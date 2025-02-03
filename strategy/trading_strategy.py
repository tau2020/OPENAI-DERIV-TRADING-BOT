import asyncio
import logging
import numpy as np
from asyncio import Lock
from datetime import datetime, time, timedelta
from typing import List, Dict, Optional
from collections import deque

from api.deriv_api import DerivAPI
from config import config
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals

class TradingStrategy:
    def __init__(
        self,
        logger: logging.Logger,
        deriv_api_client: DerivAPI,
        loginid: str = "Unknown"
    ):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.loginid = loginid

        # Concurrency management
        self.lock = Lock()
        self.trade_lock = Lock()
        self.balance_lock = Lock()
        self.trailing_stop_lock = Lock()

        # Market data storage
        self.tick_times: List[datetime] = []
        self.tick_prices: List[float] = []
        self.candle_times: List[datetime] = []
        self.candle_opens: List[float] = []
        self.candle_highs: List[float] = []
        self.candle_lows: List[float] = []
        self.candle_closes: List[float] = []
        self.candle_volumes: List[float] = []

        # Trading state
        self.active_symbols: List[str] = []
        self.current_symbol = config.SYMBOL
        self.contracts_closed: Dict[str, bool] = {}
        self.trade_open_times: Dict[str, datetime] = {}
        self.active_trades: Dict[str, Dict] = {}

        # Account information
        self.latest_balance = 0.0
        self.currency = "USD"
        self.daily_start_balance = 0.0
        self.account_info_logged = False

        # Risk management
        self.risk_manager = RiskManager(logger)
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
        self.daily_loss_limit_percent = config.DAILY_LOSS_LIMIT_PERCENT
        self.daily_profit_target_percent = config.DAILY_PROFIT_TARGET_PERCENT
        self.daily_loss_limit_reached = False
        self.daily_profit_target_reached = False

        # Dynamic trading parameters (set by rule signals)
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 3  # minutes
        self.dynamic_trailing_stop = 5.0
        # For multiplier trades, we will set the contract type based on signal:
        # "MULTUP" for BUY and "MULTDOWN" for SELL.
        self.dynamic_contract_type = ""
        self.dynamic_duration = config.DURATION
        self.entry_price = 0.0
        self.best_profit = 0.0

        # Performance tracking
        self.total_wins = 0
        self.total_losses = 0
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.trades_data: List[Dict] = []
        self.pl_times: List[datetime] = []
        self.pl_values: List[float] = []

        # Enhanced features
        self.volatility = 0.0
        self.average_true_range = 0.0
        self.MAX_SCALPING_STAKE = 100.0
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)
        self.last_strategy_evaluation = datetime.now()
        self.last_volatility_update = datetime.now()

        # Multiplier factor for multiplier trades (default 10x if not set in config)
        self.multiplier_factor = getattr(config, 'MULTIPLIER_FACTOR', 1000)

        # Initialize the new rule-based signals class
        self.rule_signals = RuleBasedSignals(logger=self.logger)

    #region Core Strategy Methods
    async def tick_callback(self, tick: Dict):
        """Handle real-time tick updates."""
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol')
        price = float(tick_data.get('quote', 0.0))
        epoch = tick_data.get('epoch', 0)
        timestamp = datetime.fromtimestamp(epoch)
        await self.append_tick_data(timestamp, price)
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True
        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    async def evaluate_and_execute_trade(self, symbol: str):
        """Main trading decision flow."""
        self.last_strategy_evaluation = datetime.now()
        if self.daily_risk_check():
            self.logger.warning("Daily risk or profit target reached. Skipping trades.")
            return
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Consecutive losses limit ({self.consecutive_losses}) reached.")
            return
        signal, confidence = await self.evaluate_strategy(symbol)
        self.logger.info(f"Strategy evaluated. Signal: {signal}, Confidence: {confidence:.2f}")
        if signal not in ["BUY", "SELL"]:
            self.logger.info("No trade executed (HOLD signal).")
            return
        if confidence < self.rule_signals.min_confidence:
            self.logger.info(f"Low confidence ({confidence:.2f}). Skipping trade.")
            return
        await self.execute_trade_safely(symbol, signal)

    async def execute_trade_safely(self, symbol: str, signal: str):
        """Enhanced trade execution with safety checks."""
        price_proposal = await self.get_price_proposal(symbol, signal)
        if not price_proposal:
            self.logger.error("No valid price proposal. Aborting trade.")
            return
        try:
            async with self.trade_lock:
                response = await self.deriv_api.buy({
                    "buy": price_proposal['id'],
                    "price": price_proposal['ask_price']
                })
                if response and 'buy' in response:
                    contract_id = response['buy']['contract_id']
                    self.initialize_trade(contract_id, price_proposal)
                    asyncio.create_task(self.monitor_trade(contract_id))
                else:
                    self.logger.error("Trade execution failed. No buy response.")
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            await self.retry_trade_execution(symbol, signal)

    async def monitor_trade(self, contract_id: str):
        """Monitor open trades and check exit conditions."""
        self.logger.info(f"Monitoring trade: {contract_id}")
        try:
            poc_stream = await self.deriv_api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": contract_id
            })
            future = asyncio.get_event_loop().create_future()
            poc_stream.subscribe(
                on_next=lambda poc: asyncio.create_task(self.handle_poc_update(poc, contract_id, future)),
                on_error=lambda e: self.logger.error(f"POC stream error: {e}"),
                on_completed=lambda: future.set_result(None)
            )
            await future
        except Exception as e:
            self.logger.error(f"Trade monitoring error: {e}")
        finally:
            await self.cleanup_trade(contract_id)

    async def handle_poc_update(self, poc_update: Dict, contract_id: str, future):
        """Handle updates for the open contract and decide on exits."""
        if contract_id in self.contracts_closed:
            return

        # Update best price from the current spot.
        poc_data = poc_update.get('proposal_open_contract', {})
        current_price = float(poc_data.get('current_spot', 0.0))
        if current_price > self.best_price:
            self.best_price = current_price
            self.logger.info(f"New best price: {self.best_price:.2f}")
        trailing_stop_price = self.best_price - self.dynamic_trailing_stop
        if current_price <= trailing_stop_price:
            await self.execute_exit(contract_id, f"Price trailing stop triggered at {trailing_stop_price:.2f}")
            return

        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))
        if is_sold:
            await self.finalize_trade(contract_id, profit, future)
            return

        # <<<< AUTO-EXIT CONDITION >>>>
        # If profit reaches or exceeds $5, exit immediately.
        if profit >= 5.0:
            await self.execute_exit(contract_id, f"Auto exit: Profit target +$5 reached (profit: {profit:.2f})")
            return
        
         # Immediate hard stop check
        if profit <= -18.0:
            await self.execute_exit(contract_id, f"Hard stop loss triggered at ${profit:.2f}")
            return

        await self.update_dynamic_parameters()
        await self.check_exit_conditions(contract_id, profit)
    #endregion

    #region Enhanced Features
    async def update_dynamic_parameters(self):
        """Update parameters based on market conditions."""
        if (datetime.now() - self.last_volatility_update).seconds > 300:
            self.volatility = await self.calculate_market_volatility()
            self.average_true_range = self.calculate_atr()
            self.last_volatility_update = datetime.now()
        async with self.trailing_stop_lock:
            if self.dynamic_trailing_stop <= 0:
                self.dynamic_trailing_stop = max(self.average_true_range * 1.5, 5.0)
                self.logger.info(f"Dynamic trailing stop set to: {self.dynamic_trailing_stop:.2f}")

    async def calculate_market_volatility(self) -> float:
        prices = np.array(self.candle_closes[-50:])
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices))
        return np.std(log_returns) * np.sqrt(252)

    def calculate_atr(self) -> float:
        if len(self.candle_highs) < self.rule_signals.atr_window:
            return 0.0
        highs = np.array(self.candle_highs[-self.rule_signals.atr_window:])
        lows = np.array(self.candle_lows[-self.rule_signals.atr_window:])
        closes = np.array(self.candle_closes[-self.rule_signals.atr_window:])
        tr = np.maximum(highs - lows,
                        np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows - np.roll(closes, 1)))
        return np.mean(tr[1:])

    async def check_exit_conditions(self, contract_id: str, profit: float):
        exit_reason = None
        # (For multiplier trades, you might use different exit thresholds.)
        if profit <= -abs(self.dynamic_stop_loss):
            exit_reason = f"Fixed stop loss at {profit:.2f}"
        elif profit >= self.dynamic_take_profit:
            exit_reason = f"Fixed take profit at {profit:.2f}"
        # A simple trailing stop example: exit if profit drops below (best_profit - dynamic_trailing_stop)
        
        
        trailing_level = self.best_price - self.dynamic_trailing_stop
        if profit <= trailing_level:
            return "Price trailing stop triggered"
        
        if profit < trailing_level:
            exit_reason = f"Trailing stop triggered (level: {trailing_level:.2f})"
        elif await self.check_holding_time(contract_id):
            exit_reason = "Max holding time exceeded"
        if exit_reason:
            await self.execute_exit(contract_id, exit_reason)

    async def check_holding_time(self, contract_id: str) -> bool:
        entry_time = self.trade_open_times.get(contract_id)
        if not entry_time:
            return False
        return (datetime.now() - entry_time).total_seconds() > self.dynamic_max_holding_time * 60

    async def execute_exit(self, contract_id: str, reason: str):
        self.logger.info(f"Exiting contract {contract_id}: {reason}")
        try:
            await self.sell_contract(contract_id)
        except Exception as e:
            self.logger.error(f"Exit failed for contract {contract_id}: {e}")
            await self.retry_sell_contract(contract_id)

    async def retry_sell_contract(self, contract_id: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)
                await self.sell_contract(contract_id)
                return
            except Exception as e:
                self.logger.warning(f"Retry {attempt+1} for selling {contract_id} failed: {e}")
        self.logger.error(f"Failed to sell contract {contract_id} after {max_retries} attempts")
    #endregion

    #region Core Functionality
    async def append_tick_data(self, timestamp: datetime, price: float):
        async with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
            if len(self.tick_prices) > config.MAX_TICK_DATA_POINTS:
                self.tick_times = self.tick_times[-config.MAX_TICK_DATA_POINTS:]
                self.tick_prices = self.tick_prices[-config.MAX_TICK_DATA_POINTS:]

    async def log_account_info(self):
        try:
            account_status_response = await self.deriv_api.get_account_status()
            account_status = account_status_response.get('get_account_status', {})
            if account_status:
                balance_response = await self.deriv_api.balance()
                balance_data = balance_response.get('balance', {})
                balance = float(balance_data.get('balance', 0.0))
                currency = balance_data.get('currency', 'USD')
                account_type = 'Real' if account_status.get('is_virtual', 1) == 0 else 'Demo'
                async with self.lock:
                    self.latest_balance = balance
                    self.currency = currency
                    if self.daily_start_balance == 0.0:
                        self.daily_start_balance = balance
                self.logger.info(f"Account Balance: {balance} {currency}")
                self.logger.info(f"Account Type: {account_type}")
                self.logger.info(f"Daily Start Balance: {self.daily_start_balance}")
            else:
                self.logger.error("Failed to retrieve account information.")
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")

    async def log_current_balance(self):
        try:
            balance_response = await self.deriv_api.balance()
            if balance_response and 'balance' in balance_response:
                balance_dict = balance_response['balance']
                new_balance = float(balance_dict.get('balance', 0.0))
                new_currency = balance_dict.get('currency', 'USD')
                async with self.lock:
                    self.latest_balance = new_balance
                    self.currency = new_currency
                self.logger.info(f"Updated Account Balance: {new_balance} {new_currency}")
            else:
                self.logger.error("Balance response missing or invalid.")
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")

    def should_evaluate_strategy(self) -> bool:
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        if elapsed < config.STRATEGY_EVALUATION_INTERVAL:
            return False
        now_local = datetime.now().time()
        if not (self.allowed_trading_start <= now_local <= self.allowed_trading_end):
            self.logger.debug(f"Outside trading window: {self.allowed_trading_start} - {self.allowed_trading_end}")
            return False
        return True

    def daily_risk_check(self) -> bool:
        # Example: use a maximum drawdown check
        max_drawdown = 0.03  # 3%
        peak_balance = max(self.pl_values) if self.pl_values else self.daily_start_balance
        current_drawdown = (peak_balance - self.latest_balance) / peak_balance
        if current_drawdown >= max_drawdown:
            self.logger.warning(f"Max drawdown {current_drawdown:.2%} reached")
            return True
        # Alternatively, use smaller loss increments
        loss_increment = self.daily_start_balance * 0.01  # 1%
        if (self.daily_start_balance - self.latest_balance) >= loss_increment:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                return True
        return False

    async def evaluate_strategy(self, symbol: str):
        try:
            await self.fetch_historical_data(symbol, use_candles=True)
            await self.fetch_candle_history(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return "HOLD", 0.0
        if len(self.candle_closes) < 8:
            self.logger.debug("Insufficient candle data. HOLD.")
            return "HOLD", 0.0
        # Build a list of candle dictionaries for the rule-based signal generator
        candle_data = []
        for i in range(len(self.candle_closes)):
            candle_data.append({
                'open': self.candle_opens[i],
                'high': self.candle_highs[i],
                'low': self.candle_lows[i],
                'close': self.candle_closes[i],
                'volume': self.candle_volumes[i]
            })
        self.rule_signals.account_balance = self.latest_balance
        rule_signal_data = self.rule_signals.generate_candle_signal(candle_data)
        if rule_signal_data["signal"] == "HOLD":
            self.logger.debug("Rule-based signal is HOLD. No trade executed.")
            return "HOLD", 0.0
        self.logger.info(f"Rule-based candidate signal: {rule_signal_data}")
        self.dynamic_stop_loss = rule_signal_data["stop_loss"]
        self.dynamic_take_profit = rule_signal_data["take_profit"]
        self.dynamic_confidence = rule_signal_data["confidence"]
        # Now, instead of using a fixed stake, we calculate a multiplier.
        # Multiply the calculated position size by the multiplier factor.
        base_size = rule_signal_data["position_size"]
        self.dynamic_position_size = round(base_size * self.multiplier_factor, 2)
        self.dynamic_max_holding_time = rule_signal_data["max_holding_time"]
        self.dynamic_trailing_stop = rule_signal_data["trailing_stop"]
        # Set contract type based on signal: for multiplier trades, use MULTUP for BUY and MULTDOWN for SELL.
        if rule_signal_data["signal"] == "BUY":
            self.dynamic_contract_type = "MULTUP"
        elif rule_signal_data["signal"] == "SELL":
            self.dynamic_contract_type = "MULTDOWN"
        self.logger.info(
            f"Final rule-based signal: {rule_signal_data['signal']}, SL: {self.dynamic_stop_loss}, "
            f"TP: {self.dynamic_take_profit}, Conf: {self.dynamic_confidence}, "
            f"Multiplier: {self.dynamic_position_size}"
        )
        return rule_signal_data["signal"], rule_signal_data["confidence"]

    async def fetch_historical_data(self, symbol: str, use_candles: bool = False):
        style = "candles" if use_candles else "ticks"
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "start": int(datetime.utcnow().timestamp()) - config.HISTORICAL_DATA_DURATION,
            "end": "latest",
            "count": config.MAX_TICK_DATA_POINTS,
            "style": style
        }
        if use_candles:
            request["granularity"] = 60
        try:
            response = await self.deriv_api.ticks_history(request)
            if style == "candles":
                candles = response.get('candles', [])
                if not candles:
                    self.logger.error(f"No candle data returned for {symbol}")
                    return
                async with self.lock:
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    for c in candles:
                        ts = c.get('epoch', 0)
                        price = float(c.get('close', 0.0))
                        self.tick_times.append(datetime.fromtimestamp(ts))
                        self.tick_prices.append(price)
                self.logger.info(f"Fetched {len(candles)} candles for {symbol}.")
            else:
                history = response.get('history', {})
                times = history.get('times', [])
                prices = history.get('prices', [])
                if not times or not prices:
                    self.logger.error(f"No tick data returned for {symbol}")
                    return
                async with self.lock:
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    for t, p in zip(times, prices):
                        self.tick_times.append(datetime.fromtimestamp(int(t)))
                        self.tick_prices.append(float(p))
                self.logger.info(f"Fetched {len(self.tick_prices)} ticks for {symbol}.")
        except Exception as e:
            self.logger.error(f"Error fetching historical data ({style}) for {symbol}: {e}")

    async def fetch_candle_history(self, symbol: str):
        end_epoch = int(datetime.utcnow().timestamp())
        start_epoch = end_epoch - config.HISTORICAL_DATA_DURATION
        request = {
            "ticks_history": symbol,
            "start": start_epoch,
            "end": "latest",
            "adjust_start_time": 1,
            "style": "candles",
            "granularity": 60,
            "count": config.MAX_TICK_DATA_POINTS
        }
        try:
            response = await self.deriv_api.ticks_history(request)
            candles = response.get('candles', [])
            if not candles:
                self.logger.error(f"No candle data returned for {symbol}")
                return
            async with self.lock:
                self.candle_times.clear()
                self.candle_opens.clear()
                self.candle_highs.clear()
                self.candle_lows.clear()
                self.candle_closes.clear()
                self.candle_volumes.clear()
                for c in candles:
                    epoch = c.get('epoch', 0)
                    o = float(c.get('open', 0.0))
                    h = float(c.get('high', 0.0))
                    l = float(c.get('low', 0.0))
                    close = float(c.get('close', 0.0))
                    vol = float(c.get('volume', 1.0))
                    self.candle_times.append(datetime.fromtimestamp(epoch))
                    self.candle_opens.append(o)
                    self.candle_highs.append(h)
                    self.candle_lows.append(l)
                    self.candle_closes.append(close)
                    self.candle_volumes.append(vol)
            self.logger.info(f"Fetched {len(candles)} candles for {symbol}.")
        except Exception as e:
            self.logger.error(f"Error fetching candle data for {symbol}: {e}")

    async def subscribe_live_ticks(self, symbol: str):
        request = {
            "ticks_history": symbol,
            "end": "latest",
            "style": "ticks",
            "adjust_start_time": 1,
            "count": 10,
            "subscribe": 1
        }
        try:
            tick_stream = await self.deriv_api.ticks_history(request)
            def on_next(tick_update):
                if "tick" in tick_update:
                    asyncio.create_task(self.tick_callback(tick_update))
            def on_error(error):
                self.logger.error(f"Live tick error for {symbol}: {error}")
            def on_completed():
                self.logger.info(f"Live tick subscription completed for {symbol}")
            tick_stream.subscribe(on_next=on_next, on_error=on_error, on_completed=on_completed)
            self.logger.info(f"Subscribed to live ticks for {symbol}")
        except Exception as e:
            self.logger.error(f"Error subscribing to live ticks for {symbol}: {e}")

    async def get_price_proposal(self, symbol: str, signal: str):
        try:
            proposal_request = {
                "proposal": 1,
                # Use dynamic_position_size as the multiplier amount
                "amount": 50,
                # Change basis from stake to multiplier
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,
                "currency": config.CURRENCY,            
                "symbol": symbol,
                "multiplier": self.multiplier_factor
            }
            response = await self.deriv_api.proposal(proposal_request)
            if response and 'proposal' in response:
                ask_price = response['proposal'].get('ask_price', 'N/A')
                self.logger.info(f"Price proposal received: {ask_price}")
                return response['proposal']
            else:
                self.logger.error("Failed to get price proposal.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting price proposal: {e}")
            return None

    async def sell_contract(self, contract_id: str, future: Optional[asyncio.Future] = None):
        try:
            request = {"sell": contract_id, "price": 0}
            response = await self.deriv_api.sell(request)
            if response and 'sell' in response:
                self.logger.info(f"Contract {contract_id} sold successfully.")
            else:
                self.logger.error(f"Failed to sell contract {contract_id}. Response: {response}")
        except Exception as e:
            self.logger.error(f"Error selling contract {contract_id}: {e}")
        finally:
            self.contracts_closed[contract_id] = True
            if contract_id in self.trade_open_times:
                del self.trade_open_times[contract_id]
            await self.log_current_balance()
            if future and not future.done():
                future.set_result(None)

    async def record_trade_results(self, profit: float, contract_id: str):
        if abs(profit) < 0.01:
            real_profit = await self.deriv_api.get_contract_profit(contract_id)
            if real_profit is not None:
                profit = real_profit
        if profit > 0:
            self.cumulative_profit += profit
            self.total_wins += 1
            self.trades_data.append({"profit": profit, "timestamp": datetime.now()})
            self.logger.info(f"Trade closed with profit: +{profit:.2f}")
            self.consecutive_losses = 0
        elif profit < 0:
            self.cumulative_loss += abs(profit)
            self.total_losses += 1
            self.logger.info(f"Trade closed with loss: {profit:.2f}")
            self.consecutive_losses += 1
        else:
            self.logger.info("No profit or loss for this contract.")
        net_pl = self.cumulative_profit - self.cumulative_loss
        self.pl_times.append(datetime.now())
        self.pl_values.append(net_pl)
        self.daily_pl = self.latest_balance - self.daily_start_balance
        self.logger.info(f"Updated daily P/L: {self.daily_pl:.2f}, Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}")

    def set_active_symbols(self, symbols: List[str]):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")
    #endregion

    #region Helper Methods
    def initialize_trade(self, contract_id: str, price_proposal: Dict):
        self.entry_price = float(price_proposal['ask_price'])
        self.best_price = self.entry_price  # Initialize best_price
        self.trade_open_times[contract_id] = datetime.now()
        self.logger.info(f"Trade initialized at {self.entry_price:.2f}")

    async def finalize_trade(self, contract_id: str, profit: float, future: asyncio.Future):
        self.contracts_closed[contract_id] = True
        await self.record_trade_results(profit, contract_id)
        await self.log_current_balance()
        if not future.done():
            future.set_result(None)

    async def cleanup_trade(self, contract_id: str):
        if contract_id in self.trade_open_times:
            del self.trade_open_times[contract_id]
        self.logger.info(f"Trade cleanup completed: {contract_id}")

    async def retry_trade_execution(self, symbol: str, signal: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)
                price_proposal = await self.get_price_proposal(symbol, signal)
                if price_proposal:
                    await self.execute_trade_safely(symbol, signal)
                    return
            except Exception as e:
                self.logger.warning(f"Trade retry {attempt+1} failed: {e}")
    #endregion
