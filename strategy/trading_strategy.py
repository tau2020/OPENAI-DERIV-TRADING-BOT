# trading_strategy.py
import asyncio
import logging
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Optional, Tuple
from collections import deque

from api.deriv_api import DerivAPI
from config import config
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals

class TradingStrategy:
    """
    Coordinates market data processing, signal generation, risk management,
    and order execution for the Deriv platform.
    """
    def __init__(self, logger: logging.Logger, deriv_api_client: DerivAPI, loginid: str = "Unknown"):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.loginid = loginid

        # Concurrency locks.
        self.lock = asyncio.Lock()
        self.trade_lock = asyncio.Lock()
        self.balance_lock = asyncio.Lock()
        self.trailing_stop_lock = asyncio.Lock()
        self.data_lock = asyncio.Lock()

        # Performance and trade history.
        self.performance_stats = {
            'wins': 0,
            'losses': 0,
            'cumulative_profit': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'max_profit': 0.0,
            'max_loss': 0.0
        }
        self.trades_data: List[Dict] = []
        self.pl_times: List[datetime] = []
        self.pl_values: List[float] = []

        # Market data buffers.
        self.tick_times = deque(maxlen=config.MAX_TICK_DATA_POINTS)
        self.tick_prices = deque(maxlen=config.MAX_TICK_DATA_POINTS)
        self.candle_times: List[datetime] = []
        self.candle_opens: List[float] = []
        self.candle_highs: List[float] = []
        self.candle_lows: List[float] = []
        self.candle_closes: List[float] = []
        self.candle_volumes: List[float] = []

        # Trading state.
        self.active_symbols: List[str] = []
        self.current_symbol = config.SYMBOL
        self.contracts_closed: Dict[str, bool] = {}
        self.trade_open_times: Dict[str, datetime] = {}
        self.active_trades: Dict[str, Dict] = {}

        # Account information.
        self.latest_balance = 0.0
        self.currency = "USD"
        self.account_info_logged = False

        # Dynamic trading parameters.
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 2  # in minutes
        self.dynamic_trailing_stop = 0.0
        self.dynamic_contract_type = ""
        self.dynamic_duration = config.DURATION
        self.entry_price = 0.0
        self.best_price = 0.0
        self.conversion_factor = 1.0

        # Volatility & ATR.
        self.volatility = 0.0
        self.average_true_range = 0.0

        # Trading session window.
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)
        self.last_strategy_evaluation = datetime.now()
        self.last_volatility_update = datetime.now()

        # Multiplier factor (only used for non-CALL/PUT instruments).
        self.multiplier_factor = getattr(config, 'MULTIPLIER_FACTOR', 2000)

        # Risk manager & signal generator.
        self.risk_manager = RiskManager(
            logger=self.logger,
            max_daily_drawdown=config.MAX_DAILY_DRAWDOWN,
            max_consecutive_losses=config.MAX_CONSECUTIVE_LOSSES,
            daily_loss_limit_percent=config.DAILY_LOSS_LIMIT_PERCENT,
            daily_profit_target_percent=config.DAILY_PROFIT_TARGET_PERCENT
        )
        self.rule_signals = RuleBasedSignals(logger=self.logger)

    async def tick_callback(self, tick: Dict):
        """
        Process an incoming tick update.
        """
        try:
            tick_data = tick.get('tick', {})
            symbol = tick_data.get('symbol', self.current_symbol)
            price = float(tick_data.get('quote', 0.0))
            epoch = tick_data.get('epoch', 0)
            timestamp = datetime.fromtimestamp(epoch)
            await self.append_tick_data(timestamp, price)
            self.rule_signals.process_tick(timestamp, price)

            if not self.account_info_logged:
                await self.log_account_info()
                self.account_info_logged = True

            if self.should_evaluate_strategy():
                asyncio.create_task(self.evaluate_and_execute_trade(symbol))
        except Exception as e:
            self.logger.error(f"Error in tick_callback: {e}")

    async def evaluate_and_execute_trade(self, symbol: str):
        """
        Evaluates market conditions and executes a trade if signals and risk conditions are met.
        """
        self.last_strategy_evaluation = datetime.now()

        if self.risk_manager.risk_limits_reached():
            self.logger.warning("Risk limits reached. Skipping trade evaluation.")
            return
        if self.risk_manager.consecutive_losses >= self.risk_manager.max_consecutive_losses:
            self.logger.warning("Maximum consecutive losses reached. Skipping trade evaluation.")
            return

        signal, confidence, stop_loss, take_profit = await self.evaluate_strategy(symbol)
        if signal not in ["HOLD"]:
            stop_loss = stop_loss or 1000.0
            take_profit = take_profit or 1000.0

        # For CALL/PUT options: use CALL for BUY signals, PUT for SELL signals.
        self.dynamic_stop_loss = stop_loss
        self.dynamic_take_profit = take_profit
        self.dynamic_contract_type = "CALL" if signal == "BUY" else "PUT"

        if signal not in ["BUY", "SELL"]:
            self.logger.info("Signal is HOLD; no trade executed.")
            return

        await self.execute_trade_safely(symbol, signal)

    async def execute_trade_safely(self, symbol: str, signal: str):
        """
        Executes a trade with concurrency controls and error handling.
        """
        price_proposal = await self.get_price_proposal(symbol, signal)
        if not price_proposal:
            self.logger.error("Invalid price proposal. Aborting trade execution.")
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
                    self.logger.error("Trade execution failed. No valid response received.")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")
            await self.retry_trade_execution(symbol, signal)

    async def monitor_trade(self, contract_id: str):
        """
        Monitors an open trade for exit conditions.
        """
        self.logger.info(f"Monitoring trade: {contract_id}")
        try:
            poc_stream = await self.deriv_api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": contract_id
            })
            start_time = datetime.now()
            min_holding_period = 30  # seconds
            future = asyncio.get_event_loop().create_future()

            def check_holding_time():
                return (datetime.now() - start_time).seconds > min_holding_period

            poc_stream.subscribe(
                on_next=lambda poc: asyncio.create_task(
                    self.handle_poc_update(poc, contract_id, future)
                ) if check_holding_time() else None,
                on_error=lambda e: self.logger.error(f"POC stream error: {e}"),
                on_completed=lambda: future.set_result(None)
            )
            await future
        except Exception as e:
            self.logger.error(f"Error monitoring trade {contract_id}: {e}")
        finally:
            await self.cleanup_trade(contract_id)

    async def handle_poc_update(self, poc_update: Dict, contract_id: str, future: asyncio.Future):
        if contract_id in self.contracts_closed:
            return
        poc_data = poc_update.get('proposal_open_contract', {})
        try:
            current_price = float(poc_data.get('current_spot', 0.0))
        except (TypeError, ValueError):
            self.logger.error("Invalid current spot in POC update.")
            return
        normalized_current_price = current_price * self.conversion_factor
        normalized_best_price = self.best_price * self.conversion_factor
        if normalized_current_price > normalized_best_price:
            self.best_price = current_price
            self.logger.info(f"Best price updated to: {current_price:.2f}")
        atr = self.average_true_range or 1.0
        trailing_stop_offset = max(atr * 2.0, atr * 0.5)
        if self.dynamic_contract_type == "CALL":  # long (call) position
            trailing_stop_threshold = normalized_best_price - trailing_stop_offset
        else:  # short (put) position: trailing stop above best price
            trailing_stop_threshold = normalized_best_price + trailing_stop_offset
        if (self.dynamic_contract_type == "CALL" and normalized_current_price <= trailing_stop_threshold) or \
           (self.dynamic_contract_type == "PUT" and normalized_current_price >= trailing_stop_threshold):
            await self.execute_exit(contract_id, f"Trailing stop triggered at {trailing_stop_threshold:.2f}")
            return
        try:
            profit = float(poc_data.get('profit', 0.0))
        except (TypeError, ValueError):
            profit = 0.0
        if poc_data.get('is_sold', False):
            await self.finalize_trade(contract_id, profit, future)
            return
        await self.update_dynamic_parameters()
        await self.check_exit_conditions(contract_id, profit)

    async def update_dynamic_parameters(self):
        """
        Updates dynamic parameters (ATR and trailing stop) based on recent volatility.
        """
        if (datetime.now() - self.last_volatility_update).seconds > 300:
            self.volatility = await self.calculate_market_volatility()
            self.average_true_range = self.rule_signals.calculate_atr(window=14) or 1.0
            self.last_volatility_update = datetime.now()

        # For CALL/PUT options we use ATR-based trailing stop.
        async with self.trailing_stop_lock:
            if self.dynamic_trailing_stop <= 0:
                self.dynamic_trailing_stop = max(self.average_true_range * 1.2, 3.0)
                self.logger.info(f"Dynamic trailing stop updated to: {self.dynamic_trailing_stop:.2f} (CALL/PUT mode)")

    async def calculate_market_volatility(self) -> float:
        prices = np.array(self.candle_closes[-50:])
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns) * np.sqrt(252))

    async def check_exit_conditions(self, contract_id: str, profit: float):
        exit_reason = None
        if profit >= self.dynamic_take_profit:
            exit_reason = f"Take profit reached: {profit:.2f}"
        elif profit <= -abs(self.dynamic_stop_loss):
            exit_reason = f"Stop loss reached: {profit:.2f}"
        else:
            trailing_level = self.best_price - self.dynamic_trailing_stop
            if profit < trailing_level:
                exit_reason = f"Trailing stop triggered at {trailing_level:.2f}"
            elif await self.check_holding_time(contract_id):
                exit_reason = "Maximum holding time exceeded"
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
            self.logger.error(f"Error exiting contract {contract_id}: {e}")
            await self.retry_sell_contract(contract_id)

    async def retry_sell_contract(self, contract_id: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)
                await self.sell_contract(contract_id)
                return
            except Exception as e:
                self.logger.warning(f"Sell retry {attempt + 1} for {contract_id} failed: {e}")
        self.logger.error(f"Failed to sell contract {contract_id} after {max_retries} attempts")

    async def append_tick_data(self, timestamp: datetime, price: float):
        async with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
           
    async def log_account_info(self):
        try:
            account_status_response = await self.deriv_api.get_account_status()
            account_status = account_status_response.get('get_account_status', {})
            if account_status:
                balance_response = await self.deriv_api.balance()
                balance_data = balance_response.get('balance', {})
                balance = float(balance_data.get('balance', 0.0))
                currency = balance_data.get('currency', 'USD')
                async with self.lock:
                    self.latest_balance = balance
                    self.currency = currency
                    if self.risk_manager.daily_start_balance == 0.0:
                        self.risk_manager.initialize_account(balance)
                self.logger.info(f"Account Balance: {balance:.2f} {currency}")
            else:
                self.logger.error("Account info not available.")
        except Exception as e:
            self.logger.error(f"Error retrieving account info: {e}")

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
                self.logger.info(f"Updated Account Balance: {new_balance:.2f} {new_currency}")
                self.risk_manager.update_balance(new_balance)
            else:
                self.logger.error("Invalid balance response.")
        except Exception as e:
            self.logger.error(f"Error retrieving balance: {e}")

    def should_evaluate_strategy(self) -> bool:
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        if elapsed < config.STRATEGY_EVALUATION_INTERVAL:
            return False
        now_local = datetime.now().time()
        if not (self.allowed_trading_start <= now_local <= self.allowed_trading_end):
            self.logger.debug("Outside permitted trading hours.")
            return False
        return True

    async def evaluate_strategy(self, symbol: str) -> Tuple[str, float, float, float]:
        try:
            await asyncio.gather(
                self.fetch_historical_data(symbol),
                self.fetch_market_data(symbol)
            )
            signal_info = self.rule_signals.generate_signal()
            self.logger.info(f"Strategy evaluated for {symbol}: {signal_info}")
            if signal_info['signal'] == "HOLD":
                return "HOLD", 0.0, 0.0, 0.0
            return (signal_info['signal'], signal_info['confidence'],
                    signal_info['stop_loss'], signal_info['take_profit'])
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {e}")
            return "HOLD", 0.0, 0.0, 0.0

    async def fetch_historical_data(self, symbol: str, use_candles: bool = False):
        style = "candles" if use_candles else "ticks"
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "start": int(datetime.utcnow().timestamp()) - config.DURATION,
            "end": "latest",
            "count": config.MAX_TICK_DATA_POINTS,
            "style": style
        }
        if use_candles:
            request["granularity"] = 60
        try:
            response = await asyncio.wait_for(self.deriv_api.ticks_history(request), timeout=10.0)
            if style == "candles":
                candles = response.get('candles', [])
                if not candles:
                    self.logger.error(f"No candle data for {symbol}")
                    return
                async with self.lock:
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    new_times = [datetime.fromtimestamp(c.get('epoch', 0)) for c in candles]
                    new_prices = [float(c.get('close', 0.0)) for c in candles]
                    self.tick_times.extend(new_times)
                    self.tick_prices.extend(new_prices)
                self.logger.info(f"Retrieved {len(candles)} candles for {symbol}.")
            else:
                history = response.get('history', {})
                times = history.get('times', [])
                prices = history.get('prices', [])
                if not times or not prices:
                    self.logger.error(f"No tick data for {symbol}")
                    return
                async with self.lock:
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    new_times = [datetime.fromtimestamp(int(t)) for t in times]
                    new_prices = [float(p) for p in prices]
                    self.tick_times.extend(new_times)
                    self.tick_prices.extend(new_prices)
                self.logger.info(f"Retrieved {len(self.tick_prices)} ticks for {symbol}.")
        except asyncio.TimeoutError:
            self.logger.error("Historical data request timed out.")
        except Exception as e:
            self.logger.error(f"Error fetching historical {style} data for {symbol}: {e}")

    async def fetch_market_data(self, symbol: str):
        request = {
            "ticks_history": symbol,
            "start": int(datetime.utcnow().timestamp()) - config.DURATION,
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
            async with self.data_lock:
                self.candle_times = [datetime.fromtimestamp(c.get('epoch', 0)) for c in candles]
                self.candle_opens = [float(c.get('open', 0.0)) for c in candles]
                self.candle_highs = [float(c.get('high', 0.0)) for c in candles]
                self.candle_lows = [float(c.get('low', 0.0)) for c in candles]
                self.candle_closes = [float(c.get('close', 0.0)) for c in candles]
                self.candle_volumes = [float(c.get('volume', 1.0)) for c in candles]
                self.rule_signals.reset_buffers()
                self.rule_signals.update_buffers_batch(
                    self.candle_opens, self.candle_highs, self.candle_lows,
                    self.candle_closes, self.candle_volumes
                )
            self.logger.info(f"Market data updated for {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")

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
            stake_amount = self.dynamic_position_size if self.dynamic_position_size > 0 else getattr(config, "BASE_STAKE", 100)
            max_stake = getattr(config, 'MAX_STAKE', 2000.0)
            stake_amount = min(stake_amount, max_stake)
            proposal_request = {
                "proposal": 1,
                "amount": stake_amount,
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,
                "currency": config.CURRENCY,
                "symbol": symbol,
            }
            # For instruments other than CALL/PUT options, include multiplier.
            if getattr(config, "INSTRUMENT_TYPE", "underlying") != "call_put":
                proposal_request["multiplier"] = self.multiplier_factor
            response = await self.deriv_api.proposal(proposal_request)
            if response and 'proposal' in response:
                self.logger.info(f"Price proposal: {response['proposal'].get('ask_price', 'N/A')}")
                return response['proposal']
            else:
                self.logger.error("Price proposal request failed.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting price proposal: {e}")
            return None

    async def sell_contract(self, contract_id: str, future: Optional[asyncio.Future] = None):
        if contract_id in self.contracts_closed:
            self.logger.info(f"Contract {contract_id} already closed.")
            return
        try:
            request = {"sell": contract_id, "price": 0}
            response = await self.deriv_api.sell(request)
            if response and 'sell' in response:
                self.logger.info(f"Contract {contract_id} sold successfully.")
            else:
                self.logger.error(f"Sell failed for contract {contract_id}. Response: {response}")
        except Exception as e:
            self.logger.exception(f"Error selling contract {contract_id}")
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

        entry_time = self.trade_open_times.get(contract_id, datetime.now())
        trade_duration = (datetime.now() - entry_time).seconds

        if profit > 0:
            self.performance_stats['cumulative_profit'] += profit
            self.performance_stats['wins'] += 1
            self.performance_stats['total_trades'] += 1
            self.performance_stats['max_profit'] = max(self.performance_stats.get('max_profit', profit), profit)
            self.trades_data.append({"profit": profit, "timestamp": datetime.now(), "duration": trade_duration})
            self.logger.info(f"Trade closed with profit: +{profit:.2f}")
        elif profit < 0:
            self.performance_stats['cumulative_profit'] -= abs(profit)
            self.performance_stats['losses'] += 1
            self.performance_stats['total_trades'] += 1
            self.performance_stats['max_loss'] = min(self.performance_stats.get('max_loss', profit), profit)
            self.logger.info(f"Trade closed with loss: {profit:.2f}")
        else:
            self.logger.info("Trade closed with no P/L.")

        net_pl = self.performance_stats['cumulative_profit']
        self.pl_times.append(datetime.now())
        self.pl_values.append(net_pl)
        self.logger.info(f"Updated net P/L: {net_pl:.2f}")
        self.risk_manager.record_trade(profit)

    def set_active_symbols(self, symbols: List[str]):
        self.active_symbols = symbols

    def initialize_trade(self, contract_id: str, price_proposal: Dict):
        self.entry_price = float(price_proposal['ask_price'])
        self.best_price = self.entry_price
        if self.tick_prices:
            last_tick_price = self.tick_prices[-1]
            self.conversion_factor = self.entry_price / last_tick_price if last_tick_price != 0 else 1.0
        else:
            self.conversion_factor = 1.0
        self.trade_open_times[contract_id] = datetime.now()
        self.logger.info(f"Initialized trade at price: {self.entry_price:.2f}, conversion factor: {self.conversion_factor:.4f}")

    async def finalize_trade(self, contract_id: str, profit: float, future: asyncio.Future):
        self.contracts_closed[contract_id] = True
        await self.record_trade_results(profit, contract_id)
        await self.log_current_balance()
        if not future.done():
            future.set_result(None)

    async def cleanup_trade(self, contract_id: str):
        if contract_id in self.trade_open_times:
            del self.trade_open_times[contract_id]
        self.logger.info(f"Trade cleanup complete for: {contract_id}")

    async def retry_trade_execution(self, symbol: str, signal: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)
                price_proposal = await self.get_price_proposal(symbol, signal)
                if price_proposal:
                    await self.execute_trade_safely(symbol, signal)
                    return
            except Exception as e:
                self.logger.warning(f"Trade retry {attempt + 1} failed: {e}")
