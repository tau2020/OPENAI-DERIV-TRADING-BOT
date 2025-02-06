#!/usr/bin/env python3
"""
TradingStrategy: Production-grade trading strategy for the Deriv platform.

This class integrates the Deriv API client, a robust risk manager, and an advanced
rule-based signals generator. It subscribes to live tick data, maintains historical
candle data, and executes trades based on signals that are fused from multiple indicators.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Optional, Tuple

# Import API client, configuration, risk manager, and the advanced rule-based signals.
from api.deriv_api import DerivAPI  
from config import config  # Contains settings like SYMBOL, MAX_TICK_DATA_POINTS, etc.
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals


class TradingStrategy:
    def __init__(self, logger: logging.Logger, deriv_api_client: DerivAPI, loginid: str = "Unknown"):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.loginid = loginid

        # ---------------- Concurrency Locks ----------------
        # These locks ensure that critical sections modifying shared state are thread-safe.
        self.lock = asyncio.Lock()                # General-purpose data lock.
        self.trade_lock = asyncio.Lock()          # Serializes trade executions.
        self.balance_lock = asyncio.Lock()        # Protects account balance updates.
        self.trailing_stop_lock = asyncio.Lock()  # Protects trailing stop updates.
        self.data_lock = asyncio.Lock()           # Protects market data buffers.

        # ---------------- Performance & Trade History ----------------
        self.performance_stats = {
            'wins': 0,
            'losses': 0,
            'cumulative_profit': 0.0,
            'max_drawdown': 0.0
        }
        self.trades_data: List[Dict] = []
        self.pl_times: List[datetime] = []
        self.pl_values: List[float] = []

        # ---------------- Market Data Buffers ----------------
        self.tick_times: List[datetime] = []
        self.tick_prices: List[float] = []
        self.candle_times: List[datetime] = []
        self.candle_opens: List[float] = []
        self.candle_highs: List[float] = []
        self.candle_lows: List[float] = []
        self.candle_closes: List[float] = []
        self.candle_volumes: List[float] = []

        # ---------------- Trading State ----------------
        self.active_symbols: List[str] = []
        self.current_symbol = config.SYMBOL
        self.contracts_closed: Dict[str, bool] = {}
        self.trade_open_times: Dict[str, datetime] = {}
        self.active_trades: Dict[str, Dict] = {}

        # ---------------- Account Information ----------------
        self.latest_balance = 0.0
        self.currency = "USD"
        self.account_info_logged = False

        # ---------------- Dynamic Trading Parameters ----------------
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

        # ---------------- Volatility & ATR ----------------
        self.volatility = 0.0
        self.average_true_range = 0.0

        # ---------------- Trading Session Window ----------------
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)
        self.last_strategy_evaluation = datetime.now()
        self.last_volatility_update = datetime.now()

        # ---------------- Trade Multiplier ----------------
        self.multiplier_factor = getattr(config, 'MULTIPLIER_FACTOR', 1000)

        # ---------------- Risk Manager & Signals ----------------
        self.risk_manager = RiskManager(
            logger=self.logger,
            max_daily_drawdown=config.MAX_DAILY_DRAWDOWN,
            max_consecutive_losses=config.MAX_CONSECUTIVE_LOSSES,
            daily_loss_limit_percent=config.DAILY_LOSS_LIMIT_PERCENT,
            daily_profit_target_percent=config.DAILY_PROFIT_TARGET_PERCENT
        )
        # The advanced rule-based signals module uses sophisticated weighted fusion.
        self.rule_signals = RuleBasedSignals(logger=self.logger)

    # ------------------ Core Strategy Methods ------------------

    async def tick_callback(self, tick: Dict):
        """Process an incoming tick update."""
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol', self.current_symbol)
        try:
            price = float(tick_data.get('quote', 0.0))
        except (TypeError, ValueError):
            self.logger.error("Invalid tick price received.")
            return
        epoch = tick_data.get('epoch', 0)
        timestamp = datetime.fromtimestamp(epoch)
        await self.append_tick_data(timestamp, price)

        # Feed tick to the rule-based signal aggregator (which updates internal candles).
        self.rule_signals.process_tick(timestamp, price)

        # Log account info on first tick.
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

        # Evaluate and possibly execute a trade if the strategy interval has elapsed.
        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    async def evaluate_and_execute_trade(self, symbol: str):
        """
        Evaluates the current market state and executes a trade if signals and risk conditions are met.
        This method uses advanced signal fusion output and dynamic risk thresholds.
        """
        self.last_strategy_evaluation = datetime.now()

        # Check risk limits before further evaluation.
        if self.risk_manager.risk_limits_reached():
            self.logger.warning("Risk limits reached. Skipping trade evaluation.")
            return
        if self.risk_manager.consecutive_losses >= self.risk_manager.max_consecutive_losses:
            self.logger.warning("Maximum consecutive losses reached. Skipping trade evaluation.")
            return

        # Evaluate strategy: fetch historical and market data, then generate a signal.
        signal, confidence, stop_loss, take_profit = await self.evaluate_strategy(symbol)

        # For this PhD-level version, we dynamically set trade parameters.
        self.dynamic_stop_loss = -10.0   # Example: hard stop loss threshold (can be dynamic)
        self.dynamic_take_profit = 5.0     # Example: hard take profit threshold
        self.dynamic_contract_type = "MULTUP" if signal == "BUY" else "MULTDOWN"

        # Only trade if signal is decisive and meets the confidence threshold.
        if signal not in ["BUY", "SELL"]:
            self.logger.info("Signal is HOLD; no trade executed.")
            return
        if confidence < self.rule_signals.min_confidence:
            self.logger.info(f"Signal confidence too low ({confidence:.2f}). Trade skipped.")
            return

        # Execute the trade with robust error handling.
        await self.execute_trade_safely(symbol, signal)

    async def execute_trade_safely(self, symbol: str, signal: str):
        """Execute a trade with concurrency controls and robust error handling."""
        price_proposal = await self.get_price_proposal(symbol, signal)
        if not price_proposal:
            self.logger.error("Price proposal invalid. Aborting trade execution.")
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
        Monitors an open trade by subscribing to open contract updates.
        Exits the trade based on trailing stops, profit targets, or time limits.
        """
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
            self.logger.error(f"Error monitoring trade {contract_id}: {e}")
        finally:
            await self.cleanup_trade(contract_id)

    async def handle_poc_update(self, poc_update: Dict, contract_id: str, future: asyncio.Future):
        """Handles real-time updates for an open contract and triggers exit if conditions are met."""
        if contract_id in self.contracts_closed:
            return
        poc_data = poc_update.get('proposal_open_contract', {})
        try:
            current_price = float(poc_data.get('current_spot', 0.0))
        except (TypeError, ValueError):
            self.logger.error("Invalid current spot received in POC update.")
            return

        if current_price > self.best_price:
            self.best_price = current_price
            self.logger.info(f"Best price updated to: {self.best_price:.2f}")

        # Calculate trailing stop and check exit conditions.
        trailing_stop_price = self.best_price - self.dynamic_trailing_stop
        if current_price <= trailing_stop_price:
            await self.execute_exit(contract_id, f"Trailing stop triggered at {trailing_stop_price:.2f}")
            return

        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))
        if is_sold:
            await self.finalize_trade(contract_id, profit, future)
            return

        # Update dynamic parameters and check for other exit triggers.
        await self.update_dynamic_parameters()
        await self.check_exit_conditions(contract_id, profit)

    async def update_dynamic_parameters(self):
        """
        Periodically updates dynamic parameters (volatility, ATR, trailing stops) from market data.
        This adaptive approach ensures our risk thresholds respond to changing conditions.
        """
        if (datetime.now() - self.last_volatility_update).seconds > 300:
            self.volatility = await self.calculate_market_volatility()
            self.average_true_range = self.rule_signals.calculate_atr(window=14)
            self.last_volatility_update = datetime.now()
        async with self.trailing_stop_lock:
            if self.dynamic_trailing_stop <= 0:
                self.dynamic_trailing_stop = max(self.average_true_range * 1.2, 3.0)
                self.logger.info(f"Dynamic trailing stop updated to: {self.dynamic_trailing_stop:.2f}")

    async def calculate_market_volatility(self) -> float:
        """Calculates annualized volatility from recent candle closes."""
        prices = np.array(self.candle_closes[-50:])
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns) * np.sqrt(252))

    async def check_exit_conditions(self, contract_id: str, profit: float):
        """
        Checks for exit conditions based on profit thresholds, trailing stops, or maximum holding time.
        An exit is triggered if any condition is met.
        """
        exit_reason = None
        if profit >= 5.0:
            exit_reason = f"Max profit reached: {profit:.2f}"
        elif profit <= -abs(self.dynamic_stop_loss):
            exit_reason = f"Stop loss reached: {profit:.2f}"
        elif profit >= self.dynamic_take_profit:
            exit_reason = f"Take profit reached: {profit:.2f}"
        else:
            trailing_level = self.best_price - self.dynamic_trailing_stop
            if profit < trailing_level:
                exit_reason = f"Trailing stop triggered at {trailing_level:.2f}"
            elif await self.check_holding_time(contract_id):
                exit_reason = "Maximum holding time exceeded"
        if exit_reason:
            await self.execute_exit(contract_id, exit_reason)

    async def check_holding_time(self, contract_id: str) -> bool:
        """Returns True if the trade has been open longer than the allowed holding time."""
        entry_time = self.trade_open_times.get(contract_id)
        if not entry_time:
            return False
        return (datetime.now() - entry_time).total_seconds() > self.dynamic_max_holding_time * 60

    async def execute_exit(self, contract_id: str, reason: str):
        """Executes an exit from a trade with error handling and retry logic."""
        self.logger.info(f"Exiting contract {contract_id}: {reason}")
        try:
            await self.sell_contract(contract_id)
        except Exception as e:
            self.logger.error(f"Error exiting contract {contract_id}: {e}")
            await self.retry_sell_contract(contract_id)

    async def retry_sell_contract(self, contract_id: str, max_retries: int = 3):
        """Retries selling a contract a fixed number of times with exponential backoff."""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)
                await self.sell_contract(contract_id)
                return
            except Exception as e:
                self.logger.warning(f"Sell retry {attempt + 1} for {contract_id} failed: {e}")
        self.logger.error(f"Failed to sell contract {contract_id} after {max_retries} attempts")

    async def append_tick_data(self, timestamp: datetime, price: float):
        """Appends tick data to in-memory buffers, ensuring they do not exceed the maximum length."""
        async with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
            if len(self.tick_prices) > config.MAX_TICK_DATA_POINTS:
                self.tick_times = self.tick_times[-config.MAX_TICK_DATA_POINTS:]
                self.tick_prices = self.tick_prices[-config.MAX_TICK_DATA_POINTS:]

    async def log_account_info(self):
        """Logs the account balance and initializes risk manager with the starting balance."""
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
        """Updates and logs the current account balance."""
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
        """Determines whether enough time has elapsed and market conditions allow strategy evaluation."""
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        if elapsed < config.STRATEGY_EVALUATION_INTERVAL:
            return False
        now_local = datetime.now().time()
        if not (self.allowed_trading_start <= now_local <= self.allowed_trading_end):
            self.logger.debug("Outside permitted trading hours.")
            return False
        return True

    async def evaluate_strategy(self, symbol: str) -> Tuple[str, float, float, float]:
        """
        Fetches historical and market data, updates the rule-based signals buffers, and generates a trading signal.
        
        Returns:
          A tuple (signal, confidence, stop_loss, take_profit).
          If no trade is advised, the tuple will indicate HOLD with zero confidence.
        """
        try:
            await self.fetch_historical_data(symbol)
            await self.fetch_market_data(symbol)  # Updates the RuleBasedSignals buffers.
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
        """
        Fetches historical market data (ticks or candles) and updates internal tick buffers.
        This historical data is used for indicator initialization and backfilling buffers.
        """
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
                    self.logger.error(f"No candle data for {symbol}")
                    return
                async with self.lock:
                    self.tick_times = []
                    self.tick_prices = []
                    for c in candles:
                        ts = c.get('epoch', 0)
                        price = float(c.get('close', 0.0))
                        self.tick_times.append(datetime.fromtimestamp(ts))
                        self.tick_prices.append(price)
                self.logger.info(f"Retrieved {len(candles)} candles for {symbol}.")
            else:
                history = response.get('history', {})
                times = history.get('times', [])
                prices = history.get('prices', [])
                if not times or not prices:
                    self.logger.error(f"No tick data for {symbol}")
                    return
                async with self.lock:
                    self.tick_times = []
                    self.tick_prices = []
                    for t, p in zip(times, prices):
                        self.tick_times.append(datetime.fromtimestamp(int(t)))
                        self.tick_prices.append(float(p))
                self.logger.info(f"Retrieved {len(self.tick_prices)} ticks for {symbol}.")
        except Exception as e:
            self.logger.error(f"Error fetching historical {style} data for {symbol}: {e}")

    async def fetch_market_data(self, symbol: str):
        """
        Fetches recent candle data and updates the internal buffers of the rule-based signals module.
        This is critical for real-time indicator updates.
        """
        request = {
            "ticks_history": symbol,
            "start": int(datetime.utcnow().timestamp()) - config.HISTORICAL_DATA_DURATION,
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
                self.candle_times = []
                self.candle_opens = []
                self.candle_highs = []
                self.candle_lows = []
                self.candle_closes = []
                self.candle_volumes = []
                self.rule_signals.reset_buffers()
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
                    # Update rule-based signals buffers.
                    self.rule_signals.update_buffers(open_price=o, high=h, low=l, close=close, volume=vol)
            self.logger.info(f"Market data updated for {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")

    async def subscribe_live_ticks(self, symbol: str):
        """
        Subscribes to a live tick stream for a given symbol.
        The asynchronous callback passes each tick to the tick_callback method.
        """
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
        """
        Retrieves a price proposal from the Deriv API using dynamic parameters.
        The stake amount is determined dynamically and capped by the maximum allowed stake.
        """
        try:
            stake_amount = self.dynamic_position_size if self.dynamic_position_size > 0 else 50
            max_stake = getattr(config, 'MAX_STAKE', 2000.0)
            stake_amount = min(stake_amount, max_stake)
            proposal_request = {
                "proposal": 1,
                "amount": stake_amount,
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,
                "currency": config.CURRENCY,
                "symbol": symbol,
                "multiplier": self.multiplier_factor
            }
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
        """
        Sells an open contract by sending a sell request to the Deriv API.
        On success, logs the sale; on failure, logs an error.
        """
        try:
            request = {"sell": contract_id, "price": 0}
            response = await self.deriv_api.sell(request)
            if response and 'sell' in response:
                self.logger.info(f"Contract {contract_id} sold successfully.")
            else:
                self.logger.error(f"Sell failed for contract {contract_id}. Response: {response}")
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
        """
        Records the result of a trade. If the calculated profit is near zero,
        attempts to retrieve a more accurate profit figure from the API.
        Updates performance statistics accordingly.
        """
        if abs(profit) < 0.01:
            real_profit = await self.deriv_api.get_contract_profit(contract_id)
            if real_profit is not None:
                profit = real_profit
        if profit > 0:
            self.performance_stats['cumulative_profit'] += profit
            self.performance_stats['wins'] += 1
            self.trades_data.append({"profit": profit, "timestamp": datetime.now()})
            self.logger.info(f"Trade closed with profit: +{profit:.2f}")
        elif profit < 0:
            self.performance_stats['cumulative_profit'] -= abs(profit)
            self.performance_stats['losses'] += 1
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

    # ------------------ Helper Methods ------------------

    def initialize_trade(self, contract_id: str, price_proposal: Dict):
        """Initializes trade state with entry price and timestamp."""
        self.entry_price = float(price_proposal['ask_price'])
        self.best_price = self.entry_price
        self.trade_open_times[contract_id] = datetime.now()
        self.logger.info(f"Initialized trade at price: {self.entry_price:.2f}")

    async def finalize_trade(self, contract_id: str, profit: float, future: asyncio.Future):
        """Finalizes a trade by recording results and updating account balance."""
        self.contracts_closed[contract_id] = True
        await self.record_trade_results(profit, contract_id)
        await self.log_current_balance()
        if not future.done():
            future.set_result(None)

    async def cleanup_trade(self, contract_id: str):
        """Cleans up trade state after a trade has been exited."""
        if contract_id in self.trade_open_times:
            del self.trade_open_times[contract_id]
        self.logger.info(f"Trade cleanup complete for: {contract_id}")

    async def retry_trade_execution(self, symbol: str, signal: str, max_retries: int = 3):
        """Retries executing a trade if errors occur, with exponential backoff."""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)
                price_proposal = await self.get_price_proposal(symbol, signal)
                if price_proposal:
                    await self.execute_trade_safely(symbol, signal)
                    return
            except Exception as e:
                self.logger.warning(f"Trade retry {attempt + 1} failed: {e}")

# End of TradingStrategy class.
