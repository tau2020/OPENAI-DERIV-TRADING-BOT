import asyncio
import logging
from asyncio import Lock
from datetime import datetime
import statistics

# For optional GARCH approach:
# from arch import arch_model
import numpy as np

from api.deriv_api import DerivAPI
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai
from utils.risk_manager import RiskManager
from utils.anthropic_utils import ClaudeTradingClient
from strategy.rule_based_signals import RuleBasedSignals


class TradingStrategy:
    def __init__(
        self,
        logger: logging.Logger,
        deriv_api_client: DerivAPI,
        openai_api_key: str,
        anthropic_api_key: str,
        loginid: str = "Unknown"
    ):
        """
        Initializes the TradingStrategy class with advanced metrics computations and AI integration,
        plus daily risk controls and safer risk management.
        """
        self.logger = logger
        self.deriv_api = deriv_api_client
        
        # Use asyncio.Lock for concurrency in an async environment
        self.lock = Lock()

        # Basic tracking
        self.total_wins = 0
        self.total_losses = 0
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.trades_data = []

        # P/L over time
        self.pl_times = []
        self.pl_values = []

        # Price/time series for ticks
        self.tick_times = []
        self.tick_prices = []

        # Candle arrays for advanced signals
        self.candle_times = []
        self.candle_opens = []
        self.candle_highs = []
        self.candle_lows = []
        self.candle_closes = []
        self.candle_volumes = []

        # Active symbols & general state
        self.active_symbols = []
        self.current_symbol = config.SYMBOL
        self.account_info_logged = False
        self.last_strategy_evaluation = datetime.now()
        
        # Latest known account balance/currency
        self.latest_balance = 0.0
        self.currency = "USD"

        # Derived/AI dynamic trade parameters (default values)
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 60
        self.dynamic_trailing_stop = 0.0
        self.dynamic_contract_type = "CALL"
        self.dynamic_duration = config.DURATION
        self.last_signal = "HOLD"
        self.best_profit = 0.0
        self.entry_price = 0.0

        # Track contracts that have been closed to avoid duplicates
        self.contracts_closed = {}
        # Track open times for each contract if we want to enforce max holding time
        self.trade_open_times = {}

        # AI Clients (OpenAI, Anthropic, etc.)
        self.openai_client = initialize_openai(openai_api_key, self.logger)
        self.claude_client = ClaudeTradingClient(logger=self.logger, api_key=anthropic_api_key)

        # Risk Manager
        self.risk_manager = RiskManager(logger)
        # Candle-based or tick-based rule signals
        self.rule_signals = RuleBasedSignals(logger=self.logger)

        # Minimal confidence threshold for general logic
        self.MIN_CONFIDENCE_THRESHOLD = 0.75

        # Store the loginid for logging/trace
        self.loginid = loginid

        # -----------------------
        # Additional Risk Control
        # -----------------------
        self.daily_start_balance = 0.0
        self.daily_pl = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
        self.daily_loss_limit_percent = config.DAILY_LOSS_LIMIT_PERCENT
        self.daily_profit_target_percent = config.DAILY_PROFIT_TARGET_PERCENT
        self.daily_loss_limit_reached = False
        self.daily_profit_target_reached = False

    async def tick_callback(self, tick):
        """
        Called each time a new tick arrives (from subscription).
        """
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol')
        price = float(tick_data.get('quote', 0.0))
        epoch = tick_data.get('epoch', 0)
        timestamp = datetime.fromtimestamp(epoch)

        await self.append_tick_data(timestamp, price)

        # Log account info once
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

        # Periodically run the strategy
        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    async def append_tick_data(self, timestamp, price):
        """
        Store new tick data in self.tick_times / self.tick_prices with concurrency lock.
        """
        async with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
            # Truncate if we exceed configured max
            if len(self.tick_prices) > config.MAX_TICK_DATA_POINTS:
                self.tick_times = self.tick_times[-config.MAX_TICK_DATA_POINTS:]
                self.tick_prices = self.tick_prices[-config.MAX_TICK_DATA_POINTS:]

    async def log_account_info(self):
        """
        Fetch and log account information once. Also sets the daily_start_balance if not set.
        """
        self.logger.info(f"Name (Login ID): {self.loginid}")
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
        """
        Retrieve and log the current account balance. 
        Safely parse the numeric field from the Deriv API response.
        """
        try:
            balance_response = await self.deriv_api.balance()
            if balance_response and 'balance' in balance_response:
                balance_dict = balance_response['balance']
                new_balance_str = balance_dict.get('balance', 0.0)
                new_currency = balance_dict.get('currency', 'USD')

                try:
                    new_balance = float(new_balance_str)
                except (ValueError, TypeError):
                    self.logger.error(f"Balance in unexpected format: {new_balance_str}")
                    new_balance = 0.0

                async with self.lock:
                    self.latest_balance = new_balance
                    self.currency = new_currency

                self.logger.info(f"Updated Account Balance: {new_balance} {new_currency}")
            else:
                self.logger.error(f"Balance response missing or invalid: {balance_response}")

        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")

    def should_evaluate_strategy(self):
        """
        Determine if we should evaluate the strategy based on elapsed time.
        """
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        return elapsed >= config.STRATEGY_EVALUATION_INTERVAL

    def daily_risk_check(self):
        """
        Checks whether the daily loss limit or daily profit target has been reached.
        If so, halts new trades until conditions reset.
        """
        current_loss = self.daily_start_balance - self.latest_balance
        loss_limit = self.daily_start_balance * self.daily_loss_limit_percent
        profit_target = self.daily_start_balance * self.daily_profit_target_percent

        # Check daily loss limit
        if current_loss >= loss_limit and not self.daily_loss_limit_reached:
            self.daily_loss_limit_reached = True
            self.logger.warning("Daily loss limit reached. Halting new trades.")

        # Check daily profit target
        current_profit = self.latest_balance - self.daily_start_balance
        if current_profit >= profit_target and not self.daily_profit_target_reached:
            self.daily_profit_target_reached = True
            self.logger.info("Daily profit target reached. Consider stopping or reducing risk.")

        return (self.daily_loss_limit_reached or self.daily_profit_target_reached)

    async def evaluate_and_execute_trade(self, symbol: str):
        """
        Evaluate the strategy and possibly place a trade based on the final signal.
        """
        self.last_strategy_evaluation = datetime.now()

        # 1) Check daily risk flags
        if self.daily_risk_check():
            self.logger.warning("Daily risk limit or profit target triggered. Skipping new trades.")
            return

        # 2) Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(
                f"Consecutive loss limit reached ({self.consecutive_losses}). "
                f"Pausing trading until conditions reset or manual override."
            )
            return

        # 3) Evaluate signals
        signal, confidence = await self.evaluate_strategy(symbol)
        self.logger.info(f"Strategy evaluated. Final Signal: {signal}, Confidence: {confidence:.2f}")

        if signal not in ["BUY", "SELL"]:
            self.logger.info("No trade action taken (HOLD).")
            return

        # 4) Risk-based position size
        # For example, max risk per trade = config.MAX_RISK_PER_TRADE * self.latest_balance
        max_trade_risk = config.MAX_RISK_PER_TRADE * self.latest_balance

        entry_price = 0.0
        if self.tick_prices:
            entry_price = self.tick_prices[-1]

        # If no valid stop_loss is set (or zero), define a small fallback
        stop_distance = abs(entry_price - (self.dynamic_stop_loss or (entry_price * 0.98)))
        if stop_distance <= 0:
            # fallback (avoid dividing by zero)
            stop_distance = 1.0

        # Very naive formula: position_size = risk budget / risk per contract
        self.dynamic_position_size = max_trade_risk / stop_distance

        # Ensure minimal position size of 1 if the calculation yields something smaller
        if self.dynamic_position_size < 1:
            self.logger.info(f"Position size {self.dynamic_position_size:.2f} below 1. Setting to 1.")
            self.dynamic_position_size = 1.0

        # If the suggested position size is beyond the entire balance, skip or scale down
        if self.dynamic_position_size > self.latest_balance:
            self.logger.warning(
                f"Position size {self.dynamic_position_size:.2f} exceeds current balance "
                f"({self.latest_balance:.2f}). Skipping trade."
            )
            return

        price_proposal = await self.get_price_proposal(symbol, signal)
        if not price_proposal:
            self.logger.error("Failed to get price proposal. Trade not executed.")
            return

        # 5) Execute trade
        await self.execute_trade(symbol, signal, price_proposal)

    #
    # EVALUATE STRATEGY (INTEGRATES ADVANCED METRICS)
    #

    async def evaluate_strategy(self, symbol: str):
        """
        Evaluate signals from both candle-based rule signals and advanced AI-based analysis,
        then combine into a final signal and parameters.
        """
        try:
            await self.fetch_historical_data(symbol, use_candles=False)
            await self.fetch_candle_history(symbol)
        except Exception as e:
            self.logger.error(f"Error in fetch_historical_data: {e}")
            return "HOLD", 0.0

        if len(self.candle_closes) < 20:
            self.logger.debug("Not enough candle data to evaluate.")
            return "HOLD", 0.0

        # 1) Rule-based signal
        rule_signal_data = self.rule_signals.generate_candle_signal(
            candle_times=self.candle_times,
            candle_opens=self.candle_opens,
            candle_highs=self.candle_highs,
            candle_lows=self.candle_lows,
            candle_closes=self.candle_closes,
            candle_volumes=self.candle_volumes,
            recent_trades_win_loss_ratio=1.0,  # placeholder
            account_balance=self.latest_balance
        )

        # 2) Advanced metrics
        atr_volatility = self.compute_atr(period=14)
        volume_spike = self.compute_volume_spike_factor(lookback=20)
        wl_ratio = self.compute_weighted_win_loss_ratio(lookback=20)

        # 3) AI-based signal
        openai_signal_data = await get_trading_signal_from_openai(
            client=self.openai_client,
            symbol=self.current_symbol,
            prices=self.candle_closes.copy(),
            times=self.candle_times.copy(),
            logger=self.logger,
            recent_volume=volume_spike,
            recent_volatility=atr_volatility,
            recent_trades_win_loss_ratio=wl_ratio,
            account_balance=self.latest_balance
        )

        # 4) Combine signals
        final_signal, final_confidence = self.combine_signals(rule_signal_data, openai_signal_data)

        # 5) Update dynamic fields
        self.dynamic_stop_loss = final_signal["stop_loss"]
        self.dynamic_take_profit = final_signal["take_profit"]
        self.dynamic_confidence = final_signal["confidence"]
        self.dynamic_position_size = final_signal["position_size"]
        self.dynamic_max_holding_time = final_signal["max_holding_time"]
        self.dynamic_trailing_stop = final_signal["trailing_stop"]
        self.last_signal = final_signal["signal"]

        # Map signal => contract type
        if final_signal["signal"] == "BUY":
            self.dynamic_contract_type = "CALL"
        elif final_signal["signal"] == "SELL":
            self.dynamic_contract_type = "PUT"
        else:
            self.dynamic_contract_type = "CALL"

        return final_signal["signal"], final_confidence

    def combine_signals(self, rule_signal_data, openai_signal_data):
        """
        Combine rule-based and AI signals. 
        Incorporates user-defined parameters for final signal:
          - signal
          - stop_loss
          - take_profit
          - confidence
          - position_size
          - max_holding_time
          - trailing_stop
        """
        ai_signal = openai_signal_data["signal"]
        ai_conf = openai_signal_data["confidence"]
        rule_signal = rule_signal_data["signal"]
        rule_conf = rule_signal_data["confidence"]

        # Decide which source of trade parameters to trust based on AI confidence
        if ai_conf >= 0.7:
            final_params = {
                "stop_loss": openai_signal_data["stop_loss"],
                "take_profit": openai_signal_data["take_profit"],
                "position_size": openai_signal_data["position_size"],
                "max_holding_time": openai_signal_data["max_holding_time"],
                "trailing_stop": openai_signal_data["trailing_stop"]
            }
            chosen_signal = ai_signal
            chosen_conf = ai_conf
        else:
            final_params = {
                "stop_loss": rule_signal_data["stop_loss"],
                "take_profit": rule_signal_data["take_profit"],
                "position_size": rule_signal_data["position_size"],
                "max_holding_time": rule_signal_data["max_holding_time"],
                "trailing_stop": rule_signal_data["trailing_stop"]
            }
            chosen_signal = rule_signal
            chosen_conf = rule_conf

        # If AI says HOLD => final is HOLD
        if ai_signal == "HOLD":
            final_signal = dict(openai_signal_data)
            final_signal.update(final_params)
            return final_signal, ai_conf

        # If AI & rule match => boost AI confidence
        if ai_signal == rule_signal and ai_signal in ["BUY", "SELL"]:
            merged_conf = min(1.0, ai_conf + 0.1)
            final_signal = dict(openai_signal_data)
            final_signal["confidence"] = merged_conf
            final_signal.update(final_params)
            self.logger.info("Rule-based and AI match; boosting AI confidence by 0.1")
            return final_signal, merged_conf

        # If they differ => degrade AI confidence if rule_conf is strong
        if rule_conf >= 0.8 and ai_conf > 0.1:
            ai_conf -= 0.1
            self.logger.info("Rule-based strongly disagrees; degrading AI confidence by 0.1")

        final_signal = dict(openai_signal_data)
        final_signal["confidence"] = ai_conf
        final_signal["signal"] = chosen_signal  # prefer the "chosen_signal" based on confidence logic
        final_signal.update(final_params)
        return final_signal, ai_conf

    #
    # HISTORICAL DATA FETCHING
    #

    async def fetch_historical_data(self, symbol: str, use_candles: bool = False):
        """
        Fetch either TICK or CANDLE data from the Deriv API and store it in self.tick_* arrays.
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
            request["granularity"] = 60  # 1-min candles

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
                self.logger.info(f"Fetched {len(candles)} candles for {symbol}. (style=candles)")

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
                self.logger.info(f"Fetched {len(self.tick_prices)} ticks for {symbol}. (style=ticks)")

        except Exception as e:
            self.logger.error(f"Error fetching historical data ({style}) for {symbol}: {e}")

    async def fetch_candle_history(self, symbol: str):
        """
        Fill the self.candle_* arrays for candle-based signals.
        """
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
        """
        Subscribe to live ticks for the specified symbol.
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

    #
    # TRADE EXECUTION & MONITORING
    #

    def validate_and_adjust_levels(self, signal, entry_price, stop_loss, take_profit):
        """
        Validate that we have a positive initial risk and adjust if needed.
        Returns (stop_loss, take_profit, bool_is_valid).
        """
        if signal == "BUY":
            # Ensure stop_loss < entry, take_profit > entry
            if (entry_price - stop_loss) <= 0:
                self.logger.error("Unable to ensure positive initial risk for BUY.")
                return stop_loss, take_profit, False
        else:  # SELL
            # Ensure stop_loss > entry, take_profit < entry
            if (stop_loss - entry_price) <= 0:
                self.logger.error("Unable to ensure positive initial risk for SELL.")
                return stop_loss, take_profit, False

        return stop_loss, take_profit, True

    async def get_price_proposal(self, symbol, signal):
        """
        Prepare a price proposal for a new contract.
        """
        try:
            proposal_request = {
                "proposal": 1,
                "amount": self.dynamic_position_size,
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,
                "currency": config.CURRENCY,
                "duration": self.dynamic_duration,
                "duration_unit": "m",
                "symbol": symbol
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

    async def execute_trade(self, symbol, signal, price_proposal):
        """
        Execute the trade (buy) using the given proposal.
        """
        proposal_id = price_proposal['id']
        ask_price = price_proposal['ask_price']
        self.logger.info(
            f"Executing trade: Symbol={symbol}, Signal={signal}, "
            f"PositionSize={self.dynamic_position_size:.2f}, "
            f"ContractType={self.dynamic_contract_type}, Duration={self.dynamic_duration}"
        )
        try:
            response = await self.deriv_api.buy({"buy": proposal_id, "price": ask_price})
            if response and 'buy' in response:
                contract_id = response['buy']['contract_id']
                self.logger.info(f"Trade executed. Contract ID: {contract_id}")

                self.entry_price = float(ask_price)
                self.best_profit = 0.0
                self.original_stop_loss = self.dynamic_stop_loss
                self.original_take_profit = self.dynamic_take_profit

                # Refresh balance after buy
                await self.log_current_balance()

                # Monitor trade
                asyncio.create_task(self.monitor_trade(contract_id))
            else:
                self.logger.error("Trade execution failed.")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")
            await self.log_current_balance()

    async def monitor_trade(self, contract_id):
        """
        Monitor the open contract via a 'proposal_open_contract' subscription,
        applying trailing stops, risk management, etc.
        """
        self.logger.info(f"Monitoring trade. Contract ID: {contract_id}")
        try:
            poc_stream = await self.deriv_api.subscribe(
                {"proposal_open_contract": 1, "contract_id": contract_id}
            )
        except Exception as e:
            self.logger.error(f"Error subscribing to poc_stream: {e}")
            return

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def on_next(poc):
            asyncio.create_task(self.handle_poc_update(poc, contract_id, future))

        def on_error(error):
            self.logger.error(f"Error in poc_stream: {error}")
            if not future.done():
                future.set_exception(error)

        def on_completed():
            if not future.done():
                future.set_result(None)

        poc_subscription = poc_stream.subscribe(on_next=on_next, on_error=on_error, on_completed=on_completed)

        try:
            await future
        except Exception as e:
            self.logger.error(f"Exception while monitoring trade: {e}")
        finally:
            poc_subscription.dispose()
            self.logger.info(f"Stopped monitoring trade. Contract ID: {contract_id}")

    async def handle_poc_update(self, poc_update, contract_id, future):
        """
        Handle updates from the 'proposal_open_contract' stream, applying
        dynamic trailing stops, max holding times, etc.
        """
        poc_data = poc_update.get('proposal_open_contract', {})
        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))
        current_price = float(poc_data.get('bid_price', 0.0))

        if contract_id in self.contracts_closed:
            return

        if is_sold:
            self.logger.info(f"Contract {contract_id} settled. Profit: {profit}")
            await self.record_trade_results(profit, contract_id)
            self.contracts_closed[contract_id] = True
            if not future.done():
                future.set_result(None)
            # Refresh balance after contract closes
            await self.log_current_balance()
            return

        # Apply risk management or trailing logic if still open
        if current_price > 0:
            if self.last_signal == "BUY":
                new_stop_loss = self.risk_manager.dynamic_profit_safeguard(
                    entry_price=self.entry_price,
                    current_price=current_price,
                    take_profit=self.original_take_profit,
                    initial_stop_loss=self.original_stop_loss
                )
                if current_price <= new_stop_loss:
                    self.logger.info(
                        f"Price {current_price:.5f} dipped below dynamic SL {new_stop_loss:.5f}. "
                        f"Closing contract {contract_id} to safeguard profit."
                    )
                    await self.sell_contract(contract_id, future)
                    return

            elif self.last_signal == "SELL":
                new_stop_loss = self.risk_manager.dynamic_profit_safeguard(
                    entry_price=self.entry_price,
                    current_price=current_price,
                    take_profit=self.original_take_profit,
                    initial_stop_loss=self.original_stop_loss
                )
                if current_price >= new_stop_loss:
                    self.logger.info(
                        f"Price {current_price:.5f} rose above dynamic SL {new_stop_loss:.5f}. "
                        f"Closing contract {contract_id} to safeguard profit."
                    )
                    await self.sell_contract(contract_id, future)
                    return

        # Monetary stop/tp
        if profit <= -abs(self.dynamic_stop_loss):
            self.logger.info(f"Stop loss hit for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            return

        if profit >= self.dynamic_take_profit:
            self.logger.info(f"Take profit reached for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            return

        # Check for max holding time
        # If you want to enforce max holding, track open time in self.trade_open_times
        # Here we assume a simpler approach, but you can expand as needed.

    async def sell_contract(self, contract_id, future):
        """
        Force-sell the contract (close trade) when risk parameters or trailing stops trigger.
        """
        try:
            request = {"sell": contract_id, "price": 0}
            response = await self.deriv_api.sell(request)
            if response and 'sell' in response:
                self.logger.info(f"Contract {contract_id} sold successfully.")
            else:
                self.logger.error(f"Failed to sell contract {contract_id}. Full response: {response}")
        except Exception as e:
            self.logger.error(f"Error selling contract {contract_id}: {e}")
        finally:
            self.contracts_closed[contract_id] = True
            if contract_id in self.trade_open_times:
                del self.trade_open_times[contract_id]
            await self.log_current_balance()
            if not future.done():
                future.set_result(None)

    async def record_trade_results(self, profit, contract_id):
        """
        Record P/L and update cumulative metrics after a trade is closed.
        Also updates daily P/L, consecutive losses, etc.
        """
        if abs(profit) < 0.01:
            real_profit = await self.deriv_api.get_contract_profit(contract_id)
            if real_profit is not None:
                profit = real_profit

        if profit > 0:
            self.cumulative_profit += profit
            self.total_wins += 1
            self.trades_data.append({"profit": profit, "timestamp": datetime.now()})
            self.logger.info(f"Trade closed with profit: +{profit:.2f}")
            self.consecutive_losses = 0  # reset on a win
        elif profit < 0:
            self.cumulative_loss += abs(profit)
            self.total_losses += 1
            self.logger.info(f"Trade closed with loss: {profit:.2f}")
            self.consecutive_losses += 1
        else:
            self.logger.info("No profit or loss for this contract.")

        net_pl = self.cumulative_profit - self.cumulative_loss
        current_time = datetime.now()
        self.pl_times.append(current_time)
        self.pl_values.append(net_pl)

        # Update daily P/L
        self.daily_pl = self.latest_balance - self.daily_start_balance
        self.logger.info(
            f"Updated daily P/L: {self.daily_pl:.2f}, "
            f"Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}"
        )

    #
    # ADVANCED METRICS SECTION
    #

    def compute_atr(self, period=14) -> float:
        """
        Compute the Average True Range for the last `period` candles.
        """
        if len(self.candle_closes) < period + 1:
            return 0.0

        highs = self.candle_highs[-(period+1):]
        lows = self.candle_lows[-(period+1):]
        closes = self.candle_closes[-(period+1):]

        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i - 1])
            low_close_prev = abs(lows[i] - closes[i - 1])
            tr = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(tr)

        return sum(true_ranges[-period:]) / period

    def compute_volume_spike_factor(self, lookback=20) -> float:
        """
        Compare the latest candle's volume to the average volume over `lookback` candles.
        Returns ratio > 1 => volume spike, ratio < 1 => below-average volume.
        """
        if len(self.candle_volumes) < lookback:
            return 1.0

        recent_volumes = self.candle_volumes[-lookback:]
        avg_vol = sum(recent_volumes) / lookback
        current_vol = self.candle_volumes[-1]
        if avg_vol == 0:
            return 1.0
        return current_vol / avg_vol

    def compute_weighted_win_loss_ratio(self, lookback=20) -> float:
        """
        Assign linearly increasing weights to the last `lookback` trades so 
        that recent trades have a heavier weight. 
        """
        trades = self.trades_data[-lookback:]
        if not trades:
            return 1.0

        wins = 0.0
        losses = 0.0
        total_weight = 0
        for i, trade in enumerate(trades, start=1):
            weight = i  # linear weighting
            total_weight += weight
            if trade['profit'] > 0:
                wins += weight
            elif trade['profit'] < 0:
                losses += weight

        if losses == 0:
            return wins if wins > 0 else 1.0
        else:
            return wins / losses

    def set_active_symbols(self, symbols):
        """
        Update the list of active symbols (called in main.py after fetching).
        """
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")
