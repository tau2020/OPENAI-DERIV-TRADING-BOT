import asyncio
import logging
from asyncio import Lock
from datetime import datetime, time

import numpy as np

from api.deriv_api import DerivAPI
from config import config
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals
from utils.openai_utils import get_trading_signal_from_openai, initialize_openai

class TradingStrategy:
    def __init__(
        self,
        logger: logging.Logger,
        deriv_api_client: DerivAPI,
        openai_api_key: str,
        loginid: str = "Unknown"
    ):
       
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.openai_client = initialize_openai(openai_api_key, self.logger)

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

        # Derived dynamic trade parameters
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 3  # e.g. 3 minutes
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

        # Risk Manager
        self.risk_manager = RiskManager(logger)
        # Rule-based signals (scalping logic)
        self.rule_signals = RuleBasedSignals(logger=self.logger)

        # Minimal confidence threshold for trades
        self.MIN_CONFIDENCE_THRESHOLD = 0.5

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

        # Example: trade all day
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)

        # Hard cap on scalping stake
        self.MAX_SCALPING_STAKE = 100.0

    #
    # ---------------------------
    # CALLBACKS / DATA HANDLING
    # ---------------------------
    #

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

        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

        # Periodically run the strategy
        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    async def append_tick_data(self, timestamp, price):
        async with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
            if len(self.tick_prices) > config.MAX_TICK_DATA_POINTS:
                self.tick_times = self.tick_times[-config.MAX_TICK_DATA_POINTS:]
                self.tick_prices = self.tick_prices[-config.MAX_TICK_DATA_POINTS:]

    #
    # ---------------------------
    # ACCOUNT INFO
    # ---------------------------
    #

    async def log_account_info(self):
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

    #
    # ---------------------------
    # STRATEGY EVALUATION
    # ---------------------------
    #

    def should_evaluate_strategy(self):
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        if elapsed < config.STRATEGY_EVALUATION_INTERVAL:
            return False

        now_localtime = datetime.now().time()
        if not (self.allowed_trading_start <= now_localtime <= self.allowed_trading_end):
            self.logger.debug(
                f"Outside allowed trading window ({self.allowed_trading_start} - {self.allowed_trading_end})."
            )
            return False

        return True

    def daily_risk_check(self):
        current_loss = self.daily_start_balance - self.latest_balance
        loss_limit = self.daily_start_balance * self.daily_loss_limit_percent
        profit_target = self.daily_start_balance * self.daily_profit_target_percent

        if current_loss >= loss_limit and not self.daily_loss_limit_reached:
            self.daily_loss_limit_reached = True
            self.logger.warning("Daily loss limit reached. Halting new trades.")

        current_profit = self.latest_balance - self.daily_start_balance
        if current_profit >= profit_target and not self.daily_profit_target_reached:
            self.daily_profit_target_reached = True
            self.logger.info("Daily profit target reached. Consider stopping or reducing risk.")

        return (self.daily_loss_limit_reached or self.daily_profit_target_reached)

    async def evaluate_and_execute_trade(self, symbol: str):
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
        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
            self.logger.info(
                f"Signal confidence {confidence:.2f} below threshold {self.MIN_CONFIDENCE_THRESHOLD:.2f}. Skipping."
            )
            return

        # 4) Ensure position size doesn't exceed a clamp or balance
        if self.dynamic_position_size > self.MAX_SCALPING_STAKE:
            self.logger.warning(
                f"Position size {self.dynamic_position_size:.2f} above max {self.MAX_SCALPING_STAKE:.2f}. Clamping."
            )
            self.dynamic_position_size = self.MAX_SCALPING_STAKE

        if self.dynamic_position_size > self.latest_balance:
            self.logger.warning(
                f"Position size {self.dynamic_position_size:.2f} exceeds current balance "
                f"({self.latest_balance:.2f}). Skipping trade."
            )
            return

        # 5) Get a price proposal
        price_proposal = await self.get_price_proposal(symbol, signal)
        if not price_proposal:
            self.logger.error("Failed to get price proposal. Trade not executed.")
            return

        # 6) Execute trade
        await self.execute_trade(symbol, signal, price_proposal)

    async def evaluate_strategy(self, symbol: str):
        try:
            await self.fetch_historical_data(symbol, use_candles=True)
            await self.fetch_candle_history(symbol)
        except Exception as e:
            self.logger.error(f"Error in fetch_historical_data: {e}")
            return "HOLD", 0.0

        if len(self.candle_closes) < 8:
            self.logger.debug("Not enough candle data for short EMAs. HOLD.")
            return "HOLD", 0.0

        rule_signal_data = self.rule_signals.generate_candle_signal(
            candle_times=self.candle_times,
            candle_opens=self.candle_opens,
            candle_highs=self.candle_highs,
            candle_lows=self.candle_lows,
            candle_closes=self.candle_closes,
            candle_volumes=self.candle_volumes,
            recent_trades_win_loss_ratio=1.0,
            account_balance=self.latest_balance
        )
        
        
        preliminary_signal = rule_signal_data["signal"]
        if preliminary_signal == "HOLD":
            self.logger.debug("Rule-based signal is HOLD. Skipping AI evaluation.")
            return "HOLD", 0.0

        self.logger.info(f"Rule-based candidate signal: {rule_signal_data}")
        
        prices = self.candle_closes if self.candle_closes else self.tick_prices
        times = self.candle_times if self.candle_times else self.tick_times
        if not prices or not times:
            self.logger.error("Insufficient price/time data for AI evaluation.")
            return "HOLD", 0.0


        ai_signal_data = await get_trading_signal_from_openai(
            client=self.openai_client,
            symbol=symbol,
            prices=prices,
            times=times,
            logger=self.logger,
            recent_volume=self.rule_signals.calculate_recent_volume(),
            recent_volatility=self.rule_signals.calculate_volatility(prices),
            recent_trades_win_loss_ratio=1.0,
            account_balance=self.latest_balance
        )
        self.logger.info(f"AI signal data: {ai_signal_data}")
        
        
        if ai_signal_data["signal"] != preliminary_signal:
            self.logger.warning(
                f"AI signal ({ai_signal_data['signal']}) does not confirm the rule-based signal ({preliminary_signal}). Defaulting to HOLD."
            )
            return "HOLD", 0.0

        if abs(ai_signal_data["confidence"] - 1.0) > 1e-9:
            self.logger.warning(
                f"AI confidence ({ai_signal_data['confidence']}) is not exactly 1.0. Defaulting to HOLD."
            )
            return "HOLD", 0.0


        self.dynamic_stop_loss = ai_signal_data["stop_loss"]
        self.dynamic_take_profit = ai_signal_data["take_profit"]
        self.dynamic_confidence = ai_signal_data["confidence"]
        self.dynamic_position_size = ai_signal_data["position_size"]
        self.dynamic_max_holding_time = ai_signal_data["max_holding_time"]
        self.dynamic_trailing_stop = ai_signal_data["trailing_stop"]
        
        
        self.logger.info(
            f"Final merged signal: {ai_signal_data['signal']}, "
            f"SL: {self.dynamic_stop_loss}, TP: {self.dynamic_take_profit}, "
            f"Conf: {self.dynamic_confidence}, PosSize: {self.dynamic_position_size}"
        )
        return ai_signal_data["signal"], ai_signal_data["confidence"]

        

    #
    # ---------------------------
    # HISTORICAL DATA
    # ---------------------------
    #

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

    #
    # ----------------------
    # TRADE EXECUTION
    # ----------------------
    #

    async def get_price_proposal(self, symbol, signal):
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

                await self.log_current_balance()

                asyncio.create_task(self.monitor_trade(contract_id))
            else:
                self.logger.error("Trade execution failed.")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")
            await self.log_current_balance()

    async def monitor_trade(self, contract_id):
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
       
        if contract_id in self.contracts_closed:
            return

        poc_data = poc_update.get('proposal_open_contract', {})
        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))

        if is_sold:
            self.logger.info(f"Contract {contract_id} settled. Profit: {profit}")
            self.contracts_closed[contract_id] = True
            await self.record_trade_results(profit, contract_id)
            if not future.done():
                future.set_result(None)
            await self.log_current_balance()
            return

        # Optional: check monetary stop/tp in contract P/L terms
        if profit <= -abs(self.dynamic_stop_loss):
            self.logger.info(f"Stop loss hit for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            return

        if profit >= self.dynamic_take_profit:
            self.logger.info(f"Take profit reached for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            return

        # If you want time-based exit, track open time in self.trade_open_times, etc.

    async def sell_contract(self, contract_id, future):
        if contract_id in self.contracts_closed:
            # skip double-sell attempt
            return

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
        current_time = datetime.now()
        self.pl_times.append(current_time)
        self.pl_values.append(net_pl)

        self.daily_pl = self.latest_balance - self.daily_start_balance
        self.logger.info(
            f"Updated daily P/L: {self.daily_pl:.2f}, "
            f"Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}"
        )

    def set_active_symbols(self, symbols):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")
