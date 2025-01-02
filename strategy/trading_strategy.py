# strategy/trading_strategy.py

import logging
import asyncio
import statistics
import threading
from datetime import datetime
from api.deriv_api import DerivAPI
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai
from utils.risk_manager import RiskManager
from utils.anthropic_utils import ClaudeTradingClient
from strategy.rule_based_signals import RuleBasedSignals

class TradingStrategy:
    def __init__(
        self,
        logger,
        deriv_api_client: DerivAPI,
        openai_api_key: str,
        anthropic_api_key: str,
        loginid: str = "Unknown"
    ):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.lock = threading.Lock()

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

        # Candle arrays: fix for candle-based signals
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
        self.latest_balance = 0.0
        self.currency = 'USD'

        # Derived/AI dynamic trade parameters
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 60
        self.dynamic_trailing_stop = 0.0
        self.dynamic_contract_type = "MULTUP"
        self.dynamic_duration = config.DURATION
        self.last_signal = "HOLD"
        self.best_profit = 0.0
        self.entry_price = 0.0

        # For preventing double-sell
        self.contracts_closed = {}

        # AI Clients
        self.openai_client = initialize_openai(openai_api_key, self.logger)
        self.claude_client = ClaudeTradingClient(logger=self.logger, api_key=anthropic_api_key)

        # Risk Manager
        self.risk_manager = RiskManager(logger)
        # Candle-based or tick-based rule signals
        self.rule_signals = RuleBasedSignals(logger=self.logger)

        # Safety threshold
        self.MIN_CONFIDENCE_THRESHOLD = 0.75

        # Store the loginid if you want to log it
        self.loginid = loginid

    async def tick_callback(self, tick):
        """
        Called each time a new tick arrives (from subscription).
        """
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol')
        price = float(tick_data.get('quote', 0.0))
        epoch = tick_data.get('epoch', 0)
        timestamp = datetime.fromtimestamp(epoch)

        self.append_tick_data(timestamp, price)

        # Log account info once
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

        # Periodically run the strategy
        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    def append_tick_data(self, timestamp, price):
        with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
            if len(self.tick_prices) > config.MAX_TICK_DATA_POINTS:
                self.tick_times = self.tick_times[-config.MAX_TICK_DATA_POINTS:]
                self.tick_prices = self.tick_prices[-config.MAX_TICK_DATA_POINTS:]

    async def log_account_info(self):
        """
        Log once. If you want the actual loginid, use `self.loginid`.
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

                with self.lock:
                    self.latest_balance = balance
                    self.currency = currency

                self.logger.info(f"Account Balance: {balance} {currency}")
                self.logger.info(f"Account Type: {account_type}")
            else:
                self.logger.error("Failed to retrieve account information.")
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")

    async def log_current_balance(self):
        try:
            balance_response = await self.deriv_api.balance()
            if balance_response and 'balance' in balance_response:
                balance = float(balance_response['balance'])
                currency = balance_response.get('currency', 'USD')
                with self.lock:
                    self.latest_balance = balance
                    self.currency = currency
                self.logger.info(f"Updated Account Balance: {balance} {currency}")
            else:
                self.logger.error("Failed to retrieve account balance.")
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")

    def should_evaluate_strategy(self):
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        return elapsed >= config.STRATEGY_EVALUATION_INTERVAL

    async def evaluate_and_execute_trade(self, symbol: str):
        """
        Evaluate the strategy and possibly place a trade.
        """
        self.last_strategy_evaluation = datetime.now()
        signal, confidence = await self.evaluate_strategy(symbol)
        self.logger.info(f"Strategy evaluated. Final Signal: {signal}, Confidence: {confidence}")

        if signal not in ["BUY", "SELL"]:
            self.logger.info("No trade action taken (HOLD).")
            return

        # Ensure min position size
        if self.dynamic_position_size < 1:
            self.logger.info(f"Position size {self.dynamic_position_size} is below 1. Setting to 1.")
            self.dynamic_position_size = 1

        # Validate levels if we have a legit price
        if self.tick_prices:
            entry_price = self.tick_prices[-1]
            if entry_price > 0:
                adjusted_sl, adjusted_tp, valid = self.validate_and_adjust_levels(
                    signal, entry_price,
                    self.dynamic_stop_loss,
                    self.dynamic_take_profit
                )
                if not valid:
                    self.logger.warning("Cannot ensure positive initial risk. Defaulting to HOLD.")
                    return
                else:
                    self.dynamic_stop_loss = adjusted_sl
                    self.dynamic_take_profit = adjusted_tp

        self.logger.info(f"Requesting price proposal with amount: {self.dynamic_position_size}")
        price_proposal = await self.get_price_proposal(symbol, signal)
        if not price_proposal:
            self.logger.error("Failed to get price proposal. Trade not executed.")
            return

        await self.execute_trade(symbol, signal, price_proposal)

    async def evaluate_strategy(self, symbol: str):
        """
        Evaluate the next signal via candle-based rule signals + AI, then combine.
        """
        try:
            # For demonstration, we fetch TICKS + CANDLES in each evaluation
            await self.fetch_historical_data(symbol, use_candles=False)  # Ticks
            await self.fetch_candle_history(symbol)                      # Candles
        except Exception as e:
            self.logger.error(f"Error in fetch_historical_data: {e}")
            return "HOLD", 0.0

        # If we want to place trades based on candle signals:
        if len(self.candle_closes) < 20:
            self.logger.debug("Not enough candle data to evaluate.")
            return "HOLD", 0.0

        # Compute volume from candle_volumes or fallback
        volume_data = self.candle_volumes.copy()
        volatility = 0.0  # or compute from candle data
        # Could also do a candle-based W/L ratio

        # 1) Candle-based rule signal
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

        # 2) OpenAI signal (use candle closes as "prices")
        openai_signal_data = await get_trading_signal_from_openai(
            client=self.openai_client,
            symbol=self.current_symbol,
            prices=self.candle_closes.copy(),
            times=self.candle_times.copy(),
            logger=self.logger,
            recent_volume=volume_data,
            recent_volatility=volatility,
            recent_trades_win_loss_ratio=1.0,
            account_balance=self.latest_balance
        )

        # 3) Combine
        final_signal, final_confidence = self.combine_signals(rule_signal_data, openai_signal_data)

        # Store results in dynamic fields
        self.dynamic_stop_loss = final_signal["stop_loss"]
        self.dynamic_take_profit = final_signal["take_profit"]
        self.dynamic_confidence = final_signal["confidence"]
        self.dynamic_position_size = final_signal["position_size"]
        self.dynamic_max_holding_time = final_signal["max_holding_time"]
        self.dynamic_trailing_stop = final_signal["trailing_stop"]
        self.last_signal = final_signal["signal"]

        if final_signal["signal"] == "BUY":
        # For a bullish multiplier contract
            self.dynamic_contract_type = "MULTUP"
        elif final_signal["signal"] == "SELL":
            # For a bearish multiplier contract
            self.dynamic_contract_type = "MULTDOWN"
        else:
            # If it’s HOLD or anything else, we won’t open a new position
            self.dynamic_contract_type = None  # or keep "MULTUP" as a default fallback

        return final_signal["signal"], final_confidence

    def combine_signals(self, rule_signal_data, openai_signal_data):
        """
        Let AI have final say, but degrade confidence if rule-based strongly disagrees, etc.
        """
        ai_signal = openai_signal_data["signal"]
        ai_conf = openai_signal_data["confidence"]
        rule_signal = rule_signal_data["signal"]
        rule_conf = rule_signal_data["confidence"]

        # If AI says HOLD => final is HOLD
        if ai_signal == "HOLD":
            return openai_signal_data, ai_conf

        # If AI & rule match => merge
        if ai_signal == rule_signal and rule_signal in ["BUY", "SELL"]:
            final_conf = min(1.0, ai_conf + 0.1)
            merged = dict(openai_signal_data)
            merged["confidence"] = final_conf
            self.logger.info("Rule-based and AI match, boosting AI confidence by 0.1")
            return merged, final_conf

        # Else AI differs from rule => degrade AI conf if rule_conf is high
        if rule_conf >= 0.8:
            if ai_conf > 0.1:
                ai_conf -= 0.1
                self.logger.info("Rule-based strong, degrading AI confidence by 0.1")

        final_data = dict(openai_signal_data)
        final_data["confidence"] = ai_conf
        return final_data, ai_conf

    async def fetch_historical_data(self, symbol: str, use_candles: bool = False):
        """
        Ticks or Candles. We'll store them in self.tick_times / self.tick_prices
        just for reference if needed, but real trades use the candle arrays below.
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
                    self.logger.error(f"No candle data returned for {symbol}")
                    return
                with self.lock:
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    for c in candles:
                        ts = c.get('epoch', 0)
                        price = float(c.get('close', 0.0))
                        self.tick_times.append(datetime.fromtimestamp(ts))
                        self.tick_prices.append(price)
                self.logger.info(f"Fetched {len(candles)} candles for {symbol}. (style=candles)")

            else:
                # style=ticks
                history = response.get('history', {})
                times = history.get('times', [])
                prices = history.get('prices', [])
                if not times or not prices:
                    self.logger.error(f"No tick data returned for {symbol}")
                    return
                with self.lock:
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
        Fill the self.candle_times... arrays for candle-based signals.
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

            with self.lock:
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

    def validate_and_adjust_levels(self, signal, entry_price, stop_loss, take_profit):
        """
        For multiplier contracts, stop_loss and take_profit
        are set in absolute monetary terms. You could still do
        minimal validation if you like.
        """
        # You can either remove these checks entirely or adapt them:
        if stop_loss < 0:
            self.logger.warning("Stop loss cannot be negative for multipliers. Setting to 0.")
            stop_loss = 0.0
        if take_profit < 0:
            self.logger.warning("Take profit cannot be negative for multipliers. Setting to 0.")
            take_profit = 0.0

        return stop_loss, take_profit, True

    def apply_trailing_stop(self, current_price, trailing_stop, entry_price, signal):
        if signal == "BUY":
            new_stop = max(entry_price, current_price - trailing_stop)
        else:
            new_stop = min(entry_price, current_price + trailing_stop)
        return new_stop

    async def get_price_proposal(self, symbol, signal):
        try:
            proposal_request = {
                "proposal": 1,
                "amount": self.dynamic_position_size,
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,
                "currency": config.CURRENCY,
                "symbol": symbol,
                "multiplier": 100   # Accepts 100,250,500,1000,2000.
              
            }
            response = await self.deriv_api.proposal(proposal_request)
            if response and 'proposal' in response:
                ask_price = response['proposal'].get('ask_price', 'N/A')
                self.logger.info(f" Multiplier Price proposal received: {ask_price}")
                return response['proposal']
            else:
                self.logger.error("Failed to get multiplier price proposal.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting price proposal: {e}")
            return None

    async def execute_trade(self, symbol, signal, price_proposal):
        proposal_id = price_proposal['id']
        ask_price = price_proposal['ask_price']
        self.logger.info(
            f"Executing trade: Symbol={symbol}, Signal={signal}, "
            f"Amount={self.dynamic_position_size}, ContractType={self.dynamic_contract_type}, "
            f"Duration={self.dynamic_duration}"
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
                await self.monitor_trade(contract_id)
            else:
                self.logger.error("Trade execution failed.")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")

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

        poc_subscription = poc_stream.subscribe(
            on_next=on_next, on_error=on_error, on_completed=on_completed
        )

        try:
            await future
        except Exception as e:
            self.logger.error(f"Exception while monitoring trade: {e}")
        finally:
            poc_subscription.dispose()
            self.logger.info(f"Stopped monitoring trade. Contract ID: {contract_id}")

    async def handle_poc_update(self, poc_update, contract_id, future):
        poc_data = poc_update.get('proposal_open_contract', {})
        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))
        current_price = float(poc_data.get('bid_price', 0.0))

        # Already sold or closed
        if contract_id in self.contracts_closed:
            return

        if is_sold:
            self.logger.info(f"Contract {contract_id} settled. Profit: {profit}")
            await self.record_trade_results(profit, contract_id)
            self.contracts_closed[contract_id] = True
            await self.log_current_balance()
            if not future.done():
                future.set_result(None)
            return

        # Dynamic profit safeguard
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
                        f"Price {current_price} dipped below dynamic SL {new_stop_loss}. "
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
                        f"Price {current_price} rose above dynamic SL {new_stop_loss}. "
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

    async def sell_contract(self, contract_id, future):
        try:
            
            request = {
            "sell": contract_id,
            "price": 0
             }
            response = await self.deriv_api.sell(request)
            if response and 'sell' in response:
                self.logger.info(f"Contract {contract_id} sold successfully.")
            else:
                self.logger.error(f"Failed to sell contract {contract_id}. Full response: {response}")
        except Exception as e:
            self.logger.error(f"Error selling contract {contract_id}: {e}")
        finally:
            self.contracts_closed[contract_id] = True
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
            self.logger.info(f"Cumulative Profit: {self.cumulative_profit}")
        elif profit < 0:
            self.cumulative_loss += abs(profit)
            self.total_losses += 1
            self.logger.info(f"Cumulative Loss: {self.cumulative_loss}")
        else:
            self.logger.info("No profit or loss for this contract.")

        net_pl = self.cumulative_profit - self.cumulative_loss
        current_time = datetime.now()
        self.pl_times.append(current_time)
        self.pl_values.append(net_pl)

    def set_active_symbols(self, symbols):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")

    @staticmethod
    def compute_recent_win_loss_ratio(trades_data, lookback=20):
        trades = trades_data[-lookback:]
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        losses = sum(1 for trade in trades if trade['profit'] < 0)
        return (wins / losses) if losses else float(wins)
