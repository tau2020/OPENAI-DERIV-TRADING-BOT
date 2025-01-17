# trading_strategy.py

import asyncio
import logging
import numpy as np
from asyncio import Lock
from datetime import datetime, time

from api.deriv_api import DerivAPI
from config import config
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals

# NEW
from utils.openai_utils import (
    initialize_openai,
    get_trading_signal_from_openai,
    fallback_hold_signal
)

def compute_volatility(candle_closes, window=10):
    if len(candle_closes) < window:
        return 0.0
    returns = []
    for i in range(1, len(candle_closes)):
        prev_close = candle_closes[i-1]
        current_close = candle_closes[i]
        if prev_close != 0.0:
            ret = (current_close - prev_close) / prev_close
            returns.append(ret)
    recent_returns = returns[-window:]
    if not recent_returns:
        return 0.0
    return float(np.std(recent_returns))

class TradingStrategy:
    def __init__(
        self,
        logger: logging.Logger,
        deriv_api_client: DerivAPI,
        loginid: str = "Unknown",
        openai_api_key: str = None
    ):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.lock = Lock()
        self.total_wins = 0
        self.total_losses = 0
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.trades_data = []
        self.pl_times = []
        self.pl_values = []
        self.tick_times = []
        self.tick_prices = []
        self.candle_times = []
        self.candle_opens = []
        self.candle_highs = []
        self.candle_lows = []
        self.candle_closes = []
        self.candle_volumes = []
        self.active_symbols = []
        self.current_symbol = config.SYMBOL
        self.account_info_logged = False
        self.last_strategy_evaluation = datetime.now()
        self.latest_balance = 0.0
        self.currency = "USD"
        self.HARDCODED_STOP_LOSS = 100.0
        self.HARDCODED_TAKE_PROFIT = 20.0
        self.HARDCODED_TRAILING_STOP = 100.0
        self.HARDCODED_POSITION_SIZE = 1000.0
        self.HARDCODED_MAX_HOLDING_TIME = 3
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_trailing_stop = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 0
        self.dynamic_contract_type = "CALL"
        self.dynamic_duration = config.DURATION
        self.last_signal = "HOLD"
        self.best_profit = 0.0
        self.entry_price = 0.0
        self.trade_open_times = {}
        self.contracts_closed = {}
        self.risk_manager = RiskManager(logger)
        self.rule_signals = RuleBasedSignals(logger=self.logger)
        self.MIN_CONFIDENCE_THRESHOLD = 1.0
        self.daily_start_balance = 0.0
        self.daily_pl = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
        self.daily_loss_limit_percent = config.DAILY_LOSS_LIMIT_PERCENT
        self.daily_profit_target_percent = config.DAILY_PROFIT_TARGET_PERCENT
        self.daily_loss_limit_reached = False
        self.daily_profit_target_reached = False
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)
        self.MAX_SCALPING_STAKE = 100.0
        self.ai_client = None
        if openai_api_key:
            try:
                
                self.ai_client = initialize_openai(openai_api_key, logger)
                self.logger.info("AI client successfully initialized.")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI: {e}")
                self.ai_client = None

    async def tick_callback(self, tick):
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

    async def append_tick_data(self, timestamp, price):
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
        #if self.daily_risk_check():
          # # self.logger.warning("Daily risk limit or profit target triggered. Skipping new trades.")
          #  return
       # if self.consecutive_losses >= self.max_consecutive_losses:
          #  self.logger.warning(
           #     f"Consecutive loss limit reached ({self.consecutive_losses}). Pausing trading."
         #   )
       #     return
        ai_direction, ai_confidence = await self.evaluate_ai_direction(symbol)
       # if ai_direction == "HOLD":
       #     self.logger.info("AI says HOLD. No trade action.")
        #    return
        #if ai_confidence < self.MIN_CONFIDENCE_THRESHOLD:
          #  self.logger.info(
               # f"AI confidence {ai_confidence:.2f} < {self.MIN_CONFIDENCE_THRESHOLD:.2f}. Skipping trade."
           # )
           # return
        #rule_direction = await self.get_rule_based_direction(symbol)
        #if self.is_opposite_signal(ai_direction, rule_direction):
            #self.logger.warning(
               ## f"AI direction={ai_direction} but rule-based={rule_direction} => conflict => HOLD."
           # )
          #  return
        self.dynamic_stop_loss = self.HARDCODED_STOP_LOSS
        self.dynamic_take_profit = self.HARDCODED_TAKE_PROFIT
        self.dynamic_trailing_stop = self.HARDCODED_TRAILING_STOP
        self.dynamic_position_size = self.HARDCODED_POSITION_SIZE
        self.dynamic_max_holding_time = self.HARDCODED_MAX_HOLDING_TIME
        if ai_direction == "BUY":
            self.dynamic_contract_type = "CALL"
        else:
            self.dynamic_contract_type = "PUT"
        position_size = self.dynamic_position_size
        if position_size > self.MAX_SCALPING_STAKE:
            self.logger.warning(
                f"Position size {position_size:.2f} above max {self.MAX_SCALPING_STAKE:.2f}. Clamping."
            )
            position_size = self.MAX_SCALPING_STAKE
        if position_size > self.latest_balance:
            self.logger.warning(
                f"Position size {position_size:.2f} exceeds current balance {self.latest_balance:.2f}. Skipping trade."
            )
            return
        self.dynamic_position_size = position_size
        price_proposal = await self.get_price_proposal(symbol, ai_direction)
        if not price_proposal:
            self.logger.error("Failed to get price proposal. Trade not executed.")
            return
        await self.execute_trade(symbol, ai_direction, price_proposal)

    async def evaluate_ai_direction(self, symbol: str):
        if not self.ai_client:
            self.logger.warning("AI client not available. Defaulting to HOLD.")
            return ("HOLD", 0.0)
        try:
            await self.fetch_candle_history(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching candle data for AI: {e}")
        vol = compute_volatility(self.candle_closes, window=10)
        self.logger.info(f"Calculated volatility: {vol:.6f}")
        try:
            await self.fetch_historical_data(symbol, use_candles=False)
        except Exception as e:
            self.logger.error(f"Error fetching tick data for AI: {e}")
        ai_response = await get_trading_signal_from_openai(
            client=self.ai_client,
            symbol=symbol,
            logger=self.logger,
            prices=self.tick_prices,
            times=self.tick_times,
            candle_times=self.candle_times,
            candle_opens=self.candle_opens,
            candle_highs=self.candle_highs,
            candle_lows=self.candle_lows,
            candle_closes=self.candle_closes,
            candle_volumes=self.candle_volumes,
            recent_volatility=vol,
            recent_trades_win_loss_ratio=1.0,
            account_balance=self.latest_balance
        )
        if isinstance(ai_response, tuple):
            ai_response = ai_response[0]
        ai_signal = ai_response.get("signal", "HOLD").upper()
        ai_confidence = float(ai_response.get("confidence", 0.0))
        return (ai_signal, ai_confidence)

    async def get_rule_based_direction(self, symbol: str):
        try:
            await self.fetch_candle_history(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching candle data: {e}")
            return "HOLD"
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
        return rule_signal_data["signal"].upper()

    def is_opposite_signal(self, ai_direction: str, rule_direction: str) -> bool:
        if ai_direction == "BUY" and rule_direction == "SELL":
            return True
        if ai_direction == "SELL" and rule_direction == "BUY":
            return True
        return False

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
                self.logger.info(f"Fetched {len(candles)} candles for {symbol}. (candles)")
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
                self.logger.info(f"Fetched {len(self.tick_prices)} ticks for {symbol}. (ticks)")
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

    async def execute_trade(self, symbol, direction, price_proposal):
        proposal_id = price_proposal['id']
        ask_price = price_proposal['ask_price']
        self.logger.info(
            f"Executing trade: Symbol={symbol}, AI_Direction={direction}, "
            f"StopLoss={self.dynamic_stop_loss}, TakeProfit={self.dynamic_take_profit}, "
            f"TrailingStop={self.dynamic_trailing_stop}, PosSize={self.dynamic_position_size:.2f}, "
            f"Duration={self.dynamic_duration}"
        )
        try:
            response = await self.deriv_api.buy({"buy": proposal_id, "price": ask_price})
            if response and 'buy' in response:
                contract_id = response['buy']['contract_id']
                self.logger.info(f"Trade executed. Contract ID: {contract_id}")
                self.entry_price = float(ask_price)
                self.best_profit = 0.0
                self.trade_open_times[contract_id] = datetime.now()
                self.contracts_closed[contract_id] = False
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
        if contract_id not in self.contracts_closed or self.contracts_closed[contract_id] is True:
            return
        poc_data = poc_update.get('proposal_open_contract', {})
        if not poc_data:
            self.logger.debug(f"No poc_data for contract {contract_id}.")
            return
        is_sold = poc_data.get('is_sold', False)
        if is_sold:
            profit = float(poc_data.get('profit', 0.0))
            self.logger.info(f"Contract {contract_id} settled. Final Profit: {profit}")
            self.contracts_closed[contract_id] = True
            await self.record_trade_results(profit, contract_id)
            if not future.done():
                future.set_result(None)
            await self.log_current_balance()
            return
        contract_status = poc_data.get("status", "")
        is_valid_to_sell = poc_data.get("is_valid_to_sell", False)
        if contract_status != "open" or not is_valid_to_sell:
            self.logger.debug(
                f"Contract {contract_id} not open or not valid to sell. status={contract_status}, is_valid_to_sell={is_valid_to_sell}."
            )
            return
        current_profit = float(poc_data.get('profit', 0.0))
        if current_profit <= -abs(self.dynamic_stop_loss):
            self.logger.info(
                f"Stop loss triggered (profit={current_profit:.2f} <= -{abs(self.dynamic_stop_loss):.2f}). Selling {contract_id}."
            )
            await self.sell_contract(contract_id, future)
            return
        if current_profit >= self.dynamic_take_profit:
            self.logger.info(
                f"Take profit triggered (profit={current_profit:.2f} >= {self.dynamic_take_profit:.2f}). Selling {contract_id}."
            )
            await self.sell_contract(contract_id, future)
            return
        if current_profit > self.best_profit:
            self.best_profit = current_profit
        if self.dynamic_trailing_stop > 0.0:
            if current_profit < (self.best_profit - self.dynamic_trailing_stop):
                self.logger.info(
                    f"Trailing stop triggered. profit={current_profit:.2f} < "
                    f"(best_profit={self.best_profit:.2f} - trailing={self.dynamic_trailing_stop:.2f}). Selling {contract_id}."
                )
                await self.sell_contract(contract_id, future)
                return

    async def sell_contract(self, contract_id, future):
        if contract_id not in self.contracts_closed or self.contracts_closed[contract_id] is True:
            return
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
            if not future.done():
                future.set_result(None)

    async def record_trade_results(self, profit, contract_id):
        final_profit = profit
        if final_profit > 0:
            self.cumulative_profit += final_profit
            self.total_wins += 1
            self.trades_data.append({"profit": final_profit, "timestamp": datetime.now()})
            self.logger.info(f"Trade closed with profit: +{final_profit:.2f}")
            self.consecutive_losses = 0
        elif final_profit < 0:
            self.cumulative_loss += abs(final_profit)
            self.total_losses += 1
            self.logger.info(f"Trade closed with loss: {final_profit:.2f}")
            self.consecutive_losses += 1
        else:
            self.logger.info("No profit or loss.")
        net_pl = self.cumulative_profit - self.cumulative_loss
        current_time = datetime.now()
        self.pl_times.append(current_time)
        self.pl_values.append(net_pl)
        self.daily_pl = self.latest_balance - self.daily_start_balance
        self.logger.info(
            f"Updated daily P/L: {self.daily_pl:.2f}, Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}"
        )

    def set_active_symbols(self, symbols):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")


