# strategy/trading_strategy.py

import logging
import asyncio
import statistics
import threading
from datetime import datetime
import requests
from api.deriv_api import DerivAPI
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai
from utils.risk_manager import RiskManager
from utils.anthropic_utils import ClaudeTradingClient
from strategy.rule_based_signals import RuleBasedSignals

class TradingStrategy:
    def __init__(self, logger, deriv_api_client: DerivAPI, openai_api_key: str, anthropic_api_key: str):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.lock = threading.Lock()
        self.total_wins = 0
        self.total_losses = 0
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.trades_data = []
        self.pl_times = []
        self.pl_values = []
        self.tick_times = []
        self.tick_prices = []
        self.active_symbols = []
        self.current_symbol = config.SYMBOL
        self.account_info_logged = False
        self.last_strategy_evaluation = datetime.now()
        self.latest_balance = 0.0
        self.currency = 'USD'
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
        self.contracts_closed = {}
        self.openai_client = initialize_openai(openai_api_key, self.logger)
        self.claude_client = ClaudeTradingClient(logger=self.logger, api_key=anthropic_api_key)
        self.risk_manager = RiskManager(logger)
        self.rule_signals = RuleBasedSignals(logger=self.logger)
        self.MIN_CONFIDENCE_THRESHOLD = 0.75

    async def tick_callback(self, tick):
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol')
        price = float(tick_data.get('quote', 0.0))
        epoch = tick_data.get('epoch', 0)
        timestamp = datetime.fromtimestamp(epoch)

        self.append_tick_data(timestamp, price)
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

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
        try:
            account_status_response = await self.deriv_api.get_account_status()
            account_status = account_status_response.get('get_account_status', {})
            if account_status:
                account_name = account_status.get('loginid', 'Unknown')
                balance_response = await self.deriv_api.balance()
                balance_data = balance_response.get('balance', {})
                balance = float(balance_data.get('balance', 0.0))
                currency = balance_data.get('currency', 'USD')
                account_type = 'Real' if account_status.get('is_virtual', 1) == 0 else 'Demo'

                with self.lock:
                    self.latest_balance = balance
                    self.currency = currency

                self.logger.info(f"Account Name (Login ID): {account_name}")
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
        self.last_strategy_evaluation = datetime.now()
        signal, confidence = await self.evaluate_strategy(symbol)
        self.logger.info(f"Strategy evaluated. Final Signal: {signal}, Confidence: {confidence}")

        if signal not in ["BUY", "SELL"]:
            self.logger.info("No trade action taken (HOLD).")
            return

        if self.dynamic_position_size < 1:
            self.logger.info(f"Position size {self.dynamic_position_size} is below 1. Setting to 1.")
            self.dynamic_position_size = 1

        if len(self.tick_prices) > 0:
            entry_price = self.tick_prices[-1]
            if entry_price > 0:
                adjusted_sl, adjusted_tp, valid = self.validate_and_adjust_levels(
                    signal,
                    entry_price,
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
        try:
            await self.fetch_historical_data(symbol)
        except Exception as e:
            self.logger.error(f"Error in fetch_historical_data: {e}")
            return "HOLD", 0.0

        with self.lock:
            prices = self.tick_prices.copy()
            times = self.tick_times.copy()

        if len(prices) < config.MIN_DATA_POINTS_FOR_STRATEGY:
            self.logger.debug("Insufficient data for strategy evaluation.")
            return "HOLD", 0.0

        volume_data = await self.compute_volume_data(symbol)
        volatility = self.compute_volatility(prices)
        win_loss_ratio = self.compute_recent_win_loss_ratio(self.trades_data, lookback=20)
        account_balance = self.latest_balance

        rule_signal_data = self.rule_signals.generate_signal(
            prices=prices,
            times=times,
            volume_data=volume_data,
            volatility=volatility,
            recent_trades_win_loss_ratio=win_loss_ratio,
            account_balance=account_balance
        )

        openai_signal_data = await get_trading_signal_from_openai(
            client=self.openai_client,
            symbol=symbol,
            prices=prices,
            times=times,
            logger=self.logger,
            recent_volume=volume_data,
            recent_volatility=volatility,
            recent_trades_win_loss_ratio=win_loss_ratio,
            account_balance=account_balance
        )

        final_signal, final_confidence = self.combine_signals(rule_signal_data, openai_signal_data)
        self.dynamic_stop_loss = final_signal["stop_loss"]
        self.dynamic_take_profit = final_signal["take_profit"]
        self.dynamic_confidence = final_signal["confidence"]
        self.dynamic_position_size = final_signal["position_size"]
        self.dynamic_max_holding_time = final_signal["max_holding_time"]
        self.dynamic_trailing_stop = final_signal["trailing_stop"]
        self.last_signal = final_signal["signal"]

        if final_signal["signal"] == "BUY":
            self.dynamic_contract_type = "CALL"
        elif final_signal["signal"] == "SELL":
            self.dynamic_contract_type = "PUT"
        else:
            self.dynamic_contract_type = "CALL"

        return final_signal["signal"], final_confidence

    # strategy/trading_strategy.py

    def combine_signals(self, rule_signal_data, openai_signal_data):
        """
        Use rule_signal_data as a safety guide, but allow AI (openai_signal_data) the final say.

        1) If AI says "HOLD", we do exactly that (AI final).
        2) If AI says "BUY"/"SELL":
            - If rule-based signal is the same, merge for improved SL/TP.
            - If rule-based signal conflicts, we still go with AI, 
            but if rule-based confidence is high (>= some threshold), we reduce AI's confidence slightly as a caution.
        3) Overall, final signal is from AI. 
        We only adjust the AI's stop_loss / take_profit slightly if the rule-based agreement is strong.
        """
        ai_signal = openai_signal_data["signal"]
        ai_conf = openai_signal_data["confidence"]
        rule_signal = rule_signal_data["signal"]
        rule_conf = rule_signal_data["confidence"]
        
        # 1) If AI says HOLD, final is HOLD outright
        if ai_signal == "HOLD":
            return openai_signal_data, ai_conf

        # 2) AI says BUY/SELL
        #   2a) If rule-based is the same (BUY/SELL) => merge levels, keep AI's final
        if ai_signal == rule_signal and rule_signal in ["BUY", "SELL"]:
            # Merge
            merged_stop_loss = openai_signal_data["stop_loss"]
            merged_take_profit = openai_signal_data["take_profit"] 
            merged_trail = openai_signal_data["trailing_stop"]
            merged_position_size = openai_signal_data["position_size"]
            # Boost AI confidence if both match
            final_conf = min(1.0, ai_conf + 0.1)

            final_data = {
                "signal": ai_signal,
                "stop_loss": merged_stop_loss,
                "take_profit": merged_take_profit,
                "confidence": final_conf,
                "position_size": merged_position_size,
                "max_holding_time": openai_signal_data["max_holding_time"],
                "trailing_stop": merged_trail
            }
            return final_data, final_conf

        #   2b) AI says BUY/SELL, rule-based disagrees
        #       If rule-based is HOLD or the opposite side, we still do AI, 
        #       but if rule_conf is quite high, degrade AI confidence a bit as a safety measure
        degrade_threshold = 0.8
        if rule_conf >= degrade_threshold:
            # Degrade AI confidence by 0.1 if it's > 0
            if ai_conf > 0.1:
                ai_conf -= 0.1

        # Return AI final
        final_data = {
            "signal": ai_signal,
            "stop_loss": openai_signal_data["stop_loss"],
            "take_profit": openai_signal_data["take_profit"],
            "confidence": ai_conf,
            "position_size": openai_signal_data["position_size"],
            "max_holding_time": openai_signal_data["max_holding_time"],
            "trailing_stop": openai_signal_data["trailing_stop"]
        }
        return final_data, ai_conf


    def force_hold(self):
        return {
            "signal": "HOLD",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "max_holding_time": 60,
            "trailing_stop": 0.0
        }

    async def fetch_historical_data(self, symbol):
        end_time = int(datetime.utcnow().timestamp())
        start_time = end_time - config.HISTORICAL_DATA_DURATION
        response = await self.deriv_api.ticks_history({
            "ticks_history": symbol,
            "start": start_time,
            "end": end_time,
            "style": "ticks",
            "count": config.MAX_TICK_DATA_POINTS
        })
        history = response.get('history', {})
        if not history:
            self.logger.error(f"Failed to fetch historical data for {symbol}")
            return
        with self.lock:
            times = [datetime.fromtimestamp(int(ts)) for ts in history.get('times', [])]
            prices = [float(p) for p in history.get('prices', [])]
            self.tick_times.clear()
            self.tick_prices.clear()
            self.tick_times.extend(times)
            self.tick_prices.extend(prices)

    def validate_and_adjust_levels(self, signal, entry_price, stop_loss, take_profit):
        if signal == "BUY":
            if stop_loss >= entry_price - 1:
                self.logger.warning(f"Stop Loss {stop_loss} for BUY.")
            if take_profit <= entry_price + 1:
                self.logger.warning(f"Take Profit {take_profit} for BUY.")
            if (entry_price - stop_loss) <= 0:
                self.logger.error("Unable to ensure positive initial risk for BUY after adjustments.")
                return stop_loss, take_profit, False
        else:
            if stop_loss <= entry_price + 1:
                self.logger.warning(f"Stop Loss {stop_loss} for SELL.")
            if take_profit >= entry_price - 1:
                self.logger.warning(f"Take Profit {take_profit} for SELL.")
            if (stop_loss - entry_price) <= 0:
                self.logger.error("Unable to ensure positive initial risk for SELL after adjustments.")
                return stop_loss, take_profit, False
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
            on_next=on_next,
            on_error=on_error,
            on_completed=on_completed
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
            response = await self.deriv_api.sell()
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

    async def compute_volume_data(self, symbol: str, lookback: int = 5):
        volume_data = []
        try:
            end_time = int(datetime.utcnow().timestamp())
            start_time = end_time - (60 * lookback)
            response = await self.deriv_api.ticks_history({
                "ticks_history": symbol,
                "start": start_time,
                "end": end_time,
                "style": "ticks",
                "count": config.MAX_TICK_DATA_POINTS
            })
            history = response.get('history', {})
            if history and 'prices' in history and 'times' in history:
                volume_data = [1 for _ in history['prices']]
        except Exception as e:
            self.logger.error(f"Error computing volume data for {symbol}: {e}")
        return volume_data

    def compute_volatility(self, prices, window: int = 20):
        if len(prices) < window + 1:
            return 0.0
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)
        recent_returns = returns[-window:]
        if not recent_returns:
            return 0.0
        avg_return = statistics.mean(recent_returns)
        variance = sum((r - avg_return) ** 2 for r in recent_returns) / len(recent_returns)
        return variance ** 0.5
