# strategy/trading_strategy.py

import logging
import asyncio
import statistics
import threading
from datetime import datetime
import requests
from api.deriv_api import DerivAPI  # Make sure your path is correct
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai
from utils.risk_manager import RiskManager
from utils.anthropic_utils import ClaudeTradingClient
from strategy.rule_based_signals import RuleBasedSignals

class TradingStrategy:
    def __init__(self, logger, deriv_api_client: DerivAPI, openai_api_key: str, anthropic_api_key: str):
        """
        A TradingStrategy that fetches real-time data, calls AI for signals, 
        and executes trades on Deriv with risk controls.
        """
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.lock = threading.Lock()

        self.total_wins = 0
        self.total_losses = 0
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.pl_times = []
        self.pl_values = []
        self.rule_signals = RuleBasedSignals(logger=self.logger)
        self.tick_times = []
        self.tick_prices = []
        self.trades_data = []
        self.last_strategy_evaluation = datetime.now()
        self.account_info_logged = False
        self.active_symbols = []
        self.current_symbol = config.SYMBOL
        self.latest_balance = 0.0
        self.currency = 'USD'

        # Dynamic trade parameters (updated after each AI call)
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 60
        self.dynamic_trailing_stop = 0.0
        self.dynamic_contract_type = "CALL"
        self.last_signal = "HOLD"
        self.dynamic_duration = config.DURATION
        self.best_profit = 0.0  # Track best profit for trailing logic
        self.entry_price = 0.0

        # Keep track of closed contracts to avoid double-sell
        self.contracts_closed = {}

        # Initialize AI clients
        self.openai_client = initialize_openai(openai_api_key, self.logger)
        self.claude_client = ClaudeTradingClient(logger=self.logger, api_key=anthropic_api_key)

        # Risk Manager
        self.risk_manager = RiskManager(logger)

        # Min confidence threshold to filter out borderline signals
        self.MIN_CONFIDENCE_THRESHOLD = 0.75  # example

    async def tick_callback(self, tick):
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol')
        price = float(tick_data.get('quote', 0.0))
        epoch = tick_data.get('epoch')
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

            # Truncate if needed
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
        await self.fetch_historical_data(symbol)

        # Evaluate strategy (AI signals, etc.)
        signal, confidence = await self.evaluate_strategy(config.SYMBOL)
        self.logger.info(f"Strategy evaluated. Final Signal: {signal}, Confidence: {confidence}")

        # Contract type from signal
        if signal == "BUY":
            self.dynamic_contract_type = "CALL"
        elif signal == "SELL":
            self.dynamic_contract_type = "PUT"
        else:
            self.dynamic_contract_type = "CALL"  # default if HOLD

        # Enforce min position size
        if self.dynamic_position_size < 1:
            self.logger.info(f"Position size {self.dynamic_position_size} is below 1. Setting to 1.")
            self.dynamic_position_size = 1

        # Validate + adjust
        if signal in ["BUY", "SELL"] and len(self.tick_prices) > 0:
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
                    signal = "HOLD"
                else:
                    self.dynamic_stop_loss = adjusted_sl
                    self.dynamic_take_profit = adjusted_tp

        self.last_signal = signal

        if signal in ["BUY", "SELL"]:
            self.logger.info(f"Requesting price proposal with amount: {self.dynamic_position_size}")
            price_proposal = await self.get_price_proposal(symbol, signal)
            if price_proposal:
                await self.execute_trade(symbol, signal, price_proposal)
            else:
                self.logger.error("Failed to get price proposal. Trade not executed.")
        else:
            self.logger.info("No trade action taken.")

    async def fetch_historical_data(self, symbol):
        # Only do partial or 1-time fetch if needed
        try:
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
            if history:
                with self.lock:
                    times = [datetime.fromtimestamp(int(ts)) for ts in history.get('times', [])]
                    prices = [float(p) for p in history.get('prices', [])]
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    self.tick_times.extend(times)
                    self.tick_prices.extend(prices)
            else:
                self.logger.error(f"Failed to fetch historical data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")

    async def evaluate_strategy(self, symbol: str):
        """
        1) Fetch historical data via ticks_history
        2) Generate rule-based signals
        3) Execute or hold
        """
        self.last_strategy_evaluation = datetime.now()
        await self.fetch_historical_data(symbol)

        with self.lock:
            prices = self.tick_prices.copy()
            times = self.tick_times.copy()

        if len(prices) < config.MIN_DATA_POINTS_FOR_STRATEGY:
            self.logger.debug("Insufficient data for strategy evaluation.")
            return

        volume_data = await self.compute_volume_data(symbol)
        volatility = self.compute_volatility(prices)
        sentiment_score = 0.0
        win_loss_ratio = self.compute_recent_win_loss_ratio(self.trades_data, 20)
        account_balance = self.latest_balance

        signal_data = self.rule_signals.generate_signal(
            prices=prices,
            times=times,
            volume_data=volume_data,
            volatility=volatility,
            sentiment_score=sentiment_score,
            recent_trades_win_loss_ratio=win_loss_ratio,
            account_balance=account_balance
        )

        final_signal = signal_data["signal"]
        final_conf = signal_data["confidence"]
        if final_conf < self.MIN_CONFIDENCE_THRESHOLD:
            self.logger.info(
                f"Confidence {final_conf:.2f} < {self.MIN_CONFIDENCE_THRESHOLD}, forcing HOLD."
            )
            final_signal = "HOLD"

        self.dynamic_stop_loss = signal_data["stop_loss"]
        self.dynamic_take_profit = signal_data["take_profit"]
        self.dynamic_confidence = final_conf
        self.dynamic_position_size = signal_data["position_size"]
        self.dynamic_max_holding_time = signal_data["max_holding_time"]
        self.dynamic_trailing_stop = signal_data["trailing_stop"]
        self.last_signal = final_signal

        if final_signal == "BUY":
            self.dynamic_contract_type = "CALL"
        elif final_signal == "SELL":
            self.dynamic_contract_type = "PUT"
        else:
            self.dynamic_contract_type = "CALL"

        if final_signal in ["BUY", "SELL"] and len(prices) > 0:
            entry_price = prices[-1]
            sl, tp, valid = self.validate_and_adjust_levels(
                final_signal,
                entry_price,
                self.dynamic_stop_loss,
                self.dynamic_take_profit
            )
            if not valid:
                self.logger.warning("Cannot ensure positive initial risk. Defaulting to HOLD.")
                final_signal = "HOLD"
            else:
                self.dynamic_stop_loss = sl
                self.dynamic_take_profit = tp

        self.last_signal = final_signal
        if final_signal in ["BUY", "SELL"]:
            self.logger.info(f"Requesting price proposal with amount: {self.dynamic_position_size}")
            proposal = await self.get_price_proposal(symbol, final_signal)
            if proposal:
                await self.execute_trade(symbol, final_signal, proposal)
            else:
                self.logger.error("Failed to get price proposal. Trade not executed.")
        else:
            self.logger.info("No trade action taken (HOLD).")

    def validate_and_adjust_levels(self, signal, entry_price, stop_loss, take_profit):
        """
        Validate and adjust stops to ensure positive risk.
        """
        original_sl = stop_loss
        original_tp = take_profit
        adjustment_factor = 0.01  # 1% offset, tweak as needed

        if signal == "BUY":
            # push SL below entry if it's not
            if stop_loss >= entry_price - 1:
                stop_loss = entry_price * (1 - adjustment_factor)
                self.logger.warning(f"Adjusted SL from {original_sl} to {stop_loss} for BUY.")

            if take_profit <= entry_price + 1:
                take_profit = entry_price * (1 + adjustment_factor * 2)
                self.logger.warning(f"Adjusted TP from {original_tp} to {take_profit} for BUY.")

            if (entry_price - stop_loss) <= 0:
                self.logger.error("Unable to ensure positive initial risk for BUY after adjustments.")
                return stop_loss, take_profit, False
        else:  # SELL
            if stop_loss <= entry_price + 1:
                stop_loss = entry_price * (1 + adjustment_factor)
                self.logger.warning(f"Adjusted SL from {original_sl} to {stop_loss} for SELL.")

            if take_profit >= entry_price - 1:
                take_profit = entry_price * (1 - adjustment_factor * 2)
                self.logger.warning(f"Adjusted TP from {original_tp} to {take_profit} for SELL.")

            if (stop_loss - entry_price) <= 0:
                self.logger.error("Unable to ensure positive initial risk for SELL after adjustments.")
                return stop_loss, take_profit, False

        return stop_loss, take_profit, True

    def apply_trailing_stop(self, current_price, trailing_stop, entry_price, signal):
        """
        If you want to apply trailing stops ONLY after at least 1R profit 
        or some logic, you can do so. We'll keep it simple or optional.
        """
        # For demonstration, do a standard trailing
        # "trailing_stop" is a difference from current price
        if signal == "BUY":
            new_stop = max(entry_price, current_price - trailing_stop)
        else:
            # For SELL, the trailing approach inverts
            new_stop = min(entry_price, current_price + trailing_stop)
        return new_stop

    async def get_price_proposal(self, symbol, signal):
        try:
            proposal_request = {
                "proposal": 1,
                "amount": self.dynamic_position_size,
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,  # CALL or PUT
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

        # Guard double-sell
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

        # Optional trailing stop logic. 
        # For demonstration, we skip or do partial:
        if self.dynamic_trailing_stop > 0 and self.last_signal in ["BUY", "SELL"] and current_price > 0:
            new_stop = self.apply_trailing_stop(
                current_price, self.dynamic_trailing_stop, self.entry_price, self.last_signal
            )
            # E.g. if it's a BUY:
            if self.last_signal == "BUY" and current_price < new_stop:
                self.logger.info(f"Price dipped below trailing stop {new_stop}, closing contract {contract_id}.")
                await self.sell_contract(contract_id, future)
                return
            elif self.last_signal == "SELL":
                # For SELL, trailing logic reversed:
                if current_price > new_stop:
                    self.logger.info(f"Price rose above trailing stop {new_stop}, closing contract {contract_id}.")
                    await self.sell_contract(contract_id, future)
                    return

        # Check normal stop loss or take profit
        # Compare 'profit' to dynamic_stop_loss, dynamic_take_profit
        # But note: The AI returns them in price terms, not 'profit' terms
        # So you might do: if current_price <= dynamic_stop_loss for a BUY, etc.
        # Or do it in profit terms if you want R-based logic.

        # For demonstration, let's assume we do it in profit terms:
        if profit <= -abs(self.dynamic_stop_loss) and not is_sold:
            self.logger.info(f"Stop loss hit for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            return

        if profit >= self.dynamic_take_profit and not is_sold:
            self.logger.info(f"Take profit reached for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            return

    async def sell_contract(self, contract_id, future):
        try:
            response = await self.deriv_api.sell({"sell": contract_id, "price": 0})
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
        """
        Optionally query the real profit from Deriv's profit_table 
        if profit is near zero or you want confirmation.
        """
        # Example logic to refine profit if needed
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

        current_time = datetime.now()
        net_pl = self.cumulative_profit - self.cumulative_loss
        
        self.pl_times.append(current_time)
        self.pl_values.append(net_pl)

    def set_active_symbols(self, symbols):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")
        
        
    def calculate_win_loss_ratio(trades_data, lookback=20):
        trades = trades_data[-lookback:]  # Last 20 trades
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        losses = sum(1 for trade in trades if trade['profit'] < 0)
        return wins / losses if losses > 0 else wins
    
    
    def compute_recent_win_loss_ratio(self, trades_data, lookback=20):
        trades = trades_data[-lookback:]
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        losses = sum(1 for trade in trades if trade['profit'] < 0)
        return wins / losses if losses else float(wins)

    
    
    
    
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
                # Simple placeholder approach: treat each tick price change as 'volume' = 1
                # In a real scenario, you'd fetch or compute actual volume data
                volume_data = [1 for _ in history['prices']]
        except Exception as e:
            self.logger.error(f"Error computing volume data for {symbol}: {e}")
        return volume_data

    def compute_volatility(self, prices, window: int = 20):
        if not prices or len(prices) < window:
            return 0.0
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)
        if len(returns) < window:
            return 0.0
        recent_returns = returns[-window:]
        avg_return = statistics.mean(recent_returns)
        variance = sum((r - avg_return) ** 2 for r in recent_returns) / window
        return variance ** 0.5

    def fetch_sentiment_score(self, symbol: str):
        # Placeholder for actual sentiment API integration
        # In a real implementation, you'd fetch sentiment from a data provider
        try:
            # Example: pseudo request to some sentiment endpoint
            # response = requests.get(f"https://api.sentimentprovider.com/{symbol}")
            # sentiment_score = response.json().get('sentiment', 0.0)
            sentiment_score = 0.0
            return sentiment_score
        except Exception as e:
            self.logger.error(f"Error fetching sentiment score for {symbol}: {e}")
            return 0.0


 
