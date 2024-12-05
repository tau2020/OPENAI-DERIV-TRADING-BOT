# strategy/trading_strategy.py

import logging
from datetime import datetime
import asyncio
import threading
import matplotlib.pyplot as plt
import io
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai

class TradingStrategy:
    """
    Implements the trading strategy logic, including indicator calculations,
    decision-making, and risk management.
    """

    def __init__(self, logger, deriv_api, openai_api_key):
        self.logger = logger
        self.deriv_api = deriv_api
        self.tick_times = []
        self.tick_prices = []
        self.lock = threading.Lock()
        self.last_chart_update = datetime.now()
        self.last_strategy_evaluation = datetime.now()
        self.account_info_logged = False
        self.active_symbols = []
        self.current_symbol = config.SYMBOL
        self.latest_balance = 0.0
        self.currency = 'USD'  # Default currency, will be updated

        # Initialize OpenAI client
        self.openai_client = initialize_openai(openai_api_key, self.logger)

        # Initialize cumulative P/L trackers
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.pl_times = []
        self.pl_values = []

    async def tick_callback(self, tick):
        """
        Callback function for tick updates.
        """
        tick_data = tick.get('tick', {})
        symbol = tick_data.get('symbol')
        price = float(tick_data.get('quote', 0.0))
        epoch = tick_data.get('epoch')
        timestamp = datetime.fromtimestamp(epoch)
        self.logger.debug(f"Received tick: {tick_data}")

        # Append tick data
        self.append_tick_data(timestamp, price)

        # Log account info once
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

        # Update chart if necessary
        if self.should_update_chart():
            self.update_chart(symbol)

        # Evaluate strategy if necessary
        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    def append_tick_data(self, timestamp, price):
        """
        Appends new tick data to the price and time lists.
        """
        with self.lock:
            self.tick_times.append(timestamp)
            self.tick_prices.append(price)
            if len(self.tick_prices) > config.MAX_TICK_DATA_POINTS:
                self.tick_times = self.tick_times[-config.MAX_TICK_DATA_POINTS:]
                self.tick_prices = self.tick_prices[-config.MAX_TICK_DATA_POINTS:]
        self.logger.debug(f"Tick data appended: {timestamp}, {price}")

    async def log_account_info(self):
        """
        Logs account information such as balance and account type.
        """
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
            """
            Logs the current account balance.
            """
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


    def should_update_chart(self):
        """
        Determines if it's time to update the price and P/L chart.
        """
        elapsed = (datetime.now() - self.last_chart_update).total_seconds()
        return elapsed >= config.CHART_UPDATE_INTERVAL

    def update_chart(self, symbol):
        """
        Updates the price and P/L charts with the latest data.
        """
        with self.lock:
            fig, ax1 = plt.subplots(figsize=(10, 5))

            # Plot Price
            color = 'tab:blue'
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price', color=color)
            ax1.plot(self.tick_times, self.tick_prices, label='Price', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Plot Net Cumulative P/L
            ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Net Cumulative P/L', color=color)
            ax2.plot(self.pl_times, self.pl_values, label='Net P/L', color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # Otherwise the right y-label is slightly clipped
            plt.title(f'{symbol} Price and Net Cumulative P/L Chart')
            plt.legend(loc='upper left')

            # Save chart to an in-memory buffer or file
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            self.last_chart_update = datetime.now()
            self.logger.info("Chart updated.")

    def should_evaluate_strategy(self):
        """
        Determines if it's time to evaluate the trading strategy.
        """
        elapsed = (datetime.now() - self.last_strategy_evaluation).total_seconds()
        return elapsed >= config.STRATEGY_EVALUATION_INTERVAL

    async def evaluate_and_execute_trade(self, symbol):
        """
        Evaluates the trading strategy and executes a trade if conditions are met.
        """
        self.last_strategy_evaluation = datetime.now()

        # Fetch historical data for better analysis
        await self.fetch_historical_data(symbol)

        signal, confidence = await self.evaluate_strategy()
        self.logger.info(f"Strategy evaluated. Signal: {signal}, Confidence: {confidence}")

        if signal in ["BUY", "SELL"]:
            # Get price proposal before executing trade
            price_proposal = await self.get_price_proposal(symbol, signal)
            if price_proposal:
                # Execute trade asynchronously
                await self.execute_trade(symbol, signal, price_proposal)
            else:
                self.logger.error("Failed to get price proposal. Trade not executed.")
        else:
            self.logger.info("No trade action taken.")

    async def fetch_historical_data(self, symbol):
        """
        Fetches historical tick data for analysis.
        """
        try:
            # Using the Deriv API to get historical data
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
                self.logger.info(f"Historical data fetched for symbol: {symbol}")
                with self.lock:
                    self.tick_times = [datetime.fromtimestamp(int(ts)) for ts in history.get('times', [])]
                    self.tick_prices = [float(price) for price in history.get('prices', [])]
            else:
                self.logger.error(f"Failed to fetch historical data for symbol: {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")

    async def evaluate_strategy(self):
        """
        Evaluates the trading strategy using OpenAI assistant.
        """
        with self.lock:
            if len(self.tick_prices) < config.MIN_DATA_POINTS_FOR_STRATEGY:
                self.logger.debug("Insufficient data for strategy evaluation.")
                return "HOLD", 0.0

            prices = self.tick_prices.copy()
            times = self.tick_times.copy()

        try:
            # Use OpenAI assistant to get trading signal
            signal = await get_trading_signal_from_openai(
                self.openai_client,
                self.current_symbol,
                prices,
                times,
                self.logger
            )
            confidence = 0.8  # Adjust confidence as needed

            self.logger.info(f"Strategy evaluated using OpenAI assistant. Signal: {signal}, Confidence: {confidence}")

            return signal, confidence
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {e}")
            return "HOLD", 0.0

    async def get_price_proposal(self, symbol, signal):
        """
        Retrieves a price proposal for a contract before executing a trade.
        """
        try:
            contract_type = "CALL" if signal == "BUY" else "PUT"
            proposal_request = {
                "proposal": 1,
                "amount": config.AMOUNT,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": config.CURRENCY,
                "duration": config.DURATION,
                "duration_unit": "m",
                "symbol": symbol
            }
            response = await self.deriv_api.proposal(proposal_request)
            if response and 'proposal' in response:
                self.logger.info(f"Price proposal received: {response['proposal']['ask_price']}")
                return response['proposal']
            else:
                self.logger.error("Failed to get price proposal.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting price proposal: {e}")
            return None

    async def execute_trade(self, symbol, signal, price_proposal):
        """
        Executes a trade based on the provided price proposal.
        """
        proposal_id = price_proposal['id']
        self.logger.info(f"Executing trade: Symbol={symbol}, Signal={signal}, Amount={config.AMOUNT}")

        try:
            response = await self.deriv_api.buy({"buy": proposal_id, "price": price_proposal['ask_price']})
            if response and 'buy' in response:
                contract_id = response['buy']['contract_id']
                self.logger.info(f"Trade executed. Contract ID: {contract_id}")
                await self.monitor_trade(contract_id)
                # Balance is logged in handle_poc_update
            else:
                self.logger.error("Trade execution failed.")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")

    async def monitor_trade(self, contract_id):
        """
        Monitors the open trade for stop-loss and take-profit conditions.
        """
        self.logger.info(f"Monitoring trade. Contract ID: {contract_id}")
        
        try:
            # Subscribe to proposal_open_contract stream
            poc_stream = await self.deriv_api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        except Exception as e:
            self.logger.error(f"Error subscribing to poc_stream: {e}")
            return
        
        # Create an asyncio Future to await until the contract is settled
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        def on_next(poc):
            # Handle each update from the proposal_open_contract stream
            asyncio.create_task(self.handle_poc_update(poc, contract_id))
            
            # Check if the contract is settled
            if poc.get('is_settled'):
                if not future.done():
                    future.set_result(None)
        
        def on_error(error):
            self.logger.error(f"Error in poc_stream: {error}")
            if not future.done():
                future.set_exception(error)
        
        def on_completed():
            if not future.done():
                future.set_result(None)
        
        # Subscribe to the Observable with the defined callbacks
        poc_subscription = poc_stream.subscribe(
            on_next=on_next,
            on_error=on_error,
            on_completed=on_completed
        )
        
        try:
            # Await the future until the contract is settled
            await future
        except Exception as e:
            self.logger.error(f"Exception while monitoring trade: {e}")
        finally:
            # Dispose of the subscription when done
            poc_subscription.dispose()
            self.logger.info(f"Stopped monitoring trade. Contract ID: {contract_id}")

    async def handle_poc_update(self, poc_update, contract_id):
        """
        Handles updates from the proposal_open_contract stream.
        """
        poc_data = poc_update.get('proposal_open_contract', {})
        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))

        if is_sold:
            self.logger.info(f"Contract {contract_id} has been sold or expired.")
            self.logger.info(f"Profit/Loss for Contract {contract_id}: {profit}")
            
            if profit > 0:
                self.cumulative_profit += profit
                self.logger.info(f"Cumulative Profit: {self.cumulative_profit}")
            elif profit < 0:
                self.cumulative_loss += abs(profit)
                self.logger.info(f"Cumulative Loss: {self.cumulative_loss}")
            else:
                self.logger.info("No profit or loss for this contract.")

            # Record P/L for charting
            current_time = datetime.now()
            net_pl = self.cumulative_profit - self.cumulative_loss
            self.pl_times.append(current_time)
            self.pl_values.append(net_pl)

            await self.log_current_balance()
            return

        self.logger.debug(f"Current profit: {profit}")

        if profit >= config.TAKE_PROFIT:
            self.logger.info(f"Take profit reached: {profit}")
            try:
                await self.deriv_api.sell({"sell": contract_id, "price": 0})
            except Exception as e:
                self.logger.error(f"Error executing take profit sell: {e}")
        elif profit <= -config.STOP_LOSS:
            self.logger.info(f"Stop loss reached: {profit}")
            try:
                await self.deriv_api.sell({"sell": contract_id, "price": 0})
            except Exception as e:
                self.logger.error(f"Error executing stop loss sell: {e}")

    def set_active_symbols(self, symbols):
        """
        Sets the list of active symbols available for trading.
        """
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")
