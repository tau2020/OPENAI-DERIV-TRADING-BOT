import logging
from datetime import datetime
import asyncio
import threading
import matplotlib.pyplot as plt
import io
from config import config
from utils.openai_utils import get_trading_signal_from_openai

class TradingStrategy:
    """
    Implements the trading strategy logic, including indicator calculations,
    decision-making, and risk management.
    """

    def __init__(self, logger, deriv_api):
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

    def set_active_symbols(self, symbols):
        """
        Sets the list of active symbols available for trading.
        """
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")

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
        account_status_response = await self.deriv_api.get_account_status()
        account_status = account_status_response.get('get_account_status', {})
        if account_status:
            account_name = account_status.get('loginid', 'Unknown')
            balance_response = await self.deriv_api.balance()
            balance_data = balance_response.get('balance', {})
            balance = balance_data.get('balance', 0.0)
            currency = balance_data.get('currency', 'USD')
            account_type = 'Real' if account_status.get('is_virtual', 1) == 0 else 'Demo'

            self.logger.info(f"Account Name (Login ID): {account_name}")
            self.logger.info(f"Account Balance: {balance} {currency}")
            self.logger.info(f"Account Type: {account_type}")
        else:
            self.logger.error("Failed to retrieve account information.")

    def should_update_chart(self):
        """
        Determines if it's time to update the price chart.
        """
        elapsed = (datetime.now() - self.last_chart_update).total_seconds()
        return elapsed >= config.CHART_UPDATE_INTERVAL

    def update_chart(self, symbol):
        """
        Updates the price chart with the latest data.
        """
        with self.lock:
            plt.figure(figsize=(10, 5))
            plt.plot(self.tick_times, self.tick_prices, label='Price')
            plt.title(f'{symbol} Price Chart')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.tight_layout()

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

        # Use OpenAI assistant to get trading signal
        signal = await get_trading_signal_from_openai(self.current_symbol, prices, times, self.logger)
        confidence = 0.8  # Adjust confidence as needed

        self.logger.info(f"Strategy evaluated using OpenAI assistant. Signal: {signal}, Confidence: {confidence}")

        return signal, confidence

    async def get_price_proposal(self, symbol, signal):
        """
        Retrieves a price proposal for a contract before executing a trade.
        """
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

    async def execute_trade(self, symbol, signal, price_proposal):
        """
        Executes a trade based on the provided price proposal.
        """
        proposal_id = price_proposal['id']
        self.logger.info(f"Executing trade: Symbol={symbol}, Signal={signal}, Amount={config.AMOUNT}")

        response = await self.deriv_api.buy({"buy": proposal_id, "price": price_proposal['ask_price']})
        if response and 'buy' in response:
            contract_id = response['buy']['contract_id']
            self.logger.info(f"Trade executed. Contract ID: {contract_id}")
            await self.monitor_trade(contract_id)
        else:
            self.logger.error("Trade execution failed.")

    async def monitor_trade(self, contract_id):
        """
        Monitors the open trade for stop-loss and take-profit conditions.
        """
        self.logger.info(f"Monitoring trade. Contract ID: {contract_id}")
        # Subscribe to proposal_open_contract stream
        poc_stream = await self.deriv_api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        poc_subscription = poc_stream.subscribe(lambda poc: asyncio.create_task(self.handle_poc_update(poc, contract_id)))

        # Keep monitoring until the contract is settled
        await poc_stream.pipe().to_future()

        # Dispose of the subscription when done
        poc_subscription.dispose()

    async def handle_poc_update(self, poc_update, contract_id):
        """
        Handles updates from the proposal_open_contract stream.
        """
        poc_data = poc_update.get('proposal_open_contract', {})
        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))

        if is_sold:
            self.logger.info(f"Contract {contract_id} has been sold or expired.")
            return

        self.logger.debug(f"Current profit: {profit}")

        if profit >= config.TAKE_PROFIT:
            self.logger.info(f"Take profit reached: {profit}")
            await self.deriv_api.sell({"sell": contract_id, "price": 0})
        elif profit <= -config.STOP_LOSS:
            self.logger.info(f"Stop loss reached: {profit}")
            await self.deriv_api.sell({"sell": contract_id, "price": 0})
