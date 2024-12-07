# strategy/trading_strategy.py

import logging
from datetime import datetime
import asyncio
import statistics
import threading
import matplotlib.pyplot as plt
import io

import requests
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai

class TradingStrategy:

    def __init__(self, logger, deriv_api, openai_api_key):
        self.logger = logger
        self.deriv_api = deriv_api
        self.total_wins = 0
        self.total_losses = 0
        self.tick_times = []
        self.tick_prices = []
        self.lock = threading.Lock()
        
        self.last_strategy_evaluation = datetime.now()
        self.account_info_logged = False
        self.active_symbols = []
        self.current_symbol = config.SYMBOL
        self.latest_balance = 0.0
        self.dynamic_stop_loss = config.STOP_LOSS  # Default from config, will be overridden
        self.dynamic_take_profit = config.TAKE_PROFIT
        self.dynamic_confidence = 0.0
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
      

        # Append tick data
        self.append_tick_data(timestamp, price)

        # Log account info once
        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

     
        
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
            
               
        

    def fetch_polygon_data(self, symbol):
        
        end_date = datetime.utcnow().strftime("%Y-%m-%d")  # Correct usage
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/{end_date}"
        params = {
            "apiKey": config.POLYGONIO_API_KEY
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            results = data.get("results", [])
            if not results:
                return "No Polygon data available."

            # Take the most recent day's data
            recent_data = results[-1]
            o, h, l, c, v = recent_data["o"], recent_data["h"], recent_data["l"], recent_data["c"], recent_data["v"]
            polygon_summary = f"Recent OHLC: O={o}, H={h}, L={l}, C={c}, Volume={v}"
            return polygon_summary
        except Exception as e:
            self.logger.error(f"Error fetching Polygon data: {e}")
            return "No Polygon data available."

    async def evaluate_strategy(self):
            with self.lock:
                if len(self.tick_prices) < config.MIN_DATA_POINTS_FOR_STRATEGY:
                    self.logger.debug("Insufficient data for strategy evaluation.")
                    return "HOLD", 0.0

                prices = self.tick_prices.copy()
                times = self.tick_times.copy()

            recent_volume_data = self.calculate_recent_volume()
            recent_volatility = self.calculate_volatility(prices)
            sentiment_score = self.get_market_sentiment()
            recent_trades_win_loss_ratio = self.calculate_win_loss_ratio()
            account_balance = self.latest_balance

            # For now, still no actual fundamentals or polygon data
           
            polygon_data = "No Polygon data available."
            news_headlines = []

            try:
                result = await get_trading_signal_from_openai(
                    self.openai_client,
                    self.current_symbol,
                    prices,
                    times,
                    self.logger,
                    recent_volume=recent_volume_data,
                    recent_volatility=recent_volatility,
                    sentiment_score=sentiment_score,
                    recent_trades_win_loss_ratio=recent_trades_win_loss_ratio,
                    account_balance=account_balance,
                    news_headlines=news_headlines,
                    
                    polygon_data=polygon_data
                )

                signal = result["signal"]
                self.dynamic_stop_loss = result["stop_loss"]
                self.dynamic_take_profit = result["take_profit"]
                confidence = result["confidence"]

                self.logger.info(
                    f"Enhanced strategy evaluation: {signal}, SL={self.dynamic_stop_loss}, "
                    f"TP={self.dynamic_take_profit}, Confidence={confidence}"
                )

                return signal, confidence
            except Exception as e:
                self.logger.error(f"Error evaluating strategy: {e}")
                return "HOLD", 0.0
        
        
        
    def calculate_recent_volume(self, lookback=50):
         
            with self.lock:
                recent_ticks = self.tick_prices[-lookback:]
                # The 'volume' is just the number of ticks. We can also scale it by some factor if we want.
                approx_volume = len(recent_ticks)

            # Return a simple list with one volume data point or generate a series
            # For demonstration, let's just return a series showing some variation.
            # If we wanted multiple data points, we could take multiple smaller segments and count ticks in each.
            return [approx_volume for _ in range(5)]  # Just replicate it 5 times as an example


    def calculate_volatility(self, prices, annualize=False):
            """
            Calculate volatility as the standard deviation of log returns or percentage returns.
            Here we will use a simple percentage change between consecutive ticks.
            If annualize=True, we can scale it, but since this is tick data, scaling might be arbitrary.

            volatility = stddev of returns
            returns = (price[i] - price[i-1]) / price[i-1]
            """
            if len(prices) < 2:
                return 0.0

            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)

            if not returns:
                return 0.0

            vol = statistics.pstdev(returns)  # population std dev
            # If desired, annualize or scale by sqrt of periods, etc.
            # Without time context, we just return raw std dev.
            return vol

    def get_market_sentiment(self, window=20):
            """
            Approximate sentiment from recent price action.
            If the latest price is above the average of the last `window` prices, sentiment is positive.
            If below, sentiment is negative.
            Scale sentiment between -1 and 1 depending on how far from the mean the price is.

            sentiment = (current_price - avg_price) / (avg_price) clipped to [-1,1].
            """
            with self.lock:
                if len(self.tick_prices) < window:
                    window_prices = self.tick_prices
                else:
                    window_prices = self.tick_prices[-window:]

            if not window_prices:
                return 0.0

            avg_price = sum(window_prices) / len(window_prices)
            current_price = window_prices[-1]

            if avg_price == 0:
                return 0.0

            raw_sentiment = (current_price - avg_price) / avg_price
            # Clip sentiment to range [-1,1]
            sentiment = max(-1.0, min(1.0, raw_sentiment))
            return sentiment

    def calculate_win_loss_ratio(self):
            """
            Compute the ratio of total wins to total losses.
            If no losses, and we have wins, ratio can be high.
            If no trades at all, return 1.0 as neutral.
            """
            wins = self.total_wins
            losses = self.total_losses

            if wins == 0 and losses == 0:
                return 1.0  # neutral if no trades

            if losses == 0:
                # If we never lost, ratio is very high, let's cap it at something
                return float(wins)  # Could also return a large number like 10.0

            return wins / losses   


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
        poc_data = poc_update.get('proposal_open_contract', {})
        is_sold = poc_data.get('is_sold', False)
        profit = float(poc_data.get('profit', 0.0))

        if is_sold:
            self.logger.info(f"Contract {contract_id} has been sold or expired.")
            self.logger.info(f"Profit/Loss for Contract {contract_id}: {profit}")

            if profit > 0:
                self.cumulative_profit += profit
                self.total_wins += 1
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

            await self.log_current_balance()
            return

    def set_active_symbols(self, symbols):
        """
        Sets the list of active symbols available for trading.
        """
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")
 
 
    def fetch_recent_news(self, symbol):
        """
        Fetch recent market news related to the provided symbol using the News API.
        """
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": config.NEWS_API_KEY,
            "pageSize": 5
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            articles = data.get("articles", [])
            news_list = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                news_list.append(f"{title}: {description}")
            return news_list
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
        
        
        
        

        
        
        
        