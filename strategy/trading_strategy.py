# strategy/trading_strategy.py

import logging
from datetime import datetime
import asyncio
import statistics
import threading
import matplotlib.pyplot as plt
import io
import requests
from api.deriv_api import DerivAPI  # Import DerivAPI class directly
from config import config
from utils.openai_utils import initialize_openai, get_trading_signal_from_openai
from utils.risk_manager import RiskManager
from utils.anthropic_utils import ClaudeTradingClient

class TradingStrategy:

    def __init__(self, logger, deriv_api_client: DerivAPI, openai_api_key: str, anthropic_api_key: str):
        """
        Initialize the TradingStrategy with required dependencies.
        :param logger: Logger instance
        :param deriv_api_client: An instance of DerivAPI class
        :param openai_api_key: API key for OpenAI
        :param anthropic_api_key: API key for Anthropic
        """
        self.logger = logger
        self.deriv_api = deriv_api_client  # Now clearly named and typed
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

        # Dynamic trade parameters
        self.dynamic_stop_loss = 0.0  
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 5
        self.dynamic_trailing_stop = 0.0
        self.dynamic_contract_type = "CALL"
        self.dynamic_duration = config.DURATION
        self.currency = 'USD'

        # Track best profit for trailing stop logic
        self.best_profit = 0.0
        self.entry_price = 0.0

        # Initialize AI clients
        self.openai_client = initialize_openai(openai_api_key, self.logger)
        self.claude_client = ClaudeTradingClient(logger=self.logger, api_key=anthropic_api_key)

        # Initialize cumulative P/L trackers
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.pl_times = []
        self.pl_values = []
        self.risk_manager = RiskManager(logger)

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

    async def evaluate_and_execute_trade(self, symbol):
        self.last_strategy_evaluation = datetime.now()
        await self.fetch_historical_data(symbol)

        signal, confidence = await self.evaluate_strategy()
        self.logger.info(f"Strategy evaluated. Final Signal: {signal}, Confidence: {confidence}")

        if signal in ["BUY", "SELL"]:
            price_proposal = await self.get_price_proposal(symbol, signal)
            if price_proposal:
                await self.execute_trade(symbol, signal, price_proposal)
            else:
                self.logger.error("Failed to get price proposal. Trade not executed.")
        else:
            self.logger.info("No trade action taken.")

    async def fetch_historical_data(self, symbol):
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
                self.logger.info(f"Historical data fetched for symbol: {symbol}")
                with self.lock:
                    times = [datetime.fromtimestamp(int(ts)) for ts in history.get('times', [])]
                    prices = [float(price) for price in history.get('prices', [])]
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    self.tick_times.extend(times)
                    self.tick_prices.extend(prices)
            else:
                self.logger.error(f"Failed to fetch historical data for symbol: {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")

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
        polygon_data = "No Polygon data available."
        news_headlines = []

        openai_result = await get_trading_signal_from_openai(
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

       

        openai_signal = openai_result["signal"]
        openai_confidence = openai_result["confidence"]
       

        self.logger.info(f"OpenAI recommendation: {openai_result}")
     

        
        final_result = openai_result
        signal = final_result["signal"]
        self.dynamic_stop_loss = final_result["stop_loss"]
        self.dynamic_take_profit = final_result["take_profit"]
        self.dynamic_confidence = final_result["confidence"]
        self.dynamic_position_size = final_result["position_size"] if final_result["position_size"] > 0 else config.AMOUNT
        self.dynamic_max_holding_time = final_result["max_holding_time"]
        self.dynamic_trailing_stop = final_result["trailing_stop"]
        self.confidence = final_result["confidence"]
        self.dynamic_contract_type = final_result.get("contract_type", "CALL")  
        self.dynamic_duration = final_result.get("duration", config.DURATION)

        self.logger.info(
            f"Final decision: {signal}, SL={self.dynamic_stop_loss}, "
            f"TP={self.dynamic_take_profit}, Confidence={self.dynamic_confidence}, "
            f"PositionSize={self.dynamic_position_size}, MaxHoldingTime={self.dynamic_max_holding_time}, "
            f"TrailingStop={self.dynamic_trailing_stop}, ContractType={self.dynamic_contract_type}, "
            f"Duration={self.dynamic_duration}"
        )

        return signal, self.confidence

    def calculate_recent_volume(self, lookback=50):
        with self.lock:
            approx_volume = min(len(self.tick_prices), lookback)
        return [approx_volume for _ in range(5)]

    def calculate_volatility(self, prices):
        if len(prices) < 2:
            return 0.0
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0]
        if not returns:
            return 0.0
        return statistics.pstdev(returns)

    def get_market_sentiment(self, window=20):
        with self.lock:
            window_prices = self.tick_prices[-window:] if len(self.tick_prices) >= window else self.tick_prices[:]
        if not window_prices:
            return 0.0
        avg_price = sum(window_prices) / len(window_prices)
        current_price = window_prices[-1]
        if avg_price == 0:
            return 0.0
        raw_sentiment = (current_price - avg_price) / avg_price
        return max(-1.0, min(1.0, raw_sentiment))

    def calculate_win_loss_ratio(self):
        wins = self.total_wins
        losses = self.total_losses
        if wins == 0 and losses == 0:
            return 1.0
        if losses == 0:
            return float(wins)
        return wins / losses

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
                self.logger.info(f"Price proposal received: {response['proposal']['ask_price']}")
                return response['proposal']
            else:
                self.logger.error("Failed to get price proposal.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting price proposal: {e}")
            return None

    async def execute_trade(self, symbol, signal, price_proposal):
        proposal_id = price_proposal['id']
        self.logger.info(f"Executing trade: Symbol={symbol}, Signal={signal}, Amount={self.dynamic_position_size}, ContractType={self.dynamic_contract_type}, Duration={self.dynamic_duration}")

        try:
            response = await self.deriv_api.buy({"buy": proposal_id, "price": price_proposal['ask_price']})
            if response and 'buy' in response:
                contract_id = response['buy']['contract_id']
                self.logger.info(f"Trade executed. Contract ID: {contract_id}")

                self.entry_price = float(price_proposal['ask_price'])
                self.best_profit = 0.0

                await self.monitor_trade(contract_id)
            else:
                self.logger.error("Trade execution failed.")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")

    async def monitor_trade(self, contract_id):
        self.logger.info(f"Monitoring trade. Contract ID: {contract_id}")
        
        try:
            poc_stream = await self.deriv_api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
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
        
        # Update best profit
        if profit > self.best_profit:
            self.best_profit = profit

        # Check conditions
        if profit <= -abs(self.dynamic_stop_loss) and not is_sold:
            self.logger.info(f"Stop loss hit for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)

        if profit >= self.dynamic_take_profit and not is_sold:
            self.logger.info(f"Take profit reached for contract {contract_id}. Selling contract.")
            await self.sell_contract(contract_id, future)
            
            
                    

        if is_sold:
            self.logger.info(f"Contract {contract_id} settled. Profit: {profit}")
            self.record_trade_results(profit)
            await self.log_current_balance()
            if not future.done():
                future.set_result(None)

            
                
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
            if not future.done():
                future.set_result(None)
           

    def record_trade_results(self, profit):
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

    def set_active_symbols(self, symbols):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")

    def fetch_recent_news(self, symbol):
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
