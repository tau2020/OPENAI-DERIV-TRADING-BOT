import asyncio
import uuid
import json
import logging
import websockets
from typing import Any, Callable, Dict, Optional
from config import config

class DerivAPI:
    """
    Handles communication with the Deriv API.
    """

    def __init__(self, api_url: str, api_token: str, logger: logging.Logger):
        self.api_url = api_url
        self.api_token = api_token
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.logger = logger
        self.rate_limit = asyncio.Semaphore(config.MAX_API_CALLS_PER_SECOND)
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.subscriptions: Dict[str, Callable] = {}
        self.receiver_task: Optional[asyncio.Task] = None
        self.connected = False
        self.should_reconnect = True
        self.reconnect_delay = 5  # Delay in seconds before attempting to reconnect

    async def connect(self):
        """
        Establishes a connection to the Deriv API and authorizes the session.
        """
        while self.should_reconnect:
            try:
                self.logger.info("Attempting to connect to Deriv API.")
                async with websockets.connect(
                    self.api_url,
                    ping_interval=None,
                ) as websocket:
                    self.connection = websocket

                    # Authorize with API token
                    auth_request = {"authorize": self.api_token}
                    self.logger.debug(f"Sending authorization request: {auth_request}")
                    response = await self.send_request(auth_request)
                    if response and 'authorize' in response:
                        self.logger.info("Successfully authorized with Deriv API.")
                        self.connected = True

                        # Start the message receiver loop
                        self.receiver_task = asyncio.create_task(self.receive_messages())
                        self.logger.debug("Message receiver task started.")

                        # Resubscribe to previous subscriptions
                        await self.resubscribe_all()

                        # Keep the connection open
                        await self.receiver_task
                    else:
                        self.logger.error("Authorization failed.")
                        self.connected = False
                        await self.close()
            except Exception as e:
                self.logger.error(f"Failed to connect to Deriv API: {e}")
                self.connected = False
                await self.close()

            self.logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
            await asyncio.sleep(self.reconnect_delay)

    async def send_request(self, request: dict) -> Optional[dict]:
        """
        Sends a request to the Deriv API with rate limiting.
        """
        async with self.rate_limit:
            try:
                if 'req_id' not in request:
                    request['req_id'] = uuid.uuid4().int & (1 << 32) - 1
                req_id = request['req_id']
                future = asyncio.get_running_loop().create_future()
                self.pending_requests[req_id] = future
                self.logger.debug(f"Sending request: {request}")
                await self.connection.send(json.dumps(request))
                self.logger.debug(f"Request sent: {request}")
                response = await future
                self.logger.debug(f"Response received for req_id {req_id}: {response}")
                return response
            except Exception as e:
                self.logger.error(f"Error sending request: {e}")
                return None

    async def receive_messages(self):
        """
        Receives messages from the Deriv API and dispatches them to the appropriate handlers.
        """
        try:
            self.logger.debug("Message receiver loop started.")
            async for message in self.connection:
                response_data = json.loads(message)
                self.logger.debug(f"Received message: {response_data}")

                # Handle different types of messages
                if 'req_id' in response_data:
                    await self.handle_response(response_data)
                elif 'tick' in response_data:
                    await self.handle_subscription(response_data)
                elif response_data.get('msg_type') == 'ping':
                    self.logger.debug("Ping response received.")
                elif 'error' in response_data:
                    await self.handle_error(response_data)
                else:
                    self.logger.warning(f"Unhandled message: {response_data}")
        except asyncio.CancelledError:
            self.logger.debug("Message receiver task cancelled.")
        except Exception as e:
            self.logger.error(f"Error in receive_messages: {e}")
        finally:
            self.logger.debug("Message receiver loop terminated.")
            self.connected = False
            await self.close()

    async def handle_response(self, response_data: dict):
        """
        Handles responses matched by req_id.
        """
        req_id = response_data['req_id']
        future = self.pending_requests.pop(req_id, None)
        if future:
            if 'error' in response_data:
                self.logger.error(f"Error in response for req_id {req_id}: {response_data['error']}")
                future.set_exception(Exception(response_data['error']['message']))
            else:
                self.logger.debug(f"Setting result for req_id {req_id}.")
                future.set_result(response_data)
        else:
            self.logger.warning(f"Received response for unknown req_id: {req_id}")

    async def handle_subscription(self, response_data: dict):
        """
        Handles subscription updates such as tick data.
        """
        if 'tick' in response_data:
            symbol = response_data['tick']['symbol']
            callback = self.subscriptions.get(symbol)
            if callback:
                self.logger.debug(f"Dispatching tick data for symbol: {symbol}")
                await callback(response_data['tick'])
            else:
                self.logger.warning(f"No callback found for symbol: {symbol}")
        else:
            self.logger.debug(f"Unhandled subscription message: {response_data}")

    async def handle_error(self, response_data: dict):
        """
        Handles error messages from the API.
        """
        error = response_data.get('error', {})
        code = error.get('code', 'Unknown')
        message = error.get('message', 'No error message provided.')
        self.logger.error(f"API Error [{code}]: {message}")

    async def resubscribe_all(self):
        """
        Resubscribes to all previously subscribed symbols after reconnection.
        """
        self.logger.info("Resubscribing to all previous tick subscriptions.")
        subscriptions = self.subscriptions.copy()
        self.subscriptions.clear()
        for symbol, callback in subscriptions.items():
            await self.subscribe_ticks(symbol, callback)

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        """
        Subscribes to tick updates for a given symbol.
        """
        ticks_request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": uuid.uuid4().int & (1 << 32) - 1
        }
        self.logger.debug(f"Subscribing to ticks for symbol: {symbol}")
        response = await self.send_request(ticks_request)
        if response and 'tick' in response:
            subscribe_id = response['tick']['id']
            self.subscriptions[symbol] = callback
            self.logger.info(f"Subscribed to ticks for {symbol}, subscription ID: {subscribe_id}")
        elif response and 'error' in response:
            await self.handle_error(response)
        else:
            self.logger.error(f"Failed to subscribe to ticks for symbol: {symbol}")

    async def close(self):
        """
        Closes the connection to the Deriv API.
        """
        self.logger.debug("Closing connection to Deriv API.")
        self.connected = False
        if self.connection:
            try:
                await self.connection.close()
                self.logger.info("Closed connection to Deriv API.")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
                self.logger.debug("Message receiver task cancelled.")
            except asyncio.CancelledError:
                self.logger.debug("Message receiver task cancellation confirmed.")

    # Additional methods for API interactions
    async def get_active_symbols(self):
        """
        Retrieves a list of active symbols available for trading.
        """
        request = {"active_symbols": "full"}
        response = await self.send_request(request)
        if response and 'active_symbols' in response:
            return response['active_symbols']
        else:
            self.logger.error("Failed to retrieve active symbols.")
            return None

    async def get_contracts_for_symbol(self, symbol):
        """
        Retrieves available contracts for a given symbol.
        """
        request = {"contracts_for": symbol}
        response = await self.send_request(request)
        if response and 'contracts_for' in response:
            return response['contracts_for']
        else:
            self.logger.error(f"Failed to retrieve contracts for symbol: {symbol}")
            return None

    async def get_ticks_history(self, symbol, start_time, end_time):
        """
        Retrieves historical tick data for a given symbol.
        """
        request = {
            "ticks_history": symbol,
            "start": start_time,
            "end": end_time,
            "style": "ticks",
            "count": config.MAX_TICK_DATA_POINTS
        }
        response = await self.send_request(request)
        if response and 'history' in response:
            return response['history']
        else:
            self.logger.error(f"Failed to retrieve tick history for symbol: {symbol}")
            return None

    async def get_account_info(self):
        """
        Retrieves account information.
        """
        request = {"get_account_status": 1}
        response = await self.send_request(request)
        if response and 'get_account_status' in response:
            return response['get_account_status']
        else:
            self.logger.error("Failed to retrieve account information.")
            return None

    async def get_balance(self):
        """
        Retrieves account balance information.
        """
        request = {"balance": 1}
        response = await self.send_request(request)
        if response and 'balance' in response:
            return response
        else:
            self.logger.error("Failed to retrieve account balance.")
            return None

    async def get_contract_profit(self, contract_id):
        """
        Retrieves the profit of an open contract.
        """
        request = {"profit_table": 1, "contract_id": contract_id}
        response = await self.send_request(request)
        if response and 'profit_table' in response:
            # Extract profit information
            transactions = response['profit_table'].get('transactions', [])
            if transactions:
                profit = float(transactions[0].get('profit', 0.0))
                return profit
            else:
                return None
        else:
            self.logger.error(f"Failed to retrieve profit for contract ID: {contract_id}")
            return None

    async def sell_contract(self, contract_id):
        """
        Sells an open contract to close the position.
        """
        request = {"sell": contract_id, "price": 0}
        response = await self.send_request(request)
        if response and 'sell' in response:
            self.logger.info(f"Contract {contract_id} sold successfully.")
        else:
            self.logger.error(f"Failed to sell contract ID: {contract_id}")
