import asyncio
import logging
import numpy as np
from asyncio import Lock
from datetime import datetime, time

from api.deriv_api import DerivAPI
from config import config
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals


def compute_volatility(candle_closes, window=10):
    """
    Computes standard deviation of returns over a given window.
    Used for potential volatility checks, if needed.
    """
    if len(candle_closes) < window:
        return 0.0
    returns = []
    for i in range(1, len(candle_closes)):
        prev_close = candle_closes[i - 1]
        current_close = candle_closes[i]
        if prev_close != 0.0:
            ret = (current_close - prev_close) / prev_close
            returns.append(ret)
    recent_returns = returns[-window:]
    if not recent_returns:
        return 0.0
    return float(np.std(recent_returns))


class TradingStrategy:
    """
    TradingStrategy class that:
      - Fetches candle data
      - Generates a rule-based micro-scalper signal
      - Gets a price proposal from Deriv
      - Executes a real buy contract
      - Monitors the contract via 'proposal_open_contract' updates
      - Enforces price-based or profit-based SL/TP/trailing stops
    """

    def __init__(
        self,
        logger: logging.Logger,
        deriv_api_client: DerivAPI,
        loginid: str = "Unknown"
    ):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.lock = Lock()
        self.loginid = loginid

        # Tick buffers
        self.tick_times = []
        self.tick_prices = []

        # Candle buffers
        self.candle_times = []
        self.candle_opens = []
        self.candle_highs = []
        self.candle_lows = []
        self.candle_closes = []
        self.candle_volumes = []

        # Strategy signals
        self.rule_signals = RuleBasedSignals(logger=logger)

        # Account info
        self.latest_balance = 0.0
        self.currency = "USD"
        self.daily_start_balance = 0.0
        self.daily_pl = 0.0
        self.account_info_logged = False

        # P/L tracking
        self.total_wins = 0
        self.total_losses = 0
        self.cumulative_profit = 0.0
        self.cumulative_loss = 0.0
        self.trades_data = []
        self.pl_times = []
        self.pl_values = []
        self.consecutive_losses = 0

        # Risk
        self.risk_manager = RiskManager(logger)
        self.daily_loss_limit_percent = config.DAILY_LOSS_LIMIT_PERCENT
        self.daily_profit_target_percent = config.DAILY_PROFIT_TARGET_PERCENT
        self.daily_loss_limit_reached = False
        self.daily_profit_target_reached = False
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES

        # Scheduling
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)
        self.last_strategy_evaluation = datetime.now()

        # Limits & dynamic fields
        self.MAX_SCALPING_STAKE = 1.0
        self.dynamic_stop_loss_points = 0.0
        self.dynamic_take_profit_points = 0.0
        self.dynamic_trailing_stop_points = 0.0
        self.dynamic_position_size = 0.0

        # Basic contract state
        self.dynamic_duration = config.DURATION
        self.dynamic_contract_type = "MULTIPLIER"

        # Active symbols
        self.active_symbols = []
        self.current_symbol = config.SYMBOL

        # Storage for open contracts
        self.contracts_closed = {}
        self.contract_spot_info = {}
        self.trade_open_times = {}

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

    def set_active_symbols(self, symbols):
        self.active_symbols = symbols
        self.logger.info(f"Active symbols set. Total symbols: {len(symbols)}")

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

    async def get_price_proposal(self, symbol: str, direction: str):
        """
        Request a price proposal from Deriv's API for a trade setup
        using your dynamic_position_size and dynamic_contract_type.
        """
        try:
            proposal_request = {
                "proposal": 1,
                "amount": self.dynamic_position_size,
                "basis": "stake",
                "contract_type": self.dynamic_contract_type,
                "currency": config.CURRENCY,
                "symbol": symbol,
                "multiplier"    : config.MULTIPLIER
               
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

    async def evaluate_and_execute_trade(self, symbol: str):
        """
        Main entry point for evaluating the rule-based micro-scalper and
        placing a real trade on Deriv (with get_price_proposal + buy).
        """
        # 1) Daily checks
        #if self.daily_risk_check():
           # self.logger.warning("Daily limit or profit target reached => skipping trade.")
            #return

        # 2) Consecutive loss checks
        #if self.consecutive_losses >= self.max_consecutive_losses:
            #self.logger.warning(f"Consecutive loss limit ({self.consecutive_losses}) reached => skipping trade.")
           # return

        # 3) Fetch updated candle data
        await self.fetch_candle_history(symbol)

        # 4) Generate rule-based signal
        rule_signal_data = self.rule_signals.generate_candle_signal(
            candle_times=self.candle_times,
            candle_opens=self.candle_opens,
            candle_highs=self.candle_highs,
            candle_lows=self.candle_lows,
            candle_closes=self.candle_closes,
            candle_volumes=self.candle_volumes,
            account_balance=self.latest_balance
        )
        self.last_strategy_evaluation = datetime.now()

        if rule_signal_data["signal"] == "HOLD":
            self.logger.info("Rule-based signal => HOLD. No trade executed.")
            return

        # 5) Extract price-based SL/TP/trailing
        self.dynamic_stop_loss_points = rule_signal_data["stop_loss_points"]
        self.dynamic_take_profit_points = rule_signal_data["take_profit_points"]
        self.dynamic_trailing_stop_points = rule_signal_data["trailing_stop_points"]
        self.dynamic_position_size = rule_signal_data["position_size"]

        direction = rule_signal_data["signal"].upper()
        if direction == "BUY":
            self.dynamic_contract_type = "MULTUP"
        else:
            self.dynamic_contract_type = "MULTDOWN"

        # 6) Get a real price proposal from Deriv
        price_proposal = await self.get_price_proposal(symbol, direction)
        if not price_proposal:
            self.logger.error("No price proposal => trade not executed.")
            return

        # 7) Execute a real buy
        ask_price = price_proposal["ask_price"]
        proposal_id = price_proposal["id"]
        try:
            buy_response = await self.deriv_api.buy({"buy": proposal_id, "price": ask_price})
            if "buy" not in buy_response:
                self.logger.error(f"Trade execution failed. Response={buy_response}")
                return
            contract_id = buy_response["buy"]["contract_id"]
            self.logger.info(f"Trade executed. Contract ID: {contract_id}")
        except Exception as e:
            self.logger.error(f"Error during trade execution: {e}")
            return

        # 8) Initialize local data for price-based monitoring
        start_spot = float(buy_response["buy"].get("spot", 0.0))
        self.contracts_closed[contract_id] = False
        self.contract_spot_info[contract_id] = {
            "entry_type": direction,
            "entry_spot": start_spot,
            "best_spot": start_spot
        }
        self.trade_open_times[contract_id] = datetime.now()

        self.logger.info(
            f"Contract={contract_id}, dir={direction}, start_spot={start_spot:.5f}, "
            f"SL_points={self.dynamic_stop_loss_points:.5f}, TP_points={self.dynamic_take_profit_points:.5f}, "
            f"trail_points={self.dynamic_trailing_stop_points:.5f}, stake={self.dynamic_position_size:.2f}"
        )

        # 9) Monitor the contract
        asyncio.create_task(self.monitor_trade_spot_based(contract_id))

    async def monitor_trade_spot_based(self, contract_id: str):
        self.logger.info(f"Monitoring trade (price-based) for {contract_id}")
        try:
            poc_stream = await self.deriv_api.subscribe(
                {"proposal_open_contract": 1, "contract_id": contract_id}
            )
        except Exception as e:
            self.logger.error(f"Error subscribing to open contract {contract_id}: {e}")
            return

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def on_next(update):
            asyncio.create_task(self.handle_spot_update(update, contract_id, future))

        def on_error(error):
            self.logger.error(f"Open contract error for {contract_id}: {error}")
            if not future.done():
                future.set_exception(error)

        def on_completed():
            if not future.done():
                future.set_result(None)

        subscription = poc_stream.subscribe(on_next=on_next, on_error=on_error, on_completed=on_completed)

        try:
            await future
        except Exception as ex:
            self.logger.error(f"Exception while monitoring {contract_id}: {ex}")
        finally:
            subscription.dispose()
            self.logger.info(f"Stopped monitoring trade {contract_id}")


    async def force_close_contract(self, contract_id: str, future):
        """
        Force-closes the contract via Deriv 'sell' request. 
        If successful, logs and sets the contract as closed.
        """
        if contract_id not in self.contracts_closed or self.contracts_closed[contract_id]:
            return

        try:
            self.logger.info(f"[force_close_contract] Selling contract {contract_id} now...")
            request = {"sell": contract_id, "price": 0}
            response = await self.deriv_api.sell(request)
            if response and "sell" in response:
                self.logger.info(f"[force_close_contract] {contract_id} sold successfully.")
            else:
                self.logger.error(f"[force_close_contract] Failed to sell {contract_id}. Response={response}")
        except Exception as e:
            self.logger.error(f"[force_close_contract] Error selling {contract_id}: {e}")
        finally:
            self.contracts_closed[contract_id] = True
            if contract_id in self.trade_open_times:
                del self.trade_open_times[contract_id]
            if not future.done():
                future.set_result(None)



    async def handle_spot_update(self, poc_update, contract_id: str, future):
        """
        Uses the 'current_spot' field from the open contract data to enforce
        price-based stop-loss, take-profit, and trailing stop logic.
        If 'entry_spot' was never set (0.0), we initialize it on the first valid
        'current_spot' update. This ensures we have a proper baseline for SL/TP checks.
        """
        if contract_id not in self.contracts_closed or self.contracts_closed[contract_id]:
            return

        poc_data = poc_update.get("proposal_open_contract", {})
        if not poc_data:
            return

        # If the contract is sold already, finalize P/L
        is_sold = poc_data.get("is_sold", False)
        if is_sold:
            final_profit = float(poc_data.get("profit", 0.0))
            self.logger.info(f"Contract {contract_id} closed by system. Final Profit: {final_profit}")
            self.contracts_closed[contract_id] = True
            await self.record_trade_results(final_profit, contract_id)
            if not future.done():
                future.set_result(None)
            await self.log_current_balance()
            return

        # Extract the current spot
        current_spot = float(poc_data.get("current_spot", 0.0))

        # Retrieve stored contract info
        entry_info = self.contract_spot_info.get(contract_id, {})
        entry_type = entry_info.get("entry_type", "BUY")   # "BUY" or "SELL"
        entry_spot = entry_info.get("entry_spot", 0.0)
        best_spot = entry_info.get("best_spot", entry_spot)

        # If entry_spot was never set properly, initialize now
        if entry_spot == 0.0 and current_spot != 0.0:
            self.logger.info(f"Initializing entry_spot for {contract_id} to {current_spot:.5f}")
            entry_info["entry_spot"] = current_spot
            entry_info["best_spot"] = current_spot
            entry_spot = current_spot
            best_spot = current_spot

        # Update best_spot if the trade moves further in our favor
        if entry_type == "BUY":
            if current_spot > best_spot:
                entry_info["best_spot"] = current_spot
                best_spot = current_spot
        else:  # "SELL"
            if current_spot < best_spot:
                entry_info["best_spot"] = current_spot
                best_spot = current_spot

        # Now apply your price-based SL, TP, trailing-stop
        # We'll assume self.dynamic_*_points are available in this class.

        if entry_type == "BUY" and entry_spot > 0.0:
            # Stop Loss
            if current_spot <= (entry_spot - self.dynamic_stop_loss_points):
                self.logger.info(
                    f"[StopLoss] BUY {contract_id}: current_spot={current_spot:.5f} <= "
                    f"(entry_spot={entry_spot:.5f} - SL={self.dynamic_stop_loss_points:.5f})"
                )
                await self.force_close_contract(contract_id, future)
                return

            # Take Profit
            if current_spot >= (entry_spot + self.dynamic_take_profit_points):
                self.logger.info(
                    f"[TakeProfit] BUY {contract_id}: current_spot={current_spot:.5f} >= "
                    f"(entry_spot={entry_spot:.5f} + TP={self.dynamic_take_profit_points:.5f})"
                )
                await self.force_close_contract(contract_id, future)
                return

            # Trailing Stop
            if self.dynamic_trailing_stop_points > 0.0 and best_spot > entry_spot:
                if current_spot < (best_spot - self.dynamic_trailing_stop_points):
                    self.logger.info(
                        f"[TrailingStop] BUY {contract_id}: current_spot={current_spot:.5f} < "
                        f"(best_spot={best_spot:.5f} - trail={self.dynamic_trailing_stop_points:.5f})"
                    )
                    await self.force_close_contract(contract_id, future)
                    return

        elif entry_type == "SELL" and entry_spot > 0.0:
            # Stop Loss
            if current_spot >= (entry_spot + self.dynamic_stop_loss_points):
                self.logger.info(
                    f"[StopLoss] SELL {contract_id}: current_spot={current_spot:.5f} >= "
                    f"(entry_spot={entry_spot:.5f} + SL={self.dynamic_stop_loss_points:.5f})"
                )
                await self.force_close_contract(contract_id, future)
                return

            # Take Profit
            if current_spot <= (entry_spot - self.dynamic_take_profit_points):
                self.logger.info(
                    f"[TakeProfit] SELL {contract_id}: current_spot={current_spot:.5f} <= "
                    f"(entry_spot={entry_spot:.5f} - TP={self.dynamic_take_profit_points:.5f})"
                )
                await self.force_close_contract(contract_id, future)
                return

            # Trailing Stop
            if self.dynamic_trailing_stop_points > 0.0 and best_spot < entry_spot:
                if current_spot > (best_spot + self.dynamic_trailing_stop_points):
                    self.logger.info(
                        f"[TrailingStop] SELL {contract_id}: current_spot={current_spot:.5f} > "
                        f"(best_spot={best_spot:.5f} + trail={self.dynamic_trailing_stop_points:.5f})"
                    )
                    await self.force_close_contract(contract_id, future)
                    return

        # Check price-based SL/TP/trailing => similar to code from previous examples
        # e.g. if current_spot <= entry_spot - dynamic_stop_loss_points => close, etc.
        # The snippet is omitted for brevity. Add your own logic here.

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
            self.logger.info("Trade closed with no profit or loss.")

        net_pl = self.cumulative_profit - self.cumulative_loss
        current_time = datetime.now()
        self.pl_times.append(current_time)
        self.pl_values.append(net_pl)
        self.daily_pl = self.latest_balance - self.daily_start_balance
        self.logger.info(
            f"Updated daily P/L: {self.daily_pl:.2f}, "
            f"Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}"
        )

    # Optional: If you want a separate 'sell_contract' or 'force_close_contract' method, place it here.
