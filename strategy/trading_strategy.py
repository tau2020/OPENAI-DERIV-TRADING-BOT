import asyncio
import logging
import numpy as np
from asyncio import Lock
from datetime import datetime, time
from typing import List, Dict, Optional
from collections import deque

from api.deriv_api import DerivAPI
from config import config
from utils.risk_manager import RiskManager
from strategy.rule_based_signals import RuleBasedSignals
from typing import DefaultDict
from collections import defaultdict


class TradingStrategy:
    # --------------------------------------------------------------------- #
    # INITIALISATION
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        logger: logging.Logger,
        deriv_api_client: DerivAPI,
        loginid: str = "Unknown",
    ):
        self.logger = logger
        self.deriv_api = deriv_api_client
        self.loginid = loginid

        # ------------- Concurrency ---------------------------------------- #
        self.lock = Lock()
        self.trade_lock = Lock()
        self.balance_lock = Lock()
        self.trailing_stop_lock = Lock()

        # ------------- Market-data buffers (deque → faster & bounded) ----- #
        max_pts = config.MAX_TICK_DATA_POINTS
        self.tick_times: deque[datetime] = deque(maxlen=max_pts)
        self.tick_prices: deque[float] = deque(maxlen=max_pts)

        self.candle_times: deque[datetime] = deque(maxlen=max_pts)
        self.candle_opens: deque[float] = deque(maxlen=max_pts)
        self.candle_highs: deque[float] = deque(maxlen=max_pts)
        self.candle_lows: deque[float] = deque(maxlen=max_pts)
        self.candle_closes: deque[float] = deque(maxlen=max_pts)
        self.candle_volumes: deque[float] = deque(maxlen=max_pts)

        # ------------- Trading state -------------------------------------- #
        self.active_symbols: List[str] = []
        self.current_symbol = config.SYMBOL
        self.contracts_closed: Dict[str, bool] = {}
        self.trade_open_times: Dict[str, datetime] = {}
        self.active_trades: Dict[str, Dict] = {}
        self.trade_directions: Dict[str, str] = {}          # NEW

        # ------------- Account info --------------------------------------- #
        self.latest_balance = 0.0
        self.currency = "USD"
        self.daily_start_balance = 0.0
        self.account_info_logged = False
        self.best_prices: DefaultDict[str, float] = defaultdict(float)
        # ------------- Risk management ------------------------------------ #
        self.risk_manager = RiskManager(logger)
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
        self.daily_loss_limit_percent = config.DAILY_LOSS_LIMIT_PERCENT
        self.daily_profit_target_percent = config.DAILY_PROFIT_TARGET_PERCENT

        # ------------- Dynamic trade params ------------------------------- #
        self.dynamic_stop_loss = 0.0
        self.dynamic_take_profit = 0.0
        self.dynamic_confidence = 0.0
        self.dynamic_position_size = 0.0
        self.dynamic_max_holding_time = 3       # minutes
        self.dynamic_trailing_stop = 0.0        # adaptive, set later
        self.dynamic_contract_type = ""
        self.dynamic_duration = config.DURATION
        self.entry_price = 0.0
        self.best_price = 0.0                   # renamed (best_profit → best_price)

        # ------------- Performance tracking ------------------------------- #
        self.total_wins = self.total_losses = 0
        self.cumulative_profit = self.cumulative_loss = 0.0
        self.trades_data: List[Dict] = []
        self.pl_times: List[datetime] = []
        self.pl_values: List[float] = []

        # ------------- Market metrics ------------------------------------- #
        self.volatility = 0.0
        self.average_true_range = 0.0
        self.allowed_trading_start = time(0, 0)
        self.allowed_trading_end = time(23, 59)
        self.last_strategy_evaluation = datetime.now()
        self.last_volatility_update = datetime.now()

        # ------------- Configurable factors ------------------------------- #
        self.multiplier_factor = getattr(config, "MULTIPLIER_FACTOR", 4000)

        # ------------- Signal engine -------------------------------------- #
        self.rule_signals = RuleBasedSignals(logger=self.logger)

    # --------------------------------------------------------------------- #
    # CORE STRATEGY METHODS
    # --------------------------------------------------------------------- #
    async def tick_callback(self, tick: Dict):
        """Handle real-time tick updates (throttled)."""
        tick_data = tick.get("tick", {})
        symbol = tick_data.get("symbol")
        price = float(tick_data.get("quote", 0.0))
        epoch = tick_data.get("epoch", 0)

        await self.append_tick_data(datetime.fromtimestamp(epoch), price)

        if not self.account_info_logged:
            await self.log_account_info()
            self.account_info_logged = True

        if self.should_evaluate_strategy():
            asyncio.create_task(self.evaluate_and_execute_trade(symbol))

    async def evaluate_and_execute_trade(self, symbol: str):
        """Main decision loop."""
        self.last_strategy_evaluation = datetime.now()

        if self.daily_risk_check() or self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning("Risk guard active – skipping evaluation")
            return

        signal, confidence = await self.evaluate_strategy(symbol)
        self.logger.info(f"Eval → {signal} ({confidence:.2f})")

        if signal not in ("BUY", "SELL") or confidence < self.rule_signals.min_confidence:
            return

        await self.execute_trade_safely(symbol, signal)

    async def execute_trade_safely(self, symbol: str, signal: str):
        """Atomic trade submit with retries & locking."""
        proposal = await self.get_price_proposal(symbol, signal)
        if not proposal:
            return

        try:
            async with self.trade_lock:
                resp = await self.deriv_api.buy({"buy": proposal["id"], "price": proposal["ask_price"]})
                if "buy" not in resp:
                    raise RuntimeError("No buy receipt")

                cid = resp["buy"]["contract_id"]
                self.trade_directions[cid] = signal          # track side
                self.initialize_trade(cid, proposal)
                asyncio.create_task(self.monitor_trade(cid))
        except Exception as e:
            self.logger.error(f"Buy error → {e}")
            await self.retry_trade_execution(symbol, signal)

    async def monitor_trade(self, contract_id: str):
        """Stream POC → exit logic."""
        self.logger.info(f"Monitor {contract_id}")
        try:
            poc_stream = await self.deriv_api.subscribe(
                {"proposal_open_contract": 1, "contract_id": contract_id}
            )
            fut = asyncio.get_event_loop().create_future()
            poc_stream.subscribe(
                on_next=lambda poc: asyncio.create_task(
                    self.handle_poc_update(poc, contract_id, fut)
                ),
                on_error=lambda e: self.logger.error(f"POC stream err: {e}"),
                on_completed=lambda: fut.set_result(None),
            )
            await fut
        finally:
            await self.cleanup_trade(contract_id)

    async def handle_poc_update(self, poc_update: Dict, cid: str, fut: asyncio.Future):
            """
            Stream update handler for a single open contract.
            • Keeps per-trade best price
            • Executes trailing-stop
            • NEW: closes immediately when profit ≥ $5
            """
            # Skip if we’ve already closed / finalised this contract
            if self.contracts_closed.get(cid):
                return

            poc = poc_update.get("proposal_open_contract", {})
            if not poc or not poc.get("is_valid_to_sell", False):
                # Contract not ready (Deriv still “waiting for entry tick”)
                return

            cur_price = float(poc.get("current_spot", 0.0))
            profit    = float(poc.get("profit", 0.0))
            side      = self.trade_directions.get(cid, "BUY")   # BUY ↔ CALL, SELL ↔ PUT

            # ── track extremum to drive trailing stop ───────────────────────── #
            if side == "BUY":
                self.best_prices[cid] = max(self.best_prices[cid] or cur_price, cur_price)
                trail_level = self.best_prices[cid] - self.dynamic_trailing_stop
                stop_hit    = cur_price <= trail_level
            else:  # SELL
                self.best_prices[cid] = min(self.best_prices[cid] or cur_price, cur_price)
                trail_level = self.best_prices[cid] + self.dynamic_trailing_stop
                stop_hit    = cur_price >= trail_level

            # ── exit conditions ─────────────────────────────────────────────── #
            if stop_hit:
                await self.execute_exit(cid, f"Trailing-stop @ {trail_level:.4f}")
                return

            if profit >= 5.0:                              # ← NEW auto-TP
                await self.execute_exit(cid, f"Auto-TP +$5 reached ({profit:.2f})")
                return

            if poc.get("is_sold", False):                  # closed on server side
                await self.finalize_trade(cid, profit, fut)
                return

            # Optional: update volatility-based params & time / fixed SL-TP checks
            await self.update_dynamic_parameters()
            await self.check_exit_conditions(cid, profit)



    # --------------------------------------------------------------------- #
    # ENHANCED FEATURES
    # --------------------------------------------------------------------- #
    async def update_dynamic_parameters(self):
        """Refresh ATR & trailing distance every 5 min."""
        if (datetime.now() - self.last_volatility_update).total_seconds() > 300:
            self.volatility = await self.calculate_market_volatility()
            self.average_true_range = self.calculate_atr()
            self.last_volatility_update = datetime.now()

        new_trail = max(self.average_true_range * 1.5, 3.0)
        async with self.trailing_stop_lock:
            if abs(new_trail - self.dynamic_trailing_stop) / (self.dynamic_trailing_stop or 1e-9) > 0.2:
                self.dynamic_trailing_stop = new_trail
                self.logger.debug(f"Trailing-stop → {self.dynamic_trailing_stop:.4f}")

    async def calculate_market_volatility(self) -> float:
        arr = np.asarray(self.candle_closes, dtype=np.float64)[-50:]
        if arr.size < 2:
            return 0.0
        return float(np.diff(np.log(arr)).std() * np.sqrt(252))

    def calculate_atr(self) -> float:
        n = self.rule_signals.atr_window
        if len(self.candle_highs) < n + 1:
            return 0.0

        h = np.asarray(self.candle_highs)[-n - 1 :]
        l = np.asarray(self.candle_lows)[-n - 1 :]
        c = np.asarray(self.candle_closes)[-n - 1 :]

        tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
        return float(tr.mean())


        # ---------------------------------------------------------------------
    # EXIT-LOGIC HELPERS  🔻 optimised
    # ---------------------------------------------------------------------
    async def check_exit_conditions(self, contract_id: str, profit: float) -> None:
        """
        Evaluate fixed SL/TP, max-holding-time (and any additional rules)
        then trigger execute_exit() once a condition is met.
        """
        if contract_id in self.contracts_closed:        # already handled
            return

        exit_reason: Optional[str] = None

        # --- fixed SL / TP ------------------------------------------------ #
        if profit <= -abs(self.dynamic_stop_loss):
            exit_reason = f"⬇ stop-loss hit ({profit:.2f})"
        elif profit >= self.dynamic_take_profit:
            exit_reason = f"⬆ take-profit hit ({profit:.2f})"

        # --- time-based exit --------------------------------------------- #
        if exit_reason is None and await self.check_holding_time(contract_id):
            exit_reason = "⏰ max hold elapsed"

        # (trailing-stop in handle_poc_update covers price-based trail)

        if exit_reason:
            await self.execute_exit(contract_id, exit_reason)

    async def check_holding_time(self, contract_id: str) -> bool:
        entry_time = self.trade_open_times.get(contract_id)
        if not entry_time:
            return False
        held_secs = (datetime.now() - entry_time).total_seconds()
        return held_secs > self.dynamic_max_holding_time * 60

    async def execute_exit(self, contract_id: str, reason: str) -> None:
        """Idempotent exit wrapper with single entry-point."""
        if self.contracts_closed.get(contract_id):
            return
        self.logger.info(f"Exit {contract_id} → {reason}")
        await self.retry_sell_contract(contract_id)

    async def retry_sell_contract(self, contract_id: str, attempts: int = 4) -> None:
        """Exponential back-off sell retries."""
        for i in range(attempts):
            try:
                await self.sell_contract(contract_id)
                return
            except Exception as e:
                delay = 1.5 ** i
                self.logger.warning(f"Sell retry {i+1}/{attempts} ({delay:.1f}s) → {e}")
                await asyncio.sleep(delay)
        self.logger.error(f"‼ sell failed after {attempts} attempts ({contract_id})")


    # ---------------------------------------------------------------------
    # CORE-FUNCTIONALITY  🔻 optimised methods
    # ---------------------------------------------------------------------
    async def append_tick_data(self, timestamp: datetime, price: float) -> None:
        """
        O(1) push into fixed-size deques – no manual trimming.
        """
        async with self.lock:
            self.tick_times.append(timestamp)   # deque(maxlen=…) auto-pops left
            self.tick_prices.append(price)


    async def log_account_info(self) -> None:
        """
        Fetch account status & balance in parallel, cache first fetch for the session.
        """
        try:
            status_resp, bal_resp = await asyncio.gather(
                self.deriv_api.get_account_status(),
                self.deriv_api.balance(),
            )

            acc_status = status_resp.get("get_account_status", {})
            bal_data   = bal_resp.get("balance", {})
            if not acc_status or not bal_data:
                self.logger.error("Account info fetch failed.")
                return

            balance   = float(bal_data.get("balance", 0.0))
            currency  = bal_data.get("currency", "USD")
            acc_type  = "Real" if acc_status.get("is_virtual", 1) == 0 else "Demo"

            async with self.lock:
                self.latest_balance = balance
                self.currency = currency
                if self.daily_start_balance == 0.0:
                    self.daily_start_balance = balance

            self.logger.info(f"Balance: {balance} {currency} | Type: {acc_type} | Day-start: {self.daily_start_balance}")
        except Exception as e:
            self.logger.error(f"Acct-info error → {e}")


    async def log_current_balance(self, retries: int = 3) -> None:
        """
        Lightweight balance poll with retry/back-off.
        """
        for i in range(retries):
            try:
                resp = await self.deriv_api.balance()
                if "balance" not in resp:
                    raise ValueError("missing 'balance'")
                bal = resp["balance"]
                async with self.lock:
                    self.latest_balance = float(bal.get("balance", 0.0))
                    self.currency       = bal.get("currency", "USD")
                self.logger.debug(f"Balance ↻ {self.latest_balance} {self.currency}")
                return
            except Exception as e:
                self.logger.warning(f"Bal retry {i+1}/{retries} → {e}")
                await asyncio.sleep(1.5 ** i)
        self.logger.error("Balance fetch failed.")


    def should_evaluate_strategy(self) -> bool:
        now = datetime.now()
        if (now - self.last_strategy_evaluation).total_seconds() < config.STRATEGY_EVALUATION_INTERVAL:
            return False
        if not (self.allowed_trading_start <= now.time() <= self.allowed_trading_end):
            return False
        return True


    def daily_risk_check(self) -> bool:
        """
        Drawdown / incremental-loss guard with peak balance tracking.
        """
        self.peak_balance = max(getattr(self, "peak_balance", self.daily_start_balance), self.latest_balance)
        dd = (self.peak_balance - self.latest_balance) / self.peak_balance
        if dd >= 0.03:
            self.logger.warning(f"⚠ max-drawdown {dd:.2%}")
            return True

        if (self.daily_start_balance - self.latest_balance) >= (self.daily_start_balance * 0.01):
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                self.logger.warning("⚠ consecutive loss guard")
                return True
        else:
            self.consecutive_losses = 0
        return False


    async def evaluate_strategy(self, symbol: str):
        """
        Concurrent historical fetch → rule-signal evaluation.
        """
        try:
            await asyncio.gather(
                self.fetch_historical_data(symbol, use_candles=True),
                self.fetch_candle_history(symbol),
            )
        except Exception as e:
            self.logger.error(f"Hist fetch error → {e}")
            return "HOLD", 0.0

        if len(self.candle_closes) < 8:
            return "HOLD", 0.0

        candle_data = [
            {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
            for o, h, l, c, v in zip(
                self.candle_opens,
                self.candle_highs,
                self.candle_lows,
                self.candle_closes,
                self.candle_volumes,
            )
        ]

        self.rule_signals.account_balance = self.latest_balance
        sig = self.rule_signals.generate_candle_signal(candle_data)
        if sig["signal"] == "HOLD":
            return "HOLD", 0.0

        # apply dynamic params
        self.dynamic_stop_loss        = sig["stop_loss"]
        self.dynamic_take_profit      = sig["take_profit"]
        self.dynamic_confidence       = sig["confidence"]
        self.dynamic_position_size    = round(sig["position_size"] * self.multiplier_factor, 2)
        self.dynamic_max_holding_time = sig["max_holding_time"]
        self.dynamic_trailing_stop    = sig["trailing_stop"]
        self.dynamic_contract_type    = "MULTUP" if sig["signal"] == "BUY" else "MULTDOWN"

        self.logger.info(f"▶ {sig['signal']} | SL {self.dynamic_stop_loss:.4f} TP {self.dynamic_take_profit:.4f} "
                        f"Conf {self.dynamic_confidence:.2f} Size {self.dynamic_position_size}")
        return sig["signal"], sig["confidence"]


    # ---------------------------------------------------------------------
    #  📈  MARKET-DATA + QUOTE HELPERS  (optimised)
    # ---------------------------------------------------------------------
    # 1) HISTORICAL DATA  ──────────────────────────────────────────────────
    async def fetch_historical_data(
        self,
        symbol: str,
        use_candles: bool = False,
        cache_secs: int = 30,
    ) -> None:
        """
        Unified tick / candle fetch with lightweight caching and deque-friendly
        updates.  When `use_candles=True` we ALSO populate the candle buffers,
        so `fetch_candle_history()` becomes a thin wrapper (kept for BC).
        """
        now = datetime.utcnow()
        last = getattr(self, "_last_hist_fetch", {}).get((symbol, use_candles))
        if last and (now - last).total_seconds() < cache_secs:
            return  # skip – data still fresh
        style = "candles" if use_candles else "ticks"
        req = {
            "ticks_history": symbol,
            "style": style,
            "start": int(now.timestamp()) - config.HISTORICAL_DATA_DURATION,
            "end": "latest",
            "count": config.MAX_TICK_DATA_POINTS,
            "adjust_start_time": 1,
        }
        if use_candles:
            req["granularity"] = 60
        try:
            res = await self.deriv_api.ticks_history(req)
            async with self.lock:
                if style == "candles":
                    cnd = res.get("candles", [])
                    if not cnd:
                        self.logger.error(f"{symbol}: empty candle history")
                        return
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    self.candle_times.clear()
                    self.candle_opens.clear()
                    self.candle_highs.clear()
                    self.candle_lows.clear()
                    self.candle_closes.clear()
                    self.candle_volumes.clear()
                    for c in cnd:
                        ts = datetime.fromtimestamp(c["epoch"])
                        self.tick_times.append(ts)
                        self.tick_prices.append(float(c["close"]))
                        self.candle_times.append(ts)
                        self.candle_opens.append(float(c["open"]))
                        self.candle_highs.append(float(c["high"]))
                        self.candle_lows.append(float(c["low"]))
                        self.candle_closes.append(float(c["close"]))
                        self.candle_volumes.append(float(c.get("volume", 1)))
                    self.logger.debug(f"{symbol}: {len(cnd)} candles")
                else:  # ticks
                    hist = res.get("history", {})
                    times = hist.get("times", [])
                    prices = hist.get("prices", [])
                    if not times or not prices:
                        self.logger.error(f"{symbol}: empty tick history")
                        return
                    self.tick_times.clear()
                    self.tick_prices.clear()
                    for t, p in zip(times, prices):
                        self.tick_times.append(datetime.fromtimestamp(int(t)))
                        self.tick_prices.append(float(p))
                    self.logger.debug(f"{symbol}: {len(prices)} ticks")
            # cache timestamp
            self._last_hist_fetch = getattr(self, "_last_hist_fetch", {})
            self._last_hist_fetch[(symbol, use_candles)] = now
        except Exception as e:
            self.logger.error(f"{symbol}: hist fetch error ({style}) → {e}")


    # 2)  LEGACY WRAPPER – retains original call-site.  ────────────────────
    async def fetch_candle_history(self, symbol: str) -> None:
        await self.fetch_historical_data(symbol, use_candles=True)


    # 3)  LIVE-TICK SUBSCRIPTION  ──────────────────────────────────────────
    async def subscribe_live_ticks(self, symbol: str) -> None:
        req = {"ticks": symbol, "subscribe": 1}
        try:
            stream = await self.deriv_api.subscribe(req)

            async def _restart(err):
                self.logger.warning(f"tick stream error: {err}; reconnecting…")
                await asyncio.sleep(2)
                await self.subscribe_live_ticks(symbol)

            stream.subscribe(
                on_next=lambda t: asyncio.create_task(self.tick_callback(t)),
                on_error=lambda e: asyncio.create_task(_restart(e)),
                on_completed=lambda: self.logger.info(f"{symbol}: tick stream closed"),
            )
            self.logger.info(f"{symbol}: tick stream subscribed")
        except Exception as e:
            self.logger.error(f"{symbol}: subscribe error → {e}")


    # 4)  PRICE PROPOSAL  ──────────────────────────────────────────────────
    async def get_price_proposal(self, symbol: str, signal: str) -> Optional[Dict]:
        req = {
            "proposal": 1,
            "amount": 50,   # use dynamic size
            "basis": "stake",
            "contract_type": self.dynamic_contract_type,
            "currency": config.CURRENCY,
            "symbol": symbol,
            "multiplier": self.multiplier_factor,
        }
        for attempt in range(3):
            try:
                rsp = await self.deriv_api.proposal(req)
                prop = rsp.get("proposal")
                if prop:
                    self.logger.debug(f"proposal ask={prop.get('ask_price')}")
                    return prop
                raise ValueError("empty proposal")
            except Exception as e:
                self.logger.warning(f"proposal retry {attempt+1}/3 → {e}")
                await asyncio.sleep(1.5 ** attempt)
        self.logger.error("proposal failed after retries")
        return None


        # ---------------------------------------------------------------------
    #  🔻 TRADE-CLOSE & RESULT HANDLERS (optimised)
    # ---------------------------------------------------------------------
    async def sell_contract(
        self,
        contract_id: str,
        future: Optional[asyncio.Future] = None,
        retries: int = 4,
    ) -> None:
        """
        Attempt to close a contract with exponential back-off.
        On first successful sell we immediately finalise records.
        """
        if self.contracts_closed.get(contract_id):
            return

        dummy_fut = False
        if future is None:
            future, dummy_fut = asyncio.get_event_loop().create_future(), True

        for i in range(retries):
            try:
                rsp = await self.deriv_api.sell({"sell": contract_id, "price": 0})
                if "sell" in rsp:
                    profit = float(rsp["sell"].get("profit", 0.0))
                    await self.finalize_trade(contract_id, profit, future)
                    return
                raise RuntimeError("no sell receipt")
            except Exception as e:
                self.logger.warning(f"Sell retry {i+1}/{retries} → {e}")
                await asyncio.sleep(1.5 ** i)

        # still here → failed
        self.logger.error(f"‼ sell failed ({contract_id}) after {retries} attempts")
        self.contracts_closed[contract_id] = True
        if not future.done():
            future.set_result(None)
        if dummy_fut:  # silence potential 'never awaited'
            await future


    async def record_trade_results(self, profit: float, contract_id: str) -> None:
        """
        Update P/L book-keeping, win/loss counters and consecutive loss guard.
        """
        # small rounding noise – pull from API if near-zero
        if abs(profit) < 0.01:
            api_profit = await self.deriv_api.get_contract_profit(contract_id)
            if api_profit is not None:
                profit = api_profit

        self.trades_data.append({"profit": profit, "timestamp": datetime.now()})

        if profit > 0:
            self.cumulative_profit += profit
            self.total_wins += 1
            self.consecutive_losses = 0
            self.logger.info(f"✓ profit  +{profit:.2f}")
        elif profit < 0:
            self.cumulative_loss += -profit
            self.total_losses += 1
            self.consecutive_losses += 1
            self.logger.info(f"✗ loss    {profit:.2f}")
        else:
            self.logger.info("↔ break-even")

        net_pl = self.cumulative_profit - self.cumulative_loss
        self.pl_times.append(datetime.now())
        self.pl_values.append(net_pl)
        self.daily_pl = self.latest_balance - self.daily_start_balance
        self.logger.debug(f"P/L day {self.daily_pl:.2f} | net {net_pl:.2f} | "
                        f"loss streak {self.consecutive_losses}/{self.max_consecutive_losses}")


    def set_active_symbols(self, symbols):
        """
        Accepts either:
        • a list of strings  → ["R_10", "R_25", …]
        • a list of dicts    → [{"symbol": "R_10", …}, …]
        and normalises to a string list for internal use / logging.
        """
        if symbols and isinstance(symbols[0], dict):
            symbols = [s.get("symbol", "") for s in symbols]

        # remove empties & deduplicate
        symbols = list(dict.fromkeys(filter(None, symbols)))

        self.active_symbols = symbols
        self.logger.info(f"Active symbols ({len(symbols)}): {', '.join(symbols)}")



    # ---------------------------------------------------------------------
    #  🔻 LIFE-CYCLE HELPERS
    # ---------------------------------------------------------------------
    def initialize_trade(self, contract_id: str, price_proposal: Dict) -> None:
        self.entry_price = float(price_proposal["ask_price"])
        self.best_price  = self.entry_price
        self.trade_open_times[contract_id] = datetime.now()
        self.logger.info(f"Open {contract_id} @ {self.entry_price:.4f}")


    async def finalize_trade(self, contract_id: str, profit: float, future: asyncio.Future):
        self.contracts_closed[contract_id] = True
        await self.record_trade_results(profit, contract_id)
        await self.log_current_balance()
        future.set_result(None)



    async def cleanup_trade(self, contract_id: str) -> None:
        self.trade_open_times.pop(contract_id, None)
        self.logger.debug(f"Cleanup {contract_id}")


    async def retry_trade_execution(
        self, symbol: str, signal: str, max_retries: int = 3
    ) -> None:
        """
        Re-obtain proposal & resubmit buy with geometric back-off.
        """
        for i in range(max_retries):
            await asyncio.sleep(1.5 ** i)
            try:
                prop = await self.get_price_proposal(symbol, signal)
                if prop:
                    await self.execute_trade_safely(symbol, signal)
                    return
            except Exception as e:
                self.logger.warning(f"Buy retry {i+1}/{max_retries} → {e}")

