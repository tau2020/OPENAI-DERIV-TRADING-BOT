import asyncio
import logging
from typing import Dict, Optional 

# ── Hedge tuning constants ─────────────────────────────────────────────
MIN_SPREAD_BUFFER = 2.8     # × ATR  (trigger must sit outside the spread)
MAX_HEDGE_DEPTH   = 1       # 0 = parent leg, 1 = one mirror, nothing more
COOLDOWN_SECS     = 45.0    # seconds between any two hedge events
# ───────────────────────────────────────────────────────────────────────

class HedgeGuard:
    """
    Plug-and-play stop-and-reverse module for Deriv multiplier trades.

    Parameters
    ----------
    strategy : TradingStrategy
        Live TradingStrategy instance – HedgeGuard calls
        ``execute_exit`` and ``execute_trade_safely`` on it.
    loss_threshold_pts : float, default -3.0
        Minimum unrealised-loss trigger (negative).  The runtime edge is
        ``min(loss_threshold_pts, -ATR*MIN_SPREAD_BUFFER)`` so it always
        exceeds the bid-ask spread.
    """

    __slots__ = (
        "_stg", "_thr", "_log",
        "_last_hedge_ts",   # throttle by time
        "_depth",           # contract_id -> hedge depth
    )

    # ------------------------------------------------------------------ #
    #  INIT
    # ------------------------------------------------------------------ #
    def __init__(self, strategy, loss_threshold_pts: float = -6.0) -> None:
        self._stg = strategy
        self._thr = float(loss_threshold_pts)

        self._last_hedge_ts: float              = 0.0          # epoch secs
        self._depth: dict[str, int]             = {}           # cid -> int
        self._log: logging.Logger               = strategy.logger.getChild(
            "HedgeGuard"
        )

    # ------------------------------------------------------------------ #
    #  CALLABLE INTERFACE – hook into POC stream
    # ------------------------------------------------------------------ #
    def __call__(self, poc_update: Dict, cid: str) -> None:      # noqa: D401
        """Non-blocking wrapper so we never stall the tick loop."""
        asyncio.create_task(self._handle(poc_update, cid))

    # ------------------------------------------------------------------ #
    #  INTERNALS
    # ------------------------------------------------------------------ #
    async def _handle(self, poc_update: Dict, cid: str) -> None:
        """
        • Arming edge = max(config_thr, ATR*MIN_SPREAD_BUFFER) *-1
        • One hedge per lineage (depth ≤ MAX_HEDGE_DEPTH)
        • Global cool-down to prevent rapid ping-pong.
        """
        poc = poc_update.get("proposal_open_contract", {})
        if not (poc and poc.get("is_valid_to_sell")):
            return

        profit = float(poc.get("profit", 0.0))

        # 1) dynamic edge (ensure trigger outside spread)
        atr_pts = self._stg.average_true_range or 20.0

        edge_thr = min(self._thr, -self._stg.average_true_range * MIN_SPREAD_BUFFER)

        # 2) depth & cool-down guards
        depth = self._depth.get(cid, 0)
        now   = asyncio.get_event_loop().time()
        if (
            profit > edge_thr  or atr_pts == 0                   # not beyond edge
            or depth >= MAX_HEDGE_DEPTH           # too many flips
            or now - self._last_hedge_ts < COOLDOWN_SECS
        ):
            return

        # 3) prep mirror data
        orig_dir = self._stg.trade_directions.get(cid, "UNKNOWN")
        opp_dir  = "SELL" if orig_dir == "BUY" else "BUY"
        stake    = float(poc.get("buy_price", 1.0))
        symbol   = (
            poc.get("underlying")                # primary
            or poc.get("symbol")                 # alt
            or self._stg.current_symbol          # fallback
        )

        self._log.warning(
            f"{cid} down {profit:.2f} pts ≤ {edge_thr:.2f} pts "
            f"→ hedge {opp_dir} @ {stake:.2f}"
        )

        # 4) execute hedge
        try:
            # close loser
            await self._stg.execute_exit(cid, "HedgeGuard trigger")
            await asyncio.sleep(3.0)             # let balance settle

            # open mirror
            self._stg.dynamic_contract_type = "CALL" if opp_dir == "BUY" else "PUT"
            self._stg.dynamic_position_size = stake
            await self._stg.execute_trade_safely(symbol, opp_dir)

            # bookkeeping
            self._last_hedge_ts    = now
            self._depth[cid]       = depth + 1          # mark parent
            child = self._latest_contract_id()
            if child:
                self._depth[child] = depth + 1          # tag mirror
            self._log.info(f"Hedged {cid} → opened {opp_dir} @ {stake:.2f}")

        except Exception as exc:
            self._log.error(f"Hedge process failed on {cid}: {exc}")

    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #
    def _latest_contract_id(self) -> Optional[str]:
        """Return the most recently added contract_id, or None."""
        if not self._stg.trade_directions:
            return None
        return next(reversed(self._stg.trade_directions))
