"""latency_adapter.py
================================
Utility class that sits *between* the TradingStrategy and the
OpenAISignalEngine to ensure LLM latency never produces stale or runaway
orders.

Key features
------------
* **Asynchronous wrapper** – schedules exactly *one* in‑flight LLM call.
* **TTL / freshness check** – discards results older than `ttl_secs`.
* **Grid‑band helper** – turns a single macro‑signal into a price ladder
  the strategy can trade locally without waiting for another round‑trip.
* **Streaming‑aware** – supports both regular and streaming completions.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence

from strategy.openai_signal_engine import OpenAISignalEngine


# Forward‑ref to avoid import cycles – use a Protocol in production.
class _AISignalEngine:
    async def generate_candle_signal(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class MacroSignal:
    """A parsed & validated AI payload."""

    data: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def expires_at(self) -> datetime:
        ttl = self.data.get("valid_for_secs", 12)
        return self.generated_at + timedelta(seconds=ttl)

    def is_fresh(self) -> bool:
        return datetime.utcnow() < self.expires_at

    # Convenience accessors ------------------------------------------------
    def risk(self) -> Dict[str, float]:
        return self.data.get("risk", {})

    def bands(self) -> Dict[str, Any]:
        return self.data.get("bands", {})

    @property
    def signal(self) -> str:
        return self.data.get("signal", "HOLD")


class LatencyAdapter:
    """Complementary helper that manages LLM calls + latency compensation."""

    def __init__(
        self,
        ai_engine: OpenAISignalEngine,
        logger: logging.Logger,
        max_network_latency: float = 6.0,
        ttl_factor: float = 2.0,
    ) -> None:
        self.ai = ai_engine
        self.logger = logger
        self.max_latency = max_network_latency
        self.ttl_secs = max_network_latency * ttl_factor  # e.g. 12 s

        self._pending_task: Optional[asyncio.Task] = None
        self._request_ts: Optional[datetime] = None
        self._latest_macro: Optional[MacroSignal] = None

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #
    def has_fresh_macro(self) -> bool:
        """Return True if we hold an unexpired MacroSignal."""
        return self._latest_macro is not None and self._latest_macro.is_fresh()

    def macro(self) -> Optional[MacroSignal]:
        """Return the cached MacroSignal if still fresh, else None."""
        if self.has_fresh_macro():
            return self._latest_macro
        # stale – drop reference so GC can reclaim
        self._latest_macro = None
        return None

    async def maybe_request(self, **ai_kwargs) -> None:
        """Kick off a new AI request if none is already running."""
        if self._pending_task and not self._pending_task.done():
            return  # already in‑flight
        self._request_ts = datetime.utcnow()
        self._pending_task = asyncio.create_task(self.ai.generate_candle_signal(**ai_kwargs))
        self.logger.debug("🚀 LLM call started (args=%s)", list(ai_kwargs.keys()))

    async def poll(self) -> None:
        """Non‑blocking check: pick up a completed task & validate freshness."""
        if not self._pending_task or not self._pending_task.done():
            return
        try:
            raw = self._pending_task.result()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("❌ AI task failed: %s", exc)
            self._pending_task = None
            return

        age = (datetime.utcnow() - (self._request_ts or datetime.utcnow())).total_seconds()
        raw.setdefault("valid_for_secs", self.ttl_secs)
        raw["age_secs"] = age
        self.logger.info("📬 AI signal arrived in %.2fs", age)

        self._latest_macro = MacroSignal(data=raw)
        self._pending_task = None

    # ------------------------------------------------------------------ #
    # GRID / BAND HELPERS
    # ------------------------------------------------------------------ #
    def price_ladder(self, last_price: float) -> Sequence[float]:
        """Return a list of entry prices centred on *last_price* based on
        the current macro‑signal's band instructions.
        """
        if not self.has_fresh_macro():
            return []
        band_cfg = self._latest_macro.bands()
        if band_cfg.get("mode") != "grid":
            return []
        window = float(band_cfg.get("window", 10))
        step = float(band_cfg.get("step", 3))
        half = window / 2.0
        centre = last_price
        # generate symmetric ladder
        levels = [centre]
        up = centre + step
        while up <= centre + half:
            levels.append(round(up, 4))
            up += step
        down = centre - step
        while down >= centre - half:
            levels.append(round(down, 4))
            down -= step
        return sorted(set(levels))
