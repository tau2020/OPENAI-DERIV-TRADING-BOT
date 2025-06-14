
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Sequence

import openai

# ---------------------------------------------------------------------------
#  Low‑level helper functions
# ---------------------------------------------------------------------------

def initialize_openai(api_key: str, logger: logging.Logger):
    """Configure the global OpenAI client and return the module for convenience."""
    openai.api_key = api_key
    logger.debug("✅ OpenAI client initialised")
    return openai


def fallback_hold_signal(logger: logging.Logger | None = None):
    """Return a HOLD signal payload and optionally log the event."""
    if logger:
        logger.info("🔄 Fallback HOLD signal returned")
    return {
        "signal": "HOLD",
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "confidence": 0.0,
        "position_size": 0.0,
        "max_holding_time": 60,
        "trailing_stop": 0.0,
    }


async def call_openai_api(
    client,
    prompt: str,
    logger: logging.Logger,
    model: str,
    retries: int = 3,
):
    """Async wrapper with exponential back‑off and rich logging."""
    logger.debug("📡 Calling OpenAI API …")
    for attempt in range(retries):
        try:
            start_ts = asyncio.get_event_loop().time()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o4-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                        "max_completion_tokens": 3000,
                    }
                ],
            )
            elapsed = asyncio.get_event_loop().time() - start_ts
            logger.info(f"🟢 OpenAI responded in {elapsed:.2f}s (tokens≈{len(prompt.split())})")
            return response
        except Exception as exc:
            if attempt < retries - 1:
                logger.warning(
                    f"OpenAI request failed (attempt {attempt + 1}/{retries}) – retrying: {exc}"
                )
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"❌ OpenAI request failed after {retries} attempts: {exc}")
                return None
    return None  # pragma: no cover – defensive


# ---------------------------------------------------------------------------
#  Prompt builder & response sanitation helpers
# ---------------------------------------------------------------------------

def build_prompt(
    symbol: str,
    prices: Sequence[float],
    times: Sequence[datetime],
    recent_volatility: float,
    recent_trades_win_loss_ratio: float,
    account_balance: float,
) -> str:
    price_data_str = "\n".join(
        f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)
    )
    return f"""
                     FOR EDUCATIONAL PURPOSES ONLY — ALGORITHM CHALLENGE FRAMEWORK 
     
                           OBJECTIVE
                           Beat DERIV'S RDBEAR & RDBULL house bot in this fast paced guessing game by calling each tick UP or DOWN.
                            
                            OUTPUT RULES
                            Return **exactly** one JSON object and nothing else.
                            No commentary, no additional keys, no wrapping quotes, no code fences.
                            
                            REQUIRED OUTPUT FORMAT
                            {{
                            "signal": "BUY" | "SELL" | "HOLD",
                            "stop\_loss": float,
                            "take\_profit": float,
                            "confidence": float,
                            "position\_size": float,
                            "max\_holding\_time": int,
                            "trailing\_stop": float
                            }}
                            
                            CONTEXT VARIABLES
                            Symbol: RDBEAR | RDBULL
                            Prices: {price_data_str}
                            Volatility: {recent_volatility}
                            WinLossRatio: {recent_trades_win_loss_ratio}
                            Balance: {account_balance}

             """


def clean_openai_response(content: str, logger: logging.Logger) -> str:
    original = content
    if content.startswith("```"):
        content = (
            content.strip("`")
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        logger.debug("✂️ Stripped markdown fences from OpenAI response")
    if len(content) != len(original):
        logger.debug("🧹 Content length after clean: %d", len(content))
    return content


def parse_signal_result(content: str, logger: logging.Logger):
    try:
        result = json.loads(content)
        logger.debug("✅ JSON parsed successfully from assistant response")
    except json.JSONDecodeError:
        logger.error("❓ Unable to parse JSON from assistant response → defaulting to HOLD")
        return None

    required_keys = {
        "signal",
        "stop_loss",
        "take_profit",
        "confidence",
        "position_size",
        "max_holding_time",
        "trailing_stop",
    }
    if not required_keys.issubset(result):
        logger.error("🚫 Assistant response missing required keys → defaulting to HOLD")
        return None
    return result


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def sanitise_signal(result: dict, logger: logging.Logger, account_balance: float):
    logger.debug("🧪 Sanitising signal data: %s", result)
    signal = (result.get("signal") or "HOLD").upper()
    stop_loss = _safe_float(result.get("stop_loss"))
    take_profit = _safe_float(result.get("take_profit"))
    confidence = _safe_float(result.get("confidence"))
    position_size = _safe_float(result.get("position_size"))
    max_holding_time = int(_safe_float(result.get("max_holding_time"), 60))
    trailing_stop = _safe_float(result.get("trailing_stop"))

    # Enforce absolute‑certainty rule
    if abs(confidence - 0.65) > 1e-6:
        logger.warning(f"⚠️ AI confidence {confidence} ≠ 1.0 → forced HOLD")
        return fallback_hold_signal(logger)

    if signal not in {"BUY", "SELL", "HOLD"}:
        logger.error(f"⚠️ Invalid signal '{signal}' → forced HOLD")
        return fallback_hold_signal(logger)

    # Clamp negatives
    stop_loss = max(0.0, stop_loss)
    take_profit = max(0.0, take_profit)
    trailing_stop = max(0.0, trailing_stop)

    # Position sizing
    min_size = 1.0
    max_size = 0.25 * account_balance
    position_size = min(max(position_size, min_size), max_size)

    clean_result = {
        "signal": signal,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "position_size": position_size,
        "max_holding_time": max_holding_time,
        "trailing_stop": trailing_stop,
    }
    logger.info("📊 Final sanitised signal → %s", clean_result)
    return clean_result


# ---------------------------------------------------------------------------
#  High‑level wrapper class
# ---------------------------------------------------------------------------

class OpenAISignalEngine:
    """Drop‑in replacement for the legacy `RuleBasedSignals` class."""

    def __init__(self, api_key: str, logger: logging.Logger, model: str = "o4-mini") -> None:
        self.logger = logger
        self.client = initialize_openai(api_key, logger)
        self.model = model
        logger.debug("🔧 OpenAISignalEngine initialised with model '%s'", model)

    # Mirror the old synchronous name but keep async internally
    async def generate_candle_signal(
        self,
        *,
        symbol: str,
        candle_times: Sequence[datetime],
        candle_closes: Sequence[float],
        account_balance: float,
        recent_volatility: float = 0.0,
        recent_trades_win_loss_ratio: float = 1.0,
    ) -> dict:
        """Generate a trading recommendation from the latest candle series."""
        self.logger.debug("🔍 Generating AI signal for %s (candles=%d)", symbol, len(candle_closes))

        prices = list(candle_closes)
        times = list(candle_times)
        if not prices or not times:
            self.logger.warning("🕳️ No candle data supplied → HOLD")
            return fallback_hold_signal(self.logger)

        prompt = build_prompt(
            symbol,
            prices,
            times,
            recent_volatility,
            recent_trades_win_loss_ratio,
            account_balance,
        )
        self.logger.debug("📝 Prompt size: %d chars", len(prompt))

        response = await call_openai_api(self.client, prompt, self.logger, self.model)
        if not response or not response.choices:
            return fallback_hold_signal(self.logger)

        content = response.choices[0].message.content.strip()
        self.logger.debug("📄 Raw OpenAI content: %.120s…", content)
        content = clean_openai_response(content, self.logger)
        result = parse_signal_result(content, self.logger)
        if result is None:
            return fallback_hold_signal(self.logger)

        return sanitise_signal(result, self.logger, account_balance)
