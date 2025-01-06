import openai
import logging
import asyncio
import json
import time

def initialize_openai(api_key, logger):
    """
    Sets the OpenAI API key and returns the openai module for usage.
    """
    openai.api_key = api_key
    logger.info("OpenAI API initialized.")
    return openai

def fallback_hold_signal():
    """
    A safe fallback response if the AI output is invalid or if the call fails.
    """
    return {
        "signal": "HOLD",
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "confidence": 0.0,
        "position_size": 0.0,
        "max_holding_time": 60,
        "trailing_stop": 0.0
    }

async def get_trading_signal_from_openai(
    client,
    symbol,
    prices,
    times,
    logger,
    recent_volume=None,
    recent_volatility=None,
    recent_trades_win_loss_ratio=None,
    account_balance=None
):
    """
    Calls an OpenAI-like API to get a trading signal in strict JSON format.
    If the AI fails or returns invalid data, returns a fallback HOLD signal.
    """

    # 1. Truncate input data to avoid sending too much text
    max_data_points = 500
    if len(prices) > max_data_points:
        logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points.")
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]

    # 2. Defaults for optional inputs
    recent_volume = recent_volume or []
    recent_volatility = recent_volatility or 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio or 1.0
    account_balance = account_balance or 1000.0

    # Log them for debugging
    logger.info(f"recent_volatility: {recent_volatility}")
    logger.info(f"recent_trades_win_loss_ratio: {recent_trades_win_loss_ratio}")
    logger.info(f"account_balance: {account_balance}")

    # 3. Build the prompt
    price_data_str = "\n".join(
        f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)
    )

    prompt = f"""
ENHANCED PROMPT (VERSION 3.5) — PROFESSIONAL PORTFOLIO MANAGER EDITION

IMPORTANT NOTE: This framework is purely for educational purposes; data is synthetic testing data.

OBJECTIVE
Grow an $8 balance into $100 by placing only trades with absolute certainty (confidence == 1.0).
Prioritize small-balance optimization while preserving high accuracy and high gains.

INSTRUCTIONS

Analysis Framework

Trend Confirmation (EMAs):
Uptrend: EMA(20) > EMA(50) > EMA(200)
Downtrend: EMA(20) < EMA(50) < EMA(200)
Confidence (0–1) must be exactly 1.0 to execute a trade:
If confidence == 1.0 and uptrend, BUY
If confidence == 1.0 and downtrend, SELL
Otherwise, HOLD

Small-Balance Risk Management
Use position sizes suited for low balances but still exploit 100% confident entries.
Implement tight stop_losses to minimize drawdowns.
Target moderate take_profit levels to lock in steady gains.
Consider trailing_stop to capture further upside.
Limit max_holding_time (hours) to reduce market exposure.

First-of-its-Kind “Synergy Strategy”
Combine short-term EMA signals with a proprietary confidence model that incorporates volatility, sentiment, and real-time order flow.
This hybrid approach ensures exact 1.0 confidence signals by dynamically filtering any conflicting indicators.
Trades are executed only when the “Synergy Strategy” flags a guaranteed directional move.

Adhere strictly to these guidelines: place trades only at confidence == 1.0, employ the Synergy Strategy for 100% guaranteed entries, and return only the JSON object in the specified format—no code fences, no quotes, no extra text.

OUTPUT RULES:
NO QUOTES
NO FORMATTING
NO CODE FENCES
NO ```
NO EXTRA TEXT 
NO EXPLANATIONS
NO ADDITIONAL KEYS

REQUIRED OUTPUT FORMAT:
{{ 
"signal": "BUY"|"SELL"|"HOLD", 
"stop_loss": float, 
"take_profit": float,
"confidence": float, 
"position_size": float, 
"max_holding_time": int, 
"trailing_stop": float 
}}

Market Data (Placeholders)
Symbol: {symbol}
Prices: {price_data_str}
Volatility: {recent_volatility}
Win/Loss Ratio: {recent_trades_win_loss_ratio}
Balance: {account_balance}
"""

    # 4. Make up to 3 attempts
    response = None
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o1-mini",  # Ensure the model name is correct/available
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                        "max_completion_tokens": 300
                    }
                ]
            )
            break
        except Exception as e:
            if attempt < 2:
                logger.warning(f"OpenAI request failed (attempt {attempt + 1}/3). Retrying...")
                time.sleep(2)
            else:
                logger.error(f"OpenAI request failed after 3 attempts: {e}")
                return fallback_hold_signal()

    # 5. Validate response
    if not response or not response.choices or len(response.choices) == 0:
        logger.error("OpenAI assistant returned no usable recommendation. Defaulting to HOLD.")
        return fallback_hold_signal()

    content = response.choices[0].message.content.strip()
    logger.info(f"OpenAI content: {content}")

    # Strip code fences if present
    if content.startswith("```"):
        content = content.strip("`").replace("```json", "").replace("```", "").strip()

    # 6. Parse JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Unable to parse JSON from the assistant response. Defaulting to HOLD.")
        return fallback_hold_signal()

    # 7. Ensure required keys exist
    required_keys = [
        "signal", 
        "stop_loss", 
        "take_profit", 
        "confidence",
        "position_size", 
        "max_holding_time", 
        "trailing_stop"
    ]
    for k in required_keys:
        if k not in result:
            logger.error(f"Missing key '{k}' in the assistant response. Defaulting to HOLD.")
            return fallback_hold_signal()

    # 8. Safe parsing of numeric fields
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except:
            return default

    signal = (result.get("signal") or "HOLD").upper()
    stop_loss = safe_float(result.get("stop_loss"), 0.0)
    take_profit = safe_float(result.get("take_profit"), 0.0)
    confidence = safe_float(result.get("confidence"), 0.0)
    position_size = safe_float(result.get("position_size"), 0.0)
    max_holding_time = int(safe_float(result.get("max_holding_time"), 60))
    trailing_stop = safe_float(result.get("trailing_stop"), 0.0)

    # 9. Enforce “confidence == 1.0” logic
    # If it’s not exactly 1.0, revert to HOLD.
    if abs(confidence - 1.0) > 1e-9:
        logger.warning(f"AI confidence is {confidence}, not exactly 1.0. Defaulting to HOLD.")
        return fallback_hold_signal()

    # Ensure signal is valid
    if signal not in ("BUY", "SELL", "HOLD"):
        logger.error(f"Invalid signal: {signal}. Defaulting to HOLD.")
        return fallback_hold_signal()

    # Clamp or validate numeric bounds
    stop_loss = max(0.0, stop_loss)
    take_profit = max(0.0, take_profit)
    trailing_stop = max(0.0, trailing_stop)

    # Example clamp for position_size
    MIN_POSITION_SIZE = 0.35
    MAX_POSITION_SIZE = .0
    position_size = max(MIN_POSITION_SIZE, min(position_size, MAX_POSITION_SIZE))

    logger.info(
        f"OpenAI final => Signal={signal}, SL={stop_loss}, TP={take_profit}, "
        f"Conf={confidence}, PosSize={position_size}, "
        f"MaxHoldingTime={max_holding_time}, TS={trailing_stop}"
    )

    return {
        "signal": signal,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "position_size": position_size,
        "max_holding_time": max_holding_time,
        "trailing_stop": trailing_stop
    }
