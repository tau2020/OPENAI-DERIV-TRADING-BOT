# utils/openai_utils.py

import openai
import logging
import asyncio
import json
import time


def initialize_openai(api_key, logger):
    openai.api_key = api_key
    logger.info("OpenAI API initialized.")
    return openai

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
    max_data_points = 500
    if len(prices) > max_data_points:
        logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points")
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]

    price_data_str = "\n".join(
        f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)
    )

    recent_volume = recent_volume or []
    recent_volatility = recent_volatility or 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio or 1.0
    account_balance = account_balance or 1000.0

   
    logger.info(f"recent_volatility: {recent_volatility}")
    logger.info(f"recent_trades_win_loss_ratio: {recent_trades_win_loss_ratio}")
    logger.info(f"account_balance: {account_balance}")

    prompt = f"""
QUANTITATIVE ANALYSIS FRAMEWORK
Version: 3.5

Return only a JSON object without any additional text or formatting.
Do not add '' or ``` to the object


ENTRY CONDITIONS:
- Uptrend: EMA(20,50,200) aligned
- Downtrend: EMA(20,50,200) aligned (descending)
- Confidence >= 0.7 => Trade

RETURN FORMAT:
{{
    "signal": "BUY"|"SELL"|"HOLD",
    "stop_loss": float,
    "take_profit": float,
    "confidence": float,
    "position_size": float,
    "max_holding_time": int,
    "trailing_stop": float
}}

MARKET DATA:
Symbol: {symbol}
Prices: {price_data_str}
Volatility: {recent_volatility}
Win/Loss: {recent_trades_win_loss_ratio}
Balance: {account_balance}
    """

    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o1-mini",
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

    if not response or not response.choices or not len(response.choices) > 0:
        logger.error("OpenAI assistant did not return a recommendation. Defaulting to HOLD.")
        return fallback_hold_signal()

    content = response.choices[0].message.content.strip()
    logger.info(f"OpenAI content: Content={content}")

    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Unable to parse JSON from the assistant response. Defaulting to HOLD.")
        return fallback_hold_signal()

    required_keys = ["signal", "stop_loss", "take_profit", "confidence",
                     "position_size", "max_holding_time", "trailing_stop"]
    for k in required_keys:
        if k not in result:
            logger.error(f"Missing key '{k}' in the assistant response.")
            return fallback_hold_signal()

    signal = (result.get("signal") or "HOLD").upper()
    if signal not in ("BUY", "SELL", "HOLD"):
        logger.error(f"Invalid signal: {signal}. Defaulting to HOLD.")
        signal = "HOLD"

    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except:
            return default

    stop_loss = safe_float(result.get("stop_loss"), 0.0)
    take_profit = safe_float(result.get("take_profit"), 0.0)
    confidence = safe_float(result.get("confidence"), 0.0)
    position_size = safe_float(result.get("position_size"), 0.0)
    max_holding_time = int(safe_float(result.get("max_holding_time"), 60.0))
    trailing_stop = safe_float(result.get("trailing_stop"), 0.0)

    if position_size < 0.35:
        position_size = max(1, position_size)

    logger.info(
        f"OpenAI assistant recommendation: Signal={signal}, SL={stop_loss}, TP={take_profit}, "
        f"Confidence={confidence}, PositionSize={position_size}, "
        f"MaxHoldingTime={max_holding_time}, TrailingStop={trailing_stop}"
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

def fallback_hold_signal():
    return {
        "signal": "HOLD",
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "confidence": 0.0,
        "position_size": 0.0,
        "max_holding_time": 60,
        "trailing_stop": 0.0
    }
