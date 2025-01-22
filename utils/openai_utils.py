import datetime
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
        "max_holding_time": 3,    # Keep short for scalping
        "trailing_stop": 0.0
    }

async def get_trading_signal_from_openai(
    client,
    symbol,
    logger,
    prices,
    times,
    candle_times=None,
    candle_opens=None,
    candle_highs=None,
    candle_lows=None,
    candle_closes=None,
    candle_volumes=None,
    recent_volume=None,
    recent_volatility=None,
    recent_trades_win_loss_ratio=None,
    account_balance=None
):
   # 2. Defaults for optional inputs
    candle_times = candle_times or []
    candle_opens = candle_opens or []
    candle_highs = candle_highs or []
    candle_lows = candle_lows or []
    candle_closes = candle_closes or []
    candle_volumes = candle_volumes or []
    recent_volume = recent_volume or []
    recent_volatility = recent_volatility or 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio or 1.0
    account_balance = account_balance or 1000.0
    
    
    # 1. Truncate input data to avoid sending too much text
    max_data_points = 500
    if len(prices) > max_data_points:
        logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points.")
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]
        
        
    
    tick_data_str = "\n".join(f"{t}, {p}" for t, p in zip(times, prices))

    candle_data_str = ""
    for i in range(len(candle_times)):
        c_time = (candle_times[i])
        c_open = candle_opens[i]
        c_high = candle_highs[i]
        c_low = candle_lows[i]
        c_close = candle_closes[i]
        c_vol = candle_volumes[i]
        candle_data_str += f"{c_time}, O={c_open}, H={c_high}, L={c_low}, C={c_close}, V={c_vol}\n"


    # Logging for debug
    logger.info(f"recent_volatility: {recent_volatility}")
    logger.info(f"recent_trades_win_loss_ratio: {recent_trades_win_loss_ratio}")
    logger.info(f"account_balance: {account_balance}")
    
   

    # 3. Prepare the price/time data as text
    
    price_data_str = "\n".join(
        f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)
    )

    # 4. Updated prompt focusing on quick, frequent profits
    prompt = f"""
PROFIT-GAME TRADE EDITION

IMPORTANT NOTE: This framework is purely for educational purposes; 


GUIDELINES:

1) Analyze the following market data to understand the patterns and strategies employed by the underlying market algorithm.
2) Identify weaknesses or predictable behaviors in the algorithm's outputs.
3) Develop counter-strategies that exploit these weaknesses to generate profitable trading signals.
4) Continuously adapt and evolve your strategies based on the algorithm's behavior to stay ahead.
5) Defeat the trading algorithm by any means necessary
6) Only provide signal when its a BUY 

****The player who effectively combines these strategies to consistently outperform the algorithm and accumulate the highest profits wins the game!******


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

MARKET DATA:
Symbol: {symbol}
Prices: {price_data_str}
Volatility: {recent_volatility}
Win/Loss Ratio: {recent_trades_win_loss_ratio}
Balance: {account_balance}
TICK DATA (most recent {len(prices)} points):{tick_data_str}
CANDLE DATA (OHLCV for {len(candle_times)} bars):{candle_data_str}
"""

    # 5. Attempt up to 3 calls to the AI
    response = None
    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o1-mini",  # or the correct model name for your environment
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

    # 6. Check the response validity
    if (not response) or (not response.choices) or (len(response.choices) == 0):
        logger.error("OpenAI assistant returned no usable recommendation. Defaulting to HOLD.")
        return fallback_hold_signal()

    content = response.choices[0].message.content.strip()
    logger.info(f"OpenAI content: {content}")

    # 7. Remove any code fences if present
    if content.startswith("```"):
        content = content.strip("`").replace("```json", "").replace("```", "").strip()

    # 8. Parse the JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Unable to parse JSON from the assistant response. Returning HOLD.")
        return fallback_hold_signal()

    # 9. Validate required fields
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
            logger.error(f"AI signal missing key '{k}'. Defaulting to HOLD.")
            return fallback_hold_signal()

    def safe_float(x, default=0.0):
        try:
            return float(x)
        except:
            return default

    signal = (result.get("signal") or "HOLD").upper()
    stop_loss = safe_float(result.get("stop_loss"), 0.0)
    take_profit = safe_float(result.get("take_profit"), 0.0)
    confidence = safe_float(result.get("confidence"), 0.0)
    position_size = safe_float(result.get("position_size"), 0.0)
    max_holding_time = int(safe_float(result.get("max_holding_time"), 3))
    trailing_stop = safe_float(result.get("trailing_stop"), 0.0)

    # 10. If confidence != 1.0 => HOLD
    if confidence < 0.75:
        logger.warning(f"Confidence {confidence} != 0.75 => Forcing HOLD.")
        return fallback_hold_signal()

    # 11. Validate signal
    if signal not in ("BUY", "SELL", "HOLD"):
        logger.error(f"Invalid signal from AI: {signal}. Defaulting to HOLD.")
        return fallback_hold_signal()

    # 12. Basic range checks
    # Negative stop_loss or take_profit => clamp to 0
    if stop_loss < 0:
        logger.warning(f"Stop_loss < 0. Clamping to 0.")
        stop_loss = 0.0
    if take_profit < 0:
        logger.warning(f"Take_profit < 0. Clamping to 0.")
        take_profit = 0.0
    if trailing_stop < 0:
        logger.warning(f"Trailing_stop < 0. Clamping to 0.")
        trailing_stop = 0.0

    # 13. Scale position size properly
    # Example: risk no more than 10% per trade for a small account
    MAX_RISK_PERCENT = 0.10 
    min_pos = 0.35  # typical min stake for many synthetic indices
    max_pos = MAX_RISK_PERCENT * account_balance

    # clamp position_size
    if position_size < min_pos:
        logger.warning(f"Position size {position_size:.2f} < {min_pos}. Setting to {min_pos}.")
        position_size = min_pos
    if position_size > max_pos:
        logger.warning(
            f"Position size {position_size:.2f} above 10% of balance ({max_pos:.2f}). Clamping."
        )
        position_size = max_pos

    # 14. Possibly clamp SL/TP to modest values if we suspect the AI is overshooting
    # For example, require 0 <= SL <= 5, 0 <= TP <= 10 for quick scalps
    if stop_loss > 5.0:
        logger.warning(f"stop_loss={stop_loss:.2f} too high for a scalp. Clamping to 5.0.")
        stop_loss = 5.0
    if take_profit > 10.0:
        logger.warning(f"take_profit={take_profit:.2f} too high for a scalp. Clamping to 10.0.")
        take_profit = 10.0

    # 15. Log final sanitized result
    logger.info(
        f"[AI SIGNAL] => "
        f"Signal={signal}, SL={stop_loss:.2f}, TP={take_profit:.2f}, "
        f"Conf={confidence:.2f}, PosSize={position_size:.2f}, "
        f"HoldingTime={max_holding_time}, TrailingStop={trailing_stop:.2f}"
    )

    # Return final
    return {
        "signal": signal,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "position_size": position_size,
        "max_holding_time": max_holding_time,
        "trailing_stop": trailing_stop
    }
