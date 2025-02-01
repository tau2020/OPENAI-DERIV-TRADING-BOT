import openai
import logging
import asyncio
import json

def initialize_openai(api_key, logger):
    """
    Sets the OpenAI API key and returns the openai module for usage.
    """
    openai.api_key = api_key
    logger.info("OpenAI API initialized.")
    return openai

def fallback_hold_signal():
    """
    Returns a safe fallback signal in case the AI output is invalid or an error occurs.
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

async def call_openai_api(client, prompt, logger, retries=3):
    """
    Calls the OpenAI API using async-to-thread and retries if errors occur.
    """
    response = None
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o3-mini",  # Replace with the correct model name if needed.
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ],
                        "max_completion_tokens": 300
                    }
                ]
            )
            return response
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"OpenAI request failed (attempt {attempt + 1}/{retries}). Retrying...")
                await asyncio.sleep(2)
            else:
                logger.error(f"OpenAI request failed after {retries} attempts: {e}")
                return None
    return response

def build_prompt(symbol, prices, times, recent_volatility, recent_trades_win_loss_ratio, account_balance, logger):
    """
    Constructs the prompt sent to the AI.
    """
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

GAME PROMPT (VERSION 3.5) — TRADING DUEL EDITION

IMPORTANT NOTE: This simulation is purely for educational purposes; all data is synthetic and for testing.

OBJECTIVE
You are in a strategic duel against a deterministic trading algorithm.
Your goal is to defeat the algorithm by countering its moves with profitable trades.
A win is achieved when your counter-move results in a profit.

GAME RULES
1. The Algorithm's Moves:
   - The algorithm executes trades based on market trends defined by Exponential Moving Averages (EMAs).
   - An uptrend is defined as: EMA(20) > EMA(50) > EMA(200)
   - A downtrend is defined as: EMA(20) < EMA(50) < EMA(200)

2. Your Counter-Moves:
   - If the algorithm is in an uptrend, your winning counter-move is SELL.
   - If the algorithm is in a downtrend, your winning counter-move is BUY.
   - In ambiguous situations or when certainty is lacking, you should choose HOLD.

3. Win Conditions:
   - Your move must be executed with absolute confidence (confidence == 1.0).
   - A win is achieved only if your move, when executed, results in a profit relative to the algorithm’s move.


   
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
    return prompt

def clean_openai_response(content, logger):
    """
    Removes any code fences or extraneous formatting from the OpenAI response.
    """
    if content.startswith("```"):
        content = content.strip("`").replace("```json", "").replace("```", "").strip()
    return content

def parse_signal_result(content, logger):
    """
    Attempts to parse the JSON response and verifies all required keys are present.
    """
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Unable to parse JSON from the assistant response. Defaulting to HOLD.")
        return None
    required_keys = [
        "signal", "stop_loss", "take_profit", "confidence",
        "position_size", "max_holding_time", "trailing_stop"
    ]
    for key in required_keys:
        if key not in result:
            logger.error(f"Missing key '{key}' in the assistant response. Defaulting to HOLD.")
            return None
    return result

def sanitize_signal_data(result, logger, account_balance):
    """
    Safely converts and clamps numeric values and enforces that confidence is exactly 1.0.
    """
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    signal = (result.get("signal") or "HOLD").upper()
    stop_loss = safe_float(result.get("stop_loss"), 0.0)
    take_profit = safe_float(result.get("take_profit"), 0.0)
    confidence = safe_float(result.get("confidence"), 0.0)
    position_size = safe_float(result.get("position_size"), 0.0)
    max_holding_time = int(safe_float(result.get("max_holding_time"), 60))
    trailing_stop = safe_float(result.get("trailing_stop"), 0.0)

    # Enforce confidence must be exactly 1.0; otherwise, default to HOLD.
    if abs(confidence - 1.0) > 1e-9:
        logger.warning(f"AI confidence is {confidence}, not exactly 1.0. Defaulting to HOLD.")
        return fallback_hold_signal()

    # Validate signal text.
    if signal not in ("BUY", "SELL", "HOLD"):
        logger.error(f"Invalid signal: {signal}. Defaulting to HOLD.")
        return fallback_hold_signal()

    # Clamp negative values to 0.
    if stop_loss < 0:
        logger.warning(f"AI stop_loss < 0 ({stop_loss}). Clamping to 0.")
        stop_loss = 0.0
    if take_profit < 0:
        logger.warning(f"AI take_profit < 0 ({take_profit}). Clamping to 0.")
        take_profit = 0.0
    if trailing_stop < 0:
        logger.warning(f"AI trailing_stop < 0 ({trailing_stop}). Clamping to 0.")
        trailing_stop = 0.0

    # Ensure position size is within acceptable limits.
    MIN_POSITION_SIZE = 1
    MAX_POSITION_SIZE = 0.25 * account_balance  # Never risk more than 50% of the account.

    if position_size < MIN_POSITION_SIZE:
        logger.warning(
            f"AI position_size={position_size:.2f} < {MIN_POSITION_SIZE:.2f}. Setting to {MIN_POSITION_SIZE:.2f}."
        )
        position_size = MIN_POSITION_SIZE
    if position_size > MAX_POSITION_SIZE:
        logger.warning(
            f"AI position_size={position_size:.2f} > 50% of balance ({MAX_POSITION_SIZE:.2f}). Clamping to {MAX_POSITION_SIZE:.2f}."
        )
        position_size = MAX_POSITION_SIZE

    logger.info(
        f"OpenAI final => Signal={signal}, SL={stop_loss}, TP={take_profit}, "
        f"Conf={confidence}, PosSize={position_size}, MaxHoldingTime={max_holding_time}, TS={trailing_stop}"
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
    Retrieves a trading signal from an OpenAI-like API in strict JSON format.
    On failure or invalid output, returns a fallback HOLD signal.
    """
    # 1. Truncate input data if necessary.
    max_data_points = 500
    if len(prices) > max_data_points:
        logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points.")
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]

    # 2. Set defaults for optional parameters.
    recent_volume = recent_volume or []
    recent_volatility = recent_volatility or 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio or 1.0
    account_balance = account_balance or 1000.0

    logger.info(f"recent_volatility: {recent_volatility}")
    logger.info(f"recent_trades_win_loss_ratio: {recent_trades_win_loss_ratio}")
    logger.info(f"account_balance: {account_balance}")

    # 3. Build the prompt.
    prompt = build_prompt(symbol, prices, times, recent_volatility, recent_trades_win_loss_ratio, account_balance, logger)

    # 4. Call the OpenAI API using our retry helper.
    response = await call_openai_api(client, prompt, logger)
    if not response or not response.choices or len(response.choices) == 0:
        logger.error("OpenAI assistant returned no usable recommendation. Defaulting to HOLD.")
        return fallback_hold_signal()

    # 5. Clean and parse the response.
    content = response.choices[0].message.content.strip()
    logger.info(f"Raw OpenAI content: {content}")
    content = clean_openai_response(content, logger)
    result = parse_signal_result(content, logger)
    if result is None:
        return fallback_hold_signal()

    # 6. Sanitize and validate the signal data.
    final_signal = sanitize_signal_data(result, logger, account_balance)
    return final_signal