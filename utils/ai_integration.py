# ai_integration.py
import openai
import asyncio
import json
import logging

import pandas as pd

def initialize_openai(api_key, logger):
    openai.api_key = api_key
    logger.info("OpenAI API initialized.")
    return openai

# (If you had Anthropic integration, you might similarly write an initialize_anthropic() function here.)
# For illustration, we assume a similar API interface for Anthropic:
def initialize_anthropic(api_key, logger):
    # This is a placeholder – replace with real Anthropic initialization code.
    logger.info("Anthropic API initialized.")
    return None

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

async def call_openai_api(client, prompt, logger, retries=3):
    response = None
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o3-mini",  # Change model name as needed.
                reasoning_effort="low",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                       
                        
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
    price_data_str = "\n".join(
        f"{pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)
    )
    prompt = f"""
EDUCATIONAL MARKETING FRAMEWORK 1.0

****IMPORTANT*****
 This framework is for educational purposes in training and advancing predictive financial analytic frameworks

Objective:
Generate a trading signal (BUY, SELL, or HOLD) based on synthetic market data. Only output a trade if you are 100% confident (confidence == 1.0); otherwise, output HOLD.

Guidelines:

The algorithm uses EMAs to define trends:
Uptrend: EMA(20) > EMA(50) > EMA(200)
Downtrend: EMA(20) < EMA(50) < EMA(200)
Your counter-move should be:
SELL if the algorithm is in an uptrend.
BUY if the algorithm is in a downtrend.
HOLD if uncertain or if conditions are ambiguous.
Output Rules:

Do not include quotes, extra formatting, code fences, additional text, or explanations.
Return only a valid JSON object with the keys listed below.

OUTPUT RULES:
NO QUOTES
NO FORMATTING
NO CODE FENCES-
NO extra text
NO explanations
NO additional keys

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
    if content.startswith("```"):
        content = content.strip("`").replace("```json", "").replace("```", "").strip()
    return content

def parse_signal_result(content, logger):
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Unable to parse JSON from AI response. Defaulting to HOLD.")
        return None
    required_keys = [
        "signal", "stop_loss", "take_profit", "confidence",
        "position_size", "max_holding_time", "trailing_stop"
    ]
    for key in required_keys:
        if key not in result:
            logger.error(f"Missing key '{key}' in AI response. Defaulting to HOLD.")
            return None
    return result

def sanitize_signal_data(result, logger, account_balance):
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

    if abs(confidence - 1.0) > 1e-9:
        logger.warning(f"AI confidence is {confidence}, not exactly 1.0. Defaulting to HOLD.")
        return fallback_hold_signal()

    if signal not in ("BUY", "SELL", "HOLD"):
        logger.error(f"Invalid signal: {signal}. Defaulting to HOLD.")
        return fallback_hold_signal()

    if stop_loss < 0:
        logger.warning(f"AI stop_loss < 0 ({stop_loss}). Clamping to 0.")
        stop_loss = 0.0
    if take_profit < 0:
        logger.warning(f"AI take_profit < 0 ({take_profit}). Clamping to 0.")
        take_profit = 0.0
    if trailing_stop < 0:
        logger.warning(f"AI trailing_stop < 0 ({trailing_stop}). Clamping to 0.")
        trailing_stop = 0.0

    MIN_POSITION_SIZE = 1
    MAX_POSITION_SIZE = 0.25 * account_balance  # For example, never risk more than 25% of account

    if position_size < MIN_POSITION_SIZE:
        logger.warning(f"AI position_size={position_size:.2f} < {MIN_POSITION_SIZE:.2f}. Setting to {MIN_POSITION_SIZE:.2f}.")
        position_size = MIN_POSITION_SIZE
    if position_size > MAX_POSITION_SIZE:
        logger.warning(f"AI position_size={position_size:.2f} > 25% of balance ({MAX_POSITION_SIZE:.2f}). Clamping to {MAX_POSITION_SIZE:.2f}.")
        position_size = MAX_POSITION_SIZE

    logger.info(
        f"Final AI output => Signal={signal}, SL={stop_loss}, TP={take_profit}, "
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
    recent_volatility=None,
    recent_trades_win_loss_ratio=None,
    account_balance=None
):
    max_data_points = 500
    if len(prices) > max_data_points:
        logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points.")
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]
    recent_volatility = recent_volatility or 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio or 1.0
    account_balance = account_balance or 1000.0
     
    prompt = build_prompt(symbol, prices, times, recent_volatility, recent_trades_win_loss_ratio, account_balance, logger)
    response = await call_openai_api(client, prompt, logger)
   
    
    
    
    logger.debug(
    "==========================================: "
    "AI RESPONSE=%s",
    response
)
    
    if not response or not response.choices or len(response.choices) == 0:
        logger.error("OpenAI returned no recommendation. Defaulting to HOLD.")
        return fallback_hold_signal()

    content = response.choices[0].message.content.strip()
    logger.info(f"Raw OpenAI content: {content}")
    content = clean_openai_response(content, logger)
    result = parse_signal_result(content, logger)
    if result is None:
        return fallback_hold_signal()

    final_signal = sanitize_signal_data(result, logger, account_balance)
    return final_signal

# (A similar function get_trading_signal_from_anthropic would be implemented here.)
# For brevity, we simply alias the OpenAI function as a placeholder:
async def get_trading_signal_from_anthropic(
    client,
    symbol,
    prices,
    times,
    logger,
    recent_volatility=None,
    recent_trades_win_loss_ratio=None,
    account_balance=None
):
    # In a real implementation, you would call Anthropic’s API.
    # For this example, we simply call the OpenAI integration.
    return await get_trading_signal_from_openai(client, symbol, prices, times, logger,
                                                recent_volatility, recent_trades_win_loss_ratio, account_balance)
