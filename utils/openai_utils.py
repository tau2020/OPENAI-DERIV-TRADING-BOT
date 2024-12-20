# utils/openai_utils.py

import openai
import logging
import asyncio
import json

def initialize_openai(api_key, logger):
    """
    Initializes the OpenAI API client with the provided API key.
    """
    openai.api_key = api_key
    logger.info("OpenAI API initialized.")
    return openai  # Return the OpenAI client for use in TradingStrategy

async def get_trading_signal_from_openai(client, symbol, prices, times, logger,
    recent_volume=None,
    recent_volatility=None,
    sentiment_score=None,
    recent_trades_win_loss_ratio=None,
    account_balance=None,
    news_headlines=None,
    polygon_data=None):

    # Limit data points
    max_data_points = 150
    if len(prices) > max_data_points:
        logger.info(f"Truncating input data from {len(prices)} to {max_data_points} points")
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]

    # Prepare price data string
    price_data_str = "\n".join(
        [f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)]
    )

    # Defaults if None
    recent_volume = recent_volume if recent_volume is not None else []
    recent_volatility = recent_volatility if recent_volatility is not None else 0.0
    sentiment_score = sentiment_score if sentiment_score is not None else 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio if recent_trades_win_loss_ratio is not None else 1.0
    account_balance = account_balance if account_balance is not None else 1000.0
    news_headlines = news_headlines or []
    polygon_data = polygon_data or "No additional market data."

    news_str = "\n".join(news_headlines[:3])

    prompt = f"""

QUANTITATIVE ANALYSIS FRAMEWORK
Version: 3.5

You are a quantitative trading bot. 
Return only a JSON object without any additional text or formatting.
Do not add ``` to the object
Do not add ```json to the object

ENTRY CONDITIONS:
1. Trend + Volume
- EMA(20,50,200) aligned
- VWAP confirmation
- Volume > 1.5x average
- Clear S/R levels

2. Position Sizing
- Base = Balance × 0.02 × Confidence
- ATR Adjusted = Base × (1/ATR_Ratio)
- Min: 1%, Max: 5% of balance

3. Risk Parameters
- Stop Loss = ATR × 1.5
- Take Profit = Stop Loss × 2
- Trail after 1R profit
- R:R minimum 1:2

4. Signal Requirements:
BUY:
- Uptrend (EMAs aligned)
- Support test
- Volume surge
- Positive sentiment

SELL:
- Downtrend (EMAs aligned)
- Resistance test
- Volume surge
- Negative sentiment

HOLD:
- Any criteria not met
- Confidence < 0.7

5. Confidence Score (Base: 0.5)
+0.1 each:
- Trend alignment
- Volume confirmation
- S/R level test
- Sentiment match
- Low volatility
- Strong performance

-0.1 each:
- High volatility
- Mixed signals
- Poor volume
- Counter-trend
- News conflict

MARKET DATA:
Symbol: {symbol}
Prices: {price_data_str}
Volatility: {recent_volatility}
Sentiment: {sentiment_score}
Win/Loss: {recent_trades_win_loss_ratio}
Balance: {account_balance}
Market: {polygon_data}
News: {news_str}

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
IMPORTANT: 

Note: This framework operates on predefined mathematical criteria for educational and testing purposes. All signals are generated systematically based on technical indicators without discretionary input.
            """

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
                        },
                    ],
                    "max_completion_tokens": 60
                }
            ]
        )

        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            logger.info(f"OpenAI content: Content={content}")

            # Remove code fences if present
            if content.startswith("```"):
                # Strip backticks and remove ```json if present
                content = content.strip("`")
                content = content.replace("```json", "").replace("```", "").strip()

            # Attempt to parse JSON
            try:
                result = json.loads(content)

                # Ensure required keys
                required_keys = ["signal", "stop_loss", "take_profit", "confidence", 
                                 "position_size", "max_holding_time", "trailing_stop"]
                for k in required_keys:
                    if k not in result:
                        logger.error(f"Missing key '{k}' in the assistant response.")
                        return fallback_hold_signal()

                # Convert nulls to defaults if any
                signal = (result.get("signal") or "HOLD").upper()
                if signal not in ["BUY", "SELL", "HOLD"]:
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
                max_holding_time = safe_float(result.get("max_holding_time"), 60.0)
                trailing_stop = safe_float(result.get("trailing_stop"), 0.0)

                # Ensure minimum position size ≥ 0.35
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

            except json.JSONDecodeError:
                logger.error("Unable to parse JSON from the assistant response. Defaulting to HOLD.")
                return fallback_hold_signal()
        else:
            logger.error("OpenAI assistant did not return a recommendation. Defaulting to HOLD.")
            return fallback_hold_signal()
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return fallback_hold_signal()

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
    
