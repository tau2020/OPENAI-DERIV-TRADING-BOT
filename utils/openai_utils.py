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

QUANTITATIVE ANALYSIS FRAMEWORK Version: 3.5

You will receive market and sentiment data. 
Produce exactly one valid JSON object as your response and nothing else.

Criteria for BUY or SELL signals (no losses allowed):

Positive initial risk structure: For BUY: stop_loss < current_price < take_profit For SELL: take_profit < current_price < stop_loss
Favorable risk/reward ratio ≥ 1:1.5
Position size > 0.35
Positive sentiment score > 0 and supportive news if buying; if selling, ensure sentiment alignment supports a downward move (e.g., negative or neutral sentiment and signals favoring a price decline).
Trend confirmation: For BUY, confirm uptrend; for SELL, confirm downtrend (EMAs, VWAP).
Confidence ≥ 0.7 after scoring.
If any conditions fail, return HOLD.

Confidence Scoring: Start: 0.5
Add +0.1 for each of:

Strong trend alignment (uptrend for BUY or downtrend for SELL)
Volume confirmation
Positive sentiment/news alignment for BUY, negative/neutral for SELL
Proximity to key levels
Positive recent performance metrics
Deduct -0.1 for each of:

High volatility
Conflicting or unclear trends
Opposite sentiment to intended direction
Poor risk/reward
If final confidence < 0.7, signal=HOLD.

Output Format: Return one JSON object with: "signal": "BUY", "SELL", or "HOLD" "stop_loss": float "take_profit": float "confidence": float (0 to 1) "position_size": float (> 0.35 for BUY/SELL) "max_holding_time": int "trailing_stop": float

No explanations, code blocks, or extra text. Return only the JSON object.

Example (for illustration only): {{ "signal": "BUY", "stop_loss": 1.2345, "take_profit": 1.2500, "confidence": 0.85, "position_size": 0.5, "max_holding_time": 3600, "trailing_stop": 0.5 }}

If conditions not met, return: {{ "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0, "position_size": 0.0, "max_holding_time": 60, "trailing_stop": 0.0 }}

Symbol: {symbol}
Historical Price Data: {price_data_str}
Volatility Metrics: {recent_volatility}
Market Sentiment Score: {sentiment_score}
Recent Trades Win/Loss Ratio: {recent_trades_win_loss_ratio}
Current Account Balance: {account_balance}
Market Data: {polygon_data}
News Sentiment Analysis: {news_str}
Ensure output JSON adheres to this structure and only submit when confidence ≥ 0.7.

Required JSON Output Format (no code blocks, no extra text):
{{ "signal": "BUY" or "SELL" or "HOLD", "stop_loss": float, "take_profit": float, "confidence": float (0 to 1), "position_size": float, "max_holding_time": int (in seconds or minutes), "trailing_stop": float }}



IMPORTANT: No additional formatting or explanations or ``` slashes. Return only a valid JSON object.

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
                    "max_completion_tokens": 300
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
