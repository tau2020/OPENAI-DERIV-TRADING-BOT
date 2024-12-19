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
        Version: 2.0
        Analysis Type: Multi-Factor Technical & Fundamental Pattern Recognition

        DATA INPUTS:
        1. Price Action:
        - Symbol: {symbol}
        - Historical Price Data (Time, Price): {price_data_str}

        2. Volume Analysis:
        - Recent Volume Data: {recent_volume}


        3. Volatility Metrics:
        - Historical Volatility: {recent_volatility}


        4. Market Context:
        - Market Sentiment Score: {sentiment_score} (-1 to 1)


        5. Performance Metrics:
        - Recent Trades Win/Loss Ratio: {recent_trades_win_loss_ratio}
        - Current Account Balance: {account_balance}

        6. External Data:
        - Market Data (Polygon.io): {polygon_data}
        - News Sentiment Analysis: {news_str}


        Analysis Focus:

        Evaluate short- and long-term trend direction via EMAs (20/50/200) and confirm with VWAP & relative volume.
        Use ATR and Bollinger Band width to gauge volatility; prefer stable/moderate volatility environments.
        Consider sentiment (overall > 0) and confirm alignment with price trends and volume.
        Prioritize entries near clear support/resistance levels and when recent performance metrics (win/loss ratio) are positive.
        Risk Management Guidelines:

        Size positions based on account balance and volatility; ensure stops and targets reflect ATR and key technical levels.
        Maintain minimum 1:1.5 risk/reward and use trailing stops to lock in profits when trends persist.
        Avoid trades if volatility is excessively high or if conflicting signals arise (e.g., negative news, unclear trend).
        Decision & Execution Criteria:

        Enter only when ≥3 technical and sentiment factors align:
        Trend confirmation (strong EMA alignment),
        Positive volume cues (above-average volume, VWAP support),
        Favorable sentiment (score >0, supportive news),
        Acceptable risk/reward (≥1:1.5).
        Exit if stop loss, take profit, or a clear reversal signal triggers.
        Confidence Scoring (>0.7 Target):

        Base: 0.5
        Add +0.1 each for: strong trend, volume confirmation, positive sentiment, and clear S/R levels, and recent positive performance.
        Deduct 0.1 each for: high volatility, conflicting signals, negative news, poor risk/reward, or uncertain market conditions.
        Aim to structure trades so final confidence score ≥0.7 before execution.
        Validation Steps:

        Pre-Trade: Check volumes, spreads, and avoid trading right before major news events.
        Post-Trade Setup: Ensure risk/reward, position size, and overall account risk are within set limits.
        Only execute if all criteria lead to a final confidence >0.7.

        Note: This quantitative framework combines multiple factors for robust decision-making. All signals are generated through systematic analysis of technical, fundamental, and sentiment indicators for educational and testing purposes.
        OUTPUT FORMAT:

        *****IMPORTANT*******
        - Use all provided data (fundamentals, news, sentiment, volatility) to make a high-confidence decision.
        Emphasize 
            profitability and risk-adjusted returns
        You must return a JSON object with the following keys:
        - "signal": string, one of "BUY", "SELL", or "HOLD"


        - "stop_loss": float, a recommended stop loss level relative to the current price
        - "take_profit": float, a recommended take profit level relative to the current price
        - "confidence": float, a value between 0 and 1 indicating your confidence in this signal
        - "position_size": float, the recommended fraction or quantity of the position to open
        - "max_holding_time": int, the maximum recommended holding time in seconds or minutes
        - "trailing_stop": float, the recommended trailing stop value to dynamically adjust the stop loss
        - Do not include any additional text or explanation. Do not include code blocks. Only return a valid JSON object

        Example JSON:
        {{"signal": "BUY", "stop_loss": 1.2345, "take_profit": 1.2500, "confidence": 0.85, "position_size": 12.23, "max_holding_time": 3600, "trailing_stop", 3.2}}


        ****************

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
                    "max_completion_tokens": 150
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
                    position_size = max(0.35, position_size)

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
