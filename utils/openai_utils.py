# utils/openai_utils.py

import openai
import logging
import asyncio

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

    # Limit data points for tokens
    max_data_points = 150
    if len(prices) > max_data_points:
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]

    price_data_str = "\n".join(
        [f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)]
    )

    # Provide defaults if None
    recent_volume = recent_volume if recent_volume is not None else []
    recent_volatility = recent_volatility if recent_volatility is not None else 0.0
    sentiment_score = sentiment_score if sentiment_score is not None else 0.0
    recent_trades_win_loss_ratio = recent_trades_win_loss_ratio if recent_trades_win_loss_ratio is not None else 1.0
    account_balance = account_balance if account_balance is not None else 1000.0
    
    news_headlines = news_headlines or []
   
    polygon_data = polygon_data or "No additional market data."

    # Include additional data
    volume_str = f"Recent volume data points: {recent_volume}"
    sentiment_str = f"Market sentiment score: {sentiment_score} (-1 to 1, negative to positive)"
    volatility_str = f"Recent price volatility measure (annualized): {recent_volatility}"
    performance_str = f"Recent trades win/loss ratio: {recent_trades_win_loss_ratio}, Current account balance: {account_balance}"

    news_str = "\n".join(news_headlines[:3]) 

    prompt = f"""
QUANTITATIVE ANALYSIS FRAMEWORK
Version: 1.0
Analysis Type: Systematic Technical Pattern Recognition

DATA:
- Symbol: {symbol}
- Historical Price Data (Time, Price):{price_data_str}
- Recent Volume Data: {recent_volume}
- Volatility: {recent_volatility}
- Market Sentiment Score: {sentiment_score} (-1 to 1)
- Recent Trades Win/Loss Ratio: {recent_trades_win_loss_ratio}
- Current Account Balance: {account_balance}




MARKET DATA (From Polygon.io):
{polygon_data}

NEWS HEADLINES (From NewsAPI):
{news_str}



GOAL:
- Maximize risk-adjusted returns
- Prioritize profitability over frequency of trades
- Provide a high-confidence signal only if conditions strongly favor a profitable outcome

TECHNICAL & FUNDAMENTAL CONSIDERATIONS:
- Identify trending vs ranging market conditions
- Evaluate momentum and mean-reversion indicators
- Consider sentiment score: a positive sentiment might support buying dips or continuing to hold longs, negative sentiment might justify shorting or selling rallies
- Use volatility metrics to set appropriate stop loss and take profit levels (tighter if low volatility, wider if high volatility)
- Consider recent performance: If the recent win/loss ratio is high, slightly more aggressive take profits; if low, focus on capital preservation and tighter stops

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

Do not include any additional text or explanation. Do not include code blocks. Only return a valid JSON object.

Example JSON:
{{"signal": "BUY", "stop_loss": 1.2345, "take_profit": 1.2500, "confidence": 0.85}}


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
            logger.debug(f"Raw AI Response: {content}")

            import json
            try:
                result = json.loads(content)
                required_keys = ["signal", "stop_loss", "take_profit", "confidence"]
                for k in required_keys:
                    if k not in result:
                        logger.error(f"Missing key '{k}' in the assistant response.")
                        return {"signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0}

                signal = result["signal"].upper()
                if signal not in ["BUY", "SELL", "HOLD"]:
                    logger.error(f"Invalid signal: {signal}. Defaulting to HOLD.")
                    signal = "HOLD"

                stop_loss = float(result["stop_loss"])
                take_profit = float(result["take_profit"])
                confidence = float(result["confidence"])

                logger.info(
                    f"OpenAI assistant recommendation: Signal={signal}, SL={stop_loss}, TP={take_profit}, Confidence={confidence}"
                )
                return {
                    "signal": signal,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": confidence
                }

            except json.JSONDecodeError:
                logger.error("Unable to parse JSON from the assistant response.")
                return {"signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0}
        else:
            logger.error("OpenAI assistant did not return a recommendation.")
            return {"signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0}
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return {"signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0}
