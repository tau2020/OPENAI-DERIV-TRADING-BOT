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
Version: 2.0
Analysis Type: Multi-Factor Technical & Fundamental Pattern Recognition

DATA INPUTS:
1. Price Action:
- Symbol: {symbol}
- Historical Price Data (Time, Price): {price_data_str}
- Price Trends: 
  * 20-period EMA
  * 50-period EMA
  * 200-period EMA crossing patterns

2. Volume Analysis:
- Recent Volume Data: {recent_volume}
- Volume Profile:
  * Volume-Weighted Average Price (VWAP)
  * Volume Trend Analysis
  * Relative Volume Ratio

3. Volatility Metrics:
- Historical Volatility: {recent_volatility}
- Implied Volatility (if available)
- ATR (Average True Range)
- Bollinger Bands Width

4. Market Context:
- Market Sentiment Score: {sentiment_score} (-1 to 1)
- Market Phase Identification:
  * Trending/Ranging/Reversing
  * Support/Resistance Levels
  * Key Price Levels

5. Performance Metrics:
- Recent Trades Win/Loss Ratio: {recent_trades_win_loss_ratio}
- Risk-Adjusted Returns (Sharpe Ratio)
- Maximum Drawdown
- Current Account Balance: {account_balance}

6. External Data:
- Market Data (Polygon.io): {polygon_data}
- News Sentiment Analysis: {news_str}
  * Headline Impact Score
  * News Volume
  * Sentiment Trend

RISK MANAGEMENT FRAMEWORK:
1. Position Sizing:
- Account Balance-Based Scaling
- Volatility-Adjusted Position Sizes
- Maximum Position Size Limits

2. Stop Loss Calculation:
- ATR-Based Stops
- Support/Resistance Levels
- Volatility-Adjusted Stops
- Time-Based Stops

3. Take Profit Targets:
- Risk/Reward Ratio (Minimum 1:1.5)
- Technical Resistance Levels
- Volatility-Based Targets
- Trailing Stop Implementation

DECISION CRITERIA:

1. Entry Conditions (Must Meet 3+ Criteria):
- Price Action Confirmation
- Volume Support
- Trend Alignment
- Sentiment Confirmation
- Risk/Reward Threshold
- News Impact Assessment

2. Exit Conditions (Any 1 Criterion):
- Stop Loss Hit
- Take Profit Reached
- Technical Reversal
- Sentiment Shift
- Time-Based Exit

3. Confidence Scoring (0-1):
Base Score (0.5) + Modifiers:
- Strong Trend (+0.1)
- Volume Confirmation (+0.1)
- Positive Sentiment (+0.1)
- Clear Support/Resistance (+0.1)
- Recent Success Rate (+0.1)
Deductions:
- High Volatility (-0.1)
- Conflicting Signals (-0.1)
- Negative News (-0.1)
- Poor Risk/Reward (-0.1)
- Market Uncertainty (-0.1)

SIGNAL VALIDATION:
1. Pre-Signal Checks:
- Minimum Volume Requirements
- Spread Analysis
- Market Hours Verification
- News Event Buffer

2. Post-Signal Validation:
- Risk/Reward Ratio Check
- Position Size Validation
- Account Risk Limit Check
- Maximum Open Position Check

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
            model="o1-preview",
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
            logger.info(
                    f"OpenAI content: Content={content}"
                )

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
                position_size = float(result["position_size"])
                max_holding_time = float(result["max_holding_time"])
                trailing_stop = float(result["trailing_stop"])

                logger.info(
                    f"OpenAI assistant recommendation: Signal={signal}, SL={stop_loss}, TP={take_profit}, Confidence={confidence}, position_size={position_size}, max_holding_time={max_holding_time}, trailing_stop={trailing_stop}"
                )
                return {
                    "signal": signal,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": confidence,
                    "position_size":position_size,
                    "max_holding_time":max_holding_time,
                    "trailing_stop":trailing_stop
                    
                }

            except json.JSONDecodeError:
                logger.error("Unable to parse JSON from the assistant response. Defaulting to HOLD.")
                return {
                    "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0,
                    "position_size": 0.0, "max_holding_time": 60, "trailing_stop": 0.0
                }
        else:
            logger.error("OpenAI assistant did not return a recommendation. Defaulting to HOLD.")
            return {
                "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0,
                "position_size": 0.0, "max_holding_time": 60, "trailing_stop": 0.0
            }
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return {
            "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0,
            "position_size": 0.0, "max_holding_time": 60, "trailing_stop": 0.0
        }
