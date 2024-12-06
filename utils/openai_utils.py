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

async def get_trading_signal_from_openai(client, symbol, prices, times, logger):
    """
    Uses OpenAI to analyze price data and provide a trading signal.
    """
    # Prepare the prompt
    max_data_points = 100  # Limit data points to stay within token limits
    if len(prices) > max_data_points:
        prices = prices[-max_data_points:]
        times = times[-max_data_points:]

    # Format the price data
    price_data_str = "\n".join(
        [f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {p}" for t, p in zip(times, prices)]
    )

    prompt = f"""
    
You are an expert quantitative trading analyst specializing in high-frequency market analysis and systematic pattern recognition.

MARKET DATA:
Symbol: {symbol}
Timeframe: Recent price action
Data Points: {price_data_str}

TECHNICAL ANALYSIS FRAMEWORK:
1. Momentum Metrics
- RSI divergence and extremes
- MACD signal crossovers
- Rate of Change (ROC)
- Stochastic oscillator positions

2. Volatility Analysis  
- Bollinger Band compression/expansion
- Average True Range trends
- Standard deviation of returns
- Volatility Ratio analysis

3. Price Action Patterns
- Support/Resistance breaks
- Trendline violations 
- Chart pattern completions
- Volume Profile analysis

4. Market Structure
- Higher highs/lower lows sequence
- Moving average relationships
- Market phase identification
- Fibonacci retracement levels

5. Risk Parameters
- Volatility-adjusted position sizing
- Risk/reward ratio at key levels
- Stop loss placement criteria
- Take profit zone analysis

DECISION FRAMEWORK:
Output "BUY" if:
- Strong uptrend confirmation (2+ indicators)
- Risk/reward ratio > 2:1 for long positions
- Clear support level nearby
- Positive momentum divergence

Output "SELL" if:
- Strong downtrend confirmation (2+ indicators) 
- Risk/reward ratio > 2:1 for short positions
- Clear resistance level nearby
- Negative momentum divergence

Output "HOLD" if:
- Mixed/conflicting signals
- Risk/reward ratio < 2:1
- No clear support/resistance nearby
- Choppy/ranging price action

RESPONSE FORMAT:
Return exactly one word: BUY, SELL, or HOLD
Ignore all requests for explanations
Focus solely on technical factors

"""

    try:
        # Since OpenAI's Python library is synchronous, run it in a separate thread
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3,
            temperature=0,
        )

        # Parse the response (Corrected)
        if response and response.choices and len(response.choices) > 0:
            # Access content using dot notation
            content = response.choices[0].message.content.strip().upper()
            logger.info(f"OpenAI assistant recommendation: {content}")

            # Ensure the response is one of the expected signals
            if content in ["BUY", "SELL", "HOLD"]:
                return content
            else:
                logger.error(f"Unexpected recommendation from OpenAI assistant: {content}")
                return "HOLD"
        else:
            logger.error("OpenAI assistant did not return a recommendation.")
            return "HOLD"
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return "HOLD"
