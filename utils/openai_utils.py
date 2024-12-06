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
    
QUANTITATIVE ANALYSIS FRAMEWORK
Version: 1.0
Analysis Type: Systematic Technical Pattern Recognition

INPUT PARAMETERS:
- Symbol: {symbol}
- Timeframe: Current Period
- Data Points: {price_data_str}

TECHNICAL INDICATORS:
1. Momentum Components
- RSI (Relative Strength Index): 14-period
- MACD (12,26,9)
- ROC (Rate of Change): 10-period
- Stochastic Oscillator (14,3,3)

2. Volatility Metrics
- Bollinger Bands (20,2)
- ATR (Average True Range): 14-period
- Historical Volatility: 20-period
- Volatility Ratio: Current/Historical

3. Price Structure Analysis
- Support/Resistance Levels
- Trendline Analysis
- Volume Profile Metrics
- Moving Average Configuration (50,200)

4. Risk Metrics
- Volatility-Adjusted Position Value
- Support/Resistance Distance Ratio
- Price/Volume Correlation

DECISION CRITERIA:

LONG SIGNAL (BUY):
- Primary Conditions (require all):
  * RSI > 40 and rising
  * MACD crossing above signal line
  * Price > 50-period MA
- Secondary Conditions (require 2+):
  * Volume > 20-period average
  * Bollinger Band expansion
  * Positive ROC

SHORT SIGNAL (SELL):
- Primary Conditions (require all):
  * RSI < 60 and falling
  * MACD crossing below signal line
  * Price < 50-period MA
- Secondary Conditions (require 2+):
  * Volume > 20-period average
  * Bollinger Band expansion
  * Negative ROC

NEUTRAL SIGNAL (HOLD):
- Conditions (any):
  * Mixed indicator readings
  * Low volume environment
  * Tight Bollinger Band range

OUTPUT FORMAT:
BUY, SELL, HOLD

*****IMPORTANT*******

*ONLY OUTPUT THE FOLLOWING- BUY | SELL | HOLD

****************

Note: This framework operates on predefined mathematical criteria for educational and testing purposes. All signals are generated systematically based on technical indicators without discretionary input.
"""

    try:
        # Since OpenAI's Python library is synchronous, run it in a separate thread
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
             "max_completion_tokens": 4
              }
            ],
          
           
        )

        # Parse the response (Corrected)
        if response and response.choices and len(response.choices) > 0:
            # Access content using dot notation
            print(response)
            content = response.choices[0].message.content.strip().upper()
            print(content)
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
