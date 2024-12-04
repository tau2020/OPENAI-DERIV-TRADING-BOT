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

    prompt = f"""You are an expert forex trading assistant.
Based on the recent price data for the symbol {symbol}, analyze the chart patterns and provide a trading recommendation.
Your recommendation should be one of: BUY, SELL, or HOLD.

Price Data (Time, Price):
{price_data_str}

Provide only the recommendation as one of the words: BUY, SELL, or HOLD.
"""

    try:
        # Since OpenAI's Python library is synchronous, run it in a separate thread
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4",
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
