#!/usr/bin/env python3
"""
main.py

Main entry point for the trading bot. This script sets up logging, initializes API connections,
starts the trading strategy, and runs indefinitely.
"""

import asyncio
import logging
from utils.logger import setup_logger  # Custom logger setup (if any)
from strategy.trading_strategy import TradingStrategy
from config import config
from deriv_api import DerivAPI


def setup_logging() -> logging.Logger:
    """
    Set up logging for the trading bot.
    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)
    # Ensure no duplicate handlers are added in repeated runs.
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


async def main():
    """
    Main asynchronous entry point for the trading bot.
    It initializes the Deriv API, authorizes the connection, creates the trading strategy,
    subscribes to live tick updates, and keeps the bot running.
    """
    logger = setup_logging()
    logger.info("Starting trading bot...")

    # Initialize Deriv API client with your app ID and logger.
    deriv_api = DerivAPI(app_id=config.DERIV_APP_ID, logger=logger)
    logger.info("Deriv API instance created.")

    try:
        # Authorize using the provided API token.
        authorize_response = await deriv_api.authorize(config.DERIV_API_TOKEN)
        if 'error' in authorize_response:
            logger.error(f"Authorization failed: {authorize_response['error']['message']}")
            return
        else:
            logger.info("Authorized with Deriv API.")

        # Retrieve loginid from the authorize response (for logging and traceability).
        loginid = authorize_response.get('authorize', {}).get('loginid', 'Unknown')

        # Initialize the trading strategy.
        strategy = TradingStrategy(
            logger=logger,
            deriv_api_client=deriv_api,
            loginid=loginid
        )

        # Fetch active symbols from the API.
        active_symbols_response = await deriv_api.active_symbols({"active_symbols": "full"})
        active_symbols = active_symbols_response.get('active_symbols', [])
        if active_symbols:
            strategy.set_active_symbols(active_symbols)
            logger.info(f"Fetched {len(active_symbols)} active symbols.")
        else:
            logger.error("Failed to retrieve active symbols.")
            return

        # Subscribe to live tick data for the configured symbol.
        tick_stream = await deriv_api.subscribe({"ticks": config.SYMBOL})
        logger.info(f"Subscribed to ticks for symbol: {config.SYMBOL}")

        # Define a callback for processing incoming ticks.
        async def handle_tick(tick):
            await strategy.tick_callback(tick)

        # Subscribe with asynchronous callback.
        tick_stream.subscribe(
            on_next=lambda tick: asyncio.create_task(handle_tick(tick)),
            on_error=lambda err: logger.error(f"Tick stream error: {err}"),
            on_completed=lambda: logger.info("Tick stream subscription completed.")
        )

        # Keep the bot running indefinitely.
        await asyncio.Future()  # This future will never complete.

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await deriv_api.disconnect()
        logger.info("Disconnected from Deriv API.")
        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
