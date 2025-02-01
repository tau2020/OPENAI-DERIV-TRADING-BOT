# main.py

import asyncio
import logging
import threading
from utils.logger import setup_logger
from strategy.trading_strategy import TradingStrategy
from config import config
from deriv_api import DerivAPI


async def main():
    """
    Main entry point for the trading bot.
    Sets up logging, initializes API connections, starts the trading strategy, and runs indefinitely.
    """
    logger = setup_logger(config.LOG_LEVEL, config.LOG_FILE)
    logger.info("Starting trading bot...")

    # Initialize Deriv API
    deriv_api = DerivAPI(app_id=config.DERIV_APP_ID)
    openai_api_key = config.OPENAI_API_KEY
    logger.info("Deriv API instance created.")

    try:
        # Authorize with API token
        authorize_response = await deriv_api.authorize(config.DERIV_API_TOKEN)
        if 'error' in authorize_response:
            logger.error(f"Authorization failed: {authorize_response['error']['message']}")
            return
        else:
            logger.info("Authorized with Deriv API.")

        # Retrieve loginid from the authorize response (optional)
        loginid = authorize_response.get('authorize', {}).get('loginid', 'Unknown')

        # Initialize the AI-free TradingStrategy
        strategy = TradingStrategy(
            logger=logger,
            deriv_api_client=deriv_api,
            openai_api_key=openai_api_key,
            loginid=loginid  # pass the loginid for logging/trace if desired
        )

        # Fetch active symbols
        active_symbols_response = await deriv_api.active_symbols({"active_symbols": "full"})
        active_symbols = active_symbols_response.get('active_symbols', [])
        if active_symbols:
            strategy.set_active_symbols(active_symbols)
            logger.info(f"Fetched {len(active_symbols)} active symbols.")
        else:
            logger.error("Failed to retrieve active symbols.")
            return

        # Start receiving ticks for the configured symbol
        tick_stream = await deriv_api.subscribe({"ticks": config.SYMBOL})

        async def handle_tick(tick):
            await strategy.tick_callback(tick)

        tick_stream.subscribe(on_next=lambda t: asyncio.create_task(handle_tick(t)))
        logger.info(f"Subscribed to ticks for symbol: {config.SYMBOL}")

        # Keep running indefinitely
        await asyncio.Future()  # Blocks so the bot keeps running

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await deriv_api.disconnect()
        logger.info("Disconnected from Deriv API.")
        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
