import asyncio
import logging
from utils.logger import setup_logger
from strategy.trading_strategy import TradingStrategy
from config import config
from utils.openai_utils import initialize_openai
from deriv_api import DerivAPI

async def main():
    """
    Main entry point for the trading bot.
    Sets up logging, initializes API connections, and starts the trading strategy.
    """
    # Setup logger with configuration
    logger = setup_logger(config.LOG_LEVEL, config.LOG_FILE)
    logger.info("Starting trading bot...")

    # Initialize OpenAI API
    initialize_openai(config.OPENAI_API_KEY, logger)

    # Initialize Deriv API using the official `python_deriv_api` package
    deriv_api = DerivAPI(app_id=config.DERIV_APP_ID)
    logger.info("Deriv API instance created.")

    try:
        # Authorize with API token
        authorize_response = await deriv_api.authorize(config.DERIV_API_TOKEN)
        if 'error' in authorize_response:
            logger.error(f"Authorization failed: {authorize_response['error']['message']}")
            return
        else:
            logger.info("Authorized with Deriv API.")

        # Initialize trading strategy
        strategy = TradingStrategy(logger=logger, deriv_api=deriv_api)

        # Fetch active symbols
        active_symbols_response = await deriv_api.active_symbols({"active_symbols": "full"})
        active_symbols = active_symbols_response.get('active_symbols', [])
        if active_symbols:
            strategy.set_active_symbols(active_symbols)
        else:
            logger.error("Failed to retrieve active symbols.")
            return

        # Start receiving ticks
        tick_stream = await deriv_api.subscribe({"ticks": config.SYMBOL})
        tick_stream.subscribe(
            on_next=lambda tick: asyncio.create_task(strategy.tick_callback(tick))
        )
        logger.info(f"Subscribed to ticks for symbol: {config.SYMBOL}")

        # Run indefinitely
        await asyncio.Future()  # Keep the main coroutine running

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        await deriv_api.disconnect()
        logger.info("Disconnected from Deriv API.")
        logger.info("Trading bot stopped.")

if __name__ == "__main__":
    asyncio.run(main())
