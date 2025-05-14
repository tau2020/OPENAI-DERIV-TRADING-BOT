import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.DEBUG) -> logging.Logger:
    """
    Sets up a logger with a rotating file handler.

    Parameters:
        name (str): Name of the logger.
        log_file (str): File path for the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent logging from propagating to the root logger
    logger.propagate = False

    # Create handlers
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set to DEBUG for console

    # Create formatters and add to handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

# Example usage:
logger = setup_logger('TradingStrategyLogger', 'trading_strategy.log')
