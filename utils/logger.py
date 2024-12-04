# utils/logger.py

import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_level, log_file):
    """
    Sets up and returns a logger with the specified log level and log file.
    """
    logger = logging.getLogger('TradingBot')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)  # 1MB per file, 5 backups

    c_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    f_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
