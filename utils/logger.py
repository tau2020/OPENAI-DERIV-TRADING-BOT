import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(level, log_file):
    """
    Sets up the logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.upper())
    console_format = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    console_handler.setFormatter(console_format)

    # File handler with rotation
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(level.upper())
        file_format = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    logger.addHandler(console_handler)
    return logger
