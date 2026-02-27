import logging
import sys

import colorlog


def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with colored output.

    Args:
        name (str): The name of the logger, typically __name__.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    log_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s%(reset)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    logger.addHandler(console_handler)

    return logger
