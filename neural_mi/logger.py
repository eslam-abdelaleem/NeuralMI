# neural_mi/logger.py
"""Initializes and configures the library's logging system.

This module sets up a centralized logger for the `neural_mi` library,
allowing for consistent and controllable logging across all modules. It
provides a default logger instance and a function to easily adjust the
verbosity level.
"""
import logging
import sys
from typing import Union

def setup_logger(name: str = 'neural_mi', level: int = logging.INFO) -> logging.Logger:
    """Sets up and configures a logger instance.

    This function creates a logger with a specified name and level, and attaches
    a console handler to it. It ensures that handlers are not duplicated if
    the logger has already been configured.

    Parameters
    ----------
    name : str, optional
        The name for the logger. Defaults to 'neural_mi'.
    level : int, optional
        The logging level, as defined in the `logging` module (e.g.,
        `logging.INFO`, `logging.DEBUG`). Defaults to `logging.INFO`.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if the logger is already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create a console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

# Create a default logger instance for the library
logger = setup_logger()

def set_verbosity(level: Union[int, str]):
    """Sets the global verbosity level for the library's logger.

    This function provides a simple way to control the logging output of the
    entire library.

    Parameters
    ----------
    level : int or str
        The desired verbosity level. Can be an integer from 0 (CRITICAL) to
        4 (DEBUG), or a string ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG').
    """
    level_map = {
        0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING,
        3: logging.INFO, 4: logging.DEBUG,
        'CRITICAL': logging.CRITICAL, 'ERROR': logging.ERROR,
        'WARNING': logging.WARNING, 'INFO': logging.INFO, 'DEBUG': logging.DEBUG
    }
    
    log_level = level_map.get(level, logging.INFO)
    
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)