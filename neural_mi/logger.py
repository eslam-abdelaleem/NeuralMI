# neural_mi/logger.py

import logging
import sys
from typing import Union

def setup_logger(name: str = 'neural_mi', level: int = logging.INFO) -> logging.Logger:
    """
    Set up a library-wide logger.

    Parameters
    ----------
    name : str
        The name for the logger.
    level : int
        The logging level (e.g., logging.INFO, logging.DEBUG).

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
    """
    Set the global verbosity level for the library's logger.

    Parameters
    ----------
    level : int or str
        The desired verbosity level. Can be an integer (0-4) or a string
        ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG').
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