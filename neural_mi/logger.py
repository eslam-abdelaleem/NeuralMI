import logging
import sys

def setup_logger(name='neural_mi', level=logging.INFO):
    """
    Set up library-wide logger.

    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

# Create default logger
logger = setup_logger()

def set_verbosity(level):
    """
    Set global verbosity level.

    Parameters
    ----------
    level : {0, 1, 2, 3, 4} or int
        Verbosity level:
        - 0 or logging.CRITICAL: Only critical errors
        - 1 or logging.ERROR: Errors
        - 2 or logging.WARNING: Warnings and errors
        - 3 or logging.INFO: Info, warnings, errors (default)
        - 4 or logging.DEBUG: Debug and all above
    """
    level_map = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG
    }

    log_level = level_map.get(level, logging.INFO)

    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)