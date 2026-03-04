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

    # Use stderr so library output does not pollute stdout (e.g. when stdout is piped)
    handler = logging.StreamHandler(sys.stderr)
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

_VALID_VERBOSITY_LEVELS = {
    0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING,
    3: logging.INFO, 4: logging.DEBUG,
    'CRITICAL': logging.CRITICAL, 'ERROR': logging.ERROR,
    'WARNING': logging.WARNING, 'INFO': logging.INFO, 'DEBUG': logging.DEBUG
}

def set_verbosity(level: Union[int, str]):
    """Sets the global verbosity level for the library's logger.

    This function provides a simple way to control the logging output of the
    entire library.

    Parameters
    ----------
    level : int or str
        The desired verbosity level. Can be an integer from 0 (CRITICAL) to
        4 (DEBUG), or a string ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG').

    Raises
    ------
    ValueError
        If `level` is not a recognised verbosity level.
    """
    if level not in _VALID_VERBOSITY_LEVELS:
        raise ValueError(
            f"Invalid verbosity level: {level!r}. "
            f"Expected an integer 0–4 or one of: "
            f"{list(k for k in _VALID_VERBOSITY_LEVELS if isinstance(k, str))}."
        )
    log_level = _VALID_VERBOSITY_LEVELS[level]
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


def set_verbose(verbose: bool):
    """Convenience wrapper: set logger to INFO (verbose=True) or WARNING (verbose=False).

    Parameters
    ----------
    verbose : bool
        If True, sets the logger to INFO level so informational messages are shown.
        If False, sets the logger to WARNING level so only warnings and errors appear.
    """
    set_verbosity(3 if verbose else 2)
