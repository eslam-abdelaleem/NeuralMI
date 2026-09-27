# neural_mi/logger.py
"""Initialises and configures the library's logging system and its warnings.

This module sets up a centralized logger for the `neural_mi` library, so logging is consistent and controllable across every module. It
provides a default logger instance and a function to easily adjust the
verbosity level.
"""
import concurrent.futures
import logging
import multiprocessing
import os
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

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
# The standard-library code that runs a task inside a worker process.
_WORKER_RUNNERS = tuple(os.path.dirname(os.path.abspath(m.__file__)) + os.sep
                        for m in (multiprocessing, concurrent.futures))


def _file(frame) -> str:
    return os.path.abspath(frame.f_code.co_filename)


def user_stacklevel() -> int:
    """The `stacklevel` that points a warning at the first frame outside neural_mi.

    Pass it inside the call, as ``warnings.warn(message, category,
    stacklevel=user_stacklevel())``, so that level 1 is the function issuing the
    warning. The warning then names the line of the caller's own code that led
    to it, however deep in the library it was raised. A worker process has no
    caller's code above the library, only the machinery that runs its task, so
    there the warning names the library's entry point for that task.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and _file(frame).startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    if frame is not None and level > 1 and _file(frame).startswith(_WORKER_RUNNERS):
        return level - 1
    return level

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


def _worker_log_init(level: int) -> None:
    """Pool initializer that gives a worker the parent's logging level.

    A ``spawn`` worker imports this module fresh, so the module-level
    ``setup_logger()`` runs again at its default INFO and the parent's choice
    does not carry over. Without this, ``run(verbose=False)`` silences the
    parent and every worker still prints its own informational output, both noisy and duplicated across processes.


    Runs once per worker process instead of once per task.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def worker_init_args():
    """``(initializer, initargs)`` propagating the current level to a Pool.

    Returns
    -------
    tuple
        Pass straight through to ``Pool(initializer=..., initargs=...)``.
    """
    return _worker_log_init, (logger.level,)


def set_verbose(verbose: bool):
    """Convenience wrapper: set logger to INFO (verbose=True) or WARNING (verbose=False).

    Parameters
    ----------
    verbose : bool
        If True, sets the logger to INFO level so informational messages are shown.
        If False, sets the logger to WARNING level so only warnings and errors appear.
    """
    set_verbosity(3 if verbose else 2)
