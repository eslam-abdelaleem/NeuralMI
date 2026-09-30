# neural_mi/logger.py
"""Initialises and configures the library's logging system and its warnings.

This module sets up a centralized logger for the `neural_mi` library, so logging is consistent and controllable across every module. It
provides a default logger instance and a function to easily adjust the
verbosity level.
"""
import concurrent.futures
import importlib.util
import logging
import multiprocessing
import os
import sys
from typing import Optional, Union

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

# The library's logger. It shows warnings and errors until nmi.set_verbosity()
# or nmi.set_verbose(True) asks for more.
logger = setup_logger(level=logging.WARNING)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
# torch calls back into the library (a network's forward runs under
# torch.nn.Module.__call__), so its frames are passed through as well.
_TORCH_DIR = os.path.abspath(importlib.util.find_spec('torch').submodule_search_locations[0]) + os.sep
_PASS_THROUGH = (_PACKAGE_DIR, _TORCH_DIR)
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
    to it, however deep in the library it was raised, and through any torch
    code the library runs under, such as a network's forward. A worker process has no
    caller's code above the library, only the machinery that runs its task, so
    there the warning names the library's entry point for that task.
    """
    frame = sys._getframe(1)
    level = 1
    last_library = None
    while frame is not None and _file(frame).startswith(_PASS_THROUGH):
        if _file(frame).startswith(_PACKAGE_DIR):
            last_library = level
        frame = frame.f_back
        level += 1
    if frame is not None and last_library is not None and _file(frame).startswith(_WORKER_RUNNERS):
        return last_library
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
            f"Expected an integer from 0 to 4 or one of "
            f"{list(k for k in _VALID_VERBOSITY_LEVELS if isinstance(k, str))}."
        )
    log_level = _VALID_VERBOSITY_LEVELS[level]
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


def _worker_log_init(level: int) -> None:
    """Pool initializer that gives a worker the parent's logging level.

    A ``spawn`` worker imports this module fresh and starts at the library's
    default level. Without this, a level the parent chose would not reach the
    workers, and a parent at INFO or ERROR would get the workers' output at
    WARNING. Runs once per worker process.
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


# ---------------------------------------------------------------------------
# Repeated messages
#
# One call can raise the same warning many times: every chunk of a rigorous
# ladder or every worker of a grid hits the same condition. Within a call the
# first message of each kind is shown when it happens, later ones that differ
# only in their numbers are counted, and the call ends with one line per kind
# giving how often it was raised. In IPython and Jupyter the count runs over
# every call of a cell and the line comes when the cell ends. grouped_warnings()
# does the same over a block of a script. Only messages raised during a NeuralMI
# call are grouped. A worker process keeps its warnings and log messages and
# hands them back with its result, and they are raised again in the calling
# process at the caller's line.
# ---------------------------------------------------------------------------

import re as _re
import warnings as _warnings
from contextlib import contextmanager as _contextmanager

_NUMBER = _re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
_repeats = None          # the counts of the call in progress, or None
_session = None          # counts shared by the calls of a cell or a block, or None
_in_worker = False       # set once a process has run a captured task


def _kind(source: str, text: str) -> tuple:
    return source, _NUMBER.sub('#', text)


class _Repeats:
    """First text and count of each kind of message raised during one call."""

    def __init__(self):
        self.seen = {}

    def first(self, kind: tuple, text: str) -> bool:
        entry = self.seen.get(kind)
        if entry is None:
            self.seen[kind] = [1, text]
            return True
        entry[0] += 1
        return False


class _RepeatFilter(logging.Filter):
    def __init__(self, repeats: _Repeats):
        super().__init__()
        self.repeats = repeats

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        text = record.getMessage()
        return self.repeats.first(_kind('log', text), text)


def _pass_on(previous, message, category, filename, lineno, file=None, line=None):
    """Hand a warning to whatever displayed or recorded warnings before the call."""
    original = getattr(_warnings, '_showwarning_orig', None)
    implementation = getattr(_warnings, '_showwarnmsg_impl', None)
    if previous is original and implementation is not None:
        # The standard display, or a catch_warnings(record=True) list such as
        # pytest's, both of which sit behind the unreplaced showwarning.
        implementation(_warnings.WarningMessage(message, category, filename, lineno, file, line))
    else:
        previous(message, category, filename, lineno, file, line)


def _snippet(text: str, width: int = 90) -> str:
    head = text.split('. ')[0].strip()
    return head if len(head) <= width else head[:width].rstrip() + '...'


def _summarise(repeats: _Repeats, scope: str) -> None:
    for count, text in repeats.seen.values():
        if count > 1:
            logger.warning(f"'{_snippet(text)}' was raised {count} times in this {scope} "
                           f"and shown once.")


@_contextmanager
def collect_repeats():
    """Show each kind of warning and log message once per call and count the rest.

    Inside a notebook cell or a :func:`grouped_warnings` block the count is
    shared with the other calls there and reported when that scope ends.
    Filters still apply as usual: an ignored warning is never counted and an
    ``error`` filter raises at the first occurrence. A call nested inside
    another, or running inside a worker, leaves the counting to the outer one.
    """
    global _repeats
    if _repeats is not None or _in_worker:
        yield
        return
    repeats = _session if _session is not None else _Repeats()
    previous = _warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if repeats.first(_kind(category.__name__, text), text):
            _pass_on(previous, message, category, filename, lineno, file, line)

    log_filter = _RepeatFilter(repeats)
    _repeats = repeats
    _warnings.showwarning = show
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        if _warnings.showwarning is show:
            _warnings.showwarning = previous
        logger.removeFilter(log_filter)
        _repeats = None
        if repeats is not _session:
            _summarise(repeats, 'call')


def groups_repeats(fn):
    """Run a public function inside :func:`collect_repeats`, as ``run()`` is."""
    import functools

    @functools.wraps(fn)
    def grouped(*args, **kwargs):
        with collect_repeats():
            return fn(*args, **kwargs)
    return grouped


@_contextmanager
def call_verbosity(verbose: Optional[bool]):
    """The log level of one call: INFO for True, WARNING for False, unchanged for None."""
    if verbose is None:
        yield
        return
    level = logging.INFO if verbose else logging.WARNING
    previous, handlers = logger.level, [h.level for h in logger.handlers]
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)
        for handler, handler_level in zip(logger.handlers, handlers):
            handler.setLevel(handler_level)


@_contextmanager
def grouped_warnings():
    """Show each kind of NeuralMI warning once across several calls.

    Every call to ``nmi.run()`` or a named quantity inside the block shares one
    count. Each kind of warning or log message is shown the first time it comes
    up, later ones that differ only in their numbers are counted, and the block
    ends with one line per kind giving the count. Messages raised outside a
    NeuralMI call are left alone. Notebook cells are grouped this way without
    it. It is meant for loops in scripts::

        with nmi.grouped_warnings():
            for seed in range(10):
                nmi.run(x, y, seed=seed)
    """
    global _session
    if _session is not None or _in_worker:
        yield
        return
    _session = _Repeats()
    try:
        yield
    finally:
        session, _session = _session, None
        _summarise(session, 'block')


class _CellRepeats(_Repeats):
    """The counts of one IPython cell."""


def _start_cell(*_):
    global _session
    if _session is None and not _in_worker:
        _session = _CellRepeats()


def _end_cell(*_):
    global _session
    if isinstance(_session, _CellRepeats):
        session, _session = _session, None
        _summarise(session, 'cell')


def _group_repeats_by_cell(shell=None) -> bool:
    """In IPython and Jupyter, share the count of repeated messages over each cell.

    Called once when ``neural_mi`` is imported. Returns whether the cell hooks
    were registered by this call.
    """
    if shell is None:
        try:
            from IPython import get_ipython
        except ImportError:
            return False
        shell = get_ipython()
    if shell is None or getattr(shell, '_neural_mi_groups_cells', False):
        return False
    shell.events.register('pre_run_cell', _start_cell)
    shell.events.register('post_run_cell', _end_cell)
    shell._neural_mi_groups_cells = True
    return True


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.getMessage()))


class _Captured:
    """A worker's result with the warnings and log messages its task raised."""

    def __init__(self, result, caught, records):
        self.result = result
        self.caught = caught
        self.records = records


class CapturedTask:
    """Run a task function in a worker and keep what it warned or logged.

    Wrap the function handed to ``Pool.imap`` and pass the results through
    :func:`released`. The function itself must stay importable at module level
    so the wrapper pickles across a ``spawn`` boundary.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, task):
        global _in_worker
        _in_worker = True
        handler = _ListHandler()
        handlers = logger.handlers[:]
        logger.handlers = [handler]
        try:
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter('always')
                result = self.fn(task)
        finally:
            logger.handlers = handlers
        kept = [(w.category.__module__, w.category.__qualname__, str(w.message)) for w in caught]
        return _Captured(result, kept, handler.records)


def _category(module: str, qualname: str) -> type:
    try:
        obj = importlib.import_module(module)
        for part in qualname.split('.'):
            obj = getattr(obj, part)
        if isinstance(obj, type) and issubclass(obj, Warning):
            return obj
    except Exception:
        pass
    return UserWarning


# A worker's results are read inside a progress bar's loop, so tqdm's frames sit
# between the library and the caller too.
_TQDM_DIR = os.path.abspath(importlib.util.find_spec('tqdm').submodule_search_locations[0]) + os.sep


def _caller_location():
    """File, line and module of the first frame outside neural_mi, torch and tqdm."""
    frame = sys._getframe(1)
    while frame is not None and _file(frame).startswith(_PASS_THROUGH + (_TQDM_DIR,)):
        frame = frame.f_back
    if frame is None:
        return __file__, 1, __name__
    return frame.f_code.co_filename, frame.f_lineno, frame.f_globals.get('__name__', '__main__')


def released(results):
    """Yield worker results in order, raising each one's warnings and log messages here."""
    for item in results:
        if isinstance(item, _Captured):
            filename, lineno, module = _caller_location()
            for mod, qualname, text in item.caught:
                _warnings.warn_explicit(text, _category(mod, qualname), filename, lineno, module=module)
            for level, text in item.records:
                logger.log(level, text)
            yield item.result
        else:
            yield item
