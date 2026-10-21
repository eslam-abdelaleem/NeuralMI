"""A call shows each kind of warning or log message once and counts the repeats.

The count comes through the channel of the message it counts: a warning of the
same category for a warning, a log line at the same level for a log message.
"""
import logging
import warnings

import pytest

import neural_mi as nmi
from neural_mi.logger import collect_repeats, logger

CAPPED = 'training samples available'
COUNTED = 'shown once'


def _messages(caught, part, counts=False):
    """The caught warnings containing ``part``: the counts of repeats, or the warnings themselves."""
    return [w for w in caught if part in str(w.message) and (COUNTED in str(w.message)) == counts]


def _small_rigorous(n_workers):
    # 300 samples and batches of 256: every rung past gamma=1 caps its batch,
    # so the same warning comes up once per chunk.
    x, y = nmi.generators.generate_correlated_gaussians(300, 2, 1.0, use_torch=False, seed=0)
    return nmi.run(x, y, mode='rigorous',
                   rigorous=nmi.Rigorous(gamma_range=range(1, 6), min_gamma_points=3),
                   training=nmi.Training(n_epochs=2, batch_size=256),
                   n_workers=n_workers, seed=0, show_progress=False)


def _run_recording(n_workers, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            _small_rigorous(n_workers)
    return caught, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize('n_workers', [1, 2], ids=['in-process', 'workers'])
def test_a_repeated_warning_is_shown_once_and_counted(n_workers, caplog):
    caught, _ = _run_recording(n_workers, caplog)
    assert len(_messages(caught, CAPPED)) == 1
    [count] = _messages(caught, 'batch_size=256', counts=True)
    assert 'times in this call' in str(count.message) and count.category is UserWarning


def test_a_warning_from_a_worker_names_the_callers_line(caplog):
    caught, _ = _run_recording(2, caplog)
    capped = _messages(caught, CAPPED)
    assert capped and capped[0].filename == __file__


def test_messages_that_differ_beyond_their_numbers_are_all_shown():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with collect_repeats():
            for n in (10, 20, 30):
                warnings.warn(f"value {n} is too large", UserWarning)
            warnings.warn("something else", UserWarning)
    assert [str(w.message) for w in caught] == [
        'value 10 is too large', 'something else',
        "'value 10 is too large' was raised 3 times in this call and shown once."]


def test_an_error_filter_still_raises_at_the_first_occurrence():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UserWarning, match='value 1'):
            with collect_repeats():
                warnings.warn("value 1 is too large", UserWarning)


def test_an_ignored_warning_is_neither_shown_nor_counted():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore')
        with collect_repeats():
            for n in (1, 2):
                warnings.warn(f"value {n} is too large", UserWarning)
        warnings.simplefilter('always')     # a count raised now would be recorded
    assert not caught


def test_a_repeated_log_message_is_shown_once_and_counted_at_its_level(caplog):
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        with collect_repeats():
            for n in (1, 2, 3):
                logger.error(f"rung {n} failed")
    records = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert records == [(logging.ERROR, 'rung 1 failed'),
                       (logging.ERROR, "'rung 1 failed' was raised 3 times in this call and shown once.")]


def test_a_nested_call_leaves_the_counting_to_the_outer_one(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            with collect_repeats():
                warnings.warn("value 1 is too large", UserWarning)
                with collect_repeats():
                    warnings.warn("value 2 is too large", UserWarning)
    assert [str(w.message) for w in caught] == [
        'value 1 is too large', "'value 1 is too large' was raised 2 times in this call and shown once."]


# ---------------------------------------------------------------------------
# Across calls: a notebook cell or a grouped_warnings() block
# ---------------------------------------------------------------------------

def _three_calls():
    """Three NeuralMI calls that each raise the same kind of warning once."""
    for n in (10, 20, 30):
        with collect_repeats():
            warnings.warn(f"Only {n} samples reach the estimator.", UserWarning)


def _shown_and_summary(caught, scope):
    shown = _messages(caught, 'reach the estimator')
    summary = _messages(caught, f'was raised 3 times in this {scope}', counts=True)
    return len(shown), len(summary)


def test_a_block_shows_a_warning_repeated_across_calls_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with nmi.grouped_warnings():
            _three_calls()
    assert _shown_and_summary(caught, 'block') == (1, 1)


def test_without_a_block_each_call_shows_its_own(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _three_calls()
    assert len(_messages(caught, 'reach the estimator')) == 3


class _Shell:
    """Stands in for IPython's shell: it keeps the registered cell callbacks."""

    class _Events:
        def __init__(self):
            self.callbacks = {}

        def register(self, name, fn):
            self.callbacks.setdefault(name, []).append(fn)

    def __init__(self):
        self.events = self._Events()

    def run_cell(self, body):
        for fn in self.events.callbacks.get('pre_run_cell', []):
            fn(None)
        try:
            body()
        finally:
            for fn in self.events.callbacks.get('post_run_cell', []):
                fn(None)


def test_a_notebook_cell_shows_a_warning_repeated_across_calls_once():
    from neural_mi.logger import _group_repeats_by_cell
    shell = _Shell()
    assert _group_repeats_by_cell(shell)
    assert not _group_repeats_by_cell(shell)          # registered once
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        shell.run_cell(_three_calls)
    assert _shown_and_summary(caught, 'cell') == (1, 1)
    # The count names the line the warning first named, not the code that ends the cell.
    [shown] = _messages(caught, 'reach the estimator')
    [summary] = _messages(caught, 'was raised 3 times in this cell', counts=True)
    assert (summary.filename, summary.lineno) == (shown.filename, shown.lineno)


def test_a_cell_leaves_warnings_outside_neural_mi_calls_alone():
    from neural_mi.logger import _group_repeats_by_cell
    shell = _Shell()
    _group_repeats_by_cell(shell)

    def body():
        for _ in range(3):
            warnings.warn("A warning of the user's own.", UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        shell.run_cell(body)
    assert len([w for w in caught if "user's own" in str(w.message)]) == 3


# ---------------------------------------------------------------------------
# The log level of a call
# ---------------------------------------------------------------------------

def _logged_during_a_call(verbose):
    from neural_mi.logger import call_verbosity
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record.levelname)

    keep = Keep(level=logging.DEBUG)
    logger.addHandler(keep)
    try:
        with call_verbosity(verbose):
            logger.info("an informational message")
            logger.warning("a warning")
    finally:
        logger.removeHandler(keep)
    return records


@pytest.mark.parametrize('verbose, shown', [(None, []), (False, ['WARNING']),
                                            (True, ['INFO', 'WARNING'])])
def test_set_verbosity_holds_unless_a_call_asks_for_a_level(verbose, shown):
    previous = logger.level
    nmi.set_verbosity('ERROR')
    try:
        assert _logged_during_a_call(verbose) == shown
        assert logger.level == logging.ERROR          # restored after the call
    finally:
        logger.setLevel(previous)
        for handler in logger.handlers:
            handler.setLevel(previous)


def test_create_dataset_groups_its_repeats_like_run(monkeypatch):
    """A loop of create_dataset calls in one block shows each warning once."""
    import numpy as np
    from neural_mi.data import create_dataset, handler
    original = handler.resolve_step_size

    def warning_step(*args, **kwargs):
        warnings.warn("A data warning raised on every call.", UserWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(handler, 'resolve_step_size', warning_step)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((400, 2)), rng.standard_normal((400, 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with nmi.grouped_warnings():
            for _ in range(3):
                create_dataset(x, y, processor_type_x='continuous', processor_type_y='continuous',
                               processor_params_x={'window_size': 5},
                               processor_params_y={'window_size': 5})
    assert len(_messages(caught, 'on every call')) == 1
    assert len(_messages(caught, 'in this block', counts=True)) == 1
