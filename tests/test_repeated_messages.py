"""A call shows each kind of warning or log message once and counts the repeats."""
import logging
import warnings

import pytest

import neural_mi as nmi
from neural_mi.logger import collect_repeats, logger

CAPPED = 'training samples available'


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
    caught, logged = _run_recording(n_workers, caplog)
    capped = [w for w in caught if CAPPED in str(w.message)]
    assert len(capped) == 1
    summary = [m for m in logged if 'times in this call and shown once' in m and 'batch_size=256' in m]
    assert len(summary) == 1


def test_a_warning_from_a_worker_names_the_callers_line(caplog):
    caught, _ = _run_recording(2, caplog)
    capped = [w for w in caught if CAPPED in str(w.message)]
    assert capped and capped[0].filename == __file__


def test_messages_that_differ_beyond_their_numbers_are_all_shown():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with collect_repeats():
            for n in (10, 20, 30):
                warnings.warn(f"value {n} is too large", UserWarning)
            warnings.warn("something else", UserWarning)
    assert [str(w.message) for w in caught] == ['value 10 is too large', 'something else']


def test_an_error_filter_still_raises_at_the_first_occurrence():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UserWarning, match='value 1'):
            with collect_repeats():
                warnings.warn("value 1 is too large", UserWarning)


def test_an_ignored_warning_is_neither_shown_nor_counted(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore')
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            with collect_repeats():
                for n in (1, 2):
                    warnings.warn(f"value {n} is too large", UserWarning)
    assert not caught
    assert not any('shown once' in r.getMessage() for r in caplog.records)


def test_a_repeated_log_message_is_shown_once_and_counted(caplog):
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        with collect_repeats():
            for n in (1, 2, 3):
                logger.warning(f"rung {n} is small")
    messages = [r.getMessage() for r in caplog.records]
    assert messages[0] == 'rung 1 is small'
    assert sum('rung' in m and 'small' in m and 'shown once' not in m for m in messages) == 1
    assert any("was raised 3 times in this call and shown once" in m for m in messages)


def test_a_nested_call_leaves_the_counting_to_the_outer_one(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            with collect_repeats():
                warnings.warn("value 1 is too large", UserWarning)
                with collect_repeats():
                    warnings.warn("value 2 is too large", UserWarning)
    assert len(caught) == 1
    summaries = [r.getMessage() for r in caplog.records if 'shown once' in r.getMessage()]
    assert summaries == ["'value 1 is too large' was raised 2 times in this call and shown once."]


# ---------------------------------------------------------------------------
# Across calls: a notebook cell or a grouped_warnings() block
# ---------------------------------------------------------------------------

def _three_calls():
    """Three NeuralMI calls that each raise the same kind of warning once."""
    for n in (10, 20, 30):
        with collect_repeats():
            warnings.warn(f"Only {n} samples reach the estimator.", UserWarning)


def _shown_and_summary(caught, caplog, scope):
    shown = [w for w in caught if 'reach the estimator' in str(w.message)]
    summary = [r.getMessage() for r in caplog.records
               if f'was raised 3 times in this {scope} and shown once' in r.getMessage()]
    return len(shown), len(summary)


def test_a_block_shows_a_warning_repeated_across_calls_once(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            with nmi.grouped_warnings():
                _three_calls()
    assert _shown_and_summary(caught, caplog, 'block') == (1, 1)


def test_without_a_block_each_call_shows_its_own(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _three_calls()
    assert len([w for w in caught if 'reach the estimator' in str(w.message)]) == 3


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


def test_a_notebook_cell_shows_a_warning_repeated_across_calls_once(caplog):
    from neural_mi.logger import _group_repeats_by_cell
    shell = _Shell()
    assert _group_repeats_by_cell(shell)
    assert not _group_repeats_by_cell(shell)          # registered once
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            shell.run_cell(_three_calls)
    assert _shown_and_summary(caught, caplog, 'cell') == (1, 1)


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


def test_create_dataset_groups_its_repeats_like_run(caplog, monkeypatch):
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
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            with nmi.grouped_warnings():
                for _ in range(3):
                    create_dataset(x, y, processor_type_x='continuous', processor_type_y='continuous',
                                   processor_params_x={'window_size': 5},
                                   processor_params_y={'window_size': 5})
    assert len([w for w in caught if 'on every call' in str(w.message)]) == 1
    assert any("A data warning raised on every call" in r.getMessage() and 'in this block' in r.getMessage()
               for r in caplog.records)
