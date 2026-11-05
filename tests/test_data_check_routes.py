# tests/test_data_check_routes.py
"""Every route through run() runs the data check on the rows that reach the estimator.

The data check (run._warn_small_sample) reports the rows and warns when they
are few. It used to skip a processor grid in mode='sweep', which windowed
inside each task, and transfer entropy on every route. Each route now reaches
it, and a processor grid reaches it once per setting, with that setting's rows.
"""
import importlib
import warnings

import numpy as np
import pytest

import neural_mi as nmi

# The module, not the function neural_mi exports under the same name.
run_module = importlib.import_module('neural_mi.run')

RNG = np.random.default_rng(0)
X = (np.cumsum(RNG.standard_normal((1200, 2)), axis=0) * 0.1).astype(np.float32)
Y = (X + RNG.standard_normal((1200, 2))).astype(np.float32)
W = (X[:, :1] + RNG.standard_normal((1200, 1))).astype(np.float32)
QUICK = dict(training=nmi.Training(n_epochs=1), show_progress=False, seed=0, n_workers=1)


@pytest.fixture
def checked(monkeypatch):
    """The (count, unit) of every data check the call runs."""
    seen = []
    original = run_module._warn_small_sample

    def spy(n_samples, unit, *args, **kwargs):
        seen.append((n_samples, unit))
        return original(n_samples, unit, *args, **kwargs)

    monkeypatch.setattr(run_module, '_warn_small_sample', spy)
    return seen


def _quietly(call):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return call()


@pytest.mark.parametrize('mode, config', [('sweep', {}), ('lag', {'lag': nmi.Lag(lag_range=[0, 2])}),
                                          ('conditional', {'conditional': nmi.Conditional(w_data=W)})])
def test_a_processor_grid_checks_the_rows_of_each_setting(checked, mode, config):
    processing = nmi.Processing(x='continuous', x_params={'window_size': 5})
    grid = {'window_size': [5, 10], 'run_id': [0]}
    _quietly(lambda: nmi.run(X, Y, mode=mode, sweep_grid=grid, processing=processing, **config, **QUICK))
    counts = [n for n, _ in checked]
    assert len(counts) == 2 and counts[0] > counts[1] > 0


@pytest.mark.parametrize('route', ['series', 'processing', 'named'])
def test_transfer_entropy_checks_the_histories_it_trains_on(checked, route):
    transfer = nmi.Transfer(history_window=3)
    if route == 'named':
        _quietly(lambda: nmi.transfer_entropy(X, Y, history_window=3, **QUICK))
    elif route == 'processing':
        processing = nmi.Processing(x='continuous', x_params={'window_size': 1})
        _quietly(lambda: nmi.run(X, Y, mode='transfer', transfer=transfer, processing=processing, **QUICK))
    else:
        _quietly(lambda: nmi.run(X, Y, mode='transfer', transfer=transfer, **QUICK))
    # Three steps of each past and one of Y's future: one history per step but the first three, on
    # every route. processing= puts the streams on a grid of one-step rows, one per sample.
    assert checked == [(len(X) - 3, 'samples')]
