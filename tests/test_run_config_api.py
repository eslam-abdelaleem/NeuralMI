"""Tests that the config-based run() lowers correctly onto the engine."""
import importlib

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi import (Model, Training, Split, Output, Processing,
                       Precision, Transfer, Conditional)

# neural_mi.run is the *function* (shadowing the submodule), so grab the module explicitly.
run_module = importlib.import_module('neural_mi.run')


@pytest.fixture
def capture_engine(monkeypatch):
    """Replace the internal engine with a capturing stub; return the captured kwargs."""
    seen = {}

    def fake(x_data, y_data=None, **kw):
        seen['x'], seen['y'] = x_data, y_data
        seen['kw'] = kw
        return "ENGINE_CALLED"

    monkeypatch.setattr(run_module, '_run_flat', fake)
    return seen


def test_shared_configs_route_to_engine(capture_engine):
    out = nmi.run(
        [[1]], [[1]], mode='estimate',
        model=Model(embedding_dim=8, dropout=0.1),      # embedding_dim->base_params; dropout->flat
        training=Training(n_epochs=5),                  # named engine kwarg -> flat
        split=Split(mode='random', gap_fraction=0.0),   # renamed -> split_mode/split_gap_fraction (flat)
        estimator='smile',                              # string shorthand
        output=Output(units='nats', x_name='LFP'),      # units->output_units; x_name label
        n_workers=3, seed=7,
    )
    assert out == "ENGINE_CALLED"
    kw = capture_engine['kw']
    assert kw['dropout'] == 0.1
    assert kw['n_epochs'] == 5
    assert kw['split_mode'] == 'random'
    assert kw['split_gap_fraction'] == 0.0
    assert kw['estimator'] == 'smile'
    assert kw['output_units'] == 'nats'
    assert kw['x_name'] == 'LFP'
    assert kw['random_seed'] == 7
    assert kw['n_workers'] == 3
    assert kw['mode'] == 'estimate'
    # base_params-only keys collected into the dict, not spread as engine kwargs
    assert kw['base_params'] == {'embedding_dim': 8}


def test_processing_and_precision_route(capture_engine):
    nmi.run(
        [[1]], [[1]], mode='precision',
        processing=Processing(x='spike', x_params={'bin_size': 0.01}),
        precision=Precision(tau_grid=[0.1, 0.2], corrupt_target='y'),
    )
    kw = capture_engine['kw']
    assert kw['processor_type_x'] == 'spike'
    assert kw['processor_params_x'] == {'bin_size': 0.01}
    assert kw['tau_grid'] == [0.1, 0.2]
    assert kw['corrupt_target'] == 'y'


def test_transfer_bidirectional_is_renamed(capture_engine):
    nmi.run([[1]], [[1]], mode='transfer',
            transfer=Transfer(history_window=10, bidirectional=True))
    kw = capture_engine['kw']
    assert kw['history_window'] == 10
    assert kw['bidirectional_te'] is True          # renamed from Transfer.bidirectional
    assert 'bidirectional' not in kw               # the raw name must not leak through


def test_conditional_z_split(capture_engine):
    z = [[0.0]]
    nmi.run([[1]], [[1]], mode='conditional',
            conditional=Conditional(z_data=z, rigorous=True))
    kw = capture_engine['kw']
    assert kw['z_data'] == z
    assert kw['rigorous'] is True


def test_removed_flat_kwarg_raises(capture_engine):
    with pytest.raises(TypeError, match="config objects"):
        nmi.run([[1]], [[1]], mode='estimate', n_epochs=50)  # flat kwargs are not accepted


def test_stray_mode_config_warns(capture_engine):
    with pytest.warns(UserWarning, match="ignored"):
        nmi.run([[1]], [[1]], mode='estimate',
                precision=Precision(tau_grid=[0.1]))  # precision cfg but estimate mode


def test_end_to_end_estimate_runs():
    """A real (un-mocked) estimate through the config API returns a finite MI."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 1)).astype('float32')
    y = (x + 0.3 * rng.standard_normal((400, 1))).astype('float32')
    res = nmi.run(
        x, y, mode='estimate',
        split=Split(mode='random'),
        model=Model(embedding_dim=8, hidden_dim=32),
        training=Training(n_epochs=5, batch_size=64),
        seed=0, show_progress=False,
    )
    assert np.isfinite(res.mi_estimate)
