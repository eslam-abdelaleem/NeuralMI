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
        output=Output(units='nats'),                    # units->output_units
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


def test_output_track_spectral_history_routes(capture_engine):
    nmi.run([[1]], [[1]], mode='estimate',
            output=Output(track_spectral_history=True))
    kw = capture_engine['kw']
    assert kw['track_spectral_history'] is True


def test_transfer_bidirectional_is_renamed(capture_engine):
    nmi.run([[1]], [[1]], mode='transfer',
            transfer=Transfer(history_window=10, bidirectional=True))
    kw = capture_engine['kw']
    assert kw['history_window'] == 10
    assert kw['bidirectional_te'] is True          # Transfer.bidirectional's schema name
    assert 'bidirectional' not in kw               # the raw name must not leak through


def test_conditional_w_split(capture_engine):
    w = [[0.0]]
    nmi.run([[1]], [[1]], mode='conditional',
            conditional=Conditional(w_data=w, rigorous=True))
    kw = capture_engine['kw']
    assert kw['w_data'] == w
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


# ---------------------------------------------------------------------------
# Settings that would have no effect are named before any work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model, training, named', [
    (Model(kernel_size=5), None, "kernel_size"),
    (Model(embedding_model='gru', nhead=2), None, "nhead"),
    (Model(n_layers_head=3), None, "n_layers_head"),
    (Model(critic_type='separable'), Training(lr_head_multiplier=5.0), "lr_head_multiplier"),
    (Model(beta=2.0), None, "beta"),
    (Model(decoder_lambda=2.0), None, "decoder_lambda"),
    (Model(custom_critic=object(), embedding_dim=8), None, "embedding_dim"),
])
def test_ineffective_setting_warns(capture_engine, model, training, named):
    with pytest.warns(UserWarning, match=f"no effect in this call.*{named}"):
        nmi.run([[1]], [[1]], mode='estimate', model=model, training=training)


@pytest.mark.parametrize('mode, model', [
    ('estimate', Model(embedding_model='cnn', kernel_size=5)),
    ('estimate', Model(critic_type='hybrid', n_layers_head=3)),
    ('estimate', Model(use_variational=True, beta=2.0)),
    ('estimate', Model(use_decoder=True, decoder_lambda=2.0)),
    ('estimate', Model(embedding_model='mlp', embedding_model_y='gru', bidirectional=True)),
    ('dimensionality', Model(n_layers_head=3)),     # dimensionality trains a hybrid critic
])
def test_effective_setting_is_quiet(capture_engine, mode, model):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message="These settings have no effect")
        nmi.run([[1]], [[1]], mode=mode, model=model)


def test_setting_spelled_out_at_its_default_is_quiet(capture_engine):
    import warnings
    from neural_mi.defaults import BASE_PARAMS_SCHEMA
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message="These settings have no effect")
        nmi.run([[1]], [[1]], mode='estimate',
                model=Model(kernel_size=BASE_PARAMS_SCHEMA['kernel_size']['default']))


def test_unknown_estimator_parameter_raises_before_training(capture_engine):
    with pytest.raises(ValueError, match=r"Estimator\(params=...\) names \['clp'\].*'clip'"):
        nmi.run([[1]], [[1]], mode='estimate',
                estimator=nmi.Estimator(name='smile', params={'clp': 5.0}))
    assert 'kw' not in capture_engine


def test_known_estimator_parameter_is_accepted(capture_engine):
    nmi.run([[1]], [[1]], mode='estimate',
            estimator=nmi.Estimator(name='smile', params={'clip': 5.0}))
    assert 'kw' in capture_engine


# ---------------------------------------------------------------------------
# The defaults that apply agree with the documented ones
# ---------------------------------------------------------------------------

def test_engine_defaults_match_the_schema():
    """_run_flat's keyword defaults are the ones that apply, and defaults.py is
    the one PARAMETERS.md is checked against, so the two must agree."""
    import inspect
    from neural_mi.defaults import BASE_PARAMS_SCHEMA
    signature = inspect.signature(importlib.import_module('neural_mi.run')._run_flat)
    differ = {}
    for name, p in signature.parameters.items():
        if p.default is inspect.Parameter.empty or p.default is None or name not in BASE_PARAMS_SCHEMA:
            continue
        documented = BASE_PARAMS_SCHEMA[name].get('default')
        if documented != p.default:
            differ[name] = (p.default, documented)
    assert differ == {}


@pytest.mark.parametrize('mode, output, named', [
    ('estimate', Output(return_rotated_embeddings=True), "return_rotated_embeddings"),
    ('estimate', Output(return_embeddings=True, return_rotated_embeddings=True,
                        rotated_embeddings_per_epoch=True), "rotated_embeddings_per_epoch"),
    ('estimate', Output(return_embeddings=True, return_rotation_matrices=True),
     "return_rotation_matrices"),
    ('dimensionality', Output(return_rotation_matrices=True), "return_rotation_matrices"),
    ('dimensionality', Output(track_embeddings=True, rotated_embeddings_per_epoch=True),
     "rotated_embeddings_per_epoch"),
])
def test_rotation_setting_nothing_reads_warns(capture_engine, mode, output, named):
    with pytest.warns(UserWarning, match=f"no effect in this call.*{named}"):
        nmi.run([[1]], [[1]], mode=mode, output=output)


@pytest.mark.parametrize('mode, output', [
    ('estimate', Output(return_embeddings=True, return_rotated_embeddings=True,
                        return_rotation_matrices=True)),
    ('estimate', Output(track_embeddings=True, return_rotated_embeddings=True,
                        rotated_embeddings_per_epoch=True)),
    ('dimensionality', Output(return_embeddings=True, return_rotated_embeddings=True,
                              return_rotation_matrices=True)),
    ('dimensionality', None),
])
def test_rotation_setting_something_reads_is_quiet(capture_engine, mode, output):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message="These settings have no effect")
        nmi.run([[1]], [[1]], mode=mode, output=output)
