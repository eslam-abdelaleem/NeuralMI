# tests/test_amp_and_names.py
"""Tests for use_amp (Feature 3) and named variable support (Feature 5)."""
from dataclasses import fields

import numpy as np

import neural_mi as nmi
from neural_mi import Model, Training, Output

_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
N = 300


def _training(**over):
    """Base training config with per-test overrides (e.g. use_amp)."""
    return Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2, **over)


class TestAMP:
    """Tests for the use_amp parameter (Feature 3)."""

    def test_use_amp_auto_cpu_completes(self):
        """use_amp='auto' on CPU runs without error and returns a finite MI."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL, training=_training(use_amp='auto'), n_workers=1)
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)

    def test_use_amp_false_completes(self):
        """use_amp=False runs without error on CPU."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL, training=_training(use_amp=False), n_workers=1)
        assert np.isfinite(r.mi_estimate)

    def test_use_amp_true_cpu_no_crash(self):
        """use_amp=True on CPU must not raise (AMP silently no-ops on CPU)."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL, training=_training(use_amp=True), n_workers=1)
        assert r.mi_estimate is not None

    def test_use_amp_in_base_params_schema(self):
        """use_amp is in BASE_PARAMS_SCHEMA with default 'auto'."""
        from neural_mi.defaults import BASE_PARAMS_SCHEMA
        assert 'use_amp' in BASE_PARAMS_SCHEMA
        assert BASE_PARAMS_SCHEMA['use_amp']['default'] == 'auto'

    def test_use_amp_is_training_field(self):
        """use_amp is a field on the Training config with an unset (None) default."""
        f = {f.name: f for f in fields(Training)}
        assert 'use_amp' in f
        assert f['use_amp'].default is None  # unset -> schema default 'auto' applies

    def test_use_amp_via_training_dict(self):
        """use_amp can also be set via a plain training dict without error."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL,
                    training={'n_epochs': 3, 'batch_size': 64, 'use_amp': False},
                    n_workers=1)
        assert np.isfinite(r.mi_estimate)

    def test_use_amp_sweep_mode(self):
        """use_amp='auto' works with mode='sweep'."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='sweep',
                    sweep_grid={'embedding_dim': [4, 8]},
                    model=_MODEL, training=_training(use_amp='auto'), n_workers=1)
        assert r.dataframe is not None


class TestNamedVariables:
    """Tests for x_name, y_name, channel_names_x, channel_names_y (Feature 5)."""

    def test_x_name_stored_in_params(self):
        """x_name is stored in result.params when provided."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL, training=_training(), output=Output(x_name='LFP'), n_workers=1)
        assert r.params.get('x_name') == 'LFP'

    def test_y_name_stored_in_params(self):
        """y_name is stored in result.params when provided."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL, training=_training(), output=Output(y_name='MUA'), n_workers=1)
        assert r.params.get('y_name') == 'MUA'

    def test_both_names_stored_together(self):
        """x_name and y_name can be provided together and both appear in params."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate', model=_MODEL, training=_training(),
                    output=Output(x_name='LFP', y_name='spikes'), n_workers=1)
        assert r.params['x_name'] == 'LFP'
        assert r.params['y_name'] == 'spikes'

    def test_no_names_leaves_params_clean(self):
        """Omitting names does not add spurious keys to result.params."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x, y, mode='estimate',
                    model=_MODEL, training=_training(), n_workers=1)
        assert 'x_name' not in r.params
        assert 'y_name' not in r.params
        assert 'channel_names_x' not in r.params

    def test_channel_names_x_in_pairwise_details_self(self):
        """channel_names_x appears as variable_names_x and variable_names_y in self-pairwise details."""
        x_3ch = np.random.randn(N, 3)  # 3 channels, treated as (N, 3, 1)
        names = ['alpha', 'beta', 'gamma']
        r = nmi.run(x_3ch, mode='pairwise',
                    model=_MODEL, training=_training(),
                    output=Output(channel_names_x=names), n_workers=1)
        assert r.details.get('variable_names_x') == names
        assert r.details.get('variable_names_y') == names

    def test_channel_names_fallback_integer_when_omitted(self):
        """Without channel_names, pairwise details contain no variable_names keys."""
        x_3ch = np.random.randn(N, 3)
        r = nmi.run(x_3ch, mode='pairwise',
                    model=_MODEL, training=_training(), n_workers=1)
        # The details should not have variable_names_x injected
        assert 'variable_names_x' not in r.details
        assert 'variable_names_y' not in r.details

    def test_channel_names_are_output_fields(self):
        """channel_names_x/y are fields on the Output config."""
        names = {f.name for f in fields(Output)}
        assert 'channel_names_x' in names
        assert 'channel_names_y' in names
