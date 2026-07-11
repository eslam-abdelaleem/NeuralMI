# tests/test_amp_and_names.py
"""Tests for use_amp (Feature 3) and named variable support (Feature 5)."""
import inspect
import numpy as np
import pytest

import neural_mi as nmi

_PARAMS = {
    'n_epochs': 3, 'learning_rate': 1e-3, 'batch_size': 64,
    'patience': 2, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
}
N = 300


class TestAMP:
    """Tests for the use_amp parameter (Feature 3)."""

    def test_use_amp_auto_cpu_completes(self):
        """use_amp='auto' on CPU runs without error and returns a finite MI."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=_PARAMS, use_amp='auto', n_workers=1)
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)

    def test_use_amp_false_completes(self):
        """use_amp=False runs without error on CPU."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=_PARAMS, use_amp=False, n_workers=1)
        assert np.isfinite(r.mi_estimate)

    def test_use_amp_true_cpu_no_crash(self):
        """use_amp=True on CPU must not raise (AMP silently no-ops on CPU)."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=_PARAMS, use_amp=True, n_workers=1)
        assert r.mi_estimate is not None

    def test_use_amp_in_base_params_schema(self):
        """use_amp is in BASE_PARAMS_SCHEMA with default 'auto'."""
        from neural_mi.defaults import BASE_PARAMS_SCHEMA
        assert 'use_amp' in BASE_PARAMS_SCHEMA
        assert BASE_PARAMS_SCHEMA['use_amp']['default'] == 'auto'

    def test_use_amp_in_run_signature(self):
        """run() has use_amp as an explicit parameter with default 'auto'."""
        sig = inspect.signature(nmi.run)
        assert 'use_amp' in sig.parameters
        assert sig.parameters['use_amp'].default == 'auto'

    def test_use_amp_via_base_params(self):
        """use_amp can also be set via base_params without error."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        params = dict(_PARAMS, use_amp=False)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=params, n_workers=1)
        assert np.isfinite(r.mi_estimate)

    def test_use_amp_sweep_mode(self):
        """use_amp='auto' works with mode='sweep'."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='sweep',
                    sweep_grid={'embedding_dim': [4, 8]},
                    base_params=_PARAMS, use_amp='auto', n_workers=1)
        assert r.dataframe is not None


class TestNamedVariables:
    """Tests for x_name, y_name, channel_names_x, channel_names_y (Feature 5)."""

    def test_x_name_stored_in_params(self):
        """x_name is stored in result.params when provided."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=_PARAMS, x_name='LFP', n_workers=1)
        assert r.params.get('x_name') == 'LFP'

    def test_y_name_stored_in_params(self):
        """y_name is stored in result.params when provided."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=_PARAMS, y_name='MUA', n_workers=1)
        assert r.params.get('y_name') == 'MUA'

    def test_both_names_stored_together(self):
        """x_name and y_name can be provided together and both appear in params."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate', base_params=_PARAMS,
                    x_name='LFP', y_name='spikes', n_workers=1)
        assert r.params['x_name'] == 'LFP'
        assert r.params['y_name'] == 'spikes'

    def test_no_names_leaves_params_clean(self):
        """Omitting names does not add spurious keys to result.params."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        r = nmi.run(x_data=x, y_data=y, mode='estimate',
                    base_params=_PARAMS, n_workers=1)
        assert 'x_name' not in r.params
        assert 'y_name' not in r.params
        assert 'channel_names_x' not in r.params

    def test_channel_names_x_in_pairwise_details_self(self):
        """channel_names_x appears as variable_names_x and variable_names_y in self-pairwise details."""
        x_3ch = np.random.randn(N, 3)  # 3 channels, treated as (N, 3, 1)
        names = ['alpha', 'beta', 'gamma']
        r = nmi.run(x_data=x_3ch, mode='pairwise',
                    base_params=_PARAMS, channel_names_x=names, n_workers=1)
        assert r.details.get('variable_names_x') == names
        assert r.details.get('variable_names_y') == names

    def test_channel_names_fallback_integer_when_omitted(self):
        """Without channel_names, pairwise details contain no variable_names keys."""
        x_3ch = np.random.randn(N, 3)
        r = nmi.run(x_data=x_3ch, mode='pairwise',
                    base_params=_PARAMS, n_workers=1)
        # The details should not have variable_names_x injected
        assert 'variable_names_x' not in r.details
        assert 'variable_names_y' not in r.details

    def test_channel_names_x_in_run_signature(self):
        """run() has channel_names_x and channel_names_y as explicit parameters."""
        sig = inspect.signature(nmi.run)
        assert 'channel_names_x' in sig.parameters
        assert 'channel_names_y' in sig.parameters
        assert sig.parameters['channel_names_x'].default is None
        assert sig.parameters['channel_names_y'].default is None
