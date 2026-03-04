# tests/test_transfer.py
"""Tests for the transfer entropy analysis mode."""
import pytest
import numpy as np
import torch
import neural_mi as nmi
from neural_mi.analysis.transfer import _build_te_arrays

# Minimal training params for fast tests
_PARAMS = {
    'n_epochs': 3, 'learning_rate': 1e-3, 'batch_size': 64,
    'patience': 2, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
}

N = 300  # time samples
H = 10   # history_window


class TestBuildTeArrays:
    """Unit tests for the internal _build_te_arrays helper."""

    def test_output_shapes(self):
        x = np.random.randn(N, 2)
        y = np.random.randn(N, 3)
        x_past, y_past, y_future = _build_te_arrays(x, y, history_window=H, prediction_horizon=1)
        # n_valid = T - H - h + 1: all valid starting positions i where
        # history [i, i+H) and future [i+H, i+H+h) both fit within [0, T).
        n_valid = N - H - 1 + 1  # = N - H = 290
        assert x_past.shape == (n_valid, 2, H)
        assert y_past.shape == (n_valid, 3, H)
        assert y_future.shape == (n_valid, 3, 1)

    def test_tensors_returned(self):
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        x_past, y_past, y_future = _build_te_arrays(x, y, history_window=H)
        assert isinstance(x_past, torch.Tensor)
        assert isinstance(y_past, torch.Tensor)
        assert isinstance(y_future, torch.Tensor)

    def test_prediction_horizon(self):
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        h = 1
        x_past, _, y_future = _build_te_arrays(x, y, history_window=H, prediction_horizon=h)
        n_valid = N - H - h + 1  # correct: T - H - h + 1 valid windows
        assert x_past.shape[0] == n_valid


class TestTransferEntropy:
    """Integration tests for the mode='transfer' dispatch."""

    def test_te_returns_results_object(self):
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='transfer',
            history_window=H,
            prediction_horizon=1,
            base_params=_PARAMS,
            n_workers=1,
        )
        assert results is not None
        assert results.mode == 'transfer'
        assert results.mi_estimate is not None
        assert isinstance(results.mi_estimate, float)
        assert np.isfinite(results.mi_estimate)

    def test_te_details_keys(self):
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='transfer',
            history_window=H,
            base_params=_PARAMS,
            n_workers=1,
        )
        assert 'i_xypast_yfuture' in results.details
        assert 'i_ypast_yfuture' in results.details
        assert 'n_samples' in results.details

    def test_te_estimate_equals_difference(self):
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='transfer',
            history_window=H,
            base_params=_PARAMS,
            n_workers=1,
        )
        expected = results.details['i_xypast_yfuture'] - results.details['i_ypast_yfuture']
        assert abs(results.mi_estimate - expected) < 1e-6

    def test_te_missing_history_window_raises(self):
        """mode='transfer' without history_window should raise ValueError."""
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        with pytest.raises((ValueError, TypeError)):
            nmi.run(
                x_data=x, y_data=y,
                mode='transfer',
                base_params=_PARAMS,
                n_workers=1,
            )
