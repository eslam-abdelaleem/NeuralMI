# tests/test_pairwise.py
"""Tests for the pairwise MI matrix analysis mode."""
import pytest
import numpy as np
import pandas as pd
import torch
import neural_mi as nmi

# Minimal training params
_PARAMS = {
    'n_epochs': 3, 'learning_rate': 1e-3, 'batch_size': 64,
    'patience': 2, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
}

N = 300   # samples
N_CH = 4  # channels


class TestPairwiseMI:
    """Tests for mode='pairwise'."""

    def test_pairwise_self_returns_upper_triangle(self):
        """x_data only → upper triangle of (n_ch x n_ch) matrix = C(n_ch, 2) pairs."""
        x = torch.from_numpy(np.random.randn(N, N_CH).astype(np.float32))
        results = nmi.run(
            x_data=x,
            mode='pairwise',
            base_params=_PARAMS,
            n_workers=1,
        )
        expected_pairs = N_CH * (N_CH - 1) // 2  # C(4, 2) = 6
        assert results.dataframe is not None
        assert isinstance(results.dataframe, pd.DataFrame)
        assert len(results.dataframe) == expected_pairs

    def test_pairwise_self_dataframe_columns(self):
        """Pairwise DataFrame must have ch_x, ch_y, mi_mean, and mi_std columns."""
        x = torch.from_numpy(np.random.randn(N, N_CH).astype(np.float32))
        results = nmi.run(
            x_data=x,
            mode='pairwise',
            base_params=_PARAMS,
            n_workers=1,
        )
        for col in ('ch_x', 'ch_y', 'mi_mean', 'mi_std'):
            assert col in results.dataframe.columns, f"Missing column: {col}"
        assert 'mi_estimate' not in results.dataframe.columns, (
            "Old column 'mi_estimate' should no longer be present; use 'mi_mean'."
        )

    def test_pairwise_cross_returns_full_matrix(self):
        """With x_data and y_data → (n_ch_x × n_ch_y) pairs."""
        N_CHX, N_CHY = 3, 2
        x = torch.from_numpy(np.random.randn(N, N_CHX).astype(np.float32))
        y = torch.from_numpy(np.random.randn(N, N_CHY).astype(np.float32))
        results = nmi.run(
            x_data=x, y_data=y,
            mode='pairwise',
            base_params=_PARAMS,
            n_workers=1,
        )
        assert len(results.dataframe) == N_CHX * N_CHY

    def test_pairwise_all_estimates_finite(self):
        """Every MI mean in the pairwise matrix should be finite."""
        x = torch.from_numpy(np.random.randn(N, 3).astype(np.float32))
        results = nmi.run(
            x_data=x,
            mode='pairwise',
            base_params=_PARAMS,
            n_workers=1,
        )
        assert np.all(np.isfinite(results.dataframe['mi_mean'].values))
        assert np.all(results.dataframe['mi_std'].values >= 0)

    def test_pairwise_mode_field(self):
        """Results.mode should be 'pairwise'."""
        x = torch.from_numpy(np.random.randn(N, 2).astype(np.float32))
        results = nmi.run(
            x_data=x,
            mode='pairwise',
            base_params=_PARAMS,
            n_workers=1,
        )
        assert results.mode == 'pairwise'
