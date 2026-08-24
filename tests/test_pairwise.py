# tests/test_pairwise.py
"""Tests for the pairwise MI matrix analysis mode."""
import numpy as np
import pandas as pd
import torch
import neural_mi as nmi
from neural_mi import Model, Training

# Minimal model/training configs
_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

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
            model=_MODEL, training=_TRAINING,
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
            model=_MODEL, training=_TRAINING,
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
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert len(results.dataframe) == N_CHX * N_CHY

    def test_pairwise_all_estimates_finite(self):
        """Every MI mean in the pairwise matrix should be finite."""
        x = torch.from_numpy(np.random.randn(N, 3).astype(np.float32))
        results = nmi.run(
            x_data=x,
            mode='pairwise',
            model=_MODEL, training=_TRAINING,
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
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert results.mode == 'pairwise'

    def test_pairwise_deferred_windowing_preserves_channel_identity(self):
        """Reachability for shift_windows/shift_time defers windowing to each
        pair's own dispatch (raw per-channel slices, not a pre-windowed
        array). A slicing bug in that path could silently swap or misalign
        channels; a real, channel-selective correlation structure would catch
        it, unlike the finite-value-only checks above."""
        np.random.seed(0)
        torch.manual_seed(0)
        T = 4000
        shared = np.random.randn(T, 1).astype('float32')
        ch0 = shared + 0.05 * np.random.randn(T, 1).astype('float32')
        ch1 = shared + 0.05 * np.random.randn(T, 1).astype('float32')
        ch2 = np.random.randn(T, 1).astype('float32')  # independent of ch0/ch1
        x = np.concatenate([ch0, ch1, ch2], axis=1)

        proc = nmi.Processing(x='continuous', x_params={'window_size': 10, 'step_size': 10})
        training = Training(n_epochs=15, patience=5, batch_size=64, learning_rate=1e-3)
        results = nmi.run(x, mode='pairwise', processing=proc,
                          model=_MODEL, training=training,
                          n_workers=1, show_progress=False, seed=0)

        df = results.dataframe.set_index(['ch_x', 'ch_y'])
        mi_01 = df.loc[(0, 1), 'mi_mean']
        mi_02 = df.loc[(0, 2), 'mi_mean']
        mi_12 = df.loc[(1, 2), 'mi_mean']
        assert mi_01 > mi_02 and mi_01 > mi_12, (
            f"Correlated pair (0,1)={mi_01} should exceed independent pairs "
            f"(0,2)={mi_02}, (1,2)={mi_12} -- a channel-slicing bug in the "
            f"deferred windowing path would break this."
        )
