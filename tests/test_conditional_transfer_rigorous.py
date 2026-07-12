"""Tests for rigorous=True mode in conditional and transfer modes (Item 3)."""
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_3d(N, C, W, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, C, W)).astype(np.float32)


from neural_mi import Model, Training, Conditional, Transfer

_MODEL = Model(embedding_dim=8, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, batch_size=64, patience=1000, learning_rate=1e-3)


# ---------------------------------------------------------------------------
# Conditional rigorous mode
# ---------------------------------------------------------------------------

class TestConditionalRigorous:
    """Tests for rigorous bias correction in conditional MI estimation."""

    def test_conditional_rigorous_returns_mi_estimate(self):
        """run() with mode='conditional' and rigorous=True should return a bias-corrected MI."""
        import neural_mi as nmi
        N = 400
        x = _make_random_3d(N, 2, 8, seed=1)
        y = _make_random_3d(N, 2, 8, seed=2)
        z = _make_random_3d(N, 2, 8, seed=3)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None, "rigorous conditional should set mi_estimate"
        assert result.dataframe is not None, "rigorous conditional should set dataframe"

    def test_conditional_rigorous_details_contain_required_keys(self):
        """Rigorous conditional result must have the standard rigorous details dict."""
        import neural_mi as nmi
        N = 400
        x = _make_random_3d(N, 2, 8, seed=4)
        y = _make_random_3d(N, 2, 8, seed=5)
        z = _make_random_3d(N, 2, 8, seed=6)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        for key in ('is_reliable', 'slope', 'mi_error',
                    'gammas_used', 'fit_quality_warning', 'leverage_warning'):
            assert key in result.details, (
                f"Missing key '{key}' in result.details; "
                f"available: {sorted(result.details.keys())}"
            )

    def test_conditional_rigorous_params_flag_set(self):
        """result.params should record rigorous=True for conditional rigorous runs."""
        import neural_mi as nmi
        N = 300
        x = _make_random_3d(N, 1, 5, seed=7)
        y = _make_random_3d(N, 1, 5, seed=8)
        z = _make_random_3d(N, 1, 5, seed=9)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.params.get('rigorous'), (
            "result.params should have rigorous=True for rigorous conditional"
        )

    def test_conditional_standard_path_unaffected(self):
        """Ensure rigorous=False (default) still works for conditional mode."""
        import neural_mi as nmi
        N = 300
        x = _make_random_3d(N, 2, 6, seed=10)
        y = _make_random_3d(N, 2, 6, seed=11)
        z = _make_random_3d(N, 2, 6, seed=12)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None
        # Standard path should NOT have rigorous keys in details
        assert 'gammas_used' not in result.details


# ---------------------------------------------------------------------------
# Transfer rigorous mode
# ---------------------------------------------------------------------------

class TestTransferRigorous:
    """Tests for rigorous bias correction in transfer entropy estimation."""

    def test_transfer_rigorous_returns_mi_estimate(self):
        """run() with mode='transfer' and rigorous=True should return a bias-corrected TE."""
        import neural_mi as nmi
        T = 500
        rng = np.random.default_rng(13)
        x = rng.normal(size=(T, 2)).astype(np.float32)
        y = rng.normal(size=(T, 2)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=5, rigorous=True,
                              gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None, "rigorous transfer should set mi_estimate"
        assert result.dataframe is not None, "rigorous transfer should set dataframe"

    def test_transfer_rigorous_details_contain_required_keys(self):
        """Rigorous transfer result must have the standard rigorous details dict."""
        import neural_mi as nmi
        T = 500
        rng = np.random.default_rng(14)
        x = rng.normal(size=(T, 1)).astype(np.float32)
        y = rng.normal(size=(T, 1)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=4, rigorous=True,
                              gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        for key in ('is_reliable', 'slope', 'fit_quality_warning', 'leverage_warning'):
            assert key in result.details, (
                f"Missing key '{key}' in result.details"
            )

    def test_transfer_standard_path_unaffected(self):
        """Ensure rigorous=False (default) still works for transfer mode."""
        import neural_mi as nmi
        T = 300
        rng = np.random.default_rng(15)
        x = rng.normal(size=(T, 1)).astype(np.float32)
        y = rng.normal(size=(T, 1)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=3),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None
        # Standard path should NOT have rigorous keys
        assert 'gammas_used' not in result.details

    def test_transfer_rigorous_params_flag_set(self):
        """result.params should record rigorous=True for rigorous transfer runs."""
        import neural_mi as nmi
        T = 400
        rng = np.random.default_rng(16)
        x = rng.normal(size=(T, 1)).astype(np.float32)
        y = rng.normal(size=(T, 1)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=4, rigorous=True,
                              gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.params.get('rigorous'), (
            "result.params should have rigorous=True for rigorous transfer"
        )
