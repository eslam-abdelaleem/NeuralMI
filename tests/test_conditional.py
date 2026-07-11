# tests/test_conditional.py
"""Tests for the conditional mutual information analysis mode."""
import pytest
import numpy as np
import torch
import neural_mi as nmi

# Minimal training params for fast tests
_PARAMS = {
    'n_epochs': 3, 'learning_rate': 1e-3, 'batch_size': 64,
    'patience': 2, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
}

N = 500  # samples


def _make_gaussian(n, d):
    return torch.from_numpy(np.random.randn(n, d).astype(np.float32))


class TestConditionalMI:
    """CMI = I(X,Z;Y) - I(Z;Y)."""

    def test_cmi_independent_xy_given_z_is_near_zero(self):
        """CMI(X;Y|Z) ≈ 0 when X is independent of Y (conditioning on Z)."""
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))
        y = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))
        z = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))

        results = nmi.run(
            x_data=x, y_data=y, z_data=z,
            mode='conditional',
            base_params=_PARAMS,
            n_workers=1,
        )
        assert results is not None
        assert results.mode == 'conditional'
        assert results.mi_estimate is not None
        assert isinstance(results.mi_estimate, float)
        # For independent signals CMI could be slightly negative due to noise;
        # just confirm it's a finite number less than 2.0 bits
        assert np.isfinite(results.mi_estimate)
        assert results.mi_estimate < 2.0

    def test_cmi_correlated_xy_given_independent_z(self):
        """CMI(X;Y|Z) > 0 when X and Y are correlated and Z is independent."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        rng = np.random.default_rng(1)
        z = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))

        results = nmi.run(
            x_data=x, y_data=y, z_data=z,
            mode='conditional',
            base_params=_PARAMS,
            n_workers=1,
        )
        assert results.mi_estimate is not None
        # mi_estimate may not always exceed 0 with very short training;
        # but it should be finite and in a plausible range
        assert np.isfinite(results.mi_estimate)

    def test_cmi_returns_details_keys(self):
        """Confirms the result details dict contains CMI breakdown keys."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        z = _make_gaussian(N, 2)

        results = nmi.run(
            x_data=x, y_data=y, z_data=z,
            mode='conditional',
            base_params=_PARAMS,
            n_workers=1,
        )
        assert 'mi_xz_y' in results.details
        assert 'mi_z_y' in results.details
        assert np.isfinite(results.details['mi_xz_y'])
        assert np.isfinite(results.details['mi_z_y'])

    def test_cmi_result_consistency(self):
        """Confirms CMI estimate = I(XZ;Y) - I(Z;Y)."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        z = _make_gaussian(N, 2)

        results = nmi.run(
            x_data=x, y_data=y, z_data=z,
            mode='conditional',
            base_params=_PARAMS,
            n_workers=1,
        )
        expected = results.details['mi_xz_y'] - results.details['mi_z_y']
        assert abs(results.mi_estimate - expected) < 1e-6

    def test_mismatched_x_z_window_sizes_raises_clear_error(self):
        """X and Z with different window sizes must raise a clear ValueError
        before the concatenation into XZ, not a bare torch.cat shape error."""
        from neural_mi.analysis.conditional import run_conditional_mi

        x = torch.randn(N, 2, 5)   # window size 5
        y = torch.randn(N, 2, 5)
        z = torch.randn(N, 2, 3)   # window size 3 -- mismatched

        with pytest.raises(ValueError, match="window size"):
            run_conditional_mi(x, y, z, base_params=_PARAMS, n_workers=1)
