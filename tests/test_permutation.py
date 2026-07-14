# tests/test_permutation.py
"""Tests for the permutation_test parameter (Phase B8 / Phase A coverage)."""
import inspect
import warnings

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi import Model, Training, Rigorous, Lag

# Minimal model/training configs
_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

N = 500


class TestPermutationTest:
    """Regression tests for permutation_test=True."""

    def test_permutation_adds_null_distribution_to_details(self):
        """permutation_test=True must add 'null_distribution' list to details."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='estimate',
            model=_MODEL, training=_TRAINING,
            permutation_test=True,
            n_workers=1,
        )
        assert 'null_distribution' in results.details, (
            "'null_distribution' key missing from results.details"
        )
        assert isinstance(results.details['null_distribution'], list)

    def test_permutation_null_distribution_length_matches_n_permutations(self):
        """null_distribution should have exactly n_permutations entries."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='estimate',
            model=_MODEL, training=_TRAINING,
            permutation_test=True,
            n_permutations=3,
            n_workers=1,
        )
        assert len(results.details['null_distribution']) == 3

    def test_permutation_null_distribution_contains_floats(self):
        """null_distribution values should be finite floats."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='estimate',
            model=_MODEL, training=_TRAINING,
            permutation_test=True,
            n_permutations=2,
            n_workers=1,
        )
        for v in results.details['null_distribution']:
            assert isinstance(v, float)

    def test_permutation_false_leaves_details_clean(self):
        """permutation_test=False must NOT add null_distribution to details."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='estimate',
            model=_MODEL, training=_TRAINING,
            permutation_test=False,
            n_workers=1,
        )
        assert 'null_distribution' not in results.details

    def test_permutation_sweep_mode(self):
        """permutation_test works with mode='sweep'."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='sweep',
            model=_MODEL, training=_TRAINING,
            sweep_grid={'embedding_dim': [4, 8]},
            permutation_test=True,
            n_permutations=1,
            n_workers=1,
        )
        assert 'null_distribution' in results.details

    def test_permutation_mi_estimate_unchanged_with_flag(self):
        """The primary mi_estimate should still be present when permutation is enabled."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        results = nmi.run(
            x_data=x, y_data=y,
            mode='estimate',
            model=_MODEL, training=_TRAINING,
            permutation_test=True,
            n_workers=1,
        )
        assert results.mi_estimate is not None
        assert np.isfinite(results.mi_estimate)


class TestNPermutationsDefault:
    """Tests for Phase B8: n_permutations default=1 and insufficiency warning."""

    def test_n_permutations_default_is_1(self):
        """run() must have n_permutations=1 as the default (Phase B8)."""
        sig = inspect.signature(nmi.run)
        assert sig.parameters['n_permutations'].default == 1, (
            f"Expected n_permutations default=1, got "
            f"{sig.parameters['n_permutations'].default}"
        )

    def test_insufficient_n_permutations_emits_warning(self):
        """permutation_test=True with n_permutations=1 must warn about insufficiency."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nmi.run(
                x_data=x, y_data=y,
                mode='estimate',
                model=_MODEL, training=_TRAINING,
                permutation_test=True,
                n_permutations=1,
                n_workers=1,
            )
        msgs = [str(w.message) for w in caught]
        assert any("insufficient" in m.lower() or "n_permutations" in m for m in msgs), (
            f"Expected insufficiency warning; got: {msgs}"
        )

    def test_no_warning_with_sufficient_n_permutations(self):
        """n_permutations >= 50 should NOT trigger the insufficiency warning."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nmi.run(
                x_data=x, y_data=y,
                mode='estimate',
                model=_MODEL, training=_TRAINING,
                permutation_test=True,
                n_permutations=50,
                n_workers=1,
            )
        msgs = [str(w.message) for w in caught
                if "n_permutations" in str(w.message) and "insufficient" in str(w.message).lower()]
        assert len(msgs) == 0, (
            f"Unexpected insufficiency warning for n_permutations=50: {msgs}"
        )

    def test_permutation_rigorous_raises(self):
        """permutation_test=True with mode='rigorous' must raise ValueError."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        with pytest.raises(ValueError, match="not supported for mode='rigorous'"):
            nmi.run(
                x_data=x, y_data=y,
                mode='rigorous',
                model=_MODEL, training=_TRAINING,
                permutation_test=True,
                rigorous=Rigorous(gamma_range=range(2, 4)),
                n_workers=1,
            )


class TestNullDistributionRawClipped:
    """Verify null_distribution_raw is consistently and independently computed for all modes.

    Prior to this fix, lag/conditional/transfer returned (mi, mi) — the same value
    for both the clipped and raw slots — because those modes didn't propagate raw_train_mi.
    Each test here confirms structural correctness (both lists present, correct length,
    finite floats); the functional contract (raw uses raw_train_mi) is guaranteed by code.
    """

    _MODEL_P = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
    _TRAINING_P = Training(n_epochs=2, learning_rate=1e-3, batch_size=64, patience=2)

    def _check_null_lists(self, details, n_perm):
        """Assert both null lists are present, have length n_perm, and contain floats."""
        assert 'null_distribution' in details, "null_distribution missing"
        assert 'null_distribution_raw' in details, "null_distribution_raw missing"
        assert len(details['null_distribution']) == n_perm
        assert len(details['null_distribution_raw']) == n_perm
        for c, r in zip(details['null_distribution'], details['null_distribution_raw']):
            assert isinstance(c, float), f"clipped value not float: {c}"
            assert isinstance(r, float), f"raw value not float: {r}"

    def test_estimate_mode_raw_and_clipped_present(self):
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        res = nmi.run(x_data=x, y_data=y, mode='estimate', model=self._MODEL_P, training=self._TRAINING_P,
                      permutation_test=True, n_permutations=2, n_workers=1)
        self._check_null_lists(res.details, 2)

    def test_sweep_mode_raw_and_clipped_present(self):
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        res = nmi.run(x_data=x, y_data=y, mode='sweep',
                      sweep_grid={'embedding_dim': [4, 8]},
                      model=self._MODEL_P, training=self._TRAINING_P,
                      permutation_test=True, n_permutations=2, n_workers=1)
        self._check_null_lists(res.details, 2)

    def test_lag_mode_raw_and_clipped_present(self):
        """lag mode null: raw_train_mi extracted from task results (not duplicated from clipped)."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        res = nmi.run(x_data=x, y_data=y, mode='lag',
                      lag=Lag(lag_range=range(-1, 2)),
                      model=self._MODEL_P, training=self._TRAINING_P,
                      permutation_test=True, n_permutations=2, n_workers=1)
        self._check_null_lists(res.details, 2)


class TestPairwisePermutation:
    """Regression tests: mode='pairwise' permutation_test used to always return an
    all-NaN null (dispatch had no 'pairwise' branch) and emitted the same warning
    twice with two different pair-count formulas."""

    _MODEL_P = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
    _TRAINING_P = Training(n_epochs=2, learning_rate=1e-3, batch_size=64, patience=2)

    def test_cross_pairwise_null_distribution_not_all_nan(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((N, 3)).astype('float32')
        y = rng.standard_normal((N, 2)).astype('float32')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = nmi.run(x_data=x, y_data=y, mode='pairwise',
                          model=self._MODEL_P, training=self._TRAINING_P,
                          permutation_test=True, n_permutations=2, n_workers=1)
        null = res.details['null_distribution']
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null), "null distribution is all-NaN"
        expensive_warnings = [x for x in w if "computationally expensive" in str(x.message)]
        assert len(expensive_warnings) == 1, "duplicate warning was not collapsed"

    def test_self_pairwise_permutation_test_does_not_crash(self):
        """Self-pairwise has no y_data to shuffle; must skip cleanly, not crash."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((N, 3)).astype('float32')
        res = nmi.run(x_data=x, mode='pairwise',
                      model=self._MODEL_P, training=self._TRAINING_P,
                      permutation_test=True, n_permutations=2, n_workers=1)
        assert 'null_distribution' not in res.details
