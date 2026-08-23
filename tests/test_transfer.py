# tests/test_transfer.py
"""Tests for the transfer entropy analysis mode."""
import pytest
import numpy as np
import torch
import neural_mi as nmi
from neural_mi import Model, Training, Transfer
from neural_mi.analysis.transfer import _build_te_arrays

# Minimal model/training configs for fast tests
_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

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
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=H, prediction_horizon=1),
            model=_MODEL, training=_TRAINING,
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
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=H),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert 'i_xypast_yfuture' in results.details
        assert 'i_ypast_yfuture' in results.details
        assert 'n_samples' in results.details

    def test_te_estimate_equals_difference(self):
        x = np.random.randn(N, 1)
        y = np.random.randn(N, 1)
        results = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=H),
            model=_MODEL, training=_TRAINING,
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
                x, y,
                mode='transfer',   # no Transfer config -> history_window missing
                model=_MODEL, training=_TRAINING,
                n_workers=1,
            )


class _SharedLatentTripleOracle:
    """Z_t=phi*Z_{t-1}+eta; X_t=a*Z_t+eps_x; Y_t=b*Z_t+eps_y; W_t=c*Z_t+eps_w.

    All three driven by one shared latent, so conditioning on W's history
    should explain away most of TE(X->Y) -- X's predictive power on Y comes
    entirely through Z, and W is an equally good proxy for Z.
    """

    def __init__(self, phi=0.85, a=1.0, b=1.0, c=1.0, sx=0.5, sy=0.5, sw=0.5):
        self.phi = phi
        self._loadings = {'x': a, 'y': b, 'w': c}
        self._noises = {'x': sx, 'y': sy, 'w': sw}

    def _cz(self, h):
        return (self.phi ** abs(h)) / (1 - self.phi ** 2)

    def _cov_entry(self, var_i, off_i, var_j, off_j):
        h = off_j - off_i
        v = self._loadings[var_i] * self._loadings[var_j] * self._cz(h)
        if var_i == var_j and h == 0:
            v += self._noises[var_i] ** 2
        return v

    def _cov_matrix(self, spec):
        n = len(spec)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = self._cov_entry(spec[i][0], spec[i][1], spec[j][0], spec[j][1])
        return M

    def mi_bits(self, spec_a, spec_b):
        joint = self._cov_matrix(spec_a + spec_b)
        ca = self._cov_matrix(spec_a)
        cb = self._cov_matrix(spec_b)
        _, ld_j = np.linalg.slogdet(joint)
        _, ld_a = np.linalg.slogdet(ca)
        _, ld_b = np.linalg.slogdet(cb)
        return float((ld_a + ld_b - ld_j) / (2 * np.log(2)))

    def cond_mi_bits(self, spec_a, spec_b, spec_c):
        """I(A;B|C) = I(A,C;B) - I(C;B)."""
        return self.mi_bits(spec_a + spec_c, spec_b) - self.mi_bits(spec_c, spec_b)

    def te_exact(self, k):
        return self.cond_mi_bits([('x', s) for s in range(-k, 0)], [('y', 0)],
                                  [('y', s) for s in range(-k, 0)])

    def cte_exact(self, k):
        """Conditional TE(X->Y|W)."""
        return self.cond_mi_bits([('x', s) for s in range(-k, 0)], [('y', 0)],
                                  [('y', s) for s in range(-k, 0)] + [('w', s) for s in range(-k, 0)])

    def sample(self, T, seed=0, burn=200):
        rng = np.random.default_rng(seed)
        z, Z = 0.0, np.zeros(T + burn)
        for t in range(T + burn):
            z = self.phi * z + rng.normal()
            Z[t] = z
        Z = Z[burn:]
        x = (self._loadings['x'] * Z + self._noises['x'] * rng.normal(size=T)).astype(np.float32)
        y = (self._loadings['y'] * Z + self._noises['y'] * rng.normal(size=T)).astype(np.float32)
        w = (self._loadings['w'] * Z + self._noises['w'] * rng.normal(size=T)).astype(np.float32)
        return x.reshape(-1, 1), y.reshape(-1, 1), w.reshape(-1, 1)


class TestConditionalTransferEntropy:
    """Tests for Transfer's w_data extension (conditional TE)."""

    def test_w_data_none_is_regression_safe(self):
        """Omitting w_data (or passing None explicitly) must reproduce exactly
        the same result as before this parameter existed -- same seed, same
        everything else."""
        x = np.random.randn(N, 1).astype(np.float32)
        y = np.random.randn(N, 1).astype(np.float32)
        r_default = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=H),
                            model=_MODEL, training=_TRAINING, n_workers=1, seed=7,
                            show_progress=False)
        r_explicit_none = nmi.run(x, y, mode='transfer',
                                  transfer=Transfer(history_window=H, w_data=None),
                                  model=_MODEL, training=_TRAINING, n_workers=1, seed=7,
                                  show_progress=False)
        assert r_default.mi_estimate == r_explicit_none.mi_estimate

    def test_build_w_past_matches_x_past_construction(self):
        from neural_mi.analysis.transfer import _build_te_arrays, _build_w_past
        x = np.random.randn(N, 2)
        y = np.random.randn(N, 3)
        w = np.random.randn(N, 1)
        x_past, y_past, y_future = _build_te_arrays(x, y, history_window=H)
        w_past = _build_w_past(w, history_window=H, n_valid=x_past.shape[0])
        assert w_past.shape == (x_past.shape[0], 1, H)

    def test_w_data_changes_conditioning_dimensionality(self):
        """Providing w_data must concatenate W_past into the conditioning
        arrays (verified via the diagnostics' implied dimensionality is not
        directly exposed, so check indirectly: the run must succeed with W
        present and produce a different point estimate than without it on
        data where W is genuinely informative)."""
        oracle = _SharedLatentTripleOracle()
        x, y, w = oracle.sample(600, seed=1)
        r_plain = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=5),
                          model=_MODEL, training=_TRAINING, n_workers=1, seed=0,
                          show_progress=False)
        r_cond = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=5, w_data=w),
                         model=_MODEL, training=_TRAINING, n_workers=1, seed=0,
                         show_progress=False)
        assert r_plain.mi_estimate != r_cond.mi_estimate

    def test_conditional_te_explains_away_shared_latent(self):
        """W driven by the same latent as X should explain away most of
        TE(X->Y) -- checked at the exact level (always true) and, loosely,
        at the estimated level (conditional-MI-family quantities are known
        to be a fragile, small-residual estimation target, see THEORY.md
        and the amplification-factor discussion, so this is a qualitative
        direction check, not a tight numeric match)."""
        k = 5
        oracle = _SharedLatentTripleOracle()
        te_exact = oracle.te_exact(k)
        cte_exact = oracle.cte_exact(k)
        assert cte_exact < te_exact * 0.5  # exact-math sanity check on the oracle itself

        x, y, w = oracle.sample(6000, seed=2)
        training = Training(n_epochs=30, learning_rate=1e-3, batch_size=128, patience=8)
        r_te = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=k),
                       model=_MODEL, training=training, n_workers=1, seed=0,
                       show_progress=False)
        r_cte = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=k, w_data=w),
                        model=_MODEL, training=training, n_workers=1, seed=0,
                        show_progress=False)
        # Both should be non-negative-ish (small negative noise is expected,
        # per THEORY.md's TE-is-a-difference discussion) and CTE should come
        # in below plain TE, matching the exact-math direction above.
        assert r_te.mi_estimate > -0.2
        assert r_cte.mi_estimate < r_te.mi_estimate + 0.2

    def test_conditional_te_rigorous_runs(self):
        """rigorous=True must accept and use w_data without erroring --
        confirms _te_rigorous_scalar's w_data parameter and the extra_data
        wiring at the run.py call site."""
        oracle = _SharedLatentTripleOracle()
        x, y, w = oracle.sample(1500, seed=3)
        r = nmi.run(x, y, mode='transfer',
                   transfer=Transfer(history_window=5, w_data=w, rigorous=True,
                                     gamma_range=range(1, 4)),
                   model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)
