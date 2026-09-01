# tests/test_quantities.py
"""Tests for neural_mi/quantities.py's information-quantities convenience
functions (Stage 1: unconditioned I(A;B) on offset slices, no architecture
change).

Ground truth comes from a small, self-contained shared-latent Gaussian AR(1)
oracle (X_t = a*Z_t + noise, Y_t = b*Z_t + noise, Z_t = phi*Z_{t-1} + noise),
the same construction used to validate this taxonomy during development, kept
self-contained here rather than importing the scratch harness scripts that
aren't part of the shipped package. Mutual information between any set of
offset slices of X and/or Y is exact via the standard Gaussian log-det
formula on the process's Toeplitz autocovariance.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training
from neural_mi.analysis.offsets import build_past_future, build_cross_offset


# --------------------------------------------------------------------------
# Self-contained ground-truth oracle
# --------------------------------------------------------------------------
class _SharedLatentOracle:
    """Z_t = phi*Z_{t-1} + eta_t (eta ~ N(0,1)); X_t = a*Z_t + eps_x; Y_t = b*Z_t + eps_y."""

    def __init__(self, phi=0.85, a=1.0, b=1.0, sx=0.5, sy=0.5):
        self.phi, self.a, self.b, self.sx, self.sy = phi, a, b, sx, sy

    def _cz(self, h):
        return (self.phi ** abs(h)) / (1 - self.phi ** 2)

    def _cov_entry(self, var_i, off_i, var_j, off_j):
        h = off_j - off_i
        cz = self._cz(h)
        if var_i == 'x' and var_j == 'x':
            return self.a ** 2 * cz + (self.sx ** 2 if h == 0 else 0.0)
        if var_i == 'y' and var_j == 'y':
            return self.b ** 2 * cz + (self.sy ** 2 if h == 0 else 0.0)
        return self.a * self.b * cz  # one x, one y (order doesn't matter, cz is even)

    def _cov_matrix(self, spec):
        n = len(spec)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = self._cov_entry(spec[i][0], spec[i][1], spec[j][0], spec[j][1])
        return M

    def mi_bits(self, spec_a, spec_b):
        """Exact I(A;B) in bits for two lists of (var, offset) tuples."""
        joint = self._cov_matrix(spec_a + spec_b)
        ca = self._cov_matrix(spec_a)
        cb = self._cov_matrix(spec_b)
        _, ld_j = np.linalg.slogdet(joint)
        _, ld_a = np.linalg.slogdet(ca)
        _, ld_b = np.linalg.slogdet(cb)
        return float((ld_a + ld_b - ld_j) / (2 * np.log(2)))

    def ais_exact(self, k):
        return self.mi_bits([('x', s) for s in range(-k, 0)], [('x', 0)])

    def excess_entropy_exact(self, k, future_k):
        return self.mi_bits([('x', s) for s in range(-k, 0)],
                             [('x', s) for s in range(0, future_k)])

    def cross_predictive_exact(self, past_k, future_k):
        return self.mi_bits([('x', s) for s in range(-past_k, 0)],
                             [('y', s) for s in range(0, future_k)])

    def block_mi_exact(self, w):
        return self.mi_bits([('x', s) for s in range(w)], [('y', s) for s in range(w)])

    def instantaneous_mi_exact(self):
        return self.mi_bits([('x', 0)], [('y', 0)])

    def sample(self, T, seed=0, burn=200):
        rng = np.random.default_rng(seed)
        z = 0.0
        Z = np.zeros(T + burn)
        for t in range(T + burn):
            z = self.phi * z + rng.normal()
            Z[t] = z
        Z = Z[burn:]
        x = (self.a * Z + self.sx * rng.normal(size=T)).astype(np.float32)
        y = (self.b * Z + self.sy * rng.normal(size=T)).astype(np.float32)
        return x.reshape(-1, 1), y.reshape(-1, 1)  # (T, 1) -- one channel


_MODEL = Model(embedding_dim=8, hidden_dim=32, n_layers=1)
_TRAINING = Training(n_epochs=30, learning_rate=1e-3, batch_size=128, patience=8)


# --------------------------------------------------------------------------
# Shape/plumbing correctness (fast, no accuracy claims)
# --------------------------------------------------------------------------
class TestOffsetShapes:
    def test_build_past_future_shapes(self):
        signal = torch.randn(500, 3)
        x_past, x_future = build_past_future(signal, past_len=5, future_len=2)
        assert x_past.shape == (494, 3, 5)
        assert x_future.shape == (494, 3, 2)

    def test_build_past_future_alignment(self):
        # X_future must start exactly where X_past ends.
        signal = torch.arange(20, dtype=torch.float32).reshape(20, 1)
        x_past, x_future = build_past_future(signal, past_len=3, future_len=2)
        assert torch.equal(x_past[0, 0], torch.tensor([0., 1., 2.]))
        assert torch.equal(x_future[0, 0], torch.tensor([3., 4.]))

    def test_build_past_future_raises_when_too_short(self):
        signal = torch.randn(4, 1)
        with pytest.raises(ValueError):
            build_past_future(signal, past_len=3, future_len=3)

    def test_build_cross_offset_shapes(self):
        x = torch.randn(500, 2)
        y = torch.randn(500, 2)
        x_past, y_future = build_cross_offset(x, y, past_len=4, future_len=1)
        assert x_past.shape == (496, 2, 4)
        assert y_future.shape == (496, 2, 1)


class TestConvenienceFunctionsReturnResults:
    """Scalar-parameter calls must return a plain Results, matching mode='estimate'."""

    def test_active_information_storage_returns_results(self):
        x = torch.randn(300, 1)
        r = nmi.active_information_storage(x, k=3, model=_MODEL, training=_TRAINING,
                                            show_progress=False)
        assert isinstance(r, nmi.Results)
        assert r.mi_estimate is not None

    def test_excess_entropy_returns_results(self):
        x = torch.randn(300, 1)
        r = nmi.excess_entropy(x, k=3, future_k=2, model=_MODEL, training=_TRAINING,
                                show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_instantaneous_mi_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.instantaneous_mi(x, y, model=_MODEL, training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_cross_predictive_information_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.cross_predictive_information(x, y, past_k=3, model=_MODEL,
                                              training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_block_mi_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.block_mi(x, y, window_size=4, model=_MODEL, training=_TRAINING,
                          show_progress=False)
        assert isinstance(r, nmi.Results)


class TestConditionalTransferEntropy:
    """`conditional_transfer_entropy` is exported from the package top level and
    had zero test references anywhere in the suite.

    Deliberately a contract test, not an accuracy test. Measured against
    SharedLatentGaussian (verification V5) this quantity carries an
    error-amplification factor near 100 and a seed-to-seed spread larger than
    its own value, so it is not distinguishable from zero at any sample size a
    test can afford. Asserting a value would be asserting noise. What is worth
    pinning is that it runs, returns the documented shape, and reports the
    amplification factor a caller needs in order to know not to trust it.
    """

    def test_returns_results_with_amplification(self):
        rng = np.random.default_rng(0)
        T = 800
        w = rng.standard_normal((T, 1)).astype('float32')
        x = (0.7 * w + 0.7 * rng.standard_normal((T, 1))).astype('float32')
        y = np.empty_like(x)
        y[0] = rng.standard_normal(1)
        for t in range(1, T):                      # y follows x and w with a lag
            y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + 0.3 * w[t - 1] \
                   + 0.3 * rng.standard_normal(1)
        r = nmi.conditional_transfer_entropy(
            x, y, w, history_window=2, model=_MODEL, training=_TRAINING,
            n_workers=1, show_progress=False, seed=0)

        assert isinstance(r, nmi.Results)
        assert r.mi_estimate is not None and np.isfinite(r.mi_estimate)
        amp = r.get('amplification_factor')
        assert amp is not None, "callers need the amplification factor to read this at all"
        assert np.isfinite(amp) and amp > 0

    def test_history_window_sweep_returns_results(self):
        """The iterable path, which shares the Results shape with mode='sweep'."""
        rng = np.random.default_rng(1)
        T = 600
        w = rng.standard_normal((T, 1)).astype('float32')
        x = rng.standard_normal((T, 1)).astype('float32')
        y = rng.standard_normal((T, 1)).astype('float32')
        r = nmi.conditional_transfer_entropy(
            x, y, w, history_window=[1, 2], model=_MODEL, training=_TRAINING,
            n_workers=1, show_progress=False, seed=0)
        assert isinstance(r, nmi.Results)
        assert list(r.dataframe['history_window']) == [1, 2]
        assert 'mi_mean' in r.dataframe.columns


class TestConvenienceFunctionsSweep:
    """An iterable construction parameter must dispatch a sweep and return a
    Results shaped like mode='sweep''s, so every entry point reads the same way."""

    def test_active_information_storage_sweep_returns_results(self):
        x = torch.randn(300, 1)
        r = nmi.active_information_storage(x, k=[2, 3, 4], model=_MODEL, training=_TRAINING,
                                             n_workers=2, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert r.mode == 'sweep'
        assert r.mi_estimate is None          # a sweep is a curve, not one number
        df = r.dataframe
        assert list(df['k']) == [2, 3, 4]
        assert 'mi_mean' in df.columns
        assert df['mi_mean'].notna().all()
        assert r.get('raw_results') is not None

    def test_block_mi_sweep_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.block_mi(x, y, window_size=[2, 4], model=_MODEL, training=_TRAINING,
                           n_workers=2, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert list(r.dataframe['window_size']) == [2, 4]

    def test_sweep_matches_individual_scalar_calls(self):
        """The sweep path must not silently compute something different from
        calling the scalar path once per value (same seed, same architecture)."""
        x = torch.randn(300, 1)
        seed = 123
        r = nmi.active_information_storage(x, k=[3, 5], model=_MODEL, training=_TRAINING,
                                             n_workers=1, show_progress=False, seed=seed)
        individual = [
            nmi.active_information_storage(x, k=kv, model=_MODEL, training=_TRAINING,
                                            show_progress=False, seed=seed).mi_estimate
            for kv in [3, 5]
        ]
        np.testing.assert_allclose(r.dataframe['mi_mean'].values, individual, rtol=1e-5)


# --------------------------------------------------------------------------
# Accuracy against exact Gaussian ground truth (slower, generous tolerance --
# short training budget, so this checks "in the right ballpark", not precision)
# --------------------------------------------------------------------------
class TestAccuracyAgainstOracle:
    _oracle = _SharedLatentOracle(phi=0.85, a=1.0, b=1.0, sx=0.5, sy=0.5)

    def test_active_information_storage_accuracy(self):
        k = 5
        x, _y = self._oracle.sample(4000, seed=1)
        exact = self._oracle.ais_exact(k)
        r = nmi.active_information_storage(
            torch.from_numpy(x), k=k, model=_MODEL, training=_TRAINING,
            show_progress=False, seed=0,
        )
        assert abs(r.mi_estimate - exact) < 0.5

    def test_excess_entropy_at_least_ais(self):
        """E_X >= AIS_X always (a longer future window can only reveal more)."""
        k = 5
        x, _y = self._oracle.sample(4000, seed=1)
        ais_exact = self._oracle.ais_exact(k)
        ee_exact = self._oracle.excess_entropy_exact(k, future_k=3)
        assert ee_exact >= ais_exact - 1e-9  # exact-math sanity check on the oracle itself

        r_ais = nmi.active_information_storage(
            torch.from_numpy(x), k=k, model=_MODEL, training=_TRAINING,
            show_progress=False, seed=0,
        )
        r_ee = nmi.excess_entropy(
            torch.from_numpy(x), k=k, future_k=3, model=_MODEL, training=_TRAINING,
            show_progress=False, seed=0,
        )
        assert abs(r_ais.mi_estimate - ais_exact) < 0.5
        assert abs(r_ee.mi_estimate - ee_exact) < 0.5

    def test_cross_predictive_information_accuracy(self):
        past_k = 5
        x, y = self._oracle.sample(4000, seed=2)
        exact = self._oracle.cross_predictive_exact(past_k, future_k=1)
        r = nmi.cross_predictive_information(
            torch.from_numpy(x), torch.from_numpy(y), past_k=past_k,
            model=_MODEL, training=_TRAINING, show_progress=False, seed=0,
        )
        assert abs(r.mi_estimate - exact) < 0.5

    def test_instantaneous_mi_accuracy(self):
        x, y = self._oracle.sample(4000, seed=3)
        exact = self._oracle.instantaneous_mi_exact()
        r = nmi.instantaneous_mi(
            torch.from_numpy(x), torch.from_numpy(y),
            model=_MODEL, training=_TRAINING, show_progress=False, seed=0,
        )
        assert abs(r.mi_estimate - exact) < 0.5

    def test_block_mi_accuracy(self):
        # Block MI combines info across every position in the window (a small
        # compression task, not a single past/future split), so it converges
        # more slowly than the other four quantities at this training budget
        # -- wider tolerance here, not a sign of a construction bug (shape/
        # alignment correctness is already covered by TestOffsetShapes).
        w = 3
        x, y = self._oracle.sample(4000, seed=4)
        exact = self._oracle.block_mi_exact(w)
        r = nmi.block_mi(
            torch.from_numpy(x), torch.from_numpy(y), window_size=w,
            model=_MODEL, training=_TRAINING, show_progress=False, seed=0,
        )
        assert abs(r.mi_estimate - exact) < 1.0


# --------------------------------------------------------------------------
# show_progress must reach every per-task run(), not just the outer sweep bar
# --------------------------------------------------------------------------
class TestSweepShowProgressPropagation:
    """Regression: dispatch_tasks' own show_progress only ever controlled the
    outer per-task-loop bar; the four task functions never forwarded the
    caller's show_progress into their own inner run() call, so
    show_progress=False silently failed to suppress each sweep entry's own
    training-loop progress bar. Checked here by capturing what quantities.py's
    module-level run() actually receives, not by scraping stdout."""

    def test_active_information_storage_sweep_forwards_show_progress(self, monkeypatch):
        received = []
        real_run = nmi.quantities.run

        def _spy(*args, **kwargs):
            received.append(kwargs.get('show_progress'))
            return real_run(*args, **kwargs)

        monkeypatch.setattr(nmi.quantities, 'run', _spy)
        x = torch.randn(300, 1)
        nmi.active_information_storage(x, k=[2, 3], model=_MODEL, training=_TRAINING,
                                       n_workers=1, show_progress=False)
        assert received and all(v is False for v in received)

    def test_block_mi_sweep_forwards_show_progress(self, monkeypatch):
        received = []
        real_run = nmi.quantities.run

        def _spy(*args, **kwargs):
            received.append(kwargs.get('show_progress'))
            return real_run(*args, **kwargs)

        monkeypatch.setattr(nmi.quantities, 'run', _spy)
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        nmi.block_mi(x, y, window_size=[2, 3], model=_MODEL, training=_TRAINING,
                    n_workers=1, show_progress=False)
        assert received and all(v is False for v in received)

    def test_mi_rate_sweep_forwards_show_progress(self, monkeypatch):
        received = []
        real_run = nmi.quantities.run

        def _spy(*args, **kwargs):
            received.append(kwargs.get('show_progress'))
            return real_run(*args, **kwargs)

        monkeypatch.setattr(nmi.quantities, 'run', _spy)
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        model = Model(embedding_model='dual_branch', embedding_dim=8, hidden_dim=16, n_layers=1)
        nmi.mi_rate(x, y, h=[0, 3], W=5, model=model, training=Training(n_epochs=2, patience=1),
                   n_workers=1, show_progress=False)
        assert received and all(v is False for v in received)
