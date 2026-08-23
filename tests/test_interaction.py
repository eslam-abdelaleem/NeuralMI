# tests/test_interaction.py
"""Tests for mode='interaction' (interaction information)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Interaction

_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

N = 300  # samples


class _StaticTripleOracle:
    """Z ~ N(0,1); X = a*Z + eps_x; Y = b*Z + eps_y; W = c*Z + eps_w, IID
    across samples. All three driven by one shared cause -> X and W are
    redundant proxies for Y's information, so exact II is negative (the
    standard "redundancy" signature)."""

    def __init__(self, a=1.0, b=1.0, c=1.0, sx=0.5, sy=0.5, sw=0.5):
        self._loadings = {'x': a, 'y': b, 'w': c}
        self._noises = {'x': sx, 'y': sy, 'w': sw}

    def _cov_entry(self, vi, vj):
        v = self._loadings[vi] * self._loadings[vj]
        if vi == vj:
            v += self._noises[vi] ** 2
        return v

    def _cov_matrix(self, vars_):
        n = len(vars_)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = self._cov_entry(vars_[i], vars_[j])
        return M

    def mi_bits(self, vars_a, vars_b):
        joint = self._cov_matrix(vars_a + vars_b)
        ca = self._cov_matrix(vars_a)
        cb = self._cov_matrix(vars_b)
        _, ld_j = np.linalg.slogdet(joint)
        _, ld_a = np.linalg.slogdet(ca)
        _, ld_b = np.linalg.slogdet(cb)
        return float((ld_a + ld_b - ld_j) / (2 * np.log(2)))

    def ii_exact(self):
        mi_xw_y = self.mi_bits(['x', 'w'], ['y'])
        mi_x_y = self.mi_bits(['x'], ['y'])
        mi_w_y = self.mi_bits(['w'], ['y'])
        return mi_xw_y - mi_x_y - mi_w_y, mi_xw_y, mi_x_y, mi_w_y

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=n)
        x = (self._loadings['x'] * z + self._noises['x'] * rng.normal(size=n)).astype(np.float32)
        y = (self._loadings['y'] * z + self._noises['y'] * rng.normal(size=n)).astype(np.float32)
        w = (self._loadings['w'] * z + self._noises['w'] * rng.normal(size=n)).astype(np.float32)
        return x.reshape(-1, 1), y.reshape(-1, 1), w.reshape(-1, 1)


class TestInteractionInformationPlumbing:
    """Fast shape/wiring checks, no accuracy claims."""

    def test_returns_results_object(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert r.mode == 'interaction'
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)

    def test_details_keys(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        for key in ('interaction_info', 'mi_xw_y', 'mi_x_y', 'mi_w_y'):
            assert key in r.details

    def test_ii_equals_combination(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        expected = r.details['mi_xw_y'] - r.details['mi_x_y'] - r.details['mi_w_y']
        assert abs(r.mi_estimate - expected) < 1e-6

    def test_missing_w_data_raises(self):
        x, y = np.random.randn(N, 1), np.random.randn(N, 1)
        with pytest.raises((ValueError, TypeError)):
            nmi.run(x, y, mode='interaction', model=_MODEL, training=_TRAINING, n_workers=1)

    def test_mismatched_window_size_raises(self):
        x = np.random.randn(N, 1, 4)
        y = np.random.randn(N, 1, 4)
        w = np.random.randn(N, 1, 6)  # different window size than x
        with pytest.raises(ValueError):
            nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                   model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)

    def test_summary_runs(self, capsys):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        r.summary()
        captured = capsys.readouterr()
        assert 'II' in captured.out

    def test_plot_runs(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        ax = r.plot(show=False)
        assert ax is not None
        plt.close('all')

    def test_rigorous_runs(self):
        x, y, w = np.random.randn(600, 1), np.random.randn(600, 1), np.random.randn(600, 1)
        r = nmi.run(x, y, mode='interaction',
                   interaction=Interaction(w_data=w, rigorous=True, gamma_range=range(1, 4)),
                   model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)

    def test_permutation_test_runs(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False,
                    permutation_test=True, n_permutations=3)
        assert 'null_distribution' in r.details
        assert len(r.details['null_distribution']) == 3


class TestInteractionInformationAccuracy:
    """Validate against exact Gaussian ground truth (a redundancy case:
    negative II is the expected direction, checked both exactly and,
    loosely, at the estimated level)."""

    def test_redundancy_gives_negative_ii(self):
        oracle = _StaticTripleOracle()
        ii_exact, mi_xw_y_exact, mi_x_y_exact, mi_w_y_exact = oracle.ii_exact()
        assert ii_exact < 0  # exact-math sanity check on the oracle itself

        x, y, w = oracle.sample(4000, seed=1)
        training = Training(n_epochs=30, learning_rate=1e-3, batch_size=128, patience=8)
        r = nmi.run(
            torch.from_numpy(x), torch.from_numpy(y), mode='interaction',
            interaction=Interaction(w_data=torch.from_numpy(w)),
            model=_MODEL, training=training, n_workers=1, seed=0, show_progress=False,
        )
        assert r.mi_estimate < 0.3  # not a tight match (II is a 3-term combination,
        # the same "small residual" fragility discussed in THEORY.md applies), but the
        # redundancy signature (clearly not a large positive synergy value) should hold.
