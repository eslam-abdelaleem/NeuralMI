# tests/test_quantities.py
"""Tests for the named quantities of neural_mi/quantities.py.

Ground truth comes from a small, self-contained shared-latent Gaussian AR(1)
oracle (X_t = a*Z_t + noise, Y_t = b*Z_t + noise, Z_t = phi*Z_{t-1} + noise).
Mutual information between any set of offset slices of X and/or Y is exact via
the standard Gaussian log-det formula on the process's Toeplitz
autocovariance.
"""
import numpy as np
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

    def predictive_information_exact(self, k):
        return self.mi_bits([('x', s) for s in range(-k, 0)],
                             [('x', s) for s in range(0, k)])

    def cross_predictive_exact(self, k):
        return self.mi_bits([('x', s) for s in range(-k, 0)],
                             [('y', s) for s in range(0, k)])

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

    def test_predictive_information_returns_results(self):
        x = torch.randn(300, 1)
        r = nmi.predictive_information(x, k=3, model=_MODEL, training=_TRAINING,
                                show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_instantaneous_mi_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.instantaneous_mi(x, y, model=_MODEL, training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_cross_predictive_information_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.cross_predictive_information(x, y, k=3, model=_MODEL,
                                              training=_TRAINING, show_progress=False)
        assert isinstance(r, nmi.Results)

    def test_block_mi_returns_results(self):
        x, y = torch.randn(300, 1), torch.randn(300, 1)
        r = nmi.block_mi(x, y, window_size=4, model=_MODEL, training=_TRAINING,
                          show_progress=False)
        assert isinstance(r, nmi.Results)


class TestTransferEntropy:
    """`transfer_entropy` is the plain quantity beside its conditional variant.

        Contract tests, for the reason the conditional class below gives: on real
        recordings TE carries a seed-to-seed spread larger than its own mean, so
        asserting a value would assert noise. What is pinned is that the wrapper is
        exactly the `mode='transfer'` call it claims to be, and that its sweep path
        returns the documented shape.
    """

    @staticmethod
    def _lagged_pair(T=800, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((T, 2)).astype('float32')
        y = np.zeros_like(x)
        y[1:] = 0.6 * x[:-1] + 0.4 * rng.standard_normal((T - 1, 2))
        return x, y.astype('float32')

    def test_matches_run_mode_transfer_exactly(self):
        from neural_mi import Transfer
        x, y = self._lagged_pair()
        kw = dict(model=_MODEL, training=_TRAINING, n_workers=1,
                  show_progress=False, seed=0)
        via_wrapper = nmi.transfer_entropy(x, y, history_window=3, **kw)
        via_run = nmi.run(x, y, mode='transfer',
                          transfer=Transfer(history_window=3), **kw)
        assert via_wrapper.mi_estimate == via_run.mi_estimate
        assert via_wrapper.get('amplification_factor') is not None

    def test_history_window_sweep_returns_sweep_shaped_results(self):
        x, y = self._lagged_pair()
        r = nmi.transfer_entropy(x, y, history_window=[2, 3], model=_MODEL,
                                 training=_TRAINING, n_workers=1,
                                 show_progress=False, seed=0)
        assert isinstance(r, nmi.Results)
        assert list(r.dataframe['history_window']) == [2, 3]
        assert 'mi_mean' in r.dataframe.columns


class TestConditionalTransferEntropy:
    """`conditional_transfer_entropy`, exported from the package top level.

        A contract test, not an accuracy test. Against SharedLatentGaussian this
        quantity carries an amplification factor near 100 and a seed-to-seed
        spread larger than its own value, so it is not distinguishable from zero at
        any sample size a test can afford. What is pinned is that it runs, returns
        the documented shape, and reports the amplification factor a caller needs
        to know not to trust it.
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

    @pytest.mark.slow
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
        assert len(r.runs) == 3

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

    def test_predictive_information_at_least_ais(self):
        """E_X >= AIS_X always (a longer future window can only reveal more)."""
        k = 5
        x, _y = self._oracle.sample(4000, seed=1)
        ais_exact = self._oracle.ais_exact(k)
        ee_exact = self._oracle.predictive_information_exact(k)
        assert ee_exact >= ais_exact - 1e-9  # exact-math sanity check on the oracle itself

        r_ais = nmi.active_information_storage(
            torch.from_numpy(x), k=k, model=_MODEL, training=_TRAINING,
            show_progress=False, seed=0,
        )
        r_ee = nmi.predictive_information(
            torch.from_numpy(x), k=k, model=_MODEL, training=_TRAINING,
            show_progress=False, seed=0,
        )
        assert abs(r_ais.mi_estimate - ais_exact) < 0.5
        assert abs(r_ee.mi_estimate - ee_exact) < 0.5

    def test_cross_predictive_information_accuracy(self):
        k = 5
        x, y = self._oracle.sample(4000, seed=2)
        exact = self._oracle.cross_predictive_exact(k)
        r = nmi.cross_predictive_information(
            torch.from_numpy(x), torch.from_numpy(y), k=k,
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
    """show_progress reaches every per-value run() call of a swept quantity,
        not just the outer progress bar. Checked by capturing what quantities.py's
        module-level run() receives.
    """

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
        nmi.mi_rate(x, y, h=[0, 3], half_width=5, model=model, training=Training(n_epochs=2, patience=1),
                   n_workers=1, show_progress=False)
        assert received and all(v is False for v in received)


class TestIterableAndGridCompose:
    """A quantity's own iterable parameter is a grid key like any other, so it
    composes with sweep_grid: every value runs every configuration and repeat
    of the grid, and the result has one dataframe row per combination."""

    @staticmethod
    def _data():
        from neural_mi.generators import SharedLatentGaussian
        oracle = SharedLatentGaussian(dims={'x': 4, 'y': 4, 'w': 4}, d=2,
                                      phi=0.9, coupling=0.4, noise=1.0, seed=0)
        s = oracle.sample(T=1200, seed=0)
        return tuple(s[k].astype('float32') for k in ('x', 'y', 'w'))

    @staticmethod
    def _cheap():
        return dict(model=nmi.Model(embedding_dim=4, hidden_dim=32),
                    training=Training(n_epochs=2, batch_size=256),
                    show_progress=False, seed=0)

    @pytest.mark.parametrize("quantity", [
        'active_information_storage', 'predictive_information',
        'cross_predictive_information', 'block_mi', 'transfer_entropy',
    ])
    def test_an_iterable_and_repeats_give_one_row_per_value(self, quantity):
        x, y, _ = self._data()
        calls = {
            'active_information_storage': lambda: nmi.active_information_storage(
                x, k=[2, 3], sweep_grid={'run_id': [0, 1]}, **self._cheap()),
            'predictive_information': lambda: nmi.predictive_information(
                x, k=[2, 3], sweep_grid={'run_id': [0, 1]}, **self._cheap()),
            'cross_predictive_information': lambda: nmi.cross_predictive_information(
                x, y, k=[2, 3], sweep_grid={'run_id': [0, 1]}, **self._cheap()),
            'block_mi': lambda: nmi.block_mi(
                x, y, window_size=[2, 3], sweep_grid={'run_id': [0, 1]}, **self._cheap()),
            'transfer_entropy': lambda: nmi.transfer_entropy(
                x, y, history_window=[2, 3], sweep_grid={'run_id': [0, 1]}, **self._cheap()),
        }
        param = {'block_mi': 'window_size', 'transfer_entropy': 'history_window'}.get(quantity, 'k')
        result = calls[quantity]()
        assert list(result.dataframe[param]) == [2, 3]
        assert list(result.dataframe['n_runs']) == [2, 2]
        assert result.mi_estimate is None
        assert result.params['config_keys'] == [param]

    def test_the_parameter_inside_sweep_grid_is_refused(self):
        x, _, _ = self._data()
        with pytest.raises(ValueError, match="leave 'k' out of sweep_grid"):
            nmi.active_information_storage(x, k=[2, 3], sweep_grid={'k': [4]}, **self._cheap())

    @pytest.mark.parametrize("quantity", ['transfer_entropy', 'interaction_information'])
    def test_repeats_of_a_difference_quantity_are_kept_per_repeat(self, quantity):
        x, y, w = self._data()
        calls = {
            'transfer_entropy': lambda: nmi.transfer_entropy(
                x, y, history_window=2, sweep_grid={'run_id': [0, 1, 2]}, **self._cheap()),
            'interaction_information': lambda: nmi.interaction_information(
                x, y, w, sweep_grid={'run_id': [0, 1, 2]}, **self._cheap()),
        }
        result = calls[quantity]()
        assert len(result.runs) == 3
        assert result.mi_estimate == pytest.approx(result.runs['mi'].mean())

    def test_a_scalar_parameter_with_repeats_is_one_row(self):
        x, y, _ = self._data()
        repeats = nmi.block_mi(x, y, window_size=3, sweep_grid={'run_id': [0, 1]}, **self._cheap())
        assert len(repeats.dataframe) == 1 and repeats.dataframe['n_runs'].iloc[0] == 2
        assert len(repeats.runs) == 2
        assert 'train_ceiling_mi' in repeats.runs.columns
        assert repeats.params['window_size'] == 3

    def test_rigorous_runs_on_a_single_mi_quantity(self):
        x, _, _ = self._data()
        result = nmi.active_information_storage(
            x, k=2, rigorous=nmi.Rigorous(gamma_range=range(1, 3), min_gamma_points=2),
            **self._cheap())
        assert result.mode == 'rigorous'
        assert 'trainings' in result.details[0]


class TestConditionalQuantitiesReportTheirSpread:
    """A quantity built by subtraction now says how far it moves between runs.

    The components were averaged and then subtracted, which gives a better
    point estimate and no variance. Taking the same combination run by run
    leaves that estimate exactly where it was, because ``mean(a) - mean(b)``
    and ``mean(a - b)`` are the same number for equal-length lists, and it
    makes the spread available. That spread is what says whether a difference
    is resolved at all.
    """

    @staticmethod
    def _data():
        from neural_mi.generators import SharedLatentGaussian
        oracle = SharedLatentGaussian(dims={'x': 4, 'y': 4, 'w': 4}, d=2,
                                      phi=0.9, coupling=0.4, noise=1.0, seed=0)
        s = oracle.sample(T=1500, seed=0)
        return tuple(s[k].astype('float32') for k in ('x', 'y', 'w'))

    @staticmethod
    def _cheap():
        return dict(model=nmi.Model(embedding_dim=4, hidden_dim=32),
                    training=Training(n_epochs=3, batch_size=256),
                    show_progress=False, seed=0)

    def test_combined_spread_matches_the_paired_difference(self):
        from neural_mi.analysis.sweep import combined_spread
        joint = [1.0, 2.0, 3.0]
        marginal = [0.5, 0.5, 0.5]
        expected = float(np.std([0.5, 1.5, 2.5], ddof=1))
        assert combined_spread((joint, marginal), (1, -1)) == pytest.approx(expected)

    def test_combined_spread_handles_three_terms(self):
        from neural_mi.analysis.sweep import combined_spread
        xw, x, w = [3.0, 4.0], [1.0, 1.0], [1.0, 2.0]
        expected = float(np.std([1.0, 1.0], ddof=1))
        assert combined_spread((xw, x, w), (1, -1, -1)) == pytest.approx(expected)

    @pytest.mark.parametrize("lists", [
        (([1.0], [2.0]),),                      # one run, nothing to spread
        (([1.0, 2.0], [3.0]),),                 # a component lost a run
    ])
    def test_combined_spread_is_none_when_there_is_no_spread_to_report(self, lists):
        from neural_mi.analysis.sweep import combined_spread
        assert combined_spread(lists[0], (1, -1)) is None

    def test_the_point_estimate_is_untouched_by_the_pairing(self):
        """mean(a) - mean(b) == mean(a - b), so no existing number moves."""
        from neural_mi.analysis.sweep import combined_spread
        rng = np.random.default_rng(0)
        a, b = list(rng.normal(size=7)), list(rng.normal(size=7))
        assert np.mean(a) - np.mean(b) == pytest.approx(
            np.mean([ai - bi for ai, bi in zip(a, b)]))
        assert combined_spread((a, b), (1, -1)) is not None

    @pytest.mark.parametrize("quantity", [
        'transfer_entropy', 'conditional_transfer_entropy', 'interaction_information',
    ])
    def test_every_difference_quantity_reports_the_spread_of_its_repeats(self, quantity):
        x, y, w = self._data()
        calls = {
            'transfer_entropy': lambda: nmi.transfer_entropy(
                x, y, history_window=3, sweep_grid={'run_id': [0, 1, 2]},
                n_workers=1, **self._cheap()),
            'conditional_transfer_entropy': lambda: nmi.conditional_transfer_entropy(
                x, y, w, history_window=3, sweep_grid={'run_id': [0, 1, 2]},
                n_workers=1, **self._cheap()),
            'interaction_information': lambda: nmi.interaction_information(
                x, y, w, sweep_grid={'run_id': [0, 1, 2]},
                n_workers=1, **self._cheap()),
        }
        result = calls[quantity]()
        spread = result.get('mi_std')
        # The spread of the per-repeat differences, from the runs table, over the
        # repeats that produced a value.
        produced = result.runs['mi'][result.runs['mi'] != 0]
        if len(produced) >= 2:
            assert spread == pytest.approx(produced.std(ddof=1))
        else:
            assert np.isnan(spread)

    def test_a_single_run_reports_no_spread_rather_than_zero(self):
        x, y, _ = self._data()
        result = nmi.transfer_entropy(x, y, history_window=3, **self._cheap())
        assert np.isnan(result.get('mi_std'))


class TestOffsetQuantitiesAcceptAGrid:
    """``processing=`` builds the grid these quantities index.

        Given a Processing, the streams go onto one grid first, so offsets index
        rows that are uniformly spaced and mean the same instant in every stream.
        That is what lets them take spike times at all.
    """

    @staticmethod
    def _spiking_driven_by_position(seed=0, duration=600.0, bin_size=0.05):
        rng = np.random.default_rng(seed)
        t = np.arange(0, duration, bin_size)
        pos = np.sin(2 * np.pi * t / 40)[:, None].astype(np.float32)
        rate = 40.0 * np.clip(pos[:, 0], 0, None) + 2.0
        trains = []
        for _ in range(8):
            counts = rng.poisson(rate * bin_size)
            trains.append(np.sort(np.concatenate(
                [t[i] + rng.uniform(0, bin_size, n) for i, n in enumerate(counts) if n])))
        return t, pos, trains, bin_size

    @staticmethod
    def _cfg():
        return dict(model=Model(embedding_dim=8, hidden_dim=128, n_layers=2),
                    training=Training(n_epochs=40, batch_size=256, patience=15),
                    show_progress=False, seed=0)

    def _processing(self, t, bin_size):
        return nmi.Processing(
            x='spike', y='continuous', y_time=t,
            x_params={'window_size': bin_size, 'bin_size': bin_size,
                      'drop_empty_windows': False},
            y_params={'window_size': bin_size, 'sample_rate': 1 / bin_size})

    @pytest.mark.slow
    def test_spike_input_now_works_and_finds_real_structure(self):
        t, pos, spikes, bin_size = self._spiking_driven_by_position()
        proc = self._processing(t, bin_size)
        driven = nmi.cross_predictive_information(
            spikes, pos, k=10, processing=proc, **self._cfg())
        assert driven.mi_estimate > 0.5

    def test_and_reports_nothing_when_there_is_nothing(self):
        t, pos, spikes, bin_size = self._spiking_driven_by_position()
        rng = np.random.default_rng(1)
        randomised = [np.sort(rng.uniform(0, 600.0, len(s))) for s in spikes]
        flat = nmi.cross_predictive_information(
            randomised, pos, k=10,
            processing=self._processing(t, bin_size), **self._cfg())
        assert abs(flat.mi_estimate) < 0.2

    def test_a_raw_array_is_unchanged_by_the_new_route(self):
        """No Processing means slice the array directly, exactly as before."""
        from neural_mi.analysis.offsets import build_past_future
        ramp = torch.arange(200, dtype=torch.float32).reshape(200, 1)
        past, present = build_past_future(ramp, past_len=5, future_len=1)
        assert past.shape == (195, 1, 5)
        assert past[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert float(present[0, 0, 0]) == 5.0

    def test_transfer_entropy_refuses_a_grid_it_cannot_index(self):
        """A categorical stream spends its trailing axis on categories, which
        mode='transfer' cannot flatten without changing what a lag means."""
        t, pos, spikes, bin_size = self._spiking_driven_by_position()
        labels = (pos[:, 0] > 0).astype(np.int64)[:, None]
        with pytest.raises(ValueError, match="one time step"):
            nmi.transfer_entropy(
                spikes, labels, history_window=8,
                processing=nmi.Processing(
                    x='spike', y='categorical', y_time=t,
                    x_params={'window_size': bin_size, 'bin_size': bin_size,
                              'drop_empty_windows': False},
                    y_params={'window_size': bin_size, 'sample_rate': 1 / bin_size}),
                **self._cfg())


class TestStride:
    """``stride`` thins the rows without changing the quantity.

    ``k``, ``h``, ``W`` and ``history_window`` define the random variable;
    ``stride`` decides how densely that variable is sampled out of one
    recording. The estimate should survive a change of stride, the row count
    should not.
    """

    _oracle = _SharedLatentOracle(phi=0.85, a=1.0, b=1.0, sx=0.5, sy=0.5)

    @staticmethod
    def _ramp(T=40, C=1):
        """signal[t, 0] == t, so a row's value reveals the time it came from."""
        return torch.arange(T * C, dtype=torch.float32).reshape(T, C) / C

    # ---- stride 1 is exactly the construction that came before -------------

    def test_stride_one_matches_explicit_slicing(self):
        sig = self._ramp(30, 2)
        past, fut = build_past_future(sig, past_len=4, future_len=2, stride=1)
        expected_past = torch.stack([sig[i:i + 4].T for i in range(25)])
        expected_fut = torch.stack([sig[i + 4:i + 6].T for i in range(25)])
        assert torch.equal(past, expected_past)
        assert torch.equal(fut, expected_fut)

    # ---- row counts --------------------------------------------------------

    @pytest.mark.parametrize("stride,expected", [(1, 25), (2, 13), (3, 9), (4, 7)])
    def test_row_count(self, stride, expected):
        """T=30, past 4, future 2 leaves 25 positions; stride keeps every n-th."""
        past, fut = build_past_future(self._ramp(30), past_len=4, future_len=2,
                                      stride=stride)
        assert past.shape[0] == expected
        assert fut.shape[0] == expected

    # ---- alignment, the failure a shape check cannot catch -----------------

    @pytest.mark.parametrize("stride", [1, 2, 3, 5])
    def test_past_and_future_stay_adjacent(self, stride):
        """X_future[i] must start exactly where X_past[i] ends, at any stride."""
        past, fut = build_past_future(self._ramp(40), past_len=4, future_len=2,
                                      stride=stride)
        for i in range(past.shape[0]):
            assert float(past[i, 0, 0]) == i * stride
            assert float(fut[i, 0, 0]) == i * stride + 4

    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_dual_branch_builders_stay_aligned(self, stride):
        """The three arrays of each dual-branch quantity share their time base.

        These mix ``unfold`` output with a direct slice, so a stride applied to
        one and not the other misaligns A against B while leaving every shape
        correct and raising nothing.
        """
        from neural_mi.quantities import (_build_mi_rate_arrays,
                                          _build_inst_exchange_arrays,
                                          _build_dir_info_rate_arrays)
        x = self._ramp(60)
        y = x + 1000.0

        x_all, y0, y_past = _build_mi_rate_arrays(x, y, h=3, half_width=2, stride=stride)
        for i in range(y0.shape[0]):
            centre = float(y0[i, 0, 0]) - 1000.0
            assert float(x_all[i, 0, 2]) == centre       # centre of the 2 * half_width + 1 window
            assert float(y_past[i, 0, -1]) == centre - 1 + 1000.0

        x0, yf, _c = _build_inst_exchange_arrays(x, y, k=3, stride=stride)
        for i in range(x0.shape[0]):
            assert float(x0[i, 0, 0]) == float(yf[i, 0, 0]) - 1000.0

        a, yf2, _yp = _build_dir_info_rate_arrays(x, y, k=3, stride=stride)
        for i in range(a.shape[0]):
            assert float(a[i, 0, -1]) == float(yf2[i, 0, 0]) - 1000.0

    @pytest.mark.slow
    def test_estimate_survives_a_change_of_stride(self):
        """Same quantity, half the rows, same answer within estimator noise."""
        x, _y = self._oracle.sample(8000, seed=1)
        exact = self._oracle.ais_exact(5)
        got = [nmi.active_information_storage(
                   torch.from_numpy(x), k=5, stride=st, model=_MODEL,
                   training=_TRAINING, show_progress=False, seed=0).mi_estimate
               for st in (1, 2)]
        for value in got:
            assert abs(value - exact) < 0.5
        assert abs(got[0] - got[1]) < 0.5

    # ---- the split check has to see the real stride -----------------------

    def test_leak_check_step_carries_the_stride(self):
        """At stride 1 a one-window gap buys one sample of separation, so a
        wrong value here waves through a split that shares almost a whole
        window."""
        import neural_mi.analysis.transfer as transfer_mod
        x, y = self._oracle.sample(1200, seed=4)
        seen = {}
        real = transfer_mod._joint_marginal_difference

        def spy(a, b, c, d, base_params, *args, **kwargs):
            seen.setdefault('step', base_params.get('leak_check_step'))
            seen.setdefault('rows', a.shape[0])
            return real(a, b, c, d, base_params, *args, **kwargs)

        transfer_mod._joint_marginal_difference = spy
        try:
            nmi.transfer_entropy(torch.from_numpy(x), torch.from_numpy(y),
                                 history_window=5, stride=4, model=_MODEL,
                                 training=_TRAINING, show_progress=False, seed=0)
        finally:
            transfer_mod._joint_marginal_difference = real
        assert seen['step'] == 4
        assert seen['rows'] == (1200 - 5 - 1) // 4 + 1

    # ---- validation --------------------------------------------------------

    @pytest.mark.parametrize("bad", [0.5, 0, -1, 'two', True])
    def test_rejects_a_stride_that_is_not_a_whole_count(self, bad):
        x, _y = self._oracle.sample(400, seed=5)
        with pytest.raises(ValueError, match="stride must be"):
            nmi.active_information_storage(torch.from_numpy(x), k=3, stride=bad,
                                           model=_MODEL, training=_TRAINING,
                                           show_progress=False, seed=0)

    def test_fractional_stride_names_the_difference_from_step_size(self):
        """The message has to explain why 0.5 works for block_mi and not here."""
        x, _y = self._oracle.sample(400, seed=5)
        with pytest.raises(ValueError, match="no fractional reading"):
            nmi.active_information_storage(torch.from_numpy(x), k=3, stride=0.5,
                                           model=_MODEL, training=_TRAINING,
                                           show_progress=False, seed=0)
