# tests/test_ksg.py
"""The KSG estimator and its conditional form (neural_mi/estimators/ksg.py), alone and through run()."""
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

import neural_mi as nmi
from neural_mi.estimators import ksg as K
from neural_mi.exceptions import CombinationWarning
from neural_mi.estimators import ksg_cmi, ksg_mi
from neural_mi.generators import LatentGaussian

LN2 = np.log(2)
QUIET = dict(show_progress=False, verbose=False)


def _pair(n=2000, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return x, rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)


def _slow(n, phi, rng):
    return lfilter([1], [1, -phi], rng.standard_normal(n + 1000))[1000:]


class TestEstimator:
    def test_matches_scikit_learn(self):
        """Same algorithm as mutual_info_regression when neither gaussianizes nor excludes rows."""
        sklearn = pytest.importorskip('sklearn.feature_selection')
        x, y = _pair()
        theirs = sklearn.mutual_info_regression(x[:, None], y, n_neighbors=4, random_state=0)[0]
        assert ksg_mi(x, y, gaussianize=False, exclude=0) == pytest.approx(theirs, abs=1e-6)

    def test_recovers_gaussian_mi(self):
        x, y = _pair()
        assert ksg_mi(x, y) == pytest.approx(-0.5 * np.log(1 - 0.36), abs=0.03)
        assert abs(ksg_mi(x, _pair(seed=1)[1])) < 0.02

    def test_gaussianizing_ignores_monotone_transforms(self):
        x, y = _pair()
        assert ksg_mi(x, y) == ksg_mi(np.exp(x), np.arcsinh(3 * y))

    def test_conditional_recovers_gaussian_cmi(self):
        g = LatentGaussian.shared(dims={'x': 2, 'y': 2, 'z': 2}, d=2, phi=0.0, noise=0.6, seed=0)
        s = g.sample(5000, seed=0)
        exact = g.exact([('x', 0)], [('y', 0)], [('z', 0)])
        assert ksg_cmi(s['x'], s['y'], s['z']) / LN2 == pytest.approx(exact, abs=0.03)

    @pytest.mark.parametrize('exclude', [0, 7])
    def test_brute_force_agrees_with_the_tree(self, monkeypatch, exclude):
        rng = np.random.default_rng(0)
        x, y, z = (rng.standard_normal((800, 2)) for _ in range(3))
        y += x
        tree = ksg_mi(x, y, exclude=exclude), ksg_cmi(x, y, z, exclude=exclude)
        monkeypatch.setattr(K, 'KD_TREE_MAX_DIM', 0)
        assert (ksg_mi(x, y, exclude=exclude), ksg_cmi(x, y, z, exclude=exclude)) == pytest.approx(tree, abs=1e-12)

    def test_autocorrelation_time(self):
        rng = np.random.default_rng(0)
        assert K.autocorrelation_time(rng.standard_normal((5000, 2))) == 0
        assert K.autocorrelation_time(_slow(20000, 0.9, rng)) in range(8, 11)   # 0.9^lag < 1/e from lag 10

    def test_the_window_keeps_neighbours_in_time_from_reading_as_dependence(self):
        rng = np.random.default_rng(0)
        a, b = _slow(5000, 0.98, rng), _slow(5000, 0.98, rng)          # independent, slow
        assert ksg_mi(a, b, exclude=0) > 0.05
        assert abs(ksg_mi(a, b)) < 0.03


class TestThroughRun:
    def test_estimate(self):
        x, y = _pair()
        r = nmi.run(x[:, None], y[:, None], mode='estimate', estimator='ksg', seed=0, **QUIET)
        assert r.mi_estimate == pytest.approx(ksg_mi(x, y) / LN2, abs=1e-9)

    def test_estimator_params_reach_the_estimator(self):
        x, y = _pair()
        r = nmi.run(x[:, None], y[:, None], mode='estimate', estimator=nmi.Estimator(name='ksg', params={'k': 8}),
                    seed=0, **QUIET)
        assert r.mi_estimate == pytest.approx(ksg_mi(x, y, k=8) / LN2, abs=1e-9)
        with pytest.raises(ValueError, match="'k', 'gaussianize', 'exclude'"):
            nmi.run(x[:, None], y[:, None], mode='estimate', estimator=nmi.Estimator(name='ksg', params={'clip': 5}),
                    **QUIET)

    def test_transfer_is_the_conditional_estimate(self):
        """Joint minus storage equals Frenzel-Pompe on the rows the library builds."""
        g = LatentGaussian(np.array([[0.8, 0.0], [0.6, 0.3]]), {'x': np.array([[1.0, 0.0]]),
                                                                 'y': np.array([[0.0, 1.0]])}, noise=0.5)
        s, k = g.sample(3000, seed=1), 2
        r = nmi.transfer_entropy(s['x'], s['y'], history_window=k, estimator='ksg', seed=0, **QUIET)
        past = lambda v: np.stack([v[k - j - 1:len(v) - j - 1, 0] for j in range(k)], 1)
        direct = ksg_cmi(past(s['x']), s['y'][k:], past(s['y']), seed=r.params['random_seed']) / LN2
        assert r.mi_estimate == pytest.approx(direct, abs=1e-12)
        exact = g.exact([('x', -j) for j in range(1, k + 1)], [('y', 0)], [('y', -j) for j in range(1, k + 1)])
        assert r.mi_estimate == pytest.approx(exact, abs=0.03)

    def test_conditional_and_interaction(self):
        s = LatentGaussian.shared(dims={'x': 1, 'y': 1, 'w': 1}, d=1, phi=0.0, noise=0.6, seed=2).sample(3000, seed=0)
        x, y, w = s['x'], s['y'], s['w']
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cmi = nmi.run(x, y, mode='conditional', conditional=nmi.Conditional(w_data=w), estimator='ksg', seed=0,
                          **QUIET)
            ii = nmi.run(x, y, mode='interaction', interaction=nmi.Interaction(w_data=w), estimator='ksg', seed=0,
                         **QUIET)
        assert not [m for m in caught if issubclass(m.category, CombinationWarning)]   # the terms err together
        assert cmi.mi_estimate == pytest.approx(ksg_cmi(x, y, w, seed=0) / LN2, abs=1e-12)
        assert ii.mi_estimate == pytest.approx((ksg_cmi(w, y, x, seed=0) - ksg_mi(w, y, seed=0)) / LN2, abs=1e-12)

    def test_a_pair_of_different_lengths(self):
        """align='dual_branch' hands the joint term the pair (X, C): the conditional estimate given C."""
        rng = np.random.default_rng(0)
        a, c = rng.standard_normal((1500, 2, 3)), rng.standard_normal((1500, 1, 5))
        y = a[:, :1, :1] + c[:, :, :1] + rng.standard_normal((1500, 1, 1))
        r = nmi.run(a, y, mode='conditional', conditional=nmi.Conditional(w_data=c, align='dual_branch'),
                    estimator='ksg', seed=0, **QUIET)
        assert r.mi_estimate == pytest.approx(ksg_cmi(a, y, c, seed=0) / LN2, abs=1e-12)

    def test_rigorous_and_permutation(self):
        x, y = _pair(4000, rho=0.8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(x[:, None], y[:, None], mode='rigorous', estimator='ksg',
                        rigorous=nmi.Rigorous(gamma_range=range(1, 6)), seed=0, **QUIET)
        assert not [m for m in caught if "evaluation batch" in str(m.message)]         # KSG holds nothing out
        assert r.mi_estimate == pytest.approx(-0.5 * np.log2(1 - 0.64), abs=0.05)
        p = nmi.run(x[:, None], y[:, None], mode='estimate', estimator='ksg', permutation_test=True, n_permutations=5,
                    seed=0, **QUIET)
        assert p.get('p_value') == pytest.approx(1 / 6)

    @pytest.mark.parametrize('mode', ['dimensionality', 'precision'])
    def test_modes_that_read_embeddings_refuse_it(self, mode):
        x, y = _pair(500)
        with pytest.raises(ValueError, match='trains none'):
            nmi.run(x[:, None], y[:, None], mode=mode, estimator='ksg', **QUIET)

    def test_training_settings_warn_that_they_are_ignored(self):
        x, y = _pair(500)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(x[:, None], y[:, None], mode='estimate', estimator='ksg', training=nmi.Training(n_epochs=5),
                    **QUIET)
        assert any('n_epochs' in str(w.message) and 'trains no network' in str(w.message) for w in caught)
