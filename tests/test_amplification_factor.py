"""Tests for the error-amplification factor on chain-rule quantities."""
import warnings

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi.analysis.sweep import (amplification_factor,
                                      AMPLIFICATION_WARN_THRESHOLD)
from neural_mi.generators.oracle import SharedLatentGaussian


class TestAmplificationFactorMath:
    def test_matches_documented_two_term_formula(self):
        # THEORY.md defines it as (t1 + t2) / (t1 - t2) for a two-term difference.
        t1, t2 = 0.7674, 0.4463
        assert amplification_factor([t1, t2], t1 - t2) == pytest.approx(
            (t1 + t2) / (t1 - t2))

    def test_near_one_when_nothing_cancels(self):
        # A residual that IS essentially the joint term amplifies nothing.
        assert amplification_factor([1.0, 0.0], 1.0) == pytest.approx(1.0)

    def test_grows_without_bound_as_result_approaches_zero(self):
        joint = 0.7674
        factors = [amplification_factor([joint, m], joint - m)
                   for m in (0.20, 0.60, 0.70, 0.76)]
        assert factors == sorted(factors), "must increase as the residual shrinks"
        assert factors[-1] > 100

    def test_zero_result_is_infinite(self):
        assert amplification_factor([1.0, 1.0], 0.0) == float('inf')

    def test_three_term_combination(self):
        # Interaction information combines three estimates, not two.
        a, b, c = 0.7674, 0.5298, 0.4463
        ii = a - b - c
        assert amplification_factor([a, b, c], ii) == pytest.approx(
            (a + b + c) / abs(ii))

    def test_sign_of_result_does_not_matter(self):
        assert amplification_factor([1.0, 0.9], 0.1) == pytest.approx(
            amplification_factor([1.0, 0.9], -0.1))


def _oracle_sample(w_noise, n=4000, seed=1):
    orc = SharedLatentGaussian(dims={'x': 4, 'y': 4, 'w': 8}, d=2, phi=0.0,
                               noise={'x': 1.0, 'y': 1.0, 'w': w_noise},
                               coupling=1.0, seed=0)
    return orc, orc.sample(n, seed=seed)


_RUN = dict(training=nmi.Training(n_epochs=25, patience=10),
            split=nmi.Split(mode='random'),
            n_workers=1, seed=0, show_progress=False)


class TestAmplificationFactorReported:
    def test_conditional_mi_reports_the_factor(self):
        _orc, s = _oracle_sample(w_noise=1.0)
        r = nmi.run(s['x'], s['y'], mode='conditional',
                    conditional=nmi.Conditional(w_data=s['w']), **_RUN)
        d = r.details
        assert 'amplification_factor' in d
        # it must be consistent with the components it is derived from
        assert d['amplification_factor'] == pytest.approx(
            (abs(d['mi_xw_y']) + abs(d['mi_w_y'])) / abs(d['cmi_estimate']))

    def test_interaction_reports_the_three_term_factor(self):
        _orc, s = _oracle_sample(w_noise=1.0)
        r = nmi.run(s['x'], s['y'], mode='interaction',
                    interaction=nmi.Interaction(w_data=s['w']), **_RUN)
        d = r.details
        assert 'amplification_factor' in d
        assert d['amplification_factor'] == pytest.approx(
            (abs(d['mi_xw_y']) + abs(d['mi_x_y']) + abs(d['mi_w_y']))
            / abs(d['interaction_info']))

    def test_warns_when_w_explains_x_away(self):
        # W is a near-noiseless readout of the shared latent, so the true
        # I(X;Y|W) is 0 and the estimate is a tiny residual of two large terms.
        orc, s = _oracle_sample(w_noise=0.05)
        # w_noise is small but nonzero, so the exact value is near zero rather
        # than identically zero.
        assert orc.exact([('x', 0)], [('y', 0)], [('w', 0)]) == pytest.approx(0.0, abs=1e-4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(s['x'], s['y'], mode='conditional',
                        conditional=nmi.Conditional(w_data=s['w']), **_RUN)
        assert r.details['amplification_factor'] > AMPLIFICATION_WARN_THRESHOLD
        text = ' '.join(str(c.message) for c in caught)
        assert 'amplification' in text, "high amplification must be surfaced to the user"

    def test_no_warning_when_the_residual_is_large(self):
        # XOR: I(X;Y|W) is the whole of the joint term, so nothing cancels.
        rng = np.random.default_rng(0)
        xb = rng.integers(0, 2, 4000)
        wb = rng.integers(0, 2, 4000)
        yb = xb ^ wb
        enc = lambda b: (2.0 * b - 1.0 + rng.normal(0, 0.15, len(b))).astype('float32')[:, None]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(enc(xb), enc(yb), mode='conditional',
                        conditional=nmi.Conditional(w_data=enc(wb)),
                        model=nmi.Model(hidden_dim=64, n_layers=2), **_RUN)
        assert r.details['amplification_factor'] < 2.0
        text = ' '.join(str(c.message) for c in caught)
        assert 'amplification' not in text
