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
        d = r.dataframe.iloc[0]
        # it must be consistent with the component means it is derived from
        assert d['amplification_factor'] == pytest.approx(
            (abs(d['mi_xw_y_mean']) + abs(d['mi_w_y_mean'])) / abs(d['mi_mean']))

    @pytest.mark.slow
    def test_interaction_reports_the_three_term_factor(self):
        _orc, s = _oracle_sample(w_noise=1.0)
        r = nmi.run(s['x'], s['y'], mode='interaction',
                    interaction=nmi.Interaction(w_data=s['w']), **_RUN)
        d = r.dataframe.iloc[0]
        assert d['amplification_factor'] == pytest.approx(
            (abs(d['mi_xw_y_mean']) + abs(d['mi_x_y_mean']) + abs(d['mi_w_y_mean']))
            / abs(d['mi_mean']))

    @pytest.mark.slow
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
        assert r.get('amplification_factor') > AMPLIFICATION_WARN_THRESHOLD
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
        assert r.get('amplification_factor') < 2.0
        text = ' '.join(str(c.message) for c in caught)
        assert 'amplification' not in text


class TestCombinationWarnings:
    """What a combined quantity's warning says follows from its components."""

    NATS = {'output_units': 'nats'}

    def _raised(self, *args, **kwargs):
        from neural_mi.analysis.sweep import warn_combination
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warn_combination(*args, **kwargs)
        return [str(w.message) for w in caught]

    def test_every_component_zero_is_undefined_and_quiet(self):
        # Two networks that learned nothing combine nothing, and their own
        # warnings already say so.
        assert np.isnan(amplification_factor([0.0, 0.0], 0.0))
        assert self._raised("Conditional MI", 0.0, ("XW;Y", 0.0, 'mi_xw_y'),
                            [("W;Y", 0.0, 'mi_w_y')], self.NATS) == []

    def test_close_components_below_zero_read_as_near_zero(self):
        messages = self._raised("Conditional MI", -0.01, ("XW;Y", 0.70, 'mi_xw_y'),
                                [("W;Y", 0.71, 'mi_w_y')], self.NATS)
        assert len(messages) == 1
        assert "estimate is negative" in messages[0] and "flip the order" in messages[0]

    def test_distant_components_below_zero_blame_a_network(self):
        # At a factor near 1 the joint term fell short, which noise does not explain.
        messages = self._raised("Conditional MI", -0.03, ("XW;Y", 0.0, 'mi_xw_y'),
                                [("W;Y", 0.03, 'mi_w_y')], self.NATS)
        assert len(messages) == 1
        assert "fell short" in messages[0] and "flips" not in messages[0]
        assert "1.0x" in messages[0]

    def test_negative_interaction_information_is_a_result(self):
        # II < 0 is redundancy. Components in a possible order raise nothing.
        assert self._raised("Interaction information", -0.4, ("X,W;Y", 1.0, 'mi_xw_y'),
                            [("X;Y", 0.7, 'mi_x_y'), ("W;Y", 0.7, 'mi_w_y')],
                            self.NATS, signed=True) == []

    def test_interaction_joint_below_a_marginal_is_flagged(self):
        messages = self._raised("Interaction information", -0.9, ("X,W;Y", 0.2, 'mi_xw_y'),
                                [("X;Y", 0.6, 'mi_x_y'), ("W;Y", 0.5, 'mi_w_y')],
                                self.NATS, signed=True)
        assert len(messages) == 1
        assert "impossible order" in messages[0] and "I(X;Y)=0.6000" in messages[0]
        assert "estimate is negative" not in messages[0]

    def test_three_term_amplification_names_all_three(self):
        messages = self._raised("Interaction information", 0.01, ("X,W;Y", 1.0, 'mi_xw_y'),
                                [("X;Y", 0.5, 'mi_x_y'), ("W;Y", 0.49, 'mi_w_y')],
                                self.NATS, signed=True)
        assert len(messages) == 1
        assert "three much larger estimates" in messages[0]
        assert all(k in messages[0] for k in ("'mi_xw_y'", "'mi_x_y'", "'mi_w_y'"))

    def test_interaction_warns_about_its_own_value(self, monkeypatch):
        """The two-term piece I(X,W;Y) - I(X;Y) is I(W;Y|X), and a warning
        about it would be mislabelled. Only the three-term value is checked."""
        from neural_mi.analysis import interaction
        calls = []
        real = interaction.warn_combination
        monkeypatch.setattr(interaction, 'warn_combination',
                            lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1])
        rng = np.random.default_rng(0)
        x, y, w = (rng.standard_normal((300, 2)).astype(np.float32) for _ in range(3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(x, y, mode='interaction', interaction=nmi.Interaction(w_data=w),
                    model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=nmi.Training(n_epochs=2, batch_size=64), show_progress=False)
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert kwargs.get('signed') is True and len(args[3]) == 2
        assert not any("estimate is negative" in str(w.message) for w in caught)


class TestQuantitiesThatCannotBeNegative:
    """A combined or extrapolated value of information is reported as 0 when it
    comes out negative, with the measured value kept beside it. Interaction
    information is signed and stays as measured."""

    @staticmethod
    def _rows(mode, joint, marginal, bidirectional=False, back=None):
        from neural_mi.analysis.modes import _difference_rows, _DIFFERENCES, _TRANSFER_REVERSE
        spec = _DIFFERENCES[mode]
        raw = {spec[0][1]: [{'train_mi': v} for v in joint],
               spec[1][1]: [{'train_mi': v} for v in marginal]}
        if mode == 'interaction':
            raw[spec[2][1]] = [{'train_mi': 0.0} for _ in joint]
        if bidirectional:
            raw[_TRANSFER_REVERSE[0][1]] = [{'train_mi': v} for v in back[0]]
            raw[_TRANSFER_REVERSE[1][1]] = [{'train_mi': v} for v in back[1]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rows, _ = _difference_rows(mode, raw, 0, {}, [], list(range(len(joint))), False,
                                       bidirectional=bidirectional)
        return rows, [str(w.message) for w in caught]

    def test_a_negative_repeat_is_reported_as_zero_under_a_positive_mean(self):
        # The averaging step says which repeats produced nothing (test_results.py).
        rows, messages = self._rows('conditional', joint=[0.1, 0.5], marginal=[0.3, 0.2])
        assert [r['mi'] for r in rows] == [0.0, pytest.approx(0.3)]
        assert rows[0]['mi_raw'] == pytest.approx(-0.2)
        assert messages == []

    def test_a_negative_mean_is_left_to_the_combined_check(self):
        # The check on the combined quantity already said it, with the reason.
        rows, messages = self._rows('conditional', joint=[0.1], marginal=[0.3])
        assert rows[0]['mi'] == 0.0 and rows[0]['mi_raw'] == pytest.approx(-0.2)
        assert messages == []

    def test_interaction_information_keeps_its_sign(self):
        rows, messages = self._rows('interaction', joint=[0.1], marginal=[0.3])
        assert rows[0]['mi'] == pytest.approx(-0.2) and messages == []

    def test_both_directions_of_transfer_entropy(self):
        rows, _ = self._rows('transfer', joint=[0.4], marginal=[0.1], bidirectional=True,
                             back=([0.1], [0.3]))
        assert rows[0]['mi'] == pytest.approx(0.3)
        assert rows[0]['te_yx'] == 0.0 and rows[0]['te_yx_raw'] == pytest.approx(-0.2)
        assert rows[0]['directionality_index'] == pytest.approx(1.0)

    @pytest.mark.parametrize('quantity, floored', [('rigorous', True), ('conditional', True),
                                                   ('interaction', False)])
    def test_an_extrapolated_value(self, quantity, floored):
        from neural_mi.analysis.modes import _fit_row
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            row = _fit_row({'mi_corrected': -0.05, 'mi_error': 0.08}, False, quantity=quantity)
        if floored:
            assert row['mi'] == 0.0 and row['mi_raw'] == pytest.approx(-0.05)
            assert len(caught) == 1 and "reported as 0" in str(caught[0].message)
        else:
            assert row['mi'] == pytest.approx(-0.05) and caught == []

    def test_the_combined_check_says_what_is_reported(self):
        from neural_mi.analysis.sweep import warn_combination
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warn_combination("TE(Y→X)", -0.03, ("yx_past;x_future", 0.0, 'i_yxpast_xfuture'),
                             [("x_past;x_future", 0.03, 'i_xpast_xfuture')],
                             {'output_units': 'nats'}, raw_key='te_yx_raw')
        assert "reported as 0 nats" in str(caught[0].message)
        assert "kept as te_yx_raw" in str(caught[0].message)
