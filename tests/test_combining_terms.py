"""Every quantity combined from separately trained networks reads its terms by one rule.

``sweep.combine_terms`` pairs the terms' repeats by position. A term whose
network measured something enters as measured. A term whose network learned
nothing distinguishable from zero enters as 0, with a warning naming it. A term
whose network read negative on its training rows while its held-out value
stood clearly above zero has two readings that contradict each other: it has
no value, and neither has its repeat. The trainer words its own warning by the
same line (``epochs.stands_above_zero``).
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from neural_mi.analysis.assemble import aggregate
from neural_mi.analysis.sweep import _joint_marginal_difference, combine_terms, term_values
from neural_mi.exceptions import CombinationWarning
from neural_mi.training.epochs import CLEAR_SE, stands_above_zero


def _net(raw, test=0.5, eval_size=1000, nothing=False):
    """One network's result as the trainer reports it, in nats: the reported value is 0 when the network
    learned nothing or its training-side value came out negative, with the measured one in raw_train_mi."""
    reported = 0.0 if nothing or raw < 0 else raw
    return {'train_mi': reported, 'raw_train_mi': raw, 'test_mi': test, 'eval_size': eval_size,
            'all_mi_negative': nothing}


# A network whose held-out value stood clearly above zero and whose training-side value then came out
# negative: the conditioning term of the mi_rate run that reported its joint term alone (2.41 for 0.17).
CONTRADICTING = _net(-0.05, test=1.29)
# A network that learned nothing, as a term that is zero does.
NOTHING = _net(0.2, test=-0.01, nothing=True)
# Both readings within noise of zero: 0.001 nats held out on 100 rows is far inside 3 standard errors.
WITHIN_NOISE = _net(-0.0002, test=0.001, eval_size=100)


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = fn(*args, **kwargs)
    return out, [w for w in caught if issubclass(w.category, CombinationWarning)]


class TestStandsAboveZero:

    def test_the_line_is_clear_se_standard_errors(self):
        # sqrt(2 I / N) is the standard error of I nats on N rows, so I > CLEAR_SE sqrt(2 I / N)
        # is I > 2 CLEAR_SE^2 / N.
        line = 2 * CLEAR_SE ** 2 / 1000
        assert stands_above_zero(1.01 * line, 1000)
        assert not stands_above_zero(0.99 * line, 1000)

    @pytest.mark.parametrize('value', [0.0, -0.3, math.nan, None])
    def test_no_value_at_or_below_zero_stands_above_it(self, value):
        assert not stands_above_zero(value, 1000)

    def test_without_a_row_count_any_value_above_zero_counts(self):
        assert stands_above_zero(1e-6, None)


class TestTermValues:

    def test_a_measured_term_enters_as_measured(self):
        assert term_values([_net(0.4)]) == ([0.4], ['measured'])

    def test_a_network_that_learned_nothing_enters_as_zero_not_as_its_overfit_value(self):
        assert term_values([NOTHING]) == ([0.0], ['nothing'])

    def test_a_negative_reading_within_noise_of_zero_learned_nothing(self):
        assert term_values([WITHIN_NOISE]) == ([0.0], ['nothing'])

    def test_a_negative_reading_against_a_clear_held_out_value_fails(self):
        values, states = term_values([CONTRADICTING])
        assert math.isnan(values[0]) and states == ['failed']

    def test_a_repeat_without_a_value_fails(self):
        values, states = term_values([{'no_value': True}, {'train_mi': math.nan}])
        assert all(math.isnan(v) for v in values) and states == ['failed', 'failed']

    def test_a_nonparametric_estimate_is_a_measurement_whatever_its_sign(self):
        # A KSG term has no held-out reading, and a negative value is its bias around zero.
        assert term_values([{'train_mi': -0.01}]) == ([-0.01], ['measured'])


class TestCombineTerms:

    @staticmethod
    def _terms(joint, marginal):
        return [('J', 'j', joint, 1), ('M', 'm', marginal, -1)]

    def test_a_failed_term_fails_its_repeat_and_only_that_repeat(self):
        combined = combine_terms(self._terms([_net(1.0), _net(1.2), _net(1.4)],
                                             [_net(0.4), CONTRADICTING, _net(0.5)]))
        per_repeat = combined['per_repeat']
        assert per_repeat[0] == pytest.approx(0.6) and per_repeat[2] == pytest.approx(0.9)
        assert math.isnan(per_repeat[1])
        # Every term of the failed repeat is NaN, the healthy joint term included.
        assert math.isnan(combined['components']['j'][1]) and math.isnan(combined['components']['m'][1])
        assert combined['failed'] == [[], ['m'], []]
        assert combined['means'] == {'j': pytest.approx(1.2), 'm': pytest.approx(0.45)}
        assert combined['value'] == pytest.approx(0.75)
        assert combined['per_run'] == ([1.0, 1.4], [0.4, 0.5])

    def test_a_term_that_learned_nothing_enters_as_zero(self):
        combined = combine_terms(self._terms([_net(0.8)], [NOTHING]))
        assert combined['per_repeat'] == [pytest.approx(0.8)]
        assert combined['nothing'] == [['m']] and combined['failed'] == [[]]

    def test_with_every_repeat_failed_there_is_no_value(self):
        combined = combine_terms(self._terms([_net(1.0)], [CONTRADICTING]))
        assert math.isnan(combined['value'])
        assert all(math.isnan(v) for v in combined['means'].values())
        assert combined['per_run'] == ([], [])

    def test_a_repeat_one_term_lacks_fails(self):
        combined = combine_terms(self._terms([_net(1.0), _net(1.1)], [_net(0.4)]))
        assert combined['failed'] == [[], ['m']]
        assert combined['value'] == pytest.approx(0.6)

    def test_one_warning_names_the_failed_terms_and_counts_their_repeats(self):
        _, caught = _caught(combine_terms, self._terms([_net(1.0), _net(1.2)], [CONTRADICTING, _net(0.5)]),
                            quantity_name='Q', warn=True)
        assert len(caught) == 1
        message = str(caught[0].message)
        assert message.startswith('Q: 1 of 2 repeats are left out') and 'I(M) in 1' in message

    def test_a_warning_names_the_terms_that_learned_nothing(self):
        _, caught = _caught(combine_terms, self._terms([_net(1.0), _net(1.2)], [NOTHING, WITHIN_NOISE]),
                            quantity_name='Q', warn=True)
        assert len(caught) == 1
        message = str(caught[0].message)
        assert 'learned nothing distinguishable from zero' in message and 'I(M) in 2' in message

    def test_measured_terms_and_an_unreported_combination_are_quiet(self):
        _, caught = _caught(combine_terms, self._terms([_net(1.0)], [_net(0.4)]), quantity_name='Q', warn=True)
        assert caught == []
        _, caught = _caught(combine_terms, self._terms([_net(1.0)], [CONTRADICTING]), quantity_name='Q')
        assert caught == []


class TestDifferenceOfTwoNetworks:
    """``_joint_marginal_difference``, behind conditional MI, transfer entropy and the quantities built on
    them, mi_rate among them."""

    @staticmethod
    def _difference(monkeypatch, joint, marginal):
        from neural_mi.analysis import sweep as sweep_module
        calls = {'n': 0}

        def fake_run(self, sweep_grid=None, n_workers=1, is_proc_sweep=False, **kwargs):
            calls['n'] += 1     # the conditioning term trains first
            return marginal if calls['n'] == 1 else joint

        monkeypatch.setattr(sweep_module.ParameterSweep, 'run', fake_run)
        return _caught(_joint_marginal_difference, None, None, None, None, {'output_units': 'nats'}, None, 1,
                       quantity_name='Rate', joint_label='J', marginal_label='M',
                       joint_key='j', marginal_key='m')

    def test_a_contradicting_conditioning_term_no_longer_leaves_the_joint_term_alone(self, monkeypatch):
        (difference, mi_joint, mi_marginal, *_), caught = self._difference(
            monkeypatch, [_net(2.41, test=2.46)], [CONTRADICTING])
        assert math.isnan(difference)
        assert any('1 of 1 repeats are left out' in str(w.message) for w in caught)

    def test_the_other_repeats_still_give_the_value(self, monkeypatch):
        (difference, mi_joint, mi_marginal, *_, per_run), _ = self._difference(
            monkeypatch, [_net(2.41), _net(4.83)], [CONTRADICTING, _net(4.66)])
        assert difference == pytest.approx(0.17)
        assert (mi_joint, mi_marginal) == (pytest.approx(4.83), pytest.approx(4.66))
        assert per_run == ([4.83], [4.66])

    def test_a_conditioning_term_that_learned_nothing_enters_as_zero_and_says_so(self, monkeypatch):
        (difference, *_), caught = self._difference(monkeypatch, [_net(0.9)], [NOTHING])
        assert difference == pytest.approx(0.9)
        assert any('I(M) in 1' in str(w.message) and 'learned nothing' in str(w.message) for w in caught)


class TestInteractionInformation:

    def test_a_failed_term_fails_the_repeat(self, monkeypatch):
        from neural_mi.analysis import sweep as sweep_module
        from neural_mi.analysis.interaction import run_interaction_information
        # Training order: I(X;Y), then I(X,W;Y), then I(W;Y). The joint term reads X's channel count off
        # the I(X;Y) results.
        canned = [[{**_net(0.1), 'n_channels_x': 1}, {**_net(0.1), 'n_channels_x': 1}],
                  [_net(0.9), _net(0.8)], [CONTRADICTING, _net(0.2)]]

        def fake_run(self, sweep_grid=None, n_workers=1, is_proc_sweep=False, **kwargs):
            return canned.pop(0)

        monkeypatch.setattr(sweep_module.ParameterSweep, 'run', fake_run)
        from collections import OrderedDict
        from neural_mi.data.plan import RowPlan, Stream
        plan = RowPlan(OrderedDict((name, Stream(torch.zeros(20, 1, 1))) for name in ('x', 'w', 'y')))
        out, caught = _caught(run_interaction_information, plan, {'output_units': 'nats'})
        assert out['interaction_info'] == pytest.approx(0.8 - 0.1 - 0.2)
        assert out['mi_estimate_std'] is None or math.isnan(out['mi_estimate_std'])
        assert any('1 of 2 repeats are left out' in str(c.message) and 'I(W;Y) in 1' in str(c.message)
                   for c in caught)


class TestRepeatRows:

    @staticmethod
    def _rows(mode, terms, back=None):
        from neural_mi.analysis.modes import _DIFFERENCES, _TRANSFER_REVERSE, _difference_rows
        raw = {key: results for (_, key, _), results in zip(_DIFFERENCES[mode], terms)}
        if back is not None:
            raw.update({key: results for (_, key, _), results in zip(_TRANSFER_REVERSE, back)})
        n = len(terms[0])
        rows, _ = _difference_rows(mode, raw, 0, {}, [], list(range(n)), False,
                                   bidirectional=back is not None)
        return rows

    def test_a_failed_repeat_is_nan_and_names_its_term(self):
        rows = self._rows('conditional', [[_net(1.0), _net(1.2)], [_net(0.4), CONTRADICTING]])
        assert rows[0]['mi'] == pytest.approx(0.6) and rows[0]['failed_term'] is None
        assert math.isnan(rows[1]['mi']) and math.isnan(rows[1]['mi_xw_y'])
        assert rows[1]['failed_term'] == 'mi_w_y'

    def test_the_reverse_direction_fails_on_its_own(self):
        rows = self._rows('transfer', [[_net(0.5)], [_net(0.1)]], back=[[_net(0.3)], [CONTRADICTING]])
        assert rows[0]['mi'] == pytest.approx(0.4)
        assert math.isnan(rows[0]['te_yx']) and math.isnan(rows[0]['directionality_index'])
        assert rows[0]['failed_term'] == 'i_xpast_xfuture'

    def test_the_row_counts_the_failed_repeats_and_averages_the_others(self):
        runs = pd.DataFrame({'config_id': [0, 0, 0], 'mi': [0.6, math.nan, 0.8],
                             'mi_raw': [0.6, math.nan, 0.8]})
        row = aggregate(runs, ['config_id']).iloc[0]
        assert row['n_runs'] == 3 and row['n_failed'] == 1 and row['n_zero'] == 0
        assert row['mi_mean'] == pytest.approx(0.7)
        assert row['mi_std'] == pytest.approx(np.std([0.6, 0.8], ddof=1))


class TestRigorousRungs:

    def test_a_rung_whose_term_failed_is_left_out_like_one_that_produced_nothing(self):
        from neural_mi.analysis.rigorous import _drop_zero_rungs
        group = pd.DataFrame({'gamma': [1, 2, 2, 3, 3, 3],
                              'train_mi': [0.5, 0.4, math.nan, 0.3, 0.0, 0.35]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            kept, n_dropped = _drop_zero_rungs(group, '')
        assert n_dropped == 2
        assert kept['train_mi'].tolist() == [0.5, 0.4, 0.3, 0.35]
        assert any('2 of 6 rungs of the gamma ladder produced nothing' in str(w.message) for w in caught)


def _spikes(rng, start, rate=5.0, end=200.0, n=3):
    return [np.sort(rng.uniform(start, end, rng.poisson(rate * (end - start)))) for _ in range(n)]


class TestTermsReadTheSameRows:
    """The terms of a difference are trained in separate tasks, each building its rows from the streams it
    reads. A repeat whose terms read different rows is refused: its difference would compare different
    stretches of the data."""

    JOINT = {'count': 400, 'origin': 0.05, 'step': 0.5, 'spans': {'x': 0.5, 'y': 0.5}, 'kept': None}

    def test_terms_on_different_rows_are_refused(self):
        late = {**self.JOINT, 'count': 320, 'origin': 40.0}
        terms = [('XW;Y', 'mi_xw_y', [{**_net(0.3), '_row_identity': self.JOINT}], 1),
                 ('W;Y', 'mi_w_y', [{**_net(0.1), '_row_identity': late}], -1)]
        with pytest.raises(ValueError, match=r"Conditional MI: I\(XW;Y\) and I\(W;Y\) read different rows "
                                             r"\(400 rows starting at 0.05, one every 0.5, Y over 0.5, "
                                             r"against 320 rows starting at 40"):
            combine_terms(terms, quantity_name='Conditional MI')

    def test_every_repeat_is_checked(self):
        late = {**self.JOINT, 'origin': 40.0}
        terms = [('XW;Y', 'mi_xw_y', [{**_net(0.3), '_row_identity': self.JOINT}] * 2, 1),
                 ('W;Y', 'mi_w_y', [{**_net(0.1), '_row_identity': self.JOINT}, {**_net(0.1), '_row_identity': late}], -1)]
        with pytest.raises(ValueError, match='read different rows'):
            combine_terms(terms)

    def test_terms_on_the_same_rows_combine(self):
        terms = [('XW;Y', 'mi_xw_y', [{**_net(0.3), '_row_identity': self.JOINT}], 1),
                 ('W;Y', 'mi_w_y', [{**_net(0.1), '_row_identity': dict(self.JOINT)}], -1)]
        assert combine_terms(terms)['value'] == pytest.approx(0.2)

    @pytest.mark.parametrize('mode', ['conditional', 'interaction'])
    def test_spike_times_with_a_late_w_read_one_plan(self, mode):
        # Every term reads the rows of one plan of X, W and Y, which starts where the latest stream
        # starts: W, 40 s in.
        import neural_mi as nmi
        rng = np.random.default_rng(0)
        x, y, w = _spikes(rng, 0.0), _spikes(rng, 0.0), _spikes(rng, 40.0)
        spike = {'window_size': 0.5}
        processing = nmi.Processing(x='spike', x_params=spike, y='spike', y_params=spike, w='spike', w_params=spike)
        config = (nmi.Conditional(w_data=w) if mode == 'conditional' else nmi.Interaction(w_data=w))
        result = nmi.run(x, y, mode=mode, processing=processing, estimator='ksg', show_progress=False,
                         **{mode: config})
        sizes = set(result.details[0]['trainings']['eval_size'])
        assert len(sizes) == 1 and sizes.pop() <= 320

    @pytest.mark.parametrize('mode', ['conditional', 'interaction'])
    def test_streams_on_one_grid_combine(self, mode):
        import neural_mi as nmi
        rng = np.random.default_rng(0)
        x, w = rng.standard_normal((2000, 2)), rng.standard_normal((2000, 2))
        y = x + w + rng.standard_normal((2000, 2))
        config = (nmi.Conditional(w_data=w) if mode == 'conditional' else nmi.Interaction(w_data=w))
        result = nmi.run(x, y, mode=mode, estimator='ksg', show_progress=False, **{mode: config})
        assert np.isfinite(result.mi_estimate)
