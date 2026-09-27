# tests/test_permutation.py
"""Tests for permutation_test: one null per dataframe row, built the way the
observed value was built, with X moved in time and everything else left in place."""
import inspect
import math
import warnings
from unittest.mock import patch

import numpy as np
import torch
import pytest

import neural_mi as nmi
from neural_mi import Model, Training, Rigorous, Lag

_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

N = 500


def _pair(mi=1.0):
    return nmi.generators.generate_correlated_gaussians(N, dim=2, mi=mi)


class TestPermutationTest:

    def test_permutation_adds_a_null_and_a_p_value(self):
        x, y = _pair()
        result = nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                         permutation_test=True, n_permutations=3, n_workers=1,
                         show_progress=False)
        null = result.get('null_distribution')
        assert isinstance(null, list) and len(null) == 3
        assert all(isinstance(v, float) for v in null)
        p = result.get('p_value')
        assert 1 / 4 <= p <= 1.0

    def test_the_p_value_follows_its_definition(self):
        x, y = _pair()
        result = nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                         permutation_test=True, n_permutations=3, n_workers=1,
                         show_progress=False)
        null = [v for v in result.get('null_distribution') if not math.isnan(v)]
        expected = (1 + sum(v >= result.mi_estimate for v in null)) / (1 + len(null))
        assert result.get('p_value') == pytest.approx(expected)

    def test_permutation_false_leaves_details_clean(self):
        x, y = _pair()
        result = nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                         n_workers=1, show_progress=False)
        assert 'null_distribution' not in result.details[0]
        assert 'p_value' not in result.dataframe.columns

    def test_every_configuration_gets_its_own_null(self):
        x, y = _pair(0.5)
        with pytest.warns(UserWarning, match="all 2 configurations of sweep_grid included"):
            result = nmi.run(x, y, mode='sweep', model=_MODEL, training=_TRAINING,
                             sweep_grid={'embedding_dim': [4, 8]}, permutation_test=True,
                             n_permutations=2, n_workers=1, show_progress=False)
        for cid in (0, 1):
            assert len(result.details[cid]['null_distribution']) == 2
        assert result.dataframe['p_value'].notna().all()

    def test_every_lag_gets_its_own_null(self):
        x, y = _pair(0.5)
        result = nmi.run(x, y, mode='lag', lag=Lag(lag_range=range(-1, 2)), model=_MODEL,
                         training=_TRAINING, permutation_test=True, n_permutations=2,
                         n_workers=1, show_progress=False)
        null = result.details[0]['null_distribution']
        assert sorted(null) == [-1, 0, 1]
        assert all(len(v) == 2 for v in null.values())
        assert len(result.dataframe['p_value']) == 3

    def test_the_null_is_in_the_output_units(self):
        """The null is compared with mi_mean, so both carry the caller's units."""
        x, y = _pair()
        kw = dict(mode='estimate', model=_MODEL, training=_TRAINING, permutation_test=True,
                  n_permutations=2, n_workers=1, show_progress=False, seed=0)
        bits = nmi.run(x, y, output=nmi.Output(units='bits'), **kw)
        nats = nmi.run(x, y, output=nmi.Output(units='nats'), **kw)
        assert bits.mi_estimate == pytest.approx(nats.mi_estimate / math.log(2))
        assert bits.get('null_distribution') == pytest.approx(
            [v / math.log(2) for v in nats.get('null_distribution')])

    def test_raw_null_is_kept_beside_the_reported_one(self):
        x, y = _pair(0.5)
        result = nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                         permutation_test=True, n_permutations=2, n_workers=1,
                         show_progress=False)
        raw = result.get('null_distribution_raw')
        assert len(raw) == 2 and all(isinstance(v, float) for v in raw)

    def test_the_conditional_transfer_entropy_null_keeps_w(self):
        """Every trial of TE(X->Y|W) conditions on W, as the observed value does."""
        from neural_mi.analysis import modes
        rng = np.random.default_rng(0)
        x, y, w = (rng.standard_normal((400, 1)).astype(np.float32) for _ in range(3))
        seen = []
        original = modes._call_difference

        def spy(mode, x_, y_, w_, *args, **kwargs):
            seen.append(w_ is not None)
            return original(mode, x_, y_, w_, *args, **kwargs)

        with patch.object(modes, '_call_difference', side_effect=spy):
            nmi.run(x, y, mode='transfer', transfer=nmi.Transfer(history_window=2, w_data=w),
                    model=_MODEL, training=_TRAINING, permutation_test=True, n_permutations=2,
                    n_workers=1, show_progress=False)
        assert len(seen) == 3 and all(seen)


class TestNPermutationsDefault:

    def test_n_permutations_default_is_10(self):
        assert inspect.signature(nmi.run).parameters['n_permutations'].default == 10

    def test_a_small_n_permutations_says_what_it_can_resolve_and_costs(self):
        x, y = _pair()
        with pytest.warns(UserWarning) as caught:
            nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                    permutation_test=True, n_permutations=2, n_workers=1, show_progress=False)
        text = ' '.join(str(w.message) for w in caught)
        assert "n_permutations=2" in text
        assert "1/3" in text
        assert "100 or more" in text
        assert "2 times the call itself" in text

    def test_no_resolution_warning_at_100_permutations(self):
        """The message is given before any trial runs, so the trials are skipped here."""
        x, y = _pair()
        with patch('neural_mi.analysis.permutation.permutation_nulls', return_value=[]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                        permutation_test=True, n_permutations=100, n_workers=1,
                        show_progress=False)
        assert not [w for w in caught if "reliable p-value" in str(w.message)]

    def test_permutation_rigorous_raises(self):
        x, y = _pair()
        with pytest.raises(ValueError, match="not supported for mode='rigorous'"):
            nmi.run(x, y, mode='rigorous', model=_MODEL, training=_TRAINING,
                    permutation_test=True, rigorous=Rigorous(gamma_range=range(2, 4)),
                    n_workers=1)

    @pytest.mark.parametrize("mode", ['conditional', 'interaction', 'transfer'])
    def test_rigorous_difference_quantities_refuse_a_permutation_test(self, mode):
        rng = np.random.default_rng(0)
        x, y, w = (rng.standard_normal((300, 1)).astype(np.float32) for _ in range(3))
        cfg = {'conditional': {'conditional': nmi.Conditional(w_data=w, rigorous=True)},
               'interaction': {'interaction': nmi.Interaction(w_data=w, rigorous=True)},
               'transfer': {'transfer': nmi.Transfer(history_window=2, rigorous=True)}}[mode]
        with pytest.raises(ValueError, match="not supported with rigorous=True"):
            nmi.run(x, y, mode=mode, model=_MODEL, training=_TRAINING, permutation_test=True,
                    n_permutations=2, n_workers=1, show_progress=False, **cfg)


class TestPermutationTestProgressBar:
    """show_progress covers the permutation trials' own progress bar."""

    @pytest.mark.parametrize("show", [False, True])
    def test_show_progress_reaches_the_permutation_bar(self, show):
        from neural_mi.analysis import permutation
        x, y = _pair()
        with patch.object(permutation, 'tqdm', wraps=permutation.tqdm) as mock_tqdm:
            nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                    permutation_test=True, n_permutations=2, n_workers=1, show_progress=show)
        calls = [c for c in mock_tqdm.call_args_list if c.kwargs.get('desc') == 'Permutation test']
        assert calls
        assert all(c.kwargs.get('disable') is (not show) for c in calls)


class TestPairwisePermutation:

    _MODEL_P = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
    _TRAINING_P = Training(n_epochs=2, learning_rate=1e-3, batch_size=64, patience=2)

    def test_cross_pairwise_gets_a_null_per_pair(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((N, 3)).astype('float32')
        y = rng.standard_normal((N, 2)).astype('float32')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = nmi.run(x, y, mode='pairwise', model=self._MODEL_P, training=self._TRAINING_P,
                          permutation_test=True, n_permutations=2, n_workers=1,
                          show_progress=False)
        null = res.details[0]['null_distribution']
        assert sorted(null) == [(i, j) for i in range(3) for j in range(2)]
        assert all(len(v) == 2 and not all(np.isnan(v)) for v in null.values())
        assert res.dataframe['p_value'].notna().all()
        cost = [w for w in caught if "reruns the whole call" in str(w.message)]
        assert len(cost) == 1

    def test_self_pairwise_permutation_test_computes_no_null(self, caplog):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((N, 3)).astype('float32')
        with caplog.at_level('WARNING', logger='neural_mi'):
            res = nmi.run(x, mode='pairwise', model=self._MODEL_P, training=self._TRAINING_P,
                          permutation_test=True, n_permutations=2, n_workers=1,
                          show_progress=False)
        assert 'null_distribution' not in res.details[0]
        assert any("no effect for mode='pairwise' without y_data" in r.getMessage()
                   for r in caplog.records)


class TestConditionalInteractionRawDeferredPermutation:
    """The trials of a conditioning variable that is windowed together with X
    (mixed types under shift_windows, or spike+spike) run the same deferred
    route as the observed call."""

    _MODEL_P = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
    _TRAINING_P = Training(n_epochs=2, learning_rate=1e-3, batch_size=32, patience=2)

    @staticmethod
    def _mixed():
        rng = np.random.default_rng(0)
        T = 3000
        x = rng.standard_normal((T, 1)).astype('float32')
        w = rng.integers(0, 4, size=(T, 1)).astype('int64')
        y = rng.standard_normal((T, 1)).astype('float32')
        processing = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', w='categorical')
        return x, y, w, processing

    @pytest.mark.parametrize("mode", ['conditional', 'interaction'])
    def test_mixed_type_shift_windows_not_all_nan(self, mode):
        x, y, w, processing = self._mixed()
        cfg = ({'conditional': nmi.Conditional(w_data=w)} if mode == 'conditional'
               else {'interaction': nmi.Interaction(w_data=w)})
        res = nmi.run(x, y, mode=mode, processing=processing, model=self._MODEL_P,
                      training=Training(n_epochs=2, patience=1, shift_windows=True, batch_size=32),
                      permutation_test=True, n_permutations=2, n_workers=1, show_progress=False,
                      **cfg)
        null = res.get('null_distribution')
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null)

    def test_conditional_spike_conditioning_not_all_nan(self):
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=800, window_size=0.05, seed=0)
        res = nmi.run(
            x_spikes, y_spikes, mode='conditional',
            conditional=nmi.Conditional(w_data=w_spikes),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=self._MODEL_P, training=self._TRAINING_P,
            permutation_test=True, n_permutations=2, n_workers=1, show_progress=False,
        )
        null = res.get('null_distribution')
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null)


class TestSpikePermutationShuffle:
    """A spike population is moved in time. Reordering its list would only
    relabel neurons and leave every train, and so its alignment, intact."""

    def _make_population(self, n_neurons=4, duration=10.0, seed=0):
        rng = np.random.default_rng(seed)
        return [np.sort(rng.uniform(0, duration, size=rng.integers(5, 15)))
                for _ in range(n_neurons)]

    def test_circular_shift_preserves_spike_counts_and_bounds(self):
        from neural_mi.analysis.permutation import (_circular_shift_spike_population,
                                                    _spike_population_extent)
        y_data = self._make_population()
        t_start, t_end = _spike_population_extent(y_data, {})
        for seed in range(30):
            np.random.seed(seed)
            y_perm = _circular_shift_spike_population(y_data, t_start, t_end)
            for orig, new in zip(y_data, y_perm):
                assert len(orig) == len(new)
                assert np.all(new >= t_start - 1e-9) and np.all(new <= t_end + 1e-9)
                assert np.all(np.diff(new) >= 0), "spike times must stay sorted"

    def test_circular_shift_is_not_identity(self):
        from neural_mi.analysis.permutation import (_circular_shift_spike_population,
                                                    _spike_population_extent)
        y_data = self._make_population()
        t_start, t_end = _spike_population_extent(y_data, {})
        np.random.seed(1)
        y_perm = _circular_shift_spike_population(y_data, t_start, t_end)
        assert any(not np.allclose(orig, new) for orig, new in zip(y_data, y_perm))

    def test_block_shuffle_preserves_spike_counts_and_bounds(self):
        """A spike exactly at t_end is kept."""
        from neural_mi.analysis.permutation import (_block_shuffle_spike_population,
                                                    _spike_population_extent)
        y_data = self._make_population()
        t_start, t_end = _spike_population_extent(y_data, {})
        y_data = [np.append(st, t_end) for st in y_data]
        for seed in range(30):
            np.random.seed(seed)
            y_perm = _block_shuffle_spike_population(y_data, t_start, t_end, block_size=2.0)
            for orig, new in zip(y_data, y_perm):
                assert len(orig) == len(new), "a spike at t_end must not be dropped"
                assert np.all(new >= t_start - 1e-9) and np.all(new <= t_end + 1e-9)
                assert np.all(np.diff(new) >= 0)

    def test_spike_population_extent_uses_n_seconds_when_set(self):
        from neural_mi.analysis.permutation import _spike_population_extent
        y_data = [np.array([1.0, 2.0]), np.array([3.0])]
        t_start, t_end = _spike_population_extent(y_data, {'processor_params_x': {'n_seconds': 100.0}})
        assert (t_start, t_end) == (1.0, 100.0)

    def test_spike_population_extent_infers_from_spikes_without_n_seconds(self):
        from neural_mi.analysis.permutation import _spike_population_extent
        y_data = [np.array([1.0, 2.0]), np.array([3.0, 4.5])]
        assert _spike_population_extent(y_data, {}) == (1.0, 4.5)

    def test_invalid_permutation_shuffle_raises(self):
        x, y = np.random.randn(200, 1).astype('float32'), np.random.randn(200, 1).astype('float32')
        with pytest.raises(ValueError, match="permutation_shuffle"):
            nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                    permutation_test=True, n_permutations=2, permutation_shuffle='jitter',
                    show_progress=False)

    @pytest.mark.slow
    def test_circular_null_sits_below_a_real_estimate(self):
        """For correlated spike populations, the circular-shift null sits below
        the real estimate, as a null for a broken dependency should."""
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        # no_spike_value is pinned: the test needs a real estimate for the null
        # to sit below, and the signal that survives the spike representation
        # depends on the padding sentinel.
        r = nmi.run(
            x_spikes, y_spikes, mode='estimate',
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05,
                                                           'no_spike_value': -1.0}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32),
            permutation_test=True, n_permutations=5, permutation_shuffle='circular',
            n_workers=1, show_progress=False, seed=0,
        )
        null_mean = np.nanmean(r.get('null_distribution'))
        assert null_mean < r.mi_estimate - 0.02, (null_mean, r.mi_estimate)

    def test_block_shuffle_end_to_end(self):
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        r = nmi.run(
            x_spikes, y_spikes, mode='estimate',
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=2, patience=1),
            permutation_test=True, n_permutations=2, permutation_shuffle='block',
            n_workers=1, show_progress=False, seed=0,
        )
        null = r.get('null_distribution')
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null)


class TestModesWithoutANullDistribution:
    """A mode that cannot build a null says so and computes none."""

    @staticmethod
    def _x():
        return np.random.default_rng(0).standard_normal((300, 4)).astype(np.float32)

    @pytest.mark.parametrize("with_y", [False, True])
    def test_dimensionality_reports_and_computes_no_null(self, caplog, with_y):
        from neural_mi.config import Dimensionality
        y = {'y_data': self._x()} if with_y else {}
        with caplog.at_level('WARNING', logger='neural_mi'):
            result = nmi.run(x_data=self._x(), mode='dimensionality',
                             dimensionality=Dimensionality(n_splits=1),
                             model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                             training=Training(n_epochs=1), permutation_test=True,
                             n_permutations=3, n_workers=1, show_progress=False, **y)
        assert any("no effect for mode='dimensionality'" in r.getMessage()
                   for r in caplog.records)
        assert 'null_distribution' not in result.details[0]
        assert 'p_value' not in result.dataframe.columns

    @pytest.mark.parametrize("mode", ['rigorous', 'precision'])
    def test_the_refusal_names_the_modes_that_do_test(self, mode):
        from neural_mi.config import Precision
        cfg = ({'rigorous': Rigorous(gamma_range=[1.0, 0.5])} if mode == 'rigorous'
               else {'precision': Precision(tau_grid=[0.1])})
        with pytest.raises(ValueError) as exc:
            nmi.run(x_data=self._x(), y_data=self._x(), mode=mode,
                    model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=Training(n_epochs=1), permutation_test=True,
                    n_permutations=3, n_workers=1, show_progress=False, **cfg)
        msg = str(exc.value)
        assert 'not supported' in msg
        assert 'dimensionality' not in msg
        assert "'pairwise'" in msg


class TestTheNullMovesX:
    """A trial moves X, the source, and leaves Y and W as they are, so every
    relation that does not involve X survives into the null."""

    def test_a_trial_moves_x_and_keeps_y_and_w(self):
        from neural_mi.analysis import permutation
        rng = np.random.default_rng(0)
        x, y, w = (torch.as_tensor(rng.standard_normal((200, 2, 1)), dtype=torch.float32)
                   for _ in range(3))
        seen = {}

        def fake_produce(mode, x_, y_, w_, *args, **kwargs):
            seen.update(x=x_, y=y_, w=w_)
            return {'rows': [{'config_id': 0, 'run_id': 0, 'mi': 0.0, 'raw_train_mi': 0.0}],
                    'details': {}, 'axis_keys': []}

        with patch('neural_mi.analysis.modes.produce', side_effect=fake_produce):
            permutation._trial(('estimate', x, y, w, {}, None, {}, True, 1, 'circular'))
        assert not torch.equal(seen['x'], x)
        assert torch.equal(seen['y'], y) and torch.equal(seen['w'], w)

    @pytest.mark.parametrize("n", [20, 100, 1001])
    def test_a_circular_shift_rolls_rows_by_an_offset_away_from_zero(self, n):
        from neural_mi.analysis.permutation import shift_x
        x = torch.arange(n, dtype=torch.float32)[:, None, None]
        for seed in range(40):
            np.random.seed(seed)
            shifted = shift_x(x, {}, 'circular')
            offset = int(shifted[0, 0, 0].item())       # row 0 now holds row (n - k) % n
            k = (n - offset) % n
            assert 0.1 * n <= k <= 0.9 * n, (n, k)
            assert torch.equal(torch.roll(x, k, 0), shifted)

    def test_a_raw_series_rolls_its_samples(self):
        from neural_mi.analysis.permutation import shift_x
        x = np.arange(50, dtype=float)[:, None]
        np.random.seed(0)
        shifted = shift_x(x, {}, 'circular')
        assert sorted(shifted[:, 0]) == sorted(x[:, 0])
        assert np.all(np.diff(np.concatenate([shifted[:, 0], shifted[:1, 0]])) % 50 == 1)

    def test_a_spike_shift_stays_away_from_zero(self):
        from neural_mi.analysis.permutation import (_circular_shift_spike_population,
                                                    _spike_population_extent)
        spikes = [np.array([0.0, 100.0]), np.array([50.0])]
        t_start, t_end = _spike_population_extent(spikes, {})
        for seed in range(40):
            np.random.seed(seed)
            moved = _circular_shift_spike_population(spikes, t_start, t_end)[1][0]
            delta = (moved - 50.0) % 100.0
            assert 10.0 <= delta <= 90.0

    def test_a_block_shuffle_keeps_whole_windows_of_a_raw_series(self):
        from neural_mi.analysis.permutation import shift_x
        x = np.arange(40, dtype=float)[:, None]
        np.random.seed(3)
        shuffled = shift_x(x, {'processor_params_x': {'window_size': 5}}, 'block')[:, 0]
        blocks = shuffled.reshape(8, 5)
        assert all(np.all(np.diff(b) == 1) and b[0] % 5 == 0 for b in blocks)
        assert not np.array_equal(shuffled, x[:, 0])

    def test_a_block_shuffle_of_windowed_rows_reorders_the_windows(self):
        from neural_mi.analysis.permutation import shift_x
        x = torch.arange(30, dtype=torch.float32).reshape(10, 1, 3)
        np.random.seed(1)
        shuffled = shift_x(x, {}, 'block')
        assert sorted(shuffled[:, 0, 0].tolist()) == sorted(x[:, 0, 0].tolist())
        assert torch.equal(shuffled[:, 0, 1] - shuffled[:, 0, 0], torch.ones(10))
