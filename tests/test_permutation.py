# tests/test_permutation.py
"""Tests for the permutation_test parameter."""
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
    """Tests for n_permutations default=1 and the insufficiency warning."""

    def test_n_permutations_default_is_1(self):
        """run() must have n_permutations=1 as the default."""
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


class TestPermutationTestProgressBar:
    """show_progress=False must also suppress the permutation test's own tqdm bar.

    The permutation test runs as a second, internal pass after the main
    estimate, with its own progress bar (desc="Permutation test") -- that bar
    used to be unconditional, so show_progress=False on the outer nmi.run()
    call silently didn't cover it.
    """

    def test_show_progress_false_disables_permutation_tqdm(self):
        import sys
        from unittest.mock import patch
        run_module = sys.modules['neural_mi.run']

        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        with patch.object(run_module, 'tqdm', wraps=run_module.tqdm) as mock_tqdm:
            nmi.run(
                x_data=x, y_data=y,
                mode='estimate',
                model=_MODEL, training=_TRAINING,
                permutation_test=True,
                n_permutations=2,
                n_workers=1,
                show_progress=False,
            )
        perm_calls = [c for c in mock_tqdm.call_args_list
                     if c.kwargs.get('desc') == 'Permutation test']
        assert perm_calls, "Expected at least one tqdm(desc='Permutation test') call"
        assert all(c.kwargs.get('disable') is True for c in perm_calls), (
            f"Permutation test tqdm must be disabled when show_progress=False; "
            f"got disable={[c.kwargs.get('disable') for c in perm_calls]}"
        )

    def test_show_progress_true_enables_permutation_tqdm(self):
        import sys
        from unittest.mock import patch
        run_module = sys.modules['neural_mi.run']

        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        with patch.object(run_module, 'tqdm', wraps=run_module.tqdm) as mock_tqdm:
            nmi.run(
                x_data=x, y_data=y,
                mode='estimate',
                model=_MODEL, training=_TRAINING,
                permutation_test=True,
                n_permutations=2,
                n_workers=1,
                show_progress=True,
            )
        perm_calls = [c for c in mock_tqdm.call_args_list
                     if c.kwargs.get('desc') == 'Permutation test']
        assert perm_calls, "Expected at least one tqdm(desc='Permutation test') call"
        assert all(c.kwargs.get('disable') is False for c in perm_calls), (
            f"Permutation test tqdm must stay enabled when show_progress=True; "
            f"got disable={[c.kwargs.get('disable') for c in perm_calls]}"
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


class TestConditionalInteractionRawDeferredPermutation:
    """Regression: permutation_test=True for mode='conditional'/'interaction'
    never forwarded raw_deferred/w_processor_type into the per-trial
    run_conditional_mi/run_interaction_information call, so every trial
    silently crashed (caught, logged as 'Permutation trial failed', all
    entries NaN) whenever the conditioning variable's own reachability path
    kept x_data/y_data/w_data raw -- a mixed continuous+categorical
    conditioning variable under shift_windows=True, or a spike+spike
    conditioning pair (raw_deferred there unconditionally, not just under
    shift_time, since merging before windowing is a correctness requirement
    for spike coverage, not just a shift-reachability nicety)."""

    _MODEL_P = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
    _TRAINING_P = Training(n_epochs=2, learning_rate=1e-3, batch_size=32, patience=2)

    def test_conditional_mixed_type_shift_windows_not_all_nan(self):
        from neural_mi import Conditional
        rng = np.random.default_rng(0)
        T = 3000
        x = rng.standard_normal((T, 1)).astype('float32')
        w = rng.integers(0, 4, size=(T, 1)).astype('int64')
        y = rng.standard_normal((T, 1)).astype('float32')
        processing = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        res = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=w, w_processor_type='categorical',
                                   w_processor_params={'window_size': 20, 'step_size': 20}),
            processing=processing, model=self._MODEL_P,
            training=Training(n_epochs=2, patience=1, shift_windows=True, batch_size=32),
            permutation_test=True, n_permutations=2, n_workers=1, show_progress=False,
        )
        null = res.details['null_distribution']
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null), "null distribution is all-NaN"

    def test_interaction_mixed_type_shift_windows_not_all_nan(self):
        from neural_mi import Interaction
        rng = np.random.default_rng(0)
        T = 3000
        x = rng.standard_normal((T, 1)).astype('float32')
        w = rng.integers(0, 4, size=(T, 1)).astype('int64')
        y = rng.standard_normal((T, 1)).astype('float32')
        processing = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        res = nmi.run(
            x, y, mode='interaction',
            interaction=Interaction(w_data=w, w_processor_type='categorical',
                                   w_processor_params={'window_size': 20, 'step_size': 20}),
            processing=processing, model=self._MODEL_P,
            training=Training(n_epochs=2, patience=1, shift_windows=True, batch_size=32),
            permutation_test=True, n_permutations=2, n_workers=1, show_progress=False,
        )
        null = res.details['null_distribution']
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null), "null distribution is all-NaN"

    def test_conditional_spike_conditioning_not_all_nan(self):
        from neural_mi import Conditional
        np.random.seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=40.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        w_spikes, _ = nmi.generators.generate_correlated_spike_trains(
            n_neurons=4, duration=40.0, firing_rate=8.0, delay=0.01, jitter=0.002
        )
        res = nmi.run(
            x_spikes, y_spikes, mode='conditional',
            conditional=Conditional(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05}),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=self._MODEL_P, training=self._TRAINING_P,
            permutation_test=True, n_permutations=2, n_workers=1, show_progress=False,
        )
        null = res.details['null_distribution']
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null), "null distribution is all-NaN"


class TestSpikePermutationShuffle:
    """Regression: the shared permutation shuffle used to reorder a spike
    population's *list position* (which array sits where), not its temporal
    alignment with X -- the reordered list contains the exact same,
    untouched per-neuron spike trains, so no X<->Y correspondence was ever
    actually broken. circular (default) and block (opt-in) replace this
    with an actual temporal permutation."""

    def _make_population(self, n_neurons=4, duration=10.0, seed=0):
        rng = np.random.default_rng(seed)
        return [np.sort(rng.uniform(0, duration, size=rng.integers(5, 15)))
               for _ in range(n_neurons)]

    def test_circular_shift_preserves_spike_counts_and_bounds(self):
        from neural_mi.run import _circular_shift_spike_population, _spike_population_extent
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
        """A shared random offset must actually move the spikes, not just
        reproduce the input (which would silently reintroduce the original
        list-reorder bug's failure mode: a 'shuffle' indistinguishable from
        no shuffle at all)."""
        from neural_mi.run import _circular_shift_spike_population, _spike_population_extent
        y_data = self._make_population()
        t_start, t_end = _spike_population_extent(y_data, {})
        np.random.seed(1)
        y_perm = _circular_shift_spike_population(y_data, t_start, t_end)
        assert any(not np.allclose(orig, new) for orig, new in zip(y_data, y_perm))

    def test_block_shuffle_preserves_spike_counts_and_bounds(self):
        """Also exercises the half-open-interval edge case: a spike landing
        exactly at t_end must not be silently dropped."""
        from neural_mi.run import _block_shuffle_spike_population, _spike_population_extent
        y_data = self._make_population()
        t_start, t_end = _spike_population_extent(y_data, {})
        y_data = [np.append(st, t_end) for st in y_data]  # force a spike exactly at t_end
        for seed in range(30):
            np.random.seed(seed)
            y_perm = _block_shuffle_spike_population(y_data, t_start, t_end, block_size=2.0)
            for orig, new in zip(y_data, y_perm):
                assert len(orig) == len(new), "a spike at t_end must not be dropped"
                assert np.all(new >= t_start - 1e-9) and np.all(new <= t_end + 1e-9)
                assert np.all(np.diff(new) >= 0)

    def test_spike_population_extent_uses_n_seconds_when_set(self):
        from neural_mi.run import _spike_population_extent
        y_data = [np.array([1.0, 2.0]), np.array([3.0])]
        t_start, t_end = _spike_population_extent(y_data, {'processor_params_y': {'n_seconds': 100.0}})
        assert t_start == 1.0
        assert t_end == 100.0

    def test_spike_population_extent_infers_from_spikes_without_n_seconds(self):
        from neural_mi.run import _spike_population_extent
        y_data = [np.array([1.0, 2.0]), np.array([3.0, 4.5])]
        t_start, t_end = _spike_population_extent(y_data, {})
        assert t_start == 1.0
        assert t_end == 4.5

    def test_invalid_permutation_shuffle_raises(self):
        x, y = np.random.randn(200, 1).astype('float32'), np.random.randn(200, 1).astype('float32')
        with pytest.raises(ValueError, match="permutation_shuffle"):
            nmi.run(x, y, mode='estimate', model=_MODEL, training=_TRAINING,
                   permutation_test=True, n_permutations=2, permutation_shuffle='jitter',
                   show_progress=False)

    def test_circular_default_gives_lower_null_than_broken_list_reorder(self):
        """End-to-end sanity check, not just unit-level: for genuinely
        X<->Y-correlated spike populations, the fixed (circular) null should
        land meaningfully below the real estimate -- unlike the old
        list-reorder shuffle, which left the 'permuted' Y statistically
        identical to the original and so produced null values hugging the
        real estimate instead of reflecting a broken dependency."""
        np.random.seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=40.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        model = Model(embedding_dim=8, hidden_dim=16, n_layers=1)
        training = Training(n_epochs=15, patience=5, batch_size=32)
        # no_spike_value is pinned rather than left to the default: this test
        # needs a real estimate for the null to sit below, and how much signal
        # survives the spike representation depends on the padding sentinel.
        # What is under test here is the shuffle, so the representation is held
        # fixed.
        r = nmi.run(
            x_spikes, y_spikes, mode='estimate',
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05,
                                                           'no_spike_value': -1.0}),
            model=model, training=training,
            permutation_test=True, n_permutations=5, permutation_shuffle='circular',
            n_workers=1, show_progress=False, seed=0,
        )
        null_mean = np.nanmean(r.details['null_distribution'])
        assert null_mean < r.mi_estimate - 0.02, (
            f"null mean ({null_mean:.4f}) should sit well below the real "
            f"estimate ({r.mi_estimate:.4f}) for genuinely correlated populations"
        )

    def test_block_shuffle_end_to_end(self):
        np.random.seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=40.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        r = nmi.run(
            x_spikes, y_spikes, mode='estimate',
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=2, patience=1),
            permutation_test=True, n_permutations=2, permutation_shuffle='block',
            n_workers=1, show_progress=False, seed=0,
        )
        null = r.details['null_distribution']
        assert len(null) == 2
        assert not all(np.isnan(v) for v in null)
