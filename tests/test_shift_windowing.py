# tests/test_shift_windowing.py
"""Tests for neural_mi/data/shift_windowing.py: pair classification, the
sample-rate-aware unit conversion, the categorical reslice+encode path, and
the vectorized spike windowing it sits alongside (data/temporal.py)."""
import numpy as np
import torch
import pytest

from neural_mi.data.shift_windowing import (
    shift_family, mixed_pair_sample_rate_ok, seconds_to_samples,
    make_categorical_encoder, make_multi_categorical_encoder, WindowShifter, PairedWindowShifter,
)
from neural_mi.data.temporal import (
    CategoricalWindowDataset, relabel_categorical_data,
    SpikeWindowDataset, BinnedSpikeDataset,
)
from neural_mi.data.handler import WindowManager


class TestShiftFamily:
    def test_regular_pairs(self):
        assert shift_family('continuous', 'continuous') == 'regular'
        assert shift_family('continuous', 'categorical') == 'regular'
        assert shift_family('categorical', 'categorical') == 'regular'

    def test_spike_pair(self):
        assert shift_family('spike', 'spike') == 'spike'

    def test_mixed_pairs(self):
        assert shift_family('continuous', 'spike') == 'mixed'
        assert shift_family('spike', 'categorical') == 'mixed'

    def test_static_side_is_never_shiftable(self):
        assert shift_family(None, 'continuous') is None
        assert shift_family('spike', None) is None
        assert shift_family(None, None) is None


class TestMixedPairSampleRateOk:
    def test_true_when_regular_side_has_sample_rate(self):
        assert mixed_pair_sample_rate_ok('continuous', {'sample_rate': 100.0}, 'spike', {})
        assert mixed_pair_sample_rate_ok('spike', {}, 'categorical', {'sample_rate': 50.0})

    def test_false_when_missing(self):
        assert not mixed_pair_sample_rate_ok('continuous', {}, 'spike', {})
        assert not mixed_pair_sample_rate_ok('continuous', None, 'spike', None)


def test_seconds_to_samples():
    assert seconds_to_samples(0.5, 1.0 / 1000.0) == 500
    assert seconds_to_samples(1.0, 1.0) == 1  # no sample_rate -> already "samples"
    assert seconds_to_samples(0.01, 1.0) == 1  # rounds up to at least 1


class TestCategoricalEncoderMatchesCategoricalWindowDataset:
    """shift=0 through the reslice path must reproduce
    CategoricalWindowDataset's own (unshifted) windowing exactly, for all
    three encodings -- this is the byte-identical parity the reslice
    mechanism promises for the case it can be checked against directly."""

    @pytest.mark.parametrize("encoding", ["majority_vote", "probability", "full_trajectory"])
    def test_shift_zero_matches(self, encoding):
        rng = np.random.default_rng(0)
        T, C, K = 4000, 3, 5
        raw = rng.integers(0, K, size=(T, C)).astype(np.int64)
        window_size = step_size = 20  # non-overlapping so both paths tile identically

        wm = WindowManager(window_size=window_size, step_size=step_size,
                           t_start=0, t_end=T - window_size + 1)
        old = CategoricalWindowDataset(raw, window_manager=wm, encoding=encoding,
                                       min_coverage_fraction=0.0).data.numpy()

        arr = relabel_categorical_data(raw)
        raw_t = torch.as_tensor(arr, dtype=torch.long)
        n_categories = int(raw_t.max().item()) + 1
        encoder = make_categorical_encoder(n_categories, encoding)
        new = WindowShifter(raw_t, window_size, step_size, encoder).windows_at(0).numpy()

        n = min(old.shape[0], new.shape[0])
        assert np.allclose(old[:n], new[:n])


class TestMakeMultiCategoricalEncoder:
    """Phase 3: block-aware encoder for conditional/interaction's categorical
    X + categorical Z/W, each block keeping its own n_categories rather than
    one shared value inferred from the combined array's max."""

    @pytest.mark.parametrize("encoding", ["majority_vote", "probability", "full_trajectory"])
    def test_single_block_folds_category_axis_into_channels(self, encoding):
        """A single-block spec must reproduce make_categorical_encoder's own
        per-block content, folded into the channel axis exactly the way
        run._reshape_categorical_w_for_conditional already folds a single
        categorical conditioning variable -- the shape convention the whole
        multi-block design (letting differently-sized blocks concatenate)
        is built on."""
        rng = np.random.default_rng(0)
        n_windows, n_channels, window_size, n_categories = 5, 3, 6, 4
        raw = torch.as_tensor(rng.integers(0, n_categories, size=(n_windows, n_channels, window_size)),
                              dtype=torch.long)
        single = make_categorical_encoder(n_categories, encoding)(raw)
        multi = make_multi_categorical_encoder([(n_channels, n_categories)], encoding)(raw)
        if encoding == 'full_trajectory':
            expected = single.reshape(n_windows, n_channels, window_size, n_categories) \
                             .permute(0, 1, 3, 2).reshape(n_windows, n_channels * n_categories, window_size)
        else:
            expected = single.reshape(n_windows, n_channels * n_categories, 1)
        torch.testing.assert_close(multi, expected)

    @pytest.mark.parametrize("encoding", ["majority_vote", "probability", "full_trajectory"])
    def test_two_blocks_with_different_n_categories_are_not_conflated(self, encoding):
        """The core correctness property: block A (n_categories=3) and block
        B (n_categories=5) must each be one-hot encoded against their own
        category count -- not one shared count inferred from the combined
        array's max value (which would be 4, wrong for both blocks, and
        would silently corrupt every category index >= 4 in block A's
        one-hot as well as block B's)."""
        n_windows, window_size = 2, 4
        raw = torch.zeros(n_windows, 2, window_size, dtype=torch.long)
        raw[0, 0, :] = 2  # block A (3 categories): category 2
        raw[0, 1, :] = 4  # block B (5 categories): category 4 -- out of range for n_categories=3
        raw[1, 0, :] = 0
        raw[1, 1, :] = 1
        block_specs = [(1, 3), (1, 5)]
        encoded = make_multi_categorical_encoder(block_specs, encoding)(raw)

        # Independently re-derive each block's expected encoding via the
        # existing, already-proven single-block encoder, applied to that
        # block's own channel slice with its own n_categories.
        expected_a = make_categorical_encoder(3, encoding)(raw[:, 0:1, :])
        expected_b = make_categorical_encoder(5, encoding)(raw[:, 1:2, :])
        if encoding == 'full_trajectory':
            expected_a = expected_a.reshape(n_windows, 1, window_size, 3).permute(0, 1, 3, 2) \
                                    .reshape(n_windows, 3, window_size)
            expected_b = expected_b.reshape(n_windows, 1, window_size, 5).permute(0, 1, 3, 2) \
                                    .reshape(n_windows, 5, window_size)
        else:
            expected_a = expected_a.reshape(n_windows, 3, 1)
            expected_b = expected_b.reshape(n_windows, 5, 1)
        expected = torch.cat([expected_a, expected_b], dim=1)

        assert encoded.shape == expected.shape
        torch.testing.assert_close(encoded, expected)

    def test_channel_count_mismatch_raises(self):
        raw = torch.zeros(2, 3, 4, dtype=torch.long)
        with pytest.raises(ValueError):
            make_multi_categorical_encoder([(1, 3), (1, 5)], 'majority_vote')(raw)  # sums to 2, raw has 3


class TestPairedWindowShifterDifferentSampleRates:
    def test_shapes_stay_fixed_across_shifts(self):
        raw_x = torch.randn(10000, 2)  # 1000 Hz, 10 sec
        raw_y = torch.randn(5000, 2)   # 500 Hz, 10 sec
        period_x, period_y = 1.0 / 1000.0, 1.0 / 500.0
        wsx = seconds_to_samples(0.5, period_x)
        ssx = seconds_to_samples(0.5, period_x)
        wsy = seconds_to_samples(0.5, period_y)
        ssy = seconds_to_samples(0.5, period_y)
        shifter = PairedWindowShifter(raw_x, raw_y, wsx, ssx, wsy, ssy,
                                           period_x=period_x, period_y=period_y)
        n_windows = shifter.n_windows
        assert n_windows > 0
        for shift_x in [0, 100, 400, ssx - 1]:
            x, y = shifter.windows_at(shift_x)
            assert x.shape == (n_windows, 2, wsx)
            assert y.shape == (n_windows, 2, wsy)

    def test_same_rate_both_sides_matches_original_behavior(self):
        # period_x == period_y == 1.0 (the pre-existing, no-sample_rate case)
        # must reduce exactly to same-shift-value-both-sides.
        raw_x = torch.randn(1000, 2)
        raw_y = torch.randn(1000, 2)
        shifter = PairedWindowShifter(raw_x, raw_y, 20, 20)
        x0, y0 = shifter.windows_at(5)
        assert torch.equal(x0, raw_x[5:].unfold(0, 20, 20)[:shifter.n_windows].contiguous())
        assert torch.equal(y0, raw_y[5:].unfold(0, 20, 20)[:shifter.n_windows].contiguous())


def _reference_spike_windows(spike_trains, window_times, window_size, max_samples_per_window,
                             no_spike_value=-1.0):
    """Ground-truth reimplementation of the original (pre-vectorization)
    two-pointer loop, kept here (not imported) so this test stays a real
    independent check of the vectorized version in
    SpikeWindowDataset.move_data_to_windows, not a tautology against
    whatever the current implementation happens to do."""
    n_windows = len(window_times)
    data = np.full((n_windows, len(spike_trains), max_samples_per_window), no_spike_value, dtype=np.float32)
    for i, spikes in enumerate(spike_trains):
        if len(spikes) == 0 or n_windows == 0:
            continue
        spikes = spikes[(spikes >= window_times[0]) & (spikes < window_times[-1] + window_size)]
        L = R = 0
        for w in range(n_windows):
            w_start = window_times[w]
            w_end = w_start + window_size
            while L < len(spikes) and spikes[L] < w_start:
                L += 1
            while R < len(spikes) and spikes[R] < w_end:
                R += 1
            n_sp = min(R - L, max_samples_per_window)
            if n_sp > 0:
                data[w, i, :n_sp] = spikes[L:L + n_sp] - w_start
    return data


class TestCreateDatasetCrossUnitWarning:
    """create_dataset must flag a spike+continuous/categorical pairing that
    lacks a shared time unit, independent of whether shifting is ever used
    -- window alignment for such a pair is already questionable (see
    shift_family/mixed_pair_sample_rate_ok's docstrings)."""

    def _spikes(self, n=3, seconds=300.0, rate=5.0, seed=0):
        rng = np.random.default_rng(seed)
        return [np.sort(rng.uniform(0, seconds, rng.poisson(seconds * rate))) for _ in range(n)]

    def test_warns_without_sample_rate(self, caplog):
        from neural_mi.data.handler import create_dataset
        x = np.random.randn(3000, 2).astype('float32')
        with caplog.at_level("WARNING", logger="neural_mi"):
            create_dataset(x, self._spikes(), processor_type_x='continuous',
                           processor_params_x={'window_size': 20, 'step_size': 20},
                           processor_type_y='spike',
                           processor_params_y={'window_size': 5.0, 'step_size': 5.0})
        assert any('sample_rate' in r.message and 'meaningless' in r.message for r in caplog.records)

    def test_silent_with_sample_rate(self, caplog):
        from neural_mi.data.handler import create_dataset
        x = np.random.randn(3000, 2).astype('float32')
        with caplog.at_level("WARNING", logger="neural_mi"):
            create_dataset(x, self._spikes(), processor_type_x='continuous',
                           processor_params_x={'window_size': 5.0, 'step_size': 5.0, 'sample_rate': 100.0},
                           processor_type_y='spike',
                           processor_params_y={'window_size': 5.0, 'step_size': 5.0})
        assert not any('meaningless' in r.message for r in caplog.records)

    def test_silent_for_non_mixed_pairs(self, caplog):
        from neural_mi.data.handler import create_dataset
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        with caplog.at_level("WARNING", logger="neural_mi"):
            create_dataset(x, y, processor_type_x='continuous',
                           processor_params_x={'window_size': 20, 'step_size': 20},
                           processor_type_y='continuous',
                           processor_params_y={'window_size': 20, 'step_size': 20})
        assert not any('meaningless' in r.message for r in caplog.records)


class TestVectorizedSpikeWindowingMatchesReference:
    @pytest.mark.parametrize("window_size,step_size", [(2.0, 2.0), (5.0, 1.0), (1.0, 0.5)])
    def test_matches_two_pointer_reference(self, window_size, step_size):
        rng = np.random.default_rng(1)
        n_neurons, n_seconds, rate = 6, 300.0, 10.0
        spikes = [np.sort(rng.uniform(0, n_seconds, rng.poisson(n_seconds * rate)))
                 for _ in range(n_neurons)]
        spikes.append(np.array([]))  # empty-neuron edge case

        wm = WindowManager(window_size=window_size, step_size=step_size,
                           t_start=0, t_end=n_seconds - window_size)
        ds = SpikeWindowDataset([s.copy() for s in spikes], window_manager=wm)

        expected = _reference_spike_windows(
            [s.copy() for s in spikes], wm.window_times, window_size, ds.max_samples_per_window
        )
        assert np.array_equal(ds.data.numpy(), expected)
