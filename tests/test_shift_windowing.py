# tests/test_shift_windowing.py
"""Tests for neural_mi/data/shift_windowing.py: the sample-rate-aware unit
conversion, the categorical reslice+encode path, the bundle that keeps streams
in step, the vectorized spike windowing it sits alongside (data/temporal.py),
and which route a row plan takes (data/plan.py)."""
import warnings
import numpy as np
import torch
import pytest

from neural_mi.data.shift_windowing import (
    seconds_to_samples, resolve_step_size, make_categorical_encoder, WindowShifter,
)
from neural_mi.data.temporal import (
    CategoricalWindowDataset, relabel_categorical_data,
    SpikeWindowDataset,
)
from neural_mi.data.handler import WindowManager


def test_seconds_to_samples():
    assert seconds_to_samples(0.5, 1.0 / 1000.0) == 500
    assert seconds_to_samples(1.0, 1.0) == 1  # no sample_rate -> already "samples"
    assert seconds_to_samples(0.01, 1.0) == 1  # rounds up to at least 1


class TestResolveStepSize:
    """One step convention for both windowing paths: a fractional step is a
        fraction of the window on the shift path too.
    """

    def test_conventions(self):
        assert resolve_step_size(10, None) == 10.0     # tile without overlap
        assert resolve_step_size(10, 0.5) == 5.0       # fraction of the window
        assert resolve_step_size(10, 0.25) == 2.5
        assert resolve_step_size(10, 5) == 5.0         # already absolute
        assert resolve_step_size(10, 1) == 1.0         # 1 is absolute, not a fraction
        assert resolve_step_size(0.5, 0.5) == 0.25     # sub-second windows too

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="step_size must be > 0"):
            resolve_step_size(10, 0)
        with pytest.raises(ValueError, match="step_size must be > 0"):
            resolve_step_size(10, -1)

    def test_window_manager_agrees(self):
        for window, step in [(10, None), (10, 0.5), (10, 0.25), (10, 5), (0.5, 0.5)]:
            wm = WindowManager(window_size=window, step_size=step)
            assert wm.resolve_step() == resolve_step_size(window, step)


def test_fractional_step_reaches_the_shift_path():
    """A fractional step_size must survive into the built windows.

    Both paths are asked for a half-window step on the same raw series, and
    both must report the same step and land within one window of each other on
    the count. Before the shared convention the shift path stepped by 1 sample
    and built roughly six times as many windows.
    """
    import neural_mi as nmi

    rng = np.random.default_rng(0)
    T = 1500
    z = rng.standard_normal((T, 2))
    x = z + 0.5 * rng.standard_normal((T, 2))
    y = z + 0.5 * rng.standard_normal((T, 2))
    proc = nmi.Processing(x='continuous', y='continuous',
                          x_params={'step_size': 0.5}, y_params={'step_size': 0.5})
    counts = {}
    for shift in (True, False):
        result = nmi.block_mi(
            x, y, window_size=10, processing=proc,
            model=nmi.Model(embedding_dim=4, hidden_dim=16, n_layers=1),
            training=nmi.Training(n_epochs=2, batch_size=64, patience=2,
                                  shift_windows=shift),
            split=nmi.Split(mode='blocked'), estimator=nmi.Estimator('infonce'),
            show_progress=False, n_workers=1, seed=0,
        )
        assert result.get('row_step') == 5
        counts[shift] = result.get('train_eval_size')
    assert abs(counts[True] - counts[False]) <= 5


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


class TestBundleShifterDifferentSampleRates:
    def test_shapes_stay_fixed_across_shifts(self):
        from collections import OrderedDict
        from neural_mi.data.shift_windowing import BundleShifter, stream_spec
        raw_x = torch.randn(10000, 2)  # 1000 Hz, 10 sec
        raw_y = torch.randn(5000, 2)   # 500 Hz, 10 sec
        period_x, period_y = 1.0 / 1000.0, 1.0 / 500.0
        wsx, wsy = seconds_to_samples(0.5, period_x), seconds_to_samples(0.5, period_y)
        shifter = BundleShifter(OrderedDict(x=stream_spec(raw_x, wsx, wsx, period=period_x),
                                            y=stream_spec(raw_y, wsy, wsy, period=period_y)), shift=True)
        n_windows = shifter.n_windows
        assert n_windows > 0
        for shift_x in [0, 100, 400, wsx - 1]:
            windows = shifter.windows_at(shift_x)
            assert windows['x'].shape == (n_windows, 2, wsx)
            assert windows['y'].shape == (n_windows, 2, wsy)

    def test_a_step_longer_than_the_window_moves_every_stream_by_the_same_time(self):
        """With a step of 30 samples and windows of 10, a shift reaches 29 samples. Every stream moves by
        all of it, where a shift clamped to one window would leave Y behind X."""
        from collections import OrderedDict
        from neural_mi.data.shift_windowing import BundleShifter, stream_spec
        ramp = torch.arange(3000, dtype=torch.float32)[:, None]
        slow = torch.arange(0, 3000, 2, dtype=torch.float32)[:, None]
        shifter = BundleShifter(OrderedDict(x=stream_spec(ramp, 10, 30, period=1.0),
                                            y=stream_spec(slow, 5, 15, period=2.0)), shift=True)
        windows = shifter.windows_at(28)
        assert float(windows['x'][0, 0, 0]) == 28.0
        assert float(windows['y'][0, 0, 0]) == 28.0


def _reference_spike_windows(spike_trains, window_times, window_size, max_samples_per_window,
                             no_spike_value):
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
    lacks a shared time unit: spike times are in seconds and a regular stream
    without a clock is in samples, so their windows cover different times."""

    def _spikes(self, n=3, seconds=300.0, rate=5.0, seed=0):
        rng = np.random.default_rng(seed)
        return [np.sort(rng.uniform(0, seconds, rng.poisson(seconds * rate))) for _ in range(n)]

    def test_warns_without_sample_rate(self):
        from neural_mi.data.handler import create_dataset
        x = np.random.randn(3000, 2).astype('float32')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_dataset(x, self._spikes(), processor_type_x='continuous',
                           processor_params_x={'window_size': 5.0, 'step_size': 5.0},
                           processor_type_y='spike',
                           processor_params_y={'window_size': 5.0, 'step_size': 5.0})
        assert any('sample_rate' in str(w.message) and 'meaningless' in str(w.message) for w in caught)

    def test_silent_with_sample_rate(self):
        from neural_mi.data.handler import create_dataset
        x = np.random.randn(3000, 2).astype('float32')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_dataset(x, self._spikes(), processor_type_x='continuous',
                           processor_params_x={'window_size': 5.0, 'step_size': 5.0, 'sample_rate': 100.0},
                           processor_type_y='spike',
                           processor_params_y={'window_size': 5.0, 'step_size': 5.0})
        assert not any('meaningless' in str(w.message) for w in caught)

    def test_silent_for_non_mixed_pairs(self):
        from neural_mi.data.handler import create_dataset
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_dataset(x, y, processor_type_x='continuous',
                           processor_params_x={'window_size': 20, 'step_size': 20},
                           processor_type_y='continuous',
                           processor_params_y={'window_size': 20, 'step_size': 20})
        assert not any('meaningless' in str(w.message) for w in caught)


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

        # Take the padding sentinel from the dataset rather than assuming one.
        # What this test checks is that the vectorised windowing agrees with the
        # two-pointer reference, which is independent of the value chosen to
        # mark an empty slot.
        expected = _reference_spike_windows(
            [s.copy() for s in spikes], wm.window_times, window_size,
            ds.max_samples_per_window, no_spike_value=ds.no_spike_value,
        )
        assert np.array_equal(ds.data.numpy(), expected)


class TestBundleShifter:
    """One shift, any number of streams, each in its own sample units."""

    @staticmethod
    def _ramp(n, channels=1):
        return torch.arange(n * channels, dtype=torch.float32).reshape(n, channels)

    def _bundle(self, n_streams, window=10, step=10):
        from collections import OrderedDict
        from neural_mi.data.shift_windowing import BundleShifter
        return BundleShifter(OrderedDict(
            (chr(ord('a') + i), dict(raw=self._ramp(400), window_size=window,
                                     step_size=step, period=1.0))
            for i in range(n_streams)
        ), shift=True)

    @pytest.mark.parametrize("n_streams", [1, 2, 3, 4])
    def test_any_number_of_streams(self, n_streams):
        bundle = self._bundle(n_streams)
        windows = bundle.windows_at(0)
        assert len(windows) == n_streams
        assert len({w.shape[0] for w in windows.values()}) == 1

    @pytest.mark.parametrize("shift", [0, 3, 7])
    def test_every_stream_moves_by_the_same_time(self, shift):
        bundle = self._bundle(3)
        windows = bundle.windows_at(shift)
        starts = {name: float(w[0, 0, 0]) for name, w in windows.items()}
        assert set(starts.values()) == {float(shift)}

    def test_a_slower_stream_shifts_by_the_same_real_time(self):
        """Half the sample rate means half the samples for the same duration."""
        from collections import OrderedDict
        from neural_mi.data.shift_windowing import BundleShifter
        bundle = BundleShifter(OrderedDict((
            ('fast', dict(raw=self._ramp(400), window_size=20, step_size=20, period=1.0)),
            ('slow', dict(raw=self._ramp(200), window_size=10, step_size=10, period=2.0)),
        )), shift=True)
        windows = bundle.windows_at(8)
        assert float(windows['fast'][0, 0, 0]) == 8.0
        assert float(windows['slow'][0, 0, 0]) == 4.0     # 8 fast samples = 4 slow ones


class TestContinuousReslicePathMatchesTheGrid:
    """The reslice path must reproduce the grid path's windows exactly.

    There are two windowing implementations and there is a measured reason for
    that: re-windowing through the grid costs ~44 ms per epoch on a 200k-sample
    continuous pair, against ~0.01 ms for the view-based reslice. They cannot be
    one implementation at that ratio, so what keeps them honest is that they
    agree wherever both apply. Categorical and spike already have such a check;
    this is continuous.

    The reslice path builds full windows only, and with a margin for shifts it
    builds fewer still.
    """

    @pytest.mark.parametrize("window_size,step_size", [(10, 10), (10, 5), (7, 7), (20, 4)])
    def test_shift_zero_matches_the_grid(self, window_size, step_size):
        from neural_mi.data.handler import create_dataset
        T = 400
        ramp = np.arange(T, dtype=np.float32)[:, None]
        eager = create_dataset(
            ramp, processor_type_x='continuous',
            processor_params_x={'window_size': window_size, 'step_size': step_size},
        ).x_data
        resliced = WindowShifter(torch.as_tensor(ramp), window_size, step_size).windows_at(0)
        n = resliced.shape[0]
        assert n <= eager.shape[0]
        assert torch.equal(eager[:n], resliced[:n])

    def test_a_shift_moves_the_windows_by_exactly_that_many_samples(self):
        T, window_size = 400, 10
        ramp = np.arange(T, dtype=np.float32)[:, None]
        shifter = WindowShifter(torch.as_tensor(ramp), window_size, window_size, max_shift=window_size - 1)
        base = shifter.windows_at(0)
        for shift in (1, 4, 9):
            moved = shifter.windows_at(shift)
            assert torch.equal(moved, base + float(shift))


class TestTheTwoRoutesReadTheSameClock:
    """A time vector means the same thing on both routes of a row plan.

        The grid in time reads the timestamps, so `window_size=1.0` is one
        second. The route by sample number converts a window size into a sample
        count once, up front, and takes the sample period from the time vector
        when there is no `sample_rate`, so a 100 Hz recording gets windows a
        hundred samples wide and the streams are cut to a common duration.
    """

    WIN = {'window_size': 1.0, 'step_size': 1.0}

    def _plan(self, x, y, x_time, y_time):
        from collections import OrderedDict
        from neural_mi.data.plan import RowPlan, resolve_stream
        return RowPlan(OrderedDict(x=resolve_stream(x, x_time, 'continuous', dict(self.WIN)),
                                   y=resolve_stream(y, y_time, 'continuous', dict(self.WIN))), shift=True)

    def _eager(self, x, y, x_time, y_time):
        from neural_mi.data.handler import create_dataset
        return create_dataset(x, y, x_time=x_time, y_time=y_time,
                              processor_type_x='continuous',
                              processor_params_x=dict(self.WIN),
                              processor_type_y='continuous',
                              processor_params_y=dict(self.WIN))

    @staticmethod
    def _regular_pair():
        xt = np.arange(0, 60, 1 / 100.0)
        yt = np.arange(0, 60, 1 / 25.0)
        return xt, yt, np.sin(xt)[:, None], np.cos(yt)[:, None]

    def test_window_width_comes_from_the_timestamps(self):
        xt, yt, x, y = self._regular_pair()
        plan = self._plan(x, y, xt, yt)
        assert plan.route == 'unfold'
        windows = plan.rows()['windows']
        eager = self._eager(x, y, xt, yt)
        # One second holds 100 samples of X and 25 of Y on either route.
        assert windows['x'].shape[2] == eager.x_data.shape[2] == 100
        assert windows['y'].shape[2] == eager.y_data.shape[2] == 25

    def test_sample_rate_still_wins_over_the_timestamps(self):
        from neural_mi.data.shift_windowing import stream_period
        assert stream_period({'sample_rate': 50}, np.arange(0, 10, 1 / 100.0)) == 1 / 50.0

    def test_no_time_vector_stays_in_raw_samples(self):
        from neural_mi.data.shift_windowing import stream_period
        assert stream_period({}, None) == 1.0
        assert stream_period(None, None) == 1.0

    def test_a_gap_puts_the_plan_on_the_grid_in_time(self):
        # Windowing by sample number cannot skip a gap, so a clock with one has
        # no single period that holds across the recording. The grid in time
        # checks each window's coverage against the timestamps themselves.
        xt, _, x, _ = self._regular_pair()
        yt_gap = np.concatenate([np.arange(0, 20, 1 / 25.0), np.arange(40, 80, 1 / 25.0)])
        assert self._plan(x, np.cos(yt_gap)[:, None], xt, yt_gap).route == 'grid'

    def test_clocks_that_start_apart_put_the_plan_on_the_grid_in_time(self):
        # The grid in time starts at the latest start among the streams;
        # windowing by sample number would pair X's first window with a
        # stretch of Y five seconds earlier.
        _, yt, _, y = self._regular_pair()
        xt_late = np.arange(5, 60, 1 / 100.0)
        plan = self._plan(np.sin(xt_late)[:, None], y, xt_late, yt)
        assert plan.route == 'grid'
        assert plan.rows()['origin'] == pytest.approx(5.0)
