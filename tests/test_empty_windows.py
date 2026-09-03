# tests/test_empty_windows.py
"""Tests for ``drop_empty_windows`` and window-retention reporting.

A silent spike window and an unobserved one are indistinguishable from spike
times alone, and the library treats both as absent. That default is right for
windowed MI, where an empty window carries no pattern to learn, and wrong for
anything indexed by time offset, where silence is the thing being predicted.
``drop_empty_windows`` selects between those two estimands.

The distinction these tests protect: the flag governs the *silence* rule on
spike data only. A continuous partner's coverage rule is a missing-data test
and must keep working regardless, which is what allows a timestamped
continuous variable to mask genuinely unobserved stretches while silent spike
windows are retained.
"""
import numpy as np
import pytest

from neural_mi.data.handler import create_dataset


def _sparse_spikes(n_neurons=3, duration=20.0, rate=1.0, seed=0):
    """Firing sparse enough that many windows are genuinely silent."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.uniform(0, duration, size=rng.poisson(rate * duration)))
            for _ in range(n_neurons)]


class TestDropEmptyWindows:

    def test_default_drops_silent_windows(self):
        spikes = _sparse_spikes()
        ds = create_dataset(x_data=spikes, processor_type_x='spike',
                            processor_params_x={'window_size': 0.5})
        expected = int(20.0 / 0.5)
        assert ds.x_data.shape[0] < expected, "sparse firing should leave silent windows"
        assert ds.window_retention < 1.0

    def test_disabled_keeps_every_window(self):
        spikes = _sparse_spikes()
        kept = create_dataset(x_data=spikes, processor_type_x='spike',
                              processor_params_x={'window_size': 0.5,
                                                  'drop_empty_windows': False})
        dropped = create_dataset(x_data=spikes, processor_type_x='spike',
                                 processor_params_x={'window_size': 0.5})
        assert kept.x_data.shape[0] > dropped.x_data.shape[0]
        assert kept.window_retention == pytest.approx(1.0)

    def test_retained_windows_are_contiguous_in_time(self):
        """The point of the flag: offsets need a real time axis.

        With silent windows dropped the surviving index is not a time axis,
        so ``index i`` and ``index i+1`` are not one step apart and any
        offset-based quantity computed on it is measuring the wrong offsets.
        """
        spikes = _sparse_spikes()
        dropped = create_dataset(x_data=spikes, processor_type_x='spike',
                                 processor_params_x={'window_size': 0.5})
        kept = create_dataset(x_data=spikes, processor_type_x='spike',
                              processor_params_x={'window_size': 0.5,
                                                  'drop_empty_windows': False})
        step_dropped = np.diff(dropped.window_manager.window_times)
        step_kept = np.diff(kept.window_manager.window_times)
        assert not np.allclose(step_dropped, 0.5), "dropping should break the axis"
        assert np.allclose(step_kept, 0.5), "keeping should preserve it"

    def test_binned_at_window_size_gives_a_per_bin_series(self):
        """window_size == bin_size makes each window exactly one time point."""
        spikes = _sparse_spikes(rate=5.0)
        ds = create_dataset(x_data=spikes, processor_type_x='spike',
                            processor_params_x={'bin_size': 0.05, 'window_size': 0.05,
                                                'normalize_bins': False,
                                                'drop_empty_windows': False})
        assert ds.x_data.shape[2] == 1, "one bin per window"
        assert np.allclose(np.diff(ds.window_manager.window_times), 0.05)
        # Squeezing the singleton bin axis gives the (T, C) series offsets need.
        assert ds.x_data.squeeze(-1).ndim == 2

    def test_spikes_are_conserved_when_nothing_is_dropped(self):
        spikes = _sparse_spikes(rate=5.0)
        ds = create_dataset(x_data=spikes, processor_type_x='spike',
                            processor_params_x={'bin_size': 0.05, 'window_size': 0.05,
                                                'normalize_bins': False,
                                                'drop_empty_windows': False})
        assert int(ds.x_data.sum()) == sum(len(s) for s in spikes)


class TestReachableThroughRun:
    """The flag has to survive parameter validation, not just reach the dataset.

    Regression: ``drop_empty_windows`` was wired into the spike datasets and the
    dataset factory but omitted from ``PROCESSOR_PARAMS_SCHEMA``, so it worked
    when ``create_dataset`` was called directly and was rejected by ``run()``,
    which is the only path a user takes.
    """

    def test_run_accepts_the_flag(self):
        import neural_mi as nmi
        spikes = _sparse_spikes(n_neurons=4, duration=60.0, rate=6.0, seed=0)
        other = _sparse_spikes(n_neurons=4, duration=60.0, rate=6.0, seed=1)
        r = nmi.run(
            spikes, other, mode='estimate',
            processing=nmi.Processing(
                x='spike', x_params={'window_size': 0.25, 'drop_empty_windows': False},
                y='spike', y_params={'window_size': 0.25, 'drop_empty_windows': False}),
            model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
            training=nmi.Training(n_epochs=1, patience=1),
            n_workers=1, seed=0, show_progress=False)
        assert np.isfinite(r.mi_estimate)
        assert r.details['window_retention'] == pytest.approx(1.0)

    def test_unknown_spike_param_still_rejected(self):
        import neural_mi as nmi
        spikes = _sparse_spikes(n_neurons=2, duration=20.0, rate=5.0)
        with pytest.raises(ValueError, match="Unknown parameters"):
            nmi.run(spikes, spikes, mode='estimate',
                    processing=nmi.Processing(x='spike',
                                              x_params={'window_size': 0.5, 'nonsense': 1}),
                    model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=nmi.Training(n_epochs=1, patience=1),
                    n_workers=1, show_progress=False)


class TestContinuousRuleIsIndependent:
    """The flag must not touch the missing-data rule on a continuous side."""

    def _gappy_continuous(self, duration=20.0, gap=(8.0, 14.0), dt=0.05):
        t = np.arange(0, duration, dt)
        t = t[(t < gap[0]) | (t > gap[1])]
        return np.sin(t)[:, None].astype('float32'), t

    def test_continuous_coverage_still_drops_unobserved_windows(self):
        values, times = self._gappy_continuous()
        ds = create_dataset(x_data=values, x_time=times, processor_type_x='continuous',
                            processor_params_x={'window_size': 1.0})
        assert ds.x_data.shape[0] < 20, "the gap should cost windows"

    def test_continuous_partner_masks_gaps_while_silence_is_kept(self):
        """The mixed case: two independent rules doing two different jobs."""
        spikes = _sparse_spikes(duration=20.0, rate=1.0)
        values, times = self._gappy_continuous()
        ds = create_dataset(
            x_data=spikes, y_data=values, y_time=times,
            processor_type_x='spike',
            processor_params_x={'window_size': 1.0, 'drop_empty_windows': False},
            processor_type_y='continuous',
            processor_params_y={'window_size': 1.0},
        )
        # Silence on X is retained, so any drop must come from Y's coverage rule.
        assert ds.window_retention < 1.0
        assert ds.x_data.shape[0] == ds.y_data.shape[0]


class TestRetentionReporting:

    def test_retention_attributes_are_always_present(self):
        spikes = _sparse_spikes()
        ds = create_dataset(x_data=spikes, processor_type_x='spike',
                            processor_params_x={'window_size': 0.5})
        assert 0.0 <= ds.window_retention <= 1.0
        assert ds.n_windows_retained <= ds.n_windows_built
        assert ds.n_windows_retained == ds.x_data.shape[0]

    def test_low_retention_warns_and_names_the_side(self, caplog):
        """Sparse firing against a fixed grid should trip the warning."""
        # The warning is deduped per process, so any earlier test that tripped
        # it would consume this one's. Reset first, or this passes or fails
        # depending on test order.
        from neural_mi.data.handler import reset_retention_warnings
        reset_retention_warnings()
        spikes = _sparse_spikes(n_neurons=1, duration=60.0, rate=0.2, seed=3)
        with caplog.at_level('WARNING'):
            create_dataset(x_data=spikes, processor_type_x='spike',
                           processor_params_x={'window_size': 0.2})
        msgs = [r.message for r in caplog.records if 'Window coverage validation kept' in r.message]
        assert msgs, "low retention should warn"
        assert 'drop_empty_windows' in msgs[0], "the warning should name the remedy"

    def _estimate(self, **training_kwargs):
        import neural_mi as nmi
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 2)).astype('float32')
        y = (0.8 * x + 0.3 * rng.normal(size=(600, 2))).astype('float32')
        return nmi.run(x, y, mode='estimate',
                       processing=nmi.Processing(x='continuous', y='continuous',
                                                 x_params={'window_size': 10},
                                                 y_params={'window_size': 10}),
                       model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                       training=nmi.Training(n_epochs=1, patience=1, **training_kwargs),
                       n_workers=1, seed=0, show_progress=False)

    @pytest.mark.parametrize('deferred', [False, True],
                             ids=['run-builds-dataset', 'windowing-deferred'])
    def test_retention_is_reported_from_either_windowing_path(self, deferred):
        """Windowing happens in the task or in ``run()``, and both must report.

        With shift active (the default) the task windows for itself. With it
        off, ``run()`` windows and hands the value down, since the task then
        receives already-windowed tensors and never sees that dataset.
        """
        r = self._estimate() if deferred else self._estimate(shift_windows=False)
        assert r.details.get('window_retention') == pytest.approx(1.0)

    def test_retention_is_per_task_not_per_run(self):
        """A sweep gets one retention per row, since it genuinely varies.

        Across a window_size sweep on spike data the retained subensemble
        grows from a fraction of the recording to all of it, so a single
        run-level scalar would misdescribe every row but one.
        """
        import neural_mi as nmi
        spikes_x = _sparse_spikes(n_neurons=4, duration=120.0, rate=3.0, seed=1)
        spikes_y = _sparse_spikes(n_neurons=4, duration=120.0, rate=3.0, seed=2)
        r = nmi.run(spikes_x, spikes_y, mode='sweep',
                    sweep_grid={'window_size': [0.05, 0.5]},
                    processing=nmi.Processing(x='spike', x_params={}, y='spike', y_params={}),
                    model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                    training=nmi.Training(n_epochs=1, patience=1),
                    n_workers=1, seed=0, show_progress=False)
        raw = r.details['raw_results']
        assert 'window_retention' in raw.columns
        by_size = raw.groupby('window_size')['window_retention'].first()
        assert by_size.loc[0.05] < by_size.loc[0.5], (
            "wider windows should retain more, which is exactly why one "
            "run-level number cannot describe the sweep"
        )


class TestMixedUnitWarning:
    """The spike-plus-regular-grid unit warning must not fire on correct usage.

    Regression: the check consulted only `sample_rate`, so a caller who
    supplied an explicit time vector, which already puts the regular-grid side
    into real seconds, still received a warning telling them their window
    alignment might be meaningless. That is the one case where the pairing is
    unambiguously correct.
    """

    def _warns(self, y_time=None, y_params=None):
        import io, logging
        from neural_mi.data.handler import create_dataset
        from neural_mi.logger import logger
        rng = np.random.default_rng(0)
        spikes = _sparse_spikes(n_neurons=3, duration=30.0, rate=5.0)
        n = 400
        values = rng.normal(size=(n, 1)).astype('float32')
        times = np.linspace(0, 30.0, n)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            create_dataset(x_data=spikes, y_data=values,
                           processor_type_x='spike',
                           processor_params_x={'window_size': 1.0},
                           processor_type_y='continuous',
                           processor_params_y=y_params or {'window_size': 1.0},
                           y_time=times if y_time else None)
        finally:
            logger.removeHandler(handler)
        return "mixes 'spike'" in buf.getvalue()

    def test_explicit_time_vector_suppresses_the_warning(self):
        assert not self._warns(y_time=True)

    def test_sample_rate_suppresses_the_warning(self):
        assert not self._warns(y_params={'window_size': 1.0, 'sample_rate': 13.3})

    def test_neither_still_warns(self):
        assert self._warns()
