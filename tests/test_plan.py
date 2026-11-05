"""One row plan per quantity (neural_mi/data/plan.py).

A quantity's rows (the window times, the windows kept, the fits to the data and
the shifts) come from one plan of every stream it reads. Each term of a
difference selects its streams from the plan, so every term reads the same
rows. These tests hold the claims the plan makes: one grid over every stream,
settings resolved once, fits made per stream and the same in every term, one
offset per epoch for every stream, and one window and one step per grid.
"""
import warnings
from collections import OrderedDict

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.data.plan import PAIR, Roles, RowPlan, Stream, resolve_stream


def _spikes(rng, start, end=120.0, rate=5.0, n=3):
    return [np.sort(rng.uniform(start, end, rng.poisson(rate * (end - start)))) for _ in range(n)]


def _difference_plan(x, w, y, shift=True, **w_params):
    spike = {'window_size': 0.5}
    return RowPlan(OrderedDict(x=resolve_stream(x, None, 'spike', spike),
                               w=resolve_stream(w, None, 'spike', {**spike, **w_params}),
                               y=resolve_stream(y, None, 'spike', spike)), shift=shift)


class TestOneGridForEveryStream:

    def test_the_grid_starts_where_the_latest_stream_starts(self):
        rng = np.random.default_rng(0)
        x, y, w = _spikes(rng, 0.0), _spikes(rng, 0.0), _spikes(rng, 40.0)
        plan = _difference_plan(x, w, y)
        assert plan.rows()['origin'] == pytest.approx(min(s[0] for s in w))

    @pytest.mark.parametrize('shift', [False, True])
    def test_every_term_reads_the_same_rows(self, shift):
        rng = np.random.default_rng(1)
        x, y, w = _spikes(rng, 0.0), _spikes(rng, 0.0), _spikes(rng, 40.0)
        plan = _difference_plan(x, w, y, shift=shift)
        joint = plan.build(Roles(x=('x', 'w')))
        conditioning = plan.build(Roles(x=('w',)))
        alone = plan.build(Roles(x=('x',)))
        assert joint._row_identity == conditioning._row_identity == alone._row_identity
        assert len(joint) == len(conditioning) == len(alone) == len(plan)

    def test_a_silent_window_of_any_stream_drops_the_row_of_every_term(self):
        """drop_empty_windows, on by default, keeps a row only where every spike stream fired."""
        rng = np.random.default_rng(2)
        x, y = _spikes(rng, 0.0, rate=20.0), _spikes(rng, 0.0, rate=20.0)
        w = _spikes(rng, 0.0, rate=0.2)       # silent in most windows
        plan = _difference_plan(x, w, y, shift=False)
        spikes = np.sort(np.concatenate(w))
        for start in plan.rows()['times']:
            assert np.any((spikes >= start) & (spikes < start + 0.5))
        assert plan.rows()['retention'][2] < 0.5

    def test_the_terms_moved_rows_stay_together(self):
        """One offset per epoch moves every stream, in every term."""
        rng = np.random.default_rng(3)
        x, y, w = _spikes(rng, 0.0), _spikes(rng, 0.0), _spikes(rng, 0.0)
        plan = _difference_plan(x, w, y)
        joint = plan.build(Roles(x=('x', 'w')))
        conditioning = plan.build(Roles(x=('w',)))
        for seed in range(3):
            g1, g2 = torch.Generator(), torch.Generator()
            g1.manual_seed(seed)
            g2.manual_seed(seed)
            (_, w_joint), y_joint = joint._window_shifter.windows_at(joint._window_shifter.random_shift(g1))
            w_alone, y_alone = conditioning._window_shifter.windows_at(
                conditioning._window_shifter.random_shift(g2))
            assert torch.equal(w_joint, w_alone) and torch.equal(y_joint, y_alone)


class TestSettingsResolvedOnce:

    def test_y_reads_with_xs_settings_in_every_task(self):
        """A Y with a processor of its own and no settings reads with X's, inside a task as in run()."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2000, 2))
        params = {'processor_type_x': 'continuous', 'processor_type_y': 'categorical',
                  'processor_params_x': {'window_size': 0.2, 'sample_rate': 100.0}, 'shift_windows': True}
        plan = RowPlan.of_pair(x, rng.integers(0, 3, (2000, 1)), params)
        assert plan.streams['y'].params['sample_rate'] == 100.0
        assert plan.rows()['windows']['y'].shape[0] == len(plan)

    def test_every_setting_is_filled_from_the_schema(self):
        from neural_mi.defaults import PROCESSOR_DEFAULTS
        stream = resolve_stream([np.array([0.1, 0.4])], None, 'spike', {'window_size': 0.5})
        assert set(stream.params) == set(PROCESSOR_DEFAULTS['spike'])
        assert stream.params['drop_empty_windows'] is True and stream.params['window_size'] == 0.5

    def test_w_reads_with_its_own_settings(self, monkeypatch):
        """Interaction information reads W with W's settings, not X's."""
        seen = []
        real_init = RowPlan.__init__

        def spy(self, streams, **kwargs):
            seen.append(OrderedDict(streams))
            real_init(self, streams, **kwargs)
        monkeypatch.setattr(RowPlan, '__init__', spy)
        rng = np.random.default_rng(0)
        x, y = rng.standard_normal((600, 1)), rng.standard_normal((600, 1))
        w = rng.integers(0, 3, (600, 1))
        processing = nmi.Processing(x='continuous', x_params={'window_size': 10}, y='continuous',
                                    w='categorical', w_params={'encoding': 'probability'})
        nmi.run(x, y, mode='interaction', interaction=nmi.Interaction(w_data=w), processing=processing,
                estimator='ksg', show_progress=False)
        assert seen[0]['w'].params['encoding'] == 'probability'

    @pytest.mark.parametrize('key, value', [('window_size', 20), ('step_size', 5)])
    def test_a_stream_asking_for_another_window_or_step_is_refused(self, key, value):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1000, 2))
        with pytest.raises(ValueError, match=f"Stream 'w' asks for {key}={value}"):
            RowPlan(OrderedDict(x=resolve_stream(x, None, 'continuous', {'window_size': 10}),
                                w=resolve_stream(x, None, 'continuous', {key: value}),
                                y=resolve_stream(x, None, 'continuous', {})))


class TestFitsPerStream:

    def test_the_bursty_exclusion_of_w_is_the_same_in_every_term(self):
        """Each stream is its own dataset, so W's bursty neurons are judged among W's alone, the same
        way in the joint term as in the conditioning term."""
        rng = np.random.default_rng(4)
        x, y = _spikes(rng, 0.0), _spikes(rng, 0.0)
        w = _spikes(rng, 0.0) + [np.sort(rng.uniform(0, 120, 40000))]   # one bursty neuron
        plan = _difference_plan(x, w, y, shift=False, exclude_bursty_neurons=True)
        joint = plan.build(Roles(x=('x', 'w')))
        conditioning = plan.build(Roles(x=('w',)))
        assert joint.x_data[1].shape == conditioning.x_data.shape
        assert conditioning.x_data.shape[1] == 3      # the bursty neuron is left out
        assert torch.equal(joint.x_data[1], conditioning.x_data)


class TestMovingRows:

    def test_a_step_longer_than_the_window_moves_every_stream_together(self):
        rng = np.random.default_rng(5)
        x = np.cumsum(rng.standard_normal((6000, 1)), axis=0)
        plan = RowPlan(OrderedDict(x=resolve_stream(x, None, 'continuous', {'window_size': 10, 'step_size': 30}),
                                   y=resolve_stream(x.copy(), None, 'continuous', {})), shift=True)
        dataset = plan.build(PAIR)
        for shift in (0, 11, 29):
            xs, ys = dataset._window_shifter.windows_at(shift)
            assert torch.equal(xs, ys)

    def test_a_row_whose_moved_window_fails_keeps_its_unmoved_windows(self):
        """On the grid in time a row that would fall silent at the drawn offset keeps its windows at
        offset 0 in every stream, so a silent window never reaches training."""
        # One spike 0.1 s into each second: the kept rows are the windows that open on it, and every
        # offset within one step moves them onto silence.
        x = [np.array([float(t) + 0.1 for t in range(100)])]
        plan = RowPlan(OrderedDict(x=resolve_stream(x, None, 'spike', {'window_size': 0.5}),
                                   y=resolve_stream([x[0].copy()], None, 'spike', {})), shift=True)
        rows = plan.rows()
        moved = rows['shifter'].windows_at(0.4)
        assert torch.equal(moved['x'], rows['windows']['x'])


def test_drop_empty_windows_has_no_effect_on_rows_built_by_offset():
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.uniform(0, 100, 400))]
    processing = nmi.Processing(x='spike', x_params={'window_size': 0.1, 'bin_size': 0.1,
                                                     'drop_empty_windows': True})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            nmi.active_information_storage(spikes, k=2, processing=processing, estimator='ksg',
                                           show_progress=False)
        except Exception:
            pass
    assert any("drop_empty_windows=True has no effect on stream 'x' here" in str(w.message) for w in caught)


@pytest.mark.parametrize('window, step', [(1, None), (5, 5), (7, 3), (7, None), (10, None)])
def test_every_route_cuts_the_same_windows_from_the_same_samples(window, step):
    """A window of a stream sampled on a regular clock lies within its samples, on every route: the
    plan's unfold and grid routes, create_dataset, and the one-step grid of a quantity defined by
    offsets. A one-sample window on the last sample is cut, and a window that would run past the last
    sample is not, whatever min_coverage_fraction allows inside the recording."""
    from neural_mi.analysis.offsets import one_step_series
    from neural_mi.data.handler import create_dataset
    x = np.arange(1200, dtype='float32')[:, None]
    params = {'window_size': window} if step is None else {'window_size': window, 'step_size': step}
    stride = step or window
    expected = (1200 - window) // stride + 1
    unfold = RowPlan(OrderedDict(x=resolve_stream(x, None, 'continuous', params),
                                 y=resolve_stream(x, None, 'continuous', params)))
    built = create_dataset(OrderedDict(x=dict(data=x, processor_type='continuous', processor_params=params),
                                       y=dict(data=x, processor_type='continuous', processor_params=params)))
    assert unfold.route == 'unfold' and len(unfold) == len(built.stream('x').data) == expected
    # Y's samples start 0.6 later: the grid route, on the stretch both cover.
    t = np.arange(1200.0)
    grid = RowPlan(OrderedDict(x=resolve_stream(x, t, 'continuous', params),
                               y=resolve_stream(x, t + 0.6, 'continuous', params)))
    assert grid.route == 'grid' and len(grid) == int((1200 - window - 0.6) // stride) + 1
    if window == 1:
        stream = dict(time=None, processor_type='continuous', processor_params={'window_size': 1})
        series = one_step_series(OrderedDict(x=dict(stream, data=x), y=dict(stream, data=x)), 'test')
        assert len(series['x']) == 1200
