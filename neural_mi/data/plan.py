# neural_mi/data/plan.py
"""One row plan per quantity: every stream a quantity reads, on one grid.

A quantity's rows are the windows its processors cut from the streams it
reads: where the grid starts and ends, its one window and step, the windows
kept, the fits to the data and the offset each epoch moves them by. A
:class:`RowPlan` holds the streams with their settings resolved once, and
every task builds its rows from the whole plan. A term of a difference reads
some of the plan's streams (the conditioning term of conditional MI reads W
and Y) and only selects them (:class:`Roles`). Its rows, the windows kept, the
fits (each stream's category count, bursty exclusion and spike slot width) and
the shifts are those of every other term, because they come from the same
streams.

Two routes cut the rows. Streams on a regular clock (continuous or
categorical, with a ``sample_rate``, a regular time vector or neither) that
start together are windowed by sample number, each in its own samples, and
every full window is a row. Any other plan (a spike stream, an irregular
clock, clocks that start apart) puts every stream on one grid in time, which
starts at the latest start among the streams and ends at the earliest end, and
keeps a window only where every stream is valid. ``drop_empty_windows``, on by
default, makes a silent spike window invalid, so the rows are then the windows
where every spike stream fired.

Under ``Training(shift_windows=True)`` the rows stay the same rows and their
content moves: each epoch draws one offset within one step, and every stream's
windows start that much later. The grid leaves one step at its end for it. On
the grid in time, a row whose moved window fails a stream's rule (a silent
spike window, a continuous window across a gap) keeps its unmoved windows in
every stream for that epoch. Evaluation reads the unmoved rows.
"""
import hashlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from neural_mi.defaults import PROCESSOR_DEFAULTS, processor_setting

REGULAR = ('continuous', 'categorical')


@dataclass(frozen=True)
class Stream:
    """One stream of a plan: its data, its clock and its processor with every setting resolved.

    A stream without a processor holds rows already built, ``(N, C, W)`` or ``(N, C)``.
    """
    data: Any
    time: Any = None
    processor: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


def resolve_stream(data, time=None, processor=None, params=None) -> Stream:
    """A stream with every setting its processor takes, from ``params`` or the schema's default."""
    if processor is None:
        return Stream(data, time, None, dict(params or {}))
    if processor not in PROCESSOR_DEFAULTS:
        raise ValueError(f"Unknown processor type: '{processor}'.")
    return Stream(data, time, processor,
                  {name: processor_setting(processor, params, name) for name in PROCESSOR_DEFAULTS[processor]})


@dataclass(frozen=True)
class Roles:
    """Which of a plan's streams a term reads as X and as Y. Two X streams travel as the pair
    ``(rest of X, conditioning set)``, the joint term's X of a difference."""
    x: Tuple[str, ...] = ('x',)
    y: str = 'y'


PAIR = Roles()


class RowPlan:
    """The rows of one quantity: every stream it reads, on one grid.

    Parameters
    ----------
    streams : mapping of str to Stream
        The streams in order, X first. The grid's window and step are those
        the streams name (:func:`neural_mi.data.handler.one_grid`), and a stream
        asking for others is refused.
    shift : bool, optional
        Whether the rows move each epoch. The grid then leaves one step at its
        end.
    times : array-like, optional
        The rows' start times, given instead of derived: a chunk of a larger
        plan (:meth:`cut`) keeps its parent's rows.
    """

    def __init__(self, streams, *, shift: bool = False, times=None, identity=None):
        from neural_mi.data.handler import one_grid
        from neural_mi.data.shift_windowing import resolve_step_size
        self.streams = OrderedDict(streams)
        if not self.streams:
            raise ValueError("A row plan needs at least one stream.")
        self.shift = bool(shift)
        self._times = None if times is None else np.asarray(times, dtype=float)
        self._identity = identity
        self._retention = None
        self._rows = None
        windowed = OrderedDict((name, s.params) for name, s in self.streams.items() if s.processor is not None)
        if windowed and len(windowed) != len(self.streams):
            raise ValueError(
                f"A pre-processed stream (processor_type=None) cannot be paired with a windowed "
                f"processor_type. {sorted(set(self.streams) - set(windowed))} are pre-processed and "
                f"{sorted(windowed)} are windowed. A pre-processed stream has no time axis to align "
                f"windows against and cannot share a grid with a windowed one."
            )
        self.window = self.step = self.step_size = None
        if windowed:
            self.window, self.step_size = one_grid(windowed)
            if self.window is None:
                raise ValueError("A windowed processor needs a window_size in at least one stream's "
                                 "processor_params.")
            self.step = resolve_step_size(self.window, self.step_size)

    @classmethod
    def of_pair(cls, x_data, y_data, params: Dict[str, Any]) -> 'RowPlan':
        """The plan of an X and a Y read with the processors and settings in a task's ``params``. A Y
        without a processor or settings of its own reads with X's."""
        from neural_mi.defaults import setting
        kind_x = params.get('processor_type_x')
        kind_y = params.get('processor_type_y') or kind_x
        params_x = params.get('processor_params_x') or {}
        params_y = params.get('processor_params_y')
        streams = OrderedDict(x=resolve_stream(x_data, params.get('x_time'), kind_x, params_x))
        if y_data is not None:
            streams['y'] = resolve_stream(y_data, params.get('y_time'), kind_y,
                                          params_y if params_y is not None else params_x)
        return cls(streams, shift=setting(params, 'shift_windows'))

    def geometry(self):
        """How the rows lie in time (:mod:`neural_mi.data.rows`), or ``None`` when that is not known."""
        from neural_mi.data.rows import RowGeometry
        if self.window is not None:
            return RowGeometry(float(self.step), float(self.window))
        identity = self._identity or {}
        if identity.get('step') is not None and identity.get('spans'):
            return RowGeometry(float(identity['step']), float(max(identity['spans'].values())))
        return None

    # ------------------------------------------------------------------
    # What the plan is
    # ------------------------------------------------------------------

    @property
    def route(self) -> str:
        """``'rows'`` for rows built already, ``'unfold'`` for streams on a regular clock that start
        together, ``'grid'`` for everything else."""
        from neural_mi.data.shift_windowing import stream_period
        kinds = {s.processor for s in self.streams.values()}
        if kinds == {None}:
            return 'rows'
        if not kinds <= set(REGULAR):
            return 'grid'
        periods, starts = [], []
        for s in self.streams.values():
            period = stream_period(s.params, s.time)
            if period is None:
                return 'grid'
            periods.append(period)
            starts.append(float(np.asarray(s.time, dtype=float).ravel()[0])
                          if s.time is not None and len(np.asarray(s.time)) else 0.0)
        return 'grid' if max(starts) - min(starts) > 0.5 * max(periods) else 'unfold'

    def __len__(self) -> int:
        return len(self.rows()['times'])

    def identity(self) -> Dict[str, Any]:
        """The rows as :func:`neural_mi.data.rows.row_identity` reports them: their count, the grid's
        origin and step, each stream's window and a digest of the rows' start times."""
        if self._identity is not None:
            return dict(self._identity)
        rows = self.rows()
        times = rows['times']
        return {'count': len(times), 'origin': rows['origin'], 'step': self.step,
                'spans': dict(rows['spans']),
                'kept': hashlib.md5(np.round(np.asarray(times, dtype=float), 9).tobytes()).hexdigest()[:16]}

    # ------------------------------------------------------------------
    # The rows
    # ------------------------------------------------------------------

    def rows(self) -> Dict[str, Any]:
        """The rows at offset 0, built once per process: ``windows`` by stream name, ``times`` (each
        row's start), ``origin``, ``spans``, ``retention`` and, for a plan that shifts, ``shifter``."""
        if self._rows is None:
            self._rows = {'rows': self._rows_given, 'unfold': self._rows_unfold,
                          'grid': self._rows_grid}[self.route]()
        return self._rows

    def __getstate__(self):
        # A task builds the rows in its own process. Rows already built are
        # sent along only for a plan that holds them as its data.
        state = dict(self.__dict__)
        state['_rows'] = None
        return state

    def _rows_given(self) -> Dict[str, Any]:
        from neural_mi.utils import float_tensor, with_window_axis
        windows = OrderedDict()
        for name, s in self.streams.items():
            data = s.data if isinstance(s.data, tuple) else with_window_axis(float_tensor(s.data))
            windows[name] = data
        counts = {name: (w[0] if isinstance(w, tuple) else w).shape[0] for name, w in windows.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"The streams' rows do not line up: {counts}. Rows built before the call must "
                f"come one per row of every other stream. Pass the streams raw, with a processor, "
                f"to put them on one grid."
            )
        n = next(iter(counts.values()))
        return {'windows': windows, 'times': np.arange(n, dtype=float), 'origin': None, 'spans': {},
                'retention': self._retention, 'shifter': None}

    def _rows_unfold(self) -> Dict[str, Any]:
        from neural_mi.data.shift_windowing import (BundleShifter, _prep_shift_side, seconds_to_samples,
                                                    stream_period, stream_spec)
        specs = OrderedDict()
        for name, s in self.streams.items():
            period = stream_period(s.params, s.time)
            raw, encoder = _prep_shift_side(s.data, s.processor, s.params)
            specs[name] = stream_spec(raw, seconds_to_samples(self.window, period),
                                      seconds_to_samples(self.step, period), encoder, period)
        shifter = BundleShifter(specs, shift=self.shift)
        if shifter.n_windows <= 0:
            raise ValueError(f"The streams hold no window of {self.window} with a step of {self.step}.")
        first = next(iter(self.streams.values()))
        origin = (float(np.asarray(first.time, dtype=float).ravel()[0])
                  if first.time is not None and len(np.asarray(first.time)) else 0.0)
        reference = next(iter(specs.values()))
        step = reference['step_size'] * reference['period']
        n = shifter.n_windows
        return {'windows': shifter.windows_at(0), 'times': origin + np.arange(n) * step,
                'origin': origin, 'spans': {name: spec['window_size'] * spec['period']
                                            for name, spec in specs.items()},
                'retention': (n, n, 1.0), 'shifter': shifter if self.shift else None}

    def _rows_grid(self) -> Dict[str, Any]:
        from neural_mi.data.handler import WindowManager, create_single_dataset, report_retention
        datasets = OrderedDict((name, create_single_dataset(s.data, s.time, s.processor, s.params))
                               for name, s in self.streams.items())
        manager = WindowManager(self.window, step_size=self.step_size)
        if self._times is not None:
            times, origin = self._times, (float(self._times[0]) if len(self._times) else None)
        else:
            extents = [ds.get_temporal_extent() for ds in datasets.values()]
            origin = float(max(e[0] for e in extents))
            end = float(min(e[1] for e in extents))
            if origin >= end:
                raise ValueError(
                    f"The streams share no stretch of time: the latest starts at {origin:.6g} and "
                    f"the earliest ends at {end:.6g}."
                )
            # As WindowManager.create_windows, with one step left at the end
            # for the shifts.
            times = np.arange(origin, end - (self.step if self.shift else 0.0), self.step)
        _attach(datasets, manager, times)
        retention = None
        if self._times is None:
            valid = OrderedDict((name, ds.validate_window_coverage()) for name, ds in datasets.items())
            kept = np.logical_and.reduce(list(valid.values()))
            retention = report_retention(valid, kept)
            if not kept.any():
                raise ValueError("No valid windows after checking data coverage")
            times = times[kept]
            # Each processor keeps its own rows and reports what it keeps
            # (a continuous window mostly bridged across a gap, for one).
            manager.valid_windows = kept
            for ds in datasets.values():
                ds.remove_invalid_windows()
            manager.window_times, manager.n_windows = times, len(times)
            manager.valid_windows = np.ones(len(times), dtype=bool)
            windows = OrderedDict((name, ds.data) for name, ds in datasets.items())
        else:
            windows = OrderedDict((name, ds.data) for name, ds in datasets.items())
        shifter = GridShifter(datasets, manager, times, windows, self.step) if self.shift else None
        return {'windows': windows, 'times': times, 'origin': origin,
                'spans': {name: float(self.window) for name in datasets},
                'retention': retention, 'shifter': shifter}

    # ------------------------------------------------------------------
    # What a task does with it
    # ------------------------------------------------------------------

    def build(self, roles: Roles = PAIR, data_device: str = 'cpu'):
        """The dataset a term trains on: its roles' rows at offset 0, with a shifter attached when the
        plan shifts. The dataset carries the rows' identity and retention."""
        from neural_mi.data.handler import PairedDataset
        from neural_mi.data.static import StaticDataset
        rows = self.rows()
        x, y = pack(rows['windows'], roles)
        dataset = PairedDataset(StaticDataset(x, data_device=data_device),
                                StaticDataset(y, data_device=data_device))
        if rows['shifter'] is not None:
            dataset._window_shifter = RoleShifter(rows['shifter'], roles)
        dataset._row_identity = self.identity()
        if rows['retention'] is not None:
            dataset.n_windows_built, dataset.n_windows_retained, dataset.window_retention = rows['retention']
        return dataset

    def materialize(self) -> 'RowPlan':
        """The same rows, built once: a plan whose streams hold their rows at offset 0."""
        rows = self.rows()
        streams = OrderedDict(
            (name, Stream(window, None, None, {}))
            for name, window in rows['windows'].items())
        plan = RowPlan(streams, identity=self.identity())
        plan._retention = rows['retention']
        return plan

    def cut(self, rows) -> 'RowPlan':
        """The plan of the rows at indices ``rows``: a chunk of a rigorous ladder. Rows built already
        are taken as indexed. The rows of raw streams form one contiguous stretch, and the streams are
        cut to the stretch those rows and their shifts cover."""
        from neural_mi.data.shift_windowing import seconds_to_samples, stream_period
        rows = np.asarray(rows, dtype=int)
        if self.route == 'rows':
            windows = self.rows()['windows']
            return RowPlan(OrderedDict((name, Stream(_take_rows(windows[name], rows), None, None, s.params))
                                       for name, s in self.streams.items()))
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        if hi - lo != len(rows) or np.any(np.diff(rows) != 1):
            raise ValueError("A chunk of a plan of raw streams is one contiguous stretch of its rows.")
        times = self.rows()['times'][lo:hi]
        start, end = float(times[0]), float(times[-1]) + self.window + (self.step if self.shift else 0.0)
        if self.route == 'unfold':
            origin = self.rows()['origin']
            streams = OrderedDict()
            for name, s in self.streams.items():
                period = stream_period(s.params, s.time)
                a = int(round((start - origin) / period))
                b = a + seconds_to_samples(end - start, period)
                time = None if s.time is None else np.asarray(s.time)[a:b]
                streams[name] = Stream(_take_rows(s.data, slice(a, b)), time, s.processor, s.params)
            return RowPlan(streams, shift=self.shift)
        streams = OrderedDict((name, Stream(_cut_in_time(s, start, end), _cut_time(s, start, end),
                                            s.processor, s.params))
                              for name, s in self.streams.items())
        return RowPlan(streams, shift=self.shift, times=times)

    def replace(self, name: str, data) -> 'RowPlan':
        """The plan with one stream's data replaced and the grid kept: a permutation trial's moved X."""
        streams = OrderedDict(self.streams)
        streams[name] = Stream(data, streams[name].time, streams[name].processor, streams[name].params)
        if self.route == 'grid':
            return RowPlan(streams, shift=self.shift, times=self.rows()['times'])
        return RowPlan(streams, shift=self.shift)


@dataclass(frozen=True)
class TermRows:
    """What a term's task receives: a plan and the streams its term reads."""
    plan: RowPlan
    roles: Roles = PAIR

    def dataset(self, data_device: str = 'cpu'):
        return self.plan.build(self.roles, data_device)


def count_rows(x_data, y_data, params: Dict[str, Any]) -> int:
    """How many rows a task builds from ``x_data`` and ``y_data`` with ``params``: the rows of their plan
    when a processor windows them, otherwise the rows given."""
    if params.get('processor_type_x') is not None and getattr(x_data, 'ndim', None) != 3:
        return len(RowPlan.of_pair(x_data, y_data, params))
    return x_data.shape[0] if hasattr(x_data, 'shape') else len(x_data)


# ---------------------------------------------------------------------------
# Packing and moving
# ---------------------------------------------------------------------------

def pack(windows, roles: Roles):
    """A term's ``(x, y)`` from the rows by stream name: one X stream as it is, two as a pair."""
    parts = tuple(windows[name] for name in roles.x)
    return (parts[0] if len(parts) == 1 else parts), windows[roles.y]


class RoleShifter:
    """A plan's shifter, packed into a term's roles for the trainer."""

    def __init__(self, shifter, roles: Roles):
        self.shifter, self.roles = shifter, roles

    def random_shift(self, generator: Optional[torch.Generator] = None):
        return self.shifter.random_shift(generator)

    def windows_at(self, shift):
        return pack(self.shifter.windows_at(shift), self.roles)


class GridShifter:
    """Moves the rows of a plan on a grid in time: every stream windowed again at the rows' starts plus
    one offset. A row whose moved window fails a stream's rule keeps its unmoved windows in every
    stream."""

    def __init__(self, datasets, manager, times, base, step):
        self.datasets, self.manager, self.times, self.base, self.step = datasets, manager, times, base, step

    def random_shift(self, generator: Optional[torch.Generator] = None) -> float:
        draw = (float(torch.rand(1, generator=generator).item()) if generator is not None
                else float(np.random.uniform()))
        return draw * self.step

    def windows_at(self, shift: float):
        if not shift:
            return OrderedDict((name, window.clone()) for name, window in self.base.items())
        # What windowing the rows reports was reported at offset 0.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _attach(self.datasets, self.manager, self.times + shift, refit=False)
        valid = np.ones(len(self.times), dtype=bool)
        moved = OrderedDict()
        for name, ds in self.datasets.items():
            moved[name] = ds.data
            ok = ds.validate_window_coverage()
            if ok is not None:
                valid &= np.asarray(ok, dtype=bool)
        if not valid.all():
            stay = torch.as_tensor(~valid)
            for name in moved:
                moved[name] = moved[name].clone()
                moved[name][stay] = self.base[name][stay]
        return moved


def _attach(datasets, manager, times, refit: bool = True) -> None:
    """Put every dataset on the windows starting at ``times`` and window it."""
    manager.window_times = np.asarray(times, dtype=float)
    manager.n_windows = len(manager.window_times)
    manager.valid_windows = np.ones(manager.n_windows, dtype=bool)
    # The span the windows cover, which a processor may read to select its
    # samples (CategoricalWindowDataset).
    if manager.n_windows:
        manager.t_start = float(manager.window_times[0])
        manager.t_end = float(manager.window_times[-1]) + manager.resolve_step()
    for ds in datasets.values():
        if refit:
            ds.set_window_manager(manager)
        ds.move_data_to_windows()


def _take_rows(data, rows):
    if isinstance(data, tuple):
        return tuple(_take_rows(part, rows) for part in data)
    if torch.is_tensor(data) and not isinstance(rows, slice):
        return data[torch.as_tensor(rows)]
    return data[rows]


def _cut_in_time(stream: Stream, start: float, end: float):
    """A raw stream's data in ``[start, end)``, in its own time."""
    if stream.processor == 'spike':
        out = []
        for neuron in stream.data:
            neuron = np.asarray(neuron)
            out.append(neuron[np.searchsorted(neuron, start, side='left'):
                              np.searchsorted(neuron, end, side='left')])
        return out
    time = _clock(stream)
    a, b = np.searchsorted(time, start, side='left'), np.searchsorted(time, end, side='left')
    return stream.data[a:b]


def _cut_time(stream: Stream, start: float, end: float):
    if stream.processor == 'spike':
        return stream.time
    time = _clock(stream)
    a, b = np.searchsorted(time, start, side='left'), np.searchsorted(time, end, side='left')
    return time[a:b]


def _clock(stream: Stream) -> np.ndarray:
    """A regular stream's time vector, as its processor reads it."""
    n = len(stream.data)
    if stream.time is not None:
        return np.asarray(stream.time, dtype=float)
    rate = stream.params.get('sample_rate')
    return np.arange(n) / float(rate) if rate else np.arange(n, dtype=float)
