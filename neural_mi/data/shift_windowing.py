# neural_mi/data/shift_windowing.py
"""The windowing behind a row plan's moves, and the rows a stride thins.

A :class:`~neural_mi.data.plan.RowPlan` of streams on a regular clock windows
each stream by sample number (`WindowShifter`, through `torch.Tensor.unfold`),
in its own samples, and `BundleShifter` keeps the streams in step: a shift is
drawn in the first stream's samples, within one step, and every other stream
moves by the same real time. Every epoch yields the same count of windows, so
the rows' indices, and every train, test and evaluation index computed against
them, stay valid from shift to shift, and only the content at each index
changes. Categorical streams are re-encoded after the reslice
(`make_categorical_encoder`, matching `CategoricalWindowDataset`'s three
encodings). A stream without a `sample_rate` takes its period from its time
vector (`stream_period`) when that clock is regular.

The rows a quantity builds by offset (the named quantities, transfer entropy,
`offset_quantity`) move under ``shift_windows`` when a ``stride`` above one
thins them: they are built at every step, and `RowPhaseShifter` keeps one in
every ``stride`` from a phase drawn each epoch. At ``stride=1`` they hold every
step already, and a shift by ``s`` would only relabel row ``i`` as row
``s + i``.

Evaluation never sees a shift. The Trainer measures `test_mi` and `train_mi`,
and everything derived from them, against a snapshot of the rows taken before
any shift.
"""
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import warnings
from neural_mi.data.rows import PHASES_KEY, kept_rows, record_rows, window_rows
from neural_mi.logger import user_stacklevel


def seconds_to_samples(value: float, period: float) -> int:
    """Convert a duration in the shared ``WindowManager`` time unit
    (seconds if `period` came from a `sample_rate`, otherwise raw samples
    if `period=1.0`) to an integer sample count for one side of a pair.

    Rounds to the nearest sample; always at least 1. Needed because
    `torch.Tensor.unfold` requires an integer element count, but
    `processor_params['window_size'/'step_size']` may be given in seconds
    (whenever `sample_rate` is set), passing that straight to `unfold`
    crashes with a confusing `TypeError` for any non-integer value.
    """
    return max(1, int(round(value / period)))


def resolve_step_size(window_size: float, step_size: Optional[float]) -> float:
    """Apply the step-size convention and return an absolute step in time units.

    - ``None``         : one whole window, so consecutive windows don't overlap.
    - ``0 < step < 1`` : a fraction of ``window_size``, so ``0.25`` means a 75%
      overlap at every window size.
    - ``step >= 1``    : already absolute, used as given.

    Shared with :meth:`neural_mi.data.handler.WindowManager.resolve_step` so
    both windowing paths read a step the same way. Taking every value as
    absolute would collapse a fractional step to ``max(1, round(step))`` = 1
    sample and give a much larger overlap than the caller asked for.
    """
    if step_size is None:
        return float(window_size)
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    if step_size < 1:
        return float(step_size) * float(window_size)
    return float(step_size)


def build_shifted_windows(raw: torch.Tensor, window_size: int, step_size: int,
                          shift: int, n_windows: int) -> torch.Tensor:
    """``(T, C)`` -> ``(n_windows, C, window_size)`` via ``unfold``, starting
    at raw sample ``shift``. A view-based slide, no interpolation, no
    time-vector, cost is independent of ``shift`` and negligible relative
    to a training epoch.
    """
    usable = raw[shift:]
    windows = usable.unfold(0, window_size, step_size)[:n_windows]  # (n_windows, C, window_size)
    return windows.contiguous()


def make_categorical_encoder(n_categories: int, encoding: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Vectorized majority_vote/probability/full_trajectory encoding for a
    ``(n_windows, n_channels, window_size)`` integer-label tensor, matching
    :class:`~neural_mi.data.temporal.CategoricalWindowDataset`'s three
    encoding modes, but with no Python per-window loop, safe here because
    reslice-based windows are always a regular, fully-populated grid (no
    ragged per-window counts to track, unlike the general `WindowManager`
    path that has to handle irregular time vectors).
    """
    if encoding not in ('majority_vote', 'probability', 'full_trajectory'):
        raise ValueError(
            f"Unknown encoding '{encoding}'. Expected 'majority_vote', "
            f"'probability' or 'full_trajectory'."
        )

    def _encode(raw_windows: torch.Tensor) -> torch.Tensor:
        # raw_windows: (n_windows, n_channels, window_size) integer labels.
        n_windows, n_channels, window_size = raw_windows.shape
        # one_hot: (n_windows, n_channels, window_size, n_categories)
        one_hot = F.one_hot(raw_windows.long(), num_classes=n_categories).float()
        if encoding == 'full_trajectory':
            # Position p -> columns [p*n_categories, (p+1)*n_categories),
            # matching CategoricalWindowDataset._move_full_trajectory's layout.
            return one_hot.reshape(n_windows, n_channels, window_size * n_categories)
        counts = one_hot.sum(dim=2)  # (n_windows, n_channels, n_categories)
        if encoding == 'probability':
            return counts / window_size
        # majority_vote: one-hot of the most frequent category.
        winner = counts.argmax(dim=-1)  # (n_windows, n_channels)
        return F.one_hot(winner, num_classes=n_categories).float()

    return _encode


def safe_n_windows(n_samples: int, window_size: int, step_size: int, max_shift: int = 0) -> int:
    """Windows of ``window_size`` every ``step_size`` that fit in ``n_samples`` at every shift up to
    ``max_shift`` samples.

    The count is the same at every epoch, whichever shift is drawn, so the
    rows' indices, and the train, test and evaluation rows computed against
    them, stay valid across every shift. Shifts stay within one step, so a
    plan that shifts passes ``step_size - 1``.
    """
    return max(0, (n_samples - max_shift - window_size) // step_size + 1)


class WindowShifter:
    """Holds one raw signal plus windowing params; re-derives windows for an
    arbitrary integer shift on demand.

    Parameters
    ----------
    raw : torch.Tensor
        Shape ``(T, n_channels)``, the unwindowed signal. Integer-dtype for
        categorical data (paired with `encoder`), float for continuous.
    window_size, step_size : int
        Already converted to this side's own raw-sample-count units (see
        `seconds_to_samples`), same meaning as
        ``Processing(..., x_params={'window_size':..., 'step_size':...})``
        when no `sample_rate` is set.
    encoder : callable, optional
        Applied to each freshly-resliced ``(n_windows, C, window_size)``
        raw tensor before it's returned, e.g. `make_categorical_encoder`'s
        output for categorical data. ``None`` for continuous data (used as-is).
    """

    def __init__(self, raw: torch.Tensor, window_size: int, step_size: int,
                encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, max_shift: int = 0):
        self.raw = raw
        self.window_size = window_size
        self.step_size = step_size
        self.encoder = encoder
        self.max_shift = max_shift
        self.n_windows = safe_n_windows(raw.shape[0], window_size, step_size, max_shift)
        if self.n_windows <= 0:
            raise ValueError(
                f"Not enough samples ({raw.shape[0]}) to build even one window "
                f"of size {window_size} with step {step_size}"
                + (f" across every shift in [0, {max_shift}]." if max_shift else ".")
            )

    def windows_at(self, shift: int) -> torch.Tensor:
        raw_windows = build_shifted_windows(self.raw, self.window_size, self.step_size,
                                            shift, self.n_windows)
        return self.encoder(raw_windows) if self.encoder is not None else raw_windows

    def random_shift(self, generator: torch.Generator = None) -> int:
        """A shift within one step: every distinct set of window boundaries, and windows that start
        one sample apart never move."""
        return int(torch.randint(0, max(1, self.step_size), (1,), generator=generator).item())

    @property
    def moves(self) -> bool:
        """Whether a shift can move the windows: they start more than one sample apart."""
        return self.step_size > 1


class RowPhaseShifter:
    """Rows built by offset at every step of their series, kept one in every ``phases`` from a phase
    drawn each epoch: the shift of rows that a stride thins.

    The rows of every phase are cut to the count every phase has, so every
    train, test and evaluation index stays valid from shift to shift, and a
    shift only changes which steps the rows start on. ``x`` may be a pair of
    row arrays (the joint term's X of a difference), each cut the same way.
    """

    def __init__(self, x, y, phases: int):
        self.x, self.y, self.phases = x, y, int(phases)
        n_rows = (x[0] if isinstance(x, tuple) else x).shape[0]
        self.n_windows = kept_rows(n_rows, {PHASES_KEY: self.phases})
        if self.n_windows <= 0:
            raise ValueError(
                f"Not enough rows ({n_rows}) to keep one in every {self.phases} at every phase."
            )

    def _take(self, data, shift: int):
        rows = slice(shift, shift + self.phases * (self.n_windows - 1) + 1, self.phases)
        return tuple(d[rows] for d in data) if isinstance(data, tuple) else data[rows]

    def windows_at(self, shift: int):
        return self._take(self.x, shift), self._take(self.y, shift)

    def random_shift(self, generator: torch.Generator = None) -> int:
        return int(torch.randint(0, self.phases, (1,), generator=generator).item())

    @property
    def moves(self) -> bool:
        """Whether a shift can move the rows: one in more than one step is kept."""
        return self.phases > 1


def try_build_row_phase_dataset(x_data, y_data, params: dict, data_device: str = 'cpu'):
    """The dataset of rows built at every step and kept one in every ``_row_phases``
    (:mod:`neural_mi.data.rows`), at phase 0 with a :class:`RowPhaseShifter` attached for
    ``shift_windows``, or ``None`` for rows built any other way."""
    from neural_mi.data.static import StaticDataset
    from neural_mi.data.handler import PairedDataset
    phases = int(params.get(PHASES_KEY) or 1)
    if phases <= 1:
        return None
    shifter = RowPhaseShifter(x_data, y_data, phases)
    x_rows, y_rows = shifter.windows_at(0)
    dataset = PairedDataset(StaticDataset(x_rows, data_device=data_device),
                            StaticDataset(y_rows, data_device=data_device))
    dataset._window_shifter = shifter
    return dataset


def stream_spec(raw, window_size, step_size, encoder=None, period=1.0):
    """One stream's entry for :class:`BundleShifter`.

    ``window_size``/``step_size`` are in this stream's own raw-sample units;
    ``period`` is 1/sample_rate, or 1.0 when the stream has no sample rate.
    """
    return dict(raw=raw, window_size=window_size, step_size=step_size,
                encoder=encoder, period=period)


class BundleShifter:
    """Any number of named streams on a regular clock, windowed together in real time.

    Each stream is windowed by sample number in its own samples, with its own
    period, window and step. A shift is drawn in the first (reference)
    stream's samples, within one step, and converted into every other stream's
    samples, so every stream moves by the same real time even when their
    sample rates differ. The streams are first truncated to a common duration,
    and every stream yields the smallest count of windows any of them reaches
    at every shift.

    Returns the windows by name. Packing them into the roles an estimator
    reads is the caller's job (``pack``).
    """

    def __init__(self, streams, pack=None, shift=False):
        """
        Parameters
        ----------
        streams : mapping of str to dict
            One entry per stream, in order, the first being the reference. Each
            value takes ``raw``, ``window_size``, ``step_size``, and optionally
            ``encoder`` and ``period`` (default 1.0). ``window_size``/
            ``step_size`` are already in that stream's own raw-sample units.
        pack : callable, optional
            Maps the name-keyed windows to whatever the caller wants back. The
            default returns them by name.
        shift : bool, optional
            Whether the windows will move. Each stream then keeps one step at
            its end for the shift.
        """
        streams = OrderedDict(streams)
        if not streams:
            raise ValueError("BundleShifter needs at least one stream.")
        periods = OrderedDict((name, spec.get('period', 1.0)) for name, spec in streams.items())
        duration = min(spec['raw'].shape[0] * periods[name] for name, spec in streams.items())
        self.shifters = OrderedDict()
        for name, spec in streams.items():
            period = periods[name]
            # A duration read back in samples, with the tolerance a measured
            # period needs (1000 * 0.001 / 0.001 is 999.999...).
            n_keep = min(spec['raw'].shape[0], int(np.floor(duration / period + 1e-6)))
            self.shifters[name] = WindowShifter(
                spec['raw'][:n_keep], spec['window_size'], spec['step_size'],
                spec.get('encoder'), max_shift=spec['step_size'] - 1 if shift else 0,
            )
        self.periods = periods
        self.reference = next(iter(self.shifters))
        self.n_windows = min(sh.n_windows for sh in self.shifters.values())
        for shifter in self.shifters.values():
            shifter.n_windows = self.n_windows
        self._pack = pack

    def windows_at(self, shift):
        """Reslice every stream at ``shift``, given in the reference's samples."""
        reference_period = self.periods[self.reference]
        out = OrderedDict()
        for name, shifter in self.shifters.items():
            own_shift = shift if name == self.reference else int(round(shift * reference_period
                                                                        / self.periods[name]))
            out[name] = shifter.windows_at(max(0, min(own_shift, shifter.max_shift)))
        return self._pack(out) if self._pack is not None else out

    def random_shift(self, generator: torch.Generator = None) -> int:
        return self.shifters[self.reference].random_shift(generator)

    @property
    def moves(self) -> bool:
        """Whether a shift can move the windows, drawn in the reference stream's samples."""
        return self.shifters[self.reference].moves


def _prep_shift_side(data, proc_type, proc_params):
    """One raw stream as ``(raw_tensor, encoder_or_None)`` for a :class:`WindowShifter`: a
    categorical stream relabelled to consecutive codes, as ``CategoricalWindowDataset`` relabels it,
    with the encoder of its encoding, and a continuous one as floats."""
    if proc_type == 'categorical':
        from neural_mi.data.temporal import relabel_categorical_data
        raw = torch.as_tensor(relabel_categorical_data(data), dtype=torch.long)
        n_categories = int(raw.max().item()) + 1 if raw.numel() > 0 else 1
        return raw, make_categorical_encoder(n_categories, proc_params.get('encoding', 'majority_vote'))
    raw = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data), dtype=torch.float32)
    if raw.ndim == 1:
        raw = raw.unsqueeze(-1)
    return raw, None


def stream_period(proc_params, time_vector):
    """Seconds per sample for one stream, or ``None`` if the reslice route
    cannot serve it.

    ``sample_rate`` states the period outright. A time vector states it too,
    as long as the clock is regular: this route indexes a raw array by sample
    number and converts a window size into a sample count once, up front, so a
    clock with a gap in it has no single period that holds across the
    recording. Such a stream belongs on the eager route, where each window's
    coverage is checked against the timestamps themselves and the ones that
    straddle a gap are dropped. A stream with neither is already in raw sample
    units, period 1.
    """
    rate = (proc_params or {}).get('sample_rate')
    if rate:
        return 1.0 / rate
    if time_vector is None:
        return 1.0
    t = np.asarray(time_vector, dtype=float).ravel()
    if t.size < 2:
        return 1.0
    steps = np.diff(t)
    period = float(np.median(steps))
    if period <= 0 or not np.all(np.abs(steps - period) <= 1e-6 * period):
        return None
    return period
