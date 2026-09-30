# neural_mi/data/handler.py
import warnings

import numpy as np
from collections import OrderedDict
from collections.abc import Mapping

from .temporal import (
    ContinuousWindowDataset, SpikeWindowDataset, BinnedSpikeDataset, CategoricalWindowDataset,
)
from .static import StaticDataset
from .shift_windowing import resolve_step_size
from torch.utils.data import Dataset
from neural_mi.logger import groups_repeats, logger, user_stacklevel

_REGULAR_GRID_TYPES = ('continuous', 'categorical')


def _truncate_leading(data, min_len: int):
    """Truncate the leading (sample) dimension to ``min_len``, whether
    ``data`` is a plain tensor or a tuple of tensors sharing that dimension
    (``StaticDataset``'s compound "X-role" data for
    ``mode='conditional'(align='dual_branch')``). A naive ``data[:min_len]``
    on a raw tuple would slice the *outer* 2-element tuple instead of
    truncating each inner tensor.
    """
    if isinstance(data, tuple):
        return tuple(d[:min_len] for d in data)
    return data[:min_len]


# Retention warnings are raised where windowing happens, which is once per
# task. A rigorous run builds 55 gamma-chunks and a sweep one dataset per
# combination, so an undeduplicated warning repeats until it is ignored.
# Warned once per run: the advice does not change between tasks, and the
# per-task numbers are carried in the results frame for anyone who wants
# them. Per process, so a worker pool reports at most once per worker.
_RETENTION_WARNED = set()


def reset_retention_warnings() -> None:
    """Clear the retention-warning dedup cache, called once per ``run()``."""
    _RETENTION_WARNED.clear()
    _AMBIGUOUS_STEP_WARNED.clear()


# Warned once per (window_size, step_size) pair per process, for the same
# reason as the retention warning above: the advice does not change between
# tasks, and a sweep would otherwise repeat it once per task.
_AMBIGUOUS_STEP_WARNED = set()


class WindowManager:
    """Centralized manager for creating and aligning temporal windows."""

    def __init__(self, window_size, t_start=None, t_end=None, step_size=None):
        self.window_size = window_size
        self.step_size = step_size  # None = step of one whole window, so windows tile
        self._warn_if_step_is_ambiguous()
        self.t_start = t_start
        self.t_end = t_end
        self.window_times = None
        self.valid_windows = None
        self.n_windows = 0
        self._observers = []  # Datasets that need to be notified
        if t_start is not None and t_end is not None:
            self.create_windows()
        
    def register_observer(self, observer):
        """Register a dataset to be notified of changes."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def _notify_observers(self):
        """Notify all registered datasets of window changes (lightweight updates only)."""
        for observer in self._observers:
            observer._on_window_manager_updated()

    def update_parameters(self, window_size=None, t_start=None, t_end=None, step_size=None):
        """Update window parameters and regenerate windows."""
        if window_size is not None:
            self.window_size = window_size
        if step_size is not None:
            self.step_size = step_size
        if window_size is not None or step_size is not None:
            self._warn_if_step_is_ambiguous()
        if t_start is not None:
            self.t_start = t_start
        if t_end is not None:
            self.t_end = t_end

        if self.t_start is not None and self.t_end is not None:
            self.create_windows()
            self._notify_observers()

    def _warn_if_step_is_ambiguous(self):
        """Warn when ``step_size`` in ``(0, 1)`` meets ``window_size`` below 1.

        ``step_size`` below 1 always means a fraction of the window, and that
        rule is unambiguous while windows are counted in samples, because a
        sample step is never below 1. Windows measured in seconds break the
        assumption: sub-second windows and sub-second steps are both ordinary
        there, so the same number reads either as a fraction or as a duration
        and the two give different answers. Naming both readings is the only
        way the caller can tell which one they got.
        """
        window, step = self.window_size, self.step_size
        if window is None or step is None:
            return
        if not (0 < step < 1) or not (window < 1):
            return
        key = (float(window), float(step))
        if key in _AMBIGUOUS_STEP_WARNED:
            return
        _AMBIGUOUS_STEP_WARNED.add(key)
        applied = step * window
        overlap = 100.0 * (1.0 - applied / window)
        # Spell out how to reach the absolute step instead, which is not always
        # possible: below 1 the fraction reading is the only one available, so
        # an absolute step wider than the window cannot be expressed at all.
        if step < window:
            remedy = (f"pass step_size={step / window:g} to get {step:g} as a "
                      f"fraction of the window")
        elif step == window:
            remedy = "pass step_size=None (the default) for a step of one window"
        else:
            remedy = (f"a {step:g} step cannot be expressed at "
                      f"window_size={window:g} because every value below 1 is read "
                      f"as a fraction. Use a larger window_size or rescale your "
                      f"time unit")
        warnings.warn(
            f"step_size={step:g} was read as a fraction of window_size={window:g}. "
            f"That gives a step of {applied:g} time units and {overlap:g}% overlap "
            f"between consecutive windows. Every step_size below 1 is a fraction. "
            f"If you meant {step:g} as an absolute duration, {remedy}. This is "
            f"reported only when window_size is below 1. A fraction and a duration "
            f"are then both plausible readings of the same number.",
            UserWarning, stacklevel=user_stacklevel(),
        )

    def resolve_step(self):
        """Return the actual step size in time units.

        - None or step >= 1 : absolute step (None defaults to window_size).
        - 0 < step < 1      : fraction of window_size used as step.

        The fraction rule is what makes ``step_size=0.25`` mean 75% overlap at
        any window size. It also means an absolute step below 1 time unit
        cannot be requested directly: ask for it as a fraction. With
        ``window_size`` below 1 the two readings differ and
        :meth:`_warn_if_step_is_ambiguous` says which one was applied.

        Delegates to :func:`neural_mi.data.shift_windowing.resolve_step_size`,
        which the shift path calls too, so one convention covers both.
        """
        return resolve_step_size(self.window_size, self.step_size)

    def create_windows(self):
        """Create window times for a given temporal range."""
        if self.t_start is None or self.t_end is None:
            raise RuntimeError("t_start and t_end parameters need to be set to create windows")
        step = self.resolve_step()
        self.window_times = np.arange(self.t_start, self.t_end, step)
        self.n_windows = len(self.window_times)
        # Initialize all windows as valid - will be updated by datasets
        self.valid_windows = np.full(self.window_times.size, True, dtype=bool)

    def set_fixed_grid(self, t_start, n_windows):
        """Deterministically build exactly ``n_windows`` window starts
        beginning at ``t_start``, spaced by the resolved step.

        Used by the time-shift path instead of ``create_windows()``'s
        ``np.arange(t_start, t_end, step)`` so the window count stays
        exactly fixed across repeated shifts, ``np.arange`` can drop or
        add a trailing window across float-step boundaries as ``t_start``
        drifts, which ``create_windows()`` never has to contend with since
        it's only ever called once per (fixed) t_start/t_end pair.
        """
        step = self.resolve_step()
        self.t_start = t_start
        self.window_times = t_start + np.arange(n_windows) * step
        self.n_windows = n_windows
        self.valid_windows = np.full(n_windows, True, dtype=bool)

    def __len__(self):
        """len() will return however many valid windows there are"""
        return self.n_windows


def _trailing_width(data):
    """Size of a built dataset tensor's trailing axis.

    Returns a tuple of widths for the compound ``(a_data, c_data)`` path, and
    ``None`` when the side is absent. Used by the ``*_window_width`` properties
    on both paired dataset classes, so the two report the width the same way.
    """
    if data is None:
        return None
    if isinstance(data, tuple):
        return tuple(d.shape[-1] for d in data)
    return data.shape[-1]


class _NamedStreams:
    """Holds an ordered mapping of named streams, and the parts of the API that
    do not care whether the streams carry a window grid.

    Shared by :class:`AlignedStreams` and :class:`AlignedStaticStreams` so the
    accessors, the length and the per-stream noise/precision helpers exist once.
    What differs between the two is alignment: a grid and a validity fold on one
    side, a truncation to a common length on the other.
    """

    def _adopt_streams(self, streams, owner):
        """Store the streams in order, dropping ``None`` entries."""
        self._streams = OrderedDict(
            (name, ds) for name, ds in streams.items() if ds is not None
        )
        if not self._streams:
            raise ValueError(
                f"{owner} needs at least one stream and every entry was None."
            )

    @property
    def stream_names(self):
        """The streams' names, in order."""
        return tuple(self._streams)

    def stream(self, name):
        """One stream's dataset by name, or ``None`` if it is absent."""
        return self._streams.get(name)

    @property
    def _reference_stream(self):
        """The first stream, which fixes the sample count for the bundle."""
        return next(iter(self._streams.values()))

    def __len__(self):
        return len(self._reference_stream)

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self._streams.values())

    def apply_noise_by_name(self, amplitudes):
        """Apply noise per stream, keyed by stream name."""
        for name, amplitude in (amplitudes or {}).items():
            dataset = self._streams.get(name)
            if dataset is not None and amplitude > 0:
                dataset.apply_noise(amplitude)

    def apply_precision_by_name(self, precisions):
        """Apply precision per stream, keyed by stream name."""
        for name, precision in (precisions or {}).items():
            dataset = self._streams.get(name)
            if dataset is not None and precision > 0:
                dataset.apply_precision(precision)


class AlignedStreams(_NamedStreams, Dataset):
    """Any number of named streams sharing one window grid.

    Alignment is a fold, not a pairwise operation: the grid spans the range
    every stream covers, and a window survives only where every stream is
    valid. Writing those folds over a collection instead of over a fixed pair
    is what lets a third stream share the grid instead of deriving its own and being reconciled afterwards, the point where window origins can disagree.

    Streams are ordered, and the first is the reference for length. A ``None``
    entry is dropped, so a one-stream bundle is the same code path with one
    element and needs no special case.

    :class:`PairedTemporalDataset` is this class with the two names ``'x'`` and
    ``'y'`` and the pair-shaped accessors callers already use.
    """

    def __init__(self, streams, window_size=None,
                 t_start=None, t_end=None,
                 validate_windows=True,
                 step_size=None):
        """
        Parameters
        ----------
        streams : mapping of str to TemporalWindowDataset
            Named streams, not yet initialized with windows. Order is kept, and
            entries whose value is ``None`` are dropped.
        window_size : float
            Window size in time units, shared by every stream.
        t_start, t_end : float, optional
            Bounds for the grid. Defaults to the range every stream covers.
        validate_windows : bool, optional
            Keep only windows where every stream has data. Akin to a
            conditional MI on the presence of data.
        step_size : float or int, optional
            Step between consecutive window starts. See
            :class:`PairedTemporalDataset`.
        """
        self._adopt_streams(streams, 'AlignedStreams')
        self.validate_windows = validate_windows
        # Defined unconditionally so callers can read retention without having
        # to know whether validation ran; 1.0 is the truthful value when it did
        # not, since nothing was dropped.
        self.n_windows_built = 0
        self.n_windows_retained = 0
        self.window_retention = 1.0

        if window_size is None:
            raise ValueError("window_size must be provided")
        self.window_manager = WindowManager(window_size, step_size=step_size)
        self._initialize_windows(t_start, t_end)
        for dataset in self._streams.values():
            dataset.set_window_manager(self.window_manager)
        self._build_windows()

        logger.info(f"Created {self.window_manager.n_windows} aligned windows")

    def _initialize_windows(self, t_start, t_end):
        """Create windows over the range every stream covers."""
        extents = [ds.get_temporal_extent() for ds in self._streams.values()]
        data_start = max(start for start, _ in extents)
        data_end = min(end for _, end in extents)
        # Record the original recording bounds on the first call so they can
        # be used to clamp the range after time-shifts.
        if not hasattr(self, 'original_data_start'):
            self.original_data_start = data_start
            self.original_data_end = data_end
        # Clamp the shifted extent so it cannot exceed the original recording.
        data_start = max(data_start, self.original_data_start)
        data_end   = min(data_end,   self.original_data_end)
        # Apply user-specified bounds if provided
        final_start = t_start if t_start is not None else data_start
        final_end = t_end if t_end is not None else data_end
        if final_start >= final_end:
            raise ValueError(
                f"Invalid temporal range: t_start={final_start}, t_end={final_end}"
            )
        # Record the effective (user-bound-aware) range once, for time_shift's
        # fixed-grid margin reservation -- distinct from original_data_start/
        # _end above, which is the natural full extent and can be wider than
        # this when the caller passed an explicit t_start/t_end.
        if not hasattr(self, '_base_t_start'):
            self._base_t_start = final_start
            self._base_t_end = final_end

        self.window_manager.update_parameters(t_start=final_start, t_end=final_end)

    def _build_windows(self, time_shift=None):
        """Move every stream into windows, then keep the ones all of them share.

        Parameters
        ----------
        time_shift : float, optional
            If windows are being rebuilt due to a time shift, the offset applied.
        """
        # Step 1: Move data to ALL windows, for every stream
        for dataset in self._streams.values():
            dataset.move_data_to_windows()
        # Step 2: Validate and filter if requested
        if self.validate_windows:
            per_stream_valid = OrderedDict(
                (name, dataset.validate_window_coverage())
                for name, dataset in self._streams.items()
            )
            # Only keep windows valid for EVERY stream
            combined_valid = np.logical_and.reduce(list(per_stream_valid.values()))
            # Retention is part of the result, not just an internal detail:
            # it says what fraction of the recording the estimate actually
            # covers, and therefore whether a per-window number should be read
            # as per-window or per-active-window.
            self.n_windows_built = int(combined_valid.size)
            self.n_windows_retained = int(combined_valid.sum())
            self.window_retention = (self.n_windows_retained / self.n_windows_built
                                     if self.n_windows_built else 0.0)
            if (self.window_retention < 0.5 and self.n_windows_built
                    and 'low' not in _RETENTION_WARNED):
                _RETENTION_WARNED.add('low')
                # Named per stream, so a bundle of three says which of the three
                # did the dropping instead of leaving the caller to guess.
                dropped_by = [
                    f"{name.upper()} ({int((~valid).sum())})"
                    for name, valid in per_stream_valid.items() if not valid.all()
                ]
                logger.warning(
                    f"Window coverage validation kept {self.n_windows_retained} of "
                    f"{self.n_windows_built} windows ({self.window_retention:.1%}). "
                    f"Dropped per stream: {' and '.join(dropped_by) or 'coverage rules'}. "
                    f"The estimate describes the retained windows only. A per-window "
                    f"figure is per retained window. For spike data, "
                    f"{{'drop_empty_windows': False}} in the stream's processor "
                    f"parameters keeps silent windows and estimates the unrestricted "
                    f"quantity. Retention falls quickly as more variables are required "
                    f"to be simultaneously valid. This is reported once per run. The "
                    f"per-task values are in result.runs as 'window_retention'."
                )

            # Step 3: Update WindowManager's tracking
            self.window_manager.valid_windows = combined_valid
            self.window_manager.n_windows = int(combined_valid.sum())
            # Step 4: Apply the same mask to every stream
            for dataset in self._streams.values():
                dataset.remove_invalid_windows()
            if self.window_manager.n_windows == 0:
                raise ValueError("No valid windows after checking data coverage")
            self.window_manager.window_times = self.window_manager.window_times[combined_valid]
            self.window_manager.valid_windows = np.ones(self.window_manager.n_windows, dtype=bool)

            logger.info(
                f"Window coverage: {self.window_manager.n_windows}/{len(combined_valid)} "
                f"windows have sufficient data"
            )
        else:
            self.window_manager.n_windows = len(self.window_manager.window_times)
        self._notify_subset_views(time_shift=time_shift)

    def _notify_subset_views(self, time_shift=None):
        """
        Notify all registered subset views that windows have been rebuilt.

        Parameters
        ----------
        time_shift : offset_x, optional
            If windows were rebuilt due to time shift, contains the offset applied to x
        """
        if hasattr(self, '_subset_views'):
            for view in self._subset_views:
                view._on_dataset_updated(time_shift=time_shift)

    def _reserve_shift_margin(self):
        """Compute (once, cached) the window count that stays fixed for
        every shift offset in ``[0, window_size)``.

        The window grid slides forward by up to a full ``window_size`` over
        fixed, unmutated raw data. Reserving ``2*window_size`` off the
        effective span (one ``window_size`` so the grid can start that far
        forward without any window running past ``_base_t_end``, and
        another so ``set_fixed_grid`` never needs to look earlier than
        ``_base_t_start``) keeps the same number of windows valid across the
        whole shift range, the fixed-grid analogue of
        ``shift_windowing.safe_n_windows``'s margin for the raw-array-slicing
        case.
        """
        if not hasattr(self, '_shift_n_windows'):
            step = self.window_manager.resolve_step()
            window_size = self.window_manager.window_size
            usable_span = self._base_t_end - self._base_t_start
            margin = 2 * window_size
            n_safe = int((usable_span - margin) // step) + 1
            if n_safe < 1:
                raise ValueError(
                    f"Recording span ({usable_span:.6g}) is too short to reserve a "
                    f"safe time-shift margin ({margin:.6g} = 2 * window_size) with "
                    f"window_size={window_size}. Reduce window_size, increase the "
                    f"recording length or disable shift_time."
                )
            self._shift_n_windows = n_safe
        return self._shift_n_windows

    def shift_grid(self, offset):
        """Slide the (fixed-size) window grid forward by ``offset`` over the
        fixed, unmutated raw data.

        Does NOT rewrite any raw data, unlike the child datasets' own
        ``time_shift()`` (still reachable directly for unit-level callers),
        which mutate their stored data and derive the grid's live extent
        from it, causing the offset to cancel out of every window-membership
        test. One grid is shared by every stream, so one offset moves them
        all together.
        """
        n_windows = self._reserve_shift_margin()
        self.window_manager.set_fixed_grid(self._base_t_start + offset, n_windows)
        self._build_windows(time_shift=offset)

    def set_window_size(self, window_size):
        """Change window size and rebuild windows."""
        self.window_manager.update_parameters(window_size=window_size)
        self._build_windows()


class _PairAccessors:
    """The ``x``/``y`` names for the first two streams of a bundle.

    Both paired classes present the same surface, so it lives here once: the
    two datasets by name, their data, their per-side window
    widths, and the pair-shaped noise/precision signatures. Read-only, because
    nothing outside ``__init__`` ever rebound them; callers mutate the dataset
    objects these return.
    """

    @property
    def x_dataset(self):
        return self._streams['x']

    @property
    def y_dataset(self):
        return self._streams.get('y')

    # Small properties for convenient access to main data
    @property
    def x_data(self):
        return self.x_dataset.data

    @property
    def y_data(self):
        return self.y_dataset.data

    @property
    def x_window_width(self):
        """Width of X's built window, read off the trailing axis.

        What that axis counts depends on the processor, and only one of the
        three counts time, so read it against the processor type instead of on
        its own.

        * continuous: ``w`` time slots for ``window_size=w``, covering the
          half-open interval ``[t, t + w)``.
        * categorical: one slot per category under the default
          ``encoding='majority_vote'`` (and under ``'probability'``), and
          ``n_categories`` times the window's sample count under
          ``'full_trajectory'``. Never ``w``: the window size sets how many
          windows are built, not how wide this axis is.
        * spike: ``max_spikes_per_window`` spike slots, capped by the busiest
          window actually present, not a time axis either.

        The two sides therefore differ legitimately in a mixed-type pair, and
        indexing the tensor is otherwise the only way to see what was built.
        """
        return _trailing_width(self.x_dataset.data)

    @property
    def y_window_width(self):
        """Width of Y's built window, read off the trailing axis.

        What that axis counts depends on the processor, and only one of the
        three counts time, so read it against the processor type instead of on
        its own.

        * continuous: ``w`` time slots for ``window_size=w``, covering the
          half-open interval ``[t, t + w)``.
        * categorical: one slot per category under the default
          ``encoding='majority_vote'`` (and under ``'probability'``), and
          ``n_categories`` times the window's sample count under
          ``'full_trajectory'``. Never ``w``: the window size sets how many
          windows are built, not how wide this axis is.
        * spike: ``max_spikes_per_window`` spike slots, capped by the busiest
          window actually present, not a time axis either.

        The two sides therefore differ legitimately in a mixed-type pair, and
        indexing the tensor is otherwise the only way to see what was built.
        """
        return _trailing_width(self.y_dataset.data) if self.y_dataset is not None else None

    def __getitem__(self, idx):
        x_data = self.x_dataset[idx]
        y_data = self.y_dataset[idx] if self.y_dataset else None
        return x_data, y_data

    def apply_noise(self, amplitude_x=0, amplitude_y=0):
        """Apply noise to both datasets."""
        self.apply_noise_by_name({'x': amplitude_x, 'y': amplitude_y})

    def apply_precision(self, precision_x=0, precision_y=0):
        self.apply_precision_by_name({'x': precision_x, 'y': precision_y})


class StreamBundle(_PairAccessors, AlignedStreams):
    """Any number of aligned streams, answering to ``x``/``y`` for the first two.

    The shape every mode can consume: a conditional or interaction run holds
    ``x``, ``y`` and ``w`` on one grid here, and downstream code that only knows
    about ``x_data``/``y_data`` keeps working because those names still resolve.
    """

    def __init__(self, streams, **kwargs):
        super().__init__(streams, **kwargs)

    def time_shift(self, offset_x=0, offset_y=0):
        """Slide the (fixed-size) window grid forward by ``offset_x``.

        ``offset_y`` is accepted for backward compatibility with callers that
        pass both, but only ``offset_x`` drives the single grid, which every
        stream in the bundle shares; production call sites always pass equal
        values. See :meth:`AlignedStreams.shift_grid` for what the shift does
        and does not touch.
        """
        if offset_y != offset_x:
            logger.warning(
                f"time_shift got offset_x={offset_x} != offset_y={offset_y}. Only "
                f"offset_x is used. It shifts the single window grid every stream "
                f"shares."
            )
        self.shift_grid(offset_x)


class PairedTemporalDataset(StreamBundle):
    """Wrapper for paired X and Y datasets with temporal alignment.

    Two named streams over :class:`AlignedStreams`, plus the pair-shaped
    accessors (``x_data``, ``y_window_width``, ``apply_noise(amplitude_x=...)``)
    that callers already use. The alignment itself lives in the base class,
    where it is written as a fold and so carries any number of streams.
    """

    def __init__(self, x_dataset, y_dataset=None,
                 window_size=None,
                 t_start=None, t_end=None,
                 validate_windows=True,
                 step_size=None):
        """
        Parameters
        ----------
        x_dataset : TemporalWindowDataset
            Dataset for X variable (not yet initialized with windows)
        y_dataset : TemporalWindowDataset, optional
            Dataset for Y variable (not yet initialized with windows)
        window_size : float
            Window size in time units
        t_start : float, optional
            Start time for windows
        t_end : float, optional
            End time for windows
        validate_windows : bool, optional
            Whether to return/use only "valid" windows where data is present
            Akin to a conditional MI on the presence of data
        step_size : float or int, optional
            Step between consecutive window starts.  ``None`` (default) gives
            windows that tile without overlap (step = window_size), since a
            window covers ``[t, t + window_size)``.  Values in ``(0, 1)``
            are treated as a fraction of ``window_size`` (e.g. 0.25 → 75%
            overlap).  Values ≥ 1 are used as an absolute step in the same
            time units as ``window_size``.  An absolute step below 1 time unit
            therefore has to be asked for as a fraction: with
            ``window_size=0.5``, a 0.125 s step is ``step_size=0.25``.  A
            warning names both readings when ``window_size`` is itself below 1.
        """
        super().__init__(
            OrderedDict((('x', x_dataset), ('y', y_dataset))),
            window_size=window_size, t_start=t_start, t_end=t_end,
            validate_windows=validate_windows, step_size=step_size,
        )

    # The two streams by their historic names. Read-only: nothing outside
    # __init__ ever rebound them, only mutated the dataset objects they hold.

class AlignedStaticStreams(_NamedStreams, Dataset):
    """Any number of already-windowed streams, truncated to a common length.

    The static counterpart to :class:`AlignedStreams`: no grid and no window
    validity, since the caller has already done the windowing, but the same
    fold shape so a third stream needs no special case. Streams are ordered and
    the first is the reference.
    """

    def __init__(self, streams):
        self._adopt_streams(streams, 'AlignedStaticStreams')
        if len(self._streams) > 1:
            self._align_datasets()
        logger.info("Created PairedDataset")

    def _align_datasets(self):
        """Truncate every stream to the shortest one's sample count.

        Truncates via a leading-dimension slice on each ``self.data`` tensor,
        because ``StaticDataset.__len__`` reads ``self.data.shape[0]`` and the
        slice is what makes the length change take effect.
        """
        lengths = OrderedDict((name, len(ds)) for name, ds in self._streams.items())
        if len(set(lengths.values())) == 1:
            return
        min_len, max_len = min(lengths.values()), max(lengths.values())
        lost = max_len - min_len
        pct = 100.0 * lost / max_len
        names = " and ".join(name.upper() for name in lengths)
        counts = ", ".join("%s: %d" % (name.upper(), n) for name, n in lengths.items())
        every = "both" if len(lengths) == 2 else "all"
        logger.warning(
            f"{names} have different numbers of samples ({counts}). "
            f"Truncating {every} to {min_len} samples "
            f"({lost} samples discarded, {pct:.1f}% of the larger dataset lost)."
        )
        for dataset in self._streams.values():
            dataset.data = _truncate_leading(dataset.data, min_len)
            # Invalidate lazily-allocated data_master so it is re-cloned from
            # the truncated data the next time apply_noise/apply_precision runs.
            dataset.data_master = None


class PairedDataset(_PairAccessors, AlignedStaticStreams):
    """
    Dataset object for when both X/Y are given processor type of None. 
    Assumes user already preprocessed data as much as they want, so avoids windowing or any temporal features.
    """

    def __init__(self, x_dataset, y_dataset=None):
        """
        Parameters
        ----------
        x_dataset : StaticDataset
            Dataset for X variable
        y_dataset : StaticDataset, optional
            Dataset for Y variable
        """
        super().__init__(OrderedDict((('x', x_dataset), ('y', y_dataset))))
def create_single_dataset(data, time, proc_type, proc_params, device=None, data_device='cpu'):
    """Create a dataset for a single variable.

    Parameters
    ----------
    data : array-like
        Raw data for this variable.
    time : array-like or None
        Time vector; required for temporal processors.
    proc_type : str or None
        Processor type: ``'continuous'``, ``'spike'``, ``'categorical'``, or
        ``None`` for pre-processed static data.
    proc_params : dict or None
        Processor-specific parameters extracted from ``processor_params_x/y``.
    device : str, optional
        Compute device (reference only).
    data_device : str, optional
        Device for storing dataset tensors.  Defaults to ``'cpu'``.  Pass
        ``'auto'`` to co-locate data with the compute device (useful for
        precision analysis where the same dataset is evaluated many times).
    """
    if proc_type is None:
        return StaticDataset(data, device=device, data_device=data_device)

    if proc_type == 'continuous':
        min_cov    = (proc_params or {}).get('min_coverage_fraction', 0.2)
        sample_rate = (proc_params or {}).get('sample_rate', None)
        return ContinuousWindowDataset(data, time, device=device,
                                       min_coverage_fraction=min_cov,
                                       data_device=data_device,
                                       sample_rate=sample_rate)

    elif proc_type == 'spike':
        # Silence is only discarded on the spike side. A continuous partner's
        # own min_coverage_fraction is untouched by this, which is what lets a
        # timestamped continuous variable keep masking genuinely unobserved
        # stretches while silent spike windows are retained as data.
        drop_empty = (proc_params or {}).get('drop_empty_windows', True)
        bin_size = (proc_params or {}).get('bin_size', None)
        if bin_size is not None:
            normalize = (proc_params or {}).get('normalize_bins', True)
            return BinnedSpikeDataset(data, bin_size=bin_size, device=device,
                                      normalize=normalize, data_device=data_device,
                                      drop_empty_windows=drop_empty)
        no_spike_val        = (proc_params or {}).get('no_spike_value', 0.0)
        excl_bursty         = (proc_params or {}).get('exclude_bursty_neurons', False)
        burst_mult          = (proc_params or {}).get('burst_threshold_multiplier', 5.0)
        max_spikes_per_win  = (proc_params or {}).get('max_spikes_per_window', None)
        n_seconds           = (proc_params or {}).get('n_seconds', None)
        return SpikeWindowDataset(data, time, device=device,
                                  no_spike_value=no_spike_val,
                                  exclude_bursty_neurons=excl_bursty,
                                  burst_threshold_multiplier=burst_mult,
                                  data_device=data_device,
                                  max_spikes_per_window=max_spikes_per_win,
                                  n_seconds=n_seconds,
                                  drop_empty_windows=drop_empty)

    elif proc_type == 'categorical':
        min_cov     = (proc_params or {}).get('min_coverage_fraction', 0.2)
        encoding    = (proc_params or {}).get('encoding', 'majority_vote')
        sample_rate = (proc_params or {}).get('sample_rate', None)
        return CategoricalWindowDataset(data, time, device=device,
                                        min_coverage_fraction=min_cov,
                                        encoding=encoding,
                                        data_device=data_device,
                                        sample_rate=sample_rate)

    else:
        raise ValueError(f"Unknown processor type: '{proc_type}'.")

def _stream_specs_from_pair(x_data, y_data, x_time, y_time,
                            processor_type_x, processor_params_x,
                            processor_type_y, processor_params_y):
    """The two-argument form, as an ordered stream mapping.

    ``processor_type_y=None`` means "inherit X's", the convention callers have
    always used, so it is resolved here where the pair is still visible.
    """
    specs = OrderedDict()
    specs['x'] = dict(data=x_data, time=x_time, processor_type=processor_type_x,
                      processor_params=processor_params_x)
    if y_data is not None:
        specs['y'] = dict(
            data=y_data, time=y_time,
            processor_type=(processor_type_y if processor_type_y is not None
                            else processor_type_x),
            processor_params=(processor_params_y if processor_params_y is not None
                              else processor_params_x),
        )
    return specs


def _warn_on_mixed_units(specs):
    """Warn when a spike stream meets a regular-grid stream in a different unit.

    'spike' timestamps are always in seconds; 'continuous'/'categorical' default
    to raw sample-index units unless `sample_rate` is given. Every stream shares
    one WindowManager, which combines their extents by plain numeric min/max, so
    a mismatch there destroys the alignment outright: a 20000-sample continuous
    recording against a 40-second spike recording windows only "sample 0 to 40"
    of the continuous side, silently discarding 99.8% of it. An explicit time vector already puts a stream in real time, supplying what `sample_rate` otherwise would.
    """
    spikes = [n for n, spec in specs.items() if spec.get('processor_type') == 'spike']
    if not spikes:
        return
    for name, spec in specs.items():
        kind = spec.get('processor_type')
        if kind not in _REGULAR_GRID_TYPES or spec.get('time') is not None:
            continue
        if not (spec.get('processor_params') or {}).get('sample_rate'):
            logger.warning(
                f"processor_type mixes 'spike' ({', '.join(spikes)}) with "
                f"'{kind}' on stream {name!r} without a 'sample_rate' on it. "
                f"'spike' timestamps are always in seconds. '{kind}' is in raw "
                f"sample-index units without a sample_rate. The streams' window "
                f"boundaries may then cover different real times and make the "
                f"alignment meaningless. Set {{'sample_rate': ...}} in the processor "
                f"parameters of {name!r} or give it a time vector to put them on a "
                f"shared time unit."
            )


@groups_repeats
def create_dataset(x_data, y_data=None,
                   x_time=None, y_time=None,
                   processor_type_x=None, processor_params_x=None,
                   processor_type_y=None, processor_params_y=None,
                   device=None, data_device='cpu', validate_windows=True):
    """Build one, two, or any number of streams onto a single window grid.

    The two-argument form takes ``x_data``/``y_data`` with their own
    ``processor_type_*``/``processor_params_*``. Passing a mapping as the first
    argument builds that many named streams through the same code:

    .. code-block:: python

        create_dataset(OrderedDict((
            ('x', dict(data=spikes, processor_type='spike',
                       processor_params={'window_size': 1.0, 'bin_size': 0.1})),
            ('y', dict(data=pos, time=pos_t, processor_type='continuous',
                       processor_params={'window_size': 1.0})),
            ('w', dict(data=labels, time=pos_t, processor_type='categorical',
                       processor_params={'window_size': 1.0})),
        )))

    The mapping form is what a third stream needs. Built in its own call, a
    stream derives its own grid, whose origin is the latest start among *that
    call's* streams, so a conditioning variable built separately can sit a
    fraction of a window away from the pair it is meant to condition and share
    no window times with it at all. Every stream here lands on one grid, and a
    window survives only where all of them have data.

    Parameters
    ----------
    x_data : array-like or mapping
        Raw data for X, or a mapping of named stream specs (see above). With a
        mapping, the remaining ``x_*``/``y_*`` arguments must be left unset.
    validate_windows : bool, optional
        Keep only the windows every stream is valid on. ``False`` keeps the grid uniformly spaced, as offset-indexed quantities need.
    device, data_device
        See :func:`create_single_dataset`.

    Returns
    -------
    PairedTemporalDataset, PairedDataset, StreamBundle or AlignedStaticStreams
        The two-stream forms return the paired classes; more than two streams
        returns the general bundle. All of them expose ``stream(name)``, and the
        first two streams are also reachable as ``x_data``/``y_data``.
    """
    if isinstance(x_data, Mapping):
        if any(v is not None for v in (y_data, x_time, y_time, processor_type_x,
                                       processor_params_x, processor_type_y,
                                       processor_params_y)):
            raise ValueError(
                "create_dataset received a mapping of streams and also one of the "
                "x_*/y_* arguments. Put every stream's data, time and processor "
                "settings in the mapping or use the two-argument form."
            )
        specs = OrderedDict(x_data)
        argument_names = {name: f"stream {name!r}" for name in specs}
        pair_form = False
    else:
        specs = _stream_specs_from_pair(x_data, y_data, x_time, y_time,
                                        processor_type_x, processor_params_x,
                                        processor_type_y, processor_params_y)
        argument_names = {'x': 'x_data', 'y': 'y_data'}
        pair_form = True
    if not specs:
        raise ValueError("create_dataset needs at least one stream.")

    # A string is never valid data, and there is one way it reliably gets there:
    # create_dataset's second positional argument is y_data, so the
    # natural-looking create_dataset(x, 'continuous', {...}) passes the processor
    # type as data. Caught here, where the stream's name is known, instead of in
    # StaticDataset._prepare_one, which sees an anonymous array and can name
    # neither the argument nor the mistake.
    for name, spec in specs.items():
        if isinstance(spec.get('data'), str):
            hint = ("create_dataset's second positional argument is y_data, so "
                    "create_dataset(x, 'continuous', ...) puts the processor type "
                    "there. Pass it by keyword: create_dataset(x, "
                    "processor_type_x='continuous', processor_params_x={...}).")
            raise ValueError(
                f"{argument_names[name]} was given the string {spec['data']!r} as data. "
                f"Data must be array-like (list, numpy array or torch.Tensor). "
                + (hint if pair_form and name == 'y' else "")
            )

    params = OrderedDict(
        (name, dict(spec.get('processor_params') or {})) for name, spec in specs.items()
    )
    if len(specs) > 1:
        _warn_on_mixed_units(specs)

    # One grid means one window_size. The first stream that names one sets it,
    # and another asking for a different value is reported instead of silently
    # losing to whichever stream was built last.
    window_size = step_size = None
    for name, prm in params.items():
        this_window = prm.pop('window_size', None)
        this_step = prm.pop('step_size', None)
        if this_window is not None:
            if window_size is None:
                window_size = this_window
            elif this_window != window_size:
                logger.warning(
                    f"Stream {name!r} specifies window_size={this_window}. Every "
                    f"stream shares a single WindowManager and must use the same "
                    f"window size. window_size={window_size} is used."
                )
        if this_step is not None:
            if step_size is None:
                step_size = this_step
            elif this_step != step_size:
                logger.warning(
                    f"Stream {name!r} specifies step_size={this_step} against the "
                    f"{step_size} already set by an earlier stream. One grid carries "
                    f"one step. Using step_size={step_size}."
                )

    datasets = OrderedDict()
    for name, spec in specs.items():
        datasets[name] = create_single_dataset(
            spec.get('data'), spec.get('time'), spec.get('processor_type'),
            params[name], device=device, data_device=data_device,
        )

    static = [n for n, ds in datasets.items() if isinstance(ds, StaticDataset)]
    windowed = [n for n in datasets if n not in static]
    if static and windowed:
        raise ValueError(
            f"A pre-processed stream (processor_type=None) cannot be paired with a "
            f"windowed processor_type. {sorted(static)} are pre-processed and "
            f"{sorted(windowed)} are windowed. A pre-processed stream has no time axis "
            f"to align windows against and cannot share a grid with a windowed one. "
            f"Give every stream a real processor_type (for example 'continuous') or "
            f"pre-process all of them and pass processor_type=None throughout."
        )

    names = list(datasets)
    is_pair_shaped = names[:2] == ['x', 'y'][:len(names)] and len(names) <= 2

    if static:
        if is_pair_shaped:
            return PairedDataset(datasets['x'], datasets.get('y'))
        return AlignedStaticStreams(datasets)

    if window_size is None:
        raise ValueError(
            "A windowed processor needs a window_size in at least one stream's "
            "processor_params."
        )
    if is_pair_shaped:
        return PairedTemporalDataset(datasets['x'], datasets.get('y'),
                                     window_size=window_size, step_size=step_size,
                                     validate_windows=validate_windows)
    return StreamBundle(datasets, window_size=window_size, step_size=step_size,
                        validate_windows=validate_windows)

