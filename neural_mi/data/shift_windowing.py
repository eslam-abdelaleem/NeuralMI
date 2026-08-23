# neural_mi/data/shift_windowing.py
"""Cheap, reslice-based window-tiling shift for regularly-sampled data.

`WindowShifter` holds a raw, unwindowed `(T, n_channels)` array and
re-derives windows via `torch.Tensor.unfold` (a view, the same
sliding-window primitive used by `analysis/transfer.py`/`analysis/offsets.py`)
at an arbitrary integer sample offset, always producing exactly `n_windows`
windows so the window *count* -- and therefore every train/test/eval index
computed against it -- never changes shift-to-shift. Only the physical
content at each window index changes. This backs `Training(shift_windows=True)`.

Covers the "regular grid" processor family: `continuous` (raw values, no
encoder) and `categorical` (integer labels, with a `make_categorical_encoder`
post-step). `spike` data has no regular grid to reslice and uses
`Training(shift_time=True)` instead -- see `shift_family` for which
mechanism applies to which (X, Y) processor-type pair.
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


_REGULAR_GRID_PROCESSOR_TYPES = frozenset({'continuous', 'categorical'})


def shift_family(processor_type_x: Optional[str], processor_type_y_effective: Optional[str]) -> Optional[str]:
    """Classify an (X, Y) processor-type pair for per-epoch shift purposes.

    ``processor_type_y_effective`` must already be resolved to its effective
    value (``processor_type_y if processor_type_y is not None else
    processor_type_x`` -- ``create_dataset``'s own "None means inherit X"
    convention, ``data/handler.py``) by the caller.

    Returns
    -------
    'regular' : both sides in {'continuous', 'categorical'} (need not
        match each other -- continuous+categorical is fine). Supports
        ``Training(shift_windows=True)``.
    'spike' : both sides 'spike'. Supports ``Training(shift_time=True)`` via
        the ``PairedTemporalDataset``/``time_shift`` machinery.
    'mixed' : one side 'spike', the other in {'continuous', 'categorical'}.
        Supports ``shift_time`` *only if* the regular-grid side's
        ``processor_params`` sets ``sample_rate`` -- required so a shift
        value means the same real time on both sides (see
        ``mixed_pair_sample_rate_ok``). Callers must check that separately;
        this function only classifies the pair, it doesn't gate on it.
    None : either side is ``None`` (static/pre-processed data -- no raw
        signal to reslice from, never shiftable) or an unrecognized
        combination.
    """
    if processor_type_x is None or processor_type_y_effective is None:
        return None
    if processor_type_x in _REGULAR_GRID_PROCESSOR_TYPES and processor_type_y_effective in _REGULAR_GRID_PROCESSOR_TYPES:
        return 'regular'
    if processor_type_x == 'spike' and processor_type_y_effective == 'spike':
        return 'spike'
    _types = {processor_type_x, processor_type_y_effective}
    if 'spike' in _types and (_types & _REGULAR_GRID_PROCESSOR_TYPES):
        return 'mixed'
    return None


def mixed_pair_sample_rate_ok(processor_type_x: Optional[str], processor_params_x: Optional[dict],
                              processor_type_y_effective: Optional[str], processor_params_y: Optional[dict]) -> bool:
    """For a ``shift_family(...) == 'mixed'`` pair, check whether the
    regular-grid side (`continuous`/`categorical`) has `sample_rate` set.

    Without it, that side's shift values are in raw sample-index units
    while spike's are natively in seconds -- the same numeric shift would
    mean a different amount of real time on each side, silently breaking
    X/Y temporal alignment. With it, the existing same-value-both-sides
    ``PairedTemporalDataset.time_shift(offset_x=s, offset_y=s)`` call is
    already correct, no new arithmetic needed.
    """
    _regular_params = processor_params_x if processor_type_x in _REGULAR_GRID_PROCESSOR_TYPES else processor_params_y
    return bool((_regular_params or {}).get('sample_rate'))


def seconds_to_samples(value: float, period: float) -> int:
    """Convert a duration in the shared ``WindowManager`` time unit
    (seconds if `period` came from a `sample_rate`, otherwise raw samples
    if `period=1.0`) to an integer sample count for one side of a pair.

    Rounds to the nearest sample; always at least 1. Needed because
    `torch.Tensor.unfold` requires an integer element count, but
    `processor_params['window_size'/'step_size']` may be given in seconds
    (whenever `sample_rate` is set) -- passing that straight to `unfold`
    crashes with a confusing `TypeError` for any non-integer value.
    """
    return max(1, int(round(value / period)))


def build_shifted_windows(raw: torch.Tensor, window_size: int, step_size: int,
                          shift: int, n_windows: int) -> torch.Tensor:
    """``(T, C)`` -> ``(n_windows, C, window_size)`` via ``unfold``, starting
    at raw sample ``shift``. A view-based slide, no interpolation, no
    time-vector -- cost is independent of ``shift`` and negligible relative
    to a training epoch.
    """
    usable = raw[shift:]
    windows = usable.unfold(0, window_size, step_size)[:n_windows]  # (n_windows, C, window_size)
    return windows.contiguous()


def make_categorical_encoder(n_categories: int, encoding: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Vectorized majority_vote/probability/full_trajectory encoding for a
    ``(n_windows, n_channels, window_size)`` integer-label tensor, matching
    :class:`~neural_mi.data.temporal.CategoricalWindowDataset`'s three
    encoding modes -- but with no Python per-window loop, safe here because
    reslice-based windows are always a regular, fully-populated grid (no
    ragged per-window counts to track, unlike the general `WindowManager`
    path that has to handle irregular time vectors).
    """
    if encoding not in ('majority_vote', 'probability', 'full_trajectory'):
        raise ValueError(
            f"Unknown encoding '{encoding}'. Expected 'majority_vote', "
            f"'probability', or 'full_trajectory'."
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


def safe_n_windows(n_samples: int, window_size: int, step_size: int) -> int:
    """Window count reachable for *any* shift in ``[0, window_size)``.

    Used as the fixed count for every epoch, regardless of which shift is
    actually drawn, so window indices (and the train/test/eval splits
    computed against them) stay valid across every reshift.
    """
    worst_case_usable = n_samples - (window_size - 1)
    return max(0, (worst_case_usable - window_size) // step_size + 1)


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
        `seconds_to_samples`) -- same meaning as
        ``Processing(..., x_params={'window_size':..., 'step_size':...})``
        when no `sample_rate` is set.
    encoder : callable, optional
        Applied to each freshly-resliced ``(n_windows, C, window_size)``
        raw tensor before it's returned -- e.g. `make_categorical_encoder`'s
        output for categorical data. ``None`` for continuous data (used as-is).
    """

    def __init__(self, raw: torch.Tensor, window_size: int, step_size: int,
                encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.raw = raw
        self.window_size = window_size
        self.step_size = step_size
        self.encoder = encoder
        self.n_windows = safe_n_windows(raw.shape[0], window_size, step_size)
        if self.n_windows <= 0:
            raise ValueError(
                f"Not enough samples ({raw.shape[0]}) to build even one window "
                f"of size {window_size} with step {step_size} across every "
                f"possible shift in [0, {window_size})."
            )

    def windows_at(self, shift: int) -> torch.Tensor:
        raw_windows = build_shifted_windows(self.raw, self.window_size, self.step_size,
                                            shift, self.n_windows)
        return self.encoder(raw_windows) if self.encoder is not None else raw_windows

    def random_shift(self, generator: torch.Generator = None) -> int:
        return int(torch.randint(0, self.window_size, (1,), generator=generator).item())


class PairedWindowShifter:
    """Pairs an X and a Y ``WindowShifter``.

    Each side may have its own sample rate (hence its own `window_size`/
    `step_size` in raw-sample units, and its own `period` = 1/sample_rate,
    or 1.0 if no `sample_rate` was set). Truncates to a common *physical
    duration* first (not a common raw sample count -- these differ whenever
    `period_x != period_y`), then converts one drawn shift value into each
    side's own sample count so both sides always shift by the same real
    time. When `period_x == period_y == 1.0` (the common case -- neither
    side has a `sample_rate`), this reduces exactly to the same-sample-
    count-both-sides case.
    """

    def __init__(self, raw_x: torch.Tensor, raw_y: torch.Tensor,
                window_size_x: int, step_size_x: int,
                window_size_y: Optional[int] = None, step_size_y: Optional[int] = None,
                encoder_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                encoder_y: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                period_x: float = 1.0, period_y: float = 1.0):
        if window_size_y is None:
            window_size_y = window_size_x
        if step_size_y is None:
            step_size_y = step_size_x
        duration = min(raw_x.shape[0] * period_x, raw_y.shape[0] * period_y)
        n_x = min(raw_x.shape[0], int(duration / period_x))
        n_y = min(raw_y.shape[0], int(duration / period_y))
        self.x = WindowShifter(raw_x[:n_x], window_size_x, step_size_x, encoder_x)
        self.y = WindowShifter(raw_y[:n_y], window_size_y, step_size_y, encoder_y)
        self.n_windows = min(self.x.n_windows, self.y.n_windows)
        self.period_x = period_x
        self.period_y = period_y

    def windows_at(self, shift_x: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # shift_x is a raw-sample offset for X; convert to the equivalent
        # physical-time offset, then to Y's own sample count, so both sides
        # shift by the same real time even when periods differ.
        shift_y = int(round(shift_x * self.period_x / self.period_y))
        shift_y = max(0, min(shift_y, self.y.window_size - 1))
        return self.x.windows_at(shift_x), self.y.windows_at(shift_y)

    def random_shift(self, generator: torch.Generator = None) -> int:
        return self.x.random_shift(generator)
