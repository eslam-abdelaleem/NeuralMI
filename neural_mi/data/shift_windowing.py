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
from typing import Callable, List, Optional, Tuple

import numpy as np
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


def make_multi_categorical_encoder(block_specs: List[Tuple[int, Optional[int]]],
                                   encoding: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Block-aware analogue of :func:`make_categorical_encoder` for a raw,
    concatenated array whose channel blocks may have different
    ``n_categories`` -- e.g. `conditional`/`interaction`'s X and Z/W,
    relabeled to their own correct ``0..n-1`` range and concatenated
    separately (see ``run_conditional_mi``/``run_interaction_information``'s
    ``raw_deferred`` branch), rather than relabeled *after* concatenation
    (which would infer one shared ``n_categories`` from the combined
    array's max value, silently conflating the two blocks' category counts).

    ``block_specs`` : list of ``(n_channels, n_categories)``, one entry per
    channel block, in the same channel order the blocks were concatenated.
    ``n_categories=None`` marks a *continuous* block -- passed through
    unencoded, at its native ``window_size`` -- for a mixed continuous +
    categorical concatenation (X and the conditioning variable have
    different types but were still concatenated raw, before windowing).

    Encodes each categorical block with its own :func:`make_categorical_encoder`,
    then folds every categorical block's category axis into its channel axis
    (the same fold ``run._reshape_categorical_w_for_conditional`` already
    applies to a single categorical conditioning variable, generalized here
    to multiple blocks with independent category counts). A continuous
    block keeps its native ``window_size`` trailing axis untouched. Once
    every block has been built, any block whose trailing axis collapsed to
    size 1 (a ``majority_vote``/``probability``-encoded categorical block)
    is broadcast up to the widest trailing axis present (a continuous
    block's real ``window_size``, when one is present) before concatenating
    along the channel axis -- mirroring the same broadcast
    ``run._reshape_categorical_w_for_conditional``'s caller already applies
    for a lone categorical conditioning variable
    (``w_data.expand(-1, -1, x_data.shape[2])``). For ``full_trajectory``,
    every block already lands at ``window_size`` natively, so this broadcast
    is a no-op. For an all-categorical ``block_specs`` (no continuous
    blocks), every block is already the same width regardless, so the
    broadcast step is a no-op there too -- this function's existing
    all-categorical behavior is unchanged.
    """
    if encoding not in ('majority_vote', 'probability', 'full_trajectory'):
        raise ValueError(
            f"Unknown encoding '{encoding}'. Expected 'majority_vote', "
            f"'probability', or 'full_trajectory'."
        )
    _per_block_encoders = [make_categorical_encoder(n_cat, encoding) if n_cat is not None else None
                           for _, n_cat in block_specs]

    def _encode(raw_windows: torch.Tensor) -> torch.Tensor:
        n_windows, n_channels, window_size = raw_windows.shape
        outputs = []
        ch0 = 0
        for (n_ch, n_cat), enc in zip(block_specs, _per_block_encoders):
            block_raw = raw_windows[:, ch0:ch0 + n_ch, :]
            if n_cat is None:
                # Continuous block: pass through unencoded, native window_size axis.
                block_encoded = block_raw
            elif encoding == 'full_trajectory':
                # (n_windows, n_ch, window_size*n_cat) -> (n_windows, n_ch, window_size, n_cat)
                # -> (n_windows, n_ch, n_cat, window_size) -> (n_windows, n_ch*n_cat, window_size).
                # Un-flatten order matches make_categorical_encoder's own
                # "position p -> columns [p*n_cat, (p+1)*n_cat)" layout --
                # the same transpose run._reshape_categorical_w_for_conditional
                # already applies to a single block.
                block_encoded = enc(block_raw).reshape(n_windows, n_ch, window_size, n_cat) \
                                              .permute(0, 1, 3, 2).reshape(n_windows, n_ch * n_cat, window_size)
            else:
                # majority_vote/probability: (n_windows, n_ch, n_cat) -> (n_windows, n_ch*n_cat, 1).
                block_encoded = enc(block_raw).reshape(n_windows, n_ch * n_cat, 1)
            outputs.append(block_encoded)
            ch0 += n_ch
        if ch0 != n_channels:
            raise ValueError(
                f"block_specs channel counts sum to {ch0}, but raw_windows has "
                f"{n_channels} channels -- block_specs must partition every "
                f"channel of the concatenated array."
            )
        max_w = max(o.shape[2] for o in outputs)
        outputs = [o.expand(-1, -1, max_w) if o.shape[2] == 1 and max_w > 1 else o for o in outputs]
        return torch.cat(outputs, dim=1)

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


class DualBranchWindowShifter:
    """X, C, Y triple for ``mode='conditional'(align='dual_branch')``.

    C's ``window_size``/``step_size``/``period``/``encoder`` are independent
    of X's -- by design, that's dual_branch's entire premise (a
    ``DualBranchEmbedding`` processes A and C at their own, generally
    different, window lengths). This reuses the exact per-side
    :class:`WindowShifter` machinery :class:`PairedWindowShifter` already
    uses for X vs Y whenever *their* geometries differ, generalized here to
    a third independent side. A sibling class, not a refactor of
    ``PairedWindowShifter`` -- that class backs every already-working
    ``shift_windows`` mode today, so this avoids any risk of regressing it.

    ``windows_at(shift_x)`` returns ``((x_w, c_w), y_w)`` -- matching
    ``PairedWindowShifter``'s existing 2-tuple contract exactly, with the
    X-role element itself a nested tuple. This is the exact shape
    ``StaticDataset``/``Trainer``'s live shift-application code
    (``dataset.x_dataset.data = _x_shifted``) already knows how to store
    and index -- the already-shipped, non-shifted dual_branch path relies
    on the same tuple-as-X-role-data convention.
    """

    def __init__(self, raw_x: torch.Tensor, raw_c: torch.Tensor, raw_y: torch.Tensor,
                window_size_x: int, step_size_x: int,
                window_size_c: int, step_size_c: int,
                window_size_y: Optional[int] = None, step_size_y: Optional[int] = None,
                encoder_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                encoder_c: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                encoder_y: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                period_x: float = 1.0, period_c: float = 1.0, period_y: float = 1.0):
        if window_size_y is None:
            window_size_y = window_size_x
        if step_size_y is None:
            step_size_y = step_size_x
        duration = min(raw_x.shape[0] * period_x, raw_c.shape[0] * period_c, raw_y.shape[0] * period_y)
        n_x = min(raw_x.shape[0], int(duration / period_x))
        n_c = min(raw_c.shape[0], int(duration / period_c))
        n_y = min(raw_y.shape[0], int(duration / period_y))
        self.x = WindowShifter(raw_x[:n_x], window_size_x, step_size_x, encoder_x)
        self.c = WindowShifter(raw_c[:n_c], window_size_c, step_size_c, encoder_c)
        self.y = WindowShifter(raw_y[:n_y], window_size_y, step_size_y, encoder_y)
        self.n_windows = min(self.x.n_windows, self.c.n_windows, self.y.n_windows)
        self.period_x, self.period_c, self.period_y = period_x, period_c, period_y

    def windows_at(self, shift_x: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # shift_x is a raw-sample offset for X; convert to the equivalent
        # physical-time offset, then to C's and Y's own sample counts, so
        # every side shifts by the same real time even when periods/window
        # sizes differ -- same conversion PairedWindowShifter.windows_at
        # already applies pairwise.
        shift_c = max(0, min(int(round(shift_x * self.period_x / self.period_c)), self.c.window_size - 1))
        shift_y = max(0, min(int(round(shift_x * self.period_x / self.period_y)), self.y.window_size - 1))
        return (self.x.windows_at(shift_x), self.c.windows_at(shift_c)), self.y.windows_at(shift_y)

    def random_shift(self, generator: torch.Generator = None) -> int:
        return self.x.random_shift(generator)


def _prep_shift_side(data, proc_type, proc_params):
    """Convert one raw side's data (X, Y, or a conditioning variable) plus
    its processor type/params into ``(raw_tensor, encoder_or_None)`` for a
    :class:`WindowShifter`. Shared by :func:`try_build_shift_windows_dataset`
    and :func:`try_build_shift_windows_dataset_dual_branch` so both build
    each side the same way.
    """
    # Checked first, regardless of proc_type: for a *mixed*
    # continuous+categorical concatenation (conditional/interaction's
    # raw_deferred branch), the concatenated joint array's proc_type is
    # still whichever type X itself is (e.g. 'continuous') even though
    # one of its blocks is categorical -- so this can't be nested inside
    # the `proc_type == 'categorical'` branch below, or a continuous-X
    # joint array would never see its block specs and Z's categorical
    # channels would reach the network as raw unencoded float labels.
    _block_specs = proc_params.get('_categorical_block_specs')
    if _block_specs is not None:
        # Caller (conditional.py/interaction.py's raw_deferred branch)
        # already relabeled each categorical channel block separately
        # to its own 0..n-1 range before concatenating X with the
        # conditioning variable -- relabeling the combined array again
        # here would be redundant at best (all-categorical) and would
        # infer one shared n_categories from the combined max at worst
        # (mixed). Cast to float (not long: a mixed block may carry
        # real continuous values that .long() would truncate --
        # make_categorical_encoder's own _encode already does its own
        # internal .long() cast on just the categorical blocks before
        # one-hot encoding, so nothing downstream needs this array
        # already integer-typed) and build a per-block encoder.
        import numpy as np
        raw = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
        raw = raw.float()
        encoder = make_multi_categorical_encoder(_block_specs, proc_params.get('encoding', 'majority_vote'))
        return raw, encoder
    if proc_type == 'categorical':
        from neural_mi.data.temporal import relabel_categorical_data
        arr = relabel_categorical_data(data)  # (T, C) int32, same
        # relabeling CategoricalWindowDataset itself would apply.
        raw = torch.as_tensor(arr, dtype=torch.long)
        n_categories = int(raw.max().item()) + 1 if raw.numel() > 0 else 1
        encoder = make_categorical_encoder(n_categories, proc_params.get('encoding', 'majority_vote'))
        return raw, encoder
    import numpy as np
    raw = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data), dtype=torch.float32)
    if raw.ndim == 1:
        raw = raw.unsqueeze(-1)
    return raw, None


def try_build_shift_windows_dataset(x_data, y_data, params: dict, data_device: str = 'cpu'):
    """Build a ``shift_windows``-capable dataset for a regular-grid
    (`continuous`/`categorical`) pair from raw, unwindowed ``x_data``/
    ``y_data``, or return ``None`` if this pair doesn't qualify (caller
    should fall back to :func:`~neural_mi.data.handler.create_dataset`).

    Mutates ``params`` in place with ``leak_check_window_size``/
    ``leak_check_step`` when it builds a dataset, since the returned
    dataset has no ``window_manager`` for :class:`Trainer`'s blocked-split
    leakage check to read geometry from otherwise. Shared by
    ``task.py::run_training_task`` and ``precision.py`` so both build this
    kind of dataset the same way.
    """
    _shift_proc_x = params.get('processor_type_x')
    _shift_proc_y = params.get('processor_type_y')
    if _shift_proc_y is None:
        _shift_proc_y = _shift_proc_x  # None -> "inherit X", create_dataset's own convention
    if not (params.get('shift_windows') and _shift_proc_x in _REGULAR_GRID_PROCESSOR_TYPES
            and _shift_proc_y in _REGULAR_GRID_PROCESSOR_TYPES):
        return None

    # Builds its own PairedDataset directly from an initial (shift=0)
    # windowing and stashes the raw-array shifter Trainer.train() uses to
    # reslice every epoch. Never cached: like a temporal dataset, its
    # .data is mutated in place across the run.
    from neural_mi.data.static import StaticDataset
    from neural_mi.data.handler import PairedDataset

    _wp_x = params.get('processor_params_x') or {}
    _wp_y = params.get('processor_params_y') or _wp_x
    _window_size = _wp_x.get('window_size')
    _step_size = _wp_x.get('step_size') or _window_size
    if _window_size is None:
        raise ValueError(
            "shift_windows=True requires processor_params_x={'window_size': ...} "
            "(optionally 'step_size', defaults to window_size)."
        )
    # window_size/step_size are in the shared WindowManager unit -- seconds
    # if 'sample_rate' is set, otherwise raw samples (period=1). Convert to
    # each side's own sample count independently, since X and Y may use
    # different sample rates (see PairedWindowShifter's docstring for why
    # the truncation/shift logic must then work in a common *duration*,
    # not a common raw sample count).
    _rate_x = _wp_x.get('sample_rate')
    _rate_y = _wp_y.get('sample_rate')
    _period_x = 1.0 / _rate_x if _rate_x else 1.0
    _period_y = 1.0 / _rate_y if _rate_y else 1.0
    _window_size_x = seconds_to_samples(_window_size, _period_x)
    _step_size_x = seconds_to_samples(_step_size, _period_x)
    _window_size_y = seconds_to_samples(_window_size, _period_y)
    _step_size_y = seconds_to_samples(_step_size, _period_y)
    # dataset has no window_manager (it's a plain PairedDataset of
    # pre-shifted StaticDatasets) -- pass the geometry explicitly so the
    # blocked-split leakage check in Trainer._create_blocked_split still
    # has something to validate against.
    params['leak_check_window_size'] = _window_size_x
    params['leak_check_step'] = _step_size_x

    _raw_x, _encoder_x = _prep_shift_side(x_data, _shift_proc_x, _wp_x)
    _raw_y, _encoder_y = _prep_shift_side(y_data, _shift_proc_y, _wp_y)
    _shifter = PairedWindowShifter(
        _raw_x, _raw_y, _window_size_x, _step_size_x, _window_size_y, _step_size_y,
        encoder_x=_encoder_x, encoder_y=_encoder_y, period_x=_period_x, period_y=_period_y,
    )
    _x0, _y0 = _shifter.windows_at(0)
    dataset = PairedDataset(StaticDataset(_x0, data_device=data_device),
                            StaticDataset(_y0, data_device=data_device))
    dataset._window_shifter = _shifter
    return dataset


def try_build_shift_windows_dataset_dual_branch(x_data: Tuple, y_data, params: dict,
                                                data_device: str = 'cpu'):
    """``try_build_shift_windows_dataset``'s sibling for
    ``mode='conditional'(align='dual_branch')``: ``x_data`` is ``(a_raw,
    c_raw)`` -- X and the conditioning variable C, each raw and unwindowed,
    kept separate (never concatenated -- that's dual_branch's entire
    premise: C genuinely has its own window geometry). Returns ``None`` if
    this triple doesn't qualify (caller falls back to eager
    ``create_dataset``), matching the sibling builder's contract exactly.

    C's own processor type/params are read from
    ``params['_dual_branch_c_processor_type']``/
    ``'_dual_branch_c_processor_params']`` (populated by
    ``run_conditional_mi``'s ``align='dual_branch'`` + ``raw_deferred``
    sub-path) rather than a function parameter, since this is called from
    ``task.py::run_training_task`` with the same generic ``(x_data, y_data,
    params)`` signature every other deferred-windowing builder uses.
    """
    a_raw, c_raw = x_data
    _shift_proc_x = params.get('processor_type_x')
    _shift_proc_c = params.get('_dual_branch_c_processor_type')
    if _shift_proc_c is None:
        _shift_proc_c = _shift_proc_x
    _shift_proc_y = params.get('processor_type_y')
    if _shift_proc_y is None:
        _shift_proc_y = _shift_proc_x
    if not (params.get('shift_windows') and _shift_proc_x in _REGULAR_GRID_PROCESSOR_TYPES
            and _shift_proc_c in _REGULAR_GRID_PROCESSOR_TYPES
            and _shift_proc_y in _REGULAR_GRID_PROCESSOR_TYPES):
        return None

    from neural_mi.data.static import StaticDataset
    from neural_mi.data.handler import PairedDataset

    _wp_x = params.get('processor_params_x') or {}
    _wp_c = params.get('_dual_branch_c_processor_params') or {}
    _wp_y = params.get('processor_params_y') or _wp_x
    _window_size_x_raw = _wp_x.get('window_size')
    _step_size_x_raw = _wp_x.get('step_size') or _window_size_x_raw
    # C's own window_size -- unlike try_build_shift_windows_dataset's X/Y
    # (which share one window_size, X's), C is expected to genuinely
    # differ; falls back to X's only if C's own params don't set one.
    _window_size_c_raw = _wp_c.get('window_size') or _window_size_x_raw
    _step_size_c_raw = _wp_c.get('step_size') or _window_size_c_raw
    if _window_size_x_raw is None:
        raise ValueError(
            "shift_windows=True requires processor_params_x={'window_size': ...} "
            "(optionally 'step_size', defaults to window_size)."
        )
    _rate_x = _wp_x.get('sample_rate')
    _rate_c = _wp_c.get('sample_rate')
    _rate_y = _wp_y.get('sample_rate')
    _period_x = 1.0 / _rate_x if _rate_x else 1.0
    _period_c = 1.0 / _rate_c if _rate_c else 1.0
    _period_y = 1.0 / _rate_y if _rate_y else 1.0
    _window_size_x = seconds_to_samples(_window_size_x_raw, _period_x)
    _step_size_x = seconds_to_samples(_step_size_x_raw, _period_x)
    _window_size_c = seconds_to_samples(_window_size_c_raw, _period_c)
    _step_size_c = seconds_to_samples(_step_size_c_raw, _period_c)
    _window_size_y = seconds_to_samples(_window_size_x_raw, _period_y)
    _step_size_y = seconds_to_samples(_step_size_x_raw, _period_y)
    # dataset has no window_manager -- report X's own geometry for the
    # blocked-split leakage check, the same precedent
    # try_build_shift_windows_dataset already uses when X/Y differ.
    params['leak_check_window_size'] = _window_size_x
    params['leak_check_step'] = _step_size_x

    _raw_a, _encoder_a = _prep_shift_side(a_raw, _shift_proc_x, _wp_x)
    _raw_c, _encoder_c = _prep_shift_side(c_raw, _shift_proc_c, _wp_c)
    _raw_y, _encoder_y = _prep_shift_side(y_data, _shift_proc_y, _wp_y)
    _shifter = DualBranchWindowShifter(
        _raw_a, _raw_c, _raw_y,
        _window_size_x, _step_size_x, _window_size_c, _step_size_c, _window_size_y, _step_size_y,
        encoder_x=_encoder_a, encoder_c=_encoder_c, encoder_y=_encoder_y,
        period_x=_period_x, period_c=_period_c, period_y=_period_y,
    )
    (_a0, _c0), _y0 = _shifter.windows_at(0)
    dataset = PairedDataset(StaticDataset((_a0, _c0), data_device=data_device),
                            StaticDataset(_y0, data_device=data_device))
    dataset._window_shifter = _shifter
    return dataset


def n_windows_if_deferred(x_data, y_data, params: dict) -> int:
    """Window count for an (x_data, y_data) pair once windowing is deferred
    (raw, 2-D x_data + ``shift_windows`` requested for a regular-grid pair),
    or ``x_data.shape[0]`` unchanged otherwise (already windowed, or shift
    not requested/reachable for this pair).

    Used wherever an analytical, shift-invariant "how many windows will
    this pair actually produce" count is needed before any orchestration-
    specific windowing (a shared train/test split, a bias-correction
    chunk boundary, ...) has happened yet -- reusing the exact same
    construction :func:`try_build_shift_windows_dataset` would build, so
    this can't drift from what actually gets built later.
    """
    if not (getattr(x_data, 'ndim', None) == 2 and params.get('shift_windows')):
        return x_data.shape[0]
    if y_data is None:
        _wp_x = params.get('processor_params_x') or {}
        _window_size = _wp_x.get('window_size')
        _step_size = _wp_x.get('step_size') or _window_size
        _period_x = 1.0 / _wp_x['sample_rate'] if _wp_x.get('sample_rate') else 1.0
        return safe_n_windows(x_data.shape[0], seconds_to_samples(_window_size, _period_x),
                              seconds_to_samples(_step_size, _period_x))
    _throwaway = try_build_shift_windows_dataset(x_data, y_data, dict(params), data_device='cpu')
    return len(_throwaway) if _throwaway is not None else x_data.shape[0]


def chunk_window_range_to_raw(lo: int, hi: int, window_size: int, step_size: int) -> Tuple[int, int]:
    """Raw sample range ``[start, end)`` covering exactly ``hi - lo``
    windows' worth of content for a contiguous window-index chunk
    ``[lo, hi)``, plus the same ``window_size - 1`` margin
    :func:`safe_n_windows` reserves globally -- so
    ``safe_n_windows(end - start, window_size, step_size) == hi - lo``
    exactly, and the chunk produces exactly ``hi - lo`` windows under any
    per-epoch shift in ``[0, window_size)``, not just at shift 0.
    """
    start = lo * step_size
    end = (hi - 1) * step_size + 2 * window_size - 1
    return start, end


def chunk_window_range_to_time(lo: int, hi: int, window_size: float, step_size: float) -> Tuple[float, float]:
    """Raw time range ``[start, end)`` covering exactly ``hi - lo`` windows'
    worth of content for a contiguous window-index chunk ``[lo, hi)``, using
    the same ``2*window_size`` margin
    ``PairedTemporalDataset._reserve_shift_margin`` reserves (Phase 0's
    fixed-grid time-shift design) -- deliberately NOT
    :func:`chunk_window_range_to_raw`'s ``window_size - 1`` margin
    (``safe_n_windows``/``WindowShifter``'s, for the regular-grid
    raw-array-slicing case), which is off by one time unit against this
    margin convention. Pair with an explicit ``t_start=0.0,
    t_end=(end - start)`` on the chunk's own ``PairedTemporalDataset`` (via
    ``processor_params_x``) so its base span matches this function's
    assumption exactly, rather than a shorter, data-dependent span derived
    from wherever the sliced spikes actually happen to fall.
    """
    start = lo * step_size
    end = (hi - 1) * step_size + 2 * window_size
    return start, end


def spike_shift_grid_info(x_data: List[np.ndarray], y_data: List[np.ndarray],
                          params: dict) -> Tuple[int, float, float, float]:
    """``(n_windows, base_t_start, window_size, step_size)`` for a raw
    (ragged per-neuron spike-time list) X/Y pair once ``shift_time``
    windowing is deferred to a ``mode='rigorous'`` gamma-chunk.

    Builds a throwaway ``PairedTemporalDataset`` and primes its shift grid
    (``PairedTemporalDataset._reserve_shift_margin`` / ``time_shift`` --
    the Phase 0 fix that makes ``shift_time`` genuinely re-tile instead of
    canceling the offset out) to read off the exact margin-reserved window
    count and the grid's shared base start time, rather than re-deriving
    that ``2*window_size`` margin arithmetic here a second time.
    """
    from neural_mi.data.temporal import SpikeWindowDataset
    from neural_mi.data.handler import PairedTemporalDataset

    _wp_x = params.get('processor_params_x') or {}
    window_size = _wp_x.get('window_size')
    step_size = _wp_x.get('step_size') or window_size
    if window_size is None:
        raise ValueError(
            "shift_time=True with mode='rigorous' for a spike+spike pair "
            "requires processor_params_x={'window_size': ...} (optionally "
            "'step_size', defaults to window_size)."
        )
    x_ds = SpikeWindowDataset(x_data)
    y_ds = SpikeWindowDataset(y_data)
    paired = PairedTemporalDataset(x_ds, y_ds, window_size=window_size, step_size=step_size)
    paired.time_shift(offset_x=0.0, offset_y=0.0)
    return len(paired), paired._base_t_start, window_size, paired.window_manager.resolve_step()


def slice_spike_data_to_time_range(spike_data: List[np.ndarray], t_start: float,
                                   t_end: float) -> List[np.ndarray]:
    """Slice a ragged per-neuron spike-time list to the absolute time range
    ``[t_start, t_end)``, re-zeroed so the returned list's own t=0 matches
    ``t_start`` -- the spike-data analogue of the raw-sample slice
    :func:`chunk_window_range_to_raw`'s output drives for regular-grid data,
    so a gamma-chunk built from it looks like a genuine sub-recording
    starting at ``t_start``.
    """
    sliced = []
    for neuron_times in spike_data:
        neuron_times = np.asarray(neuron_times)
        lo = np.searchsorted(neuron_times, t_start, side='left')
        hi = np.searchsorted(neuron_times, t_end, side='left')
        sliced.append(neuron_times[lo:hi] - t_start)
    return sliced
