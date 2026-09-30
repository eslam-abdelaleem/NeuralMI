# neural_mi/data/shift_windowing.py
"""Per-epoch window shifting, and the reslicing mechanism behind ``shift_windows``.

Two mechanisms re-tile a windowed dataset from a fresh random offset every
epoch, so training sees many sets of window boundaries.

``Training(shift_windows=True)`` serves the regular-grid family, `continuous`
and `categorical` streams in any combination. `WindowShifter` holds the raw,
unwindowed `(T, n_channels)` array and re-derives the windows with
`torch.Tensor.unfold` at an integer sample offset drawn uniformly from
`[0, window_size)`. It always produces exactly `n_windows` windows, so the
window count, and every train, test and evaluation index computed against it,
stays the same from shift to shift, and only the content at each index changes.
Categorical streams are re-encoded after the reslice
(`make_categorical_encoder`, matching `CategoricalWindowDataset`'s three
encodings). Streams with different sample rates each convert the window, step
and shift to their own sample counts, so they stay aligned in real time. A
stream without `sample_rate` takes its period from its time vector
(`stream_period`) when that clock is regular. When a clock is irregular, or two
clocks start at different times, the reslice cannot line the streams up by
sample number, and the builder declines with a logged reason. The pairing then
falls back to the eager route, which checks every window against the
timestamps and starts the grid at the latest start among the streams.
`BundleShifter` holds the alignment for any number of streams, and its
subclasses pack the streams into the estimator's roles: `PairedWindowShifter`
for X and Y, `DualBranchWindowShifter` for the X, C and Y of
`align='dual_branch'`.

``Training(shift_time=True)`` serves spike streams, and a spike stream paired
with a regular one that has `sample_rate` set (`mixed_pair_sample_rate_ok`), so
that a shift means the same time on both sides. It slides the window grid over
raw data that is never modified, by interpolation for continuous streams and a
`searchsorted` rebin for spikes, and rebuilds the windows every epoch. A margin
of `2 * window_size` is reserved once before the split
(`AlignedStreams._reserve_shift_margin`), so the window count stays fixed
across every offset in `[0, window_size)`, and a recording too short for it
raises at once.

Which modes each mechanism reaches is set by the `_SHIFT_*_SAFE_MODES` tuples in
`neural_mi/run.py`: shifting needs the raw data to reach each training task and
must not disturb a comparison between runs. Transfer entropy and the named
quantities built from offsets are never shifted. Their rows are cut at every
step, one step apart, so a shift by `s` would only relabel row `i` as row
`s + i`.

Evaluation never sees a shift. The Trainer measures `test_mi` and `train_mi`,
and everything derived from them, against a snapshot of the data taken before
any shift, using the original index arrays. Since a `shift_time` training window
can drift by up to a full `window_size` from that snapshot, the blocked-split
leak check doubles its margin to `2 * window_size` while `shift_time` is on.

See `shift_family` for which mechanism applies to which pair of processors.
"""
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from neural_mi.logger import logger


_REGULAR_GRID_PROCESSOR_TYPES = frozenset({'continuous', 'categorical'})


def shift_family(processor_type_x: Optional[str], processor_type_y_effective: Optional[str]) -> Optional[str]:
    """Classify an (X, Y) processor-type pair for per-epoch shift purposes.

    ``processor_type_y_effective`` must already be resolved to its effective
    value (``processor_type_y if processor_type_y is not None else
    processor_type_x``, ``create_dataset``'s own "None means inherit X"
    convention, ``data/handler.py``) by the caller.

    Returns
    -------
    'regular' : both sides in {'continuous', 'categorical'} (need not
        match each other, continuous+categorical is fine). Supports
        ``Training(shift_windows=True)``.
    'spike' : both sides 'spike'. Supports ``Training(shift_time=True)`` via
        the ``PairedTemporalDataset``/``time_shift`` machinery.
    'mixed' : one side 'spike', the other in {'continuous', 'categorical'}.
        Supports ``shift_time`` *only if* the regular-grid side's
        ``processor_params`` sets ``sample_rate``, required so a shift
        value means the same real time on both sides (see
        ``mixed_pair_sample_rate_ok``). Callers must check that separately;
        this function only classifies the pair, it doesn't gate on it.
    None : either side is ``None`` (static/pre-processed data, no raw
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
                              processor_params_y: Optional[dict]) -> bool:
    """For a ``shift_family(...) == 'mixed'`` pair, check whether the
    regular-grid side (`continuous`/`categorical`) has `sample_rate` set.

    Without it, that side's shift values are in raw sample-index units
    while spike's are natively in seconds. The same numeric shift would
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


def make_multi_categorical_encoder(block_specs: List[Tuple[int, Optional[int]]],
                                   encoding: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Block-aware analogue of :func:`make_categorical_encoder` for a raw,
    concatenated array whose channel blocks may have different
    ``n_categories``, e.g. `conditional`/`interaction`'s X and Z/W,
    relabeled to their own correct ``0..n-1`` range and concatenated
    separately (see ``run_conditional_mi``/``run_interaction_information``'s
    ``raw_deferred`` branch), instead of relabeled *after* concatenation
    (which would infer one shared ``n_categories`` from the combined
    array's max value, silently conflating the two blocks' category counts).

    ``block_specs`` : list of ``(n_channels, n_categories)``, one entry per
    channel block, in the same channel order the blocks were concatenated.
    ``n_categories=None`` marks a *continuous* block, passed through
    unencoded, at its native ``window_size``, for a mixed continuous +
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
    along the channel axis, mirroring the same broadcast
    ``run._reshape_categorical_w_for_conditional``'s caller already applies
    for a lone categorical conditioning variable
    (``w_data.expand(-1, -1, x_data.shape[2])``). For ``full_trajectory``,
    every block already lands at ``window_size`` natively, so this broadcast
    is a no-op. For an all-categorical ``block_specs`` (no continuous
    blocks), every block is already the same width regardless, so the
    broadcast step is a no-op there too. This function's existing
    all-categorical behavior is unchanged.
    """
    if encoding not in ('majority_vote', 'probability', 'full_trajectory'):
        raise ValueError(
            f"Unknown encoding '{encoding}'. Expected 'majority_vote', "
            f"'probability' or 'full_trajectory'."
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
                f"block_specs channel counts sum to {ch0} and raw_windows has "
                f"{n_channels} channels. block_specs must partition every "
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
        `seconds_to_samples`), same meaning as
        ``Processing(..., x_params={'window_size':..., 'step_size':...})``
        when no `sample_rate` is set.
    encoder : callable, optional
        Applied to each freshly-resliced ``(n_windows, C, window_size)``
        raw tensor before it's returned, e.g. `make_categorical_encoder`'s
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


def stream_spec(raw, window_size, step_size, encoder=None, period=1.0):
    """One stream's entry for :class:`BundleShifter`.

    ``window_size``/``step_size`` are in this stream's own raw-sample units;
    ``period`` is 1/sample_rate, or 1.0 when the stream has no sample rate.
    """
    return dict(raw=raw, window_size=window_size, step_size=step_size,
                encoder=encoder, period=period)


class BundleShifter:
    """Any number of named streams, re-tiled together in real time.

    One shift is drawn against the first (reference) stream and converted into
    every other stream's own sample units, so all of them move by the same real
    time even when their sample rates, window sizes and steps differ. The
    streams are first truncated to a common *duration* instead of a common raw
    sample count, because those differ whenever the periods do.

    Returns the freshly resliced windows by name. Packing those names into the
    roles an estimator wants, ``(x, y)`` for a plain pair, ``((x, c), y)``
    when X and C travel together as one branch, is the caller's job, so any
    number of streams needs only this one implementation.

    """

    def __init__(self, streams, pack=None):
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
            default returns them by name; the two role-packing subclasses below
            supply the ``(x, y)`` and ``((x, c), y)`` shapes ``Trainer``
            expects. Packing is the only thing that ever differed between the
            two stream counts, so it is the only thing they still define.
        """
        streams = OrderedDict(streams)
        if not streams:
            raise ValueError("BundleShifter needs at least one stream.")
        periods = OrderedDict((name, spec.get('period', 1.0)) for name, spec in streams.items())
        duration = min(spec['raw'].shape[0] * periods[name]
                       for name, spec in streams.items())
        self.shifters = OrderedDict()
        for name, spec in streams.items():
            period = periods[name]
            n_keep = min(spec['raw'].shape[0], int(duration / period))
            self.shifters[name] = WindowShifter(
                spec['raw'][:n_keep], spec['window_size'], spec['step_size'],
                spec.get('encoder'),
            )
        self.periods = periods
        self.reference = next(iter(self.shifters))
        self.n_windows = min(sh.n_windows for sh in self.shifters.values())
        self._pack = pack

    def windows_at(self, shift):
        """Reslice every stream at ``shift``, given in the reference's samples."""
        reference_period = self.periods[self.reference]
        out = OrderedDict()
        for name, shifter in self.shifters.items():
            if name == self.reference:
                own_shift = shift
            else:
                own_shift = int(round(shift * reference_period / self.periods[name]))
                own_shift = max(0, min(own_shift, shifter.window_size - 1))
            out[name] = shifter.windows_at(own_shift)
        return self._pack(out) if self._pack is not None else out

    def random_shift(self, generator: torch.Generator = None) -> int:
        return self.shifters[self.reference].random_shift(generator)


class PairedWindowShifter(BundleShifter):
    """An X and a Y stream, returned as ``(x, y)``.

    Each side may have its own sample rate, and therefore its own
    ``window_size``/``step_size`` in raw-sample units and its own ``period``
    (1/sample_rate, or 1.0 when none is set). :class:`BundleShifter` does the
    alignment; this only names the two streams and packs them.
    """

    def __init__(self, raw_x: torch.Tensor, raw_y: torch.Tensor,
                 window_size_x: int, step_size_x: int,
                 window_size_y: Optional[int] = None, step_size_y: Optional[int] = None,
                 encoder_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 encoder_y: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 period_x: float = 1.0, period_y: float = 1.0):
        super().__init__(
            OrderedDict((
                ('x', stream_spec(raw_x, window_size_x, step_size_x, encoder_x, period_x)),
                ('y', stream_spec(raw_y, window_size_y or window_size_x,
                                  step_size_y or step_size_x, encoder_y, period_y)),
            )),
            pack=lambda windows: (windows['x'], windows['y']),
        )


class DualBranchWindowShifter(BundleShifter):
    """X, C and Y streams, returned as ``((x, c), y)``.

    X and C travel together as one branch, ``dual_branch``'s premise:
    C keeps its own, generally different, window geometry and is never
    concatenated onto X. That nested shape is what ``StaticDataset`` and
    ``Trainer``'s live shift-application code already store and index as the
    X-role data, so it is preserved exactly.
    """

    def __init__(self, raw_x: torch.Tensor, raw_c: torch.Tensor, raw_y: torch.Tensor,
                 window_size_x: int, step_size_x: int,
                 window_size_c: int, step_size_c: int,
                 window_size_y: Optional[int] = None, step_size_y: Optional[int] = None,
                 encoder_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 encoder_c: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 encoder_y: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 period_x: float = 1.0, period_c: float = 1.0, period_y: float = 1.0):
        super().__init__(
            OrderedDict((
                ('x', stream_spec(raw_x, window_size_x, step_size_x, encoder_x, period_x)),
                ('c', stream_spec(raw_c, window_size_c, step_size_c, encoder_c, period_c)),
                ('y', stream_spec(raw_y, window_size_y or window_size_x,
                                  step_size_y or step_size_x, encoder_y, period_y)),
            )),
            pack=lambda windows: ((windows['x'], windows['c']), windows['y']),
        )


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


def _build_shifted_dataset(stream_defs, params, data_device, pack_roles, shifter_cls_args):
    """Shared tail of the shift-windows builders.

    Everything from "per-stream units" to "a dataset with a shifter attached"
    is identical whether there are two streams or three: convert each stream's
    window geometry into its own sample units, record the reference stream's
    geometry for the blocked-split leakage check, build the shifter, and take
    the shift-0 windowing as the dataset's initial content.

    What the callers keep is what genuinely differs: which params dict each
    stream reads from, and how the resulting streams pack into the estimator's
    two roles.

    Parameters
    ----------
    stream_defs : mapping of str to tuple
        ``name -> (raw_data, processor_type, processor_params, window_size,
        step_size)``, in order, the first being the reference.
    params : dict
        Mutated in place with ``leak_check_window_size``/``leak_check_step``,
        since the returned dataset has no ``window_manager`` for
        :class:`Trainer`'s leakage check to read geometry from.
    pack_roles : callable
        Maps the shifter's name-keyed windows to ``(x_role, y_role)``.
    shifter_cls_args : callable
        Builds the role-packing shifter that gets attached to the dataset, so
        ``Trainer`` keeps receiving the two-tuple contract it expects.
    """
    from neural_mi.data.static import StaticDataset
    from neural_mi.data.handler import PairedDataset

    # Which clock each stream is on. 'y' carries its own; a conditioning stream
    # shares X's, which run.py enforces before the data gets here.
    times = {name: (params.get('y_time') if name == 'y' else params.get('x_time'))
             for name in stream_defs}
    periods = OrderedDict()
    for name, (_, _, proc_params, _, _) in stream_defs.items():
        period = stream_period(proc_params, times[name])
        if period is None:
            logger.warning(
                f"shift_windows is off for this run because stream {name!r} was given "
                f"a time vector whose sampling is irregular. The reslice route needs "
                f"one period for the whole recording. The eager route builds the "
                f"windows here and checks each one against the timestamps. Set {{'sample_rate': ...}} in that stream's processor "
                f"parameters to state a period and keep the reslice route."
            )
            return None
        periods[name] = period
    # Every stream's window boundaries have to mean the same real time. The
    # eager route earns that by starting the grid at the latest start among the
    # streams; this one indexes from each array's own first sample, so clocks
    # that begin at different times go back to the eager route.
    starts = [float(np.asarray(t, dtype=float).ravel()[0])
              for t in times.values() if t is not None and len(np.asarray(t)) > 0]
    if starts and max(starts) - min(starts) > 0.5 * max(periods.values()):
        logger.warning(
            f"shift_windows is off for this run because the streams' time vectors "
            f"start {max(starts) - min(starts):.4g} s apart. The reslice route lines "
            f"them up by sample number. The eager route builds the windows here and "
            f"starts the grid at the latest start among them."
        )
        return None

    geometry = OrderedDict()
    for name, (raw_data, proc_type, proc_params, window_size, step_size) in stream_defs.items():
        period = periods[name]
        raw, encoder = _prep_shift_side(raw_data, proc_type, proc_params)
        geometry[name] = stream_spec(
            raw, seconds_to_samples(window_size, period),
            seconds_to_samples(step_size, period), encoder, period,
        )
    reference = next(iter(geometry.values()))
    params['leak_check_window_size'] = reference['window_size']
    params['leak_check_step'] = reference['step_size']

    shifter = shifter_cls_args(geometry)
    x_role, y_role = pack_roles(shifter)
    dataset = PairedDataset(StaticDataset(x_role, data_device=data_device),
                            StaticDataset(y_role, data_device=data_device))
    dataset._window_shifter = shifter
    return dataset


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
    _wp_x = params.get('processor_params_x') or {}
    _wp_y = params.get('processor_params_y') or _wp_x
    _window_size = _wp_x.get('window_size')
    if _window_size is None:
        raise ValueError(
            "shift_windows=True requires Processing(x_params={'window_size': ...}). "
            "'step_size' is optional and defaults to window_size."
        )
    _step_size = resolve_step_size(_window_size, _wp_x.get('step_size'))
    # window_size/step_size are in the shared WindowManager unit -- seconds if
    # 'sample_rate' is set, otherwise raw samples (period=1). Each stream is
    # converted into its own sample count by _build_shifted_dataset, since X and
    # Y may use different sample rates; see BundleShifter for why the truncation
    # then works in a common duration instead of a common raw sample count.
    return _build_shifted_dataset(
        OrderedDict((
            ('x', (x_data, _shift_proc_x, _wp_x, _window_size, _step_size)),
            ('y', (y_data, _shift_proc_y, _wp_y, _window_size, _step_size)),
        )),
        params, data_device,
        pack_roles=lambda sh: sh.windows_at(0),
        shifter_cls_args=lambda g: PairedWindowShifter(
            g['x']['raw'], g['y']['raw'],
            g['x']['window_size'], g['x']['step_size'],
            g['y']['window_size'], g['y']['step_size'],
            encoder_x=g['x']['encoder'], encoder_y=g['y']['encoder'],
            period_x=g['x']['period'], period_y=g['y']['period'],
        ),
    )


def try_build_shift_windows_dataset_dual_branch(x_data: Tuple, y_data, params: dict,
                                                data_device: str = 'cpu'):
    """``try_build_shift_windows_dataset``'s sibling for
    ``mode='conditional'(align='dual_branch')``: ``x_data`` is ``(a_raw,
    c_raw)``, X and the conditioning variable C, each raw and unwindowed,
    kept separate (never concatenated. That's dual_branch's entire
    premise: C genuinely has its own window geometry). Returns ``None`` if
    this triple doesn't qualify (caller falls back to eager
    ``create_dataset``), matching the sibling builder's contract exactly.

    C's own processor type/params are read from
    ``params['_dual_branch_c_processor_type']``/
    ``'_dual_branch_c_processor_params']`` (populated by
    ``run_conditional_mi``'s ``align='dual_branch'`` + ``raw_deferred``
    sub-path) instead of a function parameter, since this is called from
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

    _wp_x = params.get('processor_params_x') or {}
    _wp_c = params.get('_dual_branch_c_processor_params') or {}
    _wp_y = params.get('processor_params_y') or _wp_x
    _window_size_x_raw = _wp_x.get('window_size')
    if _window_size_x_raw is None:
        raise ValueError(
            "shift_windows=True requires Processing(x_params={'window_size': ...}). "
            "'step_size' is optional and defaults to window_size."
        )
    _step_size_x_raw = resolve_step_size(_window_size_x_raw, _wp_x.get('step_size'))
    # C's own window_size -- unlike try_build_shift_windows_dataset's X/Y
    # (which share one window_size, X's), C is expected to genuinely
    # differ; falls back to X's only if C's own params don't set one.
    _window_size_c_raw = _wp_c.get('window_size') or _window_size_x_raw
    # C resolves its fractional step against its own window, which is the
    # whole reason C carries separate params here.
    _step_size_c_raw = resolve_step_size(_window_size_c_raw, _wp_c.get('step_size'))
    return _build_shifted_dataset(
        OrderedDict((
            ('x', (a_raw, _shift_proc_x, _wp_x, _window_size_x_raw, _step_size_x_raw)),
            ('c', (c_raw, _shift_proc_c, _wp_c, _window_size_c_raw, _step_size_c_raw)),
            ('y', (y_data, _shift_proc_y, _wp_y, _window_size_x_raw, _step_size_x_raw)),
        )),
        params, data_device,
        # X and C travel together as one branch; only the packing differs from
        # the pair builder's.
        pack_roles=lambda sh: sh.windows_at(0),
        shifter_cls_args=lambda g: DualBranchWindowShifter(
            g['x']['raw'], g['c']['raw'], g['y']['raw'],
            g['x']['window_size'], g['x']['step_size'],
            g['c']['window_size'], g['c']['step_size'],
            g['y']['window_size'], g['y']['step_size'],
            encoder_x=g['x']['encoder'], encoder_c=g['c']['encoder'],
            encoder_y=g['y']['encoder'],
            period_x=g['x']['period'], period_c=g['c']['period'],
            period_y=g['y']['period'],
        ),
    )


def categorical_to_channel_layout(windows, n_categories, encoding):
    """Put an encoded categorical stream on the ``(N, C, W)`` layout X uses.

    A categorical encoder spends the trailing axis on categories, not on time:
    ``majority_vote`` and ``probability`` collapse the window to one slot per
    category, and ``full_trajectory`` interleaves them as
    ``timepoint * n_categories + category``. Either way the result cannot be
    concatenated onto a continuous X along the channel axis until the two axes
    mean the same thing again.

    Returns ``(N, C*n_categories, W)``, with ``W = 1`` for the collapsing
    encodings; the caller broadcasts that up to X's window width. This is the
    same re-layout the eagerly-windowed path applies, shared so the shifted and
    unshifted routes agree on it instead of each having their own.
    """
    n, c, last = windows.shape
    if encoding in ('majority_vote', 'probability'):
        return windows.reshape(n, c * n_categories, 1)
    if encoding == 'full_trajectory':
        w = last // n_categories
        return windows.reshape(n, c, w, n_categories).permute(0, 1, 3, 2).reshape(
            n, c * n_categories, w)
    raise ValueError(
        f"Unknown categorical encoding {encoding!r}, cannot lay it out for "
        f"concatenation onto X."
    )


def align_trailing_axis(windows, width):
    """Broadcast a collapsed trailing axis up to ``width`` so a concat works."""
    if windows.shape[-1] == width:
        return windows
    if windows.shape[-1] == 1:
        return windows.expand(-1, -1, width)
    raise ValueError(
        f"Cannot align a trailing axis of {windows.shape[-1]} to {width}: only a "
        f"collapsed axis of 1 broadcasts."
    )

# How a tuple X-role's streams pack into the estimator's two roles. Set by
# conditional.py/interaction.py in ``params['_shift_pack']``; see
# :func:`try_build_shift_windows_dataset_tuple`.
SHIFT_PACK_DUAL_BRANCH = 'dual_branch'   # ((x, c), y): C keeps its own geometry
SHIFT_PACK_CONCAT = 'concat'             # (cat(x, w), y): the joint X-with-W leg
SHIFT_PACK_SECOND = 'second'             # (w, y): the marginal W-alone leg
SHIFT_PACK_FIRST = 'first'               # (x, y): the marginal X-alone leg


def try_build_shift_windows_dataset_tuple(x_data, y_data, params: dict,
                                          data_device: str = 'cpu'):
    """Build a shifted dataset for a tuple X-role: ``(x_raw, second_raw)``.

    The second stream is windowed *as its own stream*, with its own processor
    type, its own relabeling and its own encoder, and is combined with X only
    after windowing. Concatenating them raw beforehand would force one shared
    encoder over two blocks with possibly different category counts. One shift
    moves every stream together, so nothing needs to be merged early to stay
    in step.

    ``params['_shift_pack']`` chooses how the windowed streams become the two
    roles the estimator consumes; see the ``SHIFT_PACK_*`` constants.

    Returns ``None`` if the triple doesn't qualify, matching its sibling
    builders' contract.
    """
    x_raw, second_raw = x_data
    pack_kind = params.get('_shift_pack', SHIFT_PACK_DUAL_BRANCH)
    _proc_x = params.get('processor_type_x')
    _proc_second = (params.get('_second_processor_type')
                    or params.get('_dual_branch_c_processor_type') or _proc_x)
    _proc_y = params.get('processor_type_y') or _proc_x
    if not (params.get('shift_windows')
            and _proc_x in _REGULAR_GRID_PROCESSOR_TYPES
            and _proc_second in _REGULAR_GRID_PROCESSOR_TYPES
            and _proc_y in _REGULAR_GRID_PROCESSOR_TYPES):
        return None

    _wp_x = params.get('processor_params_x') or {}
    _wp_second = (params.get('_second_processor_params')
                  or params.get('_dual_branch_c_processor_params') or {})
    _wp_y = params.get('processor_params_y') or _wp_x
    _window_size = _wp_x.get('window_size')
    if _window_size is None:
        raise ValueError(
            "shift_windows=True requires Processing(x_params={'window_size': ...}). "
            "'step_size' is optional and defaults to window_size."
        )
    _step_size = resolve_step_size(_window_size, _wp_x.get('step_size'))
    # The second stream keeps its own window_size only where that is its
    # premise (dual_branch); a W that will be concatenated onto X after
    # windowing has to land on X's window axis to be concatenable.
    if pack_kind == SHIFT_PACK_DUAL_BRANCH:
        _window_second = _wp_second.get('window_size') or _window_size
        _step_second = resolve_step_size(_window_second, _wp_second.get('step_size'))
    else:
        _window_second, _step_second = _window_size, _step_size

    # Either side may be categorical, and a categorical encoder spends the
    # trailing axis on categories instead of time. Both are therefore laid back
    # onto a channel axis before they can be concatenated, and whichever ends up
    # collapsed is broadcast to the wider one. Handling only the second stream
    # would leave a categorical X against a continuous W failing the same way.
    def _categorical_spec(raw, proc_type, proc_params):
        if proc_type != 'categorical':
            return None
        from neural_mi.data.temporal import relabel_categorical_data
        relabeled = relabel_categorical_data(raw)
        n_categories = int(relabeled.max()) + 1 if relabeled.size else 1
        return n_categories, proc_params.get('encoding', 'majority_vote')

    _spec_x = _categorical_spec(x_raw, _proc_x, _wp_x)
    _spec_second = _categorical_spec(second_raw, _proc_second, _wp_second)

    def _lay_out(windows, spec):
        return windows if spec is None else categorical_to_channel_layout(
            windows, spec[0], spec[1])

    def _concat_x_and_second(w):
        left = _lay_out(w['x'], _spec_x)
        right = _lay_out(w['second'], _spec_second)
        width = max(left.shape[-1], right.shape[-1])
        return torch.cat([align_trailing_axis(left, width),
                          align_trailing_axis(right, width)], dim=1)

    if pack_kind == SHIFT_PACK_DUAL_BRANCH:
        pack = lambda w: ((w['x'], w['second']), w['y'])
    elif pack_kind == SHIFT_PACK_CONCAT:
        pack = lambda w: (_concat_x_and_second(w), w['y'])
    elif pack_kind == SHIFT_PACK_SECOND:
        pack = lambda w: (w['second'], w['y'])
    elif pack_kind == SHIFT_PACK_FIRST:
        pack = lambda w: (w['x'], w['y'])
    else:
        raise ValueError(f"Unknown _shift_pack {pack_kind!r}.")

    return _build_shifted_dataset(
        OrderedDict((
            ('x', (x_raw, _proc_x, _wp_x, _window_size, _step_size)),
            ('second', (second_raw, _proc_second, _wp_second, _window_second, _step_second)),
            ('y', (y_data, _proc_y, _wp_y, _window_size, _step_size)),
        )),
        params, data_device,
        pack_roles=lambda sh: sh.windows_at(0),
        shifter_cls_args=lambda g: BundleShifter(
            OrderedDict((
                ('x', g['x']), ('second', g['second']), ('y', g['y']),
            )),
            pack=pack,
        ),
    )


def n_windows_if_deferred(x_data, y_data, params: dict) -> int:
    """Window count for an (x_data, y_data) pair once windowing is deferred
    (raw, 2-D x_data + ``shift_windows`` requested for a regular-grid pair),
    or ``x_data.shape[0]`` unchanged otherwise (already windowed, or shift
    not requested/reachable for this pair).

    Used wherever an analytical, shift-invariant "how many windows will
    this pair actually produce" count is needed before any orchestration-
    specific windowing (a shared train/test split, a bias-correction
    chunk boundary, ...) has happened yet, reusing the exact same
    construction :func:`try_build_shift_windows_dataset` would build, so
    this can't drift from what actually gets built later.
    """
    if not (getattr(x_data, 'ndim', None) == 2 and params.get('shift_windows')):
        return x_data.shape[0]
    if y_data is None:
        _wp_x = params.get('processor_params_x') or {}
        _window_size = _wp_x.get('window_size')
        _step_size = resolve_step_size(_window_size, _wp_x.get('step_size'))
        _period_x = 1.0 / _wp_x['sample_rate'] if _wp_x.get('sample_rate') else 1.0
        return safe_n_windows(x_data.shape[0], seconds_to_samples(_window_size, _period_x),
                              seconds_to_samples(_step_size, _period_x))
    _throwaway = try_build_shift_windows_dataset(x_data, y_data, dict(params), data_device='cpu')
    return len(_throwaway) if _throwaway is not None else x_data.shape[0]


def chunk_window_range_to_raw(lo: int, hi: int, window_size: int, step_size: int) -> Tuple[int, int]:
    """Raw sample range ``[start, end)`` covering exactly ``hi - lo``
    windows' worth of content for a contiguous window-index chunk
    ``[lo, hi)``, plus the same ``window_size - 1`` margin
    :func:`safe_n_windows` reserves globally, so
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
    ``PairedTemporalDataset._reserve_shift_margin`` reserves for the fixed-grid
    time-shift design, deliberately not
    :func:`chunk_window_range_to_raw`'s ``window_size - 1`` margin
    (``safe_n_windows``/``WindowShifter``'s, for the regular-grid
    raw-array-slicing case), off by one time unit against this
    margin convention. Pair with an explicit ``t_start=0.0,
    t_end=(end - start)`` on the chunk's own ``PairedTemporalDataset`` (via
    ``processor_params_x``) so its base span matches this function's
    assumption exactly, instead of a shorter, data-dependent span derived
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
    through ``_reserve_shift_margin`` and ``time_shift``, the pair that makes
    ``shift_time`` re-tile instead of canceling the offset out. Priming reads
    off the exact margin-reserved window count and the grid's shared base
    start time, so the ``2*window_size`` margin arithmetic is not re-derived
    here a second time.
    """
    from neural_mi.data.temporal import SpikeWindowDataset
    from neural_mi.data.handler import PairedTemporalDataset

    _wp_x = params.get('processor_params_x') or {}
    window_size = _wp_x.get('window_size')
    # Left as given: the WindowManager applies the step convention itself,
    # and a step filled in here with window_size would be read as a fraction
    # whenever the window is below 1.
    step_size = _wp_x.get('step_size')
    if window_size is None:
        raise ValueError(
            "shift_time=True with mode='rigorous' for a spike+spike pair "
            "requires Processing(x_params={'window_size': ...}). 'step_size' is "
            "optional and defaults to window_size."
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
    ``t_start``, the spike-data analogue of the raw-sample slice
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
