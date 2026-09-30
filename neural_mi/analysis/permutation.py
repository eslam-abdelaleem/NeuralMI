# neural_mi/analysis/permutation.py
"""Permutation nulls: rerun a mode's producer with X moved in time.

A trial shifts X, the source, once and runs the producer the observed call
ran. Y, and W where there is one, stay as they are, so every relation that
does not involve X (Y's own history in transfer entropy, the Y-W link in the
conditional quantities) survives into the null, and only X's alignment with
the rest is broken. The trial's repeats are averaged into one value per
``dataframe`` row, so the observed value and every null value are built the
same way, and each row gets its own null and its own ``p_value``.

Two shuffles are available through ``permutation_shuffle``:

``'circular'`` (default)
    X is shifted by one random offset along its time axis, wrapping at the
    end. Every window, sample or spike keeps its neighbours, so X's own
    structure is preserved. Offsets within 10% of the recording's length of
    zero (in either direction) are excluded, since a near-zero shift would
    leave the dependence in place.
``'block'``
    X is cut into contiguous blocks one window long and the blocks are
    reordered. Structure within a window is kept; structure across windows
    is not.
"""
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from neural_mi.analysis.assemble import _hashable
from neural_mi.logger import CapturedTask, logger, released, user_stacklevel, worker_init_args

# Offsets closer than this fraction of the recording to zero, in either
# direction, are excluded from a circular shift.
MIN_SHIFT_FRACTION = 0.1


# ---------------------------------------------------------------------------
# Shuffles
# ---------------------------------------------------------------------------

def _is_spike_list(data: Any) -> bool:
    return isinstance(data, list) and bool(data) and not np.isscalar(data[0])


def _spike_population_extent(spikes: list, base_params: dict, side: str = 'x') -> Tuple[float, float]:
    """``(t_start, t_end)`` of a raw spike-time population.

    Follows ``SpikeDataset.get_temporal_extent()``: ``t_start`` is the earliest
    spike across neurons, and ``t_end`` is ``processor_params_<side>['n_seconds']``
    when the caller set it (a recording can extend past its last spike),
    otherwise the latest spike across neurons.
    """
    valid = [np.asarray(st) for st in spikes if len(st) > 0]
    t_start = min((st[0] for st in valid), default=0.0)
    n_seconds = (base_params.get(f'processor_params_{side}') or {}).get('n_seconds')
    if n_seconds is not None:
        t_end = float(n_seconds)
    else:
        t_end = max((st[-1] for st in valid), default=0.0)
    return float(t_start), float(t_end)


def _circular_offset(length: float, integer: bool) -> float:
    """A random offset in ``[m, length - m]``, with ``m`` a tenth of `length`."""
    lo, hi = MIN_SHIFT_FRACTION * length, (1.0 - MIN_SHIFT_FRACTION) * length
    if integer:
        lo, hi = max(1, int(math.ceil(lo))), min(int(length) - 1, int(math.floor(hi)))
        return int(np.random.randint(lo, hi + 1)) if hi >= lo else 0
    return float(np.random.uniform(lo, hi))


def _circular_shift_spike_population(spikes: list, t_start: float, t_end: float) -> list:
    """Shift every neuron's spike train by one shared random offset, wrapping
    at the recording boundary.

    One shared offset keeps the population's own cross-neuron structure, such
    as synchrony, and breaks only its alignment with the other streams.
    """
    duration = t_end - t_start
    if duration <= 0:
        return [np.asarray(st).copy() for st in spikes]
    delta = _circular_offset(duration, integer=False)
    shifted = []
    for st in spikes:
        st = np.asarray(st, dtype=float)
        if st.size == 0:
            shifted.append(st.copy())
            continue
        new_st = t_start + np.mod((st - t_start) + delta, duration)
        shifted.append(np.sort(new_st))
    return shifted


def _block_shuffle_spike_population(spikes: list, t_start: float, t_end: float,
                                    block_size: float) -> list:
    """Cut the recording into contiguous blocks of `block_size` and reorder them.

    Every spike keeps its position within its block. The blocks are
    reassembled into a spike list of the same duration, so the result goes
    through the same raw pipeline as the unshuffled data.
    """
    duration = t_end - t_start
    if duration <= 0 or block_size <= 0:
        return [np.asarray(st).copy() for st in spikes]
    n_blocks = max(1, int(round(duration / block_size)))
    edges = np.linspace(t_start, t_end, n_blocks + 1)
    order = np.random.permutation(n_blocks)
    shifted = []
    for st in spikes:
        st = np.asarray(st, dtype=float)
        pieces = []
        cursor = t_start
        for b in order:
            lo, hi = edges[b], edges[b + 1]
            # Half-open blocks, except the recording's last one, which is
            # closed so that a spike exactly at t_end is kept.
            in_block = (st >= lo) & (st < hi if b < n_blocks - 1 else st <= hi)
            pieces.append(st[in_block] - lo + cursor)
            cursor += (hi - lo)
        shifted.append(np.sort(np.concatenate(pieces)) if pieces else np.array([]))
    return shifted


def _window_rows(x: Any, base_params: Dict[str, Any], ctx: Dict[str, Any]) -> int:
    """How many rows of X make one window, the block length of a block shuffle.

    Windowed data has one window per row. A raw series that is windowed later
    has ``window_size`` samples per window, converted with the stream's
    sample rate or clock. Transfer entropy's raw series use the history length.
    """
    if getattr(x, 'ndim', 2) != 2:
        return 1
    params = base_params.get('processor_params_x') or {}
    window = params.get('window_size')
    if window is None:
        return int(ctx.get('history_window') or 1)
    if params.get('sample_rate'):
        period = 1.0 / float(params['sample_rate'])
    elif base_params.get('x_time') is not None and len(base_params['x_time']) > 1:
        period = float(np.median(np.diff(np.asarray(base_params['x_time'], dtype=float))))
    else:
        period = 1.0
    return max(1, int(round(float(window) / period)))


def _roll_rows(x: Any, offset: int) -> Any:
    if torch.is_tensor(x):
        return torch.roll(x, shifts=offset, dims=0)
    return np.roll(np.asarray(x), offset, axis=0)


def _reorder_rows(x: Any, order: np.ndarray) -> Any:
    if torch.is_tensor(x):
        return x[torch.as_tensor(order)]
    return np.asarray(x)[order]


def shift_x(x_data: Any, base_params: Dict[str, Any], permutation_shuffle: str,
            ctx: Optional[Dict[str, Any]] = None) -> Any:
    """One shifted copy of X, drawn from numpy's global generator.

    See the module docstring for the two shuffles. A spike population is moved
    in time; any other X is moved along its first (time or sample) axis.
    """
    ctx = ctx or {}
    if _is_spike_list(x_data):
        t_start, t_end = _spike_population_extent(x_data, base_params, 'x')
        if permutation_shuffle == 'block':
            block = ((base_params.get('processor_params_x') or {}).get('window_size')
                     or (t_end - t_start) / 10.0)
            return _block_shuffle_spike_population(x_data, t_start, t_end, block)
        return _circular_shift_spike_population(x_data, t_start, t_end)
    n = x_data.shape[0] if hasattr(x_data, 'shape') else len(x_data)
    if permutation_shuffle == 'block':
        rows = _window_rows(x_data, base_params, ctx)
        starts = np.arange(0, n, rows)
        order = np.concatenate([np.arange(s, min(s + rows, n))
                                for s in starts[np.random.permutation(len(starts))]])
        return _reorder_rows(x_data, order)
    return _roll_rows(x_data, _circular_offset(n, integer=True))


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def row_values(mode: str, produced: Dict[str, Any]) -> Dict[tuple, Tuple[float, float]]:
    """A producer's repeats averaged into one ``(mi, mi_raw)`` per ``dataframe`` row.

    ``mi`` follows the reported convention, as ``aggregate`` forms ``mi_mean``:
    repeats reported as 0 produced nothing and are left out, and a row with no
    other repeat is 0. ``mi_raw`` averages every repeat's measured value.
    """
    from neural_mi.analysis.modes import _DIFFERENCES

    keys = ['config_id', *produced['axis_keys']]
    raw_by_repeat: Dict[tuple, float] = {}
    if mode in _DIFFERENCES:
        signs = {col: sign for col, _, sign in _DIFFERENCES[mode]}
        for cid, entry in produced['details'].items():
            trainings = entry.get('trainings')
            if trainings is None or len(trainings) == 0 or 'component' not in trainings:
                continue
            for record in trainings.to_dict(orient='records'):
                sign = signs.get(record['component'])
                if sign is None:
                    continue
                key = (cid, record.get('run_id'))
                raw_by_repeat[key] = raw_by_repeat.get(key, 0.0) + sign * record.get('raw_train_mi', math.nan)

    grouped: Dict[tuple, List[Tuple[float, float]]] = {}
    for row in produced['rows']:
        key = tuple(_hashable(row.get(k)) for k in keys)
        if mode in _DIFFERENCES:
            raw = raw_by_repeat.get((row['config_id'], row.get('run_id')), math.nan)
        else:
            raw = row.get('raw_train_mi', row.get('mi'))
        grouped.setdefault(key, []).append((row.get('mi'), raw))

    out = {}
    for key, pairs in grouped.items():
        mi = np.array([math.nan if v is None else v for v, _ in pairs], dtype=float)
        raw = np.array([math.nan if v is None else v for _, v in pairs], dtype=float)
        produced = mi[np.isfinite(mi) & (mi != 0)]
        if produced.size:
            mi_value = float(produced.mean())
        else:
            mi_value = 0.0 if np.isfinite(mi).any() else math.nan
        out[key] = (mi_value, float(np.nanmean(raw)) if np.isfinite(raw).any() else math.nan)
    return out


def _trial(args) -> Optional[Dict[tuple, Tuple[float, float]]]:
    """One permutation trial. Module-level so it can run in a worker process."""
    from neural_mi.analysis.modes import produce
    mode, x, y, w, base_params, grid, ctx, to_bits, seed, permutation_shuffle = args
    np.random.seed(seed)
    x_null = shift_x(x, base_params, permutation_shuffle, ctx)
    try:
        # A trial's networks train on data whose dependence was destroyed on
        # purpose, so the warnings they raise (an estimate near or below zero,
        # a network that learned nothing) describe the null working as
        # intended. The observed call has already raised its own.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            produced = produce(mode, x_null, y, w, base_params, grid, ctx=ctx, to_bits=to_bits,
                               n_workers=1)
    except Exception as exc:
        logger.warning(f"Permutation trial failed: {exc}")
        return None
    return row_values(mode, produced)


# ---------------------------------------------------------------------------
# The null of every row
# ---------------------------------------------------------------------------

def permutation_nulls(mode: str, x, y, w, base_params: Dict[str, Any], grid, *, ctx: Dict[str, Any],
                      to_bits: bool, n_permutations: int, n_workers: int = 1,
                      permutation_shuffle: str = 'circular') -> List[Optional[Dict[tuple, Tuple[float, float]]]]:
    """Run `n_permutations` trials, in parallel across `n_workers`.

    Returns one entry per trial: its ``{row key: (mi, mi_raw)}``, or ``None``
    when the trial failed.
    """
    rng = np.random.default_rng(base_params.get('random_seed'))
    seeds = [int(s) for s in rng.integers(0, 2 ** 31, size=n_permutations)]
    # A null trial's networks are never saved: they would overwrite the
    # observed networks' files with networks trained on moved data.
    trial_params = {**base_params, 'show_progress': False, 'save_best_model_path': None}
    trial_params.pop('_model_labels', None)
    args = [(mode, x, y, w, trial_params, grid, ctx, to_bits, seed, permutation_shuffle)
            for seed in seeds]
    show_progress = base_params.get('show_progress', True)
    logger.info(f"Permutation test: {n_permutations} trials for mode='{mode}' across "
                f"{n_workers} worker(s).")
    if n_workers > 1 and n_permutations > 1:
        _log_init, _log_args = worker_init_args()
        with mp.get_context('spawn').Pool(processes=min(n_workers, n_permutations),
                                          initializer=_log_init, initargs=_log_args) as pool:
            trials = list(tqdm(released(pool.imap(CapturedTask(_trial), args)), total=n_permutations,
                               desc="Permutation test", leave=False, disable=not show_progress))
    else:
        trials = [_trial(a) for a in tqdm(args, desc="Permutation test", leave=False,
                                          disable=not show_progress)]
    if trials and all(t is None for t in trials):
        warnings.warn(
            f"All {n_permutations} permutation trials for mode='{mode}' failed and left no "
            f"null distribution. The log lists each failure under 'Permutation trial failed'.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    return trials


def attach_nulls(result, trials, axis_keys) -> None:
    """Store each row's null in ``details`` and add a ``p_value`` column.

    A configuration without an axis gets a list of null values under
    ``details[config_id]['null_distribution']`` (and the unclipped values under
    ``'null_distribution_raw'``). With an axis (``lag``, or ``ch_x``/``ch_y``)
    each of those is a dict keyed by the axis value. ``p_value`` is
    ``(1 + #{null >= observed}) / (1 + n)`` over the trials that succeeded.
    """
    frame = result.dataframe
    p_values = []
    for _, row in frame.iterrows():
        cid = int(row['config_id'])
        axis = tuple(_hashable(row[k]) for k in axis_keys)
        key = (cid, *axis)
        null = [t[key][0] for t in trials if t is not None and key in t]
        null_raw = [t[key][1] for t in trials if t is not None and key in t]
        entry = result.details.setdefault(cid, {})
        if axis_keys:
            label = axis[0] if len(axis) == 1 else axis
            entry.setdefault('null_distribution', {})[label] = null
            entry.setdefault('null_distribution_raw', {})[label] = null_raw
        else:
            entry['null_distribution'] = null
            entry['null_distribution_raw'] = null_raw
        valid = [v for v in null if not math.isnan(v)]
        observed = row['mi_mean']
        if valid and observed is not None and not math.isnan(observed):
            p_values.append((1 + sum(v >= observed for v in valid)) / (1 + len(valid)))
        else:
            p_values.append(math.nan)
    frame['p_value'] = p_values
