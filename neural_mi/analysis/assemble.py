# neural_mi/analysis/assemble.py
"""Turn per-repeat records into the one result shape every mode returns.

Every mode runs the same pipeline: expand the configurations, axis values and
repeats; train each network; combine the networks of a repeat into one value;
aggregate the repeats. This module holds the two ends of that pipeline that do
not depend on the mode: enumerating configurations from a ``sweep_grid``, and
assembling repeat rows into a :class:`~neural_mi.results.Results`.

A repeat row is a plain dict carrying ``config_id``, the grid keys of its
configuration, the mode's axis keys (``lag``, ``tau``, ``ch_x``/``ch_y``), a
repeat index (``run_id``), the repeat's value of the quantity
under ``mi``, and whatever per-repeat diagnostics the mode reports.
"""
import itertools
import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neural_mi.logger import user_stacklevel

# Values one trained network reports about itself. A single-network repeat
# carries these as columns of ``runs``; a multi-network repeat keeps them per
# network under ``details[config_id]['trainings']``.
NETWORK_KEYS: Tuple[str, ...] = (
    'train_mi', 'test_mi', 'raw_train_mi', 'best_epoch',
    'conservative_epoch', 'train_mi_at_peak', 'all_mi_negative',
    'eval_size', 'train_eval_size', 'test_ceiling_mi', 'train_ceiling_mi',
    'test_saturation', 'train_saturation', 'test_trace_saturated_fraction',
    'pr_eig', 'pr_singular', 'spectrum', 'decoder_recon_loss',
    'window_retention', 'n_windows_built', 'n_windows_retained',
    'leak_check_window_size', 'leak_check_step', 'model_path',
    'test_mi_history', 'train_mi_history', 'spectral_metrics_history',
)

# Per-network outputs too large or too structured for a table column: every
# key starting with one of these prefixes. They are kept per repeat under
# ``details[config_id]['embeddings'][run_id]``. The prefixes are specific enough
# to leave out model settings such as ``embedding_dim``.
EMBEDDING_PREFIXES: Tuple[str, ...] = ('embeddings_', 'embedding_history', 'embedding_rotation',
                                       'embedding_track')

REPEAT_KEYS: Tuple[str, ...] = ('run_id', 'noise_sample')

# Values a network or a combine step reports in nats, converted once when the
# caller asked for bits. Ceilings arrive already in the requested units, and
# saturations and participation ratios have none.
MI_KEYS: Tuple[str, ...] = (
    'mi', 'mi_raw', 'train_mi', 'test_mi', 'raw_train_mi', 'train_mi_at_peak',
    'mi_error', 'mi_error_pred', 'slope',
    'mi_xw_y', 'mi_w_y', 'mi_x_y', 'te_yx', 'te_yx_raw',
    'i_xypast_yfuture', 'i_ypast_yfuture', 'i_yxpast_xfuture', 'i_xpast_xfuture',
)
MI_HISTORY_KEYS: Tuple[str, ...] = ('test_mi_history', 'train_mi_history')
NATS_TO_BITS = 1.0 / math.log(2)


def convert_record(record: Dict[str, Any], to_bits: bool) -> Dict[str, Any]:
    """A copy of `record` with its MI-valued entries in the requested units."""
    if not to_bits:
        return dict(record)
    out = dict(record)
    for key in MI_KEYS:
        value = out.get(key)
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            out[key] = float(value) * NATS_TO_BITS
    for key in MI_HISTORY_KEYS:
        seq = out.get(key)
        if isinstance(seq, (list, tuple, np.ndarray)):
            out[key] = [v if v is None or (isinstance(v, float) and math.isnan(v)) else v * NATS_TO_BITS
                        for v in seq]
    return out


def config_lookup(sweep_grid: Optional[Dict[str, Sequence]]):
    """Map a task's grid values back to its ``config_id``.

    Returns ``(config_keys, configs, find)``, where ``find(task_params)`` reads
    the configuration keys out of a task's parameter dict and returns the
    ``config_id`` they belong to.
    """
    configs, _ = split_grid(sweep_grid)
    config_keys = [k for k in (sweep_grid or {}) if k != 'run_id']
    index = {tuple(_hashable(c[k]) for k in config_keys): i for i, c in enumerate(configs)}

    def find(task_params: Dict[str, Any]) -> int:
        return index[tuple(_hashable(task_params.get(k)) for k in config_keys)]

    return config_keys, configs, find


def single_network_rows(task_results: List[Dict[str, Any]], sweep_grid: Optional[Dict[str, Sequence]],
                        to_bits: bool, axis: Sequence[str] = ()):
    """Repeat rows for modes whose repeat is one trained network.

    Returns ``(rows, details, config_keys)``. Each row carries the network's
    measurements, with ``mi`` its training-side estimate (0 when its test MI
    never rose above zero). Embeddings, when requested, go to
    ``details[config_id]['embeddings'][run_id]``, or under
    ``(*axis values, run_id)`` for a mode with an axis, so that every lag keeps
    its own.
    """
    config_keys, configs, find = config_lookup(sweep_grid)
    has_run_id = 'run_id' in (sweep_grid or {})
    rows, details = [], {i: {} for i in range(len(configs))}
    for result in task_results:
        cid = find(result)
        rid = result.get('run_id', 0) if has_run_id else 0
        values = convert_record(network_values(result), to_bits)
        mi = 0.0 if result.get('all_mi_negative') else values.get('train_mi')
        row = {'config_id': cid, **{k: result.get(k) for k in config_keys},
               **{k: result.get(k) for k in axis}, 'run_id': rid, 'mi': mi, **values}
        rows.append(row)
        embeddings = embedding_values(result)
        if embeddings:
            key = (*(result.get(k) for k in axis), rid) if axis else rid
            details[cid].setdefault('embeddings', {})[key] = embeddings
    return rows, details, config_keys


def _product(grid: Dict[str, Sequence]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def split_grid(sweep_grid: Optional[Dict[str, Sequence]]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Split a ``sweep_grid`` into its configurations and its repeat indices.

    Configurations are every combination of the grid's keys other than
    ``run_id``, in the order :func:`itertools.product` produces them; their
    position is their ``config_id``. Repeats are the ``run_id`` values, or
    ``[0]`` when the grid has none.
    """
    grid = dict(sweep_grid or {})
    run_ids = list(grid.pop('run_id', [0]))
    configs = _product(grid) if grid else [{}]
    return configs, run_ids


def expand(sweep_grid: Optional[Dict[str, Sequence]]) -> List[Tuple[int, Dict[str, Any], Any]]:
    """Every ``(config_id, config, run_id)`` a grid asks for, configuration-major."""
    configs, run_ids = split_grid(sweep_grid)
    return [(cid, cfg, rid) for cid, cfg in enumerate(configs) for rid in run_ids]


def network_values(task_result: Dict[str, Any]) -> Dict[str, Any]:
    """The per-network measurements out of one training task's result dict."""
    return {k: task_result[k] for k in NETWORK_KEYS if k in task_result}


def embedding_values(task_result: Dict[str, Any]) -> Dict[str, Any]:
    """The embedding arrays out of one training task's result dict, if any."""
    return {k: v for k, v in task_result.items() if k.startswith(EMBEDDING_PREFIXES)}


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_hashable(v) for v in value)
    return value


def aggregate(runs: pd.DataFrame, group_cols: Sequence[str], value_col: str = 'mi',
              mean_cols: Iterable[str] = (), first_cols: Iterable[str] = (),
              std_cols: Iterable[str] = ()) -> pd.DataFrame:
    """One row per group: ``mi_mean``, ``mi_std``, ``n_runs``, ``n_zero`` and column means.

    A repeat reported as 0 produced nothing: its network learned nothing that
    generalises, or its value came out negative. The means and spreads use the
    repeats that produced a value, and ``n_zero`` counts the others. A group
    whose repeats all produced nothing reports 0. ``mi_std`` is the sample
    standard deviation across the repeats used and is NaN below two, never 0. ``mean_cols`` are averaged into
    ``<col>_mean`` and ``std_cols`` spread into ``<col>_std`` the same way;
    ``first_cols`` are constant within a group and copied. ``n_runs`` counts
    every repeat, including one that could not produce a number (NaN).
    Grid values that pandas cannot group on (dicts) are grouped by a hashable
    stand-in and restored afterwards.
    """
    mean_cols = [c for c in mean_cols if c in runs.columns]
    std_cols = [c for c in std_cols if c in runs.columns]
    first_cols = [c for c in first_cols if c in runs.columns and c not in group_cols]
    if runs.empty:
        cols = (list(group_cols) + ['mi_mean', 'mi_std', 'n_runs', 'n_zero']
                + [f'{c}_mean' for c in mean_cols] + first_cols)
        return pd.DataFrame(columns=cols)

    work = runs.copy()
    keyed = {}
    for col in group_cols:
        key = f'__key_{col}'
        work[key] = work[col].map(_hashable)
        keyed[col] = key
    key_cols = [keyed[c] for c in group_cols]

    rows, zero_groups = [], []
    grouper = key_cols[0] if len(key_cols) == 1 else key_cols
    for _, group in work.groupby(grouper, sort=False, dropna=False):
        values = pd.to_numeric(group[value_col], errors='coerce')
        n_rows = len(group)
        n = int(values.notna().sum())
        n_zero = int((values == 0).sum())
        if n_zero < n:
            used = values.notna() & (values != 0)
            group, values = group[used.values], values[used.values]
        row = {col: group[col].iloc[0] for col in group_cols}
        n_used = int(values.notna().sum())
        row['mi_mean'] = float(values.mean()) if n_used else math.nan
        row['mi_std'] = float(values.std(ddof=1)) if n_used >= 2 else math.nan
        row['n_runs'] = n_rows
        row['n_zero'] = n_zero
        if n >= 2 and n_zero:
            zero_groups.append((n_zero, n))
        for col in mean_cols:
            col_values = pd.to_numeric(group[col], errors='coerce')
            row[f'{col}_mean'] = float(col_values.mean()) if col_values.notna().any() else math.nan
        for col in std_cols:
            col_values = pd.to_numeric(group[col], errors='coerce')
            row[f'{col}_std'] = float(col_values.std(ddof=1)) if col_values.notna().sum() >= 2 else math.nan
        for col in first_cols:
            row[col] = group[col].iloc[0]
        rows.append(row)
    _warn_zero_repeats(zero_groups)
    return pd.DataFrame(rows)


def _warn_zero_repeats(zero_groups: List[Tuple[int, int]]) -> None:
    """Say once per result which repeats produced nothing and were left out."""
    if not zero_groups:
        return
    n_zero = sum(z for z, _ in zero_groups)
    n_all = sum(n for _, n in zero_groups)
    all_zero = sum(1 for z, n in zero_groups if z == n)
    reading = (f"{all_zero} row(s) had no repeat that produced a value and report 0. "
               if all_zero else "")
    if len(zero_groups) == 1:
        warnings.warn(
            f"{n_zero} of {n_all} repeats produced nothing (reported as 0) and are left out of "
            f"mi_mean and mi_std. {reading}A repeat that produced nothing among clearly positive "
            f"ones is most likely a failed run. When every repeat is near zero, the quantity "
            f"itself may be near zero. result.runs keeps every repeat.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    else:
        warnings.warn(
            f"{len(zero_groups)} rows of dataframe hold {n_zero} of their {n_all} repeats that "
            f"produced nothing (reported as 0). Those repeats are left out of mi_mean and "
            f"mi_std. n_zero counts them in each row. {reading}A repeat that produced "
            f"nothing among clearly positive ones is most likely a failed run. When every "
            f"repeat is near zero, the quantity itself may be near zero. result.runs keeps "
            f"every repeat.",
            UserWarning, stacklevel=user_stacklevel(),
        )


def build_results(mode: str, params: Dict[str, Any], runs_rows: List[Dict[str, Any]], *,
                  config_keys: Sequence[str] = (), axis_keys: Sequence[str] = (),
                  mean_cols: Iterable[str] = (), first_cols: Iterable[str] = (),
                  std_cols: Iterable[str] = (),
                  details: Optional[Dict[int, Dict[str, Any]]] = None,
                  row_scalars: Optional[Dict[int, Dict[str, Any]]] = None):
    """Assemble repeat rows into a :class:`Results`.

    Parameters
    ----------
    mode : str
        The analysis mode.
    params : dict
        The full resolved configuration of the call.
    runs_rows : list of dict
        One dict per repeat, each with ``config_id``, the configuration's grid
        keys, the axis keys, a repeat index and ``mi``.
    config_keys, axis_keys : sequence of str
        The grid keys that vary and the mode's own axis keys. Together with
        ``config_id`` they define one ``dataframe`` row.
    mean_cols : iterable of str
        Per-repeat columns averaged into ``<col>_mean`` (component values,
        ``test_mi``, participation ratios).
    first_cols : iterable of str
        Per-configuration columns that are constant within a row.
    details : dict, optional
        ``{config_id: {...}}`` structured diagnostics per configuration.
    row_scalars : dict, optional
        ``{config_id: {column: value}}`` scalars that belong on every
        ``dataframe`` row of a configuration (``amplification_factor``,
        ``mi_error``). They are computed by the mode,
        not averaged here, because they are functions of the aggregate.
    """
    from neural_mi.results import Results

    runs = pd.DataFrame(runs_rows)
    # A grid value given as a list (a per-layer hidden_dim) is recorded as a
    # tuple, so the column can be grouped and filtered on.
    for col in config_keys:
        if col in runs.columns:
            runs[col] = runs[col].map(lambda v: tuple(v) if isinstance(v, list) else v)
    ordered = ['config_id', *config_keys, *axis_keys]
    repeat = [k for k in REPEAT_KEYS if k in runs.columns]
    lead = [c for c in ordered + repeat + ['mi'] if c in runs.columns]
    if not runs.empty:
        runs = runs[lead + [c for c in runs.columns if c not in lead]]
        runs = runs.sort_values(['config_id'], kind='stable').reset_index(drop=True)

    params = dict(params)
    params['config_keys'] = list(config_keys)
    params['axis_keys'] = list(axis_keys)

    group_cols = [c for c in ordered if c in runs.columns]
    dataframe = aggregate(runs, group_cols, mean_cols=mean_cols, first_cols=first_cols,
                          std_cols=std_cols)
    for cid, scalars in (row_scalars or {}).items():
        mask = dataframe['config_id'] == cid
        for col, value in scalars.items():
            if col not in dataframe.columns:
                dataframe[col] = math.nan if not isinstance(value, (bool, np.bool_)) else None
                dataframe[col] = dataframe[col].astype(object)
            dataframe.loc[mask, col] = value
    for col in dataframe.columns:
        if dataframe[col].dtype == object:
            try:
                dataframe[col] = pd.to_numeric(dataframe[col])
            except (TypeError, ValueError):
                pass

    mi_estimate = float(dataframe['mi_mean'].iloc[0]) if len(dataframe) == 1 else None
    if mi_estimate is not None and math.isnan(mi_estimate):
        mi_estimate = None
    return Results(mode=mode, params=params, mi_estimate=mi_estimate,
                   dataframe=dataframe, runs=runs.reset_index(drop=True),
                   details={int(k): v for k, v in (details or {}).items()})


def merge_results(parts, sweep_grid: Dict[str, Sequence], params: Dict[str, Any],
                  mode: Optional[str] = None):
    """Combine calls that each ran part of one ``sweep_grid`` into one Results.

    Used when a grid varies a processor parameter: the data are prepared once
    per processor setting, and each preparation runs the rest of the grid.

    Parameters
    ----------
    parts : list of (dict, Results)
        The grid values a call held fixed, and that call's result. Each call ran
        the grid without those keys.
    sweep_grid : dict
        The full grid the caller passed.
    params : dict
        The call's parameters, recorded on the merged result.
    mode : str, optional
        The merged result's mode. Defaults to the first part's.
    """
    from neural_mi.results import Results

    config_keys, _, find = config_lookup(sweep_grid)
    runs_parts, frames, details = [], [], {}
    axis_keys: List[str] = []
    for fixed, part in parts:
        inner_grid = {k: v for k, v in (sweep_grid or {}).items() if k not in fixed}
        inner_configs, _ = split_grid(inner_grid)
        remap = {cid: find({**fixed, **cfg}) for cid, cfg in enumerate(inner_configs)}
        axis_keys = list(part.params.get('axis_keys') or [])
        for table, out in ((part.runs, runs_parts), (part.dataframe, frames)):
            table = table.copy()
            for key, value in fixed.items():
                table[key] = [value] * len(table)
            table['config_id'] = table['config_id'].map(remap)
            out.append(table)
        for cid, entry in part.details.items():
            details[remap[int(cid)]] = entry

    lead = ['config_id', *config_keys, *axis_keys]

    def _ordered(frames_list):
        frame = pd.concat(frames_list, ignore_index=True) if frames_list else pd.DataFrame()
        if frame.empty:
            return frame
        cols = [c for c in lead if c in frame.columns]
        frame = frame[cols + [c for c in frame.columns if c not in cols]]
        return frame.sort_values('config_id', kind='stable').reset_index(drop=True)

    runs = _ordered(runs_parts)
    dataframe = _ordered(frames)
    merged_params = dict(params)
    merged_params['sweep_grid'] = sweep_grid
    merged_params['config_keys'] = list(config_keys)
    merged_params['axis_keys'] = axis_keys
    mi_estimate = None
    if len(dataframe) == 1 and not math.isnan(float(dataframe['mi_mean'].iloc[0])):
        mi_estimate = float(dataframe['mi_mean'].iloc[0])
    mode = mode or (parts[0][1].mode if parts else params.get('mode'))
    return Results(mode=mode, params=merged_params, mi_estimate=mi_estimate,
                   dataframe=dataframe, runs=runs, details=details)
