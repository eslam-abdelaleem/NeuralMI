# neural_mi/analysis/modes.py
"""One producer per mode: prepared data in, repeat rows out.

A producer runs a mode's networks for every configuration and repeat of a
``sweep_grid`` and combines each repeat into one row (see
:mod:`neural_mi.analysis.assemble`). The observed call and every permutation
trial go through the same producer, so a null distribution is built from
exactly the rows the observed value was built from.

Producers are module-level functions so a permutation trial can be sent to a
worker process.
"""
import hashlib
import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from neural_mi.analysis.assemble import (
    config_lookup, convert_record, embedding_values, network_values, single_network_rows,
    split_grid, NATS_TO_BITS, NETWORK_KEYS,
)
from neural_mi.analysis.sweep import ParameterSweep, amplification_factor, merge_grid_values
from neural_mi.logger import user_stacklevel

# Repeat-level fit values a rigorous extrapolation reports.
_FIT_KEYS = (
    'mi_error', 'mi_error_pred', 'slope', 'is_reliable', 'linear_region_found',
    'enough_gamma_points', 'curvature_coefficient', 'curvature_se', 'curvature_t',
    'curvature_slope', 'fit_quality_warning', 'leverage_warning', 'r_squared',
    'max_abs_residual', 'loo_intercept_shift', 'gammas_used', 'chunking_mode',
    'n_tasks_created', 'saturated_gammas', 'zero_rungs',
)

# The components of each difference quantity: (column, raw-results key, sign).
_DIFFERENCES = {
    'conditional': [('mi_xw_y', 'raw_xw_y', 1), ('mi_w_y', 'raw_w_y', -1)],
    'interaction': [('mi_xw_y', 'raw_xw_y', 1), ('mi_x_y', 'raw_x_y', -1), ('mi_w_y', 'raw_w_y', -1)],
    'transfer': [('i_xypast_yfuture', 'raw_xypast_yfuture', 1),
                 ('i_ypast_yfuture', 'raw_ypast_yfuture', -1)],
}
_TRANSFER_REVERSE = [('i_yxpast_xfuture', 'raw_yxpast_xfuture', 1),
                     ('i_xpast_xfuture', 'raw_xpast_xfuture', -1)]
# Interaction information is signed. The other combined quantities are
# information and cannot be negative.
_SIGNED = ('interaction',)
_QUANTITY_NAMES = {'conditional': 'Conditional MI', 'transfer': 'TE(X→Y)',
                   'interaction': 'Interaction information', 'rigorous': 'MI'}


def _floor_at_zero(row: Dict[str, Any], key: str) -> bool:
    """Report a negative ``row[key]`` as 0 and keep the measured value as
    ``row[key + '_raw']``. Returns whether the value was negative."""
    value = row.get(key)
    negative = value is not None and not (isinstance(value, float) and math.isnan(value)) and value < 0
    row[f'{key}_raw'] = value
    if negative:
        row[key] = 0.0
    return negative


def _produced(rows, details, config_keys, *, axis_keys=(), mean_cols=('test_mi',), std_cols=(),
              first_cols=(), row_scalars=None) -> Dict[str, Any]:
    return dict(rows=rows, details=details, config_keys=list(config_keys), axis_keys=list(axis_keys),
                mean_cols=list(mean_cols), std_cols=list(std_cols), first_cols=list(first_cols),
                row_scalars=row_scalars or {})


def repeat_seed(base_seed: Optional[int], run_id: Any) -> Optional[int]:
    """A seed of its own for one repeat, derived from the call's seed.

    Used where a repeat is a whole procedure run through a function that seeds
    its own networks by position, so two repeats would otherwise draw the same
    initialisations.
    """
    if base_seed is None:
        return None
    return (int(base_seed) + int(hashlib.md5(f"repeat{run_id}".encode()).hexdigest(), 16)) % (2 ** 31)


def _repeat_params(params: Dict[str, Any], run_id: Any, has_run_id: bool) -> Dict[str, Any]:
    out = dict(params)
    if has_run_id:
        out['run_id'] = run_id
        out['random_seed'] = repeat_seed(params.get('random_seed'), run_id)
    return out


# ---------------------------------------------------------------------------
# Modes whose repeat is one network
# ---------------------------------------------------------------------------

def produce_single_network(mode, x, y, base_params, grid, *, to_bits, n_workers=1,
                           is_proc_sweep=None):
    """estimate and sweep: one network per repeat."""
    kwargs = {'n_workers': n_workers}
    if mode == 'sweep':
        results = ParameterSweep(x, y, base_params).run(grid, is_proc_sweep=is_proc_sweep, **kwargs)
    else:
        results = ParameterSweep(x, y, base_params).run(grid, **kwargs)
    rows, details, config_keys = single_network_rows(results, grid, to_bits)
    return _produced(rows, details, config_keys)


def produce_lag(x, y, base_params, grid, *, to_bits, lag_range, n_workers=1, equalize_n=False):
    """mode='lag': one network per lag and repeat, for every configuration."""
    from neural_mi.analysis.lag import run_lag_analysis
    results = run_lag_analysis(x, y, base_params, lag_range=lag_range, sweep_grid=grid,
                               n_workers=n_workers, equalize_n=equalize_n)
    rows, details, config_keys = single_network_rows(results, grid, to_bits, axis=('lag',))
    return _produced(rows, details, config_keys, axis_keys=('lag',), first_cols=('n_windows_built',))


# ---------------------------------------------------------------------------
# Rigorous: one extrapolation per configuration and repeat
# ---------------------------------------------------------------------------

def _fit_row(fit: Dict[str, Any], to_bits: bool, quantity: str = 'rigorous') -> Dict[str, Any]:
    """One fit as a row. An extrapolated value of a quantity that cannot be
    negative is reported as 0 when it comes out below zero, with the
    extrapolated value kept as ``mi_raw``."""
    row = {'mi': fit.get('mi_corrected')}
    row.update({k: fit[k] for k in _FIT_KEYS if k in fit})
    row = convert_record(row, to_bits)
    if quantity in _SIGNED:
        return row
    if _floor_at_zero(row, 'mi'):
        units = 'bits' if to_bits else 'nats'
        error = row.get('mi_error')
        spread = (f" ± {error:.4f}" if isinstance(error, (int, float)) and math.isfinite(error)
                  else "")
        warnings.warn(
            f"The extrapolated {_QUANTITY_NAMES[quantity]} is negative "
            f"({row['mi_raw']:.4f}{spread} {units}). It cannot be negative. It is reported "
            f"as 0 {units} and the extrapolated value is kept as mi_raw in result.runs. A value "
            f"this close to zero is not resolved at this sample size.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    return row


def _ladder_rows(frame: pd.DataFrame, to_bits: bool, extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    keep = [c for c in frame.columns if c in ('gamma', 'chunk', 'run_id', 'component')
            or c in NETWORK_KEYS]
    out = []
    for record in frame[keep].to_dict(orient='records'):
        out.append({**extra, **convert_record(record, to_bits)})
    return out


def produce_rigorous(x, y, base_params, grid, *, to_bits, n_workers=1, rigorous_kwargs=None):
    """mode='rigorous': the gamma ladder and one fit per configuration and repeat."""
    from neural_mi.analysis.rigorous import run_rigorous_analysis
    out = run_rigorous_analysis(x, y, base_params, sweep_grid=grid, n_workers=n_workers,
                                **(rigorous_kwargs or {}))
    config_keys, configs, find = config_lookup(grid)
    has_run_id = 'run_id' in (grid or {})
    raw = out.get('raw_results_df')
    rows, details = [], {i: {} for i in range(len(configs))}
    for fit in out.get('corrected_results', []):
        cid = find(fit)
        rid = fit.get('run_id', 0) if has_run_id else 0
        rows.append({'config_id': cid, **{k: fit.get(k) for k in config_keys}, 'run_id': rid,
                     **_fit_row(fit, to_bits)})
    if raw is not None and len(raw):
        ladders = {i: [] for i in range(len(configs))}
        for record in raw.to_dict(orient='records'):
            cid = find(record)
            rid = record.get('run_id', 0) if has_run_id else 0
            ladder = {k: record.get(k) for k in ('gamma', 'chunk') if k in record}
            ladder.update(convert_record(network_values(record), to_bits))
            ladders[cid].append({'run_id': rid, **ladder})
        for cid, recs in ladders.items():
            details[cid]['trainings'] = pd.DataFrame(recs)
    return _rigorous_produced(rows, details, config_keys)


def _rigorous_produced(rows, details, config_keys):
    """Per-configuration scalars of a rigorous result: the fit's half-width when
    there is one repeat, and how many repeats' fits are reliable."""
    by_config: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_config.setdefault(row['config_id'], []).append(row)
    scalars = {}
    for cid, reps in by_config.items():
        scalars[cid] = {
            'mi_error': reps[0].get('mi_error') if len(reps) == 1 else math.nan,
            'n_reliable': int(sum(bool(r.get('is_reliable')) for r in reps)),
        }
    return _produced(rows, details, config_keys, mean_cols=(), row_scalars=scalars)


# ---------------------------------------------------------------------------
# Difference quantities: conditional, transfer, interaction
# ---------------------------------------------------------------------------

def _component_values(results: List[Dict[str, Any]], to_bits: bool) -> List[float]:
    return [convert_record({'mi': r.get('train_mi', math.nan)}, to_bits)['mi'] for r in results]


def _difference_rows(mode, raw, cid, cfg, config_keys, run_ids, to_bits, bidirectional=False):
    """Pair the components' repeats by position and combine each pair."""
    spec = _DIFFERENCES[mode]
    values = {col: _component_values(raw.get(key) or [], to_bits) for col, key, _ in spec}
    reverse = {}
    if bidirectional:
        reverse = {col: _component_values(raw.get(key) or [], to_bits) for col, key, _ in _TRANSFER_REVERSE}
    rows, trainings = [], []
    for i, rid in enumerate(run_ids):
        comp = {col: (vals[i] if i < len(vals) else math.nan) for col, vals in values.items()}
        mi = sum(sign * comp[col] for col, _, sign in spec)
        row = {'config_id': cid, **{k: cfg.get(k) for k in config_keys}, 'run_id': rid, 'mi': mi, **comp}
        if mode not in _SIGNED:
            _floor_at_zero(row, 'mi')
        if bidirectional:
            back = {col: (vals[i] if i < len(vals) else math.nan) for col, vals in reverse.items()}
            row.update(back)
            row['te_yx'] = sum(sign * back[col] for col, _, sign in _TRANSFER_REVERSE)
            _floor_at_zero(row, 'te_yx')
            total = abs(row['mi']) + abs(row['te_yx'])
            row['directionality_index'] = (row['mi'] - row['te_yx']) / total if total > 1e-10 else 0.0
        rows.append(row)
    for col, key, _ in spec + (_TRANSFER_REVERSE if bidirectional else []):
        for i, result in enumerate(raw.get(key) or []):
            rid = run_ids[i] if i < len(run_ids) else i
            trainings.append({'run_id': rid, 'component': col,
                              **convert_record(network_values(result), to_bits)})
    return rows, trainings


def _difference_scalars(mode, rows, bidirectional=False) -> Dict[str, Any]:
    """Amplification of the configuration's aggregate, from the component means.

    It uses the measured combination, since a value reported as 0 would make
    the factor infinite."""
    spec = _DIFFERENCES[mode]
    means = [float(np.nanmean([r[col] for r in rows])) for col, _, _ in spec]
    result = float(np.nanmean([r.get('mi_raw', r['mi']) for r in rows]))
    scalars = {'amplification_factor': amplification_factor(means, result)}
    if bidirectional:
        back = [float(np.nanmean([r[col] for r in rows])) for col, _, _ in _TRANSFER_REVERSE]
        te_yx = float(np.nanmean([r.get('te_yx_raw', r['te_yx']) for r in rows]))
        scalars['amplification_factor_yx'] = amplification_factor(back, te_yx)
    return scalars


def _call_difference(mode, x, y, w, params, repeat_grid, ctx, n_workers):
    """One configuration's call into the mode's analysis function."""
    if mode == 'conditional':
        from neural_mi.analysis.conditional import run_conditional_mi
        return run_conditional_mi(x, y, w, params, sweep_grid=repeat_grid, n_workers=n_workers,
                                  align=ctx.get('align'), c_data=w,
                                  raw_deferred=ctx.get('raw_deferred', False),
                                  w_processor_type=ctx.get('w_processor_type'),
                                  c_processor_type=ctx.get('w_processor_type'),
                                  c_processor_params=ctx.get('w_processor_params'))
    if mode == 'interaction':
        from neural_mi.analysis.interaction import run_interaction_information
        return run_interaction_information(x, y, w, params, sweep_grid=repeat_grid, n_workers=n_workers,
                                           raw_deferred=ctx.get('raw_deferred', False),
                                           w_processor_type=ctx.get('w_processor_type'))
    from neural_mi.analysis.transfer import run_transfer_entropy
    return run_transfer_entropy(x, y, params, history_window=ctx['history_window'],
                                prediction_horizon=ctx.get('prediction_horizon', 1),
                                stride=ctx.get('stride', 1), sweep_grid=repeat_grid,
                                n_workers=n_workers, bidirectional=ctx.get('bidirectional', False),
                                w_data=w)


def produce_difference(mode, x, y, w, base_params, grid, *, to_bits, ctx, n_workers=1):
    """conditional, interaction, transfer: every configuration's repeats, paired."""
    configs, run_ids = split_grid(grid)
    config_keys = [k for k in (grid or {}) if k != 'run_id']
    repeat_grid = {'run_id': run_ids} if 'run_id' in (grid or {}) else None
    bidirectional = mode == 'transfer' and ctx.get('bidirectional', False)
    rows, details, scalars = [], {}, {}
    for cid, cfg in enumerate(configs):
        params = merge_grid_values(base_params, cfg)
        raw = _call_difference(mode, x, y, w, params, repeat_grid, ctx, n_workers)
        cfg_rows, trainings = _difference_rows(mode, raw, cid, cfg, config_keys, run_ids, to_bits,
                                               bidirectional=bidirectional)
        rows.extend(cfg_rows)
        details[cid] = {'trainings': pd.DataFrame(trainings)}
        if mode == 'transfer':
            details[cid]['n_samples'] = raw.get('n_samples')
        scalars[cid] = _difference_scalars(mode, cfg_rows, bidirectional=bidirectional)
    mean_cols = [col for col, _, _ in _DIFFERENCES[mode]]
    if bidirectional:
        mean_cols += [col for col, _, _ in _TRANSFER_REVERSE] + ['te_yx', 'directionality_index']
    return _produced(rows, details, config_keys, mean_cols=mean_cols, row_scalars=scalars)


def produce_difference_rigorous(mode, x, y, base_params, grid, *, to_bits, ctx, n_workers=1):
    """conditional, interaction, transfer with rigorous=True: one extrapolation
    of the combined quantity per configuration and repeat."""
    from neural_mi.analysis.rigorous import run_rigorous_scalar_analysis
    configs, run_ids = split_grid(grid)
    config_keys = [k for k in (grid or {}) if k != 'run_id']
    has_run_id = 'run_id' in (grid or {})
    rows, details = [], {}
    for cid, cfg in enumerate(configs):
        params = merge_grid_values(base_params, cfg)
        ladders = []
        for rid in run_ids:
            rep_params = _repeat_params(params, rid, has_run_id)
            fit = run_rigorous_scalar_analysis(
                scalar_fn=ctx['scalar_fn'], x_data=x, y_data=y, base_params=rep_params,
                extra_data=ctx.get('extra_data'),
                extra_kwargs=ctx.get('extra_kwargs'), n_workers=n_workers,
                raw_deferred=ctx.get('raw_deferred', False),
                **(ctx.get('temporal_kwargs') or {}), **ctx['rigorous_kwargs'])
            rows.append({'config_id': cid, **{k: cfg.get(k) for k in config_keys}, 'run_id': rid,
                         **_fit_row(fit, to_bits, quantity=mode)})
            ladder = fit.get('raw_results_df')
            if ladder is not None and len(ladder):
                frame = ladder.copy()
                frame['run_id'] = rid
                frame['component'] = 'combined'
                ladders.extend(_ladder_rows(frame, to_bits, {}))
        details[cid] = {'trainings': pd.DataFrame(ladders)}
    return _rigorous_produced(rows, details, config_keys)


# ---------------------------------------------------------------------------
# Pairwise, dimensionality, precision
# ---------------------------------------------------------------------------

def produce_pairwise(x, y, base_params, grid, *, to_bits, n_workers=1, pairs=None,
                     channel_names_x=None, channel_names_y=None):
    """One network per channel pair and repeat, for every configuration."""
    from neural_mi.analysis.pairwise import run_pairwise_mi
    configs, run_ids = split_grid(grid)
    config_keys = [k for k in (grid or {}) if k != 'run_id']
    repeat_grid = {'run_id': run_ids} if 'run_id' in (grid or {}) else None
    rows, details = [], {}
    for cid, cfg in enumerate(configs):
        params = merge_grid_values(base_params, cfg)
        raw = run_pairwise_mi(x, params, y_data=y, sweep_grid=repeat_grid, n_workers=n_workers, pairs=pairs)
        embeddings = {}
        for record in raw['records']:
            for i, run in enumerate(record['runs']):
                values = convert_record(run, to_bits)
                rid = run_ids[i] if i < len(run_ids) else i
                rows.append({'config_id': cid, **{k: cfg.get(k) for k in config_keys},
                             'ch_x': record['ch_x'], 'ch_y': record['ch_y'],
                             'run_id': rid, 'mi': values.get('train_mi'), **values})
                emb = (record.get('embeddings') or [{}] * (i + 1))[i]
                if emb:
                    # A pairwise repeat is one channel pair and one run.
                    embeddings[(record['ch_x'], record['ch_y'], rid)] = emb
        matrix = raw['mi_matrix'] * (NATS_TO_BITS if to_bits else 1.0)
        entry = {'mi_matrix': matrix, 'n_channels': raw['n_channels']}
        n_ch = raw['n_channels']
        if isinstance(n_ch, tuple):
            if channel_names_x is not None:
                entry['variable_names_y'] = list(channel_names_x)[:n_ch[0]]
            if channel_names_y is not None:
                entry['variable_names_x'] = list(channel_names_y)[:n_ch[1]]
        elif channel_names_x is not None:
            entry['variable_names_x'] = list(channel_names_x)[:n_ch]
            entry['variable_names_y'] = list(channel_names_x)[:n_ch]
        if embeddings:
            entry['embeddings'] = embeddings
        details[cid] = entry
    return _produced(rows, details, config_keys, axis_keys=('ch_x', 'ch_y'), first_cols=('eval_size',))


def produce_dimensionality(x, y, base_params, grid, *, to_bits, n_workers=1, dim_kwargs=None,
                           user_set_keys=None):
    """One curve of MI against embedding dimension per configuration.

    A repeat is one network: a restart at one embedding dimension of one split.
    Each ``dataframe`` row is one split and one embedding dimension, carrying
    the best restart (``mi_best``) and the curve (``mi_curve``), the running
    maximum of the best restarts over increasing dimension.
    """
    from neural_mi.analysis.dimensionality import _best, _curve, run_dimensionality_analysis
    configs, _ = split_grid(grid)
    config_keys = [k for k in (grid or {}) if k != 'run_id']
    dim_kwargs = dict(dim_kwargs or {})
    user_set = set(user_set_keys) if user_set_keys is not None else set(base_params)
    scale = NATS_TO_BITS if to_bits else 1.0
    rows, details = [], {}
    for cid, cfg in enumerate(configs):
        params = merge_grid_values(base_params, cfg)
        fits, info = run_dimensionality_analysis(x, params, y_data=y, n_workers=n_workers,
                                                 user_set_keys=user_set | set(cfg), **dim_kwargs)
        best = {s: _best(fits, s) for s in {f['split_id'] for f in fits}}
        curves = {s: _curve(b) for s, b in best.items()}
        entry = dict(info, plateau={s: v * scale for s, v in info['plateau'].items()})
        embeddings = {}
        for f in fits:
            s, k, r = f['split_id'], int(f['embedding_dim']), f['run_id']
            values = convert_record(network_values(f), to_bits)
            rows.append({'config_id': cid, **{c: cfg.get(c) for c in config_keys},
                         'split_id': s, 'embedding_dim': k, 'run_id': r,
                         'mi': 0.0 if f.get('all_mi_negative') else values.get('train_mi'), **values,
                         'mi_best': best[s][k] * scale, 'mi_curve': curves[s][k] * scale})
            found = embedding_values(f)
            if found:
                embeddings[(s, k, r)] = found
        if embeddings:
            entry['embeddings'] = embeddings
            at_bound = {}
            for s, k in info['dimension_at_most_per_split'].items():
                fits_at = [f for f in fits if f['split_id'] == s and int(f['embedding_dim']) == k]
                if fits_at:
                    top = max(fits_at, key=lambda f: f.get('train_mi') or -math.inf)
                    at_bound[s] = embeddings.get((s, k, top['run_id']))
            entry['embeddings_at_bound'] = at_bound
        details[cid] = entry
    rows.sort(key=lambda row: (row['config_id'], row['split_id'], row['embedding_dim'], row['run_id']))
    return _produced(rows, details, config_keys, axis_keys=('split_id', 'embedding_dim'),
                     mean_cols=('test_mi', 'pr_eig', 'pr_singular'),
                     first_cols=('mi_best', 'mi_curve'))


def produce_precision(x, y, base_params, *, to_bits, n_workers=1, precision_kwargs=None):
    """One baseline network, evaluated at every tau."""
    from neural_mi.analysis.precision import run_precision_analysis
    out = run_precision_analysis(x, y, base_params, n_workers=n_workers, **(precision_kwargs or {}))
    rows = []
    for record in out['samples']:
        value = convert_record({'mi': record['mi']}, to_bits)['mi']
        row = {'config_id': 0, 'tau': record['tau'], 'mi': value}
        # Past the point where the curve crosses zero the frozen critic's bound
        # has broken. Those values are reported as 0, measured ones kept, and
        # precision.py's warning says so. The threshold is read off the
        # measured curve, and a crossing of a positive threshold is the same
        # either way.
        _floor_at_zero(row, 'mi')
        if 'noise_sample' in record:
            row['noise_sample'] = record['noise_sample']
        rows.append(row)
    info = dict(out['details'])
    scale = NATS_TO_BITS if to_bits else 1.0
    for key in ('baseline_mi', 'threshold_value'):
        if info.get(key) is not None:
            info[key] = info[key] * scale
    for entry in (info.get('precision_thresholds') or {}).values():
        if entry.get('threshold_value') is not None:
            entry['threshold_value'] = entry['threshold_value'] * scale
    info.pop('raw_results', None)
    return _produced(rows, {0: info}, (), axis_keys=('tau',), mean_cols=())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def produce(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers=1):
    """Run `mode`'s producer. The observed call and every permutation trial come
    through here with the same arguments, except that a trial's `x` is moved in
    time (see :mod:`neural_mi.analysis.permutation`).

    `ctx` carries the mode's own settings (``lag_range``, the rigorous fit
    settings, ``history_window``, ...), prepared once by :func:`neural_mi.run`.
    """
    if mode in ('estimate', 'sweep'):
        return produce_single_network(mode, x, y, base_params, grid, to_bits=to_bits,
                                      n_workers=n_workers, is_proc_sweep=ctx.get('is_proc_sweep'))
    if mode == 'lag':
        return produce_lag(x, y, base_params, grid, to_bits=to_bits, lag_range=ctx['lag_range'],
                           n_workers=n_workers, equalize_n=ctx.get('equalize_n', False))
    if mode == 'rigorous':
        return produce_rigorous(x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers,
                                rigorous_kwargs=ctx.get('rigorous_kwargs'))
    if mode in _DIFFERENCES:
        if ctx.get('rigorous'):
            return produce_difference_rigorous(mode, x, y, base_params, grid, to_bits=to_bits,
                                               ctx=ctx, n_workers=n_workers)
        return produce_difference(mode, x, y, w, base_params, grid, to_bits=to_bits, ctx=ctx,
                                  n_workers=n_workers)
    if mode == 'pairwise':
        return produce_pairwise(x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers,
                                pairs=ctx.get('pairs'), channel_names_x=ctx.get('channel_names_x'),
                                channel_names_y=ctx.get('channel_names_y'))
    if mode == 'dimensionality':
        return produce_dimensionality(x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers,
                                      dim_kwargs=ctx.get('dim_kwargs'),
                                      user_set_keys=ctx.get('user_set_keys'))
    if mode == 'precision':
        return produce_precision(x, y, base_params, to_bits=to_bits, n_workers=n_workers,
                                 precision_kwargs=ctx.get('precision_kwargs'))
    raise ValueError(f"No producer for mode='{mode}'.")
