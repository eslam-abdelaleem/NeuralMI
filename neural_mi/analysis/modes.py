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
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from neural_mi.defaults import default_of
from neural_mi.analysis.assemble import (
    config_lookup, convert_record, embedding_values, network_values, single_network_rows,
    split_grid, NATS_TO_BITS, NETWORK_KEYS,
)
from neural_mi.analysis.sweep import (ParameterSweep, amplification_factor, combine_terms,
                                      merge_grid_values)
from neural_mi.logger import user_stacklevel
from neural_mi.analysis.dimensionality import mode_defaults as dimensionality_defaults
from neural_mi.config import (Conditional, Dimensionality, Interaction, Lag, Pairwise, Precision,
                              Rigorous, Transfer)
from neural_mi.analysis.conditional import _cmi_rigorous_scalar, run_conditional_mi
from neural_mi.analysis.interaction import _ii_rigorous_scalar, run_interaction_information
from neural_mi.analysis.transfer import _te_rigorous_scalar, run_transfer_entropy
from neural_mi.results import (_draw_axis_curve, _draw_components, _draw_dimensionality,
                               _draw_directions, _draw_extrapolation, _draw_matrix, _draw_precision,
                               _draw_training, _overlay_extrapolations, _overlay_training,
                               _summary_dimensionality, _summary_matrix, _summary_precision)
from neural_mi.validation import (check_dimensionality_settings, check_lag_settings,
                                  check_precision_settings, check_rigorous_settings)

# Repeat-level fit values a rigorous extrapolation reports.
_FIT_KEYS = (
    'mi_error', 'mi_error_pred', 'slope', 'is_reliable', 'linear_region_found',
    'enough_gamma_points', 'curvature_coefficient', 'curvature_se', 'curvature_t',
    'curvature_slope', 'fit_quality_warning', 'leverage_warning', 'r_squared',
    'max_abs_residual', 'loo_intercept_shift', 'gammas_used', 'chunking_mode',
    'n_tasks_created', 'saturated_gammas', 'zero_rungs', 'unresolved_rungs',
)

_TRANSFER_REVERSE = [('i_yxpast_xfuture', 'raw_yxpast_xfuture', 1),
                     ('i_xpast_xfuture', 'raw_xpast_xfuture', -1)]


def _floor_at_zero(row: Dict[str, Any], key: str) -> bool:
    """Report a negative ``row[key]`` as 0 and keep the measured value as
    ``row[key + '_raw']``. Returns whether the value was negative."""
    value = row.get(key)
    negative = value is not None and not (isinstance(value, float) and math.isnan(value)) and value < 0
    row[f'{key}_raw'] = value
    if negative:
        row[key] = 0.0
    return negative


def _produced(rows, details, config_keys, *, mean_cols=('test_mi',), std_cols=(),
              first_cols=(), row_scalars=None) -> Dict[str, Any]:
    """A producer's rows and how to aggregate them. The axis they run along within each configuration is
    the mode's own, declared in MODES, which produce() adds."""
    return dict(rows=rows, details=details, config_keys=list(config_keys),
                mean_cols=list(mean_cols), std_cols=list(std_cols), first_cols=list(first_cols),
                row_scalars=row_scalars or {})


def _repeat_params(params: Dict[str, Any], run_id: Any, has_run_id: bool) -> Dict[str, Any]:
    """One repeat's settings. Its run_id gives every network of the repeat its seeds (neural_mi/seeds.py)."""
    out = dict(params)
    if has_run_id:
        out['run_id'] = run_id
    return out


# ---------------------------------------------------------------------------
# Modes whose repeat is one network
# ---------------------------------------------------------------------------

def produce_single_network(mode, x, y, base_params, grid, *, to_bits, n_workers=1):
    """estimate and sweep: one network per repeat."""
    results = ParameterSweep(x, y, base_params).run(grid, n_workers=n_workers)
    rows, details, config_keys = single_network_rows(results, grid, to_bits)
    return _produced(rows, details, config_keys)


def produce_lag(x, y, base_params, grid, *, to_bits, lag_range, n_workers=1, equalize_n=False):
    """mode='lag': one network per lag and repeat, for every configuration."""
    from neural_mi.analysis.lag import run_lag_analysis
    results = run_lag_analysis(x, y, base_params, lag_range=lag_range, sweep_grid=grid,
                               n_workers=n_workers, equalize_n=equalize_n)
    rows, details, config_keys = single_network_rows(results, grid, to_bits, axis=('lag',))
    return _produced(rows, details, config_keys, first_cols=('n_windows_built',))


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

def _combined(spec, raw, n_repeats, scale):
    """One direction of a difference quantity, combined by the rule every combined quantity follows
    (:func:`neural_mi.analysis.sweep.combine_terms`), padded to ``n_repeats`` and in the reported units."""
    combined = combine_terms([(col, col, raw.get(key) or [], sign) for col, key, sign in spec])

    def padded(values):
        return [v * scale for v in values] + [math.nan] * (n_repeats - len(values))

    failed = combined['failed'] + [[col for col, _, _ in spec]] * (n_repeats - len(combined['failed']))
    return (padded(combined['per_repeat']),
            {col: padded(combined['components'][col]) for col, _, _ in spec}, failed)


def _difference_rows(mode, raw, cid, cfg, config_keys, run_ids, to_bits, bidirectional=default_of('bidirectional')):
    """Pair the components' repeats by position and combine each pair.

    A repeat in which a term failed has no value: its combination and its terms
    are NaN, and ``failed_term`` names the terms that failed.
    """
    spec = _DIFFERENCES[mode]
    scale = NATS_TO_BITS if to_bits else 1.0
    n = len(run_ids)
    values, components, failed = _combined(spec, raw, n, scale)
    if bidirectional:
        back_values, back_components, back_failed = _combined(_TRANSFER_REVERSE, raw, n, scale)
    rows, trainings = [], []
    for i, rid in enumerate(run_ids):
        comp = {col: vals[i] for col, vals in components.items()}
        row = {'config_id': cid, **{k: cfg.get(k) for k in config_keys}, 'run_id': rid,
               'mi': values[i], **comp}
        if mode not in _SIGNED:
            _floor_at_zero(row, 'mi')
        failed_terms = list(failed[i])
        if bidirectional:
            row.update({col: vals[i] for col, vals in back_components.items()})
            row['te_yx'] = back_values[i]
            _floor_at_zero(row, 'te_yx')
            failed_terms += back_failed[i]
            total = abs(row['mi']) + abs(row['te_yx'])
            if math.isnan(total):
                row['directionality_index'] = math.nan
            else:
                row['directionality_index'] = (row['mi'] - row['te_yx']) / total if total > 1e-10 else 0.0
        row['failed_term'] = ', '.join(failed_terms) if failed_terms else None
        rows.append(row)
    for col, key, _ in spec + (_TRANSFER_REVERSE if bidirectional else []):
        for i, result in enumerate(raw.get(key) or []):
            rid = run_ids[i] if i < len(run_ids) else i
            trainings.append({'run_id': rid, 'component': col,
                              **convert_record(network_values(result), to_bits)})
    return rows, trainings


def _mean_with_value(values) -> float:
    """The mean of the values that are not NaN, or NaN when there are none."""
    kept = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(kept)) if kept else math.nan


def _difference_scalars(mode, rows, bidirectional=default_of('bidirectional')) -> Dict[str, Any]:
    """Amplification of the configuration's aggregate, from the component means.

    It uses the measured combination, since a value reported as 0 would make
    the factor infinite. Repeats without a value are left out of every mean."""
    spec = _DIFFERENCES[mode]
    means = [_mean_with_value([r[col] for r in rows]) for col, _, _ in spec]
    result = _mean_with_value([r.get('mi_raw', r['mi']) for r in rows])
    scalars = {'amplification_factor': amplification_factor(means, result)}
    if bidirectional:
        back = [_mean_with_value([r[col] for r in rows]) for col, _, _ in _TRANSFER_REVERSE]
        te_yx = _mean_with_value([r.get('te_yx_raw', r['te_yx']) for r in rows])
        scalars['amplification_factor_yx'] = amplification_factor(back, te_yx)
    return scalars


def _call_difference(mode, x, y, w, params, repeat_grid, ctx, n_workers):
    """One configuration's call into the mode's analysis function, which its declaration names."""
    return MODES[mode].run_terms(x, y, w, params, repeat_grid, ctx, n_workers)


def produce_difference(mode, x, y, w, base_params, grid, *, to_bits, ctx, n_workers=1):
    """conditional, interaction, transfer: every configuration's repeats, paired."""
    configs, run_ids = split_grid(grid)
    config_keys = [k for k in (grid or {}) if k != 'run_id']
    repeat_grid = {'run_id': run_ids} if 'run_id' in (grid or {}) else None
    # Only transfer entropy's ctx can ask for both directions.
    bidirectional = bool(ctx.get('bidirectional'))
    rows, details, scalars = [], {}, {}
    for cid, cfg in enumerate(configs):
        params = merge_grid_values(base_params, cfg)
        raw = _call_difference(mode, x, y, w, params, repeat_grid, ctx, n_workers)
        cfg_rows, trainings = _difference_rows(mode, raw, cid, cfg, config_keys, run_ids, to_bits,
                                               bidirectional=bidirectional)
        rows.extend(cfg_rows)
        details[cid] = {'trainings': pd.DataFrame(trainings)}
        if raw.get('n_samples') is not None:
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
    return _produced(rows, details, config_keys, first_cols=('eval_size',))


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
    return _produced(rows, details, config_keys,
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
    return _produced(rows, {0: info}, (), mean_cols=())


# ---------------------------------------------------------------------------
# Each mode's own settings for its producer (its ctx), prepared from the call
# (run._run_flat builds ``call``). A mode that builds its own rows from the
# series puts them in call.x_run, call.y_run and call.w_run.
# ---------------------------------------------------------------------------

# Rigorous(...) settings that run_rigorous_analysis takes as keywords, beside the
# three that _run_flat names (curvature_t_threshold, min_gamma_points,
# confidence_level).
_RIGOROUS_FIT_KEYS = ('gamma_range', 'residual_threshold', 'leverage_threshold',
                      'temporal_chunking')


def _to_2d(t):
    """``(T, C, 1)`` back to ``(T, C)``, the shape transfer entropy builds its histories from."""
    if hasattr(t, 'ndim') and t.ndim == 3 and t.shape[-1] == 1:
        return t.reshape(t.shape[0], t.shape[1]).contiguous()
    return t


# The settings a mode's config gives run(), each with its type, bounds and default.
_WORKERS = {'n_workers': {'type': int, 'default': 1}}
# The fit of a rigorous ladder, in mode='rigorous' and in a difference quantity run with rigorous=True.
_FIT = {
    'curvature_t_threshold': {'type': float, 'default': 2.0},
    'min_gamma_points': {'type': int, 'default': 5},
    'confidence_level': {'type': float, 'default': 0.68},
    'residual_threshold': {'type': float, 'default': 2.5},
    'leverage_threshold': {'type': float, 'default': 0.20},
}
_DIFFERENCE_SETTINGS = {**_WORKERS, 'rigorous': {'type': bool, 'default': False},
                        'gamma_range': {'type': (range, list, type(None)), 'default': None}, **_FIT}


def _scalar_rigorous_kwargs(analysis_kwargs: dict, curvature_t_threshold: float,
                            min_gamma_points: int, confidence_level: float) -> dict:
    """The fit settings of a difference quantity run with ``rigorous=True``."""
    return {
        'gamma_range': analysis_kwargs.get('gamma_range') or range(1, 11),
        'curvature_t_threshold': analysis_kwargs.get('curvature_t_threshold', curvature_t_threshold),
        'min_gamma_points': analysis_kwargs.get('min_gamma_points', min_gamma_points),
        'confidence_level': analysis_kwargs.get('confidence_level', confidence_level),
        'residual_threshold': analysis_kwargs.get('residual_threshold', _FIT['residual_threshold']['default']),
        'leverage_threshold': analysis_kwargs.get('leverage_threshold', _FIT['leverage_threshold']['default']),
    }


def _prepare_lag(call) -> Dict[str, Any]:
    # `lag_range` reaches here already unpacked from Lag(...) by run(); the
    # analysis_kwargs fallback covers direct _run_flat callers.
    lag_range = call.lag_range if call.lag_range is not None else call.analysis_kwargs.get('lag_range')
    if lag_range is None:
        raise ValueError(
            "`lag_range` must be provided for mode='lag'. Pass it in the per-mode "
            "config: nmi.run(..., mode='lag', lag=Lag(lag_range=range(-10, 11)))."
        )
    call.record['lag_range'] = list(lag_range)
    return {'lag_range': lag_range, 'equalize_n': call.analysis_kwargs.get('equalize_n', False)}


def _prepare_precision(call) -> Dict[str, Any]:
    if call.tau_grid is None:
        raise ValueError("`tau_grid` must be provided for mode='precision'.")
    call.record['tau_grid'] = list(call.tau_grid)
    return {'precision_kwargs': dict(tau_grid=call.tau_grid, corrupt_target=call.corrupt_target,
                                     corruption_method=call.corruption_method,
                                     n_noise_samples=call.n_noise_samples,
                                     threshold_ratio=call.threshold_ratio)}


def _prepare_rigorous(call) -> Dict[str, Any]:
    fit = {k: v for k, v in call.analysis_kwargs.items() if k in _RIGOROUS_FIT_KEYS}
    fit.update(curvature_t_threshold=call.curvature_t_threshold, min_gamma_points=call.min_gamma_points,
               confidence_level=call.confidence_level)
    return {'rigorous_kwargs': fit}


def _prepare_dimensionality(call) -> Dict[str, Any]:
    return {'dim_kwargs': {k: v for k, v in call.analysis_kwargs.items() if k != 'n_workers'},
            'user_set_keys': call.user_set_keys}


def _prepare_pairwise(call) -> Dict[str, Any]:
    call.y_run = call.y_run if call.y_data is not None else None
    return {'pairs': call.analysis_kwargs.get('pairs'), 'channel_names_x': call.channel_names_x,
            'channel_names_y': call.channel_names_y}


def _rigorous_terms(call, **more) -> Dict[str, Any]:
    """The ctx of a difference quantity run with rigorous=True: the scalar its ladder fits, the fit
    settings, and whatever the quantity adds."""
    call.record['rigorous'] = True
    return dict(rigorous=True, scalar_fn=MODES[call.mode].rigorous_scalar,
                rigorous_kwargs=_scalar_rigorous_kwargs(call.analysis_kwargs, call.curvature_t_threshold,
                                                        call.min_gamma_points, call.confidence_level),
                **more)


def _prepare_terms(call) -> Dict[str, Any]:
    """conditional and interaction: call.x_run is the plan of X, W and Y, whose rows every term shares."""
    return _rigorous_terms(call) if call.rigorous else {}


def _prepare_transfer(call) -> Dict[str, Any]:
    """Transfer entropy builds its histories from the rows of a (T, C) series. With processors, every
    stream goes onto one grid of rows one step wide that tile time, which the histories index by offset:
    the one path of processing= for every quantity defined by time offsets."""
    from collections import OrderedDict
    from neural_mi.analysis.offsets import one_step_series, spec_geometry
    from neural_mi.analysis.transfer import te_spec
    from neural_mi.data.rows import kept_rows, record_rows
    from neural_mi.preprocessing import prepare
    from neural_mi.run import _encoders_scale, _warn_small_sample
    from neural_mi.utils import float_tensor, validate_window, with_window_axis
    if call.history_window is None:
        raise ValueError("`history_window` must be provided for mode='transfer'.")
    history_window = validate_window(call.history_window, 'history_window', "mode='transfer'", 1)
    prediction_horizon = validate_window(call.prediction_horizon, 'prediction_horizon', "mode='transfer'", 1)
    if call.y_data is None:
        raise ValueError("y_data must be provided for mode='transfer'.")
    for name, arr in (('x_data', call.x_data), ('y_data', call.y_data), ('w_data', call.w_data)):
        if getattr(arr, 'ndim', None) == 3 and arr.shape[-1] != 1:
            raise ValueError(
                f"mode='transfer' requires {name} of shape (n_timepoints, n_channels) "
                f"and received a 3-D array of shape {tuple(arr.shape)}. Transfer "
                f"entropy builds its histories from the raw series. Pass it unwindowed "
                f"or set Processing(...) with one-step windows to put streams of other "
                f"kinds on one grid."
            )
    if any(t is not None for t in (call.processor_type_x, call.processor_type_y, call.w_processor_type)):
        specs = OrderedDict()
        specs['x'] = dict(data=call.x_data, time=call.x_time, processor_type=call.processor_type_x,
                          processor_params=call.processor_params_x)
        specs['y'] = dict(data=call.y_data, time=call.y_time, processor_type=call.processor_type_y,
                          processor_params=call.processor_params_y)
        if call.w_data is not None:
            specs['w'] = dict(data=call.w_data, time=call.w_time, processor_type=call.w_processor_type,
                              processor_params=call.w_processor_params)
        one_step = one_step_series(specs, "mode='transfer'")
        x_run, y_run, w_run = one_step['x'], one_step['y'], one_step.get('w')
    else:
        def series(data):
            return _to_2d(with_window_axis(float_tensor(data)))
        x_run, y_run = series(call.x_data), series(call.y_data)
        w_run = series(call.w_data) if call.w_data is not None else None
    # Preprocessing(...) and the data check act on the series the histories
    # are built from, after any grid. Transfer entropy is a rate quantity,
    # which a time transform leaves unchanged.
    prepared = prepare(OrderedDict(x=x_run, y=y_run, w=w_run), call.processor_of, call.preprocessing,
                       quantity="mode='transfer'", rate=True, check=call.rows is None,
                       encoder_scales=_encoders_scale(call.base_params))
    call.x_run, call.y_run, call.w_run = prepared['x'], prepared['y'], prepared['w']
    # The histories are built here, so the training tasks read them as given.
    base_params = call.base_params
    base_params['processor_type_x'] = None
    base_params['processor_type_y'] = None
    for key in ('processor_params_x', 'processor_params_y'):
        base_params[key] = {**(base_params.get(key) or {}), 'preprocessed': True}
    ctx = {'history_window': history_window, 'prediction_horizon': prediction_horizon,
           'stride': call.stride, 'bidirectional': call.bidirectional}
    call.record['history_window'] = history_window
    # The histories are rows in time order, recorded before anything reads
    # them: the ladder's chunks, the permutation's blocks.
    histories = spec_geometry(te_spec(history_window, prediction_horizon), call.stride)
    record_rows(base_params, histories)
    # The data check, as every other route runs it: the histories every term
    # trains on, counted from the series' length, and the scale of the series
    # they are cut from.
    n_histories = max(0, int((call.x_run.shape[0] - histories.span) // call.stride) + 1)
    _warn_small_sample(kept_rows(n_histories, base_params), 'samples', base_params, call.mode,
                       rows={name: s for name, s in (('x', call.x_run), ('y', call.y_run), ('w', call.w_run))
                             if s is not None},
                       processors=call.processor_of)
    if call.rigorous:
        ctx.update(_rigorous_terms(
            call, extra_data={'w_data': call.w_run} if call.w_run is not None else None,
            extra_kwargs={'history_window': history_window, 'prediction_horizon': prediction_horizon,
                          'stride': call.stride, 'bidirectional': call.bidirectional}))
    return ctx


# Each difference quantity's terms on one configuration.

def _conditional_terms(x, y, w, params, repeat_grid, ctx, n_workers):
    # x is the plan of X, W and Y (neural_mi/data/plan.py).
    return run_conditional_mi(x, params, sweep_grid=repeat_grid, n_workers=n_workers)


def _interaction_terms(x, y, w, params, repeat_grid, ctx, n_workers):
    return run_interaction_information(x, params, sweep_grid=repeat_grid, n_workers=n_workers)


def _transfer_terms(x, y, w, params, repeat_grid, ctx, n_workers):
    return run_transfer_entropy(x, y, params, history_window=ctx['history_window'],
                                prediction_horizon=ctx.get('prediction_horizon', 1),
                                stride=ctx.get('stride', 1), sweep_grid=repeat_grid,
                                n_workers=n_workers, bidirectional=bool(ctx.get('bidirectional')),
                                w_data=w)


# ---------------------------------------------------------------------------
# The modes, each declared once
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mode:
    """Everything the library knows about a mode, declared once.

    run(), the grid and data checks, the results and the reference pages read
    these fields, and a test checks USING.md's table of what each mode accepts
    against them. Adding a mode takes a config class, a producer and one entry
    in :data:`MODES`.
    """
    # Runs the mode: (mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers).
    produce: Callable[..., Dict[str, Any]]
    # Its config class, or None for a mode that takes no config of its own.
    config: Optional[type] = None
    # The settings its config gives run(), each with its type, bounds and default.
    settings: Mapping[str, Dict[str, Any]] = field(default_factory=lambda: dict(_WORKERS))
    # The axis its producer adds within each configuration (result.params['axis_keys']).
    axis: Tuple[str, ...] = ()
    # Its row of USING.md's table: repeats, a configuration grid, rigorous.
    table: Tuple[str, str, str] = ('yes', 'yes', 'refused')
    # It runs a configuration grid and needs one.
    needs_grid: bool = False
    # For a mode that trains one network and so runs no grid: what it trains and what to use instead.
    one_network: Optional[Tuple[str, str]] = None
    # Each repeat of each configuration is one network.
    one_network_per_repeat: bool = False
    # Why a permutation null means nothing for it, or None when one does.
    permutation: Optional[str] = None
    # Why it cannot return embeddings, or None when it can.
    embeddings: Optional[str] = None
    # The grid keys it sets itself, and the setting it takes each from.
    sets_itself: Mapping[str, str] = field(default_factory=dict)
    # Why its networks train on rows other than the ones passed, which custom split indices would misread.
    rows_differ: Optional[str] = None
    # It reads a third stream, W.
    reads_w: bool = False
    # It needs Y.
    needs_y: bool = True
    # The terms of a difference quantity: (column, raw-results key, sign).
    terms: Optional[Tuple[Tuple[str, str, int], ...]] = None
    # The quantity can be negative.
    signed: bool = False
    # The quantity's name in messages.
    label: Optional[str] = None
    # It windows inside each training task.
    windows_in_task: bool = False
    # It builds its own rows from the raw series.
    builds_own_rows: bool = False
    # It reads a trained critic, which a nonparametric estimator does not train.
    reads_critic: bool = False
    # It reads the critic's embedding of each side.
    reads_embeddings: bool = False
    # Its critic when the caller names none.
    critic: Optional[str] = None
    # The settings it sets itself when the caller left them unset: (params, user_set, has_y) -> settings.
    overrides: Optional[Callable[[Dict[str, Any], set, bool], Dict[str, Any]]] = None
    # Prepares its own settings for its producer (its ctx) from the call, which run._run_flat builds.
    prepare: Optional[Callable[[Any], Dict[str, Any]]] = None
    # Checks its own settings (validation.ParameterValidator).
    check: Optional[Callable[[Any], None]] = None
    # A difference quantity's terms on one configuration, and the scalar its rigorous ladder fits.
    run_terms: Optional[Callable[..., Dict[str, Any]]] = None
    rigorous_scalar: Optional[Callable[..., Dict[str, Any]]] = None
    # Its result is an extrapolation.
    extrapolates: bool = False
    # Its figure of one configuration, the lines summary() adds, and how compare() overlays results
    # (results.py): a figure returns the axes and whether it showed them.
    draw: Optional[Callable] = None
    summarize: Optional[Callable] = None
    overlay: Optional[Callable] = None
    # It draws a figure of its own rather than into given axes.
    own_figure: bool = False
    # Its figure is per configuration, never a curve over the swept keys.
    draws_per_config: bool = False
    # Its axis is drawn like a swept key.
    groups_by_axis: bool = False
    # Each value of its axis is a network of its own. Precision's tau is one network evaluated many times.
    axis_is_networks: bool = False
    # Its quantity's symbol, the components drawn beside it, and the title of its figure.
    symbol: Optional[str] = None
    components: Tuple[Tuple[str, str], ...] = ()
    title: Optional[str] = None


def _single(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    return produce_single_network(mode, x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers)


def _lag(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    return produce_lag(x, y, base_params, grid, to_bits=to_bits, lag_range=ctx['lag_range'],
                       n_workers=n_workers, equalize_n=ctx.get('equalize_n', False))


def _rigorous(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    return produce_rigorous(x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers,
                            rigorous_kwargs=ctx.get('rigorous_kwargs'))


def _difference(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    if ctx.get('rigorous'):
        return produce_difference_rigorous(mode, x, y, base_params, grid, to_bits=to_bits, ctx=ctx,
                                           n_workers=n_workers)
    return produce_difference(mode, x, y, w, base_params, grid, to_bits=to_bits, ctx=ctx, n_workers=n_workers)


def _pairwise(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    return produce_pairwise(x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers,
                            pairs=ctx.get('pairs'), channel_names_x=ctx.get('channel_names_x'),
                            channel_names_y=ctx.get('channel_names_y'))


def _dimensionality(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    return produce_dimensionality(x, y, base_params, grid, to_bits=to_bits, n_workers=n_workers,
                                  dim_kwargs=ctx.get('dim_kwargs'), user_set_keys=ctx.get('user_set_keys'))


def _precision(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers):
    return produce_precision(x, y, base_params, to_bits=to_bits, n_workers=n_workers,
                             precision_kwargs=ctx.get('precision_kwargs'))


_TERMS_HOLD_NO_EMBEDDINGS = ("every repeat trains one network per term of the quantity, and no one "
                             "network's embeddings describe it")

MODES: Dict[str, Mode] = {
    'estimate': Mode(_single, table=('refused', 'refused', "through `mode='rigorous'`"),
                     one_network=("trains one network",
                                  "Use mode='sweep' to repeat the estimate or to sweep a setting."),
                     one_network_per_repeat=True, draw=_draw_training, overlay=_overlay_training),
    'sweep': Mode(_single, table=('yes', 'yes', "through `mode='rigorous'` with the same grid"),
                  needs_grid=True, one_network_per_repeat=True, draw=_draw_training),
    'rigorous': Mode(_rigorous, config=Rigorous,
                     settings={**_WORKERS, **_FIT,
                               # None: contiguous chunks for rows recorded in time order
                               # (neural_mi/data/rows.py), permuted otherwise; True/False overrides.
                               'temporal_chunking': {'type': (bool, type(None)), 'default': None}},
                     table=('one extrapolation per repeat', 'yes', 'built in'),
                     permutation="The extrapolation reports its own error estimate.",
                     embeddings="every repeat trains a network on each chunk of the gamma ladder",
                     rows_differ='trains on chunks of the data', label='MI', prepare=_prepare_rigorous,
                     check=check_rigorous_settings, extrapolates=True, draw=_draw_extrapolation,
                     overlay=_overlay_extrapolations),
    'lag': Mode(_lag, config=Lag, axis=('lag',),
                settings={**_WORKERS, 'lag_range': {'type': (range, list, np.ndarray), 'required': True},
                          'equalize_n': {'type': bool, 'default': False}}, sets_itself={'lag': "Lag(lag_range=...)"},
                rows_differ='trains on copies of the data shifted by each lag', windows_in_task=True,
                prepare=_prepare_lag, check=check_lag_settings, draw=_draw_axis_curve, groups_by_axis=True,
                axis_is_networks=True),
    'precision': Mode(_precision, config=Precision, axis=('tau',),
                      settings={**_WORKERS, 'tau_grid': {'type': list, 'required': True},
                                # corrupt_target: 'x' corrupts only X, 'y' only Y, 'both' corrupts both
                                # simultaneously (useful for measuring shared spike-timing precision).
                                'corrupt_target': {'type': str, 'default': 'x'},
                                'corruption_method': {'type': str, 'default': 'rounding'},
                                'n_noise_samples': {'type': int, 'default': 50},
                                'threshold_ratio': {'type': (float, list, tuple), 'default': 0.9}}, table=('refused', 'refused', 'refused'),
                      one_network=("trains one baseline network and evaluates it at every tau in tau_grid",
                                   "To compare settings, run mode='precision' once per setting."),
                      permutation="The mode reports how one trained network degrades, not an MI value to test.",
                      embeddings=("it trains one baseline network and reports how its MI degrades, so there "
                                  "is no repeat whose embeddings the result could hold"),
                      reads_critic=True, prepare=_prepare_precision, check=check_precision_settings,
                      draw=_draw_precision, summarize=_summary_precision, draws_per_config=True),
    'conditional': Mode(_difference, config=Conditional, settings=_DIFFERENCE_SETTINGS, table=('yes', 'yes', "`rigorous=True` on the mode's config"),
                        embeddings=_TERMS_HOLD_NO_EMBEDDINGS, reads_w=True,
                        terms=(('mi_xw_y', 'raw_xw_y', 1), ('mi_w_y', 'raw_w_y', -1)), label='Conditional MI',
                        prepare=_prepare_terms, run_terms=_conditional_terms,
                        rigorous_scalar=_cmi_rigorous_scalar, draw=_draw_components, symbol='I(X;Y|W)',
                        components=(('mi_xw_y', 'I(X,W;Y)'), ('mi_w_y', 'I(W;Y)')),
                        title='Conditional MI components'),
    'interaction': Mode(_difference, config=Interaction, settings=_DIFFERENCE_SETTINGS, table=('yes', 'yes', "`rigorous=True` on the mode's config"),
                        embeddings=_TERMS_HOLD_NO_EMBEDDINGS, reads_w=True,
                        terms=(('mi_xw_y', 'raw_xw_y', 1), ('mi_x_y', 'raw_x_y', -1), ('mi_w_y', 'raw_w_y', -1)),
                        signed=True, label='Interaction information', prepare=_prepare_terms,
                        run_terms=_interaction_terms, rigorous_scalar=_ii_rigorous_scalar,
                        draw=_draw_components, symbol='II',
                        components=(('mi_xw_y', 'I(X,W;Y)'), ('mi_x_y', 'I(X;Y)'), ('mi_w_y', 'I(W;Y)')),
                        title='Interaction information components'),
    'transfer': Mode(_difference, config=Transfer, settings=_DIFFERENCE_SETTINGS, table=('yes', 'yes', "`rigorous=True` on the mode's config"),
                     embeddings=_TERMS_HOLD_NO_EMBEDDINGS, rows_differ='trains on histories built from the data',
                     reads_w=True, terms=(('i_xypast_yfuture', 'raw_xypast_yfuture', 1),
                                          ('i_ypast_yfuture', 'raw_ypast_yfuture', -1)),
                     label='TE(X→Y)', builds_own_rows=True, prepare=_prepare_transfer,
                     run_terms=_transfer_terms, rigorous_scalar=_te_rigorous_scalar, draw=_draw_directions,
                     symbol='TE(X→Y)', title='Transfer entropy'),
    'pairwise': Mode(_pairwise, config=Pairwise, axis=('ch_x', 'ch_y'),
                     settings={**_WORKERS, 'pairs': {'type': (list, type(None)), 'default': None}}, needs_y=False, prepare=_prepare_pairwise,
                     draw=_draw_matrix, summarize=_summary_matrix, own_figure=True, draws_per_config=True,
                     axis_is_networks=True),
    'dimensionality': Mode(_dimensionality, config=Dimensionality, axis=('split_id', 'embedding_dim'),
                           settings={
                               **_WORKERS,
                               # The embedding dimensions to fit. None chooses them from the reference fit.
                               'embedding_dims': {'type': (list, tuple, range, type(None)), 'default': None},
                               'n_restarts': {'type': int, 'min': 1, 'default': 4},
                               'saturation_ratio': {'type': float, 'min': 0.0, 'default': 0.95},
                               # None is 64, refitted larger when its participation ratio comes close to that.
                               'reference_dim': {'type': (int, type(None)), 'min': 2, 'default': None},
                               # 'random', 'spatial', 'temporal', 'index', 'horizontal', 'vertical',
                               # 'row_interleaved', 'col_interleaved', 'diagonal' or 'antidiagonal'.
                               'split_method': {'type': str, 'default': 'random'},
                               # Random channel splits of X; the other split methods give one split.
                               'n_splits': {'type': (int, type(None)), 'min': 1, 'default': None},
                               # With split_method='temporal'.
                               'lag': {'type': int, 'default': 1},
                               # With split_method='index': the channels of X. Y is the rest.
                               'channel_indices_x': {'type': (list, type(None)), 'default': None},
                               # Warn when the reference fit's MI is this close to its evaluation ceiling.
                               'ceiling_mi_fraction': {'type': float, 'default': 0.85},
                           },
                           table=('through `n_restarts`', 'one curve per configuration', 'refused'),
                           permutation=("The mode reports a curve over embedding dimensions and the dimension "
                                        "at which it saturates. Neither has a null distribution."),
                           sets_itself={'run_id': "Dimensionality(n_restarts=...)",
                                        'embedding_dim': "Dimensionality(embedding_dims=...)"},
                           needs_y=False, reads_critic=True, reads_embeddings=True, critic='hybrid',
                           overrides=dimensionality_defaults, prepare=_prepare_dimensionality,
                           check=check_dimensionality_settings, draw=_draw_dimensionality,
                           summarize=_summary_dimensionality, own_figure=True, draws_per_config=True,
                           axis_is_networks=True),
}

# The difference quantities' terms, read by the producers above.
_DIFFERENCES = {name: list(mode.terms) for name, mode in MODES.items() if mode.terms}
_SIGNED = tuple(name for name, mode in MODES.items() if mode.signed)
_QUANTITY_NAMES = {name: mode.label for name, mode in MODES.items() if mode.label}


def produce(mode, x, y, w, base_params, grid, *, ctx, to_bits, n_workers=1):
    """Run `mode`'s producer. The observed call and every permutation trial come
    through here with the same arguments, except that a trial's `x` is moved in
    time (see :mod:`neural_mi.analysis.permutation`).

    `ctx` carries the mode's own settings (``lag_range``, the rigorous fit
    settings, ``history_window``, ...), prepared once by :func:`neural_mi.run`.
    """
    spec = MODES[mode]
    produced = spec.produce(mode, x, y, w, base_params, grid, ctx=ctx, to_bits=to_bits, n_workers=n_workers)
    return {**produced, 'axis_keys': list(spec.axis)}
