# neural_mi/analysis/rigorous.py
"""Implements the 'rigorous' bias-corrected MI analysis mode.

Provides the public function ``run_rigorous_analysis`` and the
``AnalysisWorkflow`` class that orchestrate the multi-step process for a
rigorous, bias-corrected mutual information estimate.

The estimator trains models on progressively smaller data subsets (parameterised
by *gamma*, the number of data subsets) and extrapolates the MI vs. gamma line
to gamma = 0 to obtain a bias-corrected MI value.  All per-run MI estimates are
``train_mi`` (evaluated on the large training partition at the best-generalising
checkpoint), consistent with every other analysis mode.
"""
import warnings
import numpy as np
import pandas as pd
import itertools
import uuid
import statsmodels.api as sm
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import CapturedTask, logger, released, user_stacklevel
from neural_mi.device import with_device
from neural_mi.estimators import NONPARAMETRIC
from neural_mi.parallel import worker_pool
from neural_mi.embeddings_io import with_model_labels
from neural_mi.exceptions import CombinationWarning, InsufficientDataError, TrainingError
from neural_mi.utils import _ensure_cpu
from neural_mi.data.plan import PAIR, RowPlan, TermRows
from neural_mi.data.rows import kept_rows, rows_in_time


# ---------------------------------------------------------------------------
# Internal bias-correction helpers
# ---------------------------------------------------------------------------

def _find_linear_region(group: pd.DataFrame, curvature_t_threshold: float,
                         min_gamma_points: int) -> Tuple[List[int], bool, Dict[str, float]]:
    """Finds the linear region of the MI vs. gamma plot.

    Theory: MI_estimated(N/gamma) ≈ I_true + (a/N) * gamma, so MI is linear
    in gamma (the number of data subsets).  This function iteratively removes
    the largest gamma values (smallest data chunks, most bias) and re-fits a
    quadratic model in gamma until the quadratic term is no longer
    statistically distinguishable from zero.

    **Why a t-test on the curvature.**  The test is
    ``|a2| / SE(a2) < curvature_t_threshold``: is the quadratic coefficient
    large compared with its own uncertainty?  Comparing the curvature with the
    slope (``|a2/a1| < delta``) divides by a quantity that goes to zero exactly
    when the data are cleanest. Measured, that ratio declared linearity on
    linear data only 67% of the time, at a rate set by the amount of bias (79%
    near zero bias, 100% at strong bias) and not by how linear the relation
    was, and it accepted visibly curved data when a large slope masked the
    curvature.  Normalising by ``SE(a2)`` is scale-free, needs no knowledge of
    the MI scale, and is the standard nested-model comparison; it declares
    linearity on linear data 100% of the time and still refuses curved data.

    The dependent variable is ``train_mi`` (the training-partition MI at the
    best-generalising checkpoint).  Extrapolating to gamma → 0 gives I_true.

    Returns
    -------
    gammas_to_fit : list of int
        The retained gamma values.
    converged : bool
        True if the curvature criterion was actually satisfied on the returned
        set.  False means no linear region was found; the returned set is then
        the smallest the search is permitted to consider (``min_gamma_points``,
        or the full set when that is already smaller), and it has *not* been
        validated as linear.  The search never shrinks below
        ``min_gamma_points``, so a returned set is never smaller than the floor
        purely as an artefact of the search.
    fit_stats : dict
        The quadratic-fit quantities the decision was made on, for the returned
        set: ``curvature_coefficient`` (a2), ``curvature_se`` (SE of a2),
        ``curvature_t`` (the test statistic), and ``curvature_slope`` (a1).
        Reported so the verdict can be audited instead of taken on trust.
    """
    gammas_to_fit = sorted(group['gamma'].unique())
    converged = False
    fit_stats = {'curvature_coefficient': float('nan'), 'curvature_se': float('nan'),
                 'curvature_t': float('nan'), 'curvature_slope': float('nan')}
    while True:
        # A quadratic in gamma needs at least three distinct gamma values.
        if len(gammas_to_fit) < 3:
            break
        subset = group[group['gamma'].isin(gammas_to_fit)].copy()
        if len(subset) < 3:
            break
        weights = 1 / subset['gamma'].map(subset['gamma'].value_counts())
        X_quad = sm.add_constant(np.vstack([subset['gamma'], subset['gamma']**2]).T)
        model_quad = sm.WLS(subset['train_mi'], X_quad, weights=weights).fit()
        _, a1, a2 = model_quad.params
        # bse is a pandas Series labelled ['const', 'x1', 'x2'] when endog is a
        # Series, so bse[2] is a *label* lookup that only worked on pandas < 2.0
        # via the positional fallback.  Go through numpy to index by position on
        # every pandas version.
        se_a2 = float(np.asarray(model_quad.bse)[2])
        # Is the quadratic term large relative to its OWN uncertainty?  Not
        # relative to the slope, which vanishes on exactly the clean data this
        # check should be accepting.
        t_stat = abs(a2 / se_a2) if se_a2 > 0 else float('inf')
        fit_stats = {'curvature_coefficient': float(a2), 'curvature_se': float(se_a2),
                     'curvature_t': float(t_stat), 'curvature_slope': float(a1)}
        if t_stat < curvature_t_threshold:
            converged = True
            break
        # Stop at the floor instead of stepping past it.  Popping here would
        # return a set one shorter than the minimum, which was never tested for
        # curvature and which the caller's own length check then rejects.
        if len(gammas_to_fit) <= min_gamma_points:
            break
        gammas_to_fit.pop(-1)
    return gammas_to_fit, converged, fit_stats


def _extrapolate_mi(group: pd.DataFrame, gammas_to_fit: List[int],
                     confidence_level: float) -> tuple:
    """Extrapolates MI to infinite data limit (gamma→0, i.e. 1/N→0).

    Theory: MI ≈ I_true + (a/N) * gamma, so MI is linear in gamma.
    Fits ``train_mi = intercept + slope * gamma`` via WLS and returns
    ``(intercept, mi_error, mi_error_pred, slope)``.

    The extrapolation point is gamma=0 (i.e. the infinite-data limit),
    where ``MI = intercept = I_true``.

    Two uncertainty intervals are computed at the extrapolation point
    (``gamma = 0``):

    **Confidence interval** (``mi_error``, default reported)
        Uncertainty in the *fitted mean* at the extrapolation point.  This is
        the correct interval to report when you want to quantify how well the
        bias-corrected MI estimate is determined by the data.  It only reflects
        uncertainty in the regression coefficients.

    **Prediction interval** (``mi_error_pred``)
        Uncertainty for a *single new observation* at the extrapolation point.
        Always wider than the confidence interval because it also accounts for
        residual noise.  Useful if you want a conservative bound that would
        also cover a hypothetical individual training run at infinite data.

    The ``mi_error`` (confidence interval half-width) is returned as the
    primary uncertainty measure; ``mi_error_pred`` is provided for completeness.
    """
    final_subset = group[group['gamma'].isin(gammas_to_fit)].copy()
    if len(final_subset) < 2:
        raise InsufficientDataError("Not enough points for a reliable linear fit after pruning.")

    weights = 1 / final_subset['gamma'].map(final_subset['gamma'].value_counts())
    X_linear = sm.add_constant(final_subset['gamma'])
    fit_linear = sm.WLS(final_subset['train_mi'], X_linear, weights=weights).fit()
    intercept, slope = fit_linear.params

    # Predict at gamma=0 (infinite data → I_true)
    pred = fit_linear.get_prediction(exog=[1, 0])
    alpha = 1 - confidence_level

    # Confidence interval: uncertainty in the fitted mean at gamma = 0
    ci = pred.conf_int(obs=False, alpha=alpha)[0]
    mi_error = (ci[1] - ci[0]) / 2.0

    # Prediction interval: also accounts for residual noise (always wider)
    pi = pred.conf_int(obs=True, alpha=alpha)[0]
    mi_error_pred = (pi[1] - pi[0]) / 2.0

    return intercept, mi_error, mi_error_pred, slope


def _fit_target(param_dict: Dict[str, Any]) -> str:
    """What a message calls one fit's configuration: its grid values, or this call."""
    shown = {k: v for k, v in param_dict.items() if k != 'dummy_group'}
    return str(shown) if shown else "this call"


def _fit_label(param_dict: Dict[str, Any]) -> str:
    """The prefix of a message about one fit."""
    return f"Fit for {_fit_target(param_dict)}: "


def _unfittable_ladder(kept: pd.DataFrame, where: str, all_zero: bool = False) -> Optional[Dict[str, Any]]:
    """The fit of a ladder with fewer than two values of gamma left, or None.

    A line needs two values of gamma. With none left because every rung
    produced nothing (``all_zero``), no network learned anything at any chunk
    size and the fit reports 0. With none left otherwise, or with one value of
    gamma left, the ladder cannot be extrapolated and the fit reports NaN.
    Either way it is unreliable.
    """
    gammas = sorted(int(g) for g in kept['gamma'].unique())
    if len(gammas) >= 2:
        return None
    if not gammas and all_zero:
        warnings.warn(f"{where}every rung of the gamma ladder produced nothing. The fit "
                      f"reports 0 and is unreliable.", UserWarning, stacklevel=user_stacklevel())
        value = 0.0
    elif not gammas:
        warnings.warn(f"{where}no value of gamma is left once the rungs that produced nothing or did "
                      f"not resolve are left out. The networks resolved no dependence on enough chunks, "
                      f"either because there is none or because the chunks are too small for it. The fit "
                      f"reports NaN and is unreliable. permutation_test=True on a single estimate tests "
                      f"whether there is any.", UserWarning, stacklevel=user_stacklevel())
        value = float('nan')
    else:
        warnings.warn(f"{where}only gamma={gammas[0]} has rungs left. A line "
                      f"needs two values of gamma. The fit reports NaN and is unreliable.",
                      UserWarning, stacklevel=user_stacklevel())
        value = float('nan')
    nan = float('nan')
    return {'mi_corrected': value, 'mi_error': nan, 'mi_error_pred': nan, 'slope': nan,
            'is_reliable': False, 'linear_region_found': False, 'enough_gamma_points': False,
            'gammas_used': gammas, 'fit_quality_warning': False, 'leverage_warning': False,
            'r_squared': nan, 'max_abs_residual': nan, 'loo_intercept_shift': nan}


def _per_gamma(group: pd.DataFrame, left_out: pd.Series) -> str:
    """How many rungs at each gamma ``left_out`` marks, for a message: 'gamma 2: 1 of 2, ...'."""
    per_gamma = left_out.groupby(group['gamma']).agg(['sum', 'count'])
    return ', '.join(f"gamma {int(g)}: {int(r['sum'])} of {int(r['count'])}"
                     for g, r in per_gamma.iterrows() if r['sum'] > 0)


def _drop_rungs(group: pd.DataFrame, where: str) -> Tuple[pd.DataFrame, int, int]:
    """The ladder without the rungs the fit cannot use, and how many of them produced nothing and how
    many did not resolve.

    A rung produced nothing when it is 0, because the networks behind it learned
    nothing that generalises, or NaN, because a term of a difference failed
    (:func:`neural_mi.analysis.sweep.combine_terms`). A rung did not resolve when
    the held-out curve of its network, or of the joint term of a difference
    (:func:`neural_mi.analysis.sweep.rung`), does not rise clearly above its own
    noise (``curve_diagnosis`` is ``'not resolved'``,
    :func:`neural_mi.training.epochs.choose_epoch`). Its value then says nothing
    about the dependence, and it would enter the fit anyway: noisy rungs pass
    the curvature test easily, because noise hides curvature. A rung of a
    difference carries its signed measured value otherwise. The fit leaves both
    kinds out like failed chunks. A gamma that loses half or more of its rungs
    this way drops out of the fit with the rest of them: its chunks are too
    small for the quantity, and the rungs it keeps are the networks that
    happened to learn, which tilt the line. ``min_gamma_points`` then decides
    whether enough values of gamma remain.
    """
    zero = (group['train_mi'] == 0) | group['train_mi'].isna()
    unresolved = ~zero & (group['curve_diagnosis'] == 'not resolved'
                          if 'curve_diagnosis' in group else False)
    if zero.any():
        warnings.warn(
            f"{where}{int(zero.sum())} of {len(group)} rungs of the gamma ladder produced "
            f"nothing and are left out of the fit ({_per_gamma(group, zero)}). A rung produces "
            f"nothing when the networks behind it learned nothing that generalises (it is reported "
            f"as 0), or when a term of a difference failed on its chunk (NaN). runs counts them in "
            f"zero_rungs.",
            UserWarning, stacklevel=user_stacklevel()
        )
    if unresolved.any():
        warnings.warn(
            f"{where}{int(unresolved.sum())} of {len(group)} rungs of the gamma ladder did not "
            f"resolve and are left out of the fit ({_per_gamma(group, unresolved)}). A rung does "
            f"not resolve when the held-out curve of its network, or of the joint term of a "
            f"difference, does not rise clearly above its own noise, so its value says nothing "
            f"about the dependence. runs counts them in unresolved_rungs.",
            UserWarning, stacklevel=user_stacklevel()
        )
    left_out = zero | unresolved
    per_gamma = left_out.groupby(group['gamma']).agg(['sum', 'count'])
    half = [int(g) for g, r in per_gamma.iterrows() if r['sum'] > 0 and r['sum'] >= 0.5 * r['count']]
    if half:
        warnings.warn(
            f"{where}half or more of the rungs at gamma={half} are left out, and so are those "
            f"values of gamma. Their chunks are likely too small to estimate this quantity, and "
            f"the rungs left there would be the networks that happened to learn. The fit is "
            f"unreliable with fewer than min_gamma_points values of gamma.",
            UserWarning, stacklevel=user_stacklevel()
        )
    return group[~left_out & ~group['gamma'].isin(half)], int(zero.sum()), int(unresolved.sum())


def _compute_fit_diagnostics(group: pd.DataFrame, gammas_used: List[int],
                               residual_threshold: float = 2.5,
                               leverage_threshold: float = 0.20) -> Dict[str, Any]:
    """Computes fit diagnostics for the WLS linear extrapolation.

    Performs two checks:

    Check A, Residual quality: fits the WLS line on the final subset and
    examines externally studentized residuals.  Flags if
    ``max(|r_i|) > residual_threshold``.  R² is computed and returned for
    transparency but does **not** affect ``fit_quality_warning``: with large N
    the bias across gamma is inherently small (near-flat line) so R² collapses
    toward zero even for a sound fit, making it an unreliable gate here.
    ``fit_quality_warning`` itself is also informational only, it does **not**
    affect ``is_reliable`` upstream, because the heteroscedastic WLS structure
    (low-gamma rows dominating MSE, high-gamma rows having natural noise)
    routinely produces large studentized residuals for valid fits.

    Check B, LOO γ=1 stability: refits WLS excluding all rows where
    ``gamma == 1`` and measures the relative shift in the intercept.  Flags
    if ``|I_full - I_loo| / (|I_full| + 1e-8) > leverage_threshold``.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame with at least 'gamma' and 'train_mi' columns.
    gammas_used : list of int
        The gamma values retained after ``_find_linear_region``.
    residual_threshold : float
        Maximum allowed absolute externally studentized residual.
    leverage_threshold : float
        Maximum allowed relative shift in intercept when γ=1 is left out.

    Returns
    -------
    dict
        Keys: ``fit_quality_warning``, ``leverage_warning``, ``r_squared``,
        ``max_abs_residual``, ``loo_intercept_shift``.
    """
    _empty = {
        'fit_quality_warning': False,
        'leverage_warning': False,
        'r_squared': float('nan'),
        'max_abs_residual': float('nan'),
        'loo_intercept_shift': float('nan'),
    }

    final_subset = group[group['gamma'].isin(gammas_used)].copy()
    if len(final_subset) < 3:
        return _empty

    weights = 1 / final_subset['gamma'].map(final_subset['gamma'].value_counts())
    X_linear = sm.add_constant(final_subset['gamma'])
    fit_linear = sm.WLS(final_subset['train_mi'], X_linear, weights=weights).fit()

    # ------------------------------------------------------------------
    # Check A: residual quality
    # ------------------------------------------------------------------
    # A ladder with no spread (every chunk at the same value, typically 0) leaves
    # R-squared undefined. It is reported as NaN, which the summary skips.
    with np.errstate(divide='ignore', invalid='ignore'):
        r_squared = fit_linear.rsquared

    try:
        influence = fit_linear.get_influence()
        ext_resids = influence.resid_studentized_external
    except Exception:
        denom = np.sqrt(fit_linear.mse_resid) + 1e-12
        ext_resids = fit_linear.resid / denom

    max_abs_residual = float(np.max(np.abs(ext_resids)))
    fit_quality_warning = (max_abs_residual > residual_threshold)

    # ------------------------------------------------------------------
    # Check B: LOO γ=1 stability
    # ------------------------------------------------------------------
    leverage_warning = False
    loo_intercept_shift = float('nan')

    gamma1_mask = final_subset['gamma'] == 1
    if gamma1_mask.any():
        loo_subset = final_subset[~gamma1_mask].copy()
        if len(loo_subset) >= 2:
            loo_weights = 1 / loo_subset['gamma'].map(loo_subset['gamma'].value_counts())
            X_loo = sm.add_constant(loo_subset['gamma'])
            fit_loo = sm.WLS(loo_subset['train_mi'], X_loo, weights=loo_weights).fit()

            I_full = fit_linear.params.iloc[0]   # intercept ('const')
            I_loo = fit_loo.params.iloc[0]
            delta_loo = abs(I_full - I_loo) / (abs(I_full) + 1e-8)
            loo_intercept_shift = float(delta_loo)
            leverage_warning = delta_loo > leverage_threshold
        # else: fewer than 2 points after removing γ=1, skip silently

    return {
        'fit_quality_warning': bool(fit_quality_warning),
        'leverage_warning': bool(leverage_warning),
        'r_squared': float(r_squared),
        'max_abs_residual': max_abs_residual,
        'loo_intercept_shift': loo_intercept_shift,
    }


SATURATION_WARNING_THRESHOLD = 0.85  # matches trainer.py's CEILING_PROXIMITY_WARNING_THRESHOLD


def _compute_per_gamma_diagnostics(group: pd.DataFrame, gammas_used: List[int]) -> Dict[str, Any]:
    """Per-gamma diagnostics the extrapolation currently has no visibility into.

    Both are blind spots for the same underlying reason: ``_find_linear_region``
    only looks at curvature (|a2/a1|) in gamma, and both non-stationarity and
    ceiling saturation are approximately *smooth* in gamma. They inflate or
    suppress every rung's train_mi by a similar amount without bending the
    fitted line, so a real problem can sail through that check untouched.

    Spread : the std of train_mi across the ``gamma`` independently-trained
        chunks at each gamma level. With contiguous (temporal) chunking,
        large spread at high gamma means different segments of the recording
        give different answers. That is non-stationarity showing up for free, and
        the only diagnostic the library has for it. gamma=1 has a single chunk, so its spread is always 0.0/NaN
        (not a real signal, just nothing to compare).

    Ceiling / saturation : each chunk has its own train_eval_size and
        therefore its own ceiling; the deepest gammas have the smallest
        chunks (lowest ceilings) while most influencing the fitted slope.
        Warns by name when the gammas actually used in the fit are, on
        average, saturated.
    """
    out: Dict[str, Any] = {
        'per_gamma_train_mi_spread': {},
        'per_gamma_ceiling_mi': {},
        'per_gamma_saturation': {},
    }
    has_ceiling_cols = 'train_ceiling_mi' in group.columns and 'train_saturation' in group.columns
    saturated_gammas = []
    for gamma, sub in group.groupby('gamma'):
        g = int(gamma)
        vals = sub['train_mi'].dropna()
        out['per_gamma_train_mi_spread'][g] = float(vals.std()) if len(vals) > 1 else 0.0
        if has_ceiling_cols:
            ceil_vals = sub['train_ceiling_mi'].dropna()
            sat_vals = sub['train_saturation'].dropna()
            mean_ceil = float(ceil_vals.mean()) if len(ceil_vals) else float('nan')
            mean_sat = float(sat_vals.mean()) if len(sat_vals) else float('nan')
            out['per_gamma_ceiling_mi'][g] = mean_ceil
            out['per_gamma_saturation'][g] = mean_sat
            if g in gammas_used and mean_sat == mean_sat and mean_sat > SATURATION_WARNING_THRESHOLD:
                saturated_gammas.append(g)
    if saturated_gammas:
        warnings.warn(
            f"Rigorous fit uses the saturated gamma={sorted(saturated_gammas)} "
            f"(mean train_saturation above {SATURATION_WARNING_THRESHOLD:.0%} "
            f"of their own ceiling). An extrapolation anchored on saturated "
            f"rungs cannot be trusted. The fitted slope reflects the ceiling "
            f"as much as the true bias. Raise max_eval_samples or narrow "
            f"gamma_range to leave these out.",
            UserWarning, stacklevel=user_stacklevel()
        )
        out['saturated_gammas'] = sorted(saturated_gammas)
    else:
        out['saturated_gammas'] = []
    return out


def _post_process_and_correct(df: pd.DataFrame, sweep_grid: Dict[str, Any],
                               curvature_t_threshold: float, min_gamma_points: int,
                               confidence_level: float,
                               residual_threshold: float = 2.5,
                               leverage_threshold: float = 0.20,
                               chunking_mode: str = 'permuted',
                               n_tasks_created: int = 0) -> List[Dict[str, Any]]:
    """Groups results and performs bias correction for each group."""
    valid_df = df.dropna(subset=['gamma', 'train_mi'])
    if valid_df.empty:
        raise TrainingError("Rigorous analysis failed: all training runs produced NaN MI values.")

    group_keys = list(sweep_grid.keys()) if sweep_grid else []

    corrected_results = []

    # If there are no sweep parameters, group the whole dataframe as one.
    if not group_keys:
        group_keys.append('dummy_group')
        valid_df = valid_df.copy()
        valid_df['dummy_group'] = 0

    # Group on a scalar when there is exactly one key. Passing a length-1 list
    # yields a scalar key today and a length-1 tuple in a future pandas, and
    # warns about the change on every call; the scalar form is stable across
    # both. The key is normalised to a tuple either way, so param_dict is
    # formed identically for single and multiple keys.
    grouper = group_keys[0] if len(group_keys) == 1 else group_keys

    for params, group in valid_df.groupby(grouper):
        param_values = params if isinstance(params, tuple) else (params,)
        param_dict = dict(zip(group_keys, param_values))
        n_rungs = len(group)
        group, zero_rungs, unresolved_rungs = _drop_rungs(group, _fit_label(param_dict))
        unfittable = _unfittable_ladder(group, _fit_label(param_dict), zero_rungs == n_rungs)
        if unfittable is not None:
            param_dict.update(unfittable, chunking_mode=chunking_mode,
                              n_tasks_created=n_tasks_created, zero_rungs=zero_rungs,
                              unresolved_rungs=unresolved_rungs)
            param_dict.pop('dummy_group', None)
            corrected_results.append(param_dict)
            continue

        try:
            gammas_used, linear_region_found, curvature_stats = _find_linear_region(
                group, curvature_t_threshold, min_gamma_points)
            enough_gamma_points = len(gammas_used) >= min_gamma_points
            is_reliable = enough_gamma_points and linear_region_found
            if not enough_gamma_points:
                warnings.warn(f"Fit for {_fit_target(param_dict)} is unreliable "
                              f"(final gamma points < {min_gamma_points}).",
                              UserWarning, stacklevel=user_stacklevel())
            elif not linear_region_found:
                warnings.warn(
                    f"Fit for {_fit_target(param_dict)} is unreliable. No linear region was found down to "
                    f"min_gamma_points={min_gamma_points}. The fit uses gamma={gammas_used} "
                    f"without having satisfied the curvature criterion.",
                    UserWarning, stacklevel=user_stacklevel()
                )

            mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
                group, gammas_used, confidence_level
            )

            diagnostics = _compute_fit_diagnostics(
                group, gammas_used, residual_threshold, leverage_threshold
            )
            per_gamma_diagnostics = _compute_per_gamma_diagnostics(group, gammas_used)

            if diagnostics['leverage_warning']:
                is_reliable = False
                warnings.warn(
                    f"Fit diagnostics triggered for {_fit_target(param_dict)}: "
                    f"leverage_warning={diagnostics['leverage_warning']}.",
                    UserWarning, stacklevel=user_stacklevel()
                )
            # fit_quality_warning is informational only; does not affect is_reliable
            if diagnostics['fit_quality_warning']:
                logger.debug(
                    f"fit_quality_warning=True for {param_dict} "
                    f"(large studentized residuals from heteroscedastic WLS noise, "
                    f"informational only)."
                )
            # Saturation is smooth in gamma -- the same blind spot
            # _find_linear_region's curvature check and leverage_warning both
            # have, confirmed empirically (w=50 acceptance run: 8/10 gammas
            # saturated, curvature and leverage checks both passed anyway).
            # Gate is_reliable the same way leverage_warning already does,
            # instead of leaving this purely informational -- a fit anchored
            # on saturated rungs is exactly the kind of unreliable this flag
            # exists to catch.
            if per_gamma_diagnostics['saturated_gammas']:
                is_reliable = False
                warnings.warn(
                    f"Fit for {_fit_target(param_dict)} marked unreliable: gamma="
                    f"{per_gamma_diagnostics['saturated_gammas']} used in the fit are ceiling-saturated.",
                    UserWarning, stacklevel=user_stacklevel()
                )

            param_dict.update({
                'mi_corrected': mi_corrected,
                'mi_error': mi_error,
                'mi_error_pred': mi_error_pred,
                'slope': slope,
                'is_reliable': is_reliable,
                'linear_region_found': linear_region_found,
                'enough_gamma_points': enough_gamma_points,
                **curvature_stats,
                'gammas_used': gammas_used,
                'chunking_mode': chunking_mode,
                'n_tasks_created': n_tasks_created,
                'zero_rungs': zero_rungs,
                'unresolved_rungs': unresolved_rungs,
            })
            param_dict.update(diagnostics)
            param_dict.update(per_gamma_diagnostics)
            param_dict.pop('dummy_group', None)
            corrected_results.append(param_dict)
        except InsufficientDataError as e:
            logger.error(f"Could not perform extrapolation for params {_fit_target(param_dict)}: {e}")

    return corrected_results


# ---------------------------------------------------------------------------
# AnalysisWorkflow class
# ---------------------------------------------------------------------------

class _Chunker:
    """The data of a rigorous call cut into the chunks of its gamma ladder, the one way both rigorous
    paths cut them (``AnalysisWorkflow._prepare_tasks`` and :func:`run_rigorous_scalar_analysis`).

    At each gamma one ordering of the rows is split into gamma equal chunks, so
    the chunks at gamma=2 are halves of the data at gamma=1, and every gamma
    sees the same data at a different size. Rows in time order are cut into
    contiguous stretches: a blocked split inside a chunk takes contiguous blocks
    of the order it receives, and a chunk of shuffled rows would make those
    blocks random in time, the leak the blocked split is there to prevent. Rows
    in time order are those whose builder recorded their geometry
    (:mod:`neural_mi.data.rows`), and the rows of a plan that each chunk's task
    builds itself. Other rows are cut in a random order, which keeps each chunk
    representative of the whole when rows are independent.
    ``temporal_chunking`` overrides the choice for rows that exist already.
    Contiguous chunks trade that representativeness for time order: on a
    process that drifts, chunks at high gamma sample different stretches, which
    ``per_gamma_train_mi_spread`` shows.

    Raw streams that each chunk's task windows come as their row plan
    (:class:`~neural_mi.data.plan.RowPlan`), the quantity's or, for raw X and Y
    with a processor, the pair's. A chunk is then the plan of its rows
    (:meth:`~neural_mi.data.plan.RowPlan.cut`), whose rows every task of the
    chunk reads.
    """

    def __init__(self, x_data, y_data, params: Dict[str, Any], *, temporal_chunking: Optional[bool] = None):
        if isinstance(x_data, RowPlan):
            self.plan = x_data
        elif params.get('processor_type_x') is not None and getattr(x_data, 'ndim', None) != 3:
            self.plan = RowPlan.of_pair(x_data, y_data, params)
        else:
            self.plan = None
        raw = self.plan is not None and self.plan.route != 'rows'
        if self.plan is not None:
            self.n = len(self.plan)
            in_time = raw or self.plan.geometry() is not None or rows_in_time(params)
        else:
            self.n = x_data.shape[0] if hasattr(x_data, 'shape') else len(x_data)
            in_time = rows_in_time(params)
        if raw and temporal_chunking is False:
            raise ValueError(
                "Rigorous(temporal_chunking=False) cannot apply here: each chunk's task windows a raw "
                "recording for itself, and a chunk of a recording is a stretch of it. Leave "
                "temporal_chunking unset."
            )
        self.contiguous = in_time if temporal_chunking is None else bool(temporal_chunking)
        self.mode = 'contiguous' if self.contiguous else 'permuted'
        self.reorder()

    def reorder(self) -> None:
        """Draw the ordering the chunks are cut from: time order, or a new random permutation."""
        self.order = np.arange(self.n) if self.contiguous else np.random.permutation(self.n)

    def chunks(self, gamma: int) -> List[np.ndarray]:
        """The row indices of each chunk at ``gamma``."""
        return np.array_split(self.order, gamma)

    def cut(self, chunk: np.ndarray, arrays: Dict[str, Any]) -> Dict[str, Any]:
        """``arrays`` cut to one chunk: ``'x'``, ``'y'`` and any extra array by name, or for a plan
        ``{'x': the chunk's plan, 'y': None}``."""
        if self.plan is not None:
            return {'x': self.plan.cut(chunk), 'y': None}
        return {name: _ensure_cpu(data[chunk]) for name, data in arrays.items()}


def _warn_if_chunks_small(gamma: int, chunks: List[np.ndarray], params: Dict[str, Any]) -> None:
    """Warn when the smallest chunk at ``gamma`` cannot fill one evaluation batch with its held-out rows.

    Not a gate. The line is where a chunk's held-out partition no longer fills
    a single evaluation batch, below which a chunk's value stops measuring
    anything. ``Training(min_reliable_samples=...)`` moves it. A nonparametric
    estimator holds nothing out, so the line does not apply to it.
    """
    if params.get('estimator_name') in NONPARAMETRIC:
        return
    smallest = min(kept_rows(len(c), params) for c in chunks)
    line = params.get('min_reliable_samples')
    if line is None:
        held_out = max(1.0 - float(params.get('train_fraction', 0.9)), 1e-6)
        line = int(np.ceil(params.get('batch_size', 128) / held_out))
    if smallest < line:
        warnings.warn(
            f"gamma={gamma}: the smallest data subset has {smallest} "
            f"samples. Below about {line} the held-out partition "
            f"of a chunk cannot fill one evaluation batch "
            f"(batch_size={params.get('batch_size', 128)}, "
            f"train_fraction={params.get('train_fraction', 0.9)}). "
            f"MI values from chunks this small are dominated by noise. The "
            f"fit leaves out the rungs whose held-out curves do not resolve. "
            f"The others can still be noisy, and a straight line through "
            f"noisy rungs still looks straight, so the linearity check cannot "
            f"catch it. Use a smaller gamma_range or more data. Set "
            f"Training(min_reliable_samples=...) to move this line.",
            UserWarning, stacklevel=user_stacklevel()
        )


class AnalysisWorkflow:
    """Orchestrates the rigorous, multi-step analysis for bias correction."""

    def __init__(self, x_data, y_data, base_params, **kwargs):
        """
        Parameters
        ----------
        x_data : torch.Tensor
            Preprocessed data for variable X.
        y_data : torch.Tensor
            Preprocessed data for variable Y.
        base_params : Dict[str, Any]
            A dictionary of fixed parameters for the MI estimator's trainer.
        **kwargs : Dict[str, Any]
            Additional keyword arguments to be added to ``base_params``.
        """
        self.x_data, self.y_data = x_data, y_data
        # Copy so callers of run_rigorous_analysis() (or AnalysisWorkflow
        # directly) never see their base_params dict mutated in place.
        self.base_params = dict(base_params)
        # Raw (2-D) data windowed inside each task has no per-window shape
        # yet, so task.py infers the dimensions once a chunk is windowed. A
        # spike list has no .shape at all and is left to the task as well.
        if hasattr(x_data, 'shape') and x_data.ndim != 2:
            self.base_params.update({
                'input_dim_x': int(np.prod(x_data.shape[1:])),
                'input_dim_y': int(np.prod(y_data.shape[1:])),
                'n_channels_x': x_data.shape[1],
                'n_channels_y': y_data.shape[1],
            })
        self.base_params.update(kwargs)

    def run(self, param_grid: Optional[Dict[str, List]] = None,
            gamma_range=range(1, 11),
            n_workers: Optional[int] = None,
            temporal_chunking: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """Executes the full rigorous analysis workflow.

        This involves preparing tasks for different data subsets (controlled by
        *gamma*), running them in parallel, and then applying a post-processing
        and bias correction step to the aggregated results.

        Parameters
        ----------
        param_grid : Dict[str, List], optional
            A grid of hyperparameters to sweep over in addition to the gamma sweep.
        gamma_range : range, optional
            The range of gamma values to use for data subsampling.
            Defaults to ``range(1, 11)``.
        n_workers : int, optional
            The number of worker processes to use. Defaults to 1.
        temporal_chunking : bool, optional
            Contiguous (``True``) or permuted (``False``) chunks, overriding
            the choice :class:`_Chunker` makes from the rows' recorded
            geometry. ``None`` (default) leaves the choice to it.
        **kwargs : Dict[str, Any]
            Additional keyword arguments for the bias correction, such as
            ``curvature_t_threshold``, ``min_gamma_points``, ``confidence_level``,
            ``residual_threshold`` and ``leverage_threshold``.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:

            - ``'corrected_results'`` : list of per-group correction dicts.
              Each includes ``'chunking_mode'`` (``'contiguous'`` or
              ``'permuted'``, the resolved choice above),
              ``'n_tasks_created'``, ``'per_gamma_train_mi_spread'``,
              ``'per_gamma_ceiling_mi'``, and ``'per_gamma_saturation'``.
            - ``'raw_results_df'`` (pd.DataFrame): raw sweep results with one
              row per training run.  Key columns: ``gamma``, ``train_mi``.
        """
        n_workers = n_workers if n_workers is not None else 1
        show_progress = self.base_params.get('show_progress', True)
        logger.info(f"Starting rigorous analysis with {n_workers} workers...")
        tasks = with_device(self._prepare_tasks(param_grid, gamma_range, temporal_chunking=temporal_chunking),
                            n_workers)
        if not tasks:
            return {"corrected_results": [], "raw_results_df": pd.DataFrame()}

        # gamma_range=range(1, 11) makes sum(1..10) = 55 tasks per
        # configuration. Checked, since a truncated ladder would invalidate
        # every extrapolation.
        expected_per_combo = sum(gamma_range)
        n_combos = max(1, len(param_grid or {}) and len(list(itertools.product(
            *(param_grid or {}).values()))) or 1)
        expected_total = expected_per_combo * n_combos
        if len(tasks) != expected_total:
            warnings.warn(
                f"Rigorous analysis expected {expected_total} tasks "
                f"({n_combos} combo(s) x sum(gamma_range)={expected_per_combo}) "
                f"and created {len(tasks)}. The gamma ladder may be truncated. "
                f"Check it before trusting the extrapolation.",
                UserWarning, stacklevel=user_stacklevel()
            )

        if n_workers <= 1:
            logger.info("Running rigorous analysis sequentially (n_workers=1)...")
            raw_results = [
                run_training_task(task)
                for task in tqdm(tasks, desc="Rigorous analysis",
                                 unit="task", disable=not show_progress)
            ]
        else:
            with worker_pool(n_workers) as pool:
                raw_results = list(tqdm(
                    released(pool.imap(CapturedTask(run_training_task), tasks)), total=len(tasks),
                    desc="Rigorous analysis", unit="task", disable=not show_progress
                ))

        logger.info("All training tasks finished. Performing bias correction...")
        raw_results_df = pd.DataFrame(raw_results)

        correction_kwargs = {
            'sweep_grid': param_grid,
            'curvature_t_threshold': kwargs.pop('curvature_t_threshold', 2.0),
            'min_gamma_points': kwargs.pop('min_gamma_points', 5),
            'confidence_level': kwargs.pop('confidence_level', 0.68),
            'residual_threshold': kwargs.pop('residual_threshold', 2.5),
            'leverage_threshold': kwargs.pop('leverage_threshold', 0.20),
        }

        corrected_results = _post_process_and_correct(
            raw_results_df, chunking_mode=self.resolved_chunking_mode,
            n_tasks_created=len(tasks), **correction_kwargs)
        return {"corrected_results": corrected_results, "raw_results_df": raw_results_df}

    def _prepare_tasks(self, param_grid: Optional[Dict[str, List]], gamma_range,
                       temporal_chunking: Optional[bool] = None) -> List[tuple]:
        """One training task per chunk of the gamma ladder, for every configuration of ``param_grid``.

        :class:`_Chunker` cuts the chunks, as it does for
        :func:`run_rigorous_scalar_analysis`: contiguous for rows in time order,
        permuted otherwise, and ``temporal_chunking`` overrides that. Each
        configuration draws its own ordering.
        """
        tasks = []
        run_id_base = str(uuid.uuid4())
        param_grid = param_grid or {}
        if self.base_params.get('critic_type') == 'concat' and 'embedding_dim' in param_grid:
            param_grid.pop('embedding_dim')

        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if param_grid else [{}]

        # Raw data reach this path only when each chunk's task windows them,
        # whose rows move under shift_windows. run.py windows everything else
        # once.
        chunker = _Chunker(self.x_data, self.y_data, self.base_params,
                           temporal_chunking=temporal_chunking)
        self.resolved_chunking_mode = chunker.mode

        for i_combo, params in enumerate(param_combinations):
            current_params = {**self.base_params, **params}
            chunker.reorder()
            for gamma in gamma_range:
                current_params['gamma'] = gamma
                chunks = chunker.chunks(gamma)
                _warn_if_chunks_small(gamma, chunks, current_params)
                for i_subset, subset_indices in enumerate(chunks):
                    cut = chunker.cut(subset_indices, {'x': self.x_data, 'y': self.y_data})
                    if isinstance(cut['x'], RowPlan):
                        cut = {'x': TermRows(cut['x'], PAIR), 'y': None}
                    chunk_params = current_params
                    task_run_id = f"{run_id_base}_c{i_combo}_g{gamma}_s{i_subset}"
                    # Purely deterministic per-task key for run_training_task's seeding
                    # (see task.py) -- unlike task_run_id above, excludes the random
                    # run_id_base prefix so a fixed random_seed reproduces the same
                    # task_seed (and result) on every call.
                    chunk_params = {**chunk_params, '_seed_key': f"c{i_combo}_g{gamma}_s{i_subset}",
                                    'chunk': i_subset}
                    task_params = with_model_labels(chunk_params, **params, gamma=gamma, chunk=i_subset)
                    tasks.append((cut['x'], cut['y'], task_params.copy(), task_run_id))

        self.n_tasks_created = len(tasks)
        logger.info(
            f"Rigorous analysis: created {len(tasks)} training tasks "
            f"({len(param_combinations)} param combo(s) x sum(gamma_range)={sum(gamma_range)} "
            f"chunks each, chunking={self.resolved_chunking_mode})."
        )
        return tasks


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_rigorous_analysis(
    x_data,
    y_data,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, List]] = None,
    gamma_range=range(1, 11),
    n_workers: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Estimate MI via rigorous, bias-corrected finite-data extrapolation.

    Trains models on subsets of decreasing size (parameterised by *gamma*)
    and extrapolates the relationship to the infinite-data limit.  Each
    per-run MI estimate is ``train_mi`` (training-partition MI at the
    best-generalising checkpoint), consistent with every other mode.

    Parameters
    ----------
    x_data : torch.Tensor
        Preprocessed data for variable X, shape ``(n_samples, n_channels, window_size)``.
    y_data : torch.Tensor
        Preprocessed data for variable Y, same leading dimension as *x_data*.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid (e.g. ``{'run_id': range(5)}``).
    gamma_range : range or sequence of int, optional
        Values of *gamma* (data-fraction denominators) to sweep over.
        Defaults to ``range(1, 11)``.
    n_workers : int or None, optional
        Number of parallel workers. ``None`` uses a single process.
    **kwargs
        Additional keyword arguments forwarded to ``AnalysisWorkflow.run()``.
        Common ones: ``curvature_t_threshold``, ``min_gamma_points``, ``confidence_level``,
        ``residual_threshold``, ``leverage_threshold``.

    Returns
    -------
    Dict[str, Any]
        Same dictionary returned by ``AnalysisWorkflow.run()``.  Key entries:

        - ``'corrected_results'`` : list of per-group correction dicts.
        - ``'raw_results_df'`` (pd.DataFrame): raw sweep results.
    """
    workflow = AnalysisWorkflow(x_data, y_data, base_params)
    return workflow.run(
        param_grid=sweep_grid or {},
        gamma_range=gamma_range,
        n_workers=n_workers,
        **kwargs,
    )


def _run_scalar_fn_task(args: tuple) -> Dict[str, Any]:
    """Top-level, picklable wrapper for one gamma-chunk ``scalar_fn`` call.

    Must be module-level (not a closure) so it, and the ``scalar_fn`` it
    carries, can be pickled for a ``multiprocessing`` 'spawn' pool.  Catches
    its own exceptions instead of letting them propagate through
    ``pool.imap``, which would otherwise abort every remaining task instead
    of just skipping the failed gamma-chunk (matching the sequential path's
    per-task try/except behaviour).
    """
    scalar_fn, x_sub, y_sub, params, extra_sub, extra_kwargs, gamma, chunk, chunk_size = args
    params = with_model_labels(params, gamma=gamma, chunk=chunk)
    try:
        # A chunk's combined value is one point on the ladder, never the
        # reported result, so the checks on a combined quantity stay quiet here.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', CombinationWarning)
            value = scalar_fn(x_sub, y_sub, params, **extra_sub, **(extra_kwargs or {}))
        # A scalar_fn returns its value, or a rung (sweep.rung): the value with the
        # paths of the networks it saved and the diagnosis of its joint term's curve.
        rung = value if isinstance(value, dict) else {'train_mi': value}
        return {'gamma': gamma, 'chunk': chunk, 'train_mi': rung['train_mi'], '_error': None,
                'model_path': rung.get('model_path'), 'curve_diagnosis': rung.get('curve_diagnosis')}
    except Exception as exc:
        return {'gamma': gamma, 'chunk': chunk, 'train_mi': None,
                '_error': f"gamma={gamma} (chunk size={chunk_size}): {exc}"}


def run_rigorous_scalar_analysis(
    scalar_fn,
    x_data,
    y_data,
    base_params: Dict[str, Any],
    extra_data: Optional[Dict[str, Any]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    gamma_range=range(1, 11),
    n_workers: Optional[int] = None,
    curvature_t_threshold: float = 2.0,
    min_gamma_points: int = 5,
    confidence_level: float = 0.68,
    residual_threshold: float = 2.5,
    leverage_threshold: float = 0.20,
    temporal_chunking: Optional[bool] = None,
) -> Dict[str, Any]:
    """Bias-corrected estimation of a compound scalar quantity via rigorous extrapolation.

    Used by conditional and transfer MI modes to apply the same finite-data
    bias correction to arbitrary scalar functions of the data (e.g. conditional
    MI, transfer entropy) without requiring a full ``AnalysisWorkflow`` training
    loop.

    The function calls ``scalar_fn`` on progressively smaller subsets of the
    data (controlled by *gamma*), collects the scalar outputs, and extrapolates
    to the infinite-data limit using the same WLS linear fit as the standard
    rigorous analysis.  The gamma-chunk calls are independent of one another
    and are dispatched to a ``multiprocessing`` pool of ``n_workers`` workers
    when more than one is requested, exactly like ``AnalysisWorkflow.run()``
    does for plain ``mode='rigorous'``.

    Parameters
    ----------
    scalar_fn : callable
        A **module-level, picklable** function with signature
        ``scalar_fn(x_sub, y_sub, params, **extra_sub, **extra_kwargs)``.
        It returns a single scalar MI (or MI-like) value, or a rung
        (:func:`neural_mi.analysis.sweep.rung`): a dict with the value under
        ``'train_mi'``, the saved networks under ``'model_path'`` and the
        diagnosis of the joint term's held-out curve under
        ``'curve_diagnosis'``. A rung whose diagnosis is ``'not resolved'`` is
        left out of the fit. It cannot be a closure or lambda when
        ``n_workers > 1``, since it is sent to worker processes by reference.
    x_data : array-like, shape (N, ...)
        Data for variable X.  The first axis is the sample axis.
    y_data : array-like, shape (N, ...)
        Data for variable Y.  Same leading dimension as *x_data*.
    base_params : Dict[str, Any]
        Fixed parameters passed to ``scalar_fn`` as the ``params`` positional
        argument.  A shallow copy is made for each call.
    extra_data : dict of array-like, optional
        Additional arrays (keyed by name) to subsample alongside *x_data* and
        *y_data*.  Each array must have the same leading dimension N.  The
        subsampled arrays are passed as keyword arguments to ``scalar_fn``.
    extra_kwargs : dict, optional
        Fixed keyword arguments forwarded verbatim to every ``scalar_fn`` call
        (not subsampled).
    gamma_range : range or sequence of int, optional
        Values of *gamma* to sweep over.  Defaults to ``range(1, 11)``.
    n_workers : int or None, optional
        Number of parallel worker processes for the gamma-chunk tasks.
        ``None`` or ``<= 1`` runs sequentially.  Defaults to ``None``.
    curvature_t_threshold : float, optional
        Curvature t-statistic threshold for ``_find_linear_region``: the
        linear region ends where ``|a2| / SE(a2)`` first falls below this.
        Defaults to ``2.0`` (roughly the 5% two-sided significance level).
    min_gamma_points : int, optional
        Minimum number of distinct gamma values required for a reliable fit.
        Defaults to ``5``.
    confidence_level : float, optional
        Confidence level for the extrapolation error interval.  Defaults to
        ``0.68`` (roughly ±1 σ).
    residual_threshold : float, optional
        Passed to ``_compute_fit_diagnostics``.  Defaults to ``2.5``.
    leverage_threshold : float, optional
        Passed to ``_compute_fit_diagnostics``.  Defaults to ``0.20``.
    temporal_chunking : bool, optional
        Contiguous (``True``) or permuted (``False``) chunks, overriding the
        choice :class:`_Chunker` makes from the rows' recorded geometry.
    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:

        - ``'mi_corrected'`` (float): bias-corrected scalar estimate.
        - ``'mi_error'`` (float): half-width of the confidence interval.
        - ``'slope'`` (float): slope of the WLS fit (bias per unit gamma).
        - ``'is_reliable'`` (bool): True when all of: a linear region was
          actually found (``linear_region_found``), enough gamma points were
          retained (``enough_gamma_points``), ``leverage_warning`` is False, and
          no gamma used in the fit is ceiling-saturated.
          ``fit_quality_warning`` does **not** affect this flag (see below).
        - ``'linear_region_found'`` (bool): whether the curvature criterion was
          satisfied on the gammas that were used.  False means the search
          reached ``min_gamma_points`` without the fit ever looking linear, so
          the extrapolation is being done on a set that failed the check.  This
          is reported separately from ``enough_gamma_points`` so the two
          distinct failure modes can be told apart.
        - ``'enough_gamma_points'`` (bool): whether
          ``len(gammas_used) >= min_gamma_points``.
        - ``'gammas_used'`` (list of int): gamma values in the linear region.
        - ``'raw_results_df'`` (pd.DataFrame): one row per successful chunk call.
        - ``'fit_quality_warning'`` (bool): informational only; large studentized
          residuals arising from heteroscedastic WLS noise.  Does **not** affect
          ``is_reliable``.
        - ``'leverage_warning'`` : bool
        - ``'r_squared'`` : float
        - ``'max_abs_residual'`` : float
        - ``'loo_intercept_shift'`` : float

    Raises
    ------
    InsufficientDataError
        If fewer than ``min_gamma_points`` rows are collected across all gamma
        values (i.e. almost every ``scalar_fn`` call failed).
    """
    # A plan (conditional MI, interaction information) is cut into the plans of
    # its chunks. mode='transfer' passes raw 2-D series, which its histories
    # are built from by offset and which no processor windows.
    chunker = _Chunker(x_data, y_data, base_params, temporal_chunking=temporal_chunking)
    resolved_chunking_mode = chunker.mode

    tasks = []
    for gamma in gamma_range:
        chunks = chunker.chunks(gamma)
        _warn_if_chunks_small(gamma, chunks, base_params)
        for chunk, chunk_idx in enumerate(chunks):
            cut = chunker.cut(chunk_idx, {'x': x_data, 'y': y_data, **(extra_data or {})})
            extra_sub = {key: cut[key] for key in (extra_data or {}) if key in cut}
            tasks.append((scalar_fn, cut['x'], cut['y'], dict(base_params),
                         extra_sub, extra_kwargs, gamma, chunk, len(chunk_idx)))

    show_progress = base_params.get('show_progress', True)
    effective_workers = n_workers if n_workers is not None else 1

    if effective_workers <= 1 or len(tasks) <= 1:
        raw_rows = [
            _run_scalar_fn_task(task)
            for task in tqdm(tasks, desc="Rigorous scalar analysis", unit="task",
                             disable=not show_progress)
        ]
    else:
        logger.info(
            f"Parallelising {len(tasks)} rigorous scalar-analysis tasks across "
            f"{effective_workers} workers..."
        )
        with worker_pool(effective_workers) as pool:
            raw_rows = list(tqdm(
                released(pool.imap(CapturedTask(_run_scalar_fn_task), tasks)), total=len(tasks),
                desc="Rigorous scalar analysis", unit="task", disable=not show_progress
            ))

    rows = []
    for r in raw_rows:
        if r['_error'] is not None:
            warnings.warn(f"Rigorous fit (rigorous=True): the estimate for one chunk failed: {r['_error']}",
                          UserWarning, stacklevel=user_stacklevel())
        else:
            row = {'gamma': r['gamma'], 'chunk': r['chunk'], 'train_mi': r['train_mi']}
            for key in ('model_path', 'curve_diagnosis'):
                if r.get(key):
                    row[key] = r[key]
            rows.append(row)

    if len(rows) < min_gamma_points:
        raise InsufficientDataError(
            f"The rigorous fit collected only {len(rows)} successful chunk "
            f"estimates. It needs min_gamma_points={min_gamma_points} to "
            f"extrapolate."
        )

    ladder = pd.DataFrame(rows, columns=['gamma', 'chunk', 'train_mi']
                          + [key for key in ('model_path', 'curve_diagnosis')
                             if any(key in r for r in rows)])
    # The fit uses the rungs that produced a value and resolved. The ladder
    # reported in raw_results_df keeps every rung.
    df, zero_rungs, unresolved_rungs = _drop_rungs(ladder, "Rigorous fit (rigorous=True): ")
    unfittable = _unfittable_ladder(df, "Rigorous fit (rigorous=True): ", zero_rungs == len(ladder))
    if unfittable is not None:
        return {**unfittable, 'raw_results_df': ladder, 'chunking_mode': resolved_chunking_mode,
                'n_tasks_created': len(tasks), 'zero_rungs': zero_rungs,
                'unresolved_rungs': unresolved_rungs, 'per_gamma_train_mi_spread': {}}

    gammas_used, linear_region_found, curvature_stats = _find_linear_region(
        df, curvature_t_threshold, min_gamma_points)
    try:
        mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
            df, gammas_used, confidence_level
        )
    except InsufficientDataError:
        # Pruning left too few points, fall back to all available gammas and mark
        # the result as unreliable so callers are warned.
        gammas_used = sorted(df['gamma'].unique().tolist())
        linear_region_found = False
        warnings.warn(
            f"Rigorous fit (rigorous=True): the linear region is too small after "
            f"pruning. The fit falls back to all {len(gammas_used)} gamma values and is_reliable is False.",
            UserWarning, stacklevel=user_stacklevel()
        )
        mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
            df, gammas_used, confidence_level
        )
    # Note: diagnostics uses the same gamma-based regression as _extrapolate_mi
    diagnostics = _compute_fit_diagnostics(df, gammas_used, residual_threshold, leverage_threshold)

    enough_gamma_points = len(gammas_used) >= min_gamma_points
    is_reliable = enough_gamma_points and linear_region_found
    if enough_gamma_points and not linear_region_found:
        warnings.warn(
            f"Rigorous fit (rigorous=True): no linear region was found down to "
            f"min_gamma_points={min_gamma_points}. The fit uses gamma={gammas_used} without having "
            f"satisfied the curvature criterion.",
            UserWarning, stacklevel=user_stacklevel()
        )
    if diagnostics['leverage_warning']:
        is_reliable = False
        warnings.warn(
            f"Rigorous fit (rigorous=True): fit diagnostics triggered: "
            f"leverage_warning={diagnostics['leverage_warning']}.",
            UserWarning, stacklevel=user_stacklevel()
        )
    # fit_quality_warning is informational only; does not affect is_reliable
    if diagnostics['fit_quality_warning']:
        logger.debug(
            "run_rigorous_scalar_analysis: fit_quality_warning=True "
            "(large studentized residuals from heteroscedastic WLS noise, "
            "informational only)."
        )

    # The spread across chunks is reported as in the main workflow. The
    # per-gamma ceiling and saturation are not: scalar_fn returns one float
    # per chunk, without the per-network diagnostics they are computed from.
    per_gamma_spread = {
        int(gamma): (float(sub['train_mi'].std()) if len(sub) > 1 else 0.0)
        for gamma, sub in df.groupby('gamma')
    }

    return {
        'mi_corrected': mi_corrected,
        'mi_error': mi_error,
        'mi_error_pred': mi_error_pred,
        'slope': slope,
        'is_reliable': is_reliable,
        'linear_region_found': linear_region_found,
        'enough_gamma_points': enough_gamma_points,
        **curvature_stats,
        'gammas_used': gammas_used,
        'raw_results_df': ladder,
        'chunking_mode': resolved_chunking_mode,
        'n_tasks_created': len(tasks),
        'per_gamma_train_mi_spread': per_gamma_spread,
        'zero_rungs': zero_rungs,
        'unresolved_rungs': unresolved_rungs,
        **diagnostics,
    }
