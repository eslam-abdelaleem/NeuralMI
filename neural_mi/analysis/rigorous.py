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
import numpy as np
import pandas as pd
import itertools
import uuid
import torch.multiprocessing as mp
import statsmodels.api as sm
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger, worker_init_args
from neural_mi.exceptions import InsufficientDataError, TrainingError
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu
from neural_mi.data.shift_windowing import (
    n_windows_if_deferred, shift_family, chunk_window_range_to_raw, seconds_to_samples,
    spike_shift_grid_info, slice_spike_data_to_time_range, chunk_window_range_to_time,
)


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

    **Why a t-test rather than a curvature ratio.**  The test is
    ``|a2| / SE(a2) < curvature_t_threshold``: is the quadratic coefficient
    large compared with the uncertainty in the quadratic coefficient?  The
    earlier criterion compared curvature against the *slope*
    (``|a2/a1| < delta_threshold``), which divides by a quantity that goes to
    zero exactly when the data is cleanest.  Measured consequences of that:
    on genuinely linear data it declared linearity only 67% of the time, and
    the rate depended on how much bias was present (79% at near-zero bias,
    100% at strong bias) rather than on how linear the relation was.  It also
    failed in the other direction, accepting visibly curved data when the
    slope was large enough to mask the curvature.  Normalising by ``SE(a2)``
    instead is scale-free, needs no knowledge of the MI scale, and is the
    standard nested-model comparison; it declares linearity on linear data
    100% of the time and still refuses on genuinely curved data.

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
        Reported so the verdict can be audited rather than taken on trust.
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
        # Stop at the floor rather than stepping past it.  Popping here would
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


def _compute_fit_diagnostics(group: pd.DataFrame, gammas_used: List[int],
                               residual_threshold: float = 2.5,
                               r2_threshold: float = 0.90,
                               leverage_threshold: float = 0.20) -> Dict[str, Any]:
    """Computes fit diagnostics for the WLS linear extrapolation.

    Performs two checks:

    Check A — Residual quality: fits the WLS line on the final subset and
    examines externally studentized residuals.  Flags if
    ``max(|r_i|) > residual_threshold``.  R² is computed and returned for
    transparency but does **not** affect ``fit_quality_warning``: with large N
    the bias across gamma is inherently small (near-flat line) so R² collapses
    toward zero even for a sound fit, making it an unreliable gate here.
    ``fit_quality_warning`` itself is also informational only — it does **not**
    affect ``is_reliable`` upstream, because the heteroscedastic WLS structure
    (low-gamma rows dominating MSE, high-gamma rows having natural noise)
    routinely produces large studentized residuals for valid fits.

    Check B — LOO γ=1 stability: refits WLS excluding all rows where
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
    r2_threshold : float
        Unused by this function's own logic; accepted for a uniform call
        signature across diagnostics helpers. R² is reported as a diagnostic,
        not used as a gate: with large N the bias across gamma is inherently
        small (near-flat line) so R² collapses toward zero even for a sound
        fit, and if R² is already bad there's nothing meaningful to gate on.
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
        # else: fewer than 2 points after removing γ=1 — skip silently

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
    ceiling saturation are approximately *smooth* in gamma -- they inflate or
    suppress every rung's train_mi by a similar amount without bending the
    fitted line, so a real problem can sail through that check untouched.

    Spread : the std of train_mi across the ``gamma`` independently-trained
        chunks at each gamma level. With contiguous (temporal) chunking this
        is directly informative: large spread at high gamma means different
        segments of the recording give different answers -- non-stationarity
        showing up for free, something the library has no other diagnostic
        for. gamma=1 has a single chunk, so its spread is always 0.0/NaN
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
        logger.warning(
            f"Rigorous fit uses gamma={sorted(saturated_gammas)}, which are "
            f"saturated (mean train_saturation > {SATURATION_WARNING_THRESHOLD:.0%} "
            f"of their own ceiling). An extrapolation anchored on saturated "
            f"rungs cannot be trusted -- the fitted slope reflects the ceiling "
            f"as much as the true bias. Consider increasing max_eval_samples "
            f"or narrowing gamma_range to exclude these."
        )
        out['saturated_gammas'] = sorted(saturated_gammas)
    else:
        out['saturated_gammas'] = []
    return out


def _post_process_and_correct(df: pd.DataFrame, sweep_grid: Dict[str, Any],
                               curvature_t_threshold: float, min_gamma_points: int,
                               confidence_level: float,
                               residual_threshold: float = 2.5,
                               r2_threshold: float = 0.90,
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

    for params, group in valid_df.groupby(group_keys):
        # Ensure param_dict is correctly formed for single or multiple keys
        if isinstance(params, tuple):
            param_dict = dict(zip(group_keys, params))
        else:
            param_dict = {group_keys[0]: params}

        try:
            gammas_used, linear_region_found, curvature_stats = _find_linear_region(
                group, curvature_t_threshold, min_gamma_points)
            enough_gamma_points = len(gammas_used) >= min_gamma_points
            is_reliable = enough_gamma_points and linear_region_found
            if not enough_gamma_points:
                logger.warning(f"Fit for {param_dict} is unreliable (final gamma points < {min_gamma_points}).")
            elif not linear_region_found:
                logger.warning(
                    f"Fit for {param_dict} is unreliable: no linear region was found down to "
                    f"min_gamma_points={min_gamma_points}, so the fit uses gamma={gammas_used} "
                    f"without having satisfied the curvature criterion."
                )

            mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
                group, gammas_used, confidence_level
            )

            diagnostics = _compute_fit_diagnostics(
                group, gammas_used, residual_threshold, r2_threshold, leverage_threshold
            )
            per_gamma_diagnostics = _compute_per_gamma_diagnostics(group, gammas_used)

            if diagnostics['leverage_warning']:
                is_reliable = False
                logger.warning(
                    f"Fit diagnostics triggered for {param_dict}: "
                    f"leverage_warning={diagnostics['leverage_warning']}."
                )
            # fit_quality_warning is informational only; does not affect is_reliable
            if diagnostics['fit_quality_warning']:
                logger.debug(
                    f"fit_quality_warning=True for {param_dict} "
                    f"(large studentized residuals from heteroscedastic WLS noise — "
                    f"informational only)."
                )
            # Saturation is smooth in gamma -- the same blind spot
            # _find_linear_region's curvature check and leverage_warning both
            # have, confirmed empirically (w=50 acceptance run: 8/10 gammas
            # saturated, curvature and leverage checks both passed anyway).
            # Gate is_reliable the same way leverage_warning already does,
            # rather than leaving this purely informational -- a fit anchored
            # on saturated rungs is exactly the kind of unreliable this flag
            # exists to catch.
            if per_gamma_diagnostics['saturated_gammas']:
                is_reliable = False
                logger.warning(
                    f"Fit for {param_dict} marked unreliable: gamma="
                    f"{per_gamma_diagnostics['saturated_gammas']} used in the fit are ceiling-saturated."
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
            })
            param_dict.update(diagnostics)
            param_dict.update(per_gamma_diagnostics)
            param_dict.pop('dummy_group', None)
            corrected_results.append(param_dict)
        except InsufficientDataError as e:
            logger.error(f"Could not perform extrapolation for params {param_dict}: {e}")

    return corrected_results


# ---------------------------------------------------------------------------
# AnalysisWorkflow class
# ---------------------------------------------------------------------------

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
        # Raw (2-D), unwindowed data (shift_windows reachability) has no
        # per-window shape to infer dims from yet -- defer to task.py's own
        # per-task dimension inference once each chunk is actually windowed.
        # hasattr guards against spike data (a ragged per-neuron spike-time
        # list, no .shape) reaching here directly -- never actually deferred
        # today (shift_time is not yet extended to 'rigorous'), but safer
        # than assuming a tensor.
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
            Controls how the gamma-chunk subsampling orders data (see
            ``_prepare_tasks`` for the full rationale). ``None`` (default)
            auto-detects from ``base_params['leak_check_window_size']``;
            pass ``True``/``False`` to override.
        **kwargs : Dict[str, Any]
            Additional keyword arguments for the bias correction, such as
            ``curvature_t_threshold``, ``min_gamma_points``, ``confidence_level``,
            ``residual_threshold``, ``r2_threshold``, and ``leverage_threshold``.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:

            - ``'corrected_results'`` : list of per-group correction dicts.
              Each includes ``'chunking_mode'`` (``'contiguous'`` or
              ``'permuted'``, the resolved choice above),
              ``'n_tasks_created'``, ``'per_gamma_train_mi_spread'``,
              ``'per_gamma_ceiling_mi'``, and ``'per_gamma_saturation'``.
            - ``'raw_results_df'`` : pd.DataFrame — raw sweep results with one
              row per training run.  Key columns: ``gamma``, ``train_mi``.
        """
        n_workers = n_workers if n_workers is not None else 1
        show_progress = self.base_params.get('show_progress', True)
        logger.info(f"Starting rigorous analysis with {n_workers} workers...")
        tasks = self._prepare_tasks(param_grid, gamma_range, temporal_chunking=temporal_chunking)
        if not tasks:
            return {"corrected_results": [], "raw_results_df": pd.DataFrame()}

        # Fix 3: gamma_range=range(1,11) should create sum(1..10)=55 tasks per
        # param combination -- verify rather than assume, since a silently
        # truncated ladder would invalidate every rigorous result.
        expected_per_combo = sum(gamma_range)
        n_combos = max(1, len(param_grid or {}) and len(list(itertools.product(
            *(param_grid or {}).values()))) or 1)
        expected_total = expected_per_combo * n_combos
        if len(tasks) != expected_total:
            logger.warning(
                f"Rigorous analysis: expected {expected_total} tasks "
                f"({n_combos} combo(s) x sum(gamma_range)={expected_per_combo}) "
                f"but created {len(tasks)}. The gamma ladder may be truncated -- "
                f"investigate before trusting the extrapolation."
            )

        if n_workers <= 1:
            logger.info("Running rigorous analysis sequentially (n_workers=1)...")
            raw_results = [
                run_training_task(task)
                for task in tqdm(tasks, desc="Rigorous Analysis Progress",
                                 unit="task", disable=not show_progress)
            ]
        else:
            _configure_multiprocessing()
            _log_init, _log_args = worker_init_args()
            with mp.get_context('spawn').Pool(processes=n_workers,
                                          initializer=_log_init, initargs=_log_args) as pool:
                raw_results = list(tqdm(
                    pool.imap(run_training_task, tasks), total=len(tasks),
                    desc="Rigorous Analysis Progress", unit="task", disable=not show_progress
                ))

        logger.info("All training tasks finished. Performing bias correction...")
        raw_results_df = pd.DataFrame(raw_results)

        correction_kwargs = {
            'sweep_grid': param_grid,
            'curvature_t_threshold': kwargs.pop('curvature_t_threshold', 2.0),
            'min_gamma_points': kwargs.pop('min_gamma_points', 5),
            'confidence_level': kwargs.pop('confidence_level', 0.68),
            'residual_threshold': kwargs.pop('residual_threshold', 2.5),
            'r2_threshold': kwargs.pop('r2_threshold', 0.90),
            'leverage_threshold': kwargs.pop('leverage_threshold', 0.20),
        }

        corrected_results = _post_process_and_correct(
            raw_results_df, chunking_mode=self.resolved_chunking_mode,
            n_tasks_created=len(tasks), **correction_kwargs)
        return {"corrected_results": corrected_results, "raw_results_df": raw_results_df}

    def _prepare_tasks(self, param_grid: Optional[Dict[str, List]], gamma_range,
                       temporal_chunking: Optional[bool] = None) -> List[tuple]:
        """Prepares tasks using a hierarchical master-ordering subsampling strategy.

        Generate one master ordering at the start.  For each gamma G, split
        the master ordering into G equal chunks.  This ensures:

        - The gamma=2 subsets are literally halves of the gamma=1 dataset.
        - Each gamma level sees a consistent view of the data, only varying in N.
        - The linear fit extrapolates pure N-dependent bias, not noise variation.

        The master ordering is either a random permutation or the identity
        (contiguous, time-order-preserving) ordering, chosen by
        ``temporal_chunking``:

        - ``None`` (default): auto-detect from ``leak_check_window_size`` in
          ``base_params`` (set exactly when a windowed processor built the
          data -- see ``run.py`` / Phase 1 Fix 2). Present -> contiguous.
          Absent -> permuted (unchanged behaviour for i.i.d. data).
        - ``True`` / ``False``: explicit override.

        **Why this matters.** A random permutation is the right choice for
        i.i.d. data -- it doesn't matter what order rows appear in, and it
        keeps each gamma-chunk representative of the whole dataset. But for
        temporal data, ``Trainer._create_blocked_split`` takes *contiguous
        index blocks* of whatever ordering it receives to form the train/test
        split. A shuffled ordering makes those blocks random in time on
        autocorrelated data -- exactly the leakage ``split_mode='blocked'``
        exists to prevent, reintroduced inside every rung of the bias-
        correction ladder. Contiguous chunks keep each chunk's rows in time
        order, so the blocked split within a chunk means what it's supposed
        to.

        **Trade-off, not a strict improvement.** Random chunks each spanned
        the whole recording, averaging over any non-stationarity. Contiguous
        chunks do not -- at high gamma, each chunk is a short segment, and if
        the underlying process drifts, different chunks sample different
        regimes. Watch ``per_gamma_train_mi_spread`` in the result details:
        large spread at high gamma is exactly this showing up.
        """
        tasks = []
        run_id_base = str(uuid.uuid4())
        param_grid = param_grid or {}
        if self.base_params.get('critic_type') == 'concat' and 'embedding_dim' in param_grid:
            param_grid.pop('embedding_dim')

        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if param_grid else [{}]

        # Raw (2-D), unwindowed data (shift_windows reachability): windowing
        # hasn't happened yet, so `leak_check_window_size` (normally set by
        # run.py's eager windowing) isn't present yet either -- this data is
        # temporal regardless, and N must be the shift-invariant window
        # count, not a raw sample count.
        _proc_x = self.base_params.get('processor_type_x')
        _proc_y = self.base_params.get('processor_type_y') or _proc_x
        # hasattr guards against spike data (a ragged per-neuron spike-time
        # list, no .ndim) -- only reachable when _is_spike_deferred below.
        _is_raw_deferred = (hasattr(self.x_data, 'ndim') and self.x_data.ndim == 2
                            and self.base_params.get('shift_windows')
                            and shift_family(_proc_x, _proc_y) == 'regular')
        # Raw spike data (shift_time reachability, Phase 1): a ragged
        # per-neuron spike-time list rather than a 2-D tensor -- x_data has
        # no .ndim at all in this case.
        _is_spike_deferred = (isinstance(self.x_data, list) and self.base_params.get('shift_time')
                              and shift_family(_proc_x, _proc_y) == 'spike')
        if _is_spike_deferred:
            N, _spike_base_t_start, _spike_window_size, _spike_step_size = spike_shift_grid_info(
                self.x_data, self.y_data, self.base_params)
        else:
            N = n_windows_if_deferred(self.x_data, self.y_data, self.base_params)

        is_temporal = (temporal_chunking if temporal_chunking is not None
                      else (self.base_params.get('leak_check_window_size') is not None
                            or _is_raw_deferred or _is_spike_deferred))
        self.resolved_chunking_mode = 'contiguous' if is_temporal else 'permuted'

        if _is_raw_deferred:
            # Per-side window_size/step_size in raw-sample units, computed
            # once (rigorous doesn't route param_grid combos into
            # processor_params_x/y -- see sweep.py's own routing for
            # contrast -- so these can't vary per combo here).
            _wp_x = self.base_params.get('processor_params_x') or {}
            _wp_y = self.base_params.get('processor_params_y') or _wp_x
            _window_size = _wp_x.get('window_size')
            _step_size = _wp_x.get('step_size') or _window_size
            _period_x = 1.0 / _wp_x['sample_rate'] if _wp_x.get('sample_rate') else 1.0
            _period_y = 1.0 / _wp_y['sample_rate'] if _wp_y.get('sample_rate') else 1.0
            _window_size_x = seconds_to_samples(_window_size, _period_x)
            _step_size_x = seconds_to_samples(_step_size, _period_x)
            _window_size_y = seconds_to_samples(_window_size, _period_y)
            _step_size_y = seconds_to_samples(_step_size, _period_y)

        for i_combo, params in enumerate(param_combinations):
            current_params = {**self.base_params, **params}

            master_permutation = np.arange(N) if is_temporal else np.random.permutation(N)

            for gamma in gamma_range:
                current_params['gamma'] = gamma

                # Split the master permutation into gamma equal chunks.
                # np.array_split handles uneven divisions gracefully.
                chunks = np.array_split(master_permutation, gamma)

                min_chunk_size = min(len(c) for c in chunks)
                # Not a hard threshold and deliberately not a gate.  The line is
                # placed where a chunk's held-out partition can no longer fill a
                # single evaluation batch, which is the point below which a
                # per-chunk MI value stops being a measurement of anything.
                # Override with 'min_reliable_samples' in base_params.
                min_reliable_samples = current_params.get('min_reliable_samples')
                if min_reliable_samples is None:
                    _bs = current_params.get('batch_size', 128)
                    _train_frac = current_params.get('train_fraction', 0.9)
                    _held_out = max(1.0 - float(_train_frac), 1e-6)
                    min_reliable_samples = int(np.ceil(_bs / _held_out))
                if min_chunk_size < min_reliable_samples:
                    logger.warning(
                        f"gamma={gamma}: smallest data subset has {min_chunk_size} samples, "
                        f"below the ~{min_reliable_samples} at which the held-out partition "
                        f"of a chunk can still fill one evaluation batch "
                        f"(batch_size={current_params.get('batch_size', 128)}, "
                        f"train_fraction={current_params.get('train_fraction', 0.9)}). "
                        f"This is a caution rather than a threshold: MI values from chunks "
                        f"this small are dominated by noise, and a straight line through "
                        f"noisy rungs still looks straight, so the linearity check cannot "
                        f"warn you about it. Prefer a smaller gamma_range or more data. "
                        f"Set 'min_reliable_samples' in base_params to move this line."
                    )

                for i_subset, subset_indices in enumerate(chunks):
                    if _is_raw_deferred:
                        # subset_indices is a contiguous window-index range
                        # (guaranteed by master_permutation = np.arange(N)
                        # for is_temporal=True); translate to the raw
                        # sample range that reproduces exactly this many
                        # windows under any per-epoch shift.
                        lo, hi = int(subset_indices[0]), int(subset_indices[-1]) + 1
                        rx0, rx1 = chunk_window_range_to_raw(lo, hi, _window_size_x, _step_size_x)
                        ry0, ry1 = chunk_window_range_to_raw(lo, hi, _window_size_y, _step_size_y)
                        x_subset = _ensure_cpu(self.x_data[rx0:rx1])
                        y_subset = _ensure_cpu(self.y_data[ry0:ry1])
                    elif _is_spike_deferred:
                        # Same idea as _is_raw_deferred, but the chunk-index
                        # -> raw-range translation lands in a raw *time*
                        # range (chunk_window_range_to_time -- Phase 0's
                        # 2*window_size margin, not chunk_window_range_to_raw's
                        # window_size-1 one) rather than a raw sample range,
                        # and the "raw" data is a ragged per-neuron
                        # spike-time list sliced+re-zeroed rather than
                        # tensor-sliced. X and Y share one window grid here
                        # (spike/spike, no per-side sample-rate conversion),
                        # so both sides use the same time range.
                        lo, hi = int(subset_indices[0]), int(subset_indices[-1]) + 1
                        t0_rel, t1_rel = chunk_window_range_to_time(lo, hi, _spike_window_size, _spike_step_size)
                        chunk_span = t1_rel - t0_rel
                        t0_abs = _spike_base_t_start + t0_rel
                        t1_abs = _spike_base_t_start + t1_rel
                        x_subset = slice_spike_data_to_time_range(self.x_data, t0_abs, t1_abs)
                        y_subset = slice_spike_data_to_time_range(self.y_data, t0_abs, t1_abs)
                        # Force this chunk's own PairedTemporalDataset to treat
                        # [0, chunk_span) as its full base span (exactly what
                        # chunk_window_range_to_time assumed), instead of
                        # deriving a shorter, data-dependent extent from
                        # wherever this slice's actual spikes happen to fall
                        # -- create_dataset forwards 't_start'/'t_end'
                        # straight through to PairedTemporalDataset.
                        current_params['processor_params_x'] = {
                            **(current_params.get('processor_params_x') or {}),
                            't_start': 0.0, 't_end': chunk_span,
                        }
                    else:
                        x_subset = _ensure_cpu(self.x_data[subset_indices])
                        y_subset = _ensure_cpu(self.y_data[subset_indices])
                    task_run_id = f"{run_id_base}_c{i_combo}_g{gamma}_s{i_subset}"
                    # Purely deterministic per-task key for run_training_task's seeding
                    # (see task.py) -- unlike task_run_id above, excludes the random
                    # run_id_base prefix so a fixed random_seed reproduces the same
                    # task_seed (and result) on every call.
                    current_params['_seed_key'] = f"c{i_combo}_g{gamma}_s{i_subset}"
                    tasks.append((x_subset, y_subset, current_params.copy(), task_run_id))

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
        ``residual_threshold``, ``r2_threshold``, ``leverage_threshold``.

    Returns
    -------
    Dict[str, Any]
        Same dictionary returned by ``AnalysisWorkflow.run()``.  Key entries:

        - ``'corrected_results'`` : list of per-group correction dicts.
        - ``'raw_results_df'`` : pd.DataFrame — raw sweep results.
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

    Must be module-level (not a closure) so it — and the ``scalar_fn`` it
    carries — can be pickled for a ``multiprocessing`` 'spawn' pool.  Catches
    its own exceptions rather than letting them propagate through
    ``pool.imap``, which would otherwise abort every remaining task instead
    of just skipping the failed gamma-chunk (matching the sequential path's
    per-task try/except behaviour).
    """
    scalar_fn, x_sub, y_sub, params, extra_sub, extra_kwargs, gamma, chunk_size = args
    try:
        value = scalar_fn(x_sub, y_sub, params, **extra_sub, **(extra_kwargs or {}))
        return {'gamma': gamma, 'train_mi': value, '_error': None}
    except Exception as exc:
        return {'gamma': gamma, 'train_mi': None,
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
    r2_threshold: float = 0.90,
    leverage_threshold: float = 0.20,
    verbose: bool = False,
    temporal_chunking: Optional[bool] = None,
    raw_deferred: bool = False,
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
        ``scalar_fn(x_sub, y_sub, params, **extra_sub, **extra_kwargs) -> float``.
        It must return a single scalar MI (or MI-like) value.  It cannot be a
        closure or lambda when ``n_workers > 1``, since it is sent to worker
        processes by reference.
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
    r2_threshold : float, optional
        Passed to ``_compute_fit_diagnostics`` but unused by its logic; R² is
        reported as a diagnostic, not used as a gate (see that function's
        docstring).  Defaults to ``0.90``.
    leverage_threshold : float, optional
        Passed to ``_compute_fit_diagnostics``.  Defaults to ``0.20``.
    verbose : bool, optional
        Passed to ``_find_linear_region``.  Defaults to ``False``.
    raw_deferred : bool, optional
        ``True`` when ``x_data``/``y_data`` (and every array in
        ``extra_data``) are raw, unwindowed 2-D arrays and this call should
        reproduce ``AnalysisWorkflow._prepare_tasks``'s
        ``shift_windows``-reachability chunk-to-raw-sample-range
        translation (mirrored here as its own ``_is_raw_deferred`` check).
        Set explicitly by the caller (``run.py``'s
        ``_defer_for_conditional_interaction``) rather than inferred from
        ``base_params`` alone -- this helper is shared with
        ``mode='transfer'``'s rigorous dispatch, whose own 2-D
        ``x_data``/``y_data`` are raw for an unrelated reason (built via a
        stride-1 ``unfold`` internally, deliberately excluded from both
        shift mechanisms) and must never be reinterpreted as
        window-size/step_size-chunkable data. Defaults to ``False``.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:

        - ``'mi_corrected'`` : float — bias-corrected scalar estimate.
        - ``'mi_error'`` : float — half-width of the confidence interval.
        - ``'slope'`` : float — slope of the WLS fit (bias per unit gamma).
        - ``'is_reliable'`` : bool — True when all of: a linear region was
          actually found (``linear_region_found``), enough gamma points were
          retained (``enough_gamma_points``), ``leverage_warning`` is False, and
          no gamma used in the fit is ceiling-saturated.
          ``fit_quality_warning`` does **not** affect this flag (see below).
        - ``'linear_region_found'`` : bool — whether the curvature criterion was
          satisfied on the gammas that were used.  False means the search
          reached ``min_gamma_points`` without the fit ever looking linear, so
          the extrapolation is being done on a set that failed the check.  This
          is reported separately from ``enough_gamma_points`` so the two
          distinct failure modes can be told apart.
        - ``'enough_gamma_points'`` : bool — whether
          ``len(gammas_used) >= min_gamma_points``.
        - ``'gammas_used'`` : list of int — gamma values in the linear region.
        - ``'raw_results_df'`` : pd.DataFrame — one row per successful chunk call.
        - ``'fit_quality_warning'`` : bool — informational only; large studentized
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
    # Raw (2-D), unwindowed X/Y (shift_windows reachability for
    # conditional/interaction's rigorous=True sub-path, mirroring
    # AnalysisWorkflow._prepare_tasks's _is_raw_deferred for plain
    # mode='rigorous'): windowing hasn't happened yet, so N must be the
    # shift-invariant window count, not x_data's raw sample count. Gated on
    # the explicit raw_deferred flag (not inferred from base_params alone --
    # see its docstring entry) since mode='transfer' also reaches this
    # function with genuinely raw 2-D x_data/y_data, for an unrelated reason.
    _proc_x = base_params.get('processor_type_x')
    _proc_y = base_params.get('processor_type_y') or _proc_x
    _is_raw_deferred = (raw_deferred and hasattr(x_data, 'ndim') and x_data.ndim == 2
                        and base_params.get('shift_windows') and shift_family(_proc_x, _proc_y) == 'regular')
    # Spike analogue of _is_raw_deferred above (conditional/interaction's
    # rigorous=True sub-path for a spike+spike X/conditioning-variable
    # pair): x_data is a ragged per-neuron spike-time list, no .ndim at
    # all. Gated the same explicit way -- raw_deferred plus a family check
    # -- since mode='transfer' also reaches this function with unrelated
    # raw data that must never be misinterpreted. Deliberately NOT also
    # gated on base_params.get('shift_time'), unlike _is_raw_deferred's
    # shift_windows check above: run.py's _defer_spike_conditional_interaction
    # (the only caller that ever sets raw_deferred=True for a spike+spike
    # pair) is itself unconditional on shift_time -- merging X and the
    # conditioning variable before windowing is a correctness requirement
    # for spike coverage, not an optional shift-reachability path (see its
    # comment in run.py). Requiring shift_time here too used to mean a
    # spike+spike conditioning pair with Training(shift_time=False) still
    # arrived with raw_deferred=True and a list x_data, but fell through to
    # `else: N = x_data.shape[0]` below and crashed with AttributeError.
    _is_spike_deferred = (raw_deferred and isinstance(x_data, list)
                          and shift_family(_proc_x, _proc_y) == 'spike')
    if _is_spike_deferred:
        N, _spike_base_t_start, _spike_window_size, _spike_step_size = spike_shift_grid_info(
            x_data, y_data, base_params)
    elif _is_raw_deferred:
        N = n_windows_if_deferred(x_data, y_data, base_params)
    else:
        N = x_data.shape[0]
    # Same reasoning as AnalysisWorkflow._prepare_tasks: a random ordering is
    # fine for i.i.d. scalar quantities, but this helper also backs
    # mode='transfer' (rigorous=True), which is unconditionally temporal (TE
    # is built from time-ordered history windows) -- run.py passes
    # temporal_chunking=True explicitly there, since base_params at this call
    # site predates run_transfer_entropy's own leak_check_window_size
    # injection (it's set per-gamma-chunk, inside _te_rigorous_scalar, after
    # chunking has already happened) and so isn't available to auto-detect
    # from here. Falls back to the same base_params signal as the main
    # workflow for other callers (e.g. conditional MI), where it is
    # available in time.
    is_temporal = (temporal_chunking if temporal_chunking is not None
                   else (base_params.get('leak_check_window_size') is not None
                         or _is_raw_deferred or _is_spike_deferred))
    resolved_chunking_mode = 'contiguous' if is_temporal else 'permuted'
    master_perm = np.arange(N) if is_temporal else np.random.permutation(N)

    if _is_raw_deferred:
        # Per-side window_size/step_size in raw-sample units, computed once
        # (mirrors AnalysisWorkflow._prepare_tasks exactly). extra_data
        # arrays (w_data/c_data) are translated using X's own
        # window_size/step_size too, the same convention the non-rigorous
        # raw_deferred path already uses (conditional.py/interaction.py
        # concatenate W onto X before windowing, so they share one grid) --
        # a genuinely independent per-array window_size is a separate,
        # not-yet-fixed gap (see NEURALMI_REFERENCE.md).
        _wp_x = base_params.get('processor_params_x') or {}
        _wp_y = base_params.get('processor_params_y') or _wp_x
        _window_size = _wp_x.get('window_size')
        _step_size = _wp_x.get('step_size') or _window_size
        _period_x = 1.0 / _wp_x['sample_rate'] if _wp_x.get('sample_rate') else 1.0
        _period_y = 1.0 / _wp_y['sample_rate'] if _wp_y.get('sample_rate') else 1.0
        _window_size_x = seconds_to_samples(_window_size, _period_x)
        _step_size_x = seconds_to_samples(_step_size, _period_x)
        _window_size_y = seconds_to_samples(_window_size, _period_y)
        _step_size_y = seconds_to_samples(_step_size, _period_y)

    tasks = []
    for gamma in gamma_range:
        chunks = np.array_split(master_perm, gamma)
        for chunk_idx in chunks:
            if _is_spike_deferred:
                # Same idea as _is_raw_deferred, but the chunk-index ->
                # raw-range translation lands in a raw *time* range
                # (chunk_window_range_to_time -- the 2*window_size margin
                # matching PairedTemporalDataset._reserve_shift_margin, not
                # chunk_window_range_to_raw's window_size-1 one) rather than
                # a raw sample range, and the "raw" data is a ragged
                # per-neuron spike-time list sliced+re-zeroed rather than
                # tensor-sliced. X and the conditioning variable share one
                # window grid here (spike/spike, no per-side sample-rate
                # conversion), so both use the same time range -- extra_data
                # (w_data/c_data) is translated the same way.
                lo, hi = int(chunk_idx[0]), int(chunk_idx[-1]) + 1
                t0_rel, t1_rel = chunk_window_range_to_time(lo, hi, _spike_window_size, _spike_step_size)
                chunk_span = t1_rel - t0_rel
                t0_abs = _spike_base_t_start + t0_rel
                t1_abs = _spike_base_t_start + t1_rel
                x_sub = slice_spike_data_to_time_range(x_data, t0_abs, t1_abs)
                y_sub = slice_spike_data_to_time_range(y_data, t0_abs, t1_abs)
                extra_sub = {}
                if extra_data:
                    for key, arr in extra_data.items():
                        extra_sub[key] = slice_spike_data_to_time_range(arr, t0_abs, t1_abs)
                # Force this chunk's own PairedTemporalDataset to treat
                # [0, chunk_span) as its full base span (exactly what
                # chunk_window_range_to_time assumed), instead of deriving a
                # shorter, data-dependent extent from wherever this slice's
                # actual spikes happen to fall. A fresh dict per chunk (not
                # a reassignment of the shared base_params) so this doesn't
                # leak between iterations.
                _task_base_params = {**base_params, 'processor_params_x': {
                    **(base_params.get('processor_params_x') or {}),
                    't_start': 0.0, 't_end': chunk_span,
                }}
            elif _is_raw_deferred:
                # chunk_idx is a contiguous window-index range (guaranteed by
                # master_perm = np.arange(N) for is_temporal=True); translate
                # to the raw sample range reproducing exactly this many
                # windows under any per-epoch shift.
                lo, hi = int(chunk_idx[0]), int(chunk_idx[-1]) + 1
                rx0, rx1 = chunk_window_range_to_raw(lo, hi, _window_size_x, _step_size_x)
                ry0, ry1 = chunk_window_range_to_raw(lo, hi, _window_size_y, _step_size_y)
                x_sub = _ensure_cpu(x_data[rx0:rx1])
                y_sub = _ensure_cpu(y_data[ry0:ry1])
                extra_sub = {}
                if extra_data:
                    for key, arr in extra_data.items():
                        extra_sub[key] = _ensure_cpu(arr[rx0:rx1])
                _task_base_params = base_params
            else:
                x_sub = _ensure_cpu(x_data[chunk_idx])
                y_sub = _ensure_cpu(y_data[chunk_idx])

                extra_sub = {}
                if extra_data:
                    for key, arr in extra_data.items():
                        extra_sub[key] = _ensure_cpu(arr[chunk_idx])
                _task_base_params = base_params

            tasks.append((scalar_fn, x_sub, y_sub, _task_base_params.copy(),
                         extra_sub, extra_kwargs, gamma, len(chunk_idx)))

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
        _configure_multiprocessing()
        _log_init, _log_args = worker_init_args()
        with mp.get_context('spawn').Pool(processes=effective_workers,
                                      initializer=_log_init, initargs=_log_args) as pool:
            raw_rows = list(tqdm(
                pool.imap(_run_scalar_fn_task, tasks), total=len(tasks),
                desc="Rigorous scalar analysis", unit="task", disable=not show_progress
            ))

    rows = []
    for r in raw_rows:
        if r['_error'] is not None:
            logger.warning(f"run_rigorous_scalar_analysis: scalar_fn call failed for {r['_error']}")
        else:
            rows.append({'gamma': r['gamma'], 'train_mi': r['train_mi']})

    if len(rows) < min_gamma_points:
        raise InsufficientDataError(
            f"run_rigorous_scalar_analysis collected only {len(rows)} successful "
            f"scalar_fn calls, which is fewer than min_gamma_points={min_gamma_points}. "
            f"Cannot perform reliable extrapolation."
        )

    df = pd.DataFrame(rows, columns=['gamma', 'train_mi'])

    gammas_used, linear_region_found, curvature_stats = _find_linear_region(
        df, curvature_t_threshold, min_gamma_points)
    try:
        mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
            df, gammas_used, confidence_level
        )
    except InsufficientDataError:
        # Pruning left too few points — fall back to all available gammas and mark
        # the result as unreliable so callers are warned.
        gammas_used = sorted(df['gamma'].unique().tolist())
        linear_region_found = False
        logger.warning(
            "run_rigorous_scalar_analysis: linear region too small after pruning; "
            "falling back to all %d gamma values (is_reliable will be False).",
            len(gammas_used),
        )
        mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
            df, gammas_used, confidence_level
        )
    # Note: diagnostics uses the same gamma-based regression as _extrapolate_mi
    diagnostics = _compute_fit_diagnostics(df, gammas_used, residual_threshold, r2_threshold, leverage_threshold)

    enough_gamma_points = len(gammas_used) >= min_gamma_points
    is_reliable = enough_gamma_points and linear_region_found
    if enough_gamma_points and not linear_region_found:
        logger.warning(
            "run_rigorous_scalar_analysis: no linear region was found down to "
            "min_gamma_points=%d; the fit uses gamma=%s without having satisfied "
            "the curvature criterion.", min_gamma_points, gammas_used,
        )
    if diagnostics['leverage_warning']:
        is_reliable = False
        logger.warning(
            f"run_rigorous_scalar_analysis: fit diagnostics triggered: "
            f"leverage_warning={diagnostics['leverage_warning']}."
        )
    # fit_quality_warning is informational only; does not affect is_reliable
    if diagnostics['fit_quality_warning']:
        logger.debug(
            "run_rigorous_scalar_analysis: fit_quality_warning=True "
            "(large studentized residuals from heteroscedastic WLS noise — "
            "informational only)."
        )

    # Spread-across-chunks diagnostic (Fix 2a) applies here the same way as
    # the main workflow -- train_mi is available per chunk. The per-gamma
    # ceiling/saturation half (Fix 2b) does not: scalar_fn (e.g.
    # _te_rigorous_scalar) returns a bare float, not the richer per-run dict
    # AnalysisWorkflow's tasks produce, so train_ceiling_mi/train_saturation
    # were never propagated through _run_scalar_fn_task. Extending that would
    # mean changing the scalar_fn contract for every caller of this generic
    # helper (conditional MI's rigorous path too) -- a larger change than
    # this fix, flagged for a later phase rather than done here.
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
        'raw_results_df': df,
        'chunking_mode': resolved_chunking_mode,
        'n_tasks_created': len(tasks),
        'per_gamma_train_mi_spread': per_gamma_spread,
        **diagnostics,
    }
