# neural_mi/analysis/rigorous.py
"""Implements the 'rigorous' bias-corrected MI analysis mode.

Provides the public function ``run_rigorous_analysis`` and the
``AnalysisWorkflow`` class that orchestrate the multi-step process for a
rigorous, bias-corrected mutual information estimate.

The estimator trains models on progressively smaller data subsets (parameterised
by *gamma*, the fraction of the full dataset) and extrapolates the 1/gamma → 0
intercept to obtain a bias-corrected MI value.  All per-run MI estimates are
``train_mi`` (evaluated on the large training partition at the best-generalising
checkpoint), consistent with every other analysis mode.
"""
import torch
import numpy as np
import pandas as pd
import itertools
import uuid
import torch.multiprocessing as mp
import statsmodels.api as sm
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger
from neural_mi.exceptions import InsufficientDataError, TrainingError
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu


# ---------------------------------------------------------------------------
# Internal bias-correction helpers
# ---------------------------------------------------------------------------

def _find_linear_region(group: pd.DataFrame, delta_threshold: float,
                         min_gamma_points: int, verbose: bool) -> List[int]:
    """Finds the linear region of the MI vs. 1/gamma plot.

    This function iteratively removes the largest gamma values and re-fits a
    quadratic model until the curvature (|a2/a1|) is below the
    ``delta_threshold``, indicating a sufficiently linear region.

    The dependent variable is ``train_mi`` (the training-partition MI at the
    best-generalising checkpoint), which is what we extrapolate to
    gamma → 0 (infinite data).
    """
    gammas_to_fit = sorted(group['gamma'].unique())
    while len(gammas_to_fit) >= min_gamma_points:
        subset = group[group['gamma'].isin(gammas_to_fit)].copy()
        if len(subset) < 3:
            break
        subset['inv_gamma'] = 1.0 / subset['gamma']
        weights = 1 / subset['gamma'].map(subset['gamma'].value_counts())
        X_quad = sm.add_constant(np.vstack([subset['inv_gamma'], subset['inv_gamma']**2]).T)
        model_quad = sm.WLS(subset['train_mi'], X_quad, weights=weights).fit()
        _, a1, a2 = model_quad.params
        final_delta = abs(a2 / a1) if a1 != 0 else float('inf')
        if final_delta < delta_threshold:
            break
        gammas_to_fit.pop(-1)
    return gammas_to_fit


def _extrapolate_mi(group: pd.DataFrame, gammas_to_fit: List[int],
                     confidence_level: float) -> tuple:
    """Extrapolates MI to infinite data limit (gamma→0, i.e. 1/N→0).

    Fits ``train_mi = intercept + slope * (1/gamma)`` via WLS and returns
    ``(intercept, mi_error, mi_error_pred, slope)``.

    Two uncertainty intervals are computed at the extrapolation point
    (``1/gamma = 0``):

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

    final_subset['inv_gamma'] = 1.0 / final_subset['gamma']
    weights = 1 / final_subset['gamma'].map(final_subset['gamma'].value_counts())
    X_linear = sm.add_constant(final_subset['inv_gamma'])
    fit_linear = sm.WLS(final_subset['train_mi'], X_linear, weights=weights).fit()
    intercept, slope = fit_linear.params

    pred = fit_linear.get_prediction(exog=[1, 0])
    alpha = 1 - confidence_level

    # Confidence interval: uncertainty in the fitted mean at 1/gamma = 0
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
    examines externally studentized residuals and R².  Flags if
    ``max(|r_i|) > residual_threshold`` or ``R² < r2_threshold``.

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
        Minimum acceptable R² for the linear fit.
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

    final_subset['inv_gamma'] = 1.0 / final_subset['gamma']
    weights = 1 / final_subset['gamma'].map(final_subset['gamma'].value_counts())
    X_linear = sm.add_constant(final_subset['inv_gamma'])
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
    fit_quality_warning = (max_abs_residual > residual_threshold) or (r_squared < r2_threshold)

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
            X_loo = sm.add_constant(loo_subset['inv_gamma'])
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


def _post_process_and_correct(df: pd.DataFrame, sweep_grid: Dict[str, Any],
                               delta_threshold: float, min_gamma_points: int,
                               confidence_level: float, verbose: bool,
                               residual_threshold: float = 2.5,
                               r2_threshold: float = 0.90,
                               leverage_threshold: float = 0.20) -> List[Dict[str, Any]]:
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
            gammas_used = _find_linear_region(group, delta_threshold, min_gamma_points, verbose)
            is_reliable = len(gammas_used) >= min_gamma_points
            if not is_reliable:
                logger.warning(f"Fit for {param_dict} is unreliable (final gamma points < {min_gamma_points}).")

            mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
                group, gammas_used, confidence_level
            )

            diagnostics = _compute_fit_diagnostics(
                group, gammas_used, residual_threshold, r2_threshold, leverage_threshold
            )

            if diagnostics['fit_quality_warning'] or diagnostics['leverage_warning']:
                is_reliable = False
                logger.warning(
                    f"Fit diagnostics triggered for {param_dict}: "
                    f"fit_quality_warning={diagnostics['fit_quality_warning']}, "
                    f"leverage_warning={diagnostics['leverage_warning']}."
                )

            param_dict.update({
                'mi_corrected': mi_corrected,
                'mi_error': mi_error,
                'mi_error_pred': mi_error_pred,
                'slope': slope,
                'is_reliable': is_reliable,
                'gammas_used': gammas_used,
            })
            param_dict.update(diagnostics)
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
        self.base_params = base_params
        self.base_params.update({
            'input_dim_x': x_data.shape[1] * x_data.shape[2],
            'input_dim_y': y_data.shape[1] * y_data.shape[2],
            'n_channels_x': x_data.shape[1],
            'n_channels_y': y_data.shape[1],
            **kwargs
        })

    def run(self, param_grid: Optional[Dict[str, List]] = None,
            gamma_range=range(1, 11),
            n_workers: Optional[int] = None, **kwargs) -> Dict[str, Any]:
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
        **kwargs : Dict[str, Any]
            Additional keyword arguments for the bias correction, such as
            ``delta_threshold``, ``min_gamma_points``, ``confidence_level``,
            ``residual_threshold``, ``r2_threshold``, and ``leverage_threshold``.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:

            - ``'corrected_results'`` : list of per-group correction dicts.
            - ``'raw_results_df'`` : pd.DataFrame — raw sweep results with one
              row per training run.  Key columns: ``gamma``, ``train_mi``.
        """
        n_workers = n_workers if n_workers is not None else 1
        show_progress = self.base_params.get('show_progress', True)
        logger.info(f"Starting rigorous analysis with {n_workers} workers...")
        tasks = self._prepare_tasks(param_grid, gamma_range)
        if not tasks:
            return {"corrected_results": [], "raw_results_df": pd.DataFrame()}

        if n_workers <= 1:
            logger.info("Running rigorous analysis sequentially (n_workers=1)...")
            raw_results = [
                run_training_task(task)
                for task in tqdm(tasks, desc="Rigorous Analysis Progress",
                                 unit="task", disable=not show_progress)
            ]
        else:
            _configure_multiprocessing()
            with mp.get_context('spawn').Pool(processes=n_workers) as pool:
                raw_results = list(tqdm(
                    pool.imap(run_training_task, tasks), total=len(tasks),
                    desc="Rigorous Analysis Progress", unit="task", disable=not show_progress
                ))

        logger.info("All training tasks finished. Performing bias correction...")
        raw_results_df = pd.DataFrame(raw_results)

        correction_kwargs = {
            'sweep_grid': param_grid,
            'delta_threshold': kwargs.pop('delta_threshold', 0.1),
            'min_gamma_points': kwargs.pop('min_gamma_points', 5),
            'confidence_level': kwargs.pop('confidence_level', 0.68),
            'verbose': kwargs.get('verbose', False),
            'residual_threshold': kwargs.pop('residual_threshold', 2.5),
            'r2_threshold': kwargs.pop('r2_threshold', 0.90),
            'leverage_threshold': kwargs.pop('leverage_threshold', 0.20),
        }

        corrected_results = _post_process_and_correct(raw_results_df, **correction_kwargs)
        return {"corrected_results": corrected_results, "raw_results_df": raw_results_df}

    def _prepare_tasks(self, param_grid: Optional[Dict[str, List]], gamma_range) -> List[tuple]:
        """Prepares tasks using a hierarchical master-permutation subsampling strategy.

        Generate one master permutation at the start.  For each gamma G,
        split the master permutation into G equal chunks.  This ensures:

        - The gamma=2 subsets are literally halves of the gamma=1 dataset.
        - Each gamma level sees a consistent view of the data, only varying in N.
        - The linear fit extrapolates pure N-dependent bias, not noise variation.
        """
        tasks = []
        run_id_base = str(uuid.uuid4())
        param_grid = param_grid or {}
        if self.base_params.get('critic_type') == 'concat' and 'embedding_dim' in param_grid:
            param_grid.pop('embedding_dim')

        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if param_grid else [{}]

        N = self.x_data.shape[0]

        for i_combo, params in enumerate(param_combinations):
            current_params = {**self.base_params, **params}

            master_permutation = np.random.permutation(N)

            for gamma in gamma_range:
                current_params['gamma'] = gamma

                # Split the master permutation into gamma equal chunks.
                # np.array_split handles uneven divisions gracefully.
                chunks = np.array_split(master_permutation, gamma)

                min_chunk_size = min(len(c) for c in chunks)
                min_reliable_samples = current_params.get('min_reliable_samples', 1000)
                if min_chunk_size < min_reliable_samples:
                    logger.warning(
                        f"gamma={gamma}: smallest data subset has {min_chunk_size} samples "
                        f"(threshold: {min_reliable_samples}). MI estimates at this gamma "
                        f"may be unreliable. Consider reducing gamma_range or collecting "
                        f"more data. Set 'min_reliable_samples' in base_params to adjust "
                        f"this threshold."
                    )

                for i_subset, subset_indices in enumerate(chunks):
                    x_subset = _ensure_cpu(self.x_data[subset_indices])
                    y_subset = _ensure_cpu(self.y_data[subset_indices])
                    task_run_id = f"{run_id_base}_c{i_combo}_g{gamma}_s{i_subset}"
                    tasks.append((x_subset, y_subset, current_params.copy(), task_run_id))

        logger.debug(f"Created {len(tasks)} tasks to run...")
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
        Common ones: ``delta_threshold``, ``min_gamma_points``, ``confidence_level``,
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


def run_rigorous_scalar_analysis(
    scalar_fn,
    x_data,
    y_data,
    base_params: Dict[str, Any],
    extra_data: Optional[Dict[str, Any]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    gamma_range=range(1, 11),
    delta_threshold: float = 0.1,
    min_gamma_points: int = 5,
    confidence_level: float = 0.68,
    residual_threshold: float = 2.5,
    r2_threshold: float = 0.90,
    leverage_threshold: float = 0.20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Bias-corrected estimation of a compound scalar quantity via rigorous extrapolation.

    Used by conditional and transfer MI modes to apply the same finite-data
    bias correction to arbitrary scalar functions of the data (e.g. conditional
    MI, transfer entropy) without requiring a full ``AnalysisWorkflow`` training
    loop.

    The function calls ``scalar_fn`` on progressively smaller subsets of the
    data (controlled by *gamma*), collects the scalar outputs, and extrapolates
    to the infinite-data limit using the same WLS linear fit as the standard
    rigorous analysis.

    Parameters
    ----------
    scalar_fn : callable
        A function with signature
        ``scalar_fn(x_sub, y_sub, params, **extra_sub, **extra_kwargs) -> float``.
        It must return a single scalar MI (or MI-like) value.
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
    delta_threshold : float, optional
        Curvature threshold for ``_find_linear_region``.  Defaults to ``0.1``.
    min_gamma_points : int, optional
        Minimum number of distinct gamma values required for a reliable fit.
        Defaults to ``5``.
    confidence_level : float, optional
        Confidence level for the extrapolation error interval.  Defaults to
        ``0.68`` (roughly ±1 σ).
    residual_threshold : float, optional
        Passed to ``_compute_fit_diagnostics``.  Defaults to ``2.5``.
    r2_threshold : float, optional
        Passed to ``_compute_fit_diagnostics``.  Defaults to ``0.90``.
    leverage_threshold : float, optional
        Passed to ``_compute_fit_diagnostics``.  Defaults to ``0.20``.
    verbose : bool, optional
        Passed to ``_find_linear_region``.  Defaults to ``False``.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:

        - ``'mi_corrected'`` : float — bias-corrected scalar estimate.
        - ``'mi_error'`` : float — half-width of the confidence interval.
        - ``'slope'`` : float — slope of the WLS fit (bias per unit 1/gamma).
        - ``'is_reliable'`` : bool — True if the fit passes all quality checks.
        - ``'gammas_used'`` : list of int — gamma values in the linear region.
        - ``'raw_results_df'`` : pd.DataFrame — one row per successful chunk call.
        - ``'fit_quality_warning'`` : bool
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
    N = x_data.shape[0]
    master_perm = np.random.permutation(N)

    rows = []
    for gamma in gamma_range:
        chunks = np.array_split(master_perm, gamma)
        for chunk_idx in chunks:
            x_sub = x_data[chunk_idx]
            y_sub = y_data[chunk_idx]

            extra_sub = {}
            if extra_data:
                for key, arr in extra_data.items():
                    extra_sub[key] = arr[chunk_idx]

            try:
                scalar_value = scalar_fn(
                    x_sub, y_sub, base_params.copy(),
                    **extra_sub,
                    **(extra_kwargs or {})
                )
                rows.append({'gamma': gamma, 'train_mi': scalar_value})
            except Exception as exc:
                logger.warning(
                    f"run_rigorous_scalar_analysis: scalar_fn call failed for "
                    f"gamma={gamma} (chunk size={len(chunk_idx)}): {exc}"
                )

    if len(rows) < min_gamma_points:
        raise InsufficientDataError(
            f"run_rigorous_scalar_analysis collected only {len(rows)} successful "
            f"scalar_fn calls, which is fewer than min_gamma_points={min_gamma_points}. "
            f"Cannot perform reliable extrapolation."
        )

    df = pd.DataFrame(rows, columns=['gamma', 'train_mi'])

    gammas_used = _find_linear_region(df, delta_threshold, min_gamma_points, verbose)
    try:
        mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
            df, gammas_used, confidence_level
        )
    except InsufficientDataError:
        # Pruning left too few points — fall back to all available gammas and mark
        # the result as unreliable so callers are warned.
        gammas_used = sorted(df['gamma'].unique().tolist())
        logger.warning(
            "run_rigorous_scalar_analysis: linear region too small after pruning; "
            "falling back to all %d gamma values (is_reliable will be False).",
            len(gammas_used),
        )
        mi_corrected, mi_error, mi_error_pred, slope = _extrapolate_mi(
            df, gammas_used, confidence_level
        )
    diagnostics = _compute_fit_diagnostics(df, gammas_used, residual_threshold, r2_threshold, leverage_threshold)

    is_reliable = len(gammas_used) >= min_gamma_points
    if diagnostics['fit_quality_warning'] or diagnostics['leverage_warning']:
        is_reliable = False
        logger.warning(
            f"run_rigorous_scalar_analysis: fit diagnostics triggered: "
            f"fit_quality_warning={diagnostics['fit_quality_warning']}, "
            f"leverage_warning={diagnostics['leverage_warning']}."
        )

    return {
        'mi_corrected': mi_corrected,
        'mi_error': mi_error,
        'mi_error_pred': mi_error_pred,
        'slope': slope,
        'is_reliable': is_reliable,
        'gammas_used': gammas_used,
        'raw_results_df': df,
        **diagnostics,
    }
