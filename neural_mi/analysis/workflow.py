# neural_mi/analysis/workflow.py
"""Implements the 'rigorous' analysis workflow for bias correction.

This module contains the `AnalysisWorkflow` class, which orchestrates the
multi-step process for a rigorous, bias-corrected mutual information estimate.
This involves training models on subsets of the data of varying sizes (controlled
by `gamma`) and then extrapolating the results to estimate the MI at an
infinite data limit.
"""
import torch
import numpy as np
import pandas as pd
import itertools
import uuid
import multiprocessing
from multiprocessing import cpu_count
import statsmodels.api as sm
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger
from neural_mi.exceptions import InsufficientDataError, TrainingError

def __find_linear_region(group: pd.DataFrame, delta_threshold: float,
                         min_gamma_points: int, verbose: bool) -> List[int]:
    """Finds the linear region of the MI vs. 1/gamma plot.

    This function iteratively removes the largest gamma values and re-fits a
    quadratic model until the curvature (|a2/a1|) is below the delta_threshold,
    indicating a sufficiently linear region.
    """
    gammas_to_fit = sorted(group['gamma'].unique())
    while len(gammas_to_fit) >= min_gamma_points:
        subset = group[group['gamma'].isin(gammas_to_fit)]
        if len(subset) < 3: break
        weights = 1 / subset['gamma'].map(subset['gamma'].value_counts())
        X_quad = sm.add_constant(np.vstack([subset['gamma'], subset['gamma']**2]).T)
        model_quad = sm.WLS(subset['test_mi'], X_quad, weights=weights).fit()
        _, a1, a2 = model_quad.params
        final_delta = abs(a2 / a1) if a1 != 0 else float('inf')
        if final_delta < delta_threshold: break
        gammas_to_fit.pop(-1)
    return gammas_to_fit

def __extrapolate_mi(group: pd.DataFrame, gammas_to_fit: List[int],
                     confidence_level: float) -> tuple:
    """Extrapolates MI to infinite data limit from the linear region.

    Performs a weighted least squares linear fit on the identified linear
    region of the MI vs. gamma plot. The intercept of this fit is the
    bias-corrected MI estimate.
    """
    final_subset = group[group['gamma'].isin(gammas_to_fit)]
    if len(final_subset) < 2:
        raise InsufficientDataError("Not enough points for a reliable linear fit after pruning.")
    
    weights = 1 / final_subset['gamma'].map(final_subset['gamma'].value_counts())
    X_linear = sm.add_constant(final_subset['gamma'])
    fit_linear = sm.WLS(final_subset['test_mi'], X_linear, weights=weights).fit()
    intercept, slope = fit_linear.params
    conf_interval = fit_linear.get_prediction(exog=[1, 0]).conf_int(obs=True, alpha=1-confidence_level)[0]
    mi_error = (conf_interval[1] - conf_interval[0]) / 2.0
    return intercept, mi_error, slope

def _post_process_and_correct(df: pd.DataFrame, sweep_grid: Dict[str, Any], delta_threshold: float, 
                              min_gamma_points: int, confidence_level: float, verbose: bool) -> List[Dict[str, Any]]:
    """Groups results and performs bias correction for each group."""
    valid_df = df.dropna(subset=['gamma', 'test_mi'])
    if valid_df.empty:
        raise TrainingError("Rigorous analysis failed: all training runs produced NaN MI values.")

    group_keys = list(sweep_grid.keys()) if sweep_grid else []
    
    corrected_results = []
    
    # If there are no sweep parameters, group the whole dataframe as one.
    if not group_keys:
        group_keys.append('dummy_group')
        valid_df['dummy_group'] = 0

    for params, group in valid_df.groupby(group_keys):
        # Ensure param_dict is correctly formed for single or multiple keys
        if isinstance(params, tuple):
            param_dict = dict(zip(group_keys, params))
        else:
            param_dict = {group_keys[0]: params}

        try:
            gammas_used = __find_linear_region(group, delta_threshold, min_gamma_points, verbose)
            is_reliable = len(gammas_used) >= min_gamma_points
            if not is_reliable:
                logger.warning(f"Fit for {param_dict} is unreliable (final gamma points < {min_gamma_points}).")

            mi_corrected, mi_error, slope = __extrapolate_mi(group, gammas_used, confidence_level)
            param_dict.update({
                'mi_corrected': mi_corrected, 'mi_error': mi_error, 'slope': slope,
                'is_reliable': is_reliable, 'gammas_used': gammas_used
            })
            corrected_results.append(param_dict)
        except InsufficientDataError as e:
            logger.error(f"Could not perform extrapolation for params {param_dict}: {e}")

    return corrected_results


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
            Additional keyword arguments to be added to `base_params`.
        """
        self.x_data, self.y_data = x_data, y_data
        self.base_params = base_params
        self.base_params.update({
            'input_dim_x': x_data.shape[1] * x_data.shape[2],
            'input_dim_y': y_data.shape[1] * y_data.shape[2],
            **kwargs
        })

    def run(self, param_grid: Optional[Dict[str, List]], gamma_range=range(1, 11),
            n_workers: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Executes the full rigorous analysis workflow.

        This involves preparing tasks for different data subsets (controlled by
        `gamma`), running them in parallel, and then applying a post-processing
        and bias correction step to the aggregated results.

        Parameters
        ----------
        param_grid : Dict[str, List], optional
            A grid of hyperparameters to sweep over in addition to the gamma sweep.
        gamma_range : range, optional
            The range of gamma values to use for data subsampling. Defaults to `range(1, 11)`.
        n_workers : int, optional
            The number of worker processes to use. Defaults to the number of CPU cores.
        **kwargs : Dict[str, Any]
            Additional keyword arguments for the bias correction, such as
            `delta_threshold`, `min_gamma_points`, and `confidence_level`.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the corrected results and the raw DataFrame
            from all the individual training runs.
        """
        n_workers = n_workers or cpu_count()
        logger.info(f"Starting rigorous analysis with {n_workers} workers...")
        tasks = self._prepare_tasks(param_grid, gamma_range)
        if not tasks:
            return {"corrected_results": [], "raw_results_df": pd.DataFrame()}

        with multiprocessing.get_context("spawn").Pool(processes=n_workers) as pool:
            raw_results = list(tqdm(
                pool.imap(run_training_task, tasks), total=len(tasks),
                desc="Rigorous Analysis Progress", unit="task"
            ))

        logger.info("All training tasks finished. Performing bias correction...")
        raw_results_df = pd.DataFrame(raw_results)
        
        correction_kwargs = {
            'sweep_grid': param_grid,
            'delta_threshold': kwargs.pop('delta_threshold', 0.1),
            'min_gamma_points': kwargs.pop('min_gamma_points', 5),
            'confidence_level': kwargs.pop('confidence_level', 0.68),
            'verbose': kwargs.get('verbose', False)
        }
        
        corrected_results = _post_process_and_correct(raw_results_df, **correction_kwargs)
        return {"corrected_results": corrected_results, "raw_results_df": raw_results_df}

    def _prepare_tasks(self, param_grid: Optional[Dict[str, List]], gamma_range) -> List[tuple]:
        tasks = []; run_id_base = str(uuid.uuid4())
        param_grid = param_grid or {}
        if self.base_params.get('critic_type') == 'concat' and 'embedding_dim' in param_grid:
            param_grid.pop('embedding_dim')
        
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if param_grid else [{}]

        for i_combo, params in enumerate(param_combinations):
            current_params = {**self.base_params, **params}
            for gamma in gamma_range:
                current_params['gamma'] = gamma
                indices = np.random.permutation(self.x_data.shape[0])
                for i_subset, subset_indices in enumerate(np.array_split(indices, gamma)):
                    x_subset, y_subset = self.x_data[subset_indices], self.y_data[subset_indices]
                    task_run_id = f"{run_id_base}_c{i_combo}_g{gamma}_s{i_subset}"
                    tasks.append((x_subset, y_subset, current_params.copy(), task_run_id))
        logger.debug(f"Created {len(tasks)} tasks to run...")
        return tasks