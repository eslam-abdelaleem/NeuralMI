# neural_mi/analysis/lag.py
"""Implements the 'lag' analysis workflow for finding time-delayed relationships.

This module contains the `run_lag_analysis` function, which orchestrates the
process of estimating mutual information between two variables, X and Y, across
a range of specified time lags using the nonlinear cross-correlation method.
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from neural_mi.analysis.sweep import ParameterSweep, _product_dict
from neural_mi.logger import logger
from neural_mi.utils import _shift_data


def run_lag_analysis(
    x_data: Any,
    y_data: Any,
    base_params: Dict[str, Any],
    lag_range: range,
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    equalize_n: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Estimates MI across a range of lags via nonlinear cross-correlation.

    This function iterates through each specified lag, manually shifts the y_data
    relative to x_data, and then runs a full, independent MI estimation for that
    specific lag. This process is parallelized across `n_workers`.

    Parameters
    ----------
    x_data : Any
        The raw data for variable X.
    y_data : Any
        The raw data for variable Y.
    base_params : Dict[str, Any]
        A dictionary of fixed parameters for the MI estimator's trainer.
    lag_range : range
        A range of integer lags to test.
    sweep_grid : dict, optional
        An optional dictionary of other hyperparameters to sweep over, such as
        `{'run_id': range(5)}`.
    n_workers : int, optional
        The number of parallel processes to use for the analysis.
    equalize_n : bool, optional
        If True, will equalize the number of samples in each task by truncating according to the smallest.
    **kwargs : dict
        Additional keyword arguments (currently unused in this mode).

    Returns
    -------
    List[Dict[str, Any]]
        A list of result dictionaries from all the runs in the sweep.
    """
    all_tasks = []

    proc_type_y = base_params.get('processor_type_y')
    if not proc_type_y:
        proc_type_y = base_params.get('processor_type_x')
        logger.info("`processor_type_y` not specified in `base_params`, using the same as for x.")

    # Iinfer sample_rate from processor_params to resolve unit ambiguity
    sample_rate = base_params.get('processor_params_x', {}).get('sample_rate', None)
    if sample_rate is None:
        sample_rate = base_params.get('processor_params_y', {}).get('sample_rate', None)

    other_sweep_grid = sweep_grid.copy() if sweep_grid is not None else {}
    if 'lag' in other_sweep_grid:
        logger.warning("'lag' in sweep_grid is ignored when using mode='lag'.")
        other_sweep_grid.pop('lag')

    param_combinations = _product_dict(**other_sweep_grid) if other_sweep_grid else [{}]

    logger.info(f"Preparing {len(lag_range) * len(param_combinations)} tasks for lag analysis.")
    if sample_rate is not None:
        logger.info(f"Lag units: seconds (sample_rate={sample_rate} Hz). "
                    f"Lag range [{min(lag_range)}, {max(lag_range)}]s = "
                    f"[{int(round(min(lag_range)*sample_rate))}, "
                    f"{int(round(max(lag_range)*sample_rate))}] samples.")
    else:
        logger.info(f"Lag units: samples (no sample_rate provided).")

    # Pre-compute shifted arrays for all lags to measure n_windows per lag,
    # then optionally equalize to the minimum across lags.
    shifted_pairs = {}
    for lag in lag_range:
        x_sh, y_sh = _shift_data(x_data, y_data, lag, proc_type_y, sample_rate=sample_rate)
        shifted_pairs[lag] = (x_sh, y_sh)

    if equalize_n:
        # Determine minimum number of samples/windows across all lags
        min_n = min(x_sh.shape[0] for x_sh, _ in shifted_pairs.values())
        logger.info(
            f"equalize_n=True: truncating all lags to {min_n} samples "
            f"(limited by the largest lag in the range)."
        )
        equalized = {}
        for lag, (x_sh, y_sh) in shifted_pairs.items():
            equalized[lag] = (x_sh[:min_n], y_sh[:min_n])
        shifted_pairs = equalized

    for lag in lag_range:
        x_shifted, y_shifted = shifted_pairs[lag]
        n_windows_this_lag = x_shifted.shape[0]

        for i, other_params in enumerate(param_combinations):
            task_params = {**base_params, **other_params, 'lag': lag,
                           '_n_windows_lag': n_windows_this_lag}
            run_id = other_params.get('run_id', f"lag{lag}_combo{i}")
            all_tasks.append((x_shifted, y_shifted, task_params, run_id))

    sweep_runner = ParameterSweep(x_data=None, y_data=None, base_params=base_params)
    results_list = sweep_runner._run_parallel(all_tasks, n_workers=n_workers)

    # Propagate n_windows into each result dict so it appears in the dataframe
    for result, task in zip(results_list, all_tasks):
        if isinstance(result, dict):
            result['n_windows'] = task[2].get('_n_windows_lag', None)

    return results_list











