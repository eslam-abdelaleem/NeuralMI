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
        # This check is now a safeguard; validation should happen in run.py
        raise ValueError("`processor_type_y` must be specified in `base_params` for lag analysis.")

    other_sweep_grid = sweep_grid.copy() if sweep_grid is not None else {}
    if 'lag' in other_sweep_grid:
        logger.warning("'lag' in sweep_grid is ignored when using mode='lag'.")
        other_sweep_grid.pop('lag')
        
    param_combinations = _product_dict(**other_sweep_grid) if other_sweep_grid else [{}]

    logger.info(f"Preparing {len(lag_range) * len(param_combinations)} tasks for lag analysis.")

    for lag in lag_range:
        x_shifted, y_shifted = _shift_data(x_data, y_data, lag, proc_type_y)
        
        for i, other_params in enumerate(param_combinations):
            task_params = {**base_params, **other_params, 'lag': lag}
            run_id = other_params.get('run_id', f"lag{lag}_combo{i}")
            all_tasks.append((x_shifted, y_shifted, task_params, run_id))

    # We can initialize with None data because the actual data for each task is already prepared
    sweep_runner = ParameterSweep(x_data=None, y_data=None, base_params=base_params)
    results_list = sweep_runner._run_parallel(all_tasks, n_workers=n_workers)
    
    return results_list