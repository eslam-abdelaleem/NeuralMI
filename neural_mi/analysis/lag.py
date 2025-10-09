# neural_mi/analysis/lag.py
"""Implements the 'lag' analysis workflow for finding time-delayed relationships.

This module contains the `run_lag_analysis` function, which orchestrates the
process of estimating mutual information between two variables, X and Y, across
a range of specified time lags.
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from neural_mi.analysis.sweep import ParameterSweep
from neural_mi.logger import logger
from neural_mi.utils import _shift_data

def run_lag_analysis(
    x_data: Any,
    y_data: Any,
    base_params: Dict[str, Any],
    lag_range: range,
    sweep_grid: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Estimates mutual information across a range of time lags and other optional sweep parameters.

    For each combination of parameters, this function shifts y_data relative to x_data
    and then runs a standard MI estimation.

    Parameters
    ----------
    x_data : Any
        The raw data for variable X.
    y_data : Any
        The raw data for variable Y.
    base_params : Dict[str, Any]
        A dictionary of fixed parameters for the MI estimator's trainer.
    lag_range : range
        A range of lags to test.
    sweep_grid : dict, optional
        An optional dictionary of other hyperparameters to sweep over (e.g., {'run_id': range(5)}).
    **kwargs : dict
        Additional keyword arguments passed to the `ParameterSweep`.

    Returns
    -------
    List[Dict[str, Any]]
        A list of result dictionaries from all the runs in the sweep.
    """
    # Start with the user-provided sweep grid, or an empty one
    full_sweep_grid = sweep_grid.copy() if sweep_grid is not None else {}

    # Add the lag range to the sweep grid
    if 'lag' in full_sweep_grid:
        logger.warning("'lag' in sweep_grid is being overwritten by lag_range.")
    full_sweep_grid['lag'] = list(lag_range)

    # For lag analysis, data processing is always deferred to workers.
    sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params)

    # The kwargs here now correctly contain only the arguments intended for ParameterSweep.run
    results_list = sweep.run(full_sweep_grid, is_proc_sweep=True, **kwargs)

    return results_list
