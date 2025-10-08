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
    **kwargs
) -> List[Dict[str, Any]]:
    """Estimates mutual information across a range of time lags.

    For each lag in `lag_range`, this function shifts `y_data` relative to `x_data`
    and then runs a standard MI estimation. This is useful for determining the
    timescale of predictive information between two variables.

    Parameters
    ----------
    x_data : Any
        The raw data for variable X.
    y_data : Any
        The raw data for variable Y.
    base_params : Dict[str, Any]
        A dictionary of fixed parameters for the MI estimator's trainer.
    lag_range : range
        A range of integer lags (for continuous/categorical) or float lags in
        seconds (for spike) to test.
    **kwargs : dict
        Additional keyword arguments passed to the `ParameterSweep`.

    Returns
    -------
    List[Dict[str, Any]]
        A list of result dictionaries, one for each lag, containing the
        MI estimate and the lag value.
    """
    sweep_grid = {'lag': list(lag_range)}

    # For lag analysis, data processing is always deferred to workers.
    # The necessary processor params should already be in base_params.
    sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params)
    
    results_list = sweep.run(sweep_grid, is_proc_sweep=True, **kwargs)
    
    return results_list