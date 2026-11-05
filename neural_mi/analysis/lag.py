# neural_mi/analysis/lag.py
"""Implements the 'lag' analysis workflow for finding time-delayed relationships.

This module contains the `run_lag_analysis` function, which orchestrates the
process of estimating mutual information between two variables, X and Y, across
a range of specified time lags using the nonlinear cross-correlation method.
"""
from typing import List, Dict, Any, Optional

from neural_mi.analysis.sweep import ParameterSweep, _product_dict, merge_grid_values
import warnings
from neural_mi.logger import logger, user_stacklevel
from neural_mi.embeddings_io import with_model_labels
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

    from neural_mi.defaults import stream_processing
    proc_type_x = base_params.get('processor_type_x')
    params_x = base_params.get('processor_params_x') or {}
    proc_type_y, params_y = stream_processing(base_params.get('processor_type_y'),
                                              base_params.get('processor_params_y'), proc_type_x, params_x)

    # Infer sample_rate from processor_params to resolve unit ambiguity
    sample_rate = params_x.get('sample_rate', None)
    if sample_rate is None:
        sample_rate = (params_y or {}).get('sample_rate', None)

    # run() refuses a grid over 'lag', which the mode sets itself.
    other_sweep_grid = sweep_grid.copy() if sweep_grid is not None else {}

    param_combinations = _product_dict(**other_sweep_grid) if other_sweep_grid else [{}]

    logger.info(f"Preparing {len(lag_range) * len(param_combinations)} tasks for lag analysis.")
    if sample_rate is not None:
        logger.info(f"Lag units: seconds (sample_rate={sample_rate} Hz). "
                    f"Lag range [{min(lag_range)}, {max(lag_range)}]s = "
                    f"[{int(round(min(lag_range)*sample_rate))}, "
                    f"{int(round(max(lag_range)*sample_rate))}] samples.")
    else:
        logger.info("Lag units: samples (no sample_rate provided).")

    # Pre-compute shifted arrays for all lags to measure n_windows per lag,
    # then optionally equalize to the minimum across lags.
    shifted_pairs = {}
    for lag in lag_range:
        x_sh, y_sh = _shift_data(x_data, y_data, lag, proc_type_x, proc_type_y, sample_rate=sample_rate)
        shifted_pairs[lag] = (x_sh, y_sh)

    def _n_items(data):
        """Return the number of samples/trials, supporting both arrays and lists."""
        return data.shape[0] if hasattr(data, 'shape') else len(data)

    if equalize_n:
        # Determine minimum number of samples/windows across all lags
        min_n = min(_n_items(x_sh) for x_sh, _ in shifted_pairs.values())
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
        for i, other_params in enumerate(param_combinations):
            task_params = merge_grid_values(base_params, other_params)
            task_params.update({'lag': lag})
            task_params = with_model_labels(task_params, **other_params, lag=lag)
            run_id = other_params.get('run_id', f"lag{lag}_combo{i}")
            all_tasks.append((x_shifted, y_shifted, task_params, run_id))

    sweep_runner = ParameterSweep(x_data=None, y_data=None, base_params=base_params)
    # Every lag's task windows its data and reports the windows it trained on.
    return sweep_runner._run_parallel(all_tasks, n_workers=n_workers)