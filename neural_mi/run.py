# neural_mi/run.py
"""Provides the main `run` function, the primary entry point for the library.

This module orchestrates the entire analysis pipeline, from data validation
and preprocessing to model training and results aggregation. The `run` function
acts as a unified interface for all supported analysis modes.
"""
# Safe guard for macOS problems:
import platform
import os
import tempfile
from .logger import logger

# On macOS, PyTorch multiprocessing can have issues with the default temp directory.
# This creates a local, user-owned temp directory to avoid these issues.
if platform.system() == "Darwin":
    try:
        custom_temp_dir = os.path.expanduser('~/.neural_mi_tmp')
        os.makedirs(custom_temp_dir, exist_ok=True)
        
        # Set environment variables for all tempfile-related operations
        os.environ['TMPDIR'] = custom_temp_dir
        tempfile.tempdir = custom_temp_dir
        
        logger.debug(f"Set custom temporary directory for multiprocessing: {custom_temp_dir}")
    except Exception as e:
        logger.warning(f"Could not set custom temp directory: {e}. Using system default.")


# Actual run code
import pandas as pd
import numpy as np
import torch
from typing import Union, Optional, Dict, Any, List
import multiprocessing
import platform
import random

from .analysis.workflow import AnalysisWorkflow
from .analysis.dimensionality import run_dimensionality_analysis
from .analysis.lag import run_lag_analysis
from .data.handler import DataHandler
from .estimators import ESTIMATORS
from .results import Results
from .validation import ParameterValidator, DataValidator
from .logger import logger


try:
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    logger.debug("Multiprocessing start method already set.")

def _convert_mi_units(results: Any, to_bits: bool) -> Any:
    """Recursively converts MI values in results from nats to bits."""
    if not to_bits: return results
    NATS_TO_BITS = 1 / np.log(2)
    if isinstance(results, float): return results * NATS_TO_BITS
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
        cols = ['test_mi', 'train_mi', 'mi_mean', 'mi_std', 'mi_corrected', 'mi_error', 'slope']
        for col in cols:
            if col in df.columns: df[col] *= NATS_TO_BITS
        return df
    elif isinstance(results, list) and all(isinstance(r, dict) for r in results):
        keys = ['test_mi', 'train_mi', 'mi_corrected', 'mi_error', 'slope']
        return [{**r, **{k: r.get(k, 0) * NATS_TO_BITS for k in keys if r.get(k) is not None}} for r in results]
    elif isinstance(results, dict):
        new_results = results.copy()
        if 'corrected_results' in new_results:
            new_results['corrected_results'] = _convert_mi_units(new_results['corrected_results'], to_bits)
        if 'raw_results_df' in new_results:
            new_results['raw_results_df'] = _convert_mi_units(new_results['raw_results_df'], to_bits)
        return new_results
    return results

def run(
    x_data: Union[np.ndarray, torch.Tensor, List],
    y_data: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    mode: str = 'estimate',
    processor_type: Optional[str] = None,
    processor_params: Optional[Dict[str, Any]] = None,
    processor_type_x: Optional[str] = None,
    processor_params_x: Optional[Dict[str, Any]] = None,
    processor_type_y: Optional[str] = None,
    processor_params_y: Optional[Dict[str, Any]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    sweep_grid: Optional[Dict[str, list]] = None,
    output_units: str = 'bits',
    estimator: str = 'infonce',
    estimator_params: Optional[Dict[str, Any]] = None,
    custom_critic: Optional[torch.nn.Module] = None,
    custom_embedding_cls: Optional[type] = None,
    save_best_model_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    device: Optional[str] = None,
    delta_threshold: float = 0.1,
    min_gamma_points: int = 5,
    confidence_level: float = 0.68,
    **analysis_kwargs
) -> Results:
    
    """The unified entry point for all analyses in the NeuralMI library.
    
    This function provides a single, consistent interface for various mutual
    information estimation workflows. It handles data validation, processing,
    model training, and results aggregation, returning a standardized
    :class:`~neural_mi.results.Results` object that can be easily
    inspected and plotted.
    
    Parameters
    ----------
    x_data : np.ndarray, torch.Tensor, or list
        The data for variable X. The required format depends on ``processor_type``:
        
        - 'continuous' or 'categorical': A 2D array, typically of shape
          (n_channels, n_timepoints). Data of shape (n_timepoints, n_channels)
          is also supported and will be transposed automatically.
        - 'spike': A list of 1D NumPy arrays, where each array contains
          spike times for a single channel/neuron.
    y_data : np.ndarray, torch.Tensor, or list, optional
        The data for variable Y. Required for all modes except 'dimensionality'.
        Should have the same format as ``x_data``. Defaults to None.
    mode : {'estimate', 'sweep', 'dimensionality', 'rigorous', 'lag'}, default='estimate'
        The analysis mode to run.
    processor_type_x : {'continuous', 'spike', 'categorical'}, optional
        The type of processing to apply to raw X data. If None, data is assumed
        to be pre-processed. Defaults to None.
    processor_params_x : dict, optional
        Parameters for the X data processor, e.g., ``{'window_size': 10}``.
        Defaults to None.
    processor_type_y : {'continuous', 'spike', 'categorical'}, optional
        The type of processing to apply to raw Y data. If None, data is assumed
        to be pre-processed. Defaults to None.
    processor_params_y : dict, optional
        Parameters for the Y data processor, e.g., ``{'window_size': 10}``.
        Defaults to None.
    base_params : dict, optional
        A dictionary of fixed parameters for the MI estimator's trainer. These
        are used for all runs. Common parameters include ``n_epochs``,
        ``learning_rate``, ``batch_size``, ``embedding_dim``, etc. Defaults to {}.
    sweep_grid : dict, optional
        A dictionary defining the parameter grid for 'sweep' and
        'dimensionality' modes. Keys are parameter names and values are lists
        of values to test, e.g., ``{'embedding_dim': [8, 16, 32]}``.
        Defaults to None.
    output_units : {'bits', 'nats'}, default='bits'
        The units for the final MI estimate.
    estimator : {'infonce', 'nwj', 'tuba', 'smile', 'js'}, default='infonce'
        The MI lower bound to use for estimation.
    estimator_params : dict, optional
        Additional keyword arguments for the selected estimator function.
        For example, ``{'clip': 5.0}`` for the 'smile' estimator. Defaults to None.
    custom_critic : torch.nn.Module, optional
        A pre-initialized custom critic model. If provided, all internal model
        building is skipped. ``base_params`` related to model architecture will be
        ignored. Defaults to None.
    custom_embedding_cls : type, optional
        A user-defined embedding model class (not an instance) to be used
        within the library's standard critic structures. Defaults to None.
    save_best_model_path : str, optional
        If provided, the file path where the state dictionary of the
        best-performing trained critic model will be saved. Defaults to None.
    random_seed : int, optional
        A seed for ``random``, ``numpy``, and ``torch`` to ensure reproducibility.
        Note: Full reproducibility is only guaranteed for ``n_workers=1``.
        Defaults to None.
    verbose : bool, default=True
        If True, progress bars and informational logs will be displayed.
    device : str, optional
        The compute device to use (e.g., 'cpu', 'cuda', 'mps'). If None, it
        is auto-detected. Defaults to None.
    delta_threshold : float, default=0.1
        For ``mode='rigorous'``, the curvature threshold for determining the
        linear region of the MI vs. 1/gamma plot. Lower values enforce
        stricter linearity.
    min_gamma_points : int, default=5
        For ``mode='rigorous'``, the minimum number of gamma values required to
        perform a reliable extrapolation fit after pruning non-linear points.
    confidence_level : float, default=0.68
        For ``mode='rigorous'``, the confidence level (e.g., 0.68 for ~1 std
        dev) used for the final MI estimate's error bars.
    **analysis_kwargs
        Additional keyword arguments passed to the specific analysis engine.
        Common examples include ``n_workers``, ``n_splits``, or ``gamma_range``.
        For ``mode='lag'``, this must include ``lag_range``.
    
    Returns
    -------
    neural_mi.results.Results
        A standardized object containing the analysis results, which can be
        inspected as a dataframe or plotted directly via its ``.plot()`` method.
    
    Examples
    --------
    Perform a rigorous, bias-corrected MI estimation between two variables.
    
    >>> import neural_mi as nmi
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> x_raw, y_raw = nmi.datasets.generate_nonlinear_from_latent(
    ...     n_samples=2500, latent_dim=10, observed_dim=100, mi=3.0
    ... )
    >>> # Define model and training parameters
    >>> base_params = {
    ...     'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    ...     'embedding_dim': 16, 'hidden_dim': 64
    ... }
    >>> # Run the analysis
    >>> results = nmi.run(
    ...     x_data=x_raw.T, y_data=y_raw.T,
    ...     mode='rigorous',
    ...     processor_type_x='continuous',
    ...     processor_params_x={'window_size': 1},
    ...     base_params=base_params,
    ...     n_workers=4,
    ...     random_seed=42
    ... )
    >>> mi_est = results.mi_estimate
    >>> mi_err = results.details.get('mi_error', 0.0)
    >>> print(f"Corrected MI: {mi_est:.3f} ± {mi_err:.3f} bits")
    Corrected MI: 2.953 ± 0.081 bits
    
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if random_seed is not None and analysis_kwargs.get('n_workers', 1) is not None and analysis_kwargs.get('n_workers', 1) > 1:
        logger.warning("Reproducibility with random_seed is not guaranteed with n_workers > 1.")

    # Handle old and new processor parameter styles for backward compatibility
    if processor_type is not None:
        logger.warning("`processor_type` is deprecated. Use `processor_type_x` and `processor_type_y` instead.")
        processor_type_x = processor_type_x or processor_type
        processor_type_y = processor_type_y or processor_type
    if processor_params is not None:
        logger.warning("`processor_params` is deprecated. Use `processor_params_x` and `processor_params_y` instead.")
        processor_params_x = processor_params_x or processor_params
        processor_params_y = processor_params_y or processor_params

    ParameterValidator(locals()).validate()
    DataValidator(x_data, y_data, processor_type_x, processor_type_y).validate()

    if base_params is None: base_params = {}
    base_params['output_units'] = output_units
    base_params['verbose'] = verbose
    base_params['device'] = device
    base_params['estimator_name'] = estimator
    base_params['estimator_params'] = estimator_params or {}
    base_params['custom_critic'] = custom_critic
    base_params['custom_embedding_cls'] = custom_embedding_cls
    base_params['save_best_model_path'] = save_best_model_path

    run_params = {"mode": mode, "processor_type_x": processor_type_x, "processor_params_x": processor_params_x,
                  "processor_type_y": processor_type_y, "processor_params_y": processor_params_y,
                  "base_params": base_params, "sweep_grid": sweep_grid, "output_units": output_units,
                  "estimator": estimator, "random_seed": random_seed, "delta_threshold": delta_threshold,
                  "min_gamma_points": min_gamma_points, "confidence_level": confidence_level,
                  **analysis_kwargs}

    processor_param_keys = ['window_size', 'step_size', 'n_seconds', 'max_spikes_per_window', 'data_format']
    is_proc_sweep = mode == 'sweep' and any(key in (sweep_grid or {}) for key in processor_param_keys)
    
    handler = DataHandler(x_data, y_data, processor_type_x, processor_params_x, processor_type_y, processor_params_y)

    if is_proc_sweep or mode == 'lag':
        logger.info("Detected sweep over processor or lag parameters. Deferring data processing to workers.")
        base_params['processor_type_x'] = processor_type_x
        base_params['processor_params_x'] = processor_params_x
        base_params['processor_type_y'] = processor_type_y
        base_params['processor_params_y'] = processor_params_y
        # Use handler's raw data for deferred processing
        x_run_data, y_run_data = handler.x_data, handler.y_data
    else:
        if mode == 'dimensionality':
            if y_data is not None: logger.warning("y_data is ignored for mode 'dimensionality'.")
            x_run_data, _ = handler.process()
            y_run_data = None # y_data is not used in this mode
        else:
            if y_data is None: raise ValueError(f"y_data must be provided for mode '{mode}'.")
            x_run_data, y_run_data = handler.process()

    from .analysis.sweep import ParameterSweep
    if mode == 'sweep':
        results_list = ParameterSweep(x_run_data, y_run_data, base_params).run(
            sweep_grid, is_proc_sweep=is_proc_sweep, **analysis_kwargs
        )
        df = pd.DataFrame(results_list)
        group_vars = [key for key in sweep_grid.keys() if key != 'run_id']
        agg_df = df.groupby(group_vars)['test_mi'].agg(['mean', 'std']).reset_index().rename(columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0) if group_vars else df
        primary_sweep_var = group_vars[0] if group_vars else None
        return Results(mode=mode, dataframe=_convert_mi_units(agg_df, output_units == 'bits'), params={**run_params, 'sweep_var': primary_sweep_var}, details={'raw_results': df})

    elif mode == 'estimate':
        results_list = ParameterSweep(x_run_data, y_run_data, base_params).run(sweep_grid or {}, **analysis_kwargs)
        mi = results_list[0]['test_mi'] if results_list else float('nan')
        return Results(mode=mode, mi_estimate=_convert_mi_units(mi, output_units == 'bits'), params=run_params)

    elif mode == 'dimensionality':
        from .utils import find_saturation_point
        df = run_dimensionality_analysis(x_run_data, base_params, sweep_grid, **analysis_kwargs)
        df = _convert_mi_units(df, output_units == 'bits')
        dims = find_saturation_point(df, strictness=analysis_kwargs.get('strictness', [0.1, 1.0, 15.0]))
        return Results(mode=mode, dataframe=df, params={**run_params, 'sweep_var': 'embedding_dim'}, details={'estimated_dims': dims})

    elif mode == 'rigorous':
        analysis_kwargs.update({'delta_threshold': delta_threshold, 'min_gamma_points': min_gamma_points, 'confidence_level': confidence_level})
        results = AnalysisWorkflow(x_run_data, y_run_data, base_params).run(sweep_grid or {}, **analysis_kwargs)
        results = _convert_mi_units(results, output_units == 'bits')
        corrected_list = results.get('corrected_results', [])
        details = corrected_list[0] if corrected_list else {}
        return Results(mode=mode, mi_estimate=details.get('mi_corrected'), dataframe=results.get('raw_results_df'), details=details, params=run_params)
    
    elif mode == 'lag':
        if 'lag_range' not in analysis_kwargs:
            raise ValueError("`lag_range` must be provided for mode='lag'.")
        results_list = run_lag_analysis(x_run_data, y_run_data, base_params, **analysis_kwargs)
        df = pd.DataFrame(results_list)
        agg_df = df.groupby('lag')['test_mi'].agg(['mean', 'std']).reset_index().rename(columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0)
        return Results(mode=mode, dataframe=_convert_mi_units(agg_df, output_units == 'bits'), params={**run_params, 'sweep_var': 'lag'}, details={'raw_results': df})

    else:
        raise ValueError(f"Unknown mode: '{mode}'.")