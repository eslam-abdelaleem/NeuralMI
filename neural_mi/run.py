# neural_mi/run.py

import pandas as pd
import numpy as np
import torch
from typing import Union, Optional, Dict, Any, List
import multiprocessing
import platform
import random

from .analysis.workflow import AnalysisWorkflow
from .analysis.dimensionality import run_dimensionality_analysis
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
    base_params: Optional[Dict[str, Any]] = None,
    sweep_grid: Optional[Dict[str, list]] = None,
    output_units: str = 'bits',
    estimator: str = 'infonce',
    custom_embedding_model: Optional[torch.nn.Module] = None,
    save_best_model_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    device: Optional[str] = None,
    **analysis_kwargs
) -> Results:
    """The unified entry point for all analyses in the NeuralMI library.

    This function provides a single, consistent interface for various mutual
    information estimation workflows. It handles data processing, model
    training, and analysis, returning a standardized `Results` object.

    Parameters
    ----------
    x_data : np.ndarray, torch.Tensor, or list
        The data for variable X. The required format depends on `processor_type`:
        - For 'continuous': A 2D array of shape `(n_channels, n_timepoints)`.
        - For 'spike': A list of 1D NumPy arrays, where each array contains spike times for a neuron.
        - If `processor_type` is None: Pre-processed 3D data of shape `(n_samples, n_channels, n_features)`.
    y_data : np.ndarray, torch.Tensor, or list, optional
        The data for variable Y. Required for all modes except 'dimensionality'.
        Should have the same format as `x_data`.
    mode : {'estimate', 'sweep', 'dimensionality', 'rigorous'}, default='estimate'
        The analysis mode to run:
        - 'estimate': A single, quick MI estimate.
        - 'sweep': An exploratory sweep over a grid of hyperparameters.
        - 'dimensionality': Internal information analysis of a single variable X.
        - 'rigorous': The full, bias-corrected MI estimation workflow.
    processor_type : {'continuous', 'spike'}, optional
        The type of processing to apply to raw data. If None, data is assumed to be pre-processed (3D).
    processor_params : dict, optional
        Parameters for the data processor, e.g., `{'window_size': 10}`.
    base_params : dict
        A dictionary of fixed parameters for the MI estimator's trainer, such as
        `n_epochs`, `learning_rate`, `batch_size`, `embedding_dim`, etc.
    sweep_grid : dict, optional
        A dictionary defining the parameter grid for 'sweep' and
        'dimensionality' modes, e.g., `{'embedding_dim': [8, 16, 32]}`.
    output_units : {'bits', 'nats'}, default='bits'
        The units for the final MI estimate.
    estimator : {'infonce', 'nwj', 'tuba', 'smile'}, default='infonce'
        The MI bound to use for estimation.
    custom_embedding_model : torch.nn.Module, optional
        A user-defined embedding model class (subclass of `BaseEmbedding`).
    save_best_model_path : str, optional
        If provided, the path to save the best-performing trained critic model.
    random_seed : int, optional
        A seed for random number generators to ensure reproducibility.
        Note: Full reproducibility requires `n_workers=1`.
    **analysis_kwargs : dict
        Additional keyword arguments passed to the specific analysis engine,
        such as `n_workers`, `gamma_range`, or `confidence_level`.

    Returns
    -------
    Results
        A standardized object containing the analysis results.

    Examples
    --------
    >>> import neural_mi as nmi
    >>> import numpy as np
    >>> base_params = {
    ...     'n_epochs': 20, 'learning_rate': 1e-3, 'batch_size': 64,
    ...     'patience': 5, 'embedding_dim': 16, 'hidden_dim': 64, 'n_layers': 2
    ... }

    **1. Quick Estimate with Continuous Data**
    >>> x, y = nmi.datasets.generate_correlated_gaussians(500, 5, 2.0)
    >>> results = nmi.run(
    ...     x_data=x.T, y_data=y.T, # Shape (channels, samples)
    ...     mode='estimate',
    ...     processor_type='continuous',
    ...     processor_params={'window_size': 1},
    ...     base_params=base_params,
    ...     random_seed=42
    ... )
    >>> print(f"Estimated MI: {results.mi_estimate:.3f} bits")

    **2. Sweep over Window Sizes**
    >>> x_t, y_t = nmi.datasets.generate_temporally_convolved_data(2000)
    >>> sweep_grid = {'window_size': [10, 50, 100]}
    >>> results_sweep = nmi.run(
    ...     x_data=x_t, y_data=y_t,
    ...     mode='sweep',
    ...     processor_type='continuous',
    ...     base_params=base_params,
    ...     sweep_grid=sweep_grid,
    ...     n_workers=2,
    ...     random_seed=42
    ... )
    >>> # results_sweep.plot()

    **3. Latent Dimensionality**
    >>> x_l, _ = nmi.datasets.generate_nonlinear_from_latent(1000, 4, 50, 3.0)
    >>> sweep_grid_dim = {'embedding_dim': [2, 4, 6, 8]}
    >>> results_dim = nmi.run(
    ...     x_data=x_l.T,
    ...     mode='dimensionality',
    ...     processor_type='continuous',
    ...     processor_params={'window_size': 1},
    ...     base_params=base_params,
    ...     sweep_grid=sweep_grid_dim,
    ...     n_splits=3,
    ...     random_seed=42
    ... )
    >>> print(results_dim.details['estimated_dims'])
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

    ParameterValidator(locals()).validate()
    DataValidator(x_data, y_data, processor_type).validate()

    if base_params is None: base_params = {}
    base_params['output_units'] = output_units
    base_params['verbose'] = verbose
    base_params['device'] = device

    run_params = {"mode": mode, "processor_type": processor_type, "processor_params": processor_params,
                  "base_params": base_params, "sweep_grid": sweep_grid, "output_units": output_units,
                  "estimator": estimator, "random_seed": random_seed, **analysis_kwargs}

    processor_param_keys = ['window_size', 'step_size', 'n_seconds', 'max_spikes_per_window', 'data_format']
    is_proc_sweep = mode == 'sweep' and any(key in (sweep_grid or {}) for key in processor_param_keys)

    if is_proc_sweep:
        logger.info("Detected sweep over processor parameters. Deferring data processing to workers.")
        base_params['processor_type'] = processor_type
        base_params['processor_params'] = processor_params
    else:
        if mode == 'dimensionality':
            if y_data is not None: logger.warning("y_data is ignored for mode 'dimensionality'.")
            x_data, _ = DataHandler(x_data, x_data, processor_type, processor_params).process()
        else:
            if y_data is None: raise ValueError(f"y_data must be provided for mode '{mode}'.")
            x_data, y_data = DataHandler(x_data, y_data, processor_type, processor_params).process()

    init_kwargs = {'critic_type': analysis_kwargs.pop('critic_type', 'separable'), 'estimator_name': estimator,
                   'use_variational': analysis_kwargs.pop('use_variational', False),
                   'custom_embedding_model': custom_embedding_model, 'save_best_model_path': save_best_model_path}

    from .analysis.sweep import ParameterSweep
    if mode == 'sweep':
        results_list = ParameterSweep(x_data, y_data, base_params, **init_kwargs).run(
            sweep_grid, is_proc_sweep=is_proc_sweep, **analysis_kwargs
        )
        df = pd.DataFrame(results_list)
        
        group_vars = [key for key in sweep_grid.keys() if key != 'run_id']
        if not group_vars:
             agg_df = df
        else:
            agg_df = df.groupby(group_vars)['test_mi'].agg(['mean', 'std']).reset_index().rename(columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0)
        
        primary_sweep_var = group_vars[0] if group_vars else None
        
        return Results(mode=mode, dataframe=_convert_mi_units(agg_df, output_units == 'bits'), params={**run_params, 'sweep_var': primary_sweep_var}, details={'raw_results': df})

    elif mode == 'estimate':
        results_list = ParameterSweep(x_data, y_data, base_params, **init_kwargs).run(sweep_grid or {}, **analysis_kwargs)
        mi = results_list[0]['test_mi'] if results_list else float('nan')
        return Results(mode=mode, mi_estimate=_convert_mi_units(mi, output_units == 'bits'), params=run_params)

    elif mode == 'dimensionality':
        from .utils import find_saturation_point
        df = run_dimensionality_analysis(x_data, base_params, sweep_grid, **analysis_kwargs)
        df = _convert_mi_units(df, output_units == 'bits')
        dims = find_saturation_point(df, strictness=analysis_kwargs.get('strictness', [0.1, 1.0, 15.0]))
        return Results(mode=mode, dataframe=df, params={**run_params, 'sweep_var': 'embedding_dim'}, details={'estimated_dims': dims})

    elif mode == 'rigorous':
        results = AnalysisWorkflow(x_data, y_data, base_params, **init_kwargs).run(sweep_grid or {}, **analysis_kwargs)
        results = _convert_mi_units(results, output_units == 'bits')
        corrected_list = results.get('corrected_results', [])
        details = corrected_list[0] if corrected_list else {}
        return Results(mode=mode, mi_estimate=details.get('mi_corrected'), dataframe=results.get('raw_results_df'), details=details, params=run_params)
    else:
        raise ValueError(f"Unknown mode: '{mode}'.")