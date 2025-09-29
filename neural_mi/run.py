# neural_mi/run.py

import pandas as pd
import numpy as np
import warnings
import torch
from typing import Union

# --- Imports for the worker function ---
from .analysis.workflow import AnalysisWorkflow
from .analysis.dimensionality import run_dimensionality_analysis
from .data.handler import DataHandler
from .estimators import ESTIMATORS
from .results import Results

# _convert_mi_units remains the same...
def _convert_mi_units(results, to_bits):
    # ... (no change)
    if not to_bits: return results
    NATS_TO_BITS = 1 / np.log(2)
    if isinstance(results, float): return results * NATS_TO_BITS
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
        cols_to_convert = ['test_mi', 'train_mi', 'mi_mean', 'mi_std', 'mi_corrected', 'mi_error', 'slope']
        for col in cols_to_convert:
            if col in df.columns: df[col] *= NATS_TO_BITS
        return df
        
    elif isinstance(results, list) and all(isinstance(r, dict) for r in results):
        new_results = []
        keys_to_convert = ['test_mi', 'train_mi', 'mi_corrected', 'mi_error', 'slope']
        for res_dict in results:
            new_dict = res_dict.copy()
            for key in keys_to_convert:
                if key in new_dict and new_dict[key] is not None: new_dict[key] *= NATS_TO_BITS
            new_results.append(new_dict)
        return new_results
    
    elif isinstance(results, dict):
        new_results = results.copy()
        if 'corrected_results' in new_results:
            new_results['corrected_results'] = _convert_mi_units(new_results['corrected_results'], to_bits)
        if 'raw_results_df' in new_results:
            new_results['raw_results_df'] = _convert_mi_units(new_results['raw_results_df'], to_bits)
        return new_results
        
    return results


def run(
    x_data: Union[np.ndarray, torch.Tensor],
    y_data: Union[np.ndarray, torch.Tensor] = None,
    mode: str = 'estimate',
    processor_type: str = None,
    processor_params: dict = None,
    base_params: dict = None,
    sweep_grid: dict = None,
    output_units: str = 'bits',
    estimator: str = 'infonce',
    custom_embedding_model=None,
    save_best_model_path: str = None,
    **analysis_kwargs
) -> Results:
    """The unified entry point for all analyses in the NeuralMI library.

    This function provides a single, consistent interface for various mutual
    information estimation workflows. It handles data processing, model
    training, and analysis, returning a standardized `Results` object.

    Parameters
    ----------
    x_data : np.ndarray or torch.Tensor
        The data for variable X. Can be raw data (e.g., 2D array of shape
        `(n_timepoints, n_channels)` for continuous data) or pre-processed
        3D data of shape `(n_samples, n_channels, n_features)`.
    y_data : np.ndarray or torch.Tensor, optional
        The data for variable Y. Required for all modes except 'dimensionality'.
        Should have the same format as `x_data`.
    mode : {'estimate', 'sweep', 'dimensionality', 'rigorous'}, default='estimate'
        The analysis mode to run:
        - 'estimate': A single, quick MI estimate.
        - 'sweep': An exploratory sweep over a grid of hyperparameters.
        - 'dimensionality': Internal information analysis of a single variable X.
        - 'rigorous': The full, bias-corrected MI estimation workflow.
    processor_type : {'continuous', 'spike'}, optional
        The type of processing to apply if `x_data` and `y_data` are raw.
        - 'continuous': Treats data as a continuous time-series and applies
          windowing. Assumes shape `(n_timepoints, n_channels)`.
        - 'spike': Treats data as spike times.
        If None, data is assumed to be pre-processed (3D).
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
    **analysis_kwargs : dict
        Additional keyword arguments passed to the specific analysis engine,
        such as `n_workers`, `gamma_range`, or `confidence_level`.

    Returns
    -------
    Results
        A standardized object containing the analysis results. Key attributes:
        - `mi_estimate`: The final MI point estimate (for 'estimate' and
          'rigorous' modes).
        - `dataframe`: A pandas DataFrame with detailed results (for 'sweep',
          'dimensionality', and the raw runs of 'rigorous' mode).
        - `details`: A dictionary with mode-specific metadata.
        - `plot()`: A method to generate a context-appropriate plot.

    """
    if output_units not in ['bits', 'nats']: raise ValueError("output_units must be 'bits' or 'nats'.")
    if base_params is None: raise ValueError("'base_params' must be provided.")
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator: '{estimator}'. Must be one of {list(ESTIMATORS.keys())}")

    # Store all parameters for logging in the Results object
    run_params = {
        "mode": mode,
        "processor_type": processor_type,
        "processor_params": processor_params,
        "base_params": base_params,
        "sweep_grid": sweep_grid,
        "output_units": output_units,
        "estimator": estimator,
        **analysis_kwargs
    }

    # --- 1. Centralized Data Processing ---
    if mode == 'dimensionality':
        if y_data is not None:
            warnings.warn("y_data is ignored for mode 'dimensionality'.")
        handler = DataHandler(x_data, x_data, processor_type, processor_params)
        x_data, _ = handler.process()
    else:
        if y_data is None:
            raise ValueError(f"y_data must be provided for mode '{mode}'.")
        handler = DataHandler(x_data, y_data, processor_type, processor_params)
        x_data, y_data = handler.process()

    # --- 2. Prepare analysis-specific parameters ---
    init_kwargs = {
        'critic_type': analysis_kwargs.pop('critic_type', 'separable'),
        'estimator_fn': ESTIMATORS[estimator],
        'use_variational': analysis_kwargs.pop('use_variational', False),
        'custom_embedding_model': custom_embedding_model,
        'save_best_model_path': save_best_model_path
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    to_bits = output_units == 'bits'

    from .analysis.sweep import ParameterSweep

    # --- 3. Dispatch to the correct analysis mode ---
    if mode == 'sweep':
        if sweep_grid is None: raise ValueError("A 'sweep_grid' must be provided for mode 'sweep'.")
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        results_list = sweep.run(sweep_grid=sweep_grid, **analysis_kwargs)
        results_df = _convert_mi_units(pd.DataFrame(results_list), to_bits)
        run_params['sweep_var'] = list(sweep_grid.keys())[0] if sweep_grid else None
        return Results(mode=mode, dataframe=results_df, params=run_params)

    elif mode == 'estimate':
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        results_list = sweep.run(sweep_grid=sweep_grid or {}, **analysis_kwargs)
        final_mi = results_list[0]['test_mi'] if results_list else float('nan')
        final_mi = _convert_mi_units(final_mi, to_bits)
        return Results(mode=mode, mi_estimate=final_mi, params=run_params)

    elif mode == 'dimensionality':
        if sweep_grid is None or 'embedding_dim' not in sweep_grid:
            raise ValueError("A 'sweep_grid' with 'embedding_dim' is required for mode 'dimensionality'.")
        results_df = run_dimensionality_analysis(x_data=x_data, base_params=base_params, sweep_grid=sweep_grid, **analysis_kwargs)
        results_df = _convert_mi_units(results_df, to_bits)
        run_params['sweep_var'] = 'embedding_dim'
        return Results(mode=mode, dataframe=results_df, params=run_params)

    elif mode == 'rigorous':
        workflow = AnalysisWorkflow(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        results_dict = workflow.run(param_grid=sweep_grid or {}, **analysis_kwargs)
        results_dict = _convert_mi_units(results_dict, to_bits)

        raw_df = results_dict.get('raw_results_df')
        # Handle cases with multiple sweep results in rigorous mode, take the first.
        corrected_list = results_dict.get('corrected_results', [{}])
        details = corrected_list[0] if corrected_list else {}
        mi_estimate = details.get('mi_corrected')

        return Results(
            mode=mode,
            mi_estimate=mi_estimate,
            dataframe=raw_df,
            details=details,
            params=run_params
        )

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of 'estimate', 'sweep', 'dimensionality', 'rigorous'.")