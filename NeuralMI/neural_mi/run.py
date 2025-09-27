# neural_mi/run.py

import pandas as pd
import warnings
from .analysis.sweep import ParameterSweep
from .analysis.workflow import AnalysisWorkflow
from .analysis.dimensionality import run_dimensionality_analysis

def run(
    x_data,
    y_data=None,
    mode='estimate',
    base_params=None,
    sweep_grid=None,
    **kwargs
):
    """
    The unified entry point for all analyses in the NeuralMI library.

    Args:
        x_data (torch.Tensor): The processed data for variable X.
        y_data (torch.Tensor, optional): The processed data for variable Y. Defaults to None.
        mode (str): The analysis mode to run. One of:
            - 'estimate': A single, quick MI estimate for a fixed set of hyperparameters.
            - 'sweep': An exploratory sweep over a grid of hyperparameters.
            - 'dimensionality': Internal information analysis of a single variable X.
            - 'rigorous': The full, bias-corrected MI estimation workflow.
        base_params (dict): A dictionary of fixed parameters for the Trainer.
        sweep_grid (dict, optional): A dictionary defining the parameter grid for 'sweep'
            and 'dimensionality' modes.
        **kwargs: Additional keyword arguments passed to the specific analysis engine
            (e.g., n_workers, critic_type, gamma_range, confidence_level).

    Returns:
        The results of the analysis, format depends on the mode.
    """
    if base_params is None:
        raise ValueError("'base_params' must be provided.")

    # --- Separate kwargs for constructors vs. run methods ---
    init_kwargs = {
        'critic_type': kwargs.pop('critic_type', 'separable'),
        'estimator_fn': kwargs.pop('estimator_fn', None),
        'use_variational': kwargs.pop('use_variational', False)
    }
    # Remove None values so defaults in constructors are used
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    
    # The rest of kwargs (like n_workers, gamma_range) are for the .run() methods

    if mode == 'estimate':
        if y_data is None: raise ValueError("y_data must be provided for mode 'estimate'.")
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        results = sweep.run(sweep_grid=sweep_grid or {}, **kwargs)
        return results[0]['test_mi'] if results else None

    elif mode == 'sweep':
        if y_data is None: raise ValueError("y_data must be provided for mode 'sweep'.")
        if sweep_grid is None: raise ValueError("A 'sweep_grid' must be provided for mode 'sweep'.")
        
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        return pd.DataFrame(sweep.run(sweep_grid=sweep_grid, **kwargs))

    elif mode == 'dimensionality':
        if y_data is not None: warnings.warn("y_data is ignored for mode 'dimensionality'.")
        if sweep_grid is None or 'embedding_dim' not in sweep_grid:
            raise ValueError("A 'sweep_grid' with 'embedding_dim' is required for mode 'dimensionality'.")
            
        # Dimensionality has its own kwargs, so we pass the original dict
        return run_dimensionality_analysis(
            x_data=x_data, base_params=base_params, sweep_grid=sweep_grid, **kwargs
        )

    elif mode == 'rigorous':
        if y_data is None: raise ValueError("y_data must be provided for mode 'rigorous'.")

        workflow = AnalysisWorkflow(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        return workflow.run(param_grid=sweep_grid or {}, **kwargs)

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of 'estimate', 'sweep', 'dimensionality', 'rigorous'.")