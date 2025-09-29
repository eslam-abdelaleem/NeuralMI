# neural_mi/run.py

import pandas as pd
import numpy as np
import warnings
import torch
import itertools
import uuid
from multiprocessing import Pool, cpu_count

# --- Imports for the worker function ---
import torch.optim as optim
from .analysis.sweep import _build_critic # Re-use the critic builder
from .training.trainer import Trainer
from .analysis.workflow import AnalysisWorkflow
from .analysis.dimensionality import run_dimensionality_analysis
from .data.processors import ContinuousProcessor, SpikeProcessor
from .estimators import ESTIMATORS

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

# --- A dedicated parallel worker for processing sweeps ---
def _processing_sweep_worker(args):
    """A top-level function that can be pickled for multiprocessing."""
    (raw_x, raw_y, params, base_params, init_kwargs, run_id) = args
    
    # 1. Process the data for this specific task
    proc_type = params.pop('processor_type')
    
    if proc_type == 'continuous':
        proc = ContinuousProcessor(window_size=params.get('window_size'), 
                                   step_size=params.get('step_size', 1))
        x_processed = proc.process(raw_x.numpy() if isinstance(raw_x, torch.Tensor) else raw_x)
        y_processed = proc.process(raw_y.numpy() if isinstance(raw_y, torch.Tensor) else raw_y)
    elif proc_type == 'spike':
        proc = SpikeProcessor(window_size=params.get('window_size'),
                              step_size=params.get('step_size', 1),
                              max_spikes_per_window=params.get('max_spikes_per_window'))
        # Spike processor expects a list of arrays
        x_processed = proc.process(list(raw_x))
        y_processed = proc.process(list(raw_y))
    else:
        raise NotImplementedError(f"Processor type {proc_type} not implemented for sweeps.")
        
        
    # 2. Build the final parameter set for the trainer
    task_params = base_params.copy()
    task_params.update(init_kwargs)
    task_params.update(params) # Add sweep-specific params
    task_params['input_dim_x'] = x_processed.shape[1] * x_processed.shape[2]
    task_params['input_dim_y'] = y_processed.shape[1] * y_processed.shape[2]
    
    # 3. Build model and trainer (similar to _run_training_task)
    use_variational = task_params.get('use_variational', False)
    critic = _build_critic(task_params['critic_type'], task_params, use_variational,
                           custom_embedding_model=task_params.get('custom_embedding_model'))
    optimizer = optim.Adam(critic.parameters(), lr=task_params['learning_rate'])
    trainer = Trainer(
        model=critic, estimator_fn=task_params['estimator_fn'], optimizer=optimizer,
        use_variational=use_variational, beta=task_params.get('beta', 1.0)
    )
    
    # 4. Run training
    results = trainer.train(
        x_data=x_processed, y_data=y_processed, n_epochs=task_params['n_epochs'],
        batch_size=task_params['batch_size'], patience=task_params['patience'], run_id=run_id,
        save_best_model_path=task_params.get('save_best_model_path')
    )
    
    return {**params, **results}


def run(
    x_data,
    y_data=None,
    mode='estimate',
    base_params=None,
    sweep_grid=None,
    output_units='bits',
    estimator='infonce',
    custom_embedding_model=None,
    save_best_model_path=None,
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
        estimator (str): The MI bound to use ('infonce', 'nwj', 'tuba', 'smile').
        custom_embedding_model (torch.nn.Module, optional): A user-defined embedding model class.
        save_best_model_path (str, optional): Path to save the best trained critic model.
        **kwargs: Additional keyword arguments passed to the specific analysis engine
            (e.g., n_workers, critic_type, gamma_range, confidence_level).

    Returns:
        The results of the analysis, format depends on the mode.
    """
    if output_units not in ['bits', 'nats']: raise ValueError("output_units must be 'bits' or 'nats'.")
    if base_params is None: raise ValueError("'base_params' must be provided.")
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator: '{estimator}'. Must be one of {list(ESTIMATORS.keys())}")
    
    init_kwargs = {
        'critic_type': kwargs.pop('critic_type', 'separable'),
        'estimator_fn': ESTIMATORS[estimator],
        'use_variational': kwargs.pop('use_variational', False),
        'custom_embedding_model': custom_embedding_model,
        'save_best_model_path': save_best_model_path
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    
    to_bits = output_units == 'bits'

    # We need to import the ParameterSweep class here to avoid circular imports at the top level
    from .analysis.sweep import ParameterSweep
    
    if mode == 'sweep':
        if y_data is None: raise ValueError("y_data must be provided for mode 'sweep'.")
        if sweep_grid is None: raise ValueError("A 'sweep_grid' must be provided for mode 'sweep'.")
        
        proc_type = kwargs.get('processor_type')
        
        # Fully Parallel Logic for Processing Sweeps ---
        if proc_type and ('window_size' in sweep_grid or 'max_spikes' in sweep_grid):
            keys, values = zip(*sweep_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            tasks = []
            run_id_base = str(uuid.uuid4())
            for i, params in enumerate(param_combinations):
                params['processor_type'] = proc_type # Add proc_type to each task's params
                task_run_id = f"{run_id_base}_c{i}"
                tasks.append((x_data, y_data, params, base_params, init_kwargs, task_run_id))
            
            print(f"Created {len(tasks)} parallel processing & training tasks...")
            n_workers = kwargs.get('n_workers', cpu_count())
            with Pool(processes=n_workers) as pool:
                all_results = list(pool.map(_processing_sweep_worker, tasks))
            
            results = pd.DataFrame(all_results)
        
        # --- Standard model parameter sweep on pre-processed data ---
        else:
            sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
            results = pd.DataFrame(sweep.run(sweep_grid=sweep_grid, **kwargs))
            
        return _convert_mi_units(results, to_bits)

    # --- Other modes remain the same ---
    if mode == 'estimate':
        if y_data is None: raise ValueError("y_data must be provided for mode 'estimate'.")
        # Estimate mode now requires pre-processed data
        if x_data.ndim != 3:
             raise ValueError("'estimate' mode requires pre-processed 3D data. Use a processor first.")
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        results = sweep.run(sweep_grid=sweep_grid or {}, **kwargs)
        final_mi = results[0]['test_mi'] if results else float('nan')
        return _convert_mi_units(final_mi, to_bits)

    elif mode == 'dimensionality':
        if y_data is not None: warnings.warn("y_data is ignored for mode 'dimensionality'.")
        if sweep_grid is None or 'embedding_dim' not in sweep_grid:
            raise ValueError("A 'sweep_grid' with 'embedding_dim' is required for mode 'dimensionality'.")
        results = run_dimensionality_analysis(x_data=x_data, base_params=base_params, sweep_grid=sweep_grid, **kwargs)
        return _convert_mi_units(results, to_bits)

    elif mode == 'rigorous':
        if y_data is None: raise ValueError("y_data must be provided for mode 'rigorous'.")
        workflow = AnalysisWorkflow(x_data=x_data, y_data=y_data, base_params=base_params, **init_kwargs)
        results = workflow.run(param_grid=sweep_grid or {}, **kwargs)
        return _convert_mi_units(results, to_bits)

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of 'estimate', 'sweep', 'dimensionality', 'rigorous'.")