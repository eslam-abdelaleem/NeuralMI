# neural_mi/analysis/sweep.py
"""Provides the ParameterSweep class for running hyperparameter sweeps.

This module defines the core logic for executing multiple training runs in
parallel across a grid of hyperparameters.
"""
import torch
import itertools
import uuid
import os
import torch.multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger
from neural_mi.utils import _configure_multiprocessing

def _product_dict(**kwargs: Dict[str, List]) -> List[Dict[str, Any]]:
    """Helper to create a list of dictionaries from a grid."""
    keys = kwargs.keys()
    vals = kwargs.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]

class ParameterSweep:
    """Manages the execution of a hyperparameter sweep.

    This class prepares and distributes training tasks across multiple processes
    to efficiently explore a grid of hyperparameters.
    """
    def __init__(self, x_data, y_data, base_params, **kwargs):
        """
        Parameters
        ----------
        dataset : PairedDataset
            A PairedDataset or PairedTemporalDataset object containing the data.
        base_params : Dict[str, Any]
            A dictionary of fixed parameters for the MI estimator's trainer.
        **kwargs : Dict[str, Any]
            Additional keyword arguments to be added to `base_params`.
        """
        self.x_data, self.y_data = x_data, y_data
        self.base_params = base_params
        
        # If data is already a tensor (processed), we can infer dimensions
        if isinstance(x_data, torch.Tensor) and x_data.ndim == 3:
            self.base_params.update({
                'input_dim_x': x_data.shape[1] * x_data.shape[2],
                'input_dim_y': y_data.shape[1] * y_data.shape[2] if y_data is not None else 0,
                'n_channels_x': x_data.shape[1],
                'n_channels_y': y_data.shape[1] if y_data is not None else 0,
                **kwargs
            })
        else:
             self.base_params.update(kwargs)

    def _run_parallel(self, tasks: List[tuple], n_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes a list of prepared tasks in parallel.
        """
        if not tasks:
            logger.warning("No tasks to run. Your sweep_grid might be empty.")
            return []

        # Default to sequential if n_workers is not specified or is 1
        effective_workers = n_workers if n_workers is not None else 1

        show_progress = self.base_params.get('show_progress', True)

        if effective_workers <= 1:
            logger.info("Starting parameter sweep sequentially (n_workers=1)...")
            all_results = [run_training_task(task) for task in tqdm(tasks, desc="Sequential Sweep Progress", disable=not show_progress)]
        else:
            logger.info(f"Starting parameter sweep with {effective_workers} workers...")
            _configure_multiprocessing()
            with mp.get_context("spawn").Pool(processes=effective_workers) as pool:
                all_results = list(tqdm(
                    pool.imap(run_training_task, tasks), total=len(tasks),
                    desc="Parameter Sweep Progress", unit="task", disable=not show_progress
                ))
        return all_results
    
    def _prepare_tasks(
        self,
        sweep_grid: Dict[str, List],
        is_proc_sweep: bool,
        max_samples_per_task: Optional[int],
        **kwargs,
    ) -> List[tuple]:
        """Prepares the tasks for the parameter sweep."""
        tasks = []
        run_id_base = str(uuid.uuid4())
        sweep_grid = sweep_grid or {}

        if self.base_params.get('critic_type') == 'concat' and 'embedding_dim' in sweep_grid:
            logger.warning("'embedding_dim' is not applicable for ConcatCritic and will be ignored.")
            sweep_grid.pop('embedding_dim', None)

        param_combinations = _product_dict(**sweep_grid) if sweep_grid else [{}]

        for i_combo, params in enumerate(param_combinations):
            current_params = {**self.base_params, **params}
            
            # --- SMART MODEL SAVING LOGIC ---
            base_save_path = current_params.get('save_best_model_path')
            if base_save_path and params:
                root, ext = os.path.splitext(base_save_path)
                # Create a clean suffix from the parameters being swept
                suffix = "_" + "_".join([f"{str(k)}_{str(v)}" for k, v in params.items()])
                # Remove spaces or problematic characters if any exist in the values
                suffix = suffix.replace(" ", "")
                current_params['save_best_model_path'] = f"{root}{suffix}{ext}"
            # --------------------------------
            
            # Initialize from base_params, then update from kwargs (if any), then sweep params.
            # Only inject keys that belong to the processor schema — prevents model
            # architecture params (embedding_dim, n_layers, etc.) from bleeding into
            # processor_params_x/y when both processor and model params are swept together.
            from neural_mi.defaults import PROCESSOR_PARAMS_SCHEMA
            proc_type_x = self.base_params.get('processor_type_x', 'continuous')
            proc_type_y = self.base_params.get('processor_type_y', proc_type_x)
            valid_proc_keys_x = set(PROCESSOR_PARAMS_SCHEMA.get(proc_type_x, []))
            valid_proc_keys_y = set(PROCESSOR_PARAMS_SCHEMA.get(proc_type_y, []))
            proc_params_from_sweep_x = {k: v for k, v in params.items() if k in valid_proc_keys_x}
            proc_params_from_sweep_y = {k: v for k, v in params.items() if k in valid_proc_keys_y}

            task_processor_params_x = (self.base_params.get('processor_params_x') or {}).copy()
            if 'processor_params_x' in kwargs:
                task_processor_params_x.update(kwargs['processor_params_x'])
            task_processor_params_x.update(proc_params_from_sweep_x)

            task_processor_params_y = (self.base_params.get('processor_params_y') or {}).copy()
            if 'processor_params_y' in kwargs:
                task_processor_params_y.update(kwargs['processor_params_y'])
            task_processor_params_y.update(proc_params_from_sweep_y)

            current_params.update({
                'processor_params_x': task_processor_params_x,
                'processor_params_y': task_processor_params_y,
            })
            
            if is_proc_sweep:
                task_data_x, task_data_y = self.x_data, self.y_data
            else:
                x_to_send, y_to_send = self.x_data, self.y_data
                if max_samples_per_task and self.x_data is not None and self.x_data.shape[0] > max_samples_per_task:
                    indices = np.random.choice(self.x_data.shape[0], max_samples_per_task, replace=False)
                    x_to_send = self.x_data[indices]
                    y_to_send = self.y_data[indices] if self.y_data is not None else None
                task_data_x, task_data_y = x_to_send, y_to_send

            task_run_id = f"{run_id_base}_c{i_combo}"
            tasks.append((task_data_x, task_data_y, current_params.copy(), task_run_id))
        
        logger.debug(f"Created {len(tasks)} tasks for the sweep.")
        return tasks

    def run(self, sweep_grid: Dict[str, List], is_proc_sweep: bool = False, n_workers: Optional[int] = None,
            max_samples_per_task: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """Executes the hyperparameter sweep in parallel."""
        tasks = self._prepare_tasks(sweep_grid, is_proc_sweep, max_samples_per_task, **kwargs)
        results = self._run_parallel(tasks, n_workers)
        logger.info("Parameter sweep finished.")
        return results