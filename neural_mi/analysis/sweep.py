# neural_mi/analysis/sweep.py
"""Provides the ParameterSweep class for running hyperparameter sweeps.

This module defines the core logic for executing multiple training runs in
parallel across a grid of hyperparameters.
"""
import torch
import itertools
import uuid
import multiprocessing
from multiprocessing import cpu_count
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

from neural_mi.analysis.task import run_training_task
from neural_mi.logger import logger
from neural_mi.data.handler import DataHandler

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
        x_data : Any
            The data for variable X.
        y_data : Any
            The data for variable Y.
        base_params : Dict[str, Any]
            A dictionary of fixed parameters for the MI estimator's trainer.
        **kwargs : Dict[str, Any]
            Additional keyword arguments to be added to `base_params`.
        """
        self.x_data, self.y_data = x_data, y_data
        self.base_params = base_params
        
        if isinstance(x_data, torch.Tensor) and x_data.ndim == 3:
            self.base_params.update({
                'input_dim_x': x_data.shape[1] * x_data.shape[2],
                'input_dim_y': y_data.shape[1] * y_data.shape[2],
                'n_channels_x': x_data.shape[1],
                'n_channels_y': y_data.shape[1],
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

        if effective_workers <= 1:
            logger.info("Starting parameter sweep sequentially (n_workers=1)...")
            all_results = [run_training_task(task) for task in tqdm(tasks, desc="Sequential Sweep Progress")]
        else:
            logger.info(f"Starting parameter sweep with {effective_workers} workers...")
            with multiprocessing.get_context("spawn").Pool(processes=effective_workers) as pool:
                all_results = list(tqdm(
                    pool.imap(run_training_task, tasks), total=len(tasks),
                    desc="Parameter Sweep Progress", unit="task"
                ))
        return all_results

    def run(self, sweep_grid: Dict[str, List], is_proc_sweep: bool = False, n_workers: Optional[int] = None,
            max_samples_per_task: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """Executes the hyperparameter sweep in parallel.

        This method generates a list of training tasks based on the Cartesian
        product of the `sweep_grid` parameters and distributes them to a pool
        of worker processes.

        Parameters
        ----------
        sweep_grid : Dict[str, List]
            A dictionary defining the parameter grid to sweep over.
        is_proc_sweep : bool, optional
            If True, indicates that a data processor parameter is being swept,
            meaning data processing must be deferred to the individual workers.
            Defaults to False.
        n_workers : int, optional
            The number of worker processes to use. If None, it defaults to 1 (sequential).
        max_samples_per_task : int, optional
            The maximum number of samples to use for each individual training
            task. If the dataset is larger, it will be randomly subsampled.
            This is useful for quick sweeps on large datasets. Defaults to None.
        **kwargs : Dict[str, Any]
            Additional keyword arguments to be passed to the task preparation.

        Returns
        -------
        List[Dict[str, Any]]
            A list of result dictionaries, with each dictionary containing the
            parameters and MI estimates for a single run in the sweep.
        """
        
        tasks = self._prepare_tasks(sweep_grid, is_proc_sweep, max_samples_per_task, **kwargs)
        results = self._run_parallel(tasks, n_workers)
        logger.info("Parameter sweep finished.")
        return results

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
            
            task_processor_params_x = kwargs.get('processor_params_x', {}).copy()
            task_processor_params_x.update(params)
            
            task_processor_params_y = kwargs.get('processor_params_y', {}).copy()
            task_processor_params_y.update(params)

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
                    x_to_send, y_to_send = self.x_data[indices], self.y_data[indices]
                task_data_x, task_data_y = x_to_send, y_to_send

            task_run_id = f"{run_id_base}_c{i_combo}"
            tasks.append((task_data_x, task_data_y, current_params.copy(), task_run_id))
        
        logger.debug(f"Created {len(tasks)} tasks for the sweep.")
        return tasks