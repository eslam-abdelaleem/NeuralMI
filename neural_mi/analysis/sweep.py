# neural_mi/analysis/sweep.py

import torch
import itertools
import uuid
import multiprocessing
from multiprocessing import cpu_count
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

from neural_mi.utils import run_training_task
from neural_mi.logger import logger
from neural_mi.data.handler import DataHandler

class ParameterSweep:
    def __init__(self, x_data, y_data, base_params, **kwargs):
        self.x_data, self.y_data = x_data, y_data
        self.base_params = base_params
        
        if isinstance(x_data, torch.Tensor):
            self.base_params.update({
                'input_dim_x': x_data.shape[1] * x_data.shape[2],
                'input_dim_y': y_data.shape[1] * y_data.shape[2],
                'n_channels_x': x_data.shape[1],
                'n_channels_y': y_data.shape[1],
                **kwargs
            })
        else:
             self.base_params.update(kwargs)

    # *** FIX: Added 'is_proc_sweep' to the method signature ***
    def run(self, sweep_grid: Dict[str, List], is_proc_sweep: bool = False, n_workers: Optional[int] = None,
            max_samples_per_task: Optional[int] = None) -> List[Dict[str, Any]]:
        
        if n_workers is None:
            n_workers = cpu_count()
        logger.info(f"Starting parameter sweep with {n_workers} workers...")

        tasks = self._prepare_tasks(sweep_grid, is_proc_sweep, max_samples_per_task)
        if not tasks:
            logger.warning("No tasks to run. Your sweep_grid might be empty.")
            return []

        with multiprocessing.get_context("spawn").Pool(processes=n_workers) as pool:
            all_results = list(tqdm(
                pool.imap(run_training_task, tasks), total=len(tasks),
                desc="Parameter Sweep Progress", unit="task"
            ))
        
        logger.info("Parameter sweep finished.")
        return all_results

    def _prepare_tasks(self, sweep_grid: Dict[str, List], is_proc_sweep: bool,
                       max_samples_per_task: Optional[int]) -> List[tuple]:
        tasks = []
        run_id_base = str(uuid.uuid4())
        sweep_grid = sweep_grid or {}

        if self.base_params['critic_type'] == 'concat' and 'embedding_dim' in sweep_grid:
            logger.warning("'embedding_dim' is not applicable for ConcatCritic and will be ignored.")
            sweep_grid.pop('embedding_dim', None)

        keys, values = zip(*sweep_grid.items()) if sweep_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if sweep_grid else [{}]

        for i_combo, params in enumerate(param_combinations):
            current_params = {**self.base_params, **params}
            
            if is_proc_sweep:
                task_data_x, task_data_y = self.x_data, self.y_data
            else:
                x_to_send, y_to_send = self.x_data, self.y_data
                if max_samples_per_task and self.x_data.shape[0] > max_samples_per_task:
                    indices = np.random.choice(self.x_data.shape[0], max_samples_per_task, replace=False)
                    x_to_send, y_to_send = self.x_data[indices], self.y_data[indices]
                task_data_x, task_data_y = x_to_send, y_to_send

            task_run_id = f"{run_id_base}_c{i_combo}"
            tasks.append((task_data_x, task_data_y, current_params.copy(), task_run_id))
        
        logger.debug(f"Created {len(tasks)} tasks for the sweep.")
        return tasks