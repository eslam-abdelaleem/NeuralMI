# neural_mi/analysis/sweep.py

import torch
import itertools
import uuid
import multiprocessing
from multiprocessing import cpu_count
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, List, Tuple

from neural_mi.logger import logger
from neural_mi.estimators import bounds
from neural_mi.utils import run_training_task


class ParameterSweep:
    """
    Orchestrates a parameter sweep for exploratory analysis.
    This class assumes it receives pre-processed, 3D data.
    """
    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        base_params: Dict[str, Any],
        critic_type: str = 'separable',
        estimator_name: str = 'infonce',
        use_variational: bool = False,
        **kwargs: Any
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.base_params = base_params
        self.base_params['critic_type'] = critic_type
        self.base_params['estimator_name'] = estimator_name
        self.base_params['use_variational'] = use_variational
        self.base_params['input_dim_x'] = x_data.shape[1] * x_data.shape[2]
        self.base_params['input_dim_y'] = y_data.shape[1] * y_data.shape[2]
        self.base_params.update(kwargs)


    def run(
        self,
        sweep_grid: Dict[str, List[Any]],
        n_workers: Optional[int] = None,
        max_samples_per_task: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the parameter sweep.

        Parameters
        ----------
        sweep_grid : dict
            A dictionary defining the parameter grid to sweep over.
        n_workers : int, optional
            Number of parallel workers to use. Defaults to CPU count.
        max_samples_per_task : int, optional
            If provided, subsamples the data to this many samples for each
            task in the sweep to reduce memory usage.

        Returns
        -------
        list
            A list of dictionaries, where each dictionary contains the
            results for a single run in the sweep.
        """
        if n_workers is None:
            n_workers = cpu_count()
        logger.info(f"Starting parameter sweep with {n_workers} workers...")

        x_run, y_run = self.x_data, self.y_data
        if max_samples_per_task and self.x_data.shape[0] > max_samples_per_task:
            logger.info(f"Subsampling data from {self.x_data.shape[0]} to {max_samples_per_task} samples for this sweep.")
            indices = np.random.choice(self.x_data.shape[0], max_samples_per_task, replace=False)
            x_run = self.x_data[indices]
            y_run = self.y_data[indices]

        tasks = self._prepare_tasks(sweep_grid, x_run, y_run)
        if not tasks:
            logger.info("No tasks to run. Your sweep_grid is empty.")
            return []

        # Use get_context("spawn") to create the pool, which is safer for PyTorch.
        with multiprocessing.get_context("spawn").Pool(processes=n_workers) as pool:
            all_results = list(tqdm(
                pool.imap(run_training_task, tasks),
                total=len(tasks),
                desc="Parameter Sweep"
            ))
        
        logger.info("Parameter sweep finished.")
        return all_results

    def _prepare_tasks(
        self,
        sweep_grid: Dict[str, List[Any]],
        x_data: torch.Tensor,
        y_data: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], str]]:
        """Creates a list of training tasks for each point in the grid."""
        tasks: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], str]] = []
        run_id_base = str(uuid.uuid4())

        if self.base_params['critic_type'] == 'concat' and 'embedding_dim' in sweep_grid:
            logger.warning("'embedding_dim' is not applicable for ConcatCritic and will be ignored.")
            sweep_grid.pop('embedding_dim', None)

        keys, values = zip(*sweep_grid.items()) if sweep_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if sweep_grid else [{}]

        for i_combo, params in enumerate(param_combinations):
            current_params = self.base_params.copy()
            current_params.update(params)
            task_run_id = f"{run_id_base}_c{i_combo}"
            tasks.append((x_data, y_data, current_params.copy(), task_run_id))
        
        logger.info(f"Created {len(tasks)} tasks for the sweep...")
        return tasks