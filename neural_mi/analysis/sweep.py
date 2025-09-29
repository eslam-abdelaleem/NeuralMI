# neural_mi/analysis/sweep.py

import torch
import itertools
import uuid
from multiprocessing import Pool, cpu_count

from neural_mi.estimators import bounds
from neural_mi.utils import run_training_task


class ParameterSweep:
    """
    Orchestrates a parameter sweep for exploratory analysis.
    This class assumes it receives pre-processed, 3D data.
    """
    def __init__(self, x_data, y_data, base_params, critic_type='separable', 
                 estimator_fn=bounds.infonce_lower_bound, use_variational=False, **kwargs):

        self.x_data = x_data
        self.y_data = y_data
        self.base_params = base_params
        self.base_params['critic_type'] = critic_type
        self.base_params['estimator_fn'] = estimator_fn
        self.base_params['use_variational'] = use_variational
        # We can now reliably calculate input dimensions from the 3D tensor
        self.base_params['input_dim_x'] = x_data.shape[1] * x_data.shape[2]
        self.base_params['input_dim_y'] = y_data.shape[1] * y_data.shape[2]
        # Pass through any other relevant kwargs from the run() function
        self.base_params.update(kwargs)


    def run(self, sweep_grid, n_workers=None):
        """
        Executes the parameter sweep.
        """
        if n_workers is None:
            n_workers = cpu_count()
        print(f"Starting parameter sweep with {n_workers} workers...")

        tasks = self._prepare_tasks(sweep_grid)
        if not tasks:
            print("No tasks to run. Your sweep_grid is empty.")
            return []

        with Pool(processes=n_workers) as pool:
            all_results = list(pool.map(run_training_task, tasks))
        
        print("Parameter sweep finished.")
        return all_results

    def _prepare_tasks(self, sweep_grid):
        """Creates a list of training tasks for each point in the grid."""
        tasks = []
        run_id_base = str(uuid.uuid4())

        if self.base_params['critic_type'] == 'concat' and 'embedding_dim' in sweep_grid:
            print("Warning: 'embedding_dim' is not applicable for ConcatCritic and will be ignored.")
            sweep_grid.pop('embedding_dim', None)

        keys, values = zip(*sweep_grid.items()) if sweep_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if sweep_grid else [{}]

        for i_combo, params in enumerate(param_combinations):
            current_params = self.base_params.copy()
            current_params.update(params)
            task_run_id = f"{run_id_base}_c{i_combo}"
            tasks.append((self.x_data, self.y_data, current_params.copy(), task_run_id))
        
        print(f"Created {len(tasks)} tasks for the sweep...")
        return tasks
