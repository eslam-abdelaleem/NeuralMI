# neural_mi/analysis/sweep.py

import torch
import torch.optim as optim
import itertools
import uuid
from multiprocessing import Pool, cpu_count
from neural_mi.models.embeddings import MLP, VarMLP
from neural_mi.models.critics import SeparableCritic, ConcatCritic
from neural_mi.training.trainer import Trainer
from neural_mi.estimators import bounds

# This is the same factory and task runner from workflow.py
# In a real library, we might move this to a shared `_utils.py` to avoid duplication.
def _build_critic(critic_type, embedding_params, use_variational=False):
    """Dynamically builds a critic model based on the specified type."""
    input_dim_x = embedding_params['input_dim_x']
    input_dim_y = embedding_params['input_dim_y']
    hidden_dim = embedding_params['hidden_dim']
    n_layers = embedding_params['n_layers']
    EmbeddingModel = VarMLP if use_variational else MLP
    
    if critic_type == 'separable':
        embed_dim = embedding_params['embedding_dim']
        embedding_net_x = EmbeddingModel(input_dim_x, hidden_dim, embed_dim, n_layers)
        embedding_net_y = EmbeddingModel(input_dim_y, hidden_dim, embed_dim, n_layers)
        return SeparableCritic(embedding_net_x, embedding_net_y)
    elif critic_type == 'concat':
        # Concat critic's input is the sum of the two flattened input dimensions
        concat_input_dim = input_dim_x + input_dim_y
        embedding_net = EmbeddingModel(concat_input_dim, hidden_dim, 1, n_layers)
        return ConcatCritic(embedding_net)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


def _run_training_task(args):
    """A top-level function that can be pickled for multiprocessing."""
    x_data, y_data, params, run_id = args
    use_variational = params.get('use_variational', False)
    critic = _build_critic(params['critic_type'], params, use_variational=use_variational)
    optimizer = optim.Adam(critic.parameters(), lr=params['learning_rate'])
    
    trainer = Trainer(
        model=critic, estimator_fn=params['estimator_fn'], optimizer=optimizer,
        use_variational=use_variational, beta=params.get('beta', 1.0)
    )
    
    results = trainer.train(
        x_data=x_data, y_data=y_data, n_epochs=params['n_epochs'],
        batch_size=params['batch_size'], patience=params['patience'], run_id=run_id
    )
    return {**params, **results}

class ParameterSweep:
    """
    Orchestrates a parameter sweep for exploratory analysis without bias correction.
    """
    def __init__(self, x_data, y_data, base_params, critic_type='separable', 
                 estimator_fn=bounds.infonce_lower_bound, use_variational=False):
        self.x_data = x_data
        self.y_data = y_data
        self.base_params = base_params
        self.base_params['critic_type'] = critic_type
        self.base_params['estimator_fn'] = estimator_fn
        self.base_params['use_variational'] = use_variational
        self.base_params['input_dim_x'] = x_data.shape[1] * x_data.shape[2]
        self.base_params['input_dim_y'] = y_data.shape[1] * y_data.shape[2]

    def run(self, sweep_grid, n_workers=None):
        """
        Executes the parameter sweep.

        Args:
            sweep_grid (dict): Grid of parameters to search over.
            n_workers (int, optional): Number of parallel processes. Defaults to cpu_count().

        Returns:
            list: A list of dictionaries with results for each parameter combination.
        """
        if n_workers is None:
            n_workers = cpu_count()
        print(f"Starting parameter sweep with {n_workers} workers...")

        tasks = self._prepare_tasks(sweep_grid)
        if not tasks:
            print("No tasks to run. Your sweep_grid is empty.")
            return []

        with Pool(processes=n_workers) as pool:
            all_results = list(pool.map(_run_training_task, tasks))
        
        print("Parameter sweep finished.")
        return all_results

    def _prepare_tasks(self, sweep_grid):
        """Creates a list of training tasks for each point in the grid."""
        tasks = []
        run_id_base = str(uuid.uuid4())

        if self.base_params['critic_type'] == 'concat' and 'embedding_dim' in sweep_grid:
            print("Warning: 'embedding_dim' is not applicable for ConcatCritic and will be ignored.")
            sweep_grid.pop('embedding_dim')

        keys, values = zip(*sweep_grid.items()) if sweep_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if sweep_grid else [{}]

        for i_combo, params in enumerate(param_combinations):
            current_params = self.base_params.copy()
            current_params.update(params)
            task_run_id = f"{run_id_base}_c{i_combo}"
            tasks.append((self.x_data, self.y_data, current_params.copy(), task_run_id))
        
        print(f"Created {len(tasks)} tasks for the sweep...")
        return tasks