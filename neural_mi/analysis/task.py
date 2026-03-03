# neural_mi/analysis/task.py
"""
Contains the core, parallelizable training task function.
"""
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple

from neural_mi.utils import build_critic, get_device
from neural_mi.estimators import ESTIMATORS
from neural_mi.training.trainer import Trainer
from neural_mi.logger import logger
from neural_mi.data.handler import create_dataset

def run_training_task(args: tuple) -> Dict[str, Any]:
    """A top-level function that can be pickled for multiprocessing."""
    import random as _random
    x_data, y_data, params, run_id = args

    # Deterministic per-worker seeding: derive a seed from the base seed and
    # the run_id string so every task is reproducible but unique.
    base_seed = params.get('random_seed', None)
    if base_seed is not None:
        import hashlib
        task_seed = (base_seed + int(hashlib.md5(str(run_id).encode()).hexdigest(), 16)) % (2**31)
        _random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        logger.debug(f"Task {run_id} seeded with {task_seed}.")

    
    # Check if this task needs to process raw data or if it received pre-processed tensors
    dataset = create_dataset(
        x_data, y_data,
        processor_type_x=params.get('processor_type_x'),
        processor_type_y=params.get('processor_type_y'),
        processor_params_x=params.get('processor_params_x'),
        processor_params_y=params.get('processor_params_y')
    )

    if dataset.x_data is not None and hasattr(dataset.x_data, 'shape'):
        params['input_dim_x'] = dataset.x_data.shape[1] * dataset.x_data.shape[2]
        params['n_channels_x'] = dataset.x_data.shape[1]

    if dataset.y_data is not None and hasattr(dataset.y_data, 'shape'):
        params['input_dim_y'] = dataset.y_data.shape[1] * dataset.y_data.shape[2]
        params['n_channels_y'] = dataset.y_data.shape[1]

    if params.get('custom_critic') is not None:
        critic = params['custom_critic']
        logger.debug("Using pre-initialized custom critic model. Model architecture parameters in 'base_params' will be ignored.")
    else:
        critic = build_critic(params.get('critic_type', 'separable'),
                              params,
                              params.get('custom_embedding_cls'))
                              
    optimizer = optim.Adam(critic.parameters(), lr=params['learning_rate'])
    device = get_device(params.get('device'))
    
    # Inject custom smoothing into Trainer init
    trainer = Trainer(
        model=critic.to(device), 
        estimator_fn=ESTIMATORS[params['estimator_name']], 
        optimizer=optimizer,
        device=device, 
        use_variational=params.get('use_variational', False),
        beta=params.get('beta', 512.0),
        estimator_params=params.get('estimator_params'),
        custom_smoothing_fn=params.get('custom_smoothing_fn'),
        spectral_whitening=params.get('spectral_whitening', 'std')
    )
    
    # Inject memory, logging, and spectral metrics parameters into train
    results = trainer.train(
        dataset, 
        params['n_epochs'], 
        params['batch_size'],
        patience=params['patience'], 
        run_id=run_id,
        output_units=params.get('output_units', 'nats'),
        verbose=params.get('verbose', False),
        show_progress=params.get('show_progress', True),
        save_best_model_path=params.get('save_best_model_path'),
        split_mode=params.get('split_mode', 'blocked'),
        train_indices=params.get('train_indices'),
        test_indices=params.get('test_indices'),
        max_eval_samples=params.get('max_eval_samples', 5000),
        split_gap_fraction=params.get('split_gap_fraction', 0.5),
        train_subset_size=params.get('train_subset_size'),
        track_spectral_metrics=params.get('track_spectral_metrics', False),
        spectral_output=params.get('spectral_output', 'default'),
        return_spectrum=params.get('return_spectrum', False)
    )
    
    return_params = params.copy()
    return_params.pop('custom_critic', None)
    return_params.pop('custom_embedding_cls', None)
    
    return {**return_params, **results}