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
from neural_mi.data.handler import DataHandler


def _process_task_data(x_data: Any, y_data: Any, params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function to handle data processing within a worker task."""
    
    # If data is already a pre-processed 3D tensor, just return it.
    if isinstance(x_data, torch.Tensor) and x_data.ndim == 3:
        return x_data, y_data

    # The parameters are guaranteed to be correctly populated by the upstream run() function.
    # The DataHandler itself will handle the logic of defaulting y_params to x_params if needed.
    handler = DataHandler(
        x_data, y_data,
        params.get('processor_type_x'),
        params.get('processor_params_x'),
        params.get('processor_type_y'),
        params.get('processor_params_y')
    )
    return handler.process()


def run_training_task(args: tuple) -> Dict[str, Any]:
    """A top-level function that can be pickled for multiprocessing."""
    x_data, y_data, params, run_id = args

    x_data, y_data = _process_task_data(x_data, y_data, params)
    
    # Update dimensions in params now that data is processed
    if x_data is not None:
        params['input_dim_x'] = x_data.shape[1] * x_data.shape[2]
        params['n_channels_x'] = x_data.shape[1]
    if y_data is not None:
        params['input_dim_y'] = y_data.shape[1] * y_data.shape[2]
        params['n_channels_y'] = y_data.shape[1]

    if params.get('custom_critic') is not None:
        critic = params['custom_critic']
        logger.debug("Using pre-initialized custom critic model. Model architecture parameters in 'base_params' will be ignored.")
    else:
        critic = build_critic(params.get('critic_type', 'separable'),
                              params,
                              params.get('custom_embedding_cls'))
                              
    optimizer = optim.Adam(critic.parameters(), lr=params['learning_rate'])
    device = get_device(params.get('device'))
    trainer = Trainer(model=critic.to(device), estimator_fn=ESTIMATORS[params['estimator_name']], optimizer=optimizer,
                      device=device, use_variational=params.get('use_variational', False),
                      beta=params.get('beta', 512.0),
                      estimator_params=params.get('estimator_params'))
    
    results = trainer.train(x_data, y_data, params['n_epochs'], params['batch_size'],
                                  patience=params['patience'], run_id=run_id,
                                  output_units=params.get('output_units', 'nats'),
                                  verbose=params.get('verbose', True),
                                  save_best_model_path=params.get('save_best_model_path'),
                                  split_mode=params.get('split_mode', 'blocked'),
                                  train_indices=params.get('train_indices'),
                                  test_indices=params.get('test_indices'))
    
    return_params = params.copy()
    return_params.pop('custom_critic', None)
    return_params.pop('custom_embedding_cls', None)
    
    return {**return_params, **results}