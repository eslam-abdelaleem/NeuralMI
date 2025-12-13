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
    x_data, y_data, params, run_id = args
    
    # Check if this task needs to process raw data or if it received pre-processed tensors
    # create_dataset handles both cases seamlessly.
    dataset = create_dataset(
        x_data, y_data,
        processor_type_x=params.get('processor_type_x'),
        processor_type_y=params.get('processor_type_y'),
        processor_params_x=params.get('processor_params_x'),
        processor_params_y=params.get('processor_params_y')
    )

    # Now that data is processed and in the dataset object, we can safely determine input dimensions
    # even if the task started with raw data.
    # Check for .shape attribute (handling list-based datasets like SpikeWindowDataset which might have list data_orig, but self.data should be tensor)
    # Wait, dataset.x_data returns self.data.
    # If SpikeWindowDataset, self.data IS a Tensor.
    # So why did we get AttributeError: 'list' object has no attribute 'shape'?
    # Only if self.data is NOT a Tensor.
    # self.data is initialized to None.
    # If move_data_to_windows wasn't called, it's None.
    # If move_data_to_windows failed, it's None.
    # If it was initialized with a list and not processed?
    # SpikeWindowDataset calls move_data_to_windows in init if window_manager provided.
    # create_dataset creates window_manager and calls set_window_manager.
    # set_window_manager does NOT call move_data_to_windows automatically in TemporalWindowDataset.
    # But PairedTemporalDataset calls _build_windows -> move_data_to_windows.
    # So it should be processed.
    # Unless dataset.x_data returns something else?
    # PairedTemporalDataset.x_data -> self.x_dataset.data.

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
    trainer = Trainer(
        model=critic.to(device), estimator_fn=ESTIMATORS[params['estimator_name']], optimizer=optimizer,
        device=device, use_variational=params.get('use_variational', False),
        beta=params.get('beta', 512.0),
        estimator_params=params.get('estimator_params')
    )
    
    results = trainer.train(dataset, params['n_epochs'], params['batch_size'],
        patience=params['patience'], run_id=run_id,
        output_units=params.get('output_units', 'nats'),
        verbose=params.get('verbose', True),
        save_best_model_path=params.get('save_best_model_path'),
        split_mode=params.get('split_mode', 'blocked'),
        train_indices=params.get('train_indices'),
        test_indices=params.get('test_indices')
    )
    
    return_params = params.copy()
    return_params.pop('custom_critic', None)
    return_params.pop('custom_embedding_cls', None)
    
    return {**return_params, **results}