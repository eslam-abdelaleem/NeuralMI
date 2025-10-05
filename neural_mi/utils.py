# neural_mi/utils.py

import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np

from neural_mi.estimators import ESTIMATORS
from neural_mi.models.embeddings import MLP, VarMLP, BaseEmbedding, CNN1D
from neural_mi.models.critics import SeparableCritic, ConcatCritic, BaseCritic, BilinearCritic, ConcatCriticCNN
from neural_mi.training.trainer import Trainer
from neural_mi.logger import logger
from neural_mi.data.handler import DataHandler


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Selects the appropriate device, including 'mps' for Apple Silicon.
    """
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    
def build_critic(critic_type: str, embedding_params: Dict[str, Any], 
                 use_variational: bool = False, custom_embedding_model: Optional[type] = None) -> BaseCritic:
    
    model_type = embedding_params.get('embedding_model', 'mlp')
    hidden_dim, n_layers = embedding_params['hidden_dim'], embedding_params['n_layers']
    embed_dim = embedding_params['embedding_dim']

    if critic_type == 'concat_cnn':
        if model_type.lower() != 'cnn':
            raise ValueError("critic_type='concat_cnn' requires embedding_model='cnn'.")
        
        cnn_x = CNN1D(embedding_params['n_channels_x'], hidden_dim, embed_dim, n_layers).conv_layers
        cnn_y = CNN1D(embedding_params['n_channels_y'], hidden_dim, embed_dim, n_layers).conv_layers
        
        # The decision head is also a CNN that operates on the concatenated feature maps
        decision_head = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear( (hidden_dim // 2) * embedding_params['window_size'], 1)
        )
        return ConcatCriticCNN(cnn_x, cnn_y, decision_head)

    # Logic for other critics
    if custom_embedding_model:
        EmbeddingModel = custom_embedding_model
    elif use_variational:
        EmbeddingModel = VarMLP
    elif model_type.lower() == 'cnn':
        EmbeddingModel = CNN1D
    else:
        EmbeddingModel = MLP

    if model_type.lower() == 'cnn':
        input_dim_x, input_dim_y = embedding_params['n_channels_x'], embedding_params['n_channels_y']
    else:
        input_dim_x, input_dim_y = embedding_params['input_dim_x'], embedding_params['input_dim_y']

    net_x = EmbeddingModel(input_dim_x, hidden_dim, embed_dim, n_layers)
    net_y = EmbeddingModel(input_dim_y, hidden_dim, embed_dim, n_layers)

    if critic_type == 'separable':
        return SeparableCritic(net_x, net_y)
    elif critic_type == 'bilinear':
        return BilinearCritic(net_x, net_y, embed_dim)
    elif critic_type == 'concat':
        if model_type.lower() == 'cnn':
             raise NotImplementedError("ConcatCritic with CNN is not yet supported.")
        concat_input_dim = embedding_params['input_dim_x'] + embedding_params['input_dim_y']
        concat_net = MLP(concat_input_dim, hidden_dim, 1, n_layers)
        return ConcatCritic(concat_net)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


def run_training_task(args: tuple) -> Dict[str, Any]:
    """A top-level function that can be pickled for multiprocessing."""
    x_data, y_data, params, run_id = args

    if isinstance(x_data, list) or (isinstance(x_data, np.ndarray) and x_data.ndim == 2):
        proc_type = params['processor_type']
        
        # Combine base processor params with any swept processor params
        proc_params = params.get('processor_params', {}).copy()
        for key in ['window_size', 'step_size', 'n_seconds', 'max_spikes_per_window', 'data_format']:
            if key in params:
                proc_params[key] = params[key]
        
        x_data, y_data = DataHandler(x_data, y_data, proc_type, proc_params).process()
    
    # Update dimensions in params now that data is processed
    params['input_dim_x'] = x_data.shape[1] * x_data.shape[2]
    params['input_dim_y'] = y_data.shape[1] * y_data.shape[2]
    params['n_channels_x'] = x_data.shape[1]
    params['n_channels_y'] = y_data.shape[1]

    critic = build_critic(params['critic_type'], params, params.get('use_variational', False), 
                          params.get('custom_embedding_model'))
    optimizer = optim.Adam(critic.parameters(), lr=params['learning_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(params.get('device'))
    trainer = Trainer(model=critic.to(device), estimator_fn=ESTIMATORS[params['estimator_name']], optimizer=optimizer,
                      device=device, use_variational=params.get('use_variational', False),
                      beta=params.get('beta', 1.0))
    
    # Pass verbose flag from params to the trainer
    results = trainer.train(x_data, y_data, params['n_epochs'], params['batch_size'], 
                              patience=params['patience'], run_id=run_id,
                              output_units=params.get('output_units', 'nats'),
                              verbose=params.get('verbose', True))
    return {**params, **results}


def find_saturation_point(summary_df: pd.DataFrame, param_col: str = 'embedding_dim', 
                            mean_col: str = 'mi_mean', std_col: str = 'mi_std', 
                            strictness: Union[List[float], float] = 1.0) -> Dict[float, Any]:
    if not isinstance(strictness, list): strictness = [strictness]
    df = summary_df.sort_values(param_col).reset_index()
    mi_diff = df[mean_col].diff().to_numpy()
    estimated_dims = {}
    for s in strictness:
        saturation_indices = np.where(mi_diff[1:] < (s * df[std_col].iloc[1:]))[0]
        if len(saturation_indices) > 0:
            saturation_dim = df[param_col].iloc[saturation_indices[0] + 1]
        else:
            saturation_dim = df[param_col].max()
        estimated_dims[s] = saturation_dim
    return estimated_dims