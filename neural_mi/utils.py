# neural_mi/utils.py

import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np

from neural_mi.estimators import ESTIMATORS
from neural_mi.models.embeddings import MLP, VarMLP, BaseEmbedding, CNN1D
from neural_mi.models.critics import SeparableCritic, ConcatCritic, BaseCritic, BilinearCritic, ConcatCriticCNN
from neural_mi.logger import logger

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

def _shift_data(x_data: Any, y_data: Any, lag: int, processor_type: str) -> tuple:
    """Shifts y_data relative to x_data based on the specified lag."""
    if processor_type in ['continuous', 'categorical']:
        if not isinstance(x_data, np.ndarray): x_data = np.array(x_data)
        if not isinstance(y_data, np.ndarray): y_data = np.array(y_data)
        
        if lag == 0:
            return x_data, y_data
        elif lag > 0: # y is shifted to the left (y is in the future of x)
            return x_data[:, :-lag], y_data[:, lag:]
        else: # lag < 0, y is shifted to the right (y is in the past of x)
            return x_data[:, -lag:], y_data[:, :lag]
            
    elif processor_type == 'spike':
        y_shifted = [spikes - lag for spikes in y_data] # lag is in seconds for spikes
        return x_data, y_shifted
        
    return x_data, y_data
    
def build_critic(critic_type: str, embedding_params: Dict[str, Any], 
                 custom_embedding_cls: Optional[type] = None) -> BaseCritic:
    
    use_variational = embedding_params.get('use_variational', False)
    model_type = embedding_params.get('embedding_model', 'mlp')
    hidden_dim, n_layers = embedding_params['hidden_dim'], embedding_params['n_layers']
    embed_dim = embedding_params['embedding_dim']

    if critic_type == 'concat_cnn':
        if model_type.lower() != 'cnn':
            raise ValueError("critic_type='concat_cnn' requires embedding_model='cnn'.")
        
        # Build the full embedding networks first
        cnn_x = CNN1D(embedding_params['n_channels_x'], hidden_dim, embed_dim, n_layers)
        cnn_y = CNN1D(embedding_params['n_channels_y'], hidden_dim, embed_dim, n_layers)
        
        # The decision head's input dimension is the sum of the two embedding dimensions
        decision_head_input_dim = cnn_x.embed_dim + cnn_y.embed_dim
        
        # A simple MLP is a good choice for the decision head
        decision_head = MLP(
            input_dim=decision_head_input_dim,
            hidden_dim=hidden_dim, # Can reuse hidden_dim or specify a new one
            embed_dim=1, # Output is a scalar score
            n_layers=max(1, n_layers - 1) # Typically a shallower network
        )
        return ConcatCriticCNN(cnn_x, cnn_y, decision_head)

    # Logic for other critics remains the same
    if custom_embedding_cls:
        EmbeddingModel = custom_embedding_cls
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
        concat_input_dim = embedding_params['input_dim_x'] + embedding_params['input_dim_y']
        concat_net = MLP(concat_input_dim, hidden_dim, 1, n_layers)
        return ConcatCritic(concat_net)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")



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