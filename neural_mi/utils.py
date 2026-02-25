# neural_mi/utils.py

import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np

from neural_mi.estimators import ESTIMATORS
from neural_mi.models.embeddings import MLP, VarMLP, BaseEmbedding, CNN1D, GRU, LSTM, TCN, Transformer
from neural_mi.models.critics import SeparableCritic, ConcatCritic, BaseCritic, HybridCritic
from neural_mi.logger import logger

def get_device(device_str: Optional[str] = None) -> torch.device:
    """Selects the appropriate device, including 'mps' for Apple Silicon."""
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
        # Handle PyTorch tensors explicitly to avoid NumPy 2.0 deprecation warnings
        if torch.is_tensor(x_data): 
            x_data = x_data.detach().cpu().numpy()
        elif not isinstance(x_data, np.ndarray): 
            x_data = np.array(x_data)
            
        if torch.is_tensor(y_data): 
            y_data = y_data.detach().cpu().numpy()
        elif not isinstance(y_data, np.ndarray): 
            y_data = np.array(y_data)
        
        if lag == 0:
            return x_data, y_data
        elif lag > 0: # y is shifted to the left (y is in the future of x)
            return x_data[:-lag, :], y_data[lag:, :]
        else: # lag < 0, y is shifted to the right (y is in the past of x)
            return x_data[-lag:, :], y_data[:lag, :]
            
    elif processor_type == 'spike':
        y_shifted = [spikes - lag for spikes in y_data] # lag is in seconds for spikes
        return x_data, y_shifted
        
    return x_data, y_data
    
def build_critic(critic_type: str, embedding_params: Dict[str, Any], 
                 custom_embedding_cls: Optional[type] = None) -> BaseCritic:
    """Builds and returns a critic model based on the provided parameters.

    This function expects `embedding_params` to be fully populated with
    defaults (e.g., via `ParameterValidator.apply_defaults`). It strictly
    accesses required parameters and will raise a KeyError if something is missing,
    preventing silent failures from missing defaults.
    """
    
    # Access parameters strictly to ensure defaults were applied
    use_variational = embedding_params['use_variational']
    model_type = embedding_params['embedding_model'].lower()
    hidden_dim = embedding_params['hidden_dim']
    n_layers = embedding_params['n_layers']
    embed_dim = embedding_params['embedding_dim']
    max_n_batches = embedding_params['max_n_batches']

    # --- Model Selection Logic ---
    if custom_embedding_cls:
        EmbeddingModel = custom_embedding_cls
    elif use_variational:
        EmbeddingModel = VarMLP
    elif model_type == 'cnn':
        EmbeddingModel = CNN1D
    elif model_type == 'gru':
        EmbeddingModel = GRU
    elif model_type == 'lstm':
        EmbeddingModel = LSTM
    elif model_type == 'tcn':
        EmbeddingModel = TCN
    elif model_type == 'transformer':
        EmbeddingModel = Transformer
    else: # Default to MLP
        EmbeddingModel = MLP
    
    # --- Parameter Preparation ---
    model_kwargs = {
        'hidden_dim': hidden_dim,
        'embed_dim': embed_dim,
        'n_layers': n_layers,
    }
    critic_kwargs = {
        'embed_dim': embed_dim,
        'max_n_batches': max_n_batches,
        'use_variational': use_variational
    }

    if model_type in ['cnn', 'gru', 'lstm', 'tcn', 'transformer']:
        input_dim_x, input_dim_y = embedding_params['n_channels_x'], embedding_params['n_channels_y']
        if model_type in ['cnn', 'tcn']:
            model_kwargs['kernel_size'] = embedding_params.get('kernel_size', 7 if model_type == 'cnn' else 3)
        if model_type in ['gru', 'lstm']:
            model_kwargs['bidirectional'] = embedding_params.get('bidirectional', False)
        if model_type in ['transformer']:
            model_kwargs['nhead'] = embedding_params.get('nhead', 4)
    else: # MLP
        input_dim_x, input_dim_y = embedding_params['input_dim_x'], embedding_params['input_dim_y']

    net_x = EmbeddingModel(input_dim_x, **model_kwargs)
    net_y = EmbeddingModel(input_dim_y, **model_kwargs)

    # --- Critic Assembly ---
    if critic_type == 'separable':
        return SeparableCritic(embedding_net_x=net_x, embedding_net_y=net_y, **critic_kwargs)
    elif critic_type == 'hybrid':
        decision_head_input_dim = embed_dim * 2
        decision_head = MLP(input_dim=decision_head_input_dim, hidden_dim=hidden_dim, embed_dim=1, n_layers=max(1, n_layers - 1))
        return HybridCritic(embedding_net_x=net_x, embedding_net_y=net_y, decision_head=decision_head, **critic_kwargs)
    elif critic_type == 'concat':
        concat_input_dim = input_dim_x + input_dim_y
        concat_net = MLP(input_dim=concat_input_dim, hidden_dim=hidden_dim, embed_dim=1, n_layers=n_layers)
        return ConcatCritic(embedding_net=concat_net, **critic_kwargs)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


def compute_cross_covariance_spectrum(zx: torch.Tensor, zy: torch.Tensor) -> np.ndarray:
    """Computes the singular values of the cross-covariance matrix of embeddings."""
    zx = zx - zx.mean(dim=0, keepdim=True)
    zy = zy - zy.mean(dim=0, keepdim=True)
    
    N = zx.size(0)
    if N <= 1:
        return np.array([])
        
    # Perform matmul on CPU to save VRAM for large embedding sets
    cov_xy = torch.matmul(zx.cpu().T, zy.cpu()) / (N - 1)
    _, s_xy, _ = torch.linalg.svd(cov_xy)
    
    return s_xy.numpy()

def compute_spectral_metrics(spectrum: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Computes dimensionality metrics from singular values."""
    s = np.array(spectrum)
    s = s[s > eps]
    
    metrics = {}
    
    # 1. Variance/Energy-based PR
    lam = s**2
    metrics["pr_covariance"] = (lam.sum())**2 / (lam**2).sum() if lam.sum() > 0 else 0.0

    # 2. "Soft" PR (Based on Singular Values)
    metrics["pr_singular"] = (s.sum())**2 / (s**2).sum() if s.sum() > 0 else 0.0

    # 3. Effective Rank
    if s.sum() > 0:
        p = s / s.sum()
        entropy = -np.sum(p * np.log(p))
        metrics["effective_rank"] = np.exp(entropy)
    else:
        metrics["effective_rank"] = 0.0
    
    return metrics