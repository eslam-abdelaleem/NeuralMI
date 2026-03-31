# neural_mi/utils.py

import torch
import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np

import multiprocessing as mp
import os
import platform
import tempfile

from neural_mi.estimators import ESTIMATORS
from neural_mi.models.embeddings import MLP, VariationalWrapper, BaseEmbedding, CNN1D, GRU, LSTM, TCN, Transformer
from neural_mi.models.critics import SeparableCritic, ConcatCritic, BaseCritic, HybridCritic
from neural_mi.logger import logger

def _ensure_cpu(data):
    """Move *data* to CPU if it is a tensor on a non-CPU device.

    Multiprocessing workers receive data via pickle (spawn context), which
    requires all tensors to reside on CPU — CUDA and MPS shared-memory
    mechanisms are not available across process boundaries.  Call this on
    every tensor before adding it to a Pool task tuple.

    Non-tensor inputs (numpy arrays, None, etc.) are returned unchanged.
    """
    if isinstance(data, torch.Tensor) and data.device.type != 'cpu':
        return data.cpu()
    return data


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Selects the appropriate device, including 'mps' for Apple Silicon."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

_mp_configured = False

def _configure_multiprocessing() -> None:
    """Lazily configure multiprocessing for parallel pool creation.

    Called once, lazily, just before any Pool is created. Guarded by
    _mp_configured so it is idempotent even if workflow.py and sweep.py both
    call it in the same process.
    """
    global _mp_configured
    if _mp_configured:
        return
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set — e.g. user called set_start_method themselves; respect it.
        logger.debug("Multiprocessing start method already set; skipping.")
    if platform.system() == "Darwin":
        # macOS spawn workers inherit the parent's TMPDIR which may point to an
        # app sandbox directory not accessible to child processes.
        custom_temp = tempfile.mkdtemp()
        os.environ["TMPDIR"] = custom_temp
        logger.debug(f"macOS: set TMPDIR={custom_temp} for spawn workers.")
    _mp_configured = True

def _shift_data(x_data: Any, y_data: Any, lag: int, processor_type: str,
                sample_rate: Optional[float] = None) -> tuple:
    """Shifts y_data relative to x_data based on the specified lag.

    Parameters
    ----------
    x_data : array-like
        Data for variable X.
    y_data : array-like
        Data for variable Y.
    lag : int or float
        The lag value. Units depend on processor_type and sample_rate — see above.
    processor_type : str
        One of 'continuous', 'categorical', or 'spike'.
    sample_rate : float, optional
        Samples per second for continuous/categorical data. If provided, lag is
        interpreted as seconds and converted to samples. If None, lag is treated
        as samples with a deprecation warning.
    """
    if processor_type in ['continuous', 'categorical']:
        # Convert to numpy if needed
        if torch.is_tensor(x_data):
            x_data = x_data.detach().cpu().numpy()
        elif not isinstance(x_data, np.ndarray):
            x_data = np.array(x_data)

        if torch.is_tensor(y_data):
            y_data = y_data.detach().cpu().numpy()
        elif not isinstance(y_data, np.ndarray):
            y_data = np.array(y_data)

        if sample_rate is not None:
            # Lag provided in seconds — convert to samples
            lag_samples = int(round(lag * sample_rate))
        else:
            # Legacy: treat lag as samples, but warn the user
            logger.warning(
                f"Lag units for '{processor_type}' data are ambiguous without a sample_rate. "
                f"Treating lag={lag} as samples (index offset). "
                f"To specify lag in seconds, pass 'sample_rate' in processor_params_x. "
                f"Note: spike data always uses seconds, so mixing processor types without "
                f"sample_rate will produce inconsistent lag scales."
            )
            lag_samples = int(lag)

        if lag_samples == 0:
            return x_data, y_data
        elif lag_samples > 0:
            # y is in the future of x
            return x_data[:-lag_samples, :], y_data[lag_samples:, :]
        else:
            # y is in the past of x
            return x_data[-lag_samples:, :], y_data[:lag_samples, :]

    elif processor_type == 'spike':
        # Spike times are always in seconds — lag is in seconds, no conversion needed
        y_shifted = [spikes - lag for spikes in y_data]
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
    # Select the deterministic base encoder class first; variational wrapping is
    # applied after construction so all encoder architectures benefit uniformly.
    if custom_embedding_cls:
        EmbeddingModel = custom_embedding_cls
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
    elif model_type == 'mlp':
        EmbeddingModel = MLP
    else:
        raise ValueError(f"Unknown embedding_model: {model_type}")

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
    else:  # MLP (and custom cls with MLP-like signature)
        input_dim_x, input_dim_y = embedding_params['input_dim_x'], embedding_params['input_dim_y']
        model_kwargs['use_spectral_norm'] = embedding_params.get('use_spectral_norm', True)
        model_kwargs['dropout'] = embedding_params.get('dropout', 0.0)
        model_kwargs['norm_layer'] = embedding_params.get('norm_layer', None)

    shared_encoder = embedding_params.get('shared_encoder', False)
    if shared_encoder and critic_type == 'concat':
        raise ValueError(
            "shared_encoder=True is incompatible with critic_type='concat'. "
            "ConcatCritic operates on raw concatenated inputs and has no separate "
            "embedding networks to share. Switch to critic_type='separable' or 'hybrid'."
        )

    # Build the base (deterministic) encoders
    net_x_base = EmbeddingModel(input_dim_x, **model_kwargs)
    net_y_base = net_x_base if shared_encoder else EmbeddingModel(input_dim_y, **model_kwargs)

    # Optionally wrap with VariationalWrapper — works for every encoder type
    if use_variational:
        net_x = VariationalWrapper(net_x_base, embed_dim)
        net_y = net_x if shared_encoder else VariationalWrapper(net_y_base, embed_dim)
    else:
        net_x, net_y = net_x_base, net_y_base

    # Warn when the first embedding layer is severely overparameterized.
    # Large first layers (input_dim * hidden_dim) are the most common cause of
    # overfitting in neuroscience datasets where windows are high-dimensional but
    # sample counts are modest. 500k is a practical threshold — not a hard limit.
    first_layer_params = input_dim_x * hidden_dim
    if first_layer_params > 500_000:
        logger.warning(
            f"Large first embedding layer detected: input_dim_x={input_dim_x} x "
            f"hidden_dim={hidden_dim} = {first_layer_params:,} parameters. "
            f"This may cause overfitting on small datasets. Consider reducing "
            f"window_size, hidden_dim, or using a different embedding model."
        )

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


def compute_cross_covariance_spectrum(
    zx: torch.Tensor,
    zy: torch.Tensor,
    whitening: Optional[str] = 'std'
) -> np.ndarray:
    """Computes the singular values of the cross-covariance matrix of embeddings.

    Parameters
    ----------
    zx, zy : torch.Tensor
        Embeddings of shape (n_samples, embedding_dim).
    whitening : {'std', 'zca', None}, optional
        Normalization applied before SVD.
        - 'std': divide each dimension by its standard deviation (default).
          Makes PR reflect the number of dimensions with non-trivial shared
          variance, independent of embedding output scale.
        - 'zca': full ZCA whitening (sphering). More aggressive; requires
          n_samples >> embedding_dim to be stable.
        - None: no whitening. PR will reflect raw embedding scale.
    """
    zx_np = zx.detach().cpu().float().numpy()
    zy_np = zy.detach().cpu().float().numpy()

    # Center
    zx_np = zx_np - zx_np.mean(axis=0, keepdims=True)
    zy_np = zy_np - zy_np.mean(axis=0, keepdims=True)

    N = zx_np.shape[0]
    if N <= 1:
        return np.array([])

    if whitening == 'std':
        std_x = zx_np.std(axis=0, keepdims=True)
        std_y = zy_np.std(axis=0, keepdims=True)
        # Avoid division by zero for dead dimensions
        zx_np = zx_np / np.where(std_x > 1e-8, std_x, 1.0)
        zy_np = zy_np / np.where(std_y > 1e-8, std_y, 1.0)
    elif whitening == 'zca':
        def _zca_whiten(Z):
            cov = (Z.T @ Z) / (Z.shape[0] - 1)
            U, S, _ = np.linalg.svd(cov)
            S_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(S, 1e-8)))
            W = U @ S_inv_sqrt @ U.T
            return Z @ W.T
        zx_np = _zca_whiten(zx_np)
        zy_np = _zca_whiten(zy_np)
    elif whitening is not None:
        raise ValueError(f"Unknown whitening method: '{whitening}'. Expected 'std', 'zca', or None.")

    cov_xy = (zx_np.T @ zy_np) / (N - 1)
    _, s_xy, _ = np.linalg.svd(cov_xy)

    return s_xy


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

    # 3. Effective Rank and Spectral Entropy
    if s.sum() > 0:
        p = s / s.sum()
        p = p[p > 0]  # guard against numerical zeros before log
        entropy = -np.sum(p * np.log(p))
        metrics["effective_rank"] = np.exp(entropy)
        metrics["spectral_entropy"] = float(entropy)
    else:
        metrics["effective_rank"] = 0.0
        metrics["spectral_entropy"] = 0.0

    return metrics