# neural_mi/embeddings_io.py
"""Utilities for extracting learned embeddings from saved NeuralMI models.

This module provides :func:`extract_embeddings`, a convenience function that
reloads a saved critic model and extracts its learned representations for any
pair of input arrays.  Models saved by NeuralMI >= 0.2 carry a ``build_params``
dict alongside the state dictionary so that the architecture can be rebuilt
automatically.  Older single-dict state-dict files are also supported, but
require the caller to supply ``base_params`` explicitly.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from neural_mi.logger import logger
from neural_mi.utils import build_critic, get_device


# Architecture keys that must be present to rebuild a critic
_REQUIRED_BUILD_KEYS = (
    'critic_type', 'embedding_model', 'hidden_dim', 'embedding_dim', 'n_layers',
    'input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y',
)


_EMBEDDING_BATCH = 512  # internal batch size; no effect on results


def extract_embeddings(
    model_path: str,
    x_data,
    y_data,
    base_params: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a saved critic and extract embeddings for (x_data, y_data).

    All samples are embedded in original order — no subsampling, no shuffling —
    so the returned arrays are index-aligned with the input arrays.  Inference
    is performed in mini-batches to avoid OOM errors on large datasets.

    NeuralMI saves models in one of two formats:

    - **New format** (NeuralMI >= 0.2): a dict with ``'state_dict'`` and
      ``'build_params'`` keys.  ``build_params`` contains all architecture
      hyperparameters and is used to reconstruct the critic automatically.
    - **Old format** (raw ``state_dict``): the caller must provide ``base_params``
      with the same architecture used during training.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (``.pt`` or ``.pth``).
    x_data : np.ndarray or torch.Tensor
        Input data for variable X.  Shape must match what the model was trained
        on (e.g., ``(n_samples, input_dim_x)`` for MLP, or
        ``(n_samples, n_channels, window_size)`` for CNN/GRU).
    y_data : np.ndarray or torch.Tensor
        Input data for variable Y.
    base_params : dict, optional
        Required only when loading old-format state-dict files.  Must include
        all keys needed by :func:`~neural_mi.utils.build_critic`.
    device : str, optional
        Compute device (e.g., ``'cpu'``, ``'cuda'``).  Auto-detected if None.

    Returns
    -------
    embeddings_x : np.ndarray
        Shape ``(n_samples, embedding_dim)``.
    embeddings_y : np.ndarray
        Shape ``(n_samples, embedding_dim)``.

    Examples
    --------
    >>> zx, zy = nmi.extract_embeddings(
    ...     model_path='best_model.pt',
    ...     x_data=x_test,
    ...     y_data=y_test,
    ... )
    >>> print(zx.shape)   # (n_samples, embedding_dim)
    """
    # --- Load checkpoint ---
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(loaded, dict) and 'state_dict' in loaded and 'build_params' in loaded:
        # New format
        state_dict = loaded['state_dict']
        bp = loaded['build_params']
        logger.debug(f"Loaded new-format model from {model_path} "
                     f"(critic_type={bp.get('critic_type')}, "
                     f"embedding_model={bp.get('embedding_model')}).")
    else:
        # Old format — raw state dict
        state_dict = loaded
        if base_params is None:
            raise ValueError(
                f"'{model_path}' appears to be an old-format state-dict file with no "
                f"embedded build_params.  Provide the base_params dict that was used "
                f"during training so the architecture can be reconstructed."
            )
        bp = base_params
        missing = [k for k in _REQUIRED_BUILD_KEYS if k not in bp]
        if missing:
            raise ValueError(
                f"base_params is missing required keys for critic reconstruction: "
                f"{missing}.  Please provide all architecture parameters."
            )
        logger.debug(f"Loaded old-format state dict from {model_path}.")

    # --- Rebuild critic ---
    _device = get_device(device)
    critic = build_critic(bp.get('critic_type', 'separable'), bp)
    critic.load_state_dict(state_dict, strict=True)
    critic = critic.to(_device)
    critic.eval()

    # --- Prepare tensors ---
    def _to_tensor(arr):
        if torch.is_tensor(arr):
            return arr.float()
        return torch.from_numpy(np.asarray(arr)).float()

    x_t = _to_tensor(x_data)
    y_t = _to_tensor(y_data)

    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError(
            f"x_data and y_data must have the same number of samples, "
            f"got {x_t.shape[0]} and {y_t.shape[0]}."
        )

    # --- Extract embeddings in mini-batches, preserving sample order ---
    n = x_t.shape[0]
    zx_parts, zy_parts = [], []
    with torch.no_grad():
        for start in range(0, n, _EMBEDDING_BATCH):
            end = min(start + _EMBEDDING_BATCH, n)
            bzx, bzy = critic.get_embeddings(
                x_t[start:end].to(_device),
                y_t[start:end].to(_device),
            )
            zx_parts.append(bzx.detach().cpu())
            zy_parts.append(bzy.detach().cpu())

    return torch.cat(zx_parts, dim=0).numpy(), torch.cat(zy_parts, dim=0).numpy()
