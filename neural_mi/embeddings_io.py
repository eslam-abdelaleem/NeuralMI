# neural_mi/embeddings_io.py
"""Saving trained networks, and extracting embeddings from saved ones.

``Training(save_best_model_path=...)`` saves the best epoch of every network a
call trains. A call that trains one network writes to the path as given. When a
call trains several (a grid, repeats, a rigorous ladder, the components of a
difference quantity, lags, channel pairs, splits), each network's file name
carries the labels that identify it, and every path is recorded as
``model_path`` beside that network's row in ``result.runs`` or
``details[config_id]['trainings']``. A directory in place of a file name saves
under a generated name in that directory.

:func:`extract_embeddings` reloads a saved critic and embeds any pair of input
arrays with it. A saved model carries a ``build_params`` dict beside its state
dictionary, so the architecture is rebuilt automatically. A bare state
dictionary also loads when the caller supplies ``base_params``.
"""
from __future__ import annotations

import datetime
import os
import pickle
import re

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from neural_mi.defaults import setting
from neural_mi.logger import logger
from neural_mi.utils import build_critic, float_tensor, get_device


MODEL_EXTENSION = '.pt'


def resolve_model_path(path: Optional[str], mode: str) -> Optional[str]:
    """The file stem a call saves its networks under.

    A directory (an existing one, or a path ending in a separator) becomes
    ``<dir>/neuralmi_<mode>_<timestamp>.pt``, numbered when that stem is taken.
    A file name is used as given, with ``.pt`` added when it has no extension.
    """
    if not path:
        return None
    path = os.fspath(path)
    if os.path.isdir(path) or path.endswith(('/', os.sep)):
        os.makedirs(path, exist_ok=True)
        stem = f"neuralmi_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        taken = {f.split('.')[0] for f in os.listdir(path)}
        base, n = stem, 1
        while any(t == base or t.startswith(base + '_') for t in taken):
            base, n = f"{stem}_{n}", n + 1
        return os.path.join(path, base + MODEL_EXTENSION)
    root, ext = os.path.splitext(path)
    return path if ext else root + MODEL_EXTENSION


def with_model_labels(params: Dict[str, Any], **labels) -> Dict[str, Any]:
    """A copy of `params` whose saved network is also named by `labels`.

    Returns `params` itself when nothing is being saved.
    """
    if not params.get('save_best_model_path'):
        return params
    out = dict(params)
    out['_model_labels'] = {**(params.get('_model_labels') or {}), **labels}
    return out


def _label_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return '-'.join(_label_value(v) for v in value)
    return re.sub(r'[^\w.\-]+', '', str(value))


def model_file(params: Dict[str, Any]) -> Optional[str]:
    """The file one network is saved to: the call's path plus its labels."""
    path = params.get('save_best_model_path')
    if not path:
        return None
    labels = params.get('_model_labels') or {}
    if not labels:
        return path
    root, ext = os.path.splitext(path)
    suffix = '_'.join(f"{k}-{_label_value(v)}" for k, v in labels.items())
    return f"{root}_{suffix}{ext}"


def announce_saving_several(save_path: str) -> None:
    """Say that a call will save every network it trains, where, and at what cost."""
    root, ext = os.path.splitext(save_path)
    logger.info(
        f"save_best_model_path: this call trains several networks and saves every one of "
        f"them as {root}_<labels>{ext}. The labels name each network by its grid values and "
        f"run_id, plus its gamma and chunk, component, lag, channel pair or split where the "
        f"call has them. Each network's path is recorded as 'model_path' in result.runs. "
        f"Where a repeat trains several networks the paths are in "
        f"result.details[config_id]['trainings']. Every file holds a whole network and the "
        f"files can take a lot of disk space. To keep one model, rerun the configuration you "
        f"want on its own with this path."
    )


def saved_paths(raw: Dict[str, Any]) -> list:
    """The paths of every network saved while producing `raw`, a mode's component dict."""
    return [record['model_path'] for value in raw.values() if isinstance(value, list)
            for record in value if isinstance(record, dict) and record.get('model_path')]


def _saved(value: Any) -> Any:
    """``value`` as a saved network carries it: itself, or, for a class pickle cannot store (one defined
    inside a function), a note naming it that ``extract_embeddings`` reports."""
    try:
        pickle.dumps(value)
        return value
    except Exception:
        return _Unsaved(f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', repr(value))}")


class _Unsaved(str):
    """The name of a setting's value that could not be saved with a network."""


def save_network(model: torch.nn.Module, params: Dict[str, Any], build_keys) -> Optional[str]:
    """Save `model` with what rebuilding it needs, and return the path written."""
    path = model_file(params)
    if path is None:
        return None
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    build_params = {k: _saved(params[k]) for k in build_keys if k in params}
    torch.save({'state_dict': model.state_dict(), 'build_params': build_params}, path)
    logger.debug(f"Model saved to {path}.")
    return os.path.abspath(path)


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

    All samples are embedded in original order, with no subsampling and no
    shuffling, so the returned arrays are index-aligned with the input arrays.
    Inference runs in mini-batches to avoid OOM errors on large datasets.

    A saved model takes one of two forms:

    - **Bundled**: a dict with ``'state_dict'`` and ``'build_params'`` keys.
      ``build_params`` holds every architecture hyperparameter, so the critic
      is reconstructed automatically.
    - **Bare state_dict**: the caller must provide ``base_params`` describing
      the architecture the model was trained on.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (``.pt`` or ``.pth``).
    x_data : np.ndarray or torch.Tensor, or a tuple of two
        Input data for variable X.  Shape must match what the model was trained
        on (e.g., ``(n_samples, input_dim_x)`` for MLP, or
        ``(n_samples, n_channels, window_size)`` for CNN/GRU). The joint term
        of a difference quantity reads X as the pair ``(rest of X,
        conditioning set)``, and its saved network takes that tuple.
    y_data : np.ndarray or torch.Tensor
        Input data for variable Y.
    base_params : dict, optional
        Required when loading a bare state_dict, and then holding the settings
        :func:`~neural_mi.utils.build_critic` reads. With a bundled file it
        supplies what could not be saved, such as a ``custom_embedding_cls``
        defined inside a function.
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
        bp = {**loaded['build_params'], **(base_params or {})}
        unsaved = {k: v for k, v in bp.items() if isinstance(v, _Unsaved)}
        if unsaved:
            raise ValueError(
                f"'{model_path}' was trained with {', '.join(f'{k}={v}' for k, v in unsaved.items())}, "
                f"which could not be saved with it (a class defined inside a function cannot be). "
                f"Pass it as base_params={{{', '.join(repr(k) + ': ...' for k in unsaved)}}}."
            )
        logger.debug(f"Loaded new-format model from {model_path} "
                     f"(critic_type={bp.get('critic_type')}, "
                     f"embedding_model={bp.get('embedding_model')}).")
    else:
        # Old format, raw state dict
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
    critic = build_critic(setting(bp, 'critic_type'), bp)
    critic.load_state_dict(state_dict, strict=True)
    critic = critic.to(_device)
    critic.eval()

    # --- Prepare tensors ---
    pair = isinstance(x_data, tuple)
    x_t = tuple(float_tensor(part) for part in x_data) if pair else float_tensor(x_data)
    y_t = float_tensor(y_data)

    for part in (x_t if pair else (x_t,)):
        if part.shape[0] != y_t.shape[0]:
            raise ValueError(
                f"x_data and y_data must have the same number of samples, "
                f"got {part.shape[0]} and {y_t.shape[0]}."
            )

    # --- Extract embeddings in mini-batches, preserving sample order ---
    n = y_t.shape[0]
    zx_parts, zy_parts = [], []
    with torch.no_grad():
        for start in range(0, n, _EMBEDDING_BATCH):
            end = min(start + _EMBEDDING_BATCH, n)
            bx = (tuple(part[start:end].to(_device) for part in x_t) if pair
                  else x_t[start:end].to(_device))
            bzx, bzy = critic.get_embeddings(bx, y_t[start:end].to(_device))
            zx_parts.append(bzx.detach().cpu())
            zy_parts.append(bzy.detach().cpu())

    return torch.cat(zx_parts, dim=0).numpy(), torch.cat(zy_parts, dim=0).numpy()
