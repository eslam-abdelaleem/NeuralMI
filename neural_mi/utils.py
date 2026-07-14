# neural_mi/utils.py

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as _lr_sched
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

import multiprocessing as mp
import os
import platform
import tempfile

from neural_mi.models.embeddings import (
    MLP, VariationalWrapper,
    CNN1D, CNN2D, GRU, LSTM, TCN, Transformer,
    PretrainedBackboneEmbedding,
)
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

def _shift_data(x_data: Any, y_data: Any, lag: int,
                x_processor_type: Optional[str], y_processor_type: str,
                sample_rate: Optional[float] = None) -> tuple:
    """Shifts data to apply a time lag between X and Y.

    Sign convention (shared by every branch below): for ``lag > 0``, Y is
    compared against its own future relative to X (X's early samples are
    paired with Y's later samples).

    Parameters
    ----------
    x_data : array-like
        Data for variable X.
    y_data : array-like
        Data for variable Y.
    lag : int or float
        The lag value. For 'spike' data, always seconds. For 'continuous'/
        'categorical' data, seconds if `sample_rate` is given, otherwise a raw
        sample-index offset.
    x_processor_type : str, optional
        X's processor type: one of 'continuous', 'categorical', 'spike', or
        None. Only consulted to detect the mixed-modality case (spike paired
        with non-spike); when both sides are non-spike, only
        `y_processor_type` determines how the shift is applied (both arrays
        get the same array-index treatment regardless of which specific
        non-spike type each one is).
    y_processor_type : str
        Y's processor type: one of 'continuous', 'categorical', or 'spike'.
    sample_rate : float, optional
        Samples per second for continuous/categorical data. If provided, lag is
        interpreted as seconds and converted to samples. If None, lag is treated
        as a raw sample-index offset and a warning is emitted, since the unit is
        then ambiguous relative to spike data (always seconds).
    """
    x_is_spike = x_processor_type == 'spike'
    y_is_spike = y_processor_type == 'spike'

    if x_is_spike and not y_is_spike:
        # Mixed modality: spike-X, non-spike-Y. Shift X's spike times forward
        # by `lag` (rather than shifting Y, which isn't spike data and has no
        # per-event timestamps to offset) -- this keeps the lag>0="Y is in
        # the future" convention: advancing X's clock by `lag` is equivalent
        # to comparing X's past against Y's present.
        logger.info(
            f"Mixed-modality lag: X is 'spike', Y is '{y_processor_type}'. "
            f"Shifting X's spike times by +{lag}s; Y is left at its original times."
        )
        x_shifted = [spikes + lag for spikes in x_data]
        return x_shifted, y_data

    if y_is_spike:
        # Y is spike (X may be spike too, or non-spike -- either way only Y's
        # spike times need shifting; X is unaffected in both sub-cases).
        if not x_is_spike:
            logger.info(
                f"Mixed-modality lag: Y is 'spike', X is '{x_processor_type}'. "
                f"Shifting Y's spike times by -{lag}s; X is left at its original times."
            )
        y_shifted = [spikes - lag for spikes in y_data]
        return x_data, y_shifted

    if y_processor_type in ('continuous', 'categorical'):
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
            # No sample_rate given: treat lag as a raw sample-index offset, but warn
            logger.warning(
                f"Lag units for '{y_processor_type}' data are ambiguous without a sample_rate. "
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

    return x_data, y_data

    
def build_critic(critic_type: str, embedding_params: Dict[str, Any],
                 custom_embedding_cls: Optional[type] = None) -> BaseCritic:
    """Builds and returns a critic model based on the provided parameters.

    This function expects `embedding_params` to be fully populated with
    defaults (e.g., via `ParameterValidator.apply_defaults`). It strictly
    accesses required parameters and will raise a KeyError if something is missing,
    preventing silent failures from missing defaults.

    For ``critic_type='hybrid'``, the decision head MLP that scores the
    concatenated embeddings can be configured independently of the embedding
    networks via two optional keys in ``embedding_params``:

    - ``hidden_dim_head`` (int or None): hidden width of the decision head.
      Defaults to ``None``, which resolves to ``min(64, hidden_dim)``.
    - ``n_layers_head`` (int or None): number of hidden layers in the decision
      head.  Defaults to ``None``, which resolves to ``max(1, n_layers - 1)``.

    These parameters have no effect for ``critic_type`` values other than
    ``'hybrid'``.
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
    elif model_type == 'cnn2d':
        EmbeddingModel = CNN2D
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
    elif model_type == 'pretrained_backbone':
        EmbeddingModel = PretrainedBackboneEmbedding
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

    # CNN1D, CNN2D, and all sequence models use n_channels as input_dim
    # (they operate on the channel dimension, not a flattened feature vector).
    # pretrained_backbone also uses n_channels as input_dim.
    # MLP (and custom cls) use the fully-flattened input_dim.
    _sequential_types = {'cnn', 'cnn2d', 'gru', 'lstm', 'tcn', 'transformer',
                         'pretrained_backbone'}
    if model_type in _sequential_types:
        input_dim_x, input_dim_y = embedding_params['n_channels_x'], embedding_params['n_channels_y']
        if model_type in ['cnn', 'tcn', 'cnn2d']:
            model_kwargs['kernel_size'] = embedding_params.get('kernel_size', 7 if model_type == 'cnn' else 3)
        if model_type in ['gru', 'lstm']:
            model_kwargs['bidirectional'] = embedding_params.get('bidirectional', False)
        if model_type == 'transformer':
            model_kwargs['nhead'] = embedding_params.get('nhead', 4)
        if model_type == 'pretrained_backbone':
            model_kwargs['pytorch_predefined'] = embedding_params.get('pytorch_predefined')
            model_kwargs['pretrained'] = embedding_params.get('pretrained', False)
    else:  # MLP or a custom class — both take the fully-flattened input_dim
        input_dim_x, input_dim_y = embedding_params['input_dim_x'], embedding_params['input_dim_y']
        if not custom_embedding_cls:
            # These are specific to the built-in MLP. A custom class only receives
            # the universal (hidden_dim, embed_dim, n_layers) contract, so it need
            # not accept regularisation kwargs it may know nothing about.
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

    # Build the base (deterministic) encoders.
    model_kwargs_y = model_kwargs.copy()
    net_x_base = EmbeddingModel(input_dim_x, **model_kwargs)
    net_y_base = net_x_base if shared_encoder else EmbeddingModel(input_dim_y, **model_kwargs_y)

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
    _first_hidden = hidden_dim[0] if isinstance(hidden_dim, list) else hidden_dim
    _last_hidden  = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
    first_layer_params = input_dim_x * _first_hidden
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
        _head_hidden_dim = embedding_params.get('hidden_dim_head') or min(64, _last_hidden)
        _head_n_layers = embedding_params.get('n_layers_head') or max(1, n_layers - 1)
        decision_head = MLP(input_dim=decision_head_input_dim, hidden_dim=_head_hidden_dim, embed_dim=1, n_layers=_head_n_layers)
        return HybridCritic(embedding_net_x=net_x, embedding_net_y=net_y, decision_head=decision_head, **critic_kwargs)
    elif critic_type == 'concat':
        concat_input_dim = input_dim_x + input_dim_y
        concat_net = MLP(input_dim=concat_input_dim, hidden_dim=hidden_dim, embed_dim=1, n_layers=n_layers)
        return ConcatCritic(embedding_net=concat_net, **critic_kwargs)
    else:
        raise ValueError(f"Unknown critic_type: {critic_type}")


_OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
}
_SCHEDULER_NAMES = {'cosine', 'step', 'plateau', 'cosine_warmup'}


def build_optimizer_and_scheduler(
    params: Dict[str, Any],
    critic: nn.Module,
    decoder_x: Optional[nn.Module] = None,
    decoder_y: Optional[nn.Module] = None,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """Build the optimizer and optional LR scheduler from base_params.

    Shared by task.py (per-sweep training, which may include reconstruction
    decoders and a per-head LR multiplier) and precision.py (a single
    train-once-evaluate-many baseline model, no decoders). This function
    expects ``params`` to be fully populated (e.g. via
    ``ParameterValidator.apply_defaults``): it accesses ``learning_rate`` and
    ``n_epochs`` strictly and will raise a ``KeyError`` if either is missing,
    rather than silently substituting a different default than
    ``BASE_PARAMS_SCHEMA``'s.

    Parameters
    ----------
    params : dict
        Must contain ``learning_rate``. May contain ``optimizer``,
        ``optimizer_params``, ``lr_head_multiplier``, ``scheduler``,
        ``scheduler_params`` (and ``n_epochs`` if a scheduler is requested).
    critic : nn.Module
        The critic whose parameters (plus any decoder parameters) are
        optimized.
    decoder_x, decoder_y : nn.Module, optional
        Reconstruction decoders whose parameters, if given, are added to the
        optimizer alongside the critic's. ``None`` (default) omits them.

    Returns
    -------
    tuple[torch.optim.Optimizer, scheduler or None]
    """
    _opt_val = params.get('optimizer', 'adam')
    if isinstance(_opt_val, type):
        OptCls = _opt_val
    else:
        OptCls = _OPTIMIZERS.get(str(_opt_val).lower())
        if OptCls is None:
            raise ValueError(
                f"Unknown optimizer '{_opt_val}'. "
                f"Supported names: {list(_OPTIMIZERS.keys())}. "
                f"You can also pass a torch.optim.Optimizer subclass directly."
            )

    # Collect all trainable parameters (critic + optional decoders).
    # When lr_head_multiplier is set and the critic has a decision_head (i.e. hybrid),
    # split into two param groups so the head can train at a different rate.
    _base_lr = params['learning_rate']
    _head_mult = params.get('lr_head_multiplier')
    _decoder_params = []
    if decoder_x is not None:
        _decoder_params.extend(decoder_x.parameters())
    if decoder_y is not None:
        _decoder_params.extend(decoder_y.parameters())

    if _head_mult is not None and _head_mult != 1.0 and hasattr(critic, 'decision_head'):
        _head_ids = {id(p) for p in critic.decision_head.parameters()}
        _encoder_params = [p for p in critic.parameters() if id(p) not in _head_ids]
        _encoder_params.extend(_decoder_params)
        _param_groups = [
            {'params': _encoder_params,                        'lr': _base_lr},
            {'params': list(critic.decision_head.parameters()), 'lr': _base_lr * _head_mult},
        ]
        optimizer = OptCls(_param_groups, **params.get('optimizer_params', {}))
    else:
        _all_params = list(critic.parameters()) + _decoder_params
        optimizer = OptCls(_all_params, lr=_base_lr, **params.get('optimizer_params', {}))

    _sched_val = params.get('scheduler', None)
    scheduler = None
    if _sched_val is not None:
        _sched_params = params.get('scheduler_params', {})
        n_epochs = params['n_epochs']
        if isinstance(_sched_val, type):
            scheduler = _sched_val(optimizer, **_sched_params)
        elif _sched_val == 'cosine':
            scheduler = _lr_sched.CosineAnnealingLR(optimizer, T_max=n_epochs, **_sched_params)
        elif _sched_val == 'step':
            scheduler = _lr_sched.StepLR(optimizer, step_size=max(1, n_epochs // 3), **_sched_params)
        elif _sched_val == 'plateau':
            scheduler = _lr_sched.ReduceLROnPlateau(optimizer, mode='max', **_sched_params)
        elif _sched_val == 'cosine_warmup':
            warmup = max(1, int(n_epochs * 0.1))
            warmup_sched = _lr_sched.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
            cosine_sched = _lr_sched.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup))
            scheduler = _lr_sched.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
            )
        else:
            raise ValueError(
                f"Unknown scheduler '{_sched_val}'. "
                f"Supported names: {sorted(_SCHEDULER_NAMES)}. "
                f"You can also pass a torch.optim.lr_scheduler class directly."
            )
    return optimizer, scheduler


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


def compute_cross_covariance_rotation(
    zx: np.ndarray,
    zy: np.ndarray,
    whitening: Optional[str] = 'std'
) -> Dict[str, np.ndarray]:
    """Computes a rotation that orders embedding dimensions by shared variance.

    The rotation matrices U and V are derived from the SVD of the (optionally
    whitened) cross-covariance matrix C = ZX_w.T @ ZY_w / (N-1).  The whitening
    is applied **only** to compute the rotation axes — it is NOT applied to the
    returned embeddings.  What is returned is ZX_centered @ U and ZY_centered @ V,
    i.e. the original-scale embeddings simply re-expressed in the new basis.

    This means dimension 0 of the rotated embeddings captures the most shared
    variance between the two spaces, dimension 1 the second most, and so on —
    consistent with how the Participation Ratio (PR) dimensionality estimate
    orders dimensions.

    Parameters
    ----------
    zx, zy : np.ndarray
        Embeddings, each of shape (N, d).  May also be passed as torch.Tensor;
        they will be converted automatically.
    whitening : {'std', 'zca', None}, optional
        Normalization applied before computing the cross-covariance (default
        ``'std'``).  Matches the default used by
        :func:`compute_cross_covariance_spectrum` so that the rotation is
        consistent with PR-based dimensionality estimates.
        - ``'std'``: divide each dimension by its standard deviation.
        - ``'zca'``: full ZCA whitening (sphering); requires N >> d for stability.
        - ``None``: no whitening; rotation reflects raw shared variance.

    Returns
    -------
    dict with keys:

    ``'zx_rotated'`` : np.ndarray, shape (N, d)
        Centered ZX projected onto the left singular vectors U.
    ``'zy_rotated'`` : np.ndarray, shape (N, d)
        Centered ZY projected onto the right singular vectors V.
    ``'singular_values'`` : np.ndarray, shape (min(d_x, d_y),)
        Singular values of the (whitened) cross-covariance, largest first.
    ``'rotation_x'`` : np.ndarray, shape (d_x, min(d_x, d_y))
        Left singular vectors U.  Apply as ``ZX_new @ U`` to project new data.
    ``'rotation_y'`` : np.ndarray, shape (d_y, min(d_x, d_y))
        Right singular vectors V.  Apply as ``ZY_new @ V`` to project new data.
    """
    # Accept both torch.Tensor and np.ndarray
    if hasattr(zx, 'detach'):
        zx = zx.detach().cpu().float().numpy()
    if hasattr(zy, 'detach'):
        zy = zy.detach().cpu().float().numpy()
    zx = np.asarray(zx, dtype=np.float64)
    zy = np.asarray(zy, dtype=np.float64)

    # Center (mean-subtract per dimension)
    zx_c = zx - zx.mean(axis=0, keepdims=True)
    zy_c = zy - zy.mean(axis=0, keepdims=True)

    N = zx_c.shape[0]
    d_x, d_y = zx_c.shape[1], zy_c.shape[1]
    if N <= 1:
        return {
            'zx_rotated': zx_c,
            'zy_rotated': zy_c,
            'singular_values': np.array([]),
            'rotation_x': np.eye(d_x),
            'rotation_y': np.eye(d_y),
        }

    # Whiten copies for computing rotation axes only — zx_c / zy_c are unchanged
    zx_w, zy_w = zx_c.copy(), zy_c.copy()
    if whitening == 'std':
        std_x = zx_w.std(axis=0, keepdims=True)
        std_y = zy_w.std(axis=0, keepdims=True)
        zx_w = zx_w / np.where(std_x > 1e-8, std_x, 1.0)
        zy_w = zy_w / np.where(std_y > 1e-8, std_y, 1.0)
    elif whitening == 'zca':
        def _zca(Z):
            cov = (Z.T @ Z) / (Z.shape[0] - 1)
            Uz, Sz, _ = np.linalg.svd(cov)
            W = Uz @ np.diag(1.0 / np.sqrt(np.maximum(Sz, 1e-8))) @ Uz.T
            return Z @ W.T
        zx_w, zy_w = _zca(zx_w), _zca(zy_w)
    elif whitening is not None:
        raise ValueError(f"Unknown whitening: '{whitening}'. Expected 'std', 'zca', or None.")

    cov_xy = (zx_w.T @ zy_w) / (N - 1)
    U, s, Vt = np.linalg.svd(cov_xy, full_matrices=False)
    V = Vt.T  # (d_y, min(d_x, d_y))

    return {
        'zx_rotated': zx_c @ U,   # centered, original scale, rotated
        'zy_rotated': zy_c @ V,
        'singular_values': s,
        'rotation_x': U,
        'rotation_y': V,
    }


def compute_spectral_metrics(spectrum: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Computes dimensionality metrics from singular values."""
    s = np.array(spectrum)
    s = s[s > eps]
    
    metrics = {}
    
    # 1. Variance/Energy-based PR
    lam = s**2
    metrics["pr_eig"] = (lam.sum())**2 / (lam**2).sum() if lam.sum() > 0 else 0.0

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


def anscombe_transform(counts: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Canonical Anscombe variance-stabilizing transform for count data: ``2*sqrt(x + 3/8)``.

    Maps heteroscedastic (Poisson-like) counts to an approximately unit-variance
    scale so that downstream per-channel standard deviations are comparable
    across channels with different firing rates. This is the single canonical
    stabilizer for the library; any binned-spike processing that needs
    variance stabilization should call this rather than reimplementing it.

    Parameters
    ----------
    counts : np.ndarray or torch.Tensor
        Non-negative count data, any shape.

    Returns
    -------
    Same type and shape as ``counts``.
    """
    if torch.is_tensor(counts):
        return 2.0 * torch.sqrt(counts.clamp(min=0) + 0.375)
    arr = np.asarray(counts, dtype=np.float64)
    return 2.0 * np.sqrt(np.clip(arr, 0, None) + 0.375)