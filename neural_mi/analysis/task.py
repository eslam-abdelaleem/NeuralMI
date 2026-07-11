# neural_mi/analysis/task.py
"""
Contains the core, parallelizable training task function.
"""
import gc as _gc
import threading
import warnings
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as _lr_sched
import numpy as np
from typing import Dict, Any, Tuple

from neural_mi.utils import build_critic, get_device, compute_cross_covariance_rotation
from neural_mi.estimators import ESTIMATORS
from neural_mi.training.trainer import Trainer
from neural_mi.logger import logger
from neural_mi.data.handler import create_dataset, PairedDataset

# Batch size used for embedding extraction (no effect on training).
# Large enough to keep GPU utilisation high; small enough to avoid OOM.
_EMBEDDING_BATCH = 512

# ---------------------------------------------------------------------------
# Module-level dataset cache
# ---------------------------------------------------------------------------
# Keyed by (x_data_ptr, y_data_ptr, frozenset_of_construction_params).
# Only *static* (PairedDataset) instances are cached because temporal datasets
# are mutated in-place by time_shift() during training and cannot be safely
# shared across sequential tasks.
#
# In sequential sweeps where data and processor params are constant across
# tasks, this means create_dataset() is called once and every subsequent task
# reuses the pre-built object — eliminating N-1 redundant tensor copies.
#
# In multiprocessing (spawn) mode each worker starts with an empty cache; the
# first task in a worker populates it and later tasks in the same worker
# benefit automatically.
_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: dict = {}       # {cache_key -> PairedDataset}
_DATASET_CACHE_MAXSIZE = 4      # LRU eviction when this is exceeded

# Keys that fully determine dataset construction (anything else is a model/
# training hyperparameter and does NOT affect the dataset).
_DATASET_CONSTRUCTION_KEYS = frozenset([
    'processor_type_x', 'processor_type_y',
    'processor_params_x', 'processor_params_y',
    'dataset_device',
])


def _make_hashable(v):
    """Recursively convert dicts/lists to hashable tuples."""
    if isinstance(v, dict):
        return tuple(sorted((k, _make_hashable(w)) for k, w in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_make_hashable(i) for i in v)
    return v


def _dataset_cache_key(x_data, y_data, params: dict):
    """Build a cache key from data identity + construction params."""
    dp_x = x_data.data_ptr() if isinstance(x_data, torch.Tensor) else id(x_data)
    if y_data is None:
        dp_y = None
    elif isinstance(y_data, torch.Tensor):
        dp_y = y_data.data_ptr()
    else:
        dp_y = id(y_data)
    construction = tuple(sorted(
        (k, _make_hashable(params.get(k))) for k in _DATASET_CONSTRUCTION_KEYS
    ))
    return (dp_x, dp_y, construction)


# Keys needed to reconstruct a saved model's architecture for extract_embeddings().
# Exposed at module scope so external code can inspect or verify the set.
_BUILD_PARAMS_KEYS = [
    'critic_type', 'embedding_model', 'hidden_dim', 'embedding_dim', 'n_layers',
    'input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y',
    'use_variational', 'shared_encoder',
    'kernel_size', 'bidirectional', 'nhead', 'max_n_batches',
    'dropout', 'norm_layer', 'use_spectral_norm',
    'use_decoder', 'decoder_weight', 'decoder_weight_x', 'decoder_weight_y',
    'decoder_output_activation_x', 'decoder_output_activation_y',
    # Inductive-bias architecture parameters
    'use_depthwise', 'n_sinc_filters', 'feature_fusion',
    'pytorch_predefined', 'pretrained',
    # Modality metadata injected from processor_params (needed to reconstruct
    # SpikePhysicsEmbedding and SincEmbedding architectures)
    'sample_rate_x', 'sample_rate_y', 'no_spike_value', 'embedding_window_size',
]


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

    # ------------------------------------------------------------------
    # Resolve dataset_device: 'auto' → use the compute device; anything
    # else is forwarded verbatim ('cpu', 'cuda', 'mps', …).
    # ------------------------------------------------------------------
    _compute_device = get_device(params.get('device'))
    _raw_dd = params.get('dataset_device', 'cpu')
    _data_device: str = str(_compute_device) if _raw_dd == 'auto' else (_raw_dd or 'cpu')

    # ------------------------------------------------------------------
    # Dataset construction — with module-level cache for static datasets.
    # ------------------------------------------------------------------
    _cache_key = _dataset_cache_key(x_data, y_data,
                                    {**params, 'dataset_device': _data_device})
    dataset = None
    with _DATASET_CACHE_LOCK:
        dataset = _DATASET_CACHE.get(_cache_key)

    if dataset is None:
        dataset = create_dataset(
            x_data, y_data,
            processor_type_x=params.get('processor_type_x'),
            processor_type_y=params.get('processor_type_y'),
            processor_params_x=params.get('processor_params_x'),
            processor_params_y=params.get('processor_params_y'),
            data_device=_data_device,
        )
        # Only cache immutable static datasets — temporal datasets are mutated
        # by time_shift() during training and must not be shared.
        if isinstance(dataset, PairedDataset):
            with _DATASET_CACHE_LOCK:
                if len(_DATASET_CACHE) >= _DATASET_CACHE_MAXSIZE:
                    # Evict the oldest entry (dict preserves insertion order).
                    _DATASET_CACHE.pop(next(iter(_DATASET_CACHE)))
                _DATASET_CACHE[_cache_key] = dataset
            logger.debug("Dataset cached (key=%s).", str(_cache_key)[:80])
        else:
            logger.debug("Temporal dataset — skipping cache (mutable via time_shift).")
    else:
        logger.debug("Dataset cache hit — reusing pre-built dataset.")

    # Models that natively support 4-D input (N, C, H, W):
    #   'cnn2d' — Conv2d + AdaptiveAvgPool2d; spatial structure preserved.
    #   'mlp'   — flattens C*H*W into a feature vector; spatial structure ignored.
    # Models that require 3-D input (N, C, W):
    #   'cnn'        — raises ValueError for 4-D (ambiguous channel/spatial axes).
    #   sequence models (gru, lstm, tcn, transformer) — emit UserWarning; their
    #                   forward() methods expect 3-D and will fail at the first batch
    #                   if the user proceeds.  Use 'cnn2d' or 'mlp' instead.
    _4D_NATIVE = {'cnn2d', 'mlp', 'pretrained_backbone'}

    if dataset.x_data is not None and hasattr(dataset.x_data, 'shape'):
        _x = dataset.x_data
        params['n_channels_x'] = _x.shape[1]
        if _x.ndim == 4:
            params['input_dim_x'] = _x.shape[1] * _x.shape[2] * _x.shape[3]
            _emb = params.get('embedding_model', 'mlp')
            if _emb == 'cnn':
                raise ValueError(
                    f"embedding_model='cnn' (CNN1D) does not support 4-D input "
                    f"(shape {tuple(_x.shape)}). "
                    "Use embedding_model='cnn2d' to preserve spatial structure, "
                    "or embedding_model='mlp' to process flattened C×H×W features."
                )
            elif _emb not in _4D_NATIVE:
                warnings.warn(
                    f"embedding_model='{_emb}' received 4-D input "
                    f"(shape {tuple(_x.shape)}). "
                    "Spatial dimensions H×W are not preserved by this model. "
                    "Consider embedding_model='cnn2d' for spatially-structured data.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            params['input_dim_x'] = _x.shape[1] * _x.shape[2]

    if dataset.y_data is not None and hasattr(dataset.y_data, 'shape'):
        _y = dataset.y_data
        params['n_channels_y'] = _y.shape[1]
        if _y.ndim == 4:
            params['input_dim_y'] = _y.shape[1] * _y.shape[2] * _y.shape[3]
        else:
            params['input_dim_y'] = _y.shape[1] * _y.shape[2]

    # ------------------------------------------------------------------
    # Inject modality metadata for inductive-bias embedding constructors.
    # Extracted from processor_params here (after dataset is built) so
    # build_critic() can pass them to model constructors without needing
    # direct access to the dataset or processor_params dicts.
    # Keys are prefixed/named to avoid collisions with existing params.
    # ------------------------------------------------------------------
    _pp_x = params.get('processor_params_x') or {}
    _pp_y = params.get('processor_params_y') or {}
    params['sample_rate_x'] = _pp_x.get('sample_rate')
    params['sample_rate_y'] = _pp_y.get('sample_rate')
    # no_spike_value: default -1.0 matches SpikeWindowDataset's default sentinel
    params['no_spike_value'] = _pp_x.get('no_spike_value', _pp_y.get('no_spike_value', -1.0))
    # embedding_window_size: window duration in seconds (for firing-rate computation)
    params['embedding_window_size'] = _pp_x.get('window_size', _pp_y.get('window_size'))

    if params.get('custom_critic') is not None:
        critic = params['custom_critic']
        logger.debug("Using pre-initialized custom critic model. Model architecture parameters in 'base_params' will be ignored.")
    else:
        critic = build_critic(params.get('critic_type', 'separable'),
                              params,
                              params.get('custom_embedding_cls'))

    # Build decoders if use_decoder is enabled
    decoder_x = None
    decoder_y = None
    if params.get('use_decoder', False):
        from neural_mi.models.decoders import build_decoder
        _embedding_model = params.get('embedding_model', 'mlp')
        _embed_dim = params.get('embedding_dim', params.get('hidden_dim', 64))
        _hidden_dim = params.get('hidden_dim', 64)
        _n_layers = params.get('n_layers', 2)
        _dec_act_x = params.get('decoder_output_activation_x', 'linear')
        _dec_act_y = params.get('decoder_output_activation_y', 'linear')

        _n_channels_x = params.get('n_channels_x', 1)
        _n_channels_y = params.get('n_channels_y', 1)
        _input_dim_x = params.get('input_dim_x', _n_channels_x)
        _input_dim_y = params.get('input_dim_y', _n_channels_y)
        _window_size_x = max(1, _input_dim_x // _n_channels_x)
        _window_size_y = max(1, _input_dim_y // _n_channels_y)

        decoder_x = build_decoder(
            embedding_model=_embedding_model,
            embed_dim=_embed_dim,
            hidden_dim=_hidden_dim,
            n_channels=_n_channels_x,
            window_size=_window_size_x,
            n_layers=_n_layers,
            output_activation=_dec_act_x,
            kernel_size=params.get('kernel_size', 7),
            nhead=params.get('nhead', 4),
        )
        # Separate decoder for Y if asymmetric architecture or different data type
        # For shared_encoder=True, we still build separate decoders (X and Y may differ in n_channels)
        if _n_channels_y != _n_channels_x or _window_size_y != _window_size_x or _dec_act_x != _dec_act_y:
            decoder_y = build_decoder(
                embedding_model=_embedding_model,
                embed_dim=_embed_dim,
                hidden_dim=_hidden_dim,
                n_channels=_n_channels_y,
                window_size=_window_size_y,
                n_layers=_n_layers,
                output_activation=_dec_act_y,
                kernel_size=params.get('kernel_size', 7),
                nhead=params.get('nhead', 4),
            )
        else:
            decoder_y = build_decoder(
                embedding_model=_embedding_model,
                embed_dim=_embed_dim,
                hidden_dim=_hidden_dim,
                n_channels=_n_channels_y,
                window_size=_window_size_y,
                n_layers=_n_layers,
                output_activation=_dec_act_y,
                kernel_size=params.get('kernel_size', 7),
                nhead=params.get('nhead', 4),
            )
        logger.debug(
            f"Built decoder_x ({type(decoder_x).__name__}) and decoder_y ({type(decoder_y).__name__}) "
            f"for use_decoder=True."
        )

    _OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
    }
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

    _SCHEDULER_NAMES = {'cosine', 'step', 'plateau', 'cosine_warmup'}
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
            scheduler = _lr_sched.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
        else:
            raise ValueError(
                f"Unknown scheduler '{_sched_val}'. "
                f"Supported names: {sorted(_SCHEDULER_NAMES)}. "
                f"You can also pass a torch.optim.lr_scheduler class directly."
            )

    device = _compute_device  # already resolved above

    # Resolve augmentation params: shared baseline, per-variable override.
    # None means "fall back to shared"; {} means "explicitly no augmentation".
    _aug_shared = params.get('augmentation_params', {})
    _aug_x = params.get('augmentation_params_x')
    _aug_y = params.get('augmentation_params_y')
    aug_x = _aug_x if _aug_x is not None else _aug_shared
    aug_y = _aug_y if _aug_y is not None else _aug_shared

    # Inject custom smoothing into Trainer init
    trainer = Trainer(
        model=critic.to(device),
        estimator_fn=ESTIMATORS[params['estimator_name']],
        optimizer=optimizer,
        device=device,
        use_variational=params.get('use_variational', False),
        beta=params.get('beta', 1024.0),
        estimator_params=params.get('estimator_params'),
        custom_smoothing_fn=params.get('custom_smoothing_fn'),
        spectral_whitening=params.get('spectral_whitening', 'std'),
        gradient_clip_val=params.get('gradient_clip_val', None),
        decoder_x=decoder_x,
        decoder_y=decoder_y,
        decoder_weight_x=params.get('decoder_weight_x', params.get('decoder_weight', 1.0)),
        decoder_weight_y=params.get('decoder_weight_y', params.get('decoder_weight', 1.0)),
        decoder_output_activation_x=params.get('decoder_output_activation_x', 'linear'),
        decoder_output_activation_y=params.get('decoder_output_activation_y', 'linear'),
        use_amp=params.get('use_amp', 'auto'),
        augmentation_params_x=aug_x,
        augmentation_params_y=aug_y,
    )

    # Intercept save_best_model_path to use the extended format
    # which includes build_params alongside state_dict for later extract_embeddings()
    _save_path = params.get('save_best_model_path')

    # Inject memory, logging, and spectral metrics parameters into train
    results = trainer.train(
        dataset,
        params['n_epochs'],
        params['batch_size'],
        train_fraction=params.get('train_fraction', 0.9),
        n_test_blocks=params.get('n_test_blocks', 5),
        random_time_shifting=params.get('random_time_shifting', False),
        epochs_to_max_shift=params.get('epochs_to_max_shift', 5),
        patience=params['patience'],
        smoothing_sigma=params.get('smoothing_sigma', 1.0),
        median_window=params.get('median_window', 5),
        min_improvement=params.get('min_improvement', 0.001),
        run_id=run_id,
        output_units=params.get('output_units', 'nats'),
        verbose=params.get('verbose', False),
        show_progress=params.get('show_progress', True),
        save_best_model_path=None,  # we handle saving ourselves below (new format)
        split_mode=params.get('split_mode', 'blocked'),
        train_indices=params.get('train_indices'),
        test_indices=params.get('test_indices'),
        max_eval_samples=params.get('max_eval_samples', 5000),
        split_gap_fraction=params.get('split_gap_fraction', 0.5),
        train_subset_size=params.get('train_subset_size'),
        track_spectral_metrics=params.get('track_spectral_metrics', False),
        spectral_output=params.get('spectral_output', 'default'),
        return_spectrum=params.get('return_spectrum', False),
        max_index_reduction=params.get('max_index_reduction', 0.05),
        eval_train=params.get('eval_train', False),
        peak_fraction=params.get('peak_fraction', 1.0),
        scheduler=scheduler,
        track_embeddings=params.get('track_embeddings', False),
        return_rotated_embeddings=params.get('return_rotated_embeddings', False),
        rotated_embeddings_whitening=params.get('rotated_embeddings_whitening', 'std'),
        rotated_embeddings_per_epoch=params.get('rotated_embeddings_per_epoch', False),
        return_rotation_matrices=params.get('return_rotation_matrices', False),
    )

    # Save model in extended format {'state_dict': ..., 'build_params': {...}}
    if _save_path:
        build_params = {k: params[k] for k in _BUILD_PARAMS_KEYS if k in params}
        torch.save({'state_dict': trainer.model.state_dict(), 'build_params': build_params},
                   _save_path)
        logger.debug(f"Model saved (extended format) to {_save_path}.")

    # Optionally extract embeddings from the trained model.
    # Uses the full dataset in original sample order — no capping, no shuffling —
    # so the returned arrays align index-for-index with the caller's raw data.
    if params.get('return_embeddings', False):
        _all_x = dataset.x_data
        _all_y = dataset.y_data
        if _all_y is None:
            logger.warning("return_embeddings=True but y_data is None. Skipping embedding extraction.")
        else:
            trainer.model.eval()
            _n = _all_x.shape[0]
            _zx_parts, _zy_parts = [], []
            with torch.no_grad():
                for _start in range(0, _n, _EMBEDDING_BATCH):
                    _end = min(_start + _EMBEDDING_BATCH, _n)
                    _bzx, _bzy = trainer.model.get_embeddings(
                        _all_x[_start:_end].to(device),
                        _all_y[_start:_end].to(device),
                    )
                    _zx_parts.append(_bzx.detach().cpu())
                    _zy_parts.append(_bzy.detach().cpu())
            results['embeddings_x'] = torch.cat(_zx_parts, dim=0).numpy()
            results['embeddings_y'] = torch.cat(_zy_parts, dim=0).numpy()
            logger.debug(
                f"Extracted embeddings for all {_n} samples in original order "
                f"(shape: {results['embeddings_x'].shape})."
            )

            if params.get('return_rotated_embeddings', False):
                if params.get('critic_type', 'separable') == 'concat':
                    warnings.warn(
                        "return_rotated_embeddings=True has no effect for ConcatCritic, "
                        "which has no separate embedding networks. Skipping rotation.",
                        UserWarning, stacklevel=2,
                    )
                else:
                    _whitening = params.get('rotated_embeddings_whitening', 'std')
                    _rot = compute_cross_covariance_rotation(
                        results['embeddings_x'], results['embeddings_y'],
                        whitening=_whitening,
                    )
                    results['embeddings_x_rotated'] = _rot['zx_rotated']
                    results['embeddings_y_rotated'] = _rot['zy_rotated']
                    results['embeddings_rotation_singular_values'] = _rot['singular_values']
                    if params.get('return_rotation_matrices', False):
                        results['embeddings_rotation_x'] = _rot['rotation_x']
                        results['embeddings_rotation_y'] = _rot['rotation_y']
                    logger.debug("Computed rotated embeddings (whitening=%r).", _whitening)

    return_params = params.copy()
    return_params.pop('custom_critic', None)
    return_params.pop('custom_embedding_cls', None)
    final_result = {**return_params, **results}

    # ------------------------------------------------------------------
    # Release device-bound objects so the backend allocator can reclaim
    # memory.  Model + optimizer are always on the compute device; dataset
    # tensors are on data_device (usually CPU, so this is a no-op for them).
    # ------------------------------------------------------------------
    del trainer, critic, optimizer
    if decoder_x is not None:
        del decoder_x
    if decoder_y is not None:
        del decoder_y
    # Only delete the dataset reference if it is NOT in the shared cache —
    # cached datasets are intentionally kept alive for reuse by future tasks.
    _cached = _DATASET_CACHE.get(_cache_key)
    if _cached is not dataset:
        del dataset
    if scheduler is not None:
        del scheduler

    _device_type = getattr(device, 'type', 'cpu')
    if _device_type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif _device_type == 'mps':
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    _gc.collect()


    return final_result