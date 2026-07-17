# neural_mi/analysis/task.py
"""
Contains the core, parallelizable training task function.
"""
import gc as _gc
import threading
import warnings
import weakref
import torch
import numpy as np
from typing import Dict, Any

from neural_mi.utils import build_critic, build_optimizer_and_scheduler, get_device, compute_cross_covariance_rotation
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
#
# The key is built from data_ptr()/id(), which identifies an *allocation*, not
# a specific object: if a source array is freed and a new one happens to be
# allocated at the same address, the key can collide. Each entry also stores a
# weakref to the original x_data/y_data so a hit can verify true object
# identity before being trusted; a mismatch (or a dead weakref) is treated as
# a miss and the dataset is rebuilt.
_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE: dict = {}       # {cache_key -> (PairedDataset, x_weakref, y_weakref)}
_DATASET_CACHE_MAXSIZE = 4      # LRU eviction when this is exceeded


def _safe_weakref(obj):
    """Return a weakref to obj, or None if obj is None or not weakly referenceable."""
    if obj is None:
        return None
    try:
        return weakref.ref(obj)
    except TypeError:
        return None

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
    # PretrainedBackboneEmbedding architecture parameters
    'pytorch_predefined', 'pretrained',
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
        _cache_entry = _DATASET_CACHE.get(_cache_key)
    if _cache_entry is not None:
        _cached_dataset, _x_ref, _y_ref = _cache_entry
        _x_is_live = _x_ref is None or _x_ref() is x_data
        _y_is_live = _y_ref is None or _y_ref() is y_data
        if _x_is_live and _y_is_live:
            dataset = _cached_dataset
        else:
            logger.debug("Dataset cache key collision (stale data_ptr/id) — rebuilding.")

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
                _DATASET_CACHE[_cache_key] = (dataset, _safe_weakref(x_data), _safe_weakref(y_data))
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
            params['input_height_x'] = _x.shape[2]
            params['input_width_x'] = _x.shape[3]
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
            params['input_height_y'] = _y.shape[2]
            params['input_width_y'] = _y.shape[3]
        else:
            params['input_dim_y'] = _y.shape[1] * _y.shape[2]

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
            height=params.get('input_height_x'),
            width=params.get('input_width_x'),
        )
        # Always build a dedicated decoder_y -- X and Y may differ in n_channels/
        # window_size/activation even when shared_encoder=True.
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
            height=params.get('input_height_y'),
            width=params.get('input_width_y'),
        )
        logger.debug(
            f"Built decoder_x ({type(decoder_x).__name__}) and decoder_y ({type(decoder_y).__name__}) "
            f"for use_decoder=True."
        )

    optimizer, scheduler = build_optimizer_and_scheduler(
        params, critic, decoder_x=decoder_x, decoder_y=decoder_y,
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
        track_spectral_history=params.get('track_spectral_history', False),
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
    _cache_entry = _DATASET_CACHE.get(_cache_key)
    _cached_dataset = _cache_entry[0] if _cache_entry is not None else None
    if _cached_dataset is not dataset:
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