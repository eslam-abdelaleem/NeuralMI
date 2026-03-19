# neural_mi/analysis/task.py
"""
Contains the core, parallelizable training task function.
"""
import gc as _gc
import threading
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as _lr_sched
import numpy as np
from typing import Dict, Any, Tuple

from neural_mi.utils import build_critic, get_device
from neural_mi.estimators import ESTIMATORS
from neural_mi.training.trainer import Trainer
from neural_mi.logger import logger
from neural_mi.data.handler import create_dataset, PairedDataset

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
    optimizer = OptCls(critic.parameters(), lr=params['learning_rate'],
                       **params.get('optimizer_params', {}))

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
    )

    # Intercept save_best_model_path to use the extended format
    # which includes build_params alongside state_dict for later extract_embeddings()
    _save_path = params.get('save_best_model_path')
    _BUILD_PARAMS_KEYS = [
        'critic_type', 'embedding_model', 'hidden_dim', 'embedding_dim', 'n_layers',
        'input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y',
        'use_variational', 'shared_encoder',
        'kernel_size', 'bidirectional', 'nhead', 'max_n_batches',
    ]

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
        scheduler=scheduler,
    )

    # Save model in extended format {'state_dict': ..., 'build_params': {...}}
    if _save_path:
        build_params = {k: params[k] for k in _BUILD_PARAMS_KEYS if k in params}
        torch.save({'state_dict': trainer.model.state_dict(), 'build_params': build_params},
                   _save_path)
        logger.debug(f"Model saved (extended format) to {_save_path}.")

    # Optionally extract embeddings from the trained model
    if params.get('return_embeddings', False):
        _all_x = dataset.x_data
        _all_y = dataset.y_data
        if _all_y is None:
            logger.warning("return_embeddings=True but y_data is None. Skipping embedding extraction.")
        else:
            trainer.model.eval()
            with torch.no_grad():
                _max_emb = params.get('max_eval_samples', 5000)
                _n = _all_x.shape[0]
                if _n > _max_emb:
                    _idx = np.random.choice(_n, _max_emb, replace=False)
                    _all_x = _all_x[_idx]
                    _all_y = _all_y[_idx]
                _zx, _zy = trainer.model.get_embeddings(_all_x.to(device), _all_y.to(device))
                results['embeddings_x'] = _zx.detach().cpu().numpy()
                results['embeddings_y'] = _zy.detach().cpu().numpy()

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