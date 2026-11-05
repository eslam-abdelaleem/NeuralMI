# neural_mi/defaults.py
"""
Centralized definition of default parameters and allowed keys for validation.
"""
import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Allowed parameters for base_params
# Each entry is (type_check, min_value or validation_function)
BASE_PARAMS_SCHEMA = {
    # Trainer parameters
    'n_epochs': {'type': int, 'min': 1, 'default': 300},
    'learning_rate': {'type': float, 'min': 0.0, 'default': 5e-4},
    'batch_size': {'type': int, 'min': 1, 'default': 128},
    # Training stops once the smoothed held-out curve has not improved for
    # `patience` epochs, and n_epochs caps a curve that keeps improving. Shifted
    # windows keep the curve rising for longer, and a curve without shifting often
    # tops early and falls.
    'patience': {'type': int, 'min': 0, 'default': 30},
    'output_units': {'type': str, 'default': 'bits'},
    # Real-time grids for X and Y, carried so that windowing deferred to the
    # task layer sees the same grid the eager path gets via create_dataset.
    'x_time': {'type': (object, type(None))},
    'y_time': {'type': (object, type(None))},
    'verbose': {'type': bool, 'default': False},
    'show_progress': {'type': bool, 'default': True},
    'device': {'type': (str, type(None), torch.device), 'default': None},
    'split_mode': {'type': str, 'default': 'blocked'},
    # Every epoch the windows a processor cuts start one offset later, the same
    # for every stream, drawn within one step (neural_mi/data/plan.py).
    'shift_windows': {'type': bool, 'default': True},
    'smoothing_sigma': {'type': float, 'min': 0.0, 'default': 1.0},
    'median_window': {'type': int, 'min': 1, 'default': 5},
    'min_improvement': {'type': float, 'min': 0.0, 'default': 0.001},
    'max_eval_samples': {'type': int, 'min': 1, 'default': 5000},
    'train_subset_size': {'type': (int, str, type(None)), 'min': 1, 'default': None},  # count | 'full'
    'test_subset_size': {'type': (int, str, type(None)), 'min': 1, 'default': None},   # count | 'full'
    'split_gap_fraction': {'type': float, 'min': 0.0, 'default': 0.5},
    'track_spectral_history': {'type': bool, 'default': False},
    'return_embeddings': {'type': bool, 'default': False},
    # Normalization applied to the embeddings before the cross-covariance SVD
    # behind both the participation ratios and the rotated embeddings.
    'whitening': {'type': (str, type(None)), 'default': 'std'},
    'use_spectral_norm': {'type': bool, 'default': True},
    # Every encoder's input centred and scaled channel by channel by the training
    # rows' statistics, saved with the network: None (off) or 'standard'.
    'input_scaling': {'type': (str, type(None)), 'default': None},
    'use_decoder': {'type': bool, 'default': False},
    'decoder_lambda': {'type': float, 'min': 0.0, 'default': 0.001},
    'decoder_lambda_x': {'type': (float, type(None)), 'min': 0.0, 'default': None},
    'decoder_lambda_y': {'type': (float, type(None)), 'min': 0.0, 'default': None},
    'decoder_output_activation_x': {'type': str, 'default': 'linear'},
    'decoder_output_activation_y': {'type': str, 'default': 'linear'},
    'gradient_clip_val': {'type': (float, type(None)), 'default': None},
    'save_best_model_path': {'type': (str, type(None)), 'default': None},
    'estimator_name': {'type': str, 'default': 'infonce'},
    'estimator_params': {'type': dict, 'default': {}},
    # Online augmentations applied per-batch during training only.
    # augmentation_params applies to both X and Y unless overridden by _x/_y.
    # Set augmentation_params_x or augmentation_params_y to {} to disable for one variable.
    'augmentation_params':   {'type': dict, 'default': {}},
    'augmentation_params_x': {'type': (dict, type(None)), 'default': None},
    'augmentation_params_y': {'type': (dict, type(None)), 'default': None},
    'use_variational': {'type': bool, 'default': False},
    'beta': {'type': float, 'min': 0.0, 'default': 1024.0},
    'train_fraction': {'type': float, 'min': 0.0, 'default': 0.9},
    'n_test_blocks': {'type': int, 'min': 1, 'default': 5},
    'optimizer': {'type': (str, type), 'default': 'adam'},
    'optimizer_params': {'type': dict, 'default': {}},
    'lr_head_multiplier': {'type': (float, int, type(None)), 'min': 0.0, 'default': None},  # Hybrid only; None → same LR as encoders
    'scheduler': {'type': (str, type, type(None)), 'default': None},
    'scheduler_params': {'type': dict, 'default': {}},
    # Per-epoch train MI tracking, yielding 'train_mi_history'.
    #   False, no per-epoch train evaluation.
    #   True, the locked-in train eval subset (capped by max_eval_samples).
    #   int >= 1, exactly that many training samples.
    #   float (0, 1), that fraction of the training set.
    #   1.0 / 'full', the entire training set, uncapped.
    # An unrecognised value raises instead of silently disabling tracking.
    'eval_train': {'type': (bool, float, int, str, type(None)), 'default': False},

    # Per-epoch embedding tracking.
    # Controls whether embeddings are extracted and stored at every epoch.
    # Mirroring eval_train style:
    #   False, no tracking (global default; dimensionality mode defaults to 512).
    #   True, track first 512 samples.
    #   int >= 1, track exactly that many samples (first N in the dataset).
    #   float (0, 1), track that fraction of the dataset.
    #   'full', track all samples (emits a UserWarning about memory cost).
    # The tracked subset is always the *first* N samples so that user-supplied
    # labels (passed to result.animate()) align with the original data ordering.
    'track_embeddings': {'type': (bool, float, int, str, type(None)), 'default': False},
    'return_rotated_embeddings': {'type': bool, 'default': False},
    # False (default): one global rotation derived from the best epoch, applied to all
    # tracked epochs uniformly (consistent coordinate system across epochs).
    # True: each tracked epoch gets its own SVD-based rotation (shows structure emerging).
    'rotated_embeddings_per_epoch': {'type': bool, 'default': False},
    # Whether to include U and V (the rotation matrices) in the result.
    'return_rotation_matrices': {'type': bool, 'default': False},

    # Model architecture parameters
    'shared_encoder': {'type': bool, 'default': False},
    'embedding_dim': {'type': int, 'min': 1, 'default': 64},
    # hidden_dim may be an int (uniform width) or a list of ints (per-layer widths).
    # When a list is given, n_layers is inferred from the list length and any
    # explicit n_layers value is ignored for the layers that receive a list.
    'hidden_dim': {'type': (int, list), 'default': 64},
    'n_layers': {'type': int, 'min': 0, 'default': 2},
    'n_layers_head': {'type': (int, type(None)), 'min': 1, 'default': None},  # Hybrid critic head; None → max(1, n_layers-1)
    'hidden_dim_head': {'type': (int, list, type(None)), 'default': None},  # Hybrid critic head; None → min(64, hidden_dim)
    'critic_type': {'type': str, 'default': 'separable'},
    'embedding_model': {'type': str, 'default': 'mlp'},  # 'mlp'|'cnn'|'cnn2d'|'gru'|'lstm'|'tcn'|'transformer'|'pretrained_backbone'|'lru'|'deepsets'
    'kernel_size': {'type': int, 'min': 1, 'default': 3}, # CNN/TCN
    'cnn_pooling': {'type': str, 'default': 'flatten'},   # CNN: 'flatten' keeps every position, 'mean' averages
    'bidirectional': {'type': bool, 'default': False}, # RNN
    'nhead': {'type': int, 'min': 1, 'default': 4}, # Transformer
    # Per-side encoder overrides. None means "same as X" throughout.
    'embedding_model_y': {'type': str, 'default': None},
    'embedding_dim_y': {'type': int, 'default': None},
    'hidden_dim_y': {'type': (int, list), 'default': None},
    'n_layers_y': {'type': int, 'default': None},
    'custom_embedding_cls_y': {'type': type, 'default': None},
    'max_n_batches': {'type': int, 'min': 1, 'default': 512}, # Critic chunking
    'dropout': {'type': float, 'min': 0.0, 'default': 0.0},
    # 'auto' is layer normalisation for the hybrid critic in mode='dimensionality'
    # and none otherwise. Layer norm stops the fits of that mode from stalling. It
    # also divides out each sample's overall scale and the information it carries.
    'norm_layer': {'type': (str, type(None)), 'default': 'auto'},
    # Whether embedding layers carry bias terms. Without them the network is
    # positively homogeneous, so an all-zero input embeds to exactly zero.
    # A mean-centring `norm_layer` is served by an affine-free RMSNorm when
    # this is off, since centring would otherwise undo the property.
    'bias': {'type': (bool, type(None)), 'default': True},
    # PretrainedBackboneEmbedding: torchvision model name and pretrained flag.
    'pytorch_predefined': {'type': (str, type(None)), 'default': None},
    'pretrained': {'type': bool, 'default': False},

    # Internal/Inferred parameters (usually not set by user but passed down)
    'input_dim_x': {'type': int},
    'input_dim_y': {'type': int},
    'n_channels_x': {'type': int},
    'n_channels_y': {'type': int},
    'processor_type_x': {'type': (str, type(None))},
    'processor_type_y': {'type': (str, type(None))},
    'processor_params_x': {'type': (dict, type(None))},
    'processor_params_y': {'type': (dict, type(None))},
    'custom_critic': {'type': (object, type(None))}, # class or function that builds the critic
    'custom_embedding_cls': {'type': (type, type(None))},
    'train_indices': {'type': (object, type(None))}, # numpy array
    'test_indices': {'type': (object, type(None))},
    'gamma': {'type': (int, float)}, # Rigorous
    'min_reliable_samples': {'type': int, 'min': 1, 'default': None},
    'lag': {'type': int},  # Result label: injected by run_lag_analysis per task; not a user-settable parameter.
    # Reproducibility, used by run() and task.py workers
    'random_seed': {'type': (int, type(None)), 'default': None},

    # Conservative epoch selection:
    # 1.0 (default) → use the epoch where smoothed test MI is maximised.
    # < 1.0          → use the first epoch where smoothed test MI ≥ peak_fraction * max_test_mi.
    #                  This gives a more conservative train-MI estimate by avoiding the final
    #                  noisy peak.  When < 1.0, the train MI at the actual peak epoch is also
    #                  returned in details as 'train_mi_at_peak'.
    'peak_fraction': {'type': float, 'min': 0.0, 'default': 1.0},
    # Which epoch of the held-out curve the reported values are read at: the earliest at its top
    # ('plateau_start') or its maximum ('peak'). See neural_mi/training/epochs.py.
    'epoch_rule': {'type': str, 'default': 'plateau_start'},

    # Mixed-precision (AMP) training.
    # 'auto', enable on CUDA, no-op on CPU/MPS (safe default).
    # True, explicitly enable (CUDA only; silently no-ops on other devices).
    # False, explicitly disable.
    'use_amp': {'type': (bool, str), 'default': 'auto'},

    # Memory / device layout
    # 'cpu', store dataset tensors on CPU (default; safe for long sweeps).
    # 'auto', store on the compute device (faster repeated evaluation, e.g. precision mode).
    # Any explicit device string is also accepted.
    # Precision mode overrides this to 'auto' unless the user sets it explicitly.
    'dataset_device': {'type': (str, type(None)), 'default': 'cpu'},
}

# Parameters allowed in analysis_kwargs for each mode
MODE_KWARGS_SCHEMA = {
    'estimate': {
        'n_workers': {'type': int, 'default': 1},
    },
    'sweep': {
        'n_workers': {'type': int, 'default': 1},
    },
    'dimensionality': {
        'n_workers': {'type': int, 'default': 1},
        # The embedding dimensions to fit. None chooses them from the reference fit.
        'embedding_dims': {'type': (list, tuple, range, type(None)), 'default': None},
        'n_restarts': {'type': int, 'min': 1, 'default': 4},
        'saturation_ratio': {'type': float, 'min': 0.0, 'default': 0.95},
        # None is 64, refitted larger when its participation ratio comes close to that.
        'reference_dim': {'type': (int, type(None)), 'min': 2, 'default': None},
        'split_method': {'type': str, 'default': 'random'}, # 'random'|'spatial'|'temporal'|'index'|'horizontal'|'vertical'|'row_interleaved'|'col_interleaved'|'diagonal'|'antidiagonal'
        # Random channel splits of X; the other split methods give one split.
        'n_splits': {'type': (int, type(None)), 'min': 1, 'default': None},
        'lag': {'type': int, 'default': 1}, # if split_method='temporal'
        # Required when split_method='index': list of channel indices assigned to X.
        # Y is automatically the complement (all remaining channels).
        'channel_indices_x': {'type': (list, type(None)), 'default': None},
        # Warn when the reference fit's MI is this close to its evaluation ceiling.
        'ceiling_mi_fraction': {'type': float, 'default': 0.85},
    },
    'rigorous': {
        'n_workers': {'type': int, 'default': 1},
        'curvature_t_threshold': {'type': float, 'default': 2.0},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'leverage_threshold': {'type': float, 'default': 0.20},
        # None: contiguous chunks for rows recorded in time order
        # (neural_mi/data/rows.py), permuted otherwise; True/False overrides.
        'temporal_chunking': {'type': (bool, type(None)), 'default': None},
    },
    'lag': {
        'n_workers': {'type': int, 'default': 1},
        'lag_range': {'type': (range, list, np.ndarray), 'required': True},
        'equalize_n': {'type': bool, 'default': False},
    },
    'precision': {
        'n_workers': {'type': int, 'default': 1},
        'tau_grid': {'type': list, 'required': True},
        # corrupt_target: 'x' corrupts only X, 'y' only Y, 'both' corrupts both
        # simultaneously (useful for measuring shared spike-timing precision).
        'corrupt_target': {'type': str, 'default': 'x'},
        'corruption_method': {'type': str, 'default': 'rounding'},
        'n_noise_samples': {'type': int, 'default': 50},
        'threshold_ratio': {'type': (float, list, tuple), 'default': 0.9},
    },
    'conditional': {
        'n_workers': {'type': int, 'default': 1},
        'rigorous': {'type': bool, 'default': False},
        'gamma_range': {'type': (range, list, type(None)), 'default': None},
        'curvature_t_threshold': {'type': float, 'default': 2.0},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'leverage_threshold': {'type': float, 'default': 0.20},
    },
    'transfer': {
        'n_workers': {'type': int, 'default': 1},
        'rigorous': {'type': bool, 'default': False},
        'gamma_range': {'type': (range, list, type(None)), 'default': None},
        'curvature_t_threshold': {'type': float, 'default': 2.0},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'leverage_threshold': {'type': float, 'default': 0.20},
    },
    'interaction': {
        'n_workers': {'type': int, 'default': 1},
        'rigorous': {'type': bool, 'default': False},
        'gamma_range': {'type': (range, list, type(None)), 'default': None},
        'curvature_t_threshold': {'type': float, 'default': 2.0},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'leverage_threshold': {'type': float, 'default': 0.20},
    },
    'pairwise': {
        'n_workers': {'type': int, 'default': 1},
        'pairs': {'type': (list, type(None)), 'default': None},
    },
}


# The processor settings' defaults, by processor. ``None`` means unset: a
# window_size has to be given, a step_size of None is one window, and so on.
PROCESSOR_DEFAULTS = {
    'continuous': {'window_size': None, 'step_size': None, 'min_coverage_fraction': 0.2,
                   'sample_rate': None},
    'spike': {'window_size': None, 'step_size': None, 'max_spikes_per_window': None, 'n_seconds': None,
              'sample_rate': None, 'no_spike_value': 0.0, 'bin_size': None, 'normalize_bins': False,
              'exclude_bursty_neurons': False, 'burst_threshold_multiplier': 5.0,
              'drop_empty_windows': True},
    'categorical': {'window_size': None, 'step_size': None, 'sample_rate': None,
                    'min_coverage_fraction': 0.2, 'encoding': 'majority_vote'},
}


# The settings each processor takes.
PROCESSOR_PARAMS_SCHEMA = {processor: list(defaults) for processor, defaults in PROCESSOR_DEFAULTS.items()}


def stream_processing(processor: Optional[str], params: Optional[Dict[str, Any]], x_processor: Optional[str],
                      x_params: Optional[Dict[str, Any]], data: Any = None
                      ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The processor and processor settings a second stream, Y or W, is read with: one rule for both.

    A stream without a processor of its own reads with X's. One without
    settings of its own reads with X's settings, kept to the keys its processor
    takes. Settings it was given are its own and are not merged with X's. A
    stream given as rows built already (``data`` with three dimensions) has no
    raw series for a processor to read and inherits nothing.
    """
    if processor is None and getattr(data, 'ndim', None) == 3:
        return None, params
    processor = processor if processor is not None else x_processor
    if params is None and processor is not None:
        accepted = PROCESSOR_DEFAULTS.get(processor)
        params = {k: v for k, v in (x_params or {}).items() if accepted is None or k in accepted}
    return processor, params


def processor_setting(processor: str, params: Dict[str, Any], name: str) -> Any:
    """Processor setting ``name`` from ``params``, or the schema's default for ``processor`` when
    ``params`` lacks it."""
    params = params or {}
    return params[name] if name in params else copy.copy(PROCESSOR_DEFAULTS[processor][name])


def default_of(name: str) -> Any:
    """The schema's default for setting ``name``: the one place a default is written. A mutable default comes
    back as a copy, so a caller cannot change it for every other caller."""
    return copy.copy(BASE_PARAMS_SCHEMA[name]['default'])


def setting(params: Dict[str, Any], name: str) -> Any:
    """Setting ``name`` from ``params``, or its schema default when ``params`` lacks it."""
    return params[name] if name in params else default_of(name)


def norm_layer_of(params: Dict[str, Any]) -> Optional[str]:
    """The normalisation layer the networks build: ``norm_layer``, with 'auto' and 'none' read as none.

    A mode that wants another reading of 'auto' sets norm_layer before anything is built, as
    mode='dimensionality' does for its hybrid critic."""
    norm = setting(params, 'norm_layer')
    return None if norm in ('auto', 'none') else norm
