# neural_mi/defaults.py
"""
Centralized definition of default parameters and allowed keys for validation.
"""
import numpy as np
import torch

# Allowed parameters for base_params
# Each entry is (type_check, min_value or validation_function)
BASE_PARAMS_SCHEMA = {
    # Trainer parameters
    'n_epochs': {'type': int, 'min': 1, 'default': 50},
    'learning_rate': {'type': float, 'min': 0.0, 'default': 5e-4},
    'batch_size': {'type': int, 'min': 1, 'default': 128},
    # Default of 1000 effectively disables early stopping for the default n_epochs=50.
    # To enable early stopping, set patience to a value smaller than n_epochs
    # (e.g. patience=20 with n_epochs=200).
    'patience': {'type': int, 'min': 0, 'default': 1000},
    'output_units': {'type': str, 'default': 'bits'},
    'verbose': {'type': bool, 'default': True},
    'show_progress': {'type': bool, 'default': True},
    'device': {'type': (str, type(None), torch.device), 'default': None},
    'split_mode': {'type': str, 'default': 'blocked'},
    'random_time_shifting': {'type': bool, 'default': False},
    'epochs_to_max_shift': {'type': int, 'min': 0, 'default': 5},
    'smoothing_sigma': {'type': float, 'min': 0.0, 'default': 1.0},
    'median_window': {'type': int, 'min': 1, 'default': 5},
    'min_improvement': {'type': float, 'min': 0.0, 'default': 0.001},
    'max_eval_samples': {'type': int, 'min': 1, 'default': 5000},
    'train_subset_size': {'type': (int, type(None)), 'min': 1, 'default': None},
    'split_gap_fraction': {'type': float, 'min': 0.0, 'default': 0.5},
    'spectral_mode': {'type': str, 'default': 'none'},  # 'none' | 'summary' | 'full'
    'track_spectral_metrics': {'type': bool, 'default': False},
    'spectral_output': {'type': str, 'default': 'default'},
    'return_spectrum': {'type': bool, 'default': False},
    'return_embeddings': {'type': bool, 'default': False},
    'spectral_whitening': {'type': (str, type(None)), 'default': 'std'},
    'use_spectral_norm': {'type': bool, 'default': True},
    'use_decoder': {'type': bool, 'default': False},
    'decoder_weight': {'type': float, 'min': 0.0, 'default': 1.0},
    'decoder_weight_x': {'type': (float, type(None)), 'default': None},
    'decoder_weight_y': {'type': (float, type(None)), 'default': None},
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
    'max_index_reduction': {'type': float, 'min': 0.0, 'default': 0.05},
    'optimizer': {'type': (str, type), 'default': 'adam'},
    'optimizer_params': {'type': dict, 'default': {}},
    'lr_head_multiplier': {'type': (float, int, type(None)), 'min': 0.0, 'default': None},  # Hybrid only; None → same LR as encoders
    'scheduler': {'type': (str, type, type(None)), 'default': None},
    'scheduler_params': {'type': dict, 'default': {}},
    'eval_train': {'type': (bool, float, int, type(None)), 'default': False},

    # Per-epoch embedding tracking.
    # Controls whether embeddings are extracted and stored at every epoch.
    # Mirroring eval_train style:
    #   False         — no tracking (global default; dimensionality mode defaults to 512).
    #   True          — track first 512 samples.
    #   int >= 1      — track exactly that many samples (first N in the dataset).
    #   float (0, 1)  — track that fraction of the dataset.
    #   'full'        — track all samples (emits a UserWarning about memory cost).
    # The tracked subset is always the *first* N samples so that user-supplied
    # labels (passed to result.animate()) align with the original data ordering.
    'track_embeddings': {'type': (bool, float, int, str, type(None)), 'default': False},
    'return_rotated_embeddings': {'type': bool, 'default': False},
    # Whitening applied to the cross-covariance before SVD to derive the rotation axes.
    # Does NOT affect the scale of the returned embeddings (which are always in the
    # original embedding space, just re-projected).  Matches the default used by PR.
    'rotated_embeddings_whitening': {'type': (str, type(None)), 'default': 'std'},
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
    'embedding_model': {'type': str, 'default': 'mlp'},  # 'mlp'|'cnn'|'cnn2d'|'gru'|'lstm'|'tcn'|'transformer'|'pretrained_backbone'
    'kernel_size': {'type': int, 'min': 1, 'default': 3}, # CNN/TCN
    'bidirectional': {'type': bool, 'default': False}, # RNN
    'nhead': {'type': int, 'min': 1, 'default': 4}, # Transformer
    'max_n_batches': {'type': int, 'min': 1, 'default': 512}, # Critic chunking
    'dropout': {'type': float, 'min': 0.0, 'default': 0.0},
    'norm_layer': {'type': (str, type(None)), 'default': None},
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
    'custom_critic': {'type': (object, type(None))}, # torch.nn.Module
    'custom_embedding_cls': {'type': (type, type(None))},
    'train_indices': {'type': (object, type(None))}, # numpy array
    'test_indices': {'type': (object, type(None))},
    'gamma': {'type': (int, float)}, # Rigorous
    'min_reliable_samples': {'type': int, 'min': 1, 'default': 1000},
    'lag': {'type': int},  # Result label: injected by run_lag_analysis per task; not a user-settable parameter.
    # Reproducibility — used by run() and task.py workers
    'random_seed': {'type': (int, type(None)), 'default': None},

    # Conservative epoch selection:
    # 1.0 (default) → use the epoch where smoothed test MI is maximised (current behaviour).
    # < 1.0          → use the first epoch where smoothed test MI ≥ peak_fraction * max_test_mi.
    #                  This gives a more conservative train-MI estimate by avoiding the final
    #                  noisy peak.  When < 1.0, the train MI at the actual peak epoch is also
    #                  returned in details as 'train_mi_at_peak'.
    'peak_fraction': {'type': float, 'min': 0.0, 'default': 1.0},

    # Mixed-precision (AMP) training.
    # 'auto' — enable on CUDA, no-op on CPU/MPS (safe default).
    # True   — explicitly enable (CUDA only; silently no-ops on other devices).
    # False  — explicitly disable.
    'use_amp': {'type': (bool, str), 'default': 'auto'},

    # Memory / device layout
    # 'cpu'  — store dataset tensors on CPU (default; safe for long sweeps).
    # 'auto' — store on the compute device (faster repeated evaluation, e.g. precision mode).
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
        'max_samples_per_task': {'type': int, 'default': None},
    },
    'dimensionality': {
        'n_workers': {'type': int, 'default': 1},
        'split_method': {'type': str, 'default': 'random'}, # 'random'|'spatial'|'temporal'|'index'|'horizontal'|'vertical'|'row_interleaved'|'col_interleaved'|'diagonal'|'antidiagonal'
        'n_splits': {'type': int, 'default': 5},
        'lag': {'type': int, 'default': 1}, # if split_method='temporal'
        # Required when split_method='index': list of channel indices assigned to X.
        # Y is automatically the complement (all remaining channels).
        'channel_indices_x': {'type': (list, type(None)), 'default': None},
    },
    'rigorous': {
        'n_workers': {'type': int, 'default': 1},
        'delta_threshold': {'type': float, 'default': 0.1},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'r2_threshold': {'type': float, 'default': 0.90},
        'leverage_threshold': {'type': float, 'default': 0.20},
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
        'threshold_ratio': {'type': float, 'default': 0.9},
    },
    'conditional': {
        'n_workers': {'type': int, 'default': 1},
        'rigorous': {'type': bool, 'default': False},
        'gamma_range': {'type': (range, list, type(None)), 'default': None},
        'delta_threshold': {'type': float, 'default': 0.1},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'r2_threshold': {'type': float, 'default': 0.90},
        'leverage_threshold': {'type': float, 'default': 0.20},
    },
    'transfer': {
        'n_workers': {'type': int, 'default': 1},
        'rigorous': {'type': bool, 'default': False},
        'gamma_range': {'type': (range, list, type(None)), 'default': None},
        'delta_threshold': {'type': float, 'default': 0.1},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
        'residual_threshold': {'type': float, 'default': 2.5},
        'r2_threshold': {'type': float, 'default': 0.90},
        'leverage_threshold': {'type': float, 'default': 0.20},
    },
    'pairwise': {
        'n_workers': {'type': int, 'default': 1},
    },
}

PROCESSOR_PARAMS_SCHEMA = {
    'continuous': ['window_size', 'step_size', 'min_coverage_fraction', 'sample_rate'],
    'spike': ['window_size', 'step_size', 'max_spikes_per_window', 'n_seconds', 'sample_rate',
              'no_spike_value', 'bin_size', 'normalize_bins', 'exclude_bursty_neurons', 'burst_threshold_multiplier'],
    'categorical': ['window_size', 'step_size', 'sample_rate', 'min_coverage_fraction', 'encoding'],
}
