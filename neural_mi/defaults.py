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
    'patience': {'type': int, 'min': 0, 'default': 1000}, # Very large number, no early stopping by default
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
    'track_spectral_metrics': {'type': bool, 'default': False},  # deprecated; use spectral_mode
    'spectral_output': {'type': str, 'default': 'default'},       # deprecated; use spectral_mode
    'return_spectrum': {'type': bool, 'default': False},           # deprecated; use spectral_mode
    'return_embeddings': {'type': bool, 'default': False},
    'spectral_whitening': {'type': (str, type(None)), 'default': 'std'},
    'use_spectral_norm': {'type': bool, 'default': True},
    'gradient_clip_val': {'type': (float, type(None)), 'default': None},
    'save_best_model_path': {'type': (str, type(None)), 'default': None},
    'estimator_name': {'type': str, 'default': 'infonce'},
    'estimator_params': {'type': dict, 'default': {}},
    'use_variational': {'type': bool, 'default': False},
    'beta': {'type': float, 'min': 0.0, 'default': 1024.0},
    'train_fraction': {'type': float, 'min': 0.0, 'default': 0.9},
    'n_test_blocks': {'type': int, 'min': 1, 'default': 5},
    'max_index_reduction': {'type': float, 'min': 0.0, 'default': 0.05},
    'optimizer': {'type': (str, type), 'default': 'adam'},
    'optimizer_params': {'type': dict, 'default': {}},
    'scheduler': {'type': (str, type, type(None)), 'default': None},
    'scheduler_params': {'type': dict, 'default': {}},
    'eval_train': {'type': (bool, float, int, type(None)), 'default': False},


    # Model architecture parameters
    'shared_encoder': {'type': bool, 'default': False},
    'embedding_dim': {'type': int, 'min': 1, 'default': 64},
    'hidden_dim': {'type': int, 'min': 1, 'default': 64},
    'n_layers': {'type': int, 'min': 0, 'default': 2},
    'critic_type': {'type': str, 'default': 'separable'},
    'embedding_model': {'type': str, 'default': 'mlp'},
    'kernel_size': {'type': int, 'min': 1, 'default': 3}, # CNN/TCN
    'bidirectional': {'type': bool, 'default': False}, # RNN
    'nhead': {'type': int, 'min': 1, 'default': 4}, # Transformer
    'max_n_batches': {'type': int, 'min': 1, 'default': 512}, # Critic chunking
    'dropout': {'type': float, 'min': 0.0, 'default': 0.0},
    'norm_layer': {'type': (str, type(None)), 'default': None},

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
    'lag': {'type': int}, # Lag
    # Reproducibility — used by run() and task.py workers
    'random_seed': {'type': (int, type(None)), 'default': None},
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
        'split_method': {'type': str, 'default': 'random'}, # 'random', 'spatial', or 'temporal'
        'n_splits': {'type': int, 'default': 5},
        'lag': {'type': int, 'default': 1}, # if split_method='temporal'
    },
    'rigorous': {
        'n_workers': {'type': int, 'default': 1},
        'delta_threshold': {'type': float, 'default': 0.1},
        'min_gamma_points': {'type': int, 'default': 5},
        'confidence_level': {'type': float, 'default': 0.68},
    },
    'lag': {
        'n_workers': {'type': int, 'default': 1},
        'lag_range': {'type': (range, list, np.ndarray), 'required': True},
        'equalize_n': {'type': bool, 'default': False},
    },
    'precision': {
        'n_workers': {'type': int, 'default': 1},
        'tau_grid': {'type': list, 'required': True},
        'corrupt_target': {'type': str, 'default': 'x'},
        'corruption_method': {'type': str, 'default': 'rounding'},
        'n_noise_samples': {'type': int, 'default': 50},
        'threshold_ratio': {'type': float, 'default': 0.9},
    },
    'transfer': {
        'n_workers': {'type': int, 'default': 1},
        'bidirectional_te': {'type': bool, 'default': False},
    },
}

PROCESSOR_PARAMS_SCHEMA = {
    'continuous': ['window_size', 'min_coverage_fraction', 'sample_rate'],
    'spike': ['window_size', 'max_spikes_per_window', 'n_seconds', 'sample_rate',
              'no_spike_value', 'bin_size', 'normalize_bins', 'exclude_bursty_neurons', 'burst_threshold_multiplier'],
    'categorical': ['window_size', 'sample_rate', 'min_coverage_fraction', 'encoding'],
}
