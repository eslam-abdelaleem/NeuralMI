# neural_mi/defaults.py
"""
Centralized definition of default parameters and allowed keys for validation.
"""
import torch

# Allowed parameters for base_params
# Each entry is (type_check, min_value or validation_function)
BASE_PARAMS_SCHEMA = {
    # Trainer parameters
    'n_epochs': {'type': int, 'min': 1, 'default': 50},
    'learning_rate': {'type': float, 'min': 0.0, 'default': 1e-3},
    'batch_size': {'type': int, 'min': 1, 'default': 64},
    'patience': {'type': int, 'min': 0, 'default': 10},
    'output_units': {'type': str, 'default': 'bits'},
    'verbose': {'type': bool, 'default': True},
    'device': {'type': (str, type(None), torch.device), 'default': None},
    'split_mode': {'type': str, 'default': 'blocked'},
    'random_time_shifting': {'type': bool, 'default': True},
    'epochs_to_max_shift': {'type': int, 'min': 0, 'default': 5},
    'smoothing_sigma': {'type': float, 'min': 0.0, 'default': 1.0},
    'median_window': {'type': int, 'min': 1, 'default': 5},
    'min_improvement': {'type': float, 'min': 0.0, 'default': 0.001},
    'max_eval_samples': {'type': int, 'min': 1, 'default': 5000},
    'train_subset_size': {'type': (int, type(None)), 'min': 1, 'default': None},
    'track_spectral_metrics': {'type': bool, 'default': False},
    'spectral_output': {'type': str, 'default': 'default'},
    'return_spectrum': {'type': bool, 'default': False},
    'save_best_model_path': {'type': (str, type(None)), 'default': None},
    'estimator_name': {'type': str, 'default': 'infonce'},
    'estimator_params': {'type': dict, 'default': {}},
    'use_variational': {'type': bool, 'default': False},
    'beta': {'type': float, 'default': 1.0},

    # Model architecture parameters
    'embedding_dim': {'type': int, 'min': 1, 'default': 64},
    'hidden_dim': {'type': int, 'min': 1, 'default': 64},
    'n_layers': {'type': int, 'min': 0, 'default': 2},
    'critic_type': {'type': str, 'default': 'separable'},
    'embedding_model': {'type': str, 'default': 'mlp'},
    'kernel_size': {'type': int, 'min': 1, 'default': 3}, # CNN/TCN
    'bidirectional': {'type': bool, 'default': False}, # RNN
    'nhead': {'type': int, 'min': 1, 'default': 4}, # Transformer
    'max_n_batches': {'type': int, 'min': 1, 'default': 512}, # Critic chunking

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
    'lag': {'type': int}, # Lag
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
        'split_method': {'type': str, 'default': 'random'},
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
        'lag_range': {'type': (range, list), 'required': True},
    },
    'precision': {
        'n_workers': {'type': int, 'default': 1},
        'tau_grid': {'type': list, 'required': True},
        'corrupt_target': {'type': str, 'default': 'x'},
        'corruption_method': {'type': str, 'default': 'rounding'},
        'n_noise_samples': {'type': int, 'default': 50},
        'threshold_ratio': {'type': float, 'default': 0.9},
    }
}

PROCESSOR_PARAMS_SCHEMA = {
    'continuous': ['window_size'],
    'spike': ['window_size', 'max_spikes_per_window', 'n_seconds'], # Check valid ones
    'categorical': ['window_size'],
}
