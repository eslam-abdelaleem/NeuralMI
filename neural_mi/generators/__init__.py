# neural_mi/generators/__init__.py
"""This package contains functions for generating synthetic datasets.

These functions are useful for testing, validating, and demonstrating the
capabilities of the mutual information estimators in the library.
"""
from .synthetic import (
    mi_to_rho,
    generate_correlated_gaussians,
    generate_nonlinear_from_latent,
    generate_temporally_convolved_data,
    generate_xor_data,
    generate_correlated_spike_trains,
    generate_correlated_categorical_series,
    generate_event_related_data,
    generate_linear_data,
    generate_nonlinear_data,
    generate_history_data,
    generate_full_data,
    generate_windowed_dependency_data,
    # Inductive-bias generators
    generate_modulated_spike_trains,
    generate_timing_code_spike_trains,
    generate_noisy_image_pairs,
    # Windowed generators with analytically known MI
    generate_windowed_oscillatory,
    generate_windowed_multichannel,
)

__all__ = [
    'mi_to_rho',
    'generate_correlated_gaussians',
    'generate_nonlinear_from_latent',
    'generate_temporally_convolved_data',
    'generate_xor_data',
    'generate_correlated_spike_trains',
    'generate_correlated_categorical_series',
    'generate_event_related_data',
    'generate_linear_data',
    'generate_nonlinear_data',
    'generate_history_data',
    'generate_full_data',
    'generate_windowed_dependency_data',
    # Inductive-bias generators
    'generate_modulated_spike_trains',
    'generate_timing_code_spike_trains',
    'generate_noisy_image_pairs',
    # Windowed generators with analytically known MI
    'generate_windowed_oscillatory',
    'generate_windowed_multichannel',
]
