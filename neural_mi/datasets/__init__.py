# neural_mi/datasets/__init__.py
"""This package contains functions for generating synthetic datasets.

These functions are useful for testing, validating, and demonstrating the
capabilities of the mutual information estimators in the library.
"""
from .generators import (
    generate_correlated_gaussians, 
    generate_nonlinear_from_latent,
    generate_temporally_convolved_data,
    generate_xor_data,
    generate_correlated_spike_trains,
    generate_correlated_categorical_series
)