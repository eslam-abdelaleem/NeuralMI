# neural_mi/datasets/__init__.py
from .generators import (
    generate_correlated_gaussians, 
    generate_nonlinear_from_latent,
    generate_temporally_convolved_data,
    generate_xor_data,
    generate_correlated_spike_trains
)