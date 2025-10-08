# tests/test_datasets.py
import pytest
import numpy as np
import torch
from neural_mi.datasets import (
    generate_correlated_gaussians,
    generate_nonlinear_from_latent,
    generate_temporally_convolved_data,
    generate_xor_data,
    generate_correlated_spike_trains,
    generate_correlated_categorical_series
)

class TestDatasetGenerators:
    def test_correlated_gaussians_shape_and_type(self):
        x, y = generate_correlated_gaussians(n_samples=100, dim=5, mi=2.0, use_torch=True)
        assert x.shape == (100, 5) and y.shape == (100, 5)
        assert isinstance(x, torch.Tensor)

    def test_nonlinear_from_latent_shape(self):
        x, y = generate_nonlinear_from_latent(100, 4, 50, 2.0)
        assert x.shape == (100, 50) and y.shape == (100, 50)

    def test_temporally_convolved_shape(self):
        x, y = generate_temporally_convolved_data(1000)
        assert x.shape == (1, 1000) and y.shape == (1, 1000)

    def test_xor_data_logic(self):
        x, y = generate_xor_data(100, noise=0.0)
        expected = torch.bitwise_xor(x[:, 0].long(), x[:, 1].long()).float().view(-1, 1)
        assert torch.allclose(y, expected)

    def test_spike_trains_format(self):
        pop_x, pop_y = generate_correlated_spike_trains(n_neurons=5, duration=10.0)
        assert len(pop_x) == 5 and len(pop_y) == 5
        for spikes in pop_x:
            assert isinstance(spikes, np.ndarray)
            assert np.all(spikes >= 0) and np.all(spikes <= 10.0)

    def test_generate_correlated_categorical_series_shape_and_type(self):
        x, y = generate_correlated_categorical_series(n_samples=100, n_channels=2, use_torch=False)
        assert x.shape == (2, 100) and y.shape == (2, 100)
        assert isinstance(x, np.ndarray) and x.dtype == int
        assert (x.max()) < 3