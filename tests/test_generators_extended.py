# tests/test_generators_extended.py
import pytest
import numpy as np
import torch
from neural_mi.generators import synthetic

class TestGeneratorsExtended:
    def test_nonlinear_from_latent_numpy(self):
        x, y = synthetic.generate_nonlinear_from_latent(100, 4, 50, 2.0, use_torch=False)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.shape == (100, 50) and y.shape == (100, 50)

    def test_correlated_gaussians_numpy(self):
        x, y = synthetic.generate_correlated_gaussians(n_samples=100, dim=5, mi=2.0, use_torch=False)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.shape == (100, 5) and y.shape == (100, 5)

    def test_temporally_convolved_data_numpy(self):
        x, y = synthetic.generate_temporally_convolved_data(100, use_torch=False)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.shape == (100, 1) and y.shape == (100, 1)

    def test_xor_data_numpy(self):
        x, y = synthetic.generate_xor_data(100, use_torch=False)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.shape == (100, 2) and y.shape == (100, 1)

    def test_generate_event_related_data(self):
        x, y = synthetic.generate_event_related_data(n_samples=500, lag=10, n_events=5, response_length=10, use_torch=True)
        assert x.shape == (500, 1) and y.shape == (500, 1)
        assert isinstance(x, torch.Tensor)

        x_np, y_np = synthetic.generate_event_related_data(n_samples=500, use_torch=False)
        assert isinstance(x_np, np.ndarray)

    def test_generate_linear_data(self):
        x, y = synthetic.generate_linear_data(n_samples=500, true_lag=10)
        assert x.shape == (500, 1) and y.shape == (500, 1)
        assert isinstance(x, np.ndarray)

    def test_generate_nonlinear_data(self):
        x, y = synthetic.generate_nonlinear_data(n_samples=500, true_lag=10)
        assert x.shape == (500, 1) and y.shape == (500, 1)
        assert isinstance(x, np.ndarray)

    def test_generate_history_data(self):
        x, y = synthetic.generate_history_data(n_samples=500, history_duration=10)
        assert x.shape == (500, 1) and y.shape == (500, 1)
        assert isinstance(x, np.ndarray)

    def test_generate_full_data(self):
        x, y = synthetic.generate_full_data(n_samples=500, true_lag=10, history_duration=10)
        assert x.shape == (500, 1) and y.shape == (500, 1)
        assert isinstance(x, np.ndarray)
