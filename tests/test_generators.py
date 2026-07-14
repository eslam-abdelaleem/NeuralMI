# tests/test_generators.py
import pytest
import numpy as np
import torch
from neural_mi.generators import (
    generate_correlated_gaussians,
    generate_nonlinear_from_latent,
    generate_temporally_convolved_data,
    generate_xor_data,
    generate_correlated_spike_trains,
    generate_correlated_categorical_series
)
from neural_mi.generators import synthetic

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
        assert x.shape == (1000, 1) and y.shape == (1000, 1)

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
        assert x.shape == (100, 2) and y.shape == (100, 2)
        assert isinstance(x, np.ndarray) and x.dtype == int
        assert (x.max()) < 3


class TestGeneratorsNumpyAndExtended:
    """Tests for numpy output variants and additional synthetic generators."""

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


class TestWindowedGenerators:
    """Windowed generators with analytically known MI, and the window-size
    sweep dependency generator. Previously had zero test coverage."""

    def test_windowed_oscillatory_shape_and_dtype(self):
        X, Y, true_mi = synthetic.generate_windowed_oscillatory(
            n_windows=20, n_channels=3, window_size=64, latent_mi=1.0,
        )
        assert X.shape == (20, 3, 64) and Y.shape == (20, 3, 64)
        assert X.dtype == np.float32 and Y.dtype == np.float32
        assert isinstance(true_mi, float) and true_mi > 0

    def test_windowed_oscillatory_true_mi_scales_linearly_with_n_channels(self):
        """true_mi is a deterministic function of the parameters (it does not
        depend on the random draws), so doubling n_channels must exactly
        double it."""
        _, _, mi_1ch = synthetic.generate_windowed_oscillatory(
            n_windows=5, n_channels=1, latent_mi=1.0, snr=2.0,
        )
        _, _, mi_2ch = synthetic.generate_windowed_oscillatory(
            n_windows=5, n_channels=2, latent_mi=1.0, snr=2.0,
        )
        assert mi_2ch == pytest.approx(2 * mi_1ch)

    def test_windowed_multichannel_shape_and_dtype(self):
        X, Y, true_mi = synthetic.generate_windowed_multichannel(
            n_windows=15, n_channels=4, window_size=50, latent_mi=0.5,
        )
        assert X.shape == (15, 4, 50) and Y.shape == (15, 4, 50)
        assert X.dtype == np.float32 and Y.dtype == np.float32
        assert isinstance(true_mi, float) and true_mi > 0

    def test_windowed_multichannel_true_mi_matches_oscillatory_for_one_channel(self):
        """Each channel's contribution uses the same per-channel formula as
        generate_windowed_oscillatory, so with matching window_size/sample_rate/
        snr and a single channel, the two must agree exactly."""
        common = dict(window_size=200, sample_rate=500.0, latent_mi=0.5, snr=3.0)
        _, _, multichannel_mi = synthetic.generate_windowed_multichannel(
            n_windows=10, n_channels=1, f_min_hz=4.0, f_max_hz=4.0, **common,
        )
        _, _, oscillatory_mi = synthetic.generate_windowed_oscillatory(
            n_windows=10, n_channels=1, f_carrier_hz=4.0, **common,
        )
        assert multichannel_mi == pytest.approx(oscillatory_mi, rel=1e-5)

    def test_windowed_dependency_data_shape(self):
        x, y = synthetic.generate_windowed_dependency_data(n_timepoints=500, n_channels=3)
        assert x.shape == (500, 3) and y.shape == (500, 3)

    def test_windowed_dependency_data_rejects_invalid_history_window(self):
        with pytest.raises(ValueError, match="history_window"):
            synthetic.generate_windowed_dependency_data(n_timepoints=100, history_window=0)

    def test_windowed_dependency_data_rejects_invalid_noise_level(self):
        with pytest.raises(ValueError, match="noise_level"):
            synthetic.generate_windowed_dependency_data(n_timepoints=100, noise_level=1.5)
