import pytest
import numpy as np
import torch
from neural_mi.data.processors import ContinuousProcessor, SpikeProcessor

class TestContinuousProcessor:
    def test_basic_windowing(self):
        processor = ContinuousProcessor(window_size=3, step_size=1)
        data = np.arange(10).reshape(1, -1)  # [1, 10]
        result = processor.process(data)

        assert result.shape == (8, 1, 3)  # 8 windows, 1 channel, 3 timepoints
        assert torch.equal(result[0, 0, :], torch.tensor([0., 1., 2.]))

    def test_multiple_channels(self):
        processor = ContinuousProcessor(window_size=2, step_size=1)
        data = np.random.randn(3, 10)  # 3 channels, 10 timepoints
        result = processor.process(data)

        assert result.shape == (9, 3, 2)

    def test_step_size_greater_than_one(self):
        processor = ContinuousProcessor(window_size=2, step_size=2)
        data = np.arange(10).reshape(1, -1)
        result = processor.process(data)

        assert result.shape == (5, 1, 2)  # 5 non-overlapping windows

    def test_data_too_short(self):
        processor = ContinuousProcessor(window_size=10, step_size=1)
        data = np.arange(5).reshape(1, -1)

        with pytest.raises(ValueError, match="Length of data .* is smaller than"):
            processor.process(data)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            ContinuousProcessor(window_size=0)

        with pytest.raises(ValueError):
            ContinuousProcessor(window_size=5, step_size=-1)

class TestSpikeProcessor:
    def test_basic_processing(self):
        processor = SpikeProcessor(
            window_size=1.0,
            step_size=0.5,
            max_spikes_per_window=10
        )
        spike_data = [
            np.array([0.1, 0.5, 1.5, 2.0]),
            np.array([0.3, 1.0, 1.8])
        ]
        result = processor.process(spike_data, t_start=0, t_end=3)

        assert result.shape[1] == 2  # 2 channels
        assert result.shape[2] == 10  # max_spikes_per_window

    def test_empty_spike_train(self):
        processor = SpikeProcessor(window_size=1.0, step_size=1.0, max_spikes_per_window=5)
        spike_data = [np.array([]), np.array([1.0, 2.0])]

        # Should not raise error, just return zeros for empty channel
        result = processor.process(spike_data, t_start=0, t_end=3)
        assert torch.all(result[:, 0, :] == 0)  # First channel all zeros

    def test_auto_fit_max_spikes(self):
        processor = SpikeProcessor(window_size=1.0, step_size=1.0)
        spike_data = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]  # 5 spikes in 1 second

        result = processor.process(spike_data, t_start=0, t_end=2)
        assert processor.max_spikes == 5

    def test_all_empty_raises_error(self):
        processor = SpikeProcessor(window_size=1.0, step_size=1.0)
        spike_data = [np.array([]), np.array([])]

        with pytest.raises(ValueError, match="All spike trains are empty"):
            processor.process(spike_data)