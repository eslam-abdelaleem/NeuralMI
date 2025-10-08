# tests/test_data_processors.py
import pytest
import numpy as np
import torch
from neural_mi.data.processors import ContinuousProcessor, SpikeProcessor, CategoricalProcessor, find_max_spikes_per_window
from neural_mi.exceptions import InsufficientDataError, DataShapeError

class TestContinuousProcessor:
    def test_basic_windowing(self):
        processor = ContinuousProcessor(window_size=3, step_size=1)
        data = np.arange(10).reshape(1, -1)
        result = processor.process(data)
        assert result.shape == (8, 1, 3)
        assert torch.equal(result[0, 0, :], torch.tensor([0., 1., 2.]))

    def test_step_size(self):
        processor = ContinuousProcessor(window_size=3, step_size=2)
        data = np.arange(10).reshape(1, -1)
        result = processor.process(data)
        assert result.shape == (4, 1, 3)

    def test_data_too_short_raises_error(self):
        processor = ContinuousProcessor(window_size=10)
        data = np.arange(5).reshape(1, -1)
        with pytest.raises(InsufficientDataError):
            processor.process(data)

    def test_window_size_equals_data_length(self):
        """Tests behavior when window_size is the same as the data length."""
        processor = ContinuousProcessor(window_size=10, step_size=1)
        data = np.arange(10).reshape(1, -1)
        result = processor.process(data)
        assert result.shape == (1, 1, 10)

    def test_empty_input_data(self):
        """Tests that an empty array is handled correctly."""
        processor = ContinuousProcessor(window_size=10)
        data = np.empty((2, 0)) # 2 channels, 0 timepoints
        with pytest.raises(InsufficientDataError):
            processor.process(data)

    def test_step_size_larger_than_window(self):
        """Tests non-overlapping windows."""
        processor = ContinuousProcessor(window_size=3, step_size=3)
        data = np.arange(9).reshape(1, -1)
        result = processor.process(data)
        assert result.shape == (3, 1, 3)
        assert torch.equal(result[1, 0, :], torch.tensor([3., 4., 5.]))


class TestSpikeProcessor:
    def test_basic_processing(self):
        processor = SpikeProcessor(window_size=1.0, step_size=0.5, max_spikes_per_window=5)
        spike_data = [np.array([0.1, 0.8, 1.6, 2.2]), np.array([0.4, 1.9])]
        result = processor.process(spike_data, t_start=0, t_end=3)
        assert result.shape == (5, 2, 5) # 5 windows, 2 channels, 5 max spikes
        # Check first window
        assert torch.allclose(result[0, 0, :2], torch.tensor([0.1, 0.8]))
        assert result[0, 0, 2] == 0 # Padded

    def test_find_max_spikes_utility(self):
        """Tests the new standalone utility function."""
        spike_data = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
        max_spikes = find_max_spikes_per_window(spike_data, window_size=0.3)
        assert max_spikes == 4

    def test_empty_list_of_spikes(self):
        """Tests when the input list itself is empty."""
        processor = SpikeProcessor(window_size=1.0, step_size=0.1, max_spikes_per_window=5)
        result = processor.process([])
        assert result.shape == (0, 0, 5)

    def test_list_of_empty_spike_arrays(self):
        """Tests that a list of empty arrays returns a tensor with 0 samples."""
        processor = SpikeProcessor(window_size=1.0, step_size=0.1, max_spikes_per_window=1)
        spike_data = [np.array([]), np.array([])]
        # This no longer raises an error. It should return a tensor with the
        # correct number of channels (2) but zero samples (windows).
        result = processor.process(spike_data)
        assert result.shape == (0, 2, 1)

    def test_window_larger_than_duration(self):
        """Tests when the window is too large to fit in the data range."""
        processor = SpikeProcessor(window_size=5.0, step_size=1.0, max_spikes_per_window=1)
        spike_data = [np.array([0.1, 1.2, 2.3])]
        result = processor.process(spike_data, t_start=0, t_end=3)
        # Should produce zero windows
        assert result.shape[0] == 0

    def test_truncation_of_spikes(self):
        """Tests that spikes are correctly truncated if they exceed max_spikes."""
        processor = SpikeProcessor(window_size=1.0, step_size=1.0, max_spikes_per_window=3)
        spike_data = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]
        result = processor.process(spike_data, t_start=0, t_end=1)
        assert result.shape == (1, 1, 3)
        assert torch.all(result[0, 0, :] > 0) # All 3 spots should be filled
        # Check that the 4th spike (0.4) is not present
        assert 0.4 not in result[0, 0, :].numpy()

class TestCategoricalProcessor:
    @pytest.fixture
    def categorical_data(self):
        # Data: 2 channels, 10 timepoints, 3 categories (0, 1, 2)
        return np.array([
            [0, 1, 1, 2, 0, 0, 1, 2, 2, 1],
            [1, 1, 0, 2, 1, 0, 0, 2, 1, 0]
        ])

    def test_basic_processing(self, categorical_data):
        processor = CategoricalProcessor(window_size=3, step_size=1)
        result = processor.process(categorical_data)
        
        # n_windows = 10 - 3 + 1 = 8
        # n_channels = 2
        # n_features = window_size * n_categories = 3 * 3 = 9
        assert result.shape == (8, 2, 9)
        assert result.dtype == torch.float32

        # Check the content of the first window, first channel: [0, 1, 1]
        # One-hot: [[1,0,0], [0,1,0], [0,1,0]] -> flattened -> [1,0,0,0,1,0,0,1,0]
        expected = torch.tensor([1.,0.,0., 0.,1.,0., 0.,1.,0.])
        assert torch.equal(result[0, 0, :], expected)

    def test_data_too_short_raises_error(self, categorical_data):
        processor = CategoricalProcessor(window_size=20)
        with pytest.raises(InsufficientDataError):
            processor.process(categorical_data)