import pytest
import numpy as np
import torch
from neural_mi.data.handler import DataHandler

class TestDataHandler:
    def test_channels_first_data_is_not_transposed(self):
        """
        Tests that data with data_format='channels_first' is not transposed.
        """
        x_data = np.random.randn(5, 100)  # (channels, time)
        y_data = np.random.randn(5, 100)
        processor_params = {'window_size': 10, 'data_format': 'channels_first'}

        handler = DataHandler(x_data, y_data, 'continuous', processor_params)
        x_processed, y_processed = handler.process()

        # After windowing, the shape should be (n_windows, n_channels, window_size)
        # n_windows = 100 - 10 + 1 = 91
        assert x_processed.shape == (91, 5, 10)
        assert y_processed.shape == (91, 5, 10)

    def test_channels_last_data_is_transposed(self):
        """
        Tests that data with data_format='channels_last' is correctly transposed.
        """
        x_data = np.random.randn(100, 5)  # (time, channels)
        y_data = np.random.randn(100, 5)
        processor_params = {'window_size': 10, 'data_format': 'channels_last'}

        handler = DataHandler(x_data, y_data, 'continuous', processor_params)
        x_processed, y_processed = handler.process()

        # After transposing and windowing, shape should be (n_windows, n_channels, window_size)
        # n_windows = 100 - 10 + 1 = 91
        assert x_processed.shape == (91, 5, 10)
        assert y_processed.shape == (91, 5, 10)

    def test_default_data_format_is_channels_first(self):
        """
        Tests that the default data_format is 'channels_first'.
        """
        x_data = np.random.randn(5, 100)  # (channels, time)
        y_data = np.random.randn(5, 100)
        processor_params = {'window_size': 10} # No data_format specified

        handler = DataHandler(x_data, y_data, 'continuous', processor_params)
        x_processed, y_processed = handler.process()

        assert x_processed.shape == (91, 5, 10)
        assert y_processed.shape == (91, 5, 10)

    def test_invalid_data_format_raises_error(self):
        """
        Tests that an invalid data_format string raises a ValueError.
        """
        x_data = np.random.randn(100, 5)
        y_data = np.random.randn(100, 5)
        processor_params = {'window_size': 10, 'data_format': 'invalid_format'}

        handler = DataHandler(x_data, y_data, 'continuous', processor_params)

        with pytest.raises(ValueError, match="Invalid data_format"):
            handler.process()

    def test_preprocessed_data_is_not_processed(self):
        """
        Tests that 3D data is passed through without processing when processor_type is None.
        """
        x_data = np.random.randn(100, 5, 10) # (samples, channels, features)
        y_data = np.random.randn(100, 5, 10)

        handler = DataHandler(x_data, y_data, processor_type=None)
        x_processed, y_processed = handler.process()

        assert torch.is_tensor(x_processed)
        assert torch.equal(torch.from_numpy(x_data).float(), x_processed)
        assert torch.equal(torch.from_numpy(y_data).float(), y_processed)