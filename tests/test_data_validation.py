import pytest
import torch
import numpy as np
from neural_mi.validation import DataValidator

# --- Fixtures for test data ---

@pytest.fixture
def valid_continuous_data():
    return np.random.randn(10, 100), np.random.randn(10, 100)

@pytest.fixture
def valid_spike_data():
    return [np.sort(np.random.rand(10)) for _ in range(5)], [np.sort(np.random.rand(10)) for _ in range(5)]

# --- Test Cases ---

class TestDataValidator:
    def test_valid_continuous_data_passes(self, valid_continuous_data):
        """Tests that valid continuous data passes validation."""
        x, y = valid_continuous_data
        validator = DataValidator(x, y, 'continuous')
        try:
            validator.validate()
        except (TypeError, ValueError) as e:
            pytest.fail(f"Validation failed unexpectedly for valid continuous data: {e}")

    def test_valid_spike_data_passes(self, valid_spike_data):
        """Tests that valid spike data passes validation."""
        x, y = valid_spike_data
        validator = DataValidator(x, y, 'spike')
        try:
            validator.validate()
        except (TypeError, ValueError) as e:
            pytest.fail(f"Validation failed unexpectedly for valid spike data: {e}")

    def test_invalid_data_type_raises_error(self):
        """Tests that non-numeric/list data raises TypeError."""
        x = ["a", "b", "c"]
        y = [1, 2, 3]
        validator = DataValidator(x, y, 'continuous')
        with pytest.raises(TypeError, match="must contain numeric data"):
            validator.validate()

    def test_continuous_wrong_ndim_raises_error(self, valid_continuous_data):
        """Tests that continuous data with wrong dimensions raises ValueError."""
        x, y = valid_continuous_data
        x_1d = x.flatten()
        validator = DataValidator(x_1d, y, 'continuous')
        with pytest.raises(ValueError, match="must be 2D .* or 3D"):
            validator.validate()

    def test_spike_not_list_raises_error(self, valid_continuous_data):
        """Tests that spike data which is not a list raises TypeError."""
        x, y = valid_continuous_data
        validator = DataValidator(x, y, 'spike')
        with pytest.raises(TypeError, match="must be a list of arrays for spike data"):
            validator.validate()

    def test_spike_list_of_wrong_type_raises_error(self):
        """Tests that a list containing non-arrays for spike data raises TypeError."""
        x = [1, 2, 3]
        y = [np.array([1.0])]
        validator = DataValidator(x, y, 'spike')
        with pytest.raises(TypeError, match="must be np.ndarray"):
            validator.validate()

    def test_empty_continuous_data_raises_error(self):
        """Tests that empty continuous data raises ValueError."""
        x = np.array([[],[]])
        y = np.random.randn(10, 100)
        validator = DataValidator(x, y, 'continuous')
        with pytest.raises(ValueError, match="is empty"):
            validator.validate()

    def test_data_with_nan_raises_error(self, valid_continuous_data):
        """Tests that data containing NaN values raises ValueError."""
        x, y = valid_continuous_data
        x[0, 0] = np.nan
        validator = DataValidator(x, y, 'continuous')
        with pytest.raises(ValueError, match="contains NaN values"):
            validator.validate()

    def test_spike_with_negative_times_raises_error(self, valid_spike_data):
        """Tests that spike data with negative times raises ValueError."""
        x, y = valid_spike_data
        x[0][0] = -0.1
        validator = DataValidator(x, y, 'spike')
        with pytest.raises(ValueError, match="contains negative spike times"):
            validator.validate()

    def test_unsorted_spike_data_raises_warning(self, valid_spike_data):
        """Tests that unsorted spike data raises a warning and sorts the data."""
        x, y = valid_spike_data
        x_unsorted = x.copy()
        x_unsorted[0] = np.array([0.5, 0.1, 0.9])

        with pytest.warns(UserWarning, match="spike times are not sorted"):
            validator = DataValidator(x_unsorted, y, 'spike')
            validator.validate()

        # Check that the data was sorted in-place
        assert np.array_equal(x_unsorted[0], np.array([0.1, 0.5, 0.9]))