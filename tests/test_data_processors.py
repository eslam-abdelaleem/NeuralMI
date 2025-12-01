# tests/test_data_processors.py
import pytest
import numpy as np
import torch
from neural_mi.data.handler import WindowManager, PairedTemporalDataset, PairedDataset
from neural_mi.data.temporal import ContinuousWindowDataset, SpikeWindowDataset, CategoricalWindowDataset
from neural_mi.data.static import StaticDataset

# Fixtures for data generation
@pytest.fixture
def continuous_data():
    """Generate continuous data and time vector."""
    return np.random.randn(2, 100), np.arange(100)

@pytest.fixture
def spike_data():
    """Generate spike data."""
    return [np.sort(np.random.rand(200) * 100), np.sort(np.random.rand(150) * 100)]

@pytest.fixture
def categorical_data():
    """Generate categorical data and time vector."""
    return np.random.randint(0, 3, size=(2, 100)), np.arange(100)

# WindowManager Tests
def test_window_manager_creation():
    """Test basic creation of WindowManager."""
    wm = WindowManager(window_size=1.0, t_start=0.0, t_end=10.0)
    assert wm.n_windows == 10
    assert np.allclose(wm.window_times, np.arange(0.0, 10.0, 1.0))


# ContinuousWindowDataset Tests
def test_continuous_window_dataset(continuous_data):
    """Test ContinuousWindowDataset windowing."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    assert len(dataset) == 10
    assert dataset.data.shape[0] == 10
    assert dataset.data.shape[1] == 2
    assert dataset.data.shape[2] > 0

def test_continuous_window_with_jumps():
    """Test ContinuousWindowDataset dealing with jumps in time with no data"""
    x = np.hstack((np.linspace(0, 100, 1000), np.linspace(200, 300, 1000)))
    dat = np.vstack((np.sin(x), np.cos(x)))
    window_size = 1.5
    wm = WindowManager(window_size, t_start=x[0], t_end=x[-1])
    dataset = ContinuousWindowDataset(dat, time_vector=x, window_manager=wm)
    window_on_jump_ind = np.floor(100/window_size).astype(int)
    jump_window = dataset[window_on_jump_ind]
    valid_windows = dataset.validate_window_coverage()
    assert torch.all(jump_window[:,0:5] != 0.0) # Start of window has data
    assert torch.all(jump_window[:,-5:] == 0.0) # End of window has just zeros
    assert wm.n_windows == len(valid_windows)
    # +1 is b/c last window is open interval (window_times only define start of each window)
    assert (np.sum(wm.window_times < 100) + np.sum(wm.window_times > 200) + 1) == np.sum(valid_windows)


# SpikeWindowDataset Tests
def test_spike_window_dataset(spike_data):
    """Test SpikeWindowDataset windowing."""
    wm = WindowManager(window_size=1.0, t_start=0, t_end=100)
    dataset = SpikeWindowDataset(spike_data, window_manager=wm)
    dataset.move_data_to_windows()
    assert len(dataset) == 100
    assert dataset.data.shape[0] == wm.n_windows
    assert dataset.data.shape[1] == 2
    assert dataset.data.shape[2] > 0

# CategoricalWindowDataset Tests
def test_categorical_window_dataset(categorical_data):
    """Test CategoricalWindowDataset windowing."""
    data, time = categorical_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = CategoricalWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    assert len(dataset) == 10
    assert dataset.data.shape[0] == 10
    assert dataset.data.shape[1] == 2
    assert dataset.data.shape[2] > 0
    
# PairedTemporalDataset Tests
def test_paired_temporal_dataset(continuous_data, spike_data):
    """Test PairedTemporalDataset alignment and windowing."""
    c_data, c_time = continuous_data
    s_data = spike_data

    # Initialize datasets without window manager first
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    assert paired_dataset.window_manager.n_windows > 0
    assert len(paired_dataset) == paired_dataset.window_manager.n_windows
    
    # Test __getitem__
    x_sample, y_sample = paired_dataset[0]
    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)

def test_large_time_jumps(continuous_data):
    """Test handling of large time jumps in temporal data."""
    data, time = continuous_data
    time[50:] += 100  # Create a large jump in the middle
    
    wm = WindowManager(window_size=10, t_start=0, t_end=200)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    
    # This should still run without errors
    dataset.move_data_to_windows()
    
    # Check that windows in the gap are empty (or have some fill value)
    # The validation logic should handle this.
    valid_mask = dataset.validate_window_coverage()
    
    # Windows covering the jump (e.g., from t=90 to t=150) should be invalid
    jump_windows = (wm.window_times >= 90) & (wm.window_times < 150)
    assert not np.any(valid_mask[jump_windows])

# StaticDataset Tests
def test_static_dataset():
    """Test StaticDataset with various input shapes."""
    data_2d = np.random.randn(2, 100)
    dataset_2d = StaticDataset(data_2d)
    assert dataset_2d.data.shape == (100, 2, 1)
    
    data_3d = np.random.randn(2, 100, 5)
    dataset_3d = StaticDataset(data_3d)
    assert dataset_3d.data.shape == (100, 2, 5)

# PairedDataset Tests
def test_paired_static_dataset():
    """Test PairedDataset for static data."""
    x_data = np.random.randn(2, 100)
    y_data = np.random.randn(3, 100)
    
    x_dataset = StaticDataset(x_data)
    y_dataset = StaticDataset(y_data)
    
    paired_dataset = PairedDataset(x_dataset, y_dataset)
    assert len(paired_dataset) == 100
    
    x_sample, y_sample = paired_dataset[0]
    assert x_sample.shape == (2, 1)
    assert y_sample.shape == (3, 1)

