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
    return np.random.randn(100, 2), np.arange(100)

@pytest.fixture
def spike_data():
    """Generate spike data."""
    return [np.sort(np.random.rand(200) * 100), np.sort(np.random.rand(150) * 100)]

@pytest.fixture
def categorical_data():
    """Generate categorical data and time vector."""
    return np.random.randint(0, 3, size=(100, 2)), np.arange(100)

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
    dat = np.vstack((np.sin(x), np.cos(x))).T
    window_size = 1.5
    wm = WindowManager(window_size, t_start=x[0], t_end=x[-1])
    dataset = ContinuousWindowDataset(dat, time_vector=x, window_manager=wm)
    window_on_jump_ind = np.floor(100/window_size).astype(int)
    jump_window = dataset[window_on_jump_ind]
    valid_windows = dataset.validate_window_coverage()
    assert torch.all(jump_window[:,0:5] != 0.0) # Start of window has data
    # NOTE: New interpolation logic interpolates across gaps, so zeros are not guaranteed unless we mask
    # assert torch.all(jump_window[:,-5:] == 0.0)
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
    data_2d = np.random.randn(100, 2).astype(np.float32)
    dataset_2d = StaticDataset(data_2d)
    assert dataset_2d.data.shape == (100, 2, 1)
    
    data_3d = np.random.randn(100, 2, 5).astype(np.float32)
    dataset_3d = StaticDataset(data_3d)
    assert dataset_3d.data.shape == (100, 2, 5)

# PairedDataset Tests
def test_paired_static_dataset():
    """Test PairedDataset for static data."""
    x_data = np.random.randn(100, 2).astype(np.float32)
    y_data = np.random.randn(100, 3).astype(np.float32)
    
    x_dataset = StaticDataset(x_data)
    y_dataset = StaticDataset(y_data)
    
    paired_dataset = PairedDataset(x_dataset, y_dataset)
    assert len(paired_dataset) == 100
    
    x_sample, y_sample = paired_dataset[0]
    assert x_sample.shape == (2, 1)
    assert y_sample.shape == (3, 1)


# Time Shift Tests
def test_continuous_time_shift_zero(continuous_data):
    """Test that zero time shift leaves data unchanged."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_data = dataset.data.clone()
    # Apply zero shift
    dataset.time_shift(0)
    dataset.move_data_to_windows()

    assert torch.allclose(dataset.data, original_data)


def test_continuous_time_shift_effects(continuous_data):
    """Test that time shifts have the expected effect on continuous data."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_data = dataset.data.clone()
    original_time = dataset.time_vector.copy()
    # Apply positive shift
    shift_amount = 5.0
    dataset.time_shift(shift_amount)

    # Check time vector shifted correctly
    assert np.allclose(dataset.time_vector, original_time + shift_amount)
    # Shift back to zero
    dataset.time_shift(0)
    assert np.allclose(dataset.time_vector, original_time)
    assert torch.allclose(dataset.data, original_data)


def test_spike_time_shift_zero(spike_data):
    """Test that zero time shift leaves spike data unchanged."""
    wm = WindowManager(window_size=1.0, t_start=0, t_end=100)
    dataset = SpikeWindowDataset(spike_data, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_data = dataset.data.clone()
    original_spike_times = [st.copy() for st in dataset.data_orig]
    # Apply zero shift
    dataset.time_shift(0)
    dataset.move_data_to_windows()
    
    assert torch.allclose(dataset.data, original_data)
    for orig, current in zip(original_spike_times, dataset.data_orig):
        assert np.allclose(orig, current)


def test_spike_time_shift_effects(spike_data):
    """Test that time shifts have the expected effect on spike data."""
    wm = WindowManager(window_size=1.0, t_start=0, t_end=100)
    dataset = SpikeWindowDataset(spike_data, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original spike times
    original_spike_times = [st.copy() for st in dataset.data_orig]
    # Apply positive shift
    shift_amount = 5.0
    dataset.time_shift(shift_amount)
    # Check spike times shifted correctly
    for orig, shifted in zip(original_spike_times, dataset.data_orig):
        assert np.allclose(shifted, orig + shift_amount)
    # Apply negative shift back to original
    dataset.time_shift(0)
    for orig, shifted in zip(original_spike_times, dataset.data_orig):
        assert np.allclose(shifted, orig)


def test_paired_time_shift_positive(continuous_data, spike_data):
    """Test positive time shifts on paired dataset."""
    c_data, c_time = continuous_data
    s_data = spike_data
    
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    # Store original
    original_x_time = paired_dataset.x_dataset.time_vector.copy()
    original_y_times = [st.copy() for st in paired_dataset.y_dataset.data_orig]
    original_x_data = paired_dataset.x_data.detach().clone()
    # Apply positive shifts
    shift_x = 10.0
    shift_y = 5.0
    paired_dataset.time_shift(offset_x=shift_x, offset_y=shift_y)
    # Check shifts applied correctly
    assert np.allclose(paired_dataset.x_dataset.time_vector, original_x_time + shift_x)
    for orig, shifted in zip(original_y_times, paired_dataset.y_dataset.data_orig):
        assert np.allclose(shifted, orig + shift_y)
    # Undo shifts, check back to normal
    paired_dataset.time_shift(offset_x=0.0, offset_y=0.0)
    assert torch.allclose(paired_dataset.x_data, original_x_data)


def test_paired_time_shift_negative(continuous_data, spike_data):
    """Test negative time shifts on paired dataset."""
    c_data, c_time = continuous_data
    s_data = spike_data
    
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    # Store original
    original_x_time = paired_dataset.x_dataset.time_vector.copy()
    original_y_times = [st.copy() for st in paired_dataset.y_dataset.data_orig]
    # Apply negative shifts
    shift_x = -15.0
    shift_y = -20.0
    paired_dataset.time_shift(offset_x=shift_x, offset_y=shift_y)
    
    # Check shifts applied correctly
    assert np.allclose(paired_dataset.x_dataset.time_vector, original_x_time + shift_x)
    for orig, shifted in zip(original_y_times, paired_dataset.y_dataset.data_orig):
        assert np.allclose(shifted, orig + shift_y)


def test_paired_time_shift_large(continuous_data, spike_data):
    """Test very large time shifts (up to t_end) on paired dataset."""
    c_data, c_time = continuous_data
    s_data = spike_data
    
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    # Store original
    original_x_time = paired_dataset.x_dataset.time_vector.copy()
    original_y_times = [st.copy() for st in paired_dataset.y_dataset.data_orig]
    t_end = paired_dataset.window_manager.t_end
    # Apply large shift close to t_end
    shift_amount = t_end - 10.0
    paired_dataset.time_shift(offset_x=shift_amount, offset_y=shift_amount)

    # Check shifts applied correctly
    assert np.allclose(paired_dataset.x_dataset.time_vector, original_x_time + shift_amount)
    for orig, shifted in zip(original_y_times, paired_dataset.y_dataset.data_orig):
        assert np.allclose(shifted, orig + shift_amount)


def test_paired_time_shift_very_small(continuous_data, spike_data):
    """Test very small time shifts on paired dataset."""
    c_data, c_time = continuous_data
    s_data = spike_data
    
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    # Store original
    original_x_time = paired_dataset.x_dataset.time_vector.copy()
    # Apply very small shift
    shift_amount = 0.001
    paired_dataset.time_shift(offset_x=shift_amount, offset_y=shift_amount)

    # Check shifts applied correctly (use appropriate tolerance)
    assert np.allclose(paired_dataset.x_dataset.time_vector, original_x_time + shift_amount, atol=1e-10)


def test_time_shift_roundtrip(continuous_data):
    """Test that shifting forward and back returns to original state."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_time = dataset.time_vector.copy()
    # Shift forward, then back
    dataset.time_shift(25.0)
    dataset.time_shift(0)
    
    assert np.allclose(dataset.time_vector, original_time)


# Noise Tests
def test_continuous_noise_application(continuous_data):
    """Test that noise corrupts continuous data."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_data = dataset.data.clone()
    # Apply noise
    noise_amplitude = 0.5
    dataset.apply_noise(noise_amplitude)

    # Data should be different
    assert not torch.allclose(dataset.data, original_data)
    
    # But should be close (within reasonable bounds)
    diff = torch.abs(dataset.data - original_data)
    assert torch.all(diff < noise_amplitude * 5)  # Within ~5 sigma


def test_continuous_noise_reset(continuous_data):
    """Test that reset removes noise and returns to original data."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    # Store original
    original_data = dataset.data.clone()
    # Apply noise
    dataset.apply_noise(0.5)

    # Verify data changed
    assert not torch.allclose(dataset.data, original_data)
    # Reset
    dataset.reset()
    # Should match original exactly
    assert torch.allclose(dataset.data, original_data)


def test_spike_noise_application(spike_data):
    """Test that noise corrupts spike data."""
    wm = WindowManager(window_size=1.0, t_start=0, t_end=100)
    dataset = SpikeWindowDataset(spike_data, window_manager=wm)
    dataset.move_data_to_windows()
    # Store original
    original_data = dataset.data.clone()
    # Apply temporal jitter
    jitter_amplitude = 0.1
    dataset.apply_noise(jitter_amplitude)

    # Data should be different (at spike locations)
    assert not torch.allclose(dataset.data, original_data)
    # Differences should be bounded by jitter amplitude
    diff = torch.abs(dataset.data - original_data)
    non_zero_mask = original_data != dataset.no_spike_value
    assert torch.all(diff[non_zero_mask] <= jitter_amplitude / 2 * 1.01)  # Small tolerance for rounding


def test_spike_noise_reset(spike_data):
    """Test that reset removes noise from spike data."""
    wm = WindowManager(window_size=1.0, t_start=0, t_end=100)
    dataset = SpikeWindowDataset(spike_data, window_manager=wm)
    dataset.move_data_to_windows()
    # Store original
    original_data = dataset.data.clone()
    # Apply jitter
    dataset.apply_noise(0.1)

    # Verify data changed
    assert not torch.allclose(dataset.data, original_data)
    # Reset
    dataset.reset()
    # Should match original exactly
    assert torch.allclose(dataset.data, original_data)


def test_paired_noise_both_datasets(continuous_data, spike_data):
    """Test noise application to both datasets in a pair."""
    c_data, c_time = continuous_data
    s_data = spike_data
    
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    # Store original
    original_x = paired_dataset.x_dataset.data.clone()
    original_y = paired_dataset.y_dataset.data.clone()
    # Apply noise to both
    # paired_dataset.add_noise(amplitude_x=0.5, amplitude_y=0.1)
    # FIX: Method is apply_noise, not add_noise
    paired_dataset.apply_noise(amplitude_x=0.5, amplitude_y=0.1)

    # Both should be different
    assert not torch.allclose(paired_dataset.x_dataset.data, original_x)
    assert not torch.allclose(paired_dataset.y_dataset.data, original_y)


def test_paired_noise_selective(continuous_data, spike_data):
    """Test noise application to only one dataset in a pair."""
    c_data, c_time = continuous_data
    s_data = spike_data
    
    x_dataset = ContinuousWindowDataset(c_data, c_time)
    y_dataset = SpikeWindowDataset(s_data)
    paired_dataset = PairedTemporalDataset(x_dataset, y_dataset, window_size=1.0)
    
    # Store original
    original_x = paired_dataset.x_dataset.data.clone()
    original_y = paired_dataset.y_dataset.data.clone()
    # Apply noise only to X
    # paired_dataset.add_noise(amplitude_x=0.5, amplitude_y=0)
    # FIX: Method is apply_noise
    paired_dataset.apply_noise(amplitude_x=0.5, amplitude_y=0)

    # Only X should be different
    assert not torch.allclose(paired_dataset.x_dataset.data, original_x)
    assert torch.allclose(paired_dataset.y_dataset.data, original_y)


def test_multiple_noise_applications(continuous_data):
    """Test applying noise multiple times with resets."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_data = dataset.data.clone()
    # Apply noise, reset, apply again
    dataset.apply_noise(0.3)
    noisy_data_1 = dataset.data.clone()
    dataset.reset()
    dataset.apply_noise(0.3)
    noisy_data_2 = dataset.data.clone()

    # After reset, original should be restored
    dataset.reset()
    assert torch.allclose(dataset.data, original_data)
    # Two different noise applications should produce different results
    assert not torch.allclose(noisy_data_1, noisy_data_2)


def test_noise_then_time_shift(continuous_data):
    """Test that noise and time shift operations are independent."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Store original
    original_time = dataset.time_vector.copy()
    # Apply noise
    dataset.apply_noise(0.5)
    # Time shift
    dataset.time_shift(10.0)
    dataset.move_data_to_windows() # Must update windows after shift

    # Time should have shifted regardless of noise
    assert np.allclose(dataset.time_vector, original_time + 10.0)
    # Reset should not affect time shift
    dataset.reset()
    assert np.allclose(dataset.time_vector, original_time + 10.0)


def test_precision_continuous(continuous_data):
    """Test precision/rounding on continuous data."""
    data, time = continuous_data
    wm = WindowManager(window_size=10, t_start=0, t_end=100)
    dataset = ContinuousWindowDataset(data, time, window_manager=wm)
    dataset.move_data_to_windows()
    
    # Apply precision
    precision = 0.1
    dataset.apply_precision(precision)
    # All non-zero values should be multiples of precision
    mask = dataset._data_mask
    values = dataset.data[mask] / precision
    assert torch.allclose(values, torch.round(values), atol=1e-5, rtol=1e-5)


def test_precision_spike(spike_data):
    """Test precision/rounding on spike data."""
    wm = WindowManager(window_size=1.0, t_start=0, t_end=100)
    dataset = SpikeWindowDataset(spike_data, window_manager=wm)
    dataset.move_data_to_windows()

    # Apply precision
    precision = 0.05
    dataset.apply_precision(precision)

    # All spike times should be multiples of precision
    mask = dataset._data_mask
    values = dataset.data[mask] / precision
    assert torch.allclose(values, torch.round(values), atol=1e-5, rtol=1e-5)


# --- Data Splitting Tests ---

import neural_mi as nmi

_BASE_PARAMS_SPLIT = {
    'n_epochs': 2, 'learning_rate': 1e-4, 'batch_size': 32,
    'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 2, 'patience': 1,
}


@pytest.fixture
def iid_3d_data():
    """Pre-processed 3D IID Gaussian data: (n_samples, 1, channels)."""
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=100, dim=2, mi=1.0)
    return x.unsqueeze(1), y.unsqueeze(1)


def test_random_split_mode(iid_3d_data):
    """Random split mode runs without error and returns a float MI estimate."""
    x, y = iid_3d_data
    results = nmi.run(
        x_data=x, y_data=y, mode='estimate',
        base_params=_BASE_PARAMS_SPLIT, split_mode='random', verbose=False
    )
    assert results.mi_estimate is not None
    assert not np.isnan(results.mi_estimate)


def test_custom_indices_split(iid_3d_data):
    """Providing explicit train/test indices bypasses automatic splitting."""
    x, y = iid_3d_data
    n = x.shape[0]
    idx = np.random.permutation(n)
    results = nmi.run(
        x_data=x, y_data=y, mode='estimate',
        base_params=_BASE_PARAMS_SPLIT,
        train_indices=idx[:80], test_indices=idx[80:], verbose=False
    )
    assert results.mi_estimate is not None
    assert not np.isnan(results.mi_estimate)


def test_default_blocked_split(iid_3d_data):
    """Default blocked split mode runs correctly."""
    x, y = iid_3d_data
    results = nmi.run(
        x_data=x, y_data=y, mode='estimate',
        base_params=_BASE_PARAMS_SPLIT, split_mode='blocked', verbose=False
    )
    assert results.mi_estimate is not None
    assert not np.isnan(results.mi_estimate)


# --- Mixed-Modality Alignment Tests ---

from neural_mi.data.handler import create_dataset


def test_continuous_and_spike_alignment():
    """Aligned windows from a continuous stream and a spike stream have equal sample counts."""
    time_cont = np.arange(1000) / 100.0
    x_cont = np.random.randn(1000, 2)
    y_spike, _ = nmi.generators.generate_correlated_spike_trains(duration=10.0)

    dataset = create_dataset(
        x_data=x_cont, y_data=y_spike,
        x_time=time_cont,
        processor_type_x='continuous', processor_params_x={'window_size': 0.1},
        processor_type_y='spike', processor_params_y={'window_size': 0.1}
    )
    assert dataset.x_data.shape[0] == dataset.y_data.shape[0]
    # 10 s / 0.1 s = 100 windows expected (±10 tolerance)
    assert 90 <= dataset.x_data.shape[0] <= 110


def test_different_continuous_params_alignment():
    """Two continuous streams with different lengths are aligned to the shorter one."""
    x_data = np.random.randn(100, 1)
    y_data = np.random.randn(120, 1)

    dataset = create_dataset(
        x_data=x_data, y_data=y_data,
        processor_type_x='continuous', processor_params_x={'window_size': 10},
        processor_type_y='continuous', processor_params_y={'window_size': 10}
    )
    assert dataset.x_data.shape[0] == dataset.y_data.shape[0]
    assert dataset.x_data.shape[0] >= 9


# ---------------------------------------------------------------------------
# dataset_device tests
# ---------------------------------------------------------------------------

class TestDatasetDevice:
    """Verify the dataset_device parameter correctly controls tensor placement."""

    def test_static_dataset_defaults_to_cpu(self):
        """StaticDataset should store data on CPU by default."""
        data = np.random.randn(50, 4)
        ds = StaticDataset(data)
        assert ds.data.device.type == 'cpu'
        assert ds.data_device == torch.device('cpu')

    def test_static_dataset_explicit_cpu(self):
        """Explicit data_device='cpu' keeps data on CPU."""
        data = np.random.randn(50, 4)
        ds = StaticDataset(data, data_device='cpu')
        assert ds.data.device.type == 'cpu'

    def test_static_dataset_auto_uses_compute_device(self):
        """data_device='auto' should place data on the compute device."""
        from neural_mi.utils import get_device
        compute_dev = get_device()
        data = np.random.randn(50, 4)
        ds = StaticDataset(data, data_device='auto')
        assert ds.data.device.type == compute_dev.type

    def test_create_dataset_cpu_default(self):
        """create_dataset default keeps both x and y data on CPU."""
        x = np.random.randn(60, 3)
        y = np.random.randn(60, 3)
        dataset = create_dataset(x, y)
        assert dataset.x_data.device.type == 'cpu'
        assert dataset.y_data.device.type == 'cpu'

    def test_create_dataset_explicit_cpu(self):
        """Explicit data_device='cpu' in create_dataset keeps data on CPU."""
        x = np.random.randn(60, 3)
        y = np.random.randn(60, 3)
        dataset = create_dataset(x, y, data_device='cpu')
        assert dataset.x_data.device.type == 'cpu'
        assert dataset.y_data.device.type == 'cpu'

    def test_create_dataset_auto(self):
        """data_device='auto' in create_dataset uses the compute device."""
        from neural_mi.utils import get_device
        compute_dev = get_device()
        x = np.random.randn(60, 3)
        y = np.random.randn(60, 3)
        dataset = create_dataset(x, y, data_device='auto')
        assert dataset.x_data.device.type == compute_dev.type
        assert dataset.y_data.device.type == compute_dev.type

    def test_temporal_dataset_defaults_to_cpu(self):
        """ContinuousWindowDataset should store windowed data on CPU by default."""
        data = np.random.randn(100, 2)
        time = np.arange(100, dtype=float)
        wm = WindowManager(window_size=10, t_start=0, t_end=100)
        ds = ContinuousWindowDataset(data, time, window_manager=wm)
        ds.move_data_to_windows()
        assert ds.data.device.type == 'cpu'
        assert ds.data_master.device.type == 'cpu'

    def test_temporal_dataset_auto(self):
        """ContinuousWindowDataset with data_device='auto' uses compute device."""
        from neural_mi.utils import get_device
        compute_dev = get_device()
        data = np.random.randn(100, 2)
        time = np.arange(100, dtype=float)
        wm = WindowManager(window_size=10, t_start=0, t_end=100)
        ds = ContinuousWindowDataset(data, time, window_manager=wm, data_device='auto')
        ds.move_data_to_windows()
        assert ds.data.device.type == compute_dev.type

    def test_noise_buffer_device_matches_data(self):
        """Noise buffer in apply_noise must be on the same device as self.data."""
        data = np.random.randn(100, 2)
        time = np.arange(100, dtype=float)
        wm = WindowManager(window_size=10, t_start=0, t_end=100)
        ds = ContinuousWindowDataset(data, time, window_manager=wm)
        ds.move_data_to_windows()
        ds.apply_noise(0.01)
        # After apply_noise the noise buffer should share device with self.data
        assert ds._noise_buffer.device == ds.data.device

    def test_dataset_device_propagates_through_run(self):
        """run() with dataset_device='cpu' in base_params completes without error."""
        import neural_mi as nmi
        x = np.random.randn(80, 4, 1)
        y = np.random.randn(80, 4, 1)
        result = nmi.run(x, y, base_params={
            'n_epochs': 1, 'batch_size': 32, 'patience': 1,
            'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
            'dataset_device': 'cpu',
        }, n_workers=1)
        assert hasattr(result, 'mi_estimate')

    def test_dataset_device_auto_in_run(self):
        """run() with dataset_device='auto' in base_params completes without error."""
        import neural_mi as nmi
        x = np.random.randn(80, 4, 1)
        y = np.random.randn(80, 4, 1)
        result = nmi.run(x, y, base_params={
            'n_epochs': 1, 'batch_size': 32, 'patience': 1,
            'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
            'dataset_device': 'auto',
        }, n_workers=1)
        assert hasattr(result, 'mi_estimate')