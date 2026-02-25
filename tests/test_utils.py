# tests/test_utils.py
import pytest
import torch
import numpy as np
from neural_mi.utils import get_device, build_critic, compute_cross_covariance_spectrum, compute_spectral_metrics
from neural_mi.models.critics import SeparableCritic, ConcatCritic, HybridCritic

# A list of all device types we want to test
DEVICES = ["cuda", "mps", "cpu"]

def is_available(device: str) -> bool:
    """Helper function to check if a torch device is available."""
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device == "cpu":
        return True
    return False

@pytest.mark.parametrize("target_device", DEVICES)
def test_get_device(target_device: str):
    """
    Tests that the get_device utility function correctly identifies and returns
    the specified torch.device, skipping if the hardware is unavailable.
    """
    if not is_available(target_device):
        pytest.skip(f"Device '{target_device}' not available on this system.")

    device = get_device(device_str=target_device)
    assert isinstance(device, torch.device)
    assert device.type == target_device

def test_get_device_auto_selection():
    """
    Tests the auto-selection logic of get_device when no device is specified.
    """
    device = get_device()
    assert isinstance(device, torch.device)

# A dummy set of parameters for building models
# Must include all defaults enforced by strict validation
DUMMY_EMBEDDING_PARAMS = {
    'input_dim_x': 10, 'input_dim_y': 10, 'embedding_dim': 4,
    'hidden_dim': 16, 'n_layers': 1, 'n_channels_x': 2, 'n_channels_y': 2,
    'window_size': 5,
    'use_variational': False, 'embedding_model': 'mlp', 'max_n_batches': 512,
    'kernel_size': 3, 'bidirectional': False, 'nhead': 4
}

def test_build_critic_concat():
    critic = build_critic('concat', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, ConcatCritic)


def test_build_critic_separable():
    critic = build_critic('separable', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, SeparableCritic)

def test_build_critic_hybrid():
    critic = build_critic('hybrid', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, HybridCritic)

# --- Spectral Metric Tests ---
def test_cross_covariance_spectrum_shape_and_values():
    """Test that SVD extracts correct singular values from embeddings."""
    # Create two identical embeddings (perfect correlation)
    zx = torch.randn(100, 10)
    zy = zx.clone()
    
    spectrum = compute_cross_covariance_spectrum(zx, zy)
    
    # 1. Output should be a numpy array
    assert isinstance(spectrum, np.ndarray)
    
    # 2. Maximum possible singular values is the bottleneck dimension (10)
    assert len(spectrum) == 10
    
    # 3. Singular values should be non-negative and sorted descendingly
    assert np.all(spectrum >= -1e-7)  # allow tiny numerical noise
    assert np.all(np.diff(spectrum) <= 1e-7)

def test_spectral_metrics_single_dimension():
    """Test metrics when only 1 dimension is utilized (perfectly concentrated)."""
    # A spectrum where all energy is in the first dimension
    spectrum = np.array([10.0, 0.0, 0.0, 0.0])
    metrics = compute_spectral_metrics(spectrum)
    
    assert np.isclose(metrics['pr_singular'], 1.0)
    assert np.isclose(metrics['pr_covariance'], 1.0)
    assert np.isclose(metrics['effective_rank'], 1.0)

def test_spectral_metrics_uniform_dimensions():
    """Test metrics when all dimensions are utilized equally."""
    # A spectrum where energy is perfectly distributed across 4 dimensions
    spectrum = np.array([2.0, 2.0, 2.0, 2.0])
    metrics = compute_spectral_metrics(spectrum)
    
    assert np.isclose(metrics['pr_singular'], 4.0)
    assert np.isclose(metrics['pr_covariance'], 4.0)
    assert np.isclose(metrics['effective_rank'], 4.0)