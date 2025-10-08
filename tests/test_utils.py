# tests/test_utils.py
import pytest
import torch
from neural_mi.utils import get_device, build_critic # Import build_critic
from neural_mi.models.critics import BilinearCritic, ConcatCritic, ConcatCriticCNN

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
DUMMY_EMBEDDING_PARAMS = {
    'input_dim_x': 10, 'input_dim_y': 10, 'embedding_dim': 4,
    'hidden_dim': 16, 'n_layers': 1, 'n_channels_x': 2, 'n_channels_y': 2,
    'window_size': 5
}

def test_build_critic_bilinear():
    critic = build_critic('bilinear', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, BilinearCritic)

def test_build_critic_concat():
    critic = build_critic('concat', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, ConcatCritic)

def test_build_critic_cnn_fails_with_mlp():
    with pytest.raises(ValueError):
        build_critic('concat_cnn', {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'mlp'})

def test_build_critic_concat_cnn():
    critic = build_critic('concat_cnn', {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'cnn'})
    assert isinstance(critic, ConcatCriticCNN)