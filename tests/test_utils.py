# tests/test_utils.py
import pytest
import torch
import numpy as np
from neural_mi.utils import (
    get_device, build_critic, build_optimizer_and_scheduler,
    compute_cross_covariance_spectrum, compute_spectral_metrics,
)
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


def test_build_critic_custom_embedding_minimal_signature():
    """A custom embedding class following the minimal BaseEmbedding contract
    (input_dim, hidden_dim, embed_dim, n_layers) must build without receiving
    MLP-specific kwargs (use_spectral_norm/dropout/norm_layer)."""
    import torch.nn as nn
    from neural_mi.models.embeddings import BaseEmbedding

    class MinimalCustom(BaseEmbedding):
        def __init__(self, input_dim, hidden_dim, embed_dim, n_layers):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, embed_dim))

        def forward(self, x):
            return self.net(x.view(x.shape[0], -1))

    critic = build_critic('separable', DUMMY_EMBEDDING_PARAMS,
                          custom_embedding_cls=MinimalCustom)
    assert isinstance(critic, SeparableCritic)
    assert isinstance(critic.embedding_net_x, MinimalCustom)


# --- build_optimizer_and_scheduler (C-OPTIM: shared by task.py and precision.py) ---

class TestBuildOptimizerAndScheduler:
    def _critic(self):
        return build_critic('separable', DUMMY_EMBEDDING_PARAMS)

    def test_default_adam_optimizer(self):
        critic = self._critic()
        optimizer, scheduler = build_optimizer_and_scheduler({'learning_rate': 1e-3}, critic)
        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None

    def test_named_optimizer_string(self):
        critic = self._critic()
        optimizer, _ = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3, 'optimizer': 'sgd'}, critic)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_raises_with_helpful_message(self):
        critic = self._critic()
        with pytest.raises(ValueError, match="torch.optim.Optimizer subclass"):
            build_optimizer_and_scheduler({'learning_rate': 1e-3, 'optimizer': 'nonexistent'}, critic)

    def test_missing_learning_rate_raises_keyerror(self):
        """Strict access: must not silently substitute a different default
        than BASE_PARAMS_SCHEMA's (the drift this extraction reconciled)."""
        critic = self._critic()
        with pytest.raises(KeyError):
            build_optimizer_and_scheduler({}, critic)

    def test_cosine_scheduler(self):
        critic = self._critic()
        _, scheduler = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3, 'n_epochs': 20, 'scheduler': 'cosine'}, critic)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_unknown_scheduler_raises_with_helpful_message(self):
        critic = self._critic()
        with pytest.raises(ValueError, match="torch.optim.lr_scheduler class"):
            build_optimizer_and_scheduler(
                {'learning_rate': 1e-3, 'n_epochs': 20, 'scheduler': 'nonexistent'}, critic)

    def test_decoder_params_included_in_optimizer(self):
        """Decoder parameters (task.py's use case) must be part of the
        optimized parameter set when provided."""
        import torch.nn as nn
        critic = self._critic()
        decoder = nn.Linear(4, 4)
        optimizer, _ = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3}, critic, decoder_x=decoder)
        optimized_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert all(id(p) in optimized_ids for p in decoder.parameters())

    def test_no_decoders_by_default(self):
        """precision.py's use case: omitting decoder_x/decoder_y must work
        (they default to None) and not error."""
        critic = self._critic()
        optimizer, _ = build_optimizer_and_scheduler({'learning_rate': 1e-3}, critic)
        n_critic_params = sum(1 for _ in critic.parameters())
        n_optimized = sum(len(g['params']) for g in optimizer.param_groups)
        assert n_optimized == n_critic_params

    def test_lr_head_multiplier_splits_param_groups(self):
        """hybrid critic + lr_head_multiplier must produce two param groups
        with different learning rates."""
        params = {**DUMMY_EMBEDDING_PARAMS}
        critic = build_critic('hybrid', params)
        optimizer, _ = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3, 'lr_head_multiplier': 5.0}, critic)
        assert len(optimizer.param_groups) == 2
        lrs = sorted(g['lr'] for g in optimizer.param_groups)
        assert lrs[0] == pytest.approx(1e-3)
        assert lrs[1] == pytest.approx(5e-3)


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
    assert np.isclose(metrics['pr_eig'],1.0)
    assert np.isclose(metrics['effective_rank'], 1.0)

def test_spectral_metrics_uniform_dimensions():
    """Test metrics when all dimensions are utilized equally."""
    # A spectrum where energy is perfectly distributed across 4 dimensions
    spectrum = np.array([2.0, 2.0, 2.0, 2.0])
    metrics = compute_spectral_metrics(spectrum)
    
    assert np.isclose(metrics['pr_singular'], 4.0)
    assert np.isclose(metrics['pr_eig'],4.0)
    assert np.isclose(metrics['effective_rank'], 4.0)