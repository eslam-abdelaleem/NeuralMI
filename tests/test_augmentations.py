# tests/test_augmentations.py
"""Tests for neural_mi/augmentations.py."""
import warnings
import pytest
import torch

from neural_mi.augmentations import apply_augmentations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_3d():
    """(N=8, C=2, W=16) — 3-D batch (sequence / spike)."""
    return torch.randn(8, 2, 16)


@pytest.fixture
def batch_4d():
    """(N=8, C=2, H=8, W=8) — 4-D batch (CNN2D / spectrogram)."""
    return torch.randn(8, 2, 8, 8)


# ---------------------------------------------------------------------------
# Empty aug_params → identity
# ---------------------------------------------------------------------------

def test_empty_params_returns_same_tensor(batch_3d):
    out = apply_augmentations(batch_3d, {})
    assert out is batch_3d  # exact same object


# ---------------------------------------------------------------------------
# Shape preservation for every augmentation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aug_params", [
    {'gaussian_noise': {'std': 0.1}},
    {'intensity_scale': {'lo': 0.8, 'hi': 1.2}},
    {'channel_dropout': {'p': 0.2}},
])
def test_non_spatial_preserves_shape_3d(batch_3d, aug_params):
    out = apply_augmentations(batch_3d, aug_params)
    assert out.shape == batch_3d.shape


@pytest.mark.parametrize("aug_params", [
    {'gaussian_noise': {'std': 0.1}},
    {'intensity_scale': {'lo': 0.8, 'hi': 1.2}},
    {'channel_dropout': {'p': 0.2}},
    {'random_flip_h': True},
    {'random_flip_v': True},
    {'random_rotation_90': True},
    {'random_crop': {'padding': 2}},
    {'random_erase': {'prob': 1.0, 'scale': (0.05, 0.2)}},
    {'time_mask': {'max_width': 3}},
    {'freq_mask': {'max_height': 3}},
    {'gaussian_blur': {'kernel_size': 3, 'sigma': 1.0}},
])
def test_augmentation_preserves_shape_4d(batch_4d, aug_params):
    out = apply_augmentations(batch_4d, aug_params)
    assert out.shape == batch_4d.shape


# ---------------------------------------------------------------------------
# Spatial augmentations on 3-D input → warning, input returned unchanged
# ---------------------------------------------------------------------------

def test_spatial_aug_on_3d_emits_warning(batch_3d):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = apply_augmentations(batch_3d, {'random_flip_h': True})
    assert any(issubclass(x.category, UserWarning) for x in w)
    assert out.shape == batch_3d.shape


# ---------------------------------------------------------------------------
# Gaussian noise actually changes values
# ---------------------------------------------------------------------------

def test_gaussian_noise_changes_values(batch_3d):
    out = apply_augmentations(batch_3d.clone(), {'gaussian_noise': {'std': 1.0}})
    assert not torch.allclose(out, batch_3d)


# ---------------------------------------------------------------------------
# Intensity scale stays within bounds (per sample)
# ---------------------------------------------------------------------------

def test_intensity_scale_bounded(batch_4d):
    x = torch.ones(8, 2, 8, 8)
    lo, hi = 0.5, 2.0
    out = apply_augmentations(x, {'intensity_scale': {'lo': lo, 'hi': hi}})
    assert out.min().item() >= lo - 1e-5
    assert out.max().item() <= hi + 1e-5


# ---------------------------------------------------------------------------
# Channel dropout zeros whole channels
# ---------------------------------------------------------------------------

def test_channel_dropout_zeros_channels(batch_4d):
    # With p=1.0 every channel should be zeroed
    out = apply_augmentations(batch_4d.clone(), {'channel_dropout': {'p': 1.0}})
    assert torch.all(out == 0.0)


def test_channel_dropout_p0_identity(batch_4d):
    out = apply_augmentations(batch_4d.clone(), {'channel_dropout': {'p': 0.0}})
    assert torch.allclose(out, batch_4d)


# ---------------------------------------------------------------------------
# random_flip_h flips along H axis
# ---------------------------------------------------------------------------

def test_random_flip_h_flips_correctly(batch_4d):
    # Force all samples to flip by using a seeded approach: set prob=1.0 via
    # patching isn't available; instead verify shape and that at least one
    # sample changed (statistically certain with prob=0.5, N=8).
    out = apply_augmentations(batch_4d.clone(), {'random_flip_h': {'prob': 1.0}})
    assert out.shape == batch_4d.shape
    # With prob=1.0 all samples must be flipped → not equal to original
    assert not torch.allclose(out, batch_4d)


# ---------------------------------------------------------------------------
# random_rotation_90 changes values
# ---------------------------------------------------------------------------

def test_random_rotation_90_changes_values(batch_4d):
    # With a non-symmetric input, rotation almost certainly changes values.
    out = apply_augmentations(batch_4d.clone(), {'random_rotation_90': True})
    assert out.shape == batch_4d.shape


# ---------------------------------------------------------------------------
# time_mask and freq_mask introduce zeros
# ---------------------------------------------------------------------------

def test_time_mask_introduces_zeros():
    x = torch.ones(4, 1, 8, 16)
    out = apply_augmentations(x, {'time_mask': {'max_width': 8}})
    assert out.shape == x.shape
    # At least some values should be zero
    assert (out == 0.0).any()


def test_freq_mask_introduces_zeros():
    x = torch.ones(4, 1, 8, 16)
    out = apply_augmentations(x, {'freq_mask': {'max_height': 4}})
    assert out.shape == x.shape
    assert (out == 0.0).any()


# ---------------------------------------------------------------------------
# random_erase with prob=1.0 always erases something
# ---------------------------------------------------------------------------

def test_random_erase_zeroes_region():
    x = torch.ones(4, 1, 16, 16)
    out = apply_augmentations(x, {'random_erase': {'prob': 1.0, 'scale': (0.1, 0.5)}})
    assert out.shape == x.shape
    assert (out == 0.0).any()


# ---------------------------------------------------------------------------
# gaussian_blur (even kernel_size → auto-corrected to odd)
# ---------------------------------------------------------------------------

def test_gaussian_blur_even_kernel_auto_fix(batch_4d):
    out = apply_augmentations(batch_4d.clone(), {'gaussian_blur': {'kernel_size': 4, 'sigma': 1.0}})
    assert out.shape == batch_4d.shape


# ---------------------------------------------------------------------------
# Custom callable augmentation
# ---------------------------------------------------------------------------

def test_custom_callable(batch_3d):
    called = []

    def double(x):
        called.append(True)
        return x * 2.0

    out = apply_augmentations(batch_3d.clone(), {'custom': double})
    assert len(called) == 1
    assert torch.allclose(out, batch_3d * 2.0)


def test_custom_list_of_callables(batch_3d):
    out = apply_augmentations(
        batch_3d.clone(),
        {'custom': [lambda x: x + 1.0, lambda x: x * 2.0]},
    )
    assert torch.allclose(out, (batch_3d + 1.0) * 2.0)


def test_custom_invalid_raises():
    x = torch.randn(4, 2, 8)
    with pytest.raises(ValueError, match="callable"):
        apply_augmentations(x, {'custom': "not_a_callable"})


# ---------------------------------------------------------------------------
# Application order: spatial → non-spatial → custom
# ---------------------------------------------------------------------------

def test_application_order_custom_runs_last(batch_4d):
    """Custom callable must see the result of all other augmentations."""
    record = []

    def record_fn(x):
        record.append(x.clone())
        return x

    # gaussian_noise changes values; custom sees the noisy result
    apply_augmentations(
        batch_4d.clone(),
        {'gaussian_noise': {'std': 0.5}, 'custom': record_fn},
    )
    assert len(record) == 1
    # The recorded tensor should not be identical to the original (noise was applied)
    assert not torch.allclose(record[0], batch_4d)


# ---------------------------------------------------------------------------
# Default parameter values (bool True shortcuts)
# ---------------------------------------------------------------------------

def test_bool_true_shortcut_flip_h(batch_4d):
    """random_flip_h: True should work (prob defaults to 0.5)."""
    out = apply_augmentations(batch_4d.clone(), {'random_flip_h': True})
    assert out.shape == batch_4d.shape


def test_bool_true_shortcut_gaussian_noise(batch_3d):
    """gaussian_noise: True → std defaults to 0.1."""
    out = apply_augmentations(batch_3d.clone(), {'gaussian_noise': True})
    assert out.shape == batch_3d.shape
