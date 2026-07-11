# neural_mi/augmentations.py
"""
Online data augmentations applied per-batch during training.

All functions operate on PyTorch tensors and have no external dependencies
beyond PyTorch itself.

Main entry point
----------------
apply_augmentations(x, aug_params) -> torch.Tensor

aug_params keys
---------------
Spatial (4-D input only — warns and skips for lower-dimensional input):
  random_flip_h        : True | {'prob': float}   — flip along height axis
  random_flip_v        : True | {'prob': float}   — flip along width axis
  random_rotation_90   : True                     — rotate by 0/90/180/270°
  random_crop          : {'padding': int}         — pad-then-crop
  random_erase         : {'prob': float, 'scale': (min, max)}
  time_mask            : {'max_width': int}       — mask random column range
  freq_mask            : {'max_height': int}      — mask random row range
  gaussian_blur        : {'kernel_size': int, 'sigma': float}

Non-spatial (any ndim):
  gaussian_noise       : {'std': float}
  intensity_scale      : {'lo': float, 'hi': float}
  channel_dropout      : {'p': float}

Custom (any ndim):
  custom               : callable | list[callable]
                         Each callable receives a (N, ...) tensor and must
                         return a tensor of the same shape.

Application order: spatial → non-spatial → custom.
"""

import warnings
import torch
import torch.nn.functional as F
from typing import Any, Dict

# All keys that require 4-D input
_SPATIAL_KEYS = frozenset({
    'random_flip_h', 'random_flip_v', 'random_rotation_90',
    'random_crop', 'random_erase', 'time_mask', 'freq_mask', 'gaussian_blur',
})


def apply_augmentations(x: torch.Tensor, aug_params: Dict[str, Any]) -> torch.Tensor:
    """Apply augmentations defined in *aug_params* to batch tensor *x*.

    Parameters
    ----------
    x : torch.Tensor
        Batch of shape ``(N, C, ...)`` — typically ``(N, C, W)`` or ``(N, C, H, W)``.
    aug_params : dict
        Augmentation specification.  See module docstring for valid keys.

    Returns
    -------
    torch.Tensor
        Augmented tensor, same shape as *x*.
    """
    if not aug_params:
        return x

    is_4d = (x.ndim == 4)

    # Warn once if spatial augmentations are requested on non-4-D input
    requested_spatial = _SPATIAL_KEYS & set(aug_params)
    if requested_spatial and not is_4d:
        warnings.warn(
            f"Spatial augmentations {sorted(requested_spatial)} require 4-D input "
            f"(N, C, H, W). Got {x.ndim}-D — spatial augmentations will be skipped.",
            UserWarning, stacklevel=3,
        )

    # --- Spatial augmentations (4-D only) ---
    if is_4d:
        if 'random_flip_h' in aug_params:
            cfg = aug_params['random_flip_h']
            prob = cfg.get('prob', 0.5) if isinstance(cfg, dict) else 0.5
            x = _random_flip(x, dim=2, prob=prob)

        if 'random_flip_v' in aug_params:
            cfg = aug_params['random_flip_v']
            prob = cfg.get('prob', 0.5) if isinstance(cfg, dict) else 0.5
            x = _random_flip(x, dim=3, prob=prob)

        if 'random_rotation_90' in aug_params:
            x = _random_rotation_90(x)

        if 'random_crop' in aug_params:
            cfg = aug_params['random_crop']
            padding = cfg.get('padding', 4) if isinstance(cfg, dict) else 4
            x = _random_crop(x, padding)

        if 'random_erase' in aug_params:
            cfg = aug_params['random_erase']
            if isinstance(cfg, dict):
                prob = cfg.get('prob', 0.5)
                scale = cfg.get('scale', (0.02, 0.33))
            else:
                prob, scale = 0.5, (0.02, 0.33)
            x = _random_erase(x, prob, scale)

        if 'time_mask' in aug_params:
            cfg = aug_params['time_mask']
            max_width = cfg.get('max_width', max(1, x.shape[3] // 4)) if isinstance(cfg, dict) else max(1, x.shape[3] // 4)
            x = _time_mask(x, max_width)

        if 'freq_mask' in aug_params:
            cfg = aug_params['freq_mask']
            max_height = cfg.get('max_height', max(1, x.shape[2] // 4)) if isinstance(cfg, dict) else max(1, x.shape[2] // 4)
            x = _freq_mask(x, max_height)

        if 'gaussian_blur' in aug_params:
            cfg = aug_params['gaussian_blur']
            if isinstance(cfg, dict):
                kernel_size = cfg.get('kernel_size', 3)
                sigma = cfg.get('sigma', 1.0)
            else:
                kernel_size, sigma = 3, 1.0
            x = _gaussian_blur(x, kernel_size, sigma)

    # --- Non-spatial augmentations (any ndim) ---
    if 'gaussian_noise' in aug_params:
        cfg = aug_params['gaussian_noise']
        std = cfg.get('std', 0.1) if isinstance(cfg, dict) else 0.1
        x = x + torch.randn_like(x) * std

    if 'intensity_scale' in aug_params:
        cfg = aug_params['intensity_scale']
        if isinstance(cfg, dict):
            lo, hi = cfg.get('lo', 0.8), cfg.get('hi', 1.2)
        else:
            lo, hi = 0.8, 1.2
        scale = torch.empty(
            x.shape[0], *([1] * (x.ndim - 1)), device=x.device, dtype=x.dtype
        ).uniform_(lo, hi)
        x = x * scale

    if 'channel_dropout' in aug_params:
        cfg = aug_params['channel_dropout']
        p = cfg.get('p', 0.1) if isinstance(cfg, dict) else 0.1
        x = _channel_dropout(x, p)

    # --- Custom augmentation(s) ---
    if 'custom' in aug_params:
        custom = aug_params['custom']
        if callable(custom):
            x = custom(x)
        elif isinstance(custom, (list, tuple)):
            for fn in custom:
                x = fn(x)
        else:
            raise ValueError(
                f"'custom' augmentation must be a callable or list of callables, "
                f"got {type(custom).__name__}."
            )

    return x


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _random_flip(x: torch.Tensor, dim: int, prob: float = 0.5) -> torch.Tensor:
    """Flip each sample independently along *dim* with probability *prob*."""
    mask = torch.rand(x.shape[0], device=x.device) < prob
    if not mask.any():
        return x
    result = x.clone()
    result[mask] = torch.flip(x[mask], [dim])
    return result


def _random_rotation_90(x: torch.Tensor) -> torch.Tensor:
    """Rotate each sample by 0 / 90 / 180 / 270 degrees independently."""
    ks = torch.randint(0, 4, (x.shape[0],))
    result = x.clone()
    for k in range(1, 4):  # k=0 is identity
        mask = (ks == k)
        if mask.any():
            result[mask] = torch.rot90(x[mask], k=k, dims=[2, 3])
    return result


def _random_crop(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Pad by *padding* pixels with reflect mode, then random-crop back to original size."""
    N, C, H, W = x.shape
    padded = F.pad(x, [padding] * 4, mode='reflect')
    tops  = torch.randint(0, 2 * padding + 1, (N,))
    lefts = torch.randint(0, 2 * padding + 1, (N,))
    result = torch.empty_like(x)
    for i in range(N):
        result[i] = padded[i, :, tops[i]:tops[i] + H, lefts[i]:lefts[i] + W]
    return result


def _random_erase(x: torch.Tensor, prob: float, scale: tuple) -> torch.Tensor:
    """Zero a random rectangle in each sample with probability *prob*."""
    N, C, H, W = x.shape
    result = x.clone()
    min_area, max_area = scale[0] * H * W, scale[1] * H * W
    for i in range(N):
        if torch.rand(1).item() >= prob:
            continue
        for _ in range(10):  # up to 10 attempts to find valid dimensions
            area   = torch.empty(1).uniform_(min_area, max_area).item()
            aspect = torch.empty(1).uniform_(0.3, 3.3).item()
            eh = max(1, int(round((area * aspect) ** 0.5)))
            ew = max(1, int(round((area / aspect) ** 0.5)))
            if eh <= H and ew <= W:
                top  = torch.randint(0, H - eh + 1, (1,)).item()
                left = torch.randint(0, W - ew + 1, (1,)).item()
                result[i, :, top:top + eh, left:left + ew] = 0.0
                break
    return result


def _time_mask(x: torch.Tensor, max_width: int) -> torch.Tensor:
    """Zero a random contiguous column band of width up to *max_width* per sample."""
    N, C, H, W = x.shape
    result = x.clone()
    for i in range(N):
        w = torch.randint(0, max_width + 1, (1,)).item()
        if w > 0:
            start = torch.randint(0, max(1, W - w + 1), (1,)).item()
            result[i, :, :, start:start + w] = 0.0
    return result


def _freq_mask(x: torch.Tensor, max_height: int) -> torch.Tensor:
    """Zero a random contiguous row band of height up to *max_height* per sample."""
    N, C, H, W = x.shape
    result = x.clone()
    for i in range(N):
        h = torch.randint(0, max_height + 1, (1,)).item()
        if h > 0:
            start = torch.randint(0, max(1, H - h + 1), (1,)).item()
            result[i, :, start:start + h, :] = 0.0
    return result


def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply depthwise 2-D Gaussian blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd for symmetric padding
    C = x.shape[1]
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1))          # (k, k)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)         # (1, 1, k, k)
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)  # depthwise
    return F.conv2d(x, kernel_2d, padding=kernel_size // 2, groups=C)


def _channel_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    """Zero each channel independently with probability *p*."""
    N, C = x.shape[0], x.shape[1]
    mask = (torch.rand(N, C, device=x.device) > p).to(x.dtype)
    mask = mask.view(N, C, *([1] * (x.ndim - 2)))
    return x * mask
