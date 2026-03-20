# neural_mi/analysis/conditional.py
"""Implements conditional mutual information (CMI) estimation.

CMI between X and Y given Z is computed by the chain-rule difference:
    I(X; Y | Z) = I(X, Z; Y) - I(Z; Y)

Both terms are estimated independently using ``ParameterSweep``, so the
existing MI machinery (estimators, critics, augmentation) is reused verbatim.
The conditioning variable Z is concatenated with X at the data level before
any windowing or embedding.
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from neural_mi.analysis.sweep import ParameterSweep
from neural_mi.logger import logger


def run_conditional_mi(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    z_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Estimates conditional mutual information I(X; Y | Z).

    Uses the chain-rule identity:
        I(X; Y | Z) = I(XZ; Y) - I(Z; Y)

    Both component MI values are estimated via ``ParameterSweep`` with the
    same ``base_params``, so all hyperparameters (estimator, critic, embedding,
    training schedule) are shared.

    Parameters
    ----------
    x_data : torch.Tensor
        Data for variable X, shape ``(n_samples, n_channels_x, window_size)``.
    y_data : torch.Tensor
        Data for variable Y, shape ``(n_samples, n_channels_y, window_size)``.
    z_data : torch.Tensor
        Conditioning variable Z, shape ``(n_samples, n_channels_z, window_size)``.
        Must share the same sample dimension as x_data and y_data.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. Passed to both sweep runs.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid, e.g. ``{'run_id': range(5)}``.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - ``'cmi_estimate'`` : float — point estimate of I(X;Y|Z).
        - ``'mi_xz_y'`` : float — mean test MI for I(XZ; Y).
        - ``'mi_z_y'`` : float — mean test MI for I(Z; Y).
        - ``'raw_xz_y'`` : list of result dicts from the XZ→Y sweep.
        - ``'raw_z_y'`` : list of result dicts from the Z→Y sweep.
    """
    # Normalise all inputs to the same ndim before shape comparison and cat.
    # StaticDataset delivers (N, C, 1) tensors, but z_data is often passed as
    # raw 2-D (N, C) when no z_processor_type is given.  Unsqueeze the missing
    # trailing window dimension so that torch.cat works on a consistent axis.
    def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1) if t.ndim == 2 else t

    x_data = _ensure_3d(x_data)
    y_data = _ensure_3d(y_data)
    z_data = _ensure_3d(z_data)

    # Ensure all inputs are on the same device (x_data is the reference).
    # z_data in particular may arrive as a raw CPU tensor when no
    # z_processor_type is given, while x_data/y_data may be on MPS/CUDA.
    device = x_data.device
    y_data = y_data.to(device)
    z_data = z_data.to(device)

    if x_data.shape[0] != y_data.shape[0] or x_data.shape[0] != z_data.shape[0]:
        raise ValueError(
            "x_data, y_data, and z_data must have the same number of samples. "
            f"Got shapes {tuple(x_data.shape)}, {tuple(y_data.shape)}, {tuple(z_data.shape)}."
        )

    # Build XZ by concatenating along the channel dimension (dim=1)
    xz_data = torch.cat([x_data, z_data], dim=1)

    logger.info("Conditional MI: estimating I(XZ; Y)...")
    sweep_xz_y = ParameterSweep(x_data=xz_data, y_data=y_data, base_params=base_params.copy())
    results_xz_y = sweep_xz_y.run(
        sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False
    )

    logger.info("Conditional MI: estimating I(Z; Y)...")
    sweep_z_y = ParameterSweep(x_data=z_data, y_data=y_data, base_params=base_params.copy())
    results_z_y = sweep_z_y.run(
        sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False
    )

    vals_xz_y = [r['train_mi'] for r in results_xz_y if 'train_mi' in r]
    vals_z_y = [r['train_mi'] for r in results_z_y if 'train_mi' in r]
    if not vals_xz_y:
        raise RuntimeError("Conditional MI: all I(XZ;Y) runs failed — no valid train_mi values.")
    if not vals_z_y:
        raise RuntimeError("Conditional MI: all I(Z;Y) runs failed — no valid train_mi values.")
    mi_xz_y = float(np.mean(vals_xz_y))
    mi_z_y = float(np.mean(vals_z_y))
    cmi = mi_xz_y - mi_z_y

    logger.info(
        f"Conditional MI: I(XZ;Y)={mi_xz_y:.4f}, I(Z;Y)={mi_z_y:.4f}, "
        f"I(X;Y|Z)={cmi:.4f} nats (converted to requested output_units by the caller)"
    )

    return {
        'cmi_estimate': cmi,
        'mi_xz_y': mi_xz_y,
        'mi_z_y': mi_z_y,
        'raw_xz_y': results_xz_y,
        'raw_z_y': results_z_y,
    }
