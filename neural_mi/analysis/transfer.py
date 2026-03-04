# neural_mi/analysis/transfer.py
"""Implements transfer entropy (TE) estimation.

Transfer entropy from X to Y is the conditional MI of Y's future given its
joint past with X, over Y's past alone:

    TE(X→Y) = I(y_future ; x_past | y_past)
             = I(x_past, y_past ; y_future) - I(y_past ; y_future)

Both component MI values are estimated with ``ParameterSweep``.
The past/future arrays are built internally from the raw time series using
sliding windows controlled by ``history_window`` and ``prediction_horizon``.
"""
import torch
import numpy as np
from typing import Dict, Any, Optional

from neural_mi.analysis.sweep import ParameterSweep
from neural_mi.logger import logger


def _build_te_arrays(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    history_window: int,
    prediction_horizon: int = 1,
) -> tuple:
    """Build (x_past, y_past, y_future) sliding-window arrays.

    Parameters
    ----------
    x_data : torch.Tensor
        Shape ``(T, n_channels_x)`` — raw time series for X.
    y_data : torch.Tensor
        Shape ``(T, n_channels_y)`` — raw time series for Y.
    history_window : int
        Number of past time steps to include in each past window.
    prediction_horizon : int, optional
        How many steps ahead to predict. Defaults to 1.

    Returns
    -------
    tuple of (x_past, y_past, y_future), each a torch.Tensor of shape
    ``(n_valid, n_channels, history_window)`` or
    ``(n_valid, n_channels, prediction_horizon)``.
    """
    # Accept numpy arrays and convert to tensors
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.as_tensor(np.asarray(x_data), dtype=torch.float32)
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.as_tensor(np.asarray(y_data), dtype=torch.float32)

    T = x_data.shape[0]
    # n_valid: the number of valid starting positions i such that
    #   history window [i, i+H) and future [i+H, i+H+h) both fit within [0, T).
    # Largest valid i = T - H - h  →  count = T - H - h + 1.
    n_valid = T - history_window - prediction_horizon + 1
    if n_valid <= 0:
        raise ValueError(
            f"Not enough time points to build transfer entropy arrays. "
            f"Need > history_window + prediction_horizon = "
            f"{history_window + prediction_horizon}, got T={T}."
        )

    # Build sliding windows: shape (n_valid, n_channels, history_window)
    x_past = torch.stack(
        [x_data[i: i + history_window] for i in range(n_valid)], dim=0
    ).permute(0, 2, 1)  # (n_valid, n_channels_x, history_window)

    y_past = torch.stack(
        [y_data[i: i + history_window] for i in range(n_valid)], dim=0
    ).permute(0, 2, 1)  # (n_valid, n_channels_y, history_window)

    y_future = torch.stack(
        [y_data[i + history_window: i + history_window + prediction_horizon]
         for i in range(n_valid)],
        dim=0
    ).permute(0, 2, 1)  # (n_valid, n_channels_y, prediction_horizon)

    return x_past, y_past, y_future


def run_transfer_entropy(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    base_params: Dict[str, Any],
    history_window: int,
    prediction_horizon: int = 1,
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Estimates transfer entropy TE(X→Y).

    Uses the chain-rule identity:
        TE(X→Y) = I(x_past, y_past ; y_future) - I(y_past ; y_future)

    Both component MI values are estimated via ``ParameterSweep``.

    Parameters
    ----------
    x_data : torch.Tensor
        Raw time-series data for X, shape ``(T, n_channels_x)``.
        2-D (no windowing dimension yet) — windows are built internally.
    y_data : torch.Tensor
        Raw time-series data for Y, shape ``(T, n_channels_y)``.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. ``embedding_model`` should be
        compatible with temporal data (e.g. 'cnn', 'gru', 'lstm', 'tcn').
    history_window : int
        Number of past samples to use as the history context.
    prediction_horizon : int, optional
        Number of future samples to predict. Defaults to 1.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid passed to both sweep runs.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - ``'te_estimate'`` : float — point estimate of TE(X→Y).
        - ``'i_xypast_yfuture'`` : float — mean I(x_past, y_past ; y_future).
        - ``'i_ypast_yfuture'`` : float — mean I(y_past ; y_future).
        - ``'raw_xypast_yfuture'`` : list of result dicts.
        - ``'raw_ypast_yfuture'`` : list of result dicts.
        - ``'n_samples'`` : int — number of valid sliding-window samples.
    """
    if x_data.ndim != 2 or y_data.ndim != 2:
        raise ValueError(
            "run_transfer_entropy expects 2-D inputs of shape (T, n_channels). "
            f"Got x_data.ndim={x_data.ndim}, y_data.ndim={y_data.ndim}."
        )
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError(
            "x_data and y_data must have the same number of time points. "
            f"Got {x_data.shape[0]} and {y_data.shape[0]}."
        )

    logger.info(
        f"Transfer Entropy: building windows "
        f"(history_window={history_window}, prediction_horizon={prediction_horizon})..."
    )
    x_past, y_past, y_future = _build_te_arrays(
        x_data, y_data, history_window, prediction_horizon
    )
    n_samples = x_past.shape[0]
    logger.info(f"Transfer Entropy: {n_samples} valid samples.")

    # Joint past: concatenate x_past and y_past along channel dim
    xy_past = torch.cat([x_past, y_past], dim=1)

    logger.info("Transfer Entropy: estimating I(x_past, y_past ; y_future)...")
    sweep_joint = ParameterSweep(
        x_data=xy_past, y_data=y_future, base_params=base_params.copy()
    )
    results_joint = sweep_joint.run(
        sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False
    )

    logger.info("Transfer Entropy: estimating I(y_past ; y_future)...")
    sweep_marginal = ParameterSweep(
        x_data=y_past, y_data=y_future, base_params=base_params.copy()
    )
    results_marginal = sweep_marginal.run(
        sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False
    )

    mi_joint = float(
        np.mean([r['test_mi'] for r in results_joint if 'test_mi' in r])
    )
    mi_marginal = float(
        np.mean([r['test_mi'] for r in results_marginal if 'test_mi' in r])
    )
    te = mi_joint - mi_marginal

    logger.info(
        f"Transfer Entropy: I(xy_past;y_future)={mi_joint:.4f}, "
        f"I(y_past;y_future)={mi_marginal:.4f}, TE={te:.4f}"
    )

    return {
        'te_estimate': te,
        'i_xypast_yfuture': mi_joint,   # I(x_past, y_past ; y_future)
        'i_ypast_yfuture': mi_marginal,  # I(y_past ; y_future)
        'raw_xypast_yfuture': results_joint,
        'raw_ypast_yfuture': results_marginal,
        'n_samples': n_samples,
    }
