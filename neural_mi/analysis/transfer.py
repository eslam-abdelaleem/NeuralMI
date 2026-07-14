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

from neural_mi.analysis.sweep import _joint_marginal_difference
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

    # Build sliding windows via unfold (a view, not a copy) instead of a
    # Python list comprehension + torch.stack, which would materialize three
    # large intermediate window arrays. unfold(0, size, 1) on a (T, C) tensor
    # already produces the (n_windows, C, size) layout directly, so no permute
    # is needed either.
    x_past = x_data.unfold(0, history_window, 1)[:n_valid]        # (n_valid, n_channels_x, history_window)
    y_past = y_data.unfold(0, history_window, 1)[:n_valid]        # (n_valid, n_channels_y, history_window)
    y_future = y_data[history_window:].unfold(0, prediction_horizon, 1)  # (n_valid, n_channels_y, prediction_horizon)

    return x_past, y_past, y_future


def run_transfer_entropy(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    base_params: Dict[str, Any],
    history_window: int,
    prediction_horizon: int = 1,
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    bidirectional: bool = False,
) -> Dict[str, Any]:
    """Estimates transfer entropy TE(X→Y), and optionally TE(Y→X).

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
    bidirectional : bool, optional
        If True, also compute TE(Y→X) and return a directionality index.
        Defaults to False.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``'te_xy'`` : float — point estimate of TE(X→Y).
        - ``'te_estimate'`` : float — alias for ``te_xy``.
        - ``'i_xypast_yfuture'`` : float — mean I(x_past, y_past ; y_future).
        - ``'i_ypast_yfuture'`` : float — mean I(y_past ; y_future).
        - ``'raw_xypast_yfuture'`` : list of result dicts.
        - ``'raw_ypast_yfuture'`` : list of result dicts.
        - ``'n_samples'`` : int — number of valid sliding-window samples.
        - ``'bidirectional'`` : bool — whether bidirectional TE was computed.

        If ``bidirectional=True``, additionally:

        - ``'te_yx'`` : float — point estimate of TE(Y→X).
        - ``'i_yxpast_xfuture'`` : float — mean I(y_past, x_past ; x_future).
        - ``'i_xpast_xfuture'`` : float — mean I(x_past ; x_future).
        - ``'raw_yxpast_xfuture'`` : list of result dicts.
        - ``'raw_xpast_xfuture'`` : list of result dicts.
        - ``'directionality_index'`` : float — (TE_xy - TE_yx) / (|TE_xy| + |TE_yx|).
          +1 = pure X→Y, -1 = pure Y→X, 0 = symmetric.
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

    if not bidirectional:
        logger.info(
            "Computing TE(X→Y) only. In coupled systems, consider also computing TE(Y→X) "
            "by swapping x_data and y_data and comparing both directions. Pass "
            "bidirectional=True to compute both directions automatically and obtain "
            "a directionality index."
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

    te, mi_joint, mi_marginal, results_joint, results_marginal = _joint_marginal_difference(
        xy_past, y_future, y_past, y_future,
        base_params, sweep_grid, n_workers,
        quantity_name="TE(X→Y)",
        joint_label="xy_past;y_future", marginal_label="y_past;y_future",
        joint_key="i_xypast_yfuture", marginal_key="i_ypast_yfuture",
    )

    result = {
        'te_xy': te,
        'te_estimate': te,
        'i_xypast_yfuture': mi_joint,
        'i_ypast_yfuture': mi_marginal,
        'raw_xypast_yfuture': results_joint,
        'raw_ypast_yfuture': results_marginal,
        'n_samples': n_samples,
        'bidirectional': bidirectional,
    }

    if bidirectional:
        logger.info("Transfer Entropy (bidirectional): estimating TE(Y→X)...")
        # Swap roles of X and Y to get TE(Y→X)
        y_past_b, x_past_b, x_future = _build_te_arrays(
            y_data, x_data, history_window, prediction_horizon
        )
        yx_past = torch.cat([y_past_b, x_past_b], dim=1)

        te_yx, mi_joint_yx, mi_marginal_yx, results_joint_yx, results_marginal_yx = _joint_marginal_difference(
            yx_past, x_future, x_past_b, x_future,
            base_params, sweep_grid, n_workers,
            quantity_name="TE(Y→X)",
            joint_label="yx_past;x_future", marginal_label="x_past;x_future",
            joint_key="i_yxpast_xfuture", marginal_key="i_xpast_xfuture",
        )

        # Directionality index: +1 = pure X→Y, -1 = pure Y→X, 0 = symmetric
        te_sum = abs(te) + abs(te_yx)
        directionality_index = (te - te_yx) / te_sum if te_sum > 1e-10 else 0.0

        logger.info(
            f"TE(X→Y)={te:.4f}, TE(Y→X)={te_yx:.4f}, "
            f"directionality_index={directionality_index:.4f}"
        )
        result.update({
            'te_yx': te_yx,
            'i_yxpast_xfuture': mi_joint_yx,
            'i_xpast_xfuture': mi_marginal_yx,
            'raw_yxpast_xfuture': results_joint_yx,
            'raw_xpast_xfuture': results_marginal_yx,
            'directionality_index': directionality_index,
        })

    return result
