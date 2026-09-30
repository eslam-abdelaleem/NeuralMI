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
from typing import Dict, Any, Optional

from neural_mi.analysis.sweep import (combined_spread, _joint_marginal_difference,
                                      amplification_factor)
from neural_mi.logger import logger
from neural_mi.embeddings_io import saved_paths
from neural_mi.utils import validate_stride


def _build_te_arrays(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    history_window: int,
    prediction_horizon: int = 1,
    stride: int = 1,
) -> tuple:
    """Build (x_past, y_past, y_future) sliding-window arrays.

    Parameters
    ----------
    x_data : torch.Tensor
        Shape ``(T, n_channels_x)``: raw time series for X.
    y_data : torch.Tensor
        Shape ``(T, n_channels_y)``: raw time series for Y.
    history_window : int
        Number of past time steps to include in each past window.
    prediction_horizon : int, optional
        How many steps ahead to predict. Defaults to 1.
    stride : int, optional
        Distance in samples between consecutive rows. Defaults to 1, keeping every valid starting position so that consecutive rows share
        ``history_window - 1`` of their samples. A larger stride thins the rows
        without changing which quantity is measured.

    Returns
    -------
    tuple of (x_past, y_past, y_future), each a torch.Tensor of shape
    ``(n_valid, n_channels, history_window)`` or
    ``(n_valid, n_channels, prediction_horizon)``.
    """
    # Accept numpy arrays and convert to tensors

    stride = validate_stride(stride, 'transfer entropy')
    from neural_mi.analysis.offsets import as_rows, slice_lags
    x_rows, y_rows = as_rows(x_data), as_rows(y_data)
    T = x_rows.shape[0]
    # n_positions: the number of valid starting positions i such that
    #   history window [i, i+H) and future [i+H, i+H+h) both fit within [0, T).
    # Largest valid i = T - H - h  →  count = T - H - h + 1.
    n_positions = T - history_window - prediction_horizon + 1
    if n_positions <= 0:
        raise ValueError(
            f"Not enough time points to build transfer entropy arrays. "
            f"Need > history_window + prediction_horizon = "
            f"{history_window + prediction_horizon}, got T={T}."
        )
    # Only every stride-th position is kept, so the row count is how many of
    # those steps fit inside n_positions.
    n_valid = (n_positions - 1) // stride + 1

    # Build sliding windows via unfold (a view, not a copy) instead of a
    # Python list comprehension + torch.stack, which would materialize three
    # large intermediate window arrays. unfold(0, size, 1) on a (T, C) tensor
    # already produces the (n_windows, C, size) layout directly, so no permute
    # is needed either.
    # All three slice at the same stride from the same reference positions, so
    # row i of each refers to the same time point. Truncating all three to
    # n_valid keeps that true when the strided tail lands unevenly.
    x_past = slice_lags(x_rows, history_window, stride, n_valid)
    y_past = slice_lags(y_rows, history_window, stride, n_valid)
    y_future = slice_lags(y_rows[history_window:], prediction_horizon, stride, n_valid)

    return x_past, y_past, y_future


def _build_w_past(w_data: torch.Tensor, history_window: int, n_valid: int,
                  stride: int = 1) -> torch.Tensor:
    """Build W_past, matching X_past/Y_past's construction exactly.

    Same ``history_window``, same ``stride``, truncated to the same ``n_valid``
    count, so it aligns sample-for-sample with the other arrays. The stride has
    to be passed in instead of inferred, since ``n_valid`` alone does not fix
    which positions were kept once the rows can be thinned.
    """
    from neural_mi.analysis.offsets import as_rows, slice_lags
    return slice_lags(as_rows(w_data), history_window, stride, n_valid)


def run_transfer_entropy(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    base_params: Dict[str, Any],
    history_window: int,
    prediction_horizon: int = 1,
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    bidirectional: bool = False,
    w_data: Optional[torch.Tensor] = None,
    stride: int = 1,
) -> Dict[str, Any]:
    """Estimates transfer entropy TE(X→Y), and optionally TE(Y→X).

    Uses the chain-rule identity:
        TE(X→Y) = I(x_past, y_past ; y_future) - I(y_past ; y_future)

    Both component MI values are estimated via ``ParameterSweep``.

    Parameters
    ----------
    x_data : torch.Tensor
        Raw time-series data for X, shape ``(T, n_channels_x)``.
        2-D (no windowing dimension yet), windows are built internally.
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
    w_data : torch.Tensor, optional
        Raw time-series data for a third conditioning signal W, shape
        ``(T, n_channels_w)``. When provided, computes *conditional* transfer
        entropy TE(X→Y|W) = I(y_future; x_past | y_past, w_past) instead of
        plain TE(X→Y). W_past (built the same way as X_past/Y_past, same
        ``history_window``) is folded into both the joint and marginal
        conditioning arrays. ``None`` (the default) gives plain TE. Applied to
        both directions when ``bidirectional=True``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``'te_xy'`` (float): point estimate of TE(X→Y).
        - ``'te_estimate'`` (float): alias for ``te_xy``.
        - ``'i_xypast_yfuture'`` (float): mean I(x_past, y_past ; y_future).
        - ``'i_ypast_yfuture'`` (float): mean I(y_past ; y_future).
        - ``'amplification_factor'`` (float): error-amplification factor for
          TE(X→Y), ``(|I(xy_past;y_future)| + |I(y_past;y_future)|) / |TE|``.
          Transfer entropy is the most fragile quantity in the taxonomy on this
          measure; a value >= 10 means the estimate is a small residual of two
          much larger numbers.  ``'amplification_factor_yx'`` is the same for
          TE(Y→X) when ``bidirectional=True``.  See
          :func:`neural_mi.analysis.sweep.amplification_factor`.
        - ``'raw_xypast_yfuture'`` : list of result dicts.
        - ``'raw_ypast_yfuture'`` : list of result dicts.
        - ``'n_samples'`` (int): number of valid sliding-window samples.
        - ``'bidirectional'`` (bool): whether bidirectional TE was computed.

        If ``bidirectional=True``, additionally:

        - ``'te_yx'`` (float): point estimate of TE(Y→X).
        - ``'i_yxpast_xfuture'`` (float): mean I(y_past, x_past ; x_future).
        - ``'i_xpast_xfuture'`` (float): mean I(x_past ; x_future).
        - ``'raw_yxpast_xfuture'`` : list of result dicts.
        - ``'raw_xpast_xfuture'`` : list of result dicts.
        - ``'directionality_index'`` (float): (TE_xy - TE_yx) / (|TE_xy| + |TE_yx|).
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
    # _build_te_arrays windows via unfold(0, history_window, stride), bypassing
    # WindowManager entirely, so the blocked-split leakage check (same mechanism
    # as the WindowManager path; see run.py/trainer.py) needs its window geometry
    # passed explicitly here instead. The step has to be the real stride: at
    # stride 1 a gap of one window index buys one sample of separation, so the
    # check would otherwise wave through a split that shares almost a whole
    # window. Set once and reused by every _joint_marginal_difference call below
    # (joint/marginal, both directions if bidirectional), since neither
    # history_window nor the stride changes between them.
    base_params = dict(base_params)
    base_params['leak_check_window_size'] = history_window
    base_params['leak_check_step'] = stride
    x_past, y_past, y_future = _build_te_arrays(
        x_data, y_data, history_window, prediction_horizon, stride=stride
    )
    n_samples = x_past.shape[0]
    logger.info(f"Transfer Entropy: {n_samples} valid samples.")

    # Joint past: concatenate x_past and y_past along channel dim
    xy_past = torch.cat([x_past, y_past], dim=1)
    y_past_cond = y_past
    if w_data is not None:
        w_past = _build_w_past(w_data, history_window, n_samples, stride=stride)
        xy_past = torch.cat([xy_past, w_past], dim=1)
        y_past_cond = torch.cat([y_past, w_past], dim=1)

    te, mi_joint, mi_marginal, results_joint, results_marginal, _per_run = _joint_marginal_difference(
        xy_past, y_future, y_past_cond, y_future,
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
        'amplification_factor': amplification_factor([mi_joint, mi_marginal], te),
        'mi_estimate_std': combined_spread(_per_run, (1, -1)),
        'raw_xypast_yfuture': results_joint,
        'raw_ypast_yfuture': results_marginal,
        'n_samples': n_samples,
        'bidirectional': bidirectional,
    }

    if bidirectional:
        logger.info("Transfer Entropy (bidirectional): estimating TE(Y→X)...")
        # Swap roles of X and Y to get TE(Y→X)
        y_past_b, x_past_b, x_future = _build_te_arrays(
            y_data, x_data, history_window, prediction_horizon, stride=stride
        )
        yx_past = torch.cat([y_past_b, x_past_b], dim=1)
        x_past_cond = x_past_b
        if w_data is not None:
            # Reuse the same w_past computed above -- same history_window,
            # same n_samples, so it aligns with this direction's arrays too.
            yx_past = torch.cat([yx_past, w_past], dim=1)
            x_past_cond = torch.cat([x_past_b, w_past], dim=1)

        te_yx, mi_joint_yx, mi_marginal_yx, results_joint_yx, results_marginal_yx, _per_run_yx = _joint_marginal_difference(
            yx_past, x_future, x_past_cond, x_future,
            base_params, sweep_grid, n_workers,
            quantity_name="TE(Y→X)",
            joint_label="yx_past;x_future", marginal_label="x_past;x_future",
            joint_key="i_yxpast_xfuture", marginal_key="i_xpast_xfuture", raw_key='te_yx_raw',
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
            'amplification_factor_yx': amplification_factor(
                [mi_joint_yx, mi_marginal_yx], te_yx),
            'raw_yxpast_xfuture': results_joint_yx,
            'raw_xpast_xfuture': results_marginal_yx,
            'directionality_index': directionality_index,
        })

    return result


def _te_rigorous_scalar(x_s, y_s, bp, sweep_grid=None, history_window=None,
                        prediction_horizon=1, bidirectional=False, w_data=None,
                        stride=1) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of TE.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1``, must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching the
    outer-loop-gets-workers / inner-loop-sequential convention used for
    dimensionality-mode splits.

    ``w_data`` arrives here already sliced to this gamma-chunk's samples (via
    ``run_rigorous_scalar_analysis``'s ``extra_data`` mechanism, the same one
    ``mode='conditional'``'s rigorous path also uses for its own ``w_data``),
    not the full signal, forwarded straight through to ``run_transfer_entropy``.
    """
    raw = run_transfer_entropy(
        x_s, y_s, bp,
        history_window=history_window,
        prediction_horizon=prediction_horizon,
        sweep_grid=sweep_grid,
        n_workers=1,
        bidirectional=bidirectional,
        w_data=w_data,
        stride=stride,
    )
    paths = saved_paths(raw)
    return (raw['te_estimate'], paths) if paths else raw['te_estimate']
