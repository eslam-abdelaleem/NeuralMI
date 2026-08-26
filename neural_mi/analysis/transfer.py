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

from neural_mi.analysis.sweep import (_joint_marginal_difference, _extract_embeddings,
                                      amplification_factor)
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


def _build_w_past(w_data: torch.Tensor, history_window: int, n_valid: int) -> torch.Tensor:
    """Build W_past, matching X_past/Y_past's construction exactly (same
    ``history_window``, same stride-1 unfold, truncated to the same
    ``n_valid`` count so it aligns sample-for-sample with the other arrays)."""
    if not isinstance(w_data, torch.Tensor):
        w_data = torch.as_tensor(np.asarray(w_data), dtype=torch.float32)
    return w_data.unfold(0, history_window, 1)[:n_valid]


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
    w_data : torch.Tensor, optional
        Raw time-series data for a third conditioning signal W, shape
        ``(T, n_channels_w)``. When provided, computes *conditional* transfer
        entropy TE(X→Y|W) = I(y_future; x_past | y_past, w_past) instead of
        plain TE(X→Y). W_past (built the same way as X_past/Y_past, same
        ``history_window``) is folded into both the joint and marginal
        conditioning arrays. ``None`` (the default) reproduces plain TE
        exactly, unchanged from before this parameter existed. Applied to
        both directions when ``bidirectional=True``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``'te_xy'`` : float — point estimate of TE(X→Y).
        - ``'te_estimate'`` : float — alias for ``te_xy``.
        - ``'i_xypast_yfuture'`` : float — mean I(x_past, y_past ; y_future).
        - ``'i_ypast_yfuture'`` : float — mean I(y_past ; y_future).
        - ``'amplification_factor'`` : float — error-amplification factor for
          TE(X→Y), ``(|I(xy_past;y_future)| + |I(y_past;y_future)|) / |TE|``.
          Transfer entropy is the most fragile quantity in the taxonomy on this
          measure; a value >= 10 means the estimate is a small residual of two
          much larger numbers.  ``'amplification_factor_yx'`` is the same for
          TE(Y→X) when ``bidirectional=True``.  See
          :func:`neural_mi.analysis.sweep.amplification_factor`.
        - ``'raw_xypast_yfuture'`` : list of result dicts.
        - ``'raw_ypast_yfuture'`` : list of result dicts.
        - ``'n_samples'`` : int — number of valid sliding-window samples.
        - ``'bidirectional'`` : bool — whether bidirectional TE was computed.
        - ``'embeddings_x'``, ``'embeddings_y'`` : present only when
          ``base_params['return_embeddings']`` is set -- the joint
          (xy_past;y_future) leg's learned embeddings, not the marginal
          (y_past;y_future) leg's, which trains a separate model.

        If ``bidirectional=True``, additionally:

        - ``'te_yx'`` : float — point estimate of TE(Y→X).
        - ``'i_yxpast_xfuture'`` : float — mean I(y_past, x_past ; x_future).
        - ``'i_xpast_xfuture'`` : float — mean I(x_past ; x_future).
        - ``'raw_yxpast_xfuture'`` : list of result dicts.
        - ``'raw_xpast_xfuture'`` : list of result dicts.
        - ``'directionality_index'`` : float — (TE_xy - TE_yx) / (|TE_xy| + |TE_yx|).
          +1 = pure X→Y, -1 = pure Y→X, 0 = symmetric.
        - ``'embeddings_x_yx'``, ``'embeddings_y_yx'`` : the TE(Y→X)
          direction's joint-leg embeddings, present under the same condition.
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
    # _build_te_arrays windows via unfold(0, history_window, 1) -- stride 1,
    # bypassing WindowManager entirely -- so the blocked-split leakage check
    # (same mechanism as the WindowManager path; see run.py/trainer.py) needs
    # its window geometry passed explicitly here instead. Set once; reused by
    # every _joint_marginal_difference call below (joint/marginal, both
    # directions if bidirectional), since history_window and the stride-1
    # construction don't change between them.
    base_params = dict(base_params)
    base_params['leak_check_window_size'] = history_window
    base_params['leak_check_step'] = 1
    x_past, y_past, y_future = _build_te_arrays(
        x_data, y_data, history_window, prediction_horizon
    )
    n_samples = x_past.shape[0]
    logger.info(f"Transfer Entropy: {n_samples} valid samples.")

    # Joint past: concatenate x_past and y_past along channel dim
    xy_past = torch.cat([x_past, y_past], dim=1)
    y_past_cond = y_past
    if w_data is not None:
        w_past = _build_w_past(w_data, history_window, n_samples)
        xy_past = torch.cat([xy_past, w_past], dim=1)
        y_past_cond = torch.cat([y_past, w_past], dim=1)

    te, mi_joint, mi_marginal, results_joint, results_marginal = _joint_marginal_difference(
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
        'raw_xypast_yfuture': results_joint,
        'raw_ypast_yfuture': results_marginal,
        'n_samples': n_samples,
        'bidirectional': bidirectional,
    }
    # TE is a difference of two separately-trained estimates with separate
    # ceilings -- surface both components' diagnostics (Fix 6), not just one,
    # since a healthy joint estimate can still hide a saturated marginal one
    # (or vice versa) that the difference alone wouldn't reveal.
    result['diagnostics_joint'] = _extract_diagnostics(results_joint)
    result['diagnostics_marginal'] = _extract_diagnostics(results_marginal)
    result.update(_extract_embeddings(results_joint) or {})

    if bidirectional:
        logger.info("Transfer Entropy (bidirectional): estimating TE(Y→X)...")
        # Swap roles of X and Y to get TE(Y→X)
        y_past_b, x_past_b, x_future = _build_te_arrays(
            y_data, x_data, history_window, prediction_horizon
        )
        yx_past = torch.cat([y_past_b, x_past_b], dim=1)
        x_past_cond = x_past_b
        if w_data is not None:
            # Reuse the same w_past computed above -- same history_window,
            # same n_samples, so it aligns with this direction's arrays too.
            yx_past = torch.cat([yx_past, w_past], dim=1)
            x_past_cond = torch.cat([x_past_b, w_past], dim=1)

        te_yx, mi_joint_yx, mi_marginal_yx, results_joint_yx, results_marginal_yx = _joint_marginal_difference(
            yx_past, x_future, x_past_cond, x_future,
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
        _emb_yx = _extract_embeddings(results_joint_yx)
        result.update({
            'te_yx': te_yx,
            'i_yxpast_xfuture': mi_joint_yx,
            'i_xpast_xfuture': mi_marginal_yx,
            'amplification_factor_yx': amplification_factor(
                [mi_joint_yx, mi_marginal_yx], te_yx),
            'raw_yxpast_xfuture': results_joint_yx,
            'raw_xpast_xfuture': results_marginal_yx,
            'directionality_index': directionality_index,
            'diagnostics_joint_yx': _extract_diagnostics(results_joint_yx),
            'diagnostics_marginal_yx': _extract_diagnostics(results_marginal_yx),
        })
        if _emb_yx is not None:
            result['embeddings_x_yx'] = _emb_yx['embeddings_x']
            result['embeddings_y_yx'] = _emb_yx['embeddings_y']

    return result


_DIAGNOSTIC_KEYS = (
    'eval_size', 'train_eval_size', 'test_ceiling_mi', 'train_ceiling_mi',
    'test_saturation', 'train_saturation', 'test_trace_saturated_fraction',
)


def _extract_diagnostics(task_results: list) -> Optional[Dict[str, Any]]:
    """Pull the ceiling/saturation diagnostics (Fix 6) out of a ParameterSweep
    task-result list, for the representative (last) task.

    Transfer entropy is a difference of two separately-trained estimates, each
    with its own eval_size/ceiling -- this surfaces one component's numbers
    at the top level so a caller doesn't have to dig into raw_*future lists.
    Uses the last entry (consistent with how the rest of the codebase treats
    "no natural aggregation, pick one representative result" when a
    sweep_grid produces more than one row -- see run.py's sweep-embeddings
    handling); with no sweep_grid (the common case) there is exactly one.
    """
    if not task_results:
        return None
    last = task_results[-1]
    return {k: last.get(k) for k in _DIAGNOSTIC_KEYS if k in last}


def _te_rigorous_scalar(x_s, y_s, bp, sweep_grid=None, history_window=None,
                        prediction_horizon=1, bidirectional=False, w_data=None) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of TE.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1`` -- must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching the
    outer-loop-gets-workers / inner-loop-sequential convention used for
    dimensionality-mode splits.

    ``w_data`` arrives here already sliced to this gamma-chunk's samples (via
    ``run_rigorous_scalar_analysis``'s ``extra_data`` mechanism, the same one
    ``mode='conditional'``'s rigorous path also uses for its own ``w_data``),
    not the full signal -- forwarded straight through to ``run_transfer_entropy``.
    """
    raw = run_transfer_entropy(
        x_s, y_s, bp,
        history_window=history_window,
        prediction_horizon=prediction_horizon,
        sweep_grid=sweep_grid,
        n_workers=1,
        bidirectional=bidirectional,
        w_data=w_data,
    )
    return raw['te_estimate']
