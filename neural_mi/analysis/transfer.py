# neural_mi/analysis/transfer.py
"""Implements transfer entropy (TE) estimation.

Transfer entropy from X to Y is the conditional MI of Y's future given its
joint past with X, over Y's past alone:

    TE(X→Y) = I(y_future ; x_past | y_past)
             = I(x_past, y_past ; y_future) - I(y_past ; y_future)

Both component MI values are estimated with ``ParameterSweep``. The past and
future arrays come from the raw series through the one builder of rows by
offset (:func:`neural_mi.analysis.offsets.build_rows`), with the offsets of
:func:`te_spec`, set by ``history_window`` and ``prediction_horizon``.
"""
import torch
from typing import Dict, Any, Optional

from neural_mi.analysis.sweep import (combined_spread, rung, _joint_marginal_difference,
                                      amplification_factor)
from neural_mi.analysis.offsets import build_rows, spec_geometry
from neural_mi.data.rows import PHASES_KEY, kept_rows, record_rows
from neural_mi.logger import logger
from neural_mi.utils import validate_stride


def te_spec(history_window: int, prediction_horizon: int = 1, *, source: str = 'x',
            target: str = 'y', conditioned_on_w: bool = False) -> Dict[str, Any]:
    """The offsets of transfer entropy from ``source`` to ``target``, as a spec for
    :func:`neural_mi.analysis.offsets.build_rows`.

    A is the joint past, the source's ``history_window`` steps followed by the
    target's (and W's when ``conditioned_on_w``), so the conditioning set is its
    last channels. B is the target's next ``prediction_horizon`` steps. C is the
    conditioning past alone, the marginal term's X.
    """
    past = range(-history_window, 0)
    conditioning = [(target, s) for s in past] + ([('w', s) for s in past] if conditioned_on_w else [])
    return {'A': [(source, s) for s in past] + conditioning,
            'B': [(target, s) for s in range(prediction_horizon)],
            'C': conditioning}


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
    # The rows come from the one builder of rows by offset, which gives their
    # geometry too, recorded for the leak check of every term's split
    # (neural_mi/data/rows.py). The step is the real stride: at stride 1 a gap
    # of one row buys one sample of separation.
    stride = validate_stride(stride, 'transfer entropy')
    series = {'x': x_data, 'y': y_data, **({'w': w_data} if w_data is not None else {})}
    with_w = w_data is not None
    base_params = dict(base_params)
    # Rows a stride thins shift under shift_windows: built at every step, of
    # which each task keeps one in every stride at a phase drawn each epoch.
    phases = stride if stride > 1 and base_params.get('shift_windows') else 1
    spec = te_spec(history_window, prediction_horizon, conditioned_on_w=with_w)
    xy_past, y_future, y_past_cond, _ = build_rows(series, spec, stride=1 if phases > 1 else stride)
    record_rows(base_params, spec_geometry(spec, stride))
    if phases > 1:
        base_params[PHASES_KEY] = phases
    n_samples = kept_rows(xy_past.shape[0], base_params)
    logger.info(f"Transfer Entropy: {n_samples} valid samples.")

    te, mi_joint, mi_marginal, results_joint, results_marginal, _per_run = _joint_marginal_difference(
        xy_past, y_future, y_past_cond, y_future,
        base_params, sweep_grid, n_workers,
        quantity_name="TE(X→Y)",
        joint_label="xy_past;y_future", marginal_label="y_past;y_future",
        joint_key="i_xypast_yfuture", marginal_key="i_ypast_yfuture", condition_at='last',
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
        # The same offsets with the roles of X and Y swapped, on the same rows.
        yx_past, x_future, x_past_cond, _ = build_rows(
            series, te_spec(history_window, prediction_horizon, source='y', target='x',
                            conditioned_on_w=with_w), stride=1 if phases > 1 else stride)

        te_yx, mi_joint_yx, mi_marginal_yx, results_joint_yx, results_marginal_yx, _per_run_yx = _joint_marginal_difference(
            yx_past, x_future, x_past_cond, x_future,
            base_params, sweep_grid, n_workers,
            quantity_name="TE(Y→X)",
            joint_label="yx_past;x_future", marginal_label="x_past;x_future",
            joint_key="i_yxpast_xfuture", marginal_key="i_xpast_xfuture", raw_key='te_yx_raw',
            condition_at='last',
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
    return rung(raw, (('i_xypast_yfuture', 1), ('i_ypast_yfuture', -1)))
