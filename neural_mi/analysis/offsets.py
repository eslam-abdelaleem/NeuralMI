# neural_mi/analysis/offsets.py
"""Shared offset/window construction for quantities built on plain (unconditioned)
past/future slices of one or two raw time series.

Generalizes the sliding-window construction already used by
``analysis/transfer.py``'s ``_build_te_arrays`` (built for the two-signal,
conditioned transfer-entropy case) to the simpler single-signal and
unconditioned two-signal cases needed by ``neural_mi/quantities.py``.
"""
import numpy as np
import torch
from typing import Tuple

from neural_mi.analysis.transfer import _build_te_arrays


def build_past_future(signal: torch.Tensor, past_len: int, future_len: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X_past, X_future) sliding-window arrays from one raw signal.

    ``X_past[i] = signal[i : i+past_len]``,
    ``X_future[i] = signal[i+past_len : i+past_len+future_len]``, so X_future
    starts exactly where X_past ends. Covers active information storage
    (``future_len=1``) and excess entropy (``future_len=w``): both are
    :math:`I(X_{past}; X_{future})` on this same offset shape, differing only
    in how much future is included.

    Parameters
    ----------
    signal : torch.Tensor or array-like
        Shape ``(T, n_channels)`` — raw time series.
    past_len : int
        Number of past time steps in each ``X_past`` window.
    future_len : int, default=1
        Number of future time steps in each ``X_future`` window.

    Returns
    -------
    tuple of (x_past, x_future), each shape ``(n_valid, n_channels, {past_len,future_len})``.
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.as_tensor(np.asarray(signal), dtype=torch.float32)
    T = signal.shape[0]
    n_valid = T - past_len - future_len + 1
    if n_valid <= 0:
        raise ValueError(
            f"Not enough time points to build past/future arrays. "
            f"Need > past_len + future_len = {past_len + future_len}, got T={T}."
        )
    x_past = signal.unfold(0, past_len, 1)[:n_valid]
    x_future = signal[past_len:].unfold(0, future_len, 1)[:n_valid]
    return x_past, x_future


def build_cross_offset(x: torch.Tensor, y: torch.Tensor, past_len: int, future_len: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X_past, Y_future) sliding-window arrays for cross-predictive information.

    ``X_past[i] = x[i : i+past_len]``, ``Y_future[i] = y[i+past_len : i+past_len+future_len]``,
    measuring how much a window of X's past tells you about a window of Y's
    future, unconditioned. Reuses ``_build_te_arrays``'s construction directly (it
    already builds exactly this pair, plus a ``y_past`` this quantity doesn't
    need), rather than re-implementing the same ``unfold`` logic a second
    time.

    Parameters
    ----------
    x, y : torch.Tensor or array-like
        Shape ``(T, n_channels)`` each, same leading dimension.
    past_len : int
        Number of past time steps in each ``X_past`` window.
    future_len : int, default=1
        Number of future time steps in each ``Y_future`` window.

    Returns
    -------
    tuple of (x_past, y_future), each shape ``(n_valid, n_channels, {past_len,future_len})``.
    """
    x_past, _y_past, y_future = _build_te_arrays(x, y, history_window=past_len, prediction_horizon=future_len)
    return x_past, y_future
