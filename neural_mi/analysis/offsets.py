# neural_mi/analysis/offsets.py
"""Shared offset/window construction for quantities built on plain (unconditioned)
past/future slices of one or two raw time series.

Generalizes the sliding-window construction already used by
``analysis/transfer.py``'s ``_build_te_arrays`` (built for the two-signal,
conditioned transfer-entropy case) to the simpler single-signal and
unconditioned two-signal cases needed by ``neural_mi/quantities.py``.
"""
import torch
from typing import Tuple

from neural_mi.analysis.transfer import _build_te_arrays
from neural_mi.utils import validate_stride, _as_tensor



def as_rows(data):
    """Promote a raw ``(T, C)`` series to the ``(N, C, w)`` row layout.

    A bare array *is* a grid, one sample per row, so it becomes ``w = 1``. An
    array that already came off a bundle is left alone. Every offset builder
    works on rows from here, so the same slicing serves a raw
    series and an aligned multi-modal grid.
    """
    tensor = _as_tensor(data)
    return tensor.unsqueeze(-1) if tensor.ndim == 2 else tensor


def slice_lags(rows, n_lags, stride, n_valid):
    """``n_lags`` consecutive rows at each strided position, flattened lag-major.

    Returns ``(n_valid, C, n_lags * w)``, with lag ``l``'s slot ``j`` at
    ``l * w + j``. At ``w = 1`` this is exactly ``unfold(0, n_lags, stride)``,
    so a raw series keeps the arrays it has always produced.
    """
    windows = rows.unfold(0, n_lags, stride)[:n_valid]        # (n, C, w, n_lags)
    n, channels = windows.shape[0], windows.shape[1]
    return windows.permute(0, 1, 3, 2).reshape(n, channels, -1)


def build_past_future(signal: torch.Tensor, past_len: int, future_len: int = 1,
                      stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Shape ``(T, n_channels)``: raw time series.
    past_len : int
        Number of past time steps in each ``X_past`` window.
    future_len : int, default=1
        Number of future time steps in each ``X_future`` window.
    stride : int, default=1
        Distance in samples between consecutive rows. At the default of 1 every
        valid position becomes a row, so neighbours share ``past_len - 1`` of
        their samples. Raising it thins the rows without changing the quantity.

    Returns
    -------
    tuple of (x_past, x_future), each shape ``(n_valid, n_channels, {past_len,future_len})``.
    """
    stride = validate_stride(stride, 'this quantity')
    rows = as_rows(signal)
    T = rows.shape[0]
    n_positions = T - past_len - future_len + 1
    if n_positions <= 0:
        raise ValueError(
            f"Not enough time points to build past/future arrays. "
            f"Need > past_len + future_len = {past_len + future_len}, got T={T}."
        )
    n_valid = (n_positions - 1) // stride + 1
    # Both slice from the same strided reference positions, so X_future[i]
    # still starts exactly where X_past[i] ends.
    x_past = slice_lags(rows, past_len, stride, n_valid)
    x_future = slice_lags(rows[past_len:], future_len, stride, n_valid)
    return x_past, x_future


def build_cross_offset(x: torch.Tensor, y: torch.Tensor, past_len: int, future_len: int = 1,
                       stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X_past, Y_future) sliding-window arrays for cross-predictive information.

    ``X_past[i] = x[i : i+past_len]``, ``Y_future[i] = y[i+past_len : i+past_len+future_len]``,
    measuring how much a window of X's past tells you about a window of Y's
    future, unconditioned. Reuses ``_build_te_arrays``'s construction directly (it
    already builds exactly this pair, plus a ``y_past`` this quantity doesn't
    need), instead of re-implementing the same ``unfold`` logic a second
    time.

    Parameters
    ----------
    x, y : torch.Tensor or array-like
        Shape ``(T, n_channels)`` each, same leading dimension.
    past_len : int
        Number of past time steps in each ``X_past`` window.
    future_len : int, default=1
        Number of future time steps in each ``Y_future`` window.
    stride : int, default=1
        Distance in samples between consecutive rows. See
        :func:`build_past_future`.

    Returns
    -------
    tuple of (x_past, y_future), each shape ``(n_valid, n_channels, {past_len,future_len})``.
    """
    x_past, _y_past, y_future = _build_te_arrays(x, y, history_window=past_len,
                                                 prediction_horizon=future_len, stride=stride)
    return x_past, y_future


def resolve_stream_processing(processing) -> dict:
    """Each stream's processor, parameters and clock from a ``Processing``.

    A stream with no processor of its own reads with X's, and one with no
    parameters of its own reads with X's parameters, kept to the keys its
    processor takes. Returns ``{name: (processor_type, params, time)}`` for
    ``'x'``, ``'y'`` and ``'w'``.
    """
    from neural_mi.config import Processing
    from neural_mi.defaults import PROCESSOR_PARAMS_SCHEMA
    if isinstance(processing, dict):
        processing = Processing(**processing)
    out = {}
    for name in ('x', 'y', 'w'):
        proc = getattr(processing, name)
        if proc is None:
            proc = processing.x
        params = getattr(processing, f'{name}_params')
        if params is None and name != 'x':
            accepted = PROCESSOR_PARAMS_SCHEMA.get(proc)
            params = {k: v for k, v in (processing.x_params or {}).items()
                      if accepted is None or k in accepted}
        out[name] = (proc, dict(params or {}), getattr(processing, f'{name}_time'))
    return out


def grid_rows(specs):
    """Put every stream on one grid and hand back its rows.

    Offsets index rows and read a fixed interval into each step, which holds
    only while the rows are uniformly spaced and mean the same instant in every
    stream. A raw ``(T, C)`` array satisfies that by being a grid already. Any
    other modality does not: spike times are a point process until they are
    binned, and two streams on different clocks share no row. The streams are
    built onto one shared grid by their processors and handed back as
    ``(N, C, w)`` rows, which the offset builders treat exactly as they treat a
    raw array.

    Validation is off on purpose. Dropping windows compacts the row axis and so
    destroys the uniform spacing offsets depend on: a 40 s gap in a 1 s grid
    turns the steps into ``{1.0, 41.0}``, and an offset of five rows then means
    five seconds in one place and forty-five in another. Keeping every row
    leaves the gap visible to the coverage warnings.

    Parameters
    ----------
    specs : mapping
        ``{name: dict(data=..., time=..., processor_type=..., processor_params=...)}``,
        in the order the streams should be built. Streams whose data is None
        are skipped.
    """
    from collections import OrderedDict
    from neural_mi.data.handler import create_dataset
    specs = OrderedDict((n, s) for n, s in specs.items() if s.get('data') is not None)
    bundle = create_dataset(specs, validate_windows=False)
    return OrderedDict((name, bundle.stream(name).data) for name in bundle.stream_names)


def one_step_rows(rows, caller: str):
    """Rows one time step wide, as ``(T, C)`` series, for a history built by offset.

    Transfer entropy builds its histories from a plain ``(T, C)`` series, so a
    grid row has to carry a single time step. A resolution-1 grid gives that:
    one bin per row for spikes, one sample per row for continuous data. A
    categorical stream spends its trailing axis on categories, which cannot be
    flattened without changing what a lag means, so it is refused.
    """
    wide = {n: r.shape[-1] for n, r in rows.items() if r.shape[-1] != 1}
    if wide:
        raise ValueError(
            f"{caller} needs a grid whose rows are one time step wide, and "
            f"{sorted(wide)} came back wider than that ({wide}). Set each stream's "
            f"window_size to one step (bin_size for spikes, 1/sample_rate for "
            f"continuous). A categorical stream cannot satisfy this, since its "
            f"encoder uses that axis for categories."
        )
    return {n: r.squeeze(-1) for n, r in rows.items()}
