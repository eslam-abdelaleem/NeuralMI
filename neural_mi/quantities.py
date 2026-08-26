# neural_mi/quantities.py
"""Named convenience functions for the information-quantities taxonomy.

Every quantity here is an unconditioned :math:`I(A;B)` on offset slices of
one or two raw time series, exactly what ``mode='estimate'`` already
computes -- these are thin wrappers that build the right arrays (via
``analysis/offsets.py``, or the library's own windowed ``Processing`` for
``block_mi``) and call :func:`neural_mi.run`, returning the same
:class:`~neural_mi.results.Results` object unchanged.

Each function's natural construction parameter (``k``, ``past_k``,
``window_size``) accepts either a scalar (one call, one ``Results``) or an
iterable (a parallel sweep across values via :func:`neural_mi.parallel.dispatch_tasks`,
aggregated into a plain :class:`pandas.DataFrame`). See ``neural_mi/parallel.py``
for the dispatch mechanics.
"""
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from neural_mi.run import run
from neural_mi.parallel import dispatch_tasks
from neural_mi.analysis.offsets import build_past_future, build_cross_offset
from neural_mi.analysis.transfer import _build_te_arrays


def _is_sweep(value: Any) -> bool:
    """True if *value* names multiple values to sweep, not one scalar."""
    return isinstance(value, (list, tuple, range, np.ndarray))


def _run_prebuilt_task(task: Tuple[torch.Tensor, torch.Tensor, str, Any, Dict[str, Any], bool]) -> Dict[str, Any]:
    """Module-level (picklable) dispatch target for a single mode='estimate'
    call on already-offset-constructed arrays."""
    x_arr, y_arr, param_name, param_value, run_kwargs, show_progress = task
    result = run(x_data=x_arr, y_data=y_arr, mode='estimate', n_workers=1,
                 show_progress=show_progress, **run_kwargs)
    return {param_name: param_value, 'mi_estimate': result.mi_estimate}


def _processing_with_window(run_kwargs: Dict[str, Any], window_size) -> Tuple[Any, Dict[str, Any]]:
    """Merge ``window_size`` into the caller's ``Processing``, or supply one.

    ``block_mi`` is the one quantity that does real windowing rather than
    offset slicing, so it owns ``window_size`` while the caller owns everything
    else about how their data is read. A caller passing
    ``Processing(x='spike', y='spike')`` gets spike windowing at the swept
    size; passing nothing gets the continuous default.

    Returns the resolved ``Processing`` and a copy of ``run_kwargs`` with the
    original removed, so it cannot arrive at ``run()`` twice.
    """
    from neural_mi.config import Processing
    rest = dict(run_kwargs)
    given = rest.pop('processing', None)
    if given is None:
        return Processing(x='continuous', y='continuous',
                          x_params={'window_size': window_size},
                          y_params={'window_size': window_size}), rest
    if isinstance(given, dict):
        given = Processing(**given)
    merged = replace(given)
    merged.x = given.x if given.x is not None else 'continuous'
    merged.y = given.y if given.y is not None else merged.x
    merged.x_params = {**(given.x_params or {}), 'window_size': window_size}
    merged.y_params = {**(given.y_params or {}), 'window_size': window_size}
    return merged, rest


def _run_block_mi_task(task: Tuple[Any, Any, Any, Dict[str, Any], bool]) -> Dict[str, Any]:
    """Module-level (picklable) dispatch target for one block_mi window_size."""
    x_data, y_data, window_size, run_kwargs, show_progress = task
    processing, rest = _processing_with_window(run_kwargs, window_size)
    result = run(
        x_data=x_data, y_data=y_data, mode='estimate', processing=processing,
        n_workers=1, show_progress=show_progress, **rest,
    )
    return {'window_size': window_size, 'mi_estimate': result.mi_estimate}


def active_information_storage(
    x_data, k: Union[int, list], future_k: int = 1,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Active information storage :math:`I(X_{past}; X_0)`.

    How much a window of X's own past predicts its (immediate) present.
    Built from a single signal via :func:`analysis.offsets.build_past_future`
    (``past_len=k``, ``future_len=future_k``) and estimated via
    ``mode='estimate'``.

    Parameters
    ----------
    x_data : array-like, shape (T, n_channels)
        Raw time series.
    k : int or iterable of int
        History length. An iterable sweeps every value in parallel
        (``n_workers``-dispatched) and returns a DataFrame with columns
        ``k``, ``mi_estimate``, instead of a single ``Results``.
    future_k : int, default=1
        Future window length (1 = the instantaneous present; use
        :func:`excess_entropy` for a longer future window).
    n_workers : int, default=1
        Worker processes. For a sweep, this parallelises across values of
        ``k``; for a single ``k``, forwarded to the underlying ``mode='estimate'``
        run's own ``n_workers``.
    **run_kwargs
        Forwarded to :func:`neural_mi.run` (``model=``, ``training=``, etc.).

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if _is_sweep(k):
        tasks = [
            (*build_past_future(x_data, past_len=kv, future_len=future_k), 'k', kv, run_kwargs, show_progress)
            for kv in k
        ]
        rows = dispatch_tasks(tasks, _run_prebuilt_task, n_workers=n_workers,
                               show_progress=show_progress, desc="active_information_storage sweep")
        return pd.DataFrame(rows)
    x_past, x_future = build_past_future(x_data, past_len=k, future_len=future_k)
    return run(x_data=x_past, y_data=x_future, mode='estimate',
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def excess_entropy(
    x_data, k: Union[int, list], future_k: int,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Excess entropy :math:`I(X_{past}; X_{fut})`.

    The same construction as :func:`active_information_storage`, with a
    multi-sample future window (``future_k``) instead of a single present
    time step -- so :math:`E_X \ge AIS_X` always, since a longer future
    window can only reveal more.

    Parameters
    ----------
    x_data : array-like, shape (T, n_channels)
        Raw time series.
    k : int or iterable of int
        History length. An iterable sweeps every value in parallel and
        returns a DataFrame with columns ``k``, ``mi_estimate``.
    future_k : int
        Future window length (kept fixed across a ``k``-sweep).
    n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if _is_sweep(k):
        tasks = [
            (*build_past_future(x_data, past_len=kv, future_len=future_k), 'k', kv, run_kwargs, show_progress)
            for kv in k
        ]
        rows = dispatch_tasks(tasks, _run_prebuilt_task, n_workers=n_workers,
                               show_progress=show_progress, desc="excess_entropy sweep")
        return pd.DataFrame(rows)
    x_past, x_future = build_past_future(x_data, past_len=k, future_len=future_k)
    return run(x_data=x_past, y_data=x_future, mode='estimate',
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def instantaneous_mi(x_data, y_data, n_workers: int = 1, show_progress: bool = True, **run_kwargs):
    r"""Instantaneous mutual information :math:`I(X_0; Y_0)`.

    Plain, unwindowed MI at matching time indices -- no offset construction
    needed, this is exactly what ``mode='estimate'`` computes directly on
    ``(T, n_channels)`` (or already-windowed) input. Provided for naming
    symmetry with the other quantities in this module, not for new logic.

    Parameters
    ----------
    x_data, y_data : array-like
        Same leading (time) dimension.
    n_workers, show_progress, **run_kwargs
        Forwarded to :func:`neural_mi.run`.

    Returns
    -------
    neural_mi.results.Results
    """
    return run(x_data=x_data, y_data=y_data, mode='estimate',
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def cross_predictive_information(
    x_data, y_data, past_k: Union[int, list], future_k: int = 1,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Cross-predictive information :math:`I(X_{past}; Y_{fut})`.

    How much a window of X's past tells you about a window of Y's future.
    Unconditioned -- one training run, no subtraction. Built via
    :func:`analysis.offsets.build_cross_offset`.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    past_k : int or iterable of int
        X's history length. An iterable sweeps every value in parallel and
        returns a DataFrame with columns ``past_k``, ``mi_estimate``.
    future_k : int, default=1
        Y's future window length (kept fixed across a ``past_k``-sweep).
    n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if _is_sweep(past_k):
        tasks = [
            (*build_cross_offset(x_data, y_data, past_len=pk, future_len=future_k), 'past_k', pk, run_kwargs, show_progress)
            for pk in past_k
        ]
        rows = dispatch_tasks(tasks, _run_prebuilt_task, n_workers=n_workers,
                               show_progress=show_progress, desc="cross_predictive_information sweep")
        return pd.DataFrame(rows)
    x_past, y_future = build_cross_offset(x_data, y_data, past_len=past_k, future_len=future_k)
    return run(x_data=x_past, y_data=y_future, mode='estimate',
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def block_mi(
    x_data, y_data, window_size: Union[int, list],
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Block mutual information :math:`I(X_{1:w}; Y_{1:w})`.

    Already exactly what ``mode='estimate'`` computes with a windowed
    ``Processing``; this wrapper exists for naming symmetry with the other
    quantities in this module, and to route a ``window_size`` sweep through
    the same parallel-dispatch convention as the rest, rather than new
    estimation logic. Raw (non-extensive-corrected) MI over a window of
    length ``window_size`` -- it grows with ``window_size``, see
    ``THEORY.md`` \S2.1 for why this isn't directly comparable across window
    sizes without normalizing.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    window_size : int or iterable of int
        An iterable sweeps every value in parallel and returns a DataFrame
        with columns ``window_size``, ``mi_estimate``.
    n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if _is_sweep(window_size):
        tasks = [(x_data, y_data, wv, run_kwargs, show_progress) for wv in window_size]
        rows = dispatch_tasks(tasks, _run_block_mi_task, n_workers=n_workers,
                               show_progress=show_progress, desc="block_mi sweep")
        return pd.DataFrame(rows)
    processing, rest = _processing_with_window(run_kwargs, window_size)
    return run(
        x_data=x_data, y_data=y_data, mode='estimate', processing=processing,
        n_workers=n_workers, show_progress=show_progress, **rest,
    )


def _run_transfer_task(task: Tuple[Any, Any, Any, int, str, Any, Dict[str, Any], bool]) -> Dict[str, Any]:
    """Module-level (picklable) dispatch target for one conditional_transfer_entropy history_window."""
    x_data, y_data, w_data, history_window, param_name, param_value, run_kwargs, show_progress = task
    from neural_mi.config import Transfer
    result = run(x_data=x_data, y_data=y_data, mode='transfer',
                 transfer=Transfer(history_window=history_window, w_data=w_data),
                 n_workers=1, show_progress=show_progress, **run_kwargs)
    return {param_name: param_value, 'mi_estimate': result.mi_estimate}


def conditional_transfer_entropy(
    x_data, y_data, w_data, history_window: Union[int, list],
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Conditional transfer entropy :math:`\text{TE}_{X\to Y}(W) = I(Y_0; X_{past} \mid Y_{past}, W_{past})`.

    Transfer entropy with a third signal's history folded into the
    conditioning side, controlling for how much of $X$'s apparent influence
    on $Y$ is already explained by a third process $W$. Thin wrapper around
    ``mode='transfer'``'s ``w_data`` parameter (added directly to
    ``run_transfer_entropy``, not a new estimation mechanism); no new mode.

    Parameters
    ----------
    x_data, y_data, w_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    history_window : int or iterable of int
        History length for X_past/Y_past/W_past (all three share it). An
        iterable sweeps every value in parallel and returns a DataFrame with
        columns ``history_window``, ``mi_estimate``.
    n_workers, show_progress, **run_kwargs
        Forwarded to :func:`neural_mi.run` (``model=``, ``training=``, etc.).
        ``run_kwargs`` may also include ``bidirectional=True`` (applies W to
        both directions) or ``rigorous=True``, both forwarded through
        ``transfer=Transfer(...)`` by ``mode='transfer'`` itself.

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    from neural_mi.config import Transfer
    if _is_sweep(history_window):
        tasks = [
            (x_data, y_data, w_data, hw, 'history_window', hw, run_kwargs, show_progress)
            for hw in history_window
        ]
        rows = dispatch_tasks(tasks, _run_transfer_task, n_workers=n_workers,
                               show_progress=show_progress, desc="conditional_transfer_entropy sweep")
        return pd.DataFrame(rows)
    return run(x_data=x_data, y_data=y_data, mode='transfer',
               transfer=Transfer(history_window=history_window, w_data=w_data),
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def interaction_information(x_data, y_data, w_data, n_workers: int = 1, show_progress: bool = True, **run_kwargs):
    r"""Interaction information :math:`II = I(X,W;Y) - I(X;Y) - I(W;Y)`.

    How much shared information between X and Y changes once a third
    population W is also observed. Unlike every other function in this
    module, this isn't a single :math:`I(A;B)`/:math:`I(A;B\mid C)` call --
    it's three separate MI estimates combined by a formula, so it routes
    through its own ``mode='interaction'`` rather than ``mode='estimate'``
    or ``mode='conditional'``. Thin wrapper regardless: no array
    construction of its own, ``x_data``/``y_data``/``w_data`` are used as
    given.

    Parameters
    ----------
    x_data, y_data, w_data : array-like
        Data for the three populations, same leading (sample) dimension;
        ``x_data`` and ``w_data`` must share the same window size.
    n_workers, show_progress, **run_kwargs
        Forwarded to :func:`neural_mi.run` (``model=``, ``training=``,
        ``rigorous=True``, etc.).

    Returns
    -------
    neural_mi.results.Results
    """
    from neural_mi.config import Interaction
    return run(x_data=x_data, y_data=y_data, mode='interaction',
               interaction=Interaction(w_data=w_data),
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


# --- MI rate, instantaneous exchange, directed information rate ---
#
# A and C have genuinely different window lengths for all three (not the
# small mismatch mode='conditional' already tolerates), so each routes
# through align='dual_branch' -- the caller must supply
# model=Model(embedding_model='dual_branch', ...) (or a custom
# DualBranchEmbedding subclass via custom_embedding_cls, for a non-default
# branch architecture -- see the class docstring), checked upfront below
# rather than left to fail deep inside training.

def _is_spike_input(data) -> bool:
    """Raw spike times arrive as a list of one ragged array per neuron."""
    return isinstance(data, (list, tuple)) and not isinstance(data, torch.Tensor) and (
        len(data) > 0 and all(np.ndim(np.asarray(d, dtype=object)) <= 1 for d in data)
        and not np.isscalar(data[0])
    )


def _as_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if _is_spike_input(data):
        raise TypeError(
            "Raw spike times were passed to a quantity that indexes by integer "
            "time offset, which needs a regularly sampled (n_timepoints, "
            "n_channels) series. Build one by binning at the window size and "
            "keeping silent windows, so the time axis stays contiguous:\n"
            "    ds = create_dataset(x_data=spikes, processor_type_x='spike',\n"
            "            processor_params_x={'bin_size': b, 'window_size': b,\n"
            "                                'normalize_bins': False,\n"
            "                                'drop_empty_windows': False})\n"
            "    series = ds.x_data.squeeze(-1)   # (n_bins, n_neurons)\n"
            "Dropping silent windows would leave consecutive indices more than "
            "one bin apart, so the offsets would not be the ones you asked for. "
            "block_mi is the exception: it does real windowing, so it takes "
            "spike times directly via processing=Processing(x='spike', y='spike')."
        )
    return torch.as_tensor(np.asarray(data), dtype=torch.float32)


def _require_dual_branch_model(run_kwargs: Dict[str, Any], fn_name: str) -> None:
    from neural_mi.models.embeddings import DualBranchEmbedding
    model = run_kwargs.get('model')
    embedding_model = getattr(model, 'embedding_model', None) if model is not None else None
    custom_cls = getattr(model, 'custom_embedding_cls', None) if model is not None else None
    is_dual_branch = (
        embedding_model == 'dual_branch'
        or (isinstance(custom_cls, type) and issubclass(custom_cls, DualBranchEmbedding))
    )
    if not is_dual_branch:
        raise ValueError(
            f"{fn_name} needs A and C at different window lengths, which requires "
            f"model=Model(embedding_model='dual_branch', ...) (or a DualBranchEmbedding "
            f"subclass via custom_embedding_cls). Got embedding_model={embedding_model!r}, "
            f"custom_embedding_cls={custom_cls!r}."
        )


def _build_x_zero_aligned(x_data, history_window: int, n_valid: int) -> torch.Tensor:
    """X at the same absolute time as ``_build_te_arrays``'s ``y_future``
    (single time step), for the one row per quantity here (X_0) that isn't
    already one of ``_build_te_arrays``'s three outputs."""
    x_t = _as_tensor(x_data)
    return x_t[history_window:history_window + n_valid].unsqueeze(-1)


def _build_mi_rate_arrays(x_data, y_data, h: int, W: int):
    """Build (X_all, Y_0, Y_past(h)) for MI rate.

    X_all is a two-sided window of half-width ``W`` centered on the same
    time index as Y_0 (the symmetric/two-sided rate, per THEORY.md's
    Massey-feedback note); Y_past(h) covers the h steps strictly before that
    center. Returns ``y_past=None`` when ``h=0`` (no conditioning: this
    reduces to plain instantaneous MI between X_all and Y_0).
    """
    x_t, y_t = _as_tensor(x_data), _as_tensor(y_data)
    T = x_t.shape[0]
    if T < 2 * W + 1:
        raise ValueError(
            f"Not enough time points for a two-sided window of half-width W={W}: "
            f"need T >= {2 * W + 1}, got T={T}."
        )
    start, end = max(W, h), T - W
    if end - start <= 0:
        raise ValueError(f"Not enough time points to build mi_rate arrays for W={W}, h={h} (T={T}).")
    x_all_full = x_t.unfold(0, 2 * W + 1, 1)      # window i covers [i, i+2W], centered at i+W
    x_all = x_all_full[start - W:end - W]
    y0 = y_t[start:end].unsqueeze(-1)
    y_past = None
    if h > 0:
        y_past_full = y_t.unfold(0, h, 1)          # window i covers [i, i+h-1] = offsets -h..-1 of center i+h
        y_past = y_past_full[start - h:end - h]
    return x_all, y0, y_past


def _build_inst_exchange_arrays(x_data, y_data, k: int):
    """Build (X_0, Y_0, [X_past(k)|Y_past(k)]) for instantaneous exchange.

    C concatenates X_past and Y_past channel-wise (both share window length
    k, so no dual-branch is needed for C itself, only for A vs. C). Returns
    ``c=None`` when ``k=0`` (reduces to plain instantaneous MI I(X_0;Y_0)).
    """
    if k == 0:
        x0 = _as_tensor(x_data).unsqueeze(-1)
        y0 = _as_tensor(y_data).unsqueeze(-1)
        return x0, y0, None
    x_past, y_past, y_future = _build_te_arrays(x_data, y_data, history_window=k, prediction_horizon=1)
    x_zero = _build_x_zero_aligned(x_data, k, x_past.shape[0])
    c = torch.cat([x_past, y_past], dim=1)
    return x_zero, y_future, c


def _build_dir_info_rate_arrays(x_data, y_data, k: int):
    """Build (X_past(k)+X_0, Y_0, Y_past(k)) for directed information rate.

    A spans offsets -k..0 (X's past AND its present, one step wider than
    C's -k..-1), built by concatenating X_0 onto X_past rather than relying
    on mode='conditional''s auto-trim-to-shared-start, which would silently
    drop A's last position (X_0) instead of the intended alignment. Returns
    ``c=None`` when ``k=0`` (reduces to plain instantaneous MI I(X_0;Y_0)).
    """
    if k == 0:
        x0 = _as_tensor(x_data).unsqueeze(-1)
        y0 = _as_tensor(y_data).unsqueeze(-1)
        return x0, y0, None
    x_past, y_past, y_future = _build_te_arrays(x_data, y_data, history_window=k, prediction_horizon=1)
    x_zero = _build_x_zero_aligned(x_data, k, x_past.shape[0])
    a = torch.cat([x_past, x_zero], dim=2)
    return a, y_future, y_past


def _run_dual_branch_task(task: Tuple[Any, Any, Any, str, Any, Dict[str, Any], bool]) -> Dict[str, Any]:
    """Module-level (picklable) dispatch target shared by mi_rate,
    instantaneous_exchange, and directed_information_rate: one
    mode='conditional' (align='dual_branch') call on already-built (A, B, C)
    arrays, or plain mode='estimate' when C is None (the h=0/k=0 boundary)."""
    a_arr, b_arr, c_arr, param_name, param_value, run_kwargs, show_progress = task
    if c_arr is None:
        result = run(x_data=a_arr, y_data=b_arr, mode='estimate', n_workers=1,
                     show_progress=show_progress, **run_kwargs)
    else:
        from neural_mi.config import Conditional
        result = run(x_data=a_arr, y_data=b_arr, mode='conditional',
                     conditional=Conditional(w_data=c_arr, align='dual_branch'),
                     n_workers=1, show_progress=show_progress, **run_kwargs)
    return {param_name: param_value, 'mi_estimate': result.mi_estimate}


def mi_rate(
    x_data, y_data, h: Union[int, list], W: int = 20,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""MI rate :math:`I(X_{all}; Y_0 \mid Y_{past}(h))`, the two-sided,
    per-sample information rate as :math:`h \to \infty`.

    :math:`X_{all}` is a symmetric two-sided window of half-width ``W``
    around the same time index as :math:`Y_0`, so :math:`X_{all}` and
    :math:`Y_{past}(h)` generally differ in window length: this always
    routes through ``align='dual_branch'`` (except at the ``h=0`` boundary,
    which has no conditioning at all). See ``THEORY.md`` for why the rate
    only converges to its true value once ``h`` is large enough to capture
    Y's own dependence structure.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    h : int or iterable of int
        Y's conditioning history length. An iterable sweeps every value in
        parallel and returns a DataFrame with columns ``h``, ``mi_estimate``.
    W : int, default=20
        X_all's half-width (kept fixed across an ``h``-sweep); the full
        window spans ``2W+1`` time steps.
    n_workers, show_progress : see :func:`active_information_storage`.
    **run_kwargs
        Forwarded to :func:`neural_mi.run`. Must include
        ``model=Model(embedding_model='dual_branch', ...)``
        for any ``h > 0`` (checked upfront, raises ``ValueError`` if missing
        or mismatched).

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if any((hv > 0) for hv in (h if _is_sweep(h) else [h])):
        _require_dual_branch_model(run_kwargs, 'mi_rate')
    if _is_sweep(h):
        tasks = [
            (*_build_mi_rate_arrays(x_data, y_data, hv, W), 'h', hv, run_kwargs, show_progress)
            for hv in h
        ]
        rows = dispatch_tasks(tasks, _run_dual_branch_task, n_workers=n_workers,
                               show_progress=show_progress, desc="mi_rate sweep")
        return pd.DataFrame(rows)
    a, b, c = _build_mi_rate_arrays(x_data, y_data, h, W)
    if c is None:
        return run(x_data=a, y_data=b, mode='estimate',
                   n_workers=n_workers, show_progress=show_progress, **run_kwargs)
    from neural_mi.config import Conditional
    return run(x_data=a, y_data=b, mode='conditional',
               conditional=Conditional(w_data=c, align='dual_branch'),
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def instantaneous_exchange(
    x_data, y_data, k: Union[int, list],
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Instantaneous exchange :math:`I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))`.

    How much X and Y share at the same instant, beyond what their shared
    past already predicts. :math:`A=X_0` (window length 1) and
    :math:`C=[X_{past}(k) \Vert Y_{past}(k)]` (window length k) differ in
    length whenever ``k > 0``, so this routes through ``align='dual_branch'``
    (except at the ``k=0`` boundary, which has no conditioning at all).

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    k : int or iterable of int
        Shared conditioning history length for both X_past and Y_past. An
        iterable sweeps every value in parallel and returns a DataFrame with
        columns ``k``, ``mi_estimate``.
    n_workers, show_progress : see :func:`active_information_storage`.
    **run_kwargs
        Forwarded to :func:`neural_mi.run`. Must include
        ``model=Model(embedding_model='dual_branch', ...)``
        for any ``k > 0`` (checked upfront, raises ``ValueError`` if missing
        or mismatched).

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if any((kv > 0) for kv in (k if _is_sweep(k) else [k])):
        _require_dual_branch_model(run_kwargs, 'instantaneous_exchange')
    if _is_sweep(k):
        tasks = [
            (*_build_inst_exchange_arrays(x_data, y_data, kv), 'k', kv, run_kwargs, show_progress)
            for kv in k
        ]
        rows = dispatch_tasks(tasks, _run_dual_branch_task, n_workers=n_workers,
                               show_progress=show_progress, desc="instantaneous_exchange sweep")
        return pd.DataFrame(rows)
    a, b, c = _build_inst_exchange_arrays(x_data, y_data, k)
    if c is None:
        return run(x_data=a, y_data=b, mode='estimate',
                   n_workers=n_workers, show_progress=show_progress, **run_kwargs)
    from neural_mi.config import Conditional
    return run(x_data=a, y_data=b, mode='conditional',
               conditional=Conditional(w_data=c, align='dual_branch'),
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)


def directed_information_rate(
    x_data, y_data, k: Union[int, list],
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
):
    r"""Directed information rate :math:`I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))`.

    Computed directly from its own :math:`A`/:math:`B`/:math:`C` arrays,
    not via the :math:`\text{TE}_{X\to Y} + \text{instantaneous exchange}`
    identity (exact on the oracle, but composing through TE's small,
    high-variance residual would reintroduce the fragility the direct route
    avoids; see ``THEORY.md``). :math:`A=[X_{past}(k) \Vert X_0]` (window
    length k+1) and :math:`C=Y_{past}(k)` (window length k) differ by one
    position whenever ``k > 0``, deliberately not relying on
    ``mode='conditional''``'s auto-trim (that would drop A's last position,
    :math:`X_0`, turning this into plain transfer entropy by accident), so
    this always routes through ``align='dual_branch'`` for ``k > 0``.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    k : int or iterable of int
        Shared history length for both X_past and Y_past. An iterable
        sweeps every value in parallel and returns a DataFrame with columns
        ``k``, ``mi_estimate``.
    n_workers, show_progress : see :func:`active_information_storage`.
    **run_kwargs
        Forwarded to :func:`neural_mi.run`. Must include
        ``model=Model(embedding_model='dual_branch', ...)``
        for any ``k > 0`` (checked upfront, raises ``ValueError`` if missing
        or mismatched).

    Returns
    -------
    neural_mi.results.Results or pandas.DataFrame
    """
    if any((kv > 0) for kv in (k if _is_sweep(k) else [k])):
        _require_dual_branch_model(run_kwargs, 'directed_information_rate')
    if _is_sweep(k):
        tasks = [
            (*_build_dir_info_rate_arrays(x_data, y_data, kv), 'k', kv, run_kwargs, show_progress)
            for kv in k
        ]
        rows = dispatch_tasks(tasks, _run_dual_branch_task, n_workers=n_workers,
                               show_progress=show_progress, desc="directed_information_rate sweep")
        return pd.DataFrame(rows)
    a, b, c = _build_dir_info_rate_arrays(x_data, y_data, k)
    if c is None:
        return run(x_data=a, y_data=b, mode='estimate',
                   n_workers=n_workers, show_progress=show_progress, **run_kwargs)
    from neural_mi.config import Conditional
    return run(x_data=a, y_data=b, mode='conditional',
               conditional=Conditional(w_data=c, align='dual_branch'),
               n_workers=n_workers, show_progress=show_progress, **run_kwargs)
