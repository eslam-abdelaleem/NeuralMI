# neural_mi/quantities.py
"""Named functions for the information-quantities taxonomy.

Every quantity here builds the arrays its offset pattern needs (via
``analysis/offsets.py``, or the library's own windowed ``Processing`` for
``block_mi``) and calls :func:`neural_mi.run`, returning its
:class:`~neural_mi.results.Results`. None adds estimation logic of its own.

They split by whether the pattern has a conditioning set, and that decides
the mode each one routes to:

* **Unconditioned** :math:`I(A;B)`, one network per repeat:
  :func:`active_information_storage`, :func:`predictive_information`,
  :func:`instantaneous_mi`, :func:`cross_predictive_information`,
  :func:`block_mi`.
* **Conditioned** :math:`I(A;B \\mid C)`, a chain-rule difference of two larger
  estimates: :func:`transfer_entropy` and
  :func:`conditional_transfer_entropy` (``mode='transfer'``), and
  :func:`mi_rate`, :func:`instantaneous_exchange`,
  :func:`directed_information_rate` (``mode='conditional'`` with
  ``align='dual_branch'``, since their groups have different window lengths).
* **A three-estimate combination**: :func:`interaction_information`
  (``mode='interaction'``).

Everything in the second and third groups reports an
``amplification_factor``, because a small difference of two large estimates
inherits their error magnified. Amplification says how fragile the arithmetic
is. Whether the result is distinguishable from zero is a separate question
that only repeats can settle, through ``mi_std`` or a permutation test.

Every function takes the same controls as :func:`neural_mi.run`, and they
compose the same way:

* The quantity's own parameter (``k``, ``history_window``, ``window_size``,
  ``h``, ``half_width``) takes a scalar or an iterable. An iterable is a grid
  key like any other: every value runs, in parallel across ``n_workers``, and
  the values are configurations of one :class:`~neural_mi.results.Results`.
* ``sweep_grid`` adds repeats (``run_id``) and further settings at every value.
* ``rigorous=True``, or a :class:`~neural_mi.config.Rigorous` with the fit
  settings, extrapolates each repeat to infinite data.
"""
import os
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, Optional, Union

import torch

from neural_mi.run import run
from neural_mi.results import Results
from neural_mi.parallel import dispatch_tasks
from neural_mi.analysis.assemble import merge_results
from neural_mi.analysis.offsets import (build_past_future, build_cross_offset, grid_rows,
                                        resolve_stream_processing)
from neural_mi.analysis.transfer import _build_te_arrays
from neural_mi.utils import validate_stride
from neural_mi.embeddings_io import model_file, resolve_model_path, warn_saving_several


# The fit settings a Rigorous config carries into a difference quantity's own
# config (Transfer, Conditional, Interaction).
_FIT_FIELDS = ('gamma_range', 'curvature_t_threshold', 'min_gamma_points',
               'confidence_level', 'residual_threshold', 'r2_threshold', 'leverage_threshold')


def _is_sweep(value: Any) -> bool:
    """True if *value* names several values, not one scalar."""
    import numpy as np
    return isinstance(value, (list, tuple, range, np.ndarray))


def _rigorous_settings(rigorous) -> tuple:
    """``(on, config)`` from ``rigorous=True/False`` or a ``Rigorous(...)``."""
    from neural_mi.config import Rigorous, as_config
    if rigorous is None or rigorous is False:
        return False, None
    if rigorous is True:
        return True, None
    return True, as_config(rigorous, Rigorous)


def _difference_fit(config) -> Dict[str, Any]:
    """A difference quantity's ``rigorous`` fields from a ``Rigorous(...)``."""
    fields = {'rigorous': True}
    if config is None:
        return fields
    if getattr(config, 'temporal_chunking', None) is not None:
        raise ValueError(
            "Rigorous(temporal_chunking=...) applies to single-MI quantities only. A "
            "difference quantity chunks its data contiguously whenever the data are a "
            "time series."
        )
    fields.update({k: getattr(config, k) for k in _FIT_FIELDS if getattr(config, k) is not None})
    return fields


def _call(kind: str, a, b, c, *, run_kwargs: Dict[str, Any], rigorous, n_workers: int,
          show_progress: bool, transfer: Optional[Dict[str, Any]] = None) -> Results:
    """One :func:`run` call on arrays a quantity has built.

    ``kind`` is ``'mi'`` for one MI estimate per repeat, or the mode of a
    difference quantity (``'transfer'``, ``'conditional'``, ``'interaction'``).
    A conditional quantity with no conditioning set (``c`` is None) is one MI
    estimate.
    """
    from neural_mi.config import Conditional, Interaction, Transfer
    on, config = _rigorous_settings(rigorous)
    common = dict(n_workers=n_workers, show_progress=show_progress, **run_kwargs)
    if kind == 'mi' or (kind == 'conditional' and c is None):
        if on:
            return run(a, b, mode='rigorous', rigorous=config, **common)
        mode = 'sweep' if run_kwargs.get('sweep_grid') else 'estimate'
        return run(a, b, mode=mode, **common)
    fit = _difference_fit(config) if on else {}
    if kind == 'transfer':
        return run(a, b, mode='transfer', transfer=Transfer(**(transfer or {}), w_data=c, **fit),
                   **common)
    if kind == 'conditional':
        return run(a, b, mode='conditional',
                   conditional=Conditional(w_data=c, align='dual_branch', **fit), **common)
    if kind == 'interaction':
        return run(a, b, mode='interaction', interaction=Interaction(w_data=c, **fit), **common)
    raise ValueError(f"Unknown quantity kind {kind!r}.")


def _save_path(run_kwargs: Dict[str, Any]) -> Optional[str]:
    from neural_mi.config import Training, as_config
    training = as_config(run_kwargs.get('training'), Training)
    return training.save_best_model_path if training is not None else None


def _with_save_path(run_kwargs: Dict[str, Any], quantity: str, **labels) -> Dict[str, Any]:
    """`run_kwargs` whose saved networks, if any, sit under one stem and carry `labels`.

    A directory becomes one file stem named after the quantity, shared by every
    value of an iterable parameter, and each value adds its own label.
    """
    from neural_mi.config import Training, as_config
    training = as_config(run_kwargs.get('training'), Training)
    if training is None or not training.save_best_model_path:
        return run_kwargs
    path = resolve_model_path(training.save_best_model_path, quantity)
    path = model_file({'save_best_model_path': path, '_model_labels': labels})
    return {**run_kwargs, 'training': replace(training, save_best_model_path=path)}


def _task(task) -> Results:
    """Module-level dispatch target: one value of a quantity's parameter."""
    kind, a, b, c, transfer, run_kwargs, rigorous, show_progress = task
    return _call(kind, a, b, c, run_kwargs=run_kwargs, rigorous=rigorous, n_workers=1,
                 show_progress=show_progress, transfer=transfer)


def _named(quantity: str, param: str, value, build, *, run_kwargs: Dict[str, Any], rigorous,
           n_workers: int, show_progress: bool, construction: Dict[str, Any],
           sweep_mode: str) -> Results:
    """Run a quantity at one value of its own parameter, or at every value of an
    iterable, merged into one result.

    ``build(v)`` returns ``(kind, a, b, c, transfer_kwargs)`` for value ``v``.
    ``construction`` holds the settings held fixed, recorded in ``params``.
    ``sweep_mode`` labels a merged result.
    """
    grid = dict(run_kwargs.get('sweep_grid') or {})
    if param in grid:
        raise ValueError(
            f"{quantity}() takes '{param}' as its own argument. Pass the values as "
            f"{param}=[...] and leave '{param}' out of sweep_grid."
        )
    record = {**construction, 'quantity': quantity}
    run_kwargs = _with_save_path(run_kwargs, quantity)
    if not _is_sweep(value):
        kind, a, b, c, transfer = build(value)
        result = _call(kind, a, b, c, run_kwargs=run_kwargs, rigorous=rigorous,
                       n_workers=n_workers, show_progress=show_progress, transfer=transfer)
        result.params.update({**record, param: value})
        return result
    values = list(value)
    save_path = _save_path(run_kwargs)
    if save_path:
        root, ext = os.path.splitext(save_path)
        warn_saving_several(f"{root}_{param}-<value>{ext}")
    tasks = [(*build(v), _with_save_path(run_kwargs, quantity, **{param: v}), rigorous, show_progress)
             for v in values]
    parts = dispatch_tasks(tasks, _task, n_workers=n_workers, show_progress=show_progress,
                           desc=f"{quantity} sweep")
    params = {**parts[0].params, **record}
    return merge_results([({param: v}, part) for v, part in zip(values, parts)],
                         {param: values, **grid}, params, mode=sweep_mode)


def _sweep_label(rigorous) -> str:
    return 'rigorous' if _rigorous_settings(rigorous)[0] else 'sweep'


def _on_one_grid(streams: Dict[str, Any], processing) -> Optional[Dict[str, Any]]:
    """Put a quantity's streams on one grid of rows, or None without ``processing``.

    A raw ``(T, C)`` array is a grid already, and the offset builders slice it
    directly. Spike times, or streams on different clocks, first go through
    their processors onto one shared grid (see
    :func:`neural_mi.analysis.offsets.grid_rows`). Each stream reads with its
    own processor from ``processing``: ``x``, ``y`` and ``w``, with Y and W
    falling back to X's.
    """
    if processing is None:
        return None
    resolved = resolve_stream_processing(processing)
    specs = OrderedDict()
    for name, data in streams.items():
        proc, params, time = resolved[name]
        specs[name] = dict(data=data, time=time, processor_type=proc, processor_params=params)
    return grid_rows(specs)


# ---------------------------------------------------------------------------
# Unconditioned quantities
# ---------------------------------------------------------------------------

def active_information_storage(
    x_data, k: Union[int, list], future_k: int = 1, stride: int = 1, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Active information storage :math:`I(X_{past}; X_0)`.

    How much a window of X's own past predicts its present. Built from a
    single signal via :func:`analysis.offsets.build_past_future`
    (``past_len=k``, ``future_len=future_k``).

    Parameters
    ----------
    x_data : array-like, shape (T, n_channels)
        Raw time series, or any stream ``processing`` can put on a grid.
    k : int or iterable of int
        History length. An iterable runs every value, and each value is one
        configuration of the result, with ``k`` a column of ``dataframe``.
    future_k : int, default=1
        Future window length (1 is the present; use
        :func:`predictive_information` for a longer future window).
    stride : int, default=1
        Distance in samples between consecutive rows of the arrays this builds.
        At 1, every valid position becomes a row and neighbours share all but
        one of their samples. That is the densest sampling of the recording, and
        it overstates the InfoNCE ceiling the most, since the ceiling counts rows
        without knowing how many are near copies. Raising it thins the rows
        without changing which quantity is measured. It counts whole samples:
        unlike a processor's ``step_size`` it has no fractional reading, because
        these quantities have no single window length for a fraction to refer to.
    rigorous : bool or Rigorous, default=False
        Extrapolate each repeat to infinite data, as ``mode='rigorous'`` does.
        A :class:`~neural_mi.config.Rigorous` sets the fit.
    n_workers : int, default=1
        Worker processes. An iterable ``k`` runs its values in parallel;
        otherwise they go to the call's own training tasks.
    **run_kwargs
        Forwarded to :func:`neural_mi.run`: ``model=``, ``training=``,
        ``sweep_grid=`` (repeats and settings at every ``k``),
        ``permutation_test=``, and so on. ``processing=`` puts a stream that is
        not already a regular series (spike times, a separate clock) on a grid
        of rows first.

    Returns
    -------
    neural_mi.results.Results
    """
    rows = _on_one_grid(OrderedDict((('x', x_data),)), run_kwargs.pop('processing', None))
    if rows is not None:
        x_data = rows['x']

    def build(kv):
        return ('mi', *build_past_future(x_data, past_len=kv, future_len=future_k, stride=stride),
                None, None)

    return _named('active_information_storage', 'k', k, build, run_kwargs=run_kwargs,
                  rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                  construction={'future_k': future_k, 'stride': stride},
                  sweep_mode=_sweep_label(rigorous))


def predictive_information(
    x_data, k: Union[int, list], stride: int = 1, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Predictive information :math:`I(X_{past}(k); X_{fut}(k))`.

    How much a window of a process's own past tells you about an equally long
    window of its future. The two windows have the same length by definition,
    so ``k`` sets both.

    Excess entropy is the :math:`k \to \infty` limit of this quantity. Sweep
    ``k`` to read it: where the curve plateaus, the plateau is that limit;
    where it keeps climbing, no finite excess entropy exists for the process
    and the growth law is the result.

    Parameters
    ----------
    x_data : array-like, shape (T, n_channels)
        Raw time series.
    k : int or iterable of int
        Length of both the past and the future window. See
        :func:`active_information_storage` for an iterable.
    stride, rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results
    """
    rows = _on_one_grid(OrderedDict((('x', x_data),)), run_kwargs.pop('processing', None))
    if rows is not None:
        x_data = rows['x']

    def build(kv):
        return ('mi', *build_past_future(x_data, past_len=kv, future_len=kv, stride=stride),
                None, None)

    return _named('predictive_information', 'k', k, build, run_kwargs=run_kwargs,
                  rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                  construction={'stride': stride}, sweep_mode=_sweep_label(rigorous))


def instantaneous_mi(x_data, y_data, rigorous=False, n_workers: int = 1,
                     show_progress: bool = True, **run_kwargs) -> Results:
    r"""Instantaneous mutual information :math:`I(X_0; Y_0)`.

    MI at matching time indices. ``mode='estimate'`` computes the same quantity on
    ``(T, n_channels)`` or already-windowed input. Provided so the taxonomy has
    a name for every quantity.

    Parameters
    ----------
    x_data, y_data : array-like
        Same leading (time) dimension.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results
    """
    result = _call('mi', x_data, y_data, None, run_kwargs=run_kwargs, rigorous=rigorous,
                   n_workers=n_workers, show_progress=show_progress)
    result.params['quantity'] = 'instantaneous_mi'
    return result


def cross_predictive_information(
    x_data, y_data, k: Union[int, list], stride: int = 1, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Cross-predictive information :math:`I(X_{past}(k); Y_{fut}(k))`.

    Predictive information's two-process form: one process's past against the
    other's future, both windows of length ``k``. Unconditioned, so one
    training run and no subtraction. It measures the shared predictive content
    without isolating X's unique contribution, the thing transfer entropy
    conditions for.

    The name is descriptive; the literature calls this predictive information
    too, and measures it in this cross-process form.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    k : int or iterable of int
        Length of X's past window and of Y's future window. See
        :func:`active_information_storage` for an iterable.
    stride, rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results
    """
    rows = _on_one_grid(OrderedDict((('x', x_data), ('y', y_data))),
                        run_kwargs.pop('processing', None))
    if rows is not None:
        x_data, y_data = rows['x'], rows['y']

    def build(kv):
        return ('mi', *build_cross_offset(x_data, y_data, past_len=kv, future_len=kv,
                                          stride=stride), None, None)

    return _named('cross_predictive_information', 'k', k, build, run_kwargs=run_kwargs,
                  rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                  construction={'stride': stride}, sweep_mode=_sweep_label(rigorous))


def _processing_with_window(processing, window_size):
    """The caller's ``Processing`` with ``window_size`` set for X and Y.

    ``block_mi`` is the one quantity that windows through a processor instead
    of slicing by offset, so it owns ``window_size`` while the caller owns
    everything else about how the data are read. With no ``Processing`` both
    streams are read as continuous.
    """
    from neural_mi.config import Processing
    if processing is None:
        return Processing(x='continuous', y='continuous',
                          x_params={'window_size': window_size},
                          y_params={'window_size': window_size})
    if isinstance(processing, dict):
        processing = Processing(**processing)
    merged = replace(processing)
    merged.x = processing.x if processing.x is not None else 'continuous'
    merged.y = processing.y if processing.y is not None else merged.x
    merged.x_params = {**(processing.x_params or {}), 'window_size': window_size}
    merged.y_params = {**(processing.y_params or {}), 'window_size': window_size}
    return merged


def block_mi(
    x_data, y_data, window_size: Union[float, list], rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Block mutual information :math:`I(X_{1:w}; Y_{1:w})`.

    MI between windows of length ``window_size``. ``mode='estimate'`` computes
    the same quantity with a windowed ``Processing``. It grows with
    ``window_size``; see ``THEORY.md`` for why values at different window sizes
    are not comparable without normalising.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    window_size : float or iterable of float
        An iterable runs every value through the processor grid of
        :func:`neural_mi.run`, with ``window_size`` a column of ``dataframe``.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`. ``processing=`` sets how X and Y
        are read (a spike processor, a sample rate); this function sets its
        ``window_size``.

    Returns
    -------
    neural_mi.results.Results
    """
    on, config = _rigorous_settings(rigorous)
    grid = dict(run_kwargs.pop('sweep_grid', None) or {})
    if 'window_size' in grid:
        raise ValueError(
            "block_mi() takes 'window_size' as its own argument. Pass the values as "
            "window_size=[...] and leave 'window_size' out of sweep_grid."
        )
    values = list(window_size) if _is_sweep(window_size) else [window_size]
    processing = _processing_with_window(run_kwargs.pop('processing', None), values[0])
    if _is_sweep(window_size):
        grid = {'window_size': values, **grid}
    extra = {'rigorous': config} if on else {}
    mode = 'rigorous' if on else ('sweep' if grid else 'estimate')
    result = run(x_data, y_data, mode=mode, processing=processing, sweep_grid=grid or None,
                 n_workers=n_workers, show_progress=show_progress, **extra, **run_kwargs)
    result.params['quantity'] = 'block_mi'
    if not _is_sweep(window_size):
        result.params['window_size'] = window_size
    return result


# ---------------------------------------------------------------------------
# Transfer entropy and interaction information
# ---------------------------------------------------------------------------

def transfer_entropy(
    x_data, y_data, history_window: Union[int, list], stride: int = 1,
    bidirectional: bool = False, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Transfer entropy :math:`\text{TE}_{X\to Y} = I(Y_0; X_{past} \mid Y_{past})`.

    How much of $Y$'s present is predicted by $X$'s past beyond what $Y$'s own
    past already predicts. Runs ``mode='transfer'``.

    TE is a difference of two separately trained MI estimates, and on real
    recordings the difference is often small relative to both. The result's
    ``amplification_factor`` reports
    :math:`(|I(XY_{past};Y_0)| + |I(Y_{past};Y_0)|)\,/\,|\text{TE}|`, and the
    library warns when it is large. Amplification is a property of the
    decomposition. Whether the answer is measurable is a separate question
    that repeats settle, so run several with ``sweep_grid={'run_id': ...}`` and
    compare ``mi_mean`` with ``mi_std`` before reporting a value.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension. ``mode='transfer'`` builds its
        own history and prediction arrays, so do not window these first.
    history_window : int or iterable of int
        History length for X_past and Y_past. See
        :func:`active_information_storage` for an iterable.
    stride : int, default=1
        See :func:`active_information_storage`.
    bidirectional : bool, default=False
        Also estimate :math:`\text{TE}_{Y\to X}` and the directionality index.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`. ``processing=`` puts spike
        times or streams on separate clocks onto one grid of one-step rows.

    Returns
    -------
    neural_mi.results.Results

    See Also
    --------
    conditional_transfer_entropy : the same quantity with a third signal's
        history folded into the conditioning side.
    """
    def build(hw):
        return ('transfer', x_data, y_data, None,
                {'history_window': hw, 'stride': stride, 'bidirectional': bidirectional})

    return _named('transfer_entropy', 'history_window', history_window, build,
                  run_kwargs=run_kwargs, rigorous=rigorous, n_workers=n_workers,
                  show_progress=show_progress, construction={'stride': stride},
                  sweep_mode='transfer')


def conditional_transfer_entropy(
    x_data, y_data, w_data, history_window: Union[int, list], stride: int = 1,
    bidirectional: bool = False, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Conditional transfer entropy :math:`\text{TE}_{X\to Y}(W) = I(Y_0; X_{past} \mid Y_{past}, W_{past})`.

    Transfer entropy with a third signal's history on the conditioning side,
    which controls for how much of $X$'s apparent influence on $Y$ a third
    process $W$ already explains. Runs ``mode='transfer'`` with ``w_data``.

    Parameters
    ----------
    x_data, y_data, w_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    history_window : int or iterable of int
        History length for X_past, Y_past and W_past. See
        :func:`active_information_storage` for an iterable.
    stride : int, default=1
        See :func:`active_information_storage`.
    bidirectional : bool, default=False
        Also estimate the reverse direction, conditioned on W as well.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`transfer_entropy`. ``processing=`` reads W with
        ``Processing(w=..., w_params=..., w_time=...)``, or with X's settings
        when those are unset.

    Returns
    -------
    neural_mi.results.Results
    """
    def build(hw):
        return ('transfer', x_data, y_data, w_data,
                {'history_window': hw, 'stride': stride, 'bidirectional': bidirectional})

    return _named('conditional_transfer_entropy', 'history_window', history_window, build,
                  run_kwargs=run_kwargs, rigorous=rigorous, n_workers=n_workers,
                  show_progress=show_progress, construction={'stride': stride},
                  sweep_mode='transfer')


def interaction_information(x_data, y_data, w_data, rigorous=False, n_workers: int = 1,
                            show_progress: bool = True, **run_kwargs) -> Results:
    r"""Interaction information :math:`II = I(X,W;Y) - I(X;Y) - I(W;Y)`.

    How much the information X and Y share changes once a third population W
    is also observed. It is built from three MI estimates combined by a
    formula, so it runs ``mode='interaction'``, on the data as given.

    Parameters
    ----------
    x_data, y_data, w_data : array-like
        Data for the three populations, same leading (sample) dimension.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results
    """
    result = _call('interaction', x_data, y_data, w_data, run_kwargs=run_kwargs,
                   rigorous=rigorous, n_workers=n_workers, show_progress=show_progress)
    result.params['quantity'] = 'interaction_information'
    return result


# ---------------------------------------------------------------------------
# MI rate, instantaneous exchange, directed information rate
# ---------------------------------------------------------------------------
#
# A and C have different window lengths for all three, beyond the small
# mismatch mode='conditional' trims, so each routes through
# align='dual_branch'. The caller supplies
# model=Model(embedding_model='dual_branch', ...), or a DualBranchEmbedding
# subclass through custom_embedding_cls for another branch architecture, and
# this is checked up front instead of failing inside training.

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


def _build_x_zero_aligned(x_data, history_window: int, n_valid: int,
                          stride: int = 1) -> torch.Tensor:
    """X at the same absolute time as ``_build_te_arrays``'s ``y_future``
    (single time step), for the one row per quantity here (X_0) that isn't
    already one of ``_build_te_arrays``'s three outputs.

    ``stride`` has to match the one those arrays were built at. This is the one
    place in the family that reads time points by direct slice instead of by
    ``unfold``, so a stride applied to one and not the other would misalign A
    against B while leaving every shape correct and raising nothing.
    """
    from neural_mi.analysis.offsets import as_rows
    return as_rows(x_data)[history_window::stride][:n_valid]


def _build_mi_rate_arrays(x_data, y_data, h: int, half_width: int, stride: int = 1):
    """Build (X_all, Y_0, Y_past(h)) for MI rate.

    X_all is a two-sided window of half-width ``half_width`` centred on the
    same time index as Y_0 (the symmetric, two-sided rate); Y_past(h) covers
    the h steps strictly before that centre. Returns ``y_past=None`` when
    ``h=0``: with no conditioning this is plain MI between X_all and Y_0.

    ``stride`` thins the centre times the three arrays share. All three are cut
    from the same ``[start, end)`` range before striding, so they keep the same
    length and stay aligned on the same centres.
    """
    stride = validate_stride(stride, 'mi_rate')
    from neural_mi.analysis.offsets import as_rows, slice_lags
    x_rows, y_rows = as_rows(x_data), as_rows(y_data)
    T = x_rows.shape[0]
    if T < 2 * half_width + 1:
        raise ValueError(
            f"Not enough time points for a two-sided window of half_width={half_width}: "
            f"need T >= {2 * half_width + 1}, got T={T}."
        )
    start, end = max(half_width, h), T - half_width
    if end - start <= 0:
        raise ValueError(
            f"Not enough time points to build mi_rate arrays for half_width={half_width}, "
            f"h={h} (T={T})."
        )
    # window i covers [i, i + 2 * half_width], centred at i + half_width
    x_all_full = slice_lags(x_rows, 2 * half_width + 1, 1, T - 2 * half_width)
    x_all = x_all_full[start - half_width:end - half_width:stride]
    y0 = y_rows[start:end:stride]
    y_past = None
    if h > 0:
        # window i covers [i, i+h-1] = offsets -h..-1 of centre i+h
        y_past_full = slice_lags(y_rows, h, 1, y_rows.shape[0] - h + 1)
        y_past = y_past_full[start - h:end - h:stride]
    return x_all, y0, y_past


def _build_inst_exchange_arrays(x_data, y_data, k: int, stride: int = 1):
    """Build (X_0, Y_0, [X_past(k)|Y_past(k)]) for instantaneous exchange.

    C concatenates X_past and Y_past channel-wise (both share window length
    k, so no dual branch is needed for C itself, only for A against C). Returns
    ``c=None`` when ``k=0``, where this is plain instantaneous MI I(X_0;Y_0).
    """
    if k == 0:
        from neural_mi.analysis.offsets import as_rows
        return as_rows(x_data), as_rows(y_data), None
    x_past, y_past, y_future = _build_te_arrays(x_data, y_data, history_window=k,
                                                prediction_horizon=1, stride=stride)
    x_zero = _build_x_zero_aligned(x_data, k, x_past.shape[0], stride=stride)
    c = torch.cat([x_past, y_past], dim=1)
    return x_zero, y_future, c


def _build_dir_info_rate_arrays(x_data, y_data, k: int, stride: int = 1):
    """Build (X_past(k)+X_0, Y_0, Y_past(k)) for directed information rate.

    A spans offsets -k..0 (X's past and its present, one step wider than C's
    -k..-1), built by concatenating X_0 onto X_past. mode='conditional''s trim
    to a shared start would drop A's last position (X_0) instead. Returns
    ``c=None`` when ``k=0``, where this is plain instantaneous MI I(X_0;Y_0).
    """
    if k == 0:
        from neural_mi.analysis.offsets import as_rows
        return as_rows(x_data), as_rows(y_data), None
    x_past, y_past, y_future = _build_te_arrays(x_data, y_data, history_window=k,
                                                prediction_horizon=1, stride=stride)
    x_zero = _build_x_zero_aligned(x_data, k, x_past.shape[0], stride=stride)
    a = torch.cat([x_past, x_zero], dim=2)
    return a, y_future, y_past


def mi_rate(
    x_data, y_data, h: Union[int, list], half_width: Union[int, list] = 20, stride: int = 1,
    rigorous=False, n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""MI rate :math:`I(X_{all}; Y_0 \mid Y_{past}(h))`, the two-sided,
    per-sample information rate as :math:`h \to \infty`.

    :math:`X_{all}` is a symmetric two-sided window of half-width
    ``half_width`` around the same time index as :math:`Y_0`.
    :math:`X_{all}` and :math:`Y_{past}(h)` generally differ in window length,
    so this routes through ``align='dual_branch'``. At ``h=0`` there is no
    conditioning and this routing does not apply. See ``THEORY.md`` for why the rate converges to its
    true value only once ``h`` covers Y's own dependence structure.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    h : int or iterable of int
        Y's conditioning history length. See
        :func:`active_information_storage` for an iterable.
    half_width : int or iterable of int, default=20
        X_all's half-width; the full window spans ``2 * half_width + 1`` steps.
        Either ``h`` or ``half_width`` may be an iterable, and the other stays
        fixed. Both windows change the answer, and they bias it in opposite
        directions. A curve that has flattened along one of them is therefore
        no evidence on its own. Too little conditioning history leaves Y's own
        storage in and reads high, and too narrow a window on X omits signal
        and reads low. Sweep one, fix it past its knee, then sweep the other.
        Passing iterables for both raises, since the grid multiplies training
        runs for a reading two sequential sweeps already give.
    stride : int, default=1
        See :func:`active_information_storage`.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`. ``model=`` must be
        ``Model(embedding_model='dual_branch', ...)`` whenever ``h > 0``.

    Returns
    -------
    neural_mi.results.Results
    """
    if any(hv > 0 for hv in (h if _is_sweep(h) else [h])):
        _require_dual_branch_model(run_kwargs, 'mi_rate')
    if _is_sweep(h) and _is_sweep(half_width):
        raise ValueError(
            "mi_rate takes an iterable for h or for half_width, and not for both. The "
            "two windows bias the estimate in opposite directions, so the reading you "
            "want comes from sweeping one, fixing it past its knee, then sweeping the "
            "other. A grid over both costs len(h) * len(half_width) training runs for "
            "the same conclusion."
        )
    rows = _on_one_grid(OrderedDict((('x', x_data), ('y', y_data))),
                        run_kwargs.pop('processing', None))
    if rows is not None:
        x_data, y_data = rows['x'], rows['y']
    if _is_sweep(half_width):
        param, value, construction = 'half_width', half_width, {'h': h, 'stride': stride}

        def build(v):
            return ('conditional', *_build_mi_rate_arrays(x_data, y_data, h, v, stride=stride), None)
    else:
        param, value, construction = 'h', h, {'half_width': half_width, 'stride': stride}

        def build(v):
            return ('conditional', *_build_mi_rate_arrays(x_data, y_data, v, half_width,
                                                          stride=stride), None)
    return _named('mi_rate', param, value, build, run_kwargs=run_kwargs, rigorous=rigorous,
                  n_workers=n_workers, show_progress=show_progress, construction=construction,
                  sweep_mode='conditional')


def instantaneous_exchange(
    x_data, y_data, k: Union[int, list], stride: int = 1, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Instantaneous exchange :math:`I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))`.

    How much X and Y share at the same instant, beyond what their shared
    past already predicts. :math:`A=X_0` (window length 1) and
    :math:`C=[X_{past}(k) \Vert Y_{past}(k)]` (window length k) differ in
    length whenever ``k > 0``, so this routes through ``align='dual_branch'``.
    At ``k=0`` there is no conditioning and this routing does not apply.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    k : int or iterable of int
        Conditioning history length for both X_past and Y_past. See
        :func:`active_information_storage` for an iterable.
    stride : int, default=1
        See :func:`active_information_storage`.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`mi_rate`.

    Returns
    -------
    neural_mi.results.Results
    """
    if any(kv > 0 for kv in (k if _is_sweep(k) else [k])):
        _require_dual_branch_model(run_kwargs, 'instantaneous_exchange')
    rows = _on_one_grid(OrderedDict((('x', x_data), ('y', y_data))),
                        run_kwargs.pop('processing', None))
    if rows is not None:
        x_data, y_data = rows['x'], rows['y']

    def build(kv):
        return ('conditional', *_build_inst_exchange_arrays(x_data, y_data, kv, stride=stride), None)

    return _named('instantaneous_exchange', 'k', k, build, run_kwargs=run_kwargs,
                  rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                  construction={'stride': stride}, sweep_mode='conditional')


def directed_information_rate(
    x_data, y_data, k: Union[int, list], stride: int = 1, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Directed information rate :math:`I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))`.

    Computed from its own :math:`A`/:math:`B`/:math:`C` arrays. The identity
    :math:`\text{TE}_{X\to Y} + \text{instantaneous exchange}` is exact on the
    oracle, but going through it would carry TE's small, high-variance residual
    into this estimate (see ``THEORY.md``). :math:`A=[X_{past}(k) \Vert X_0]`
    (window length k+1) and :math:`C=Y_{past}(k)` (window length k) differ by
    one position whenever ``k > 0``. ``mode='conditional'``'s trim to a shared
    start would drop A's last position, :math:`X_0`, and turn this into plain
    transfer entropy. To keep :math:`X_0`, this routes through
    ``align='dual_branch'`` for ``k > 0``.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension.
    k : int or iterable of int
        History length for both X_past and Y_past. See
        :func:`active_information_storage` for an iterable.
    stride : int, default=1
        See :func:`active_information_storage`.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`mi_rate`.

    Returns
    -------
    neural_mi.results.Results
    """
    if any(kv > 0 for kv in (k if _is_sweep(k) else [k])):
        _require_dual_branch_model(run_kwargs, 'directed_information_rate')
    rows = _on_one_grid(OrderedDict((('x', x_data), ('y', y_data))),
                        run_kwargs.pop('processing', None))
    if rows is not None:
        x_data, y_data = rows['x'], rows['y']

    def build(kv):
        return ('conditional', *_build_dir_info_rate_arrays(x_data, y_data, kv, stride=stride), None)

    return _named('directed_information_rate', 'k', k, build, run_kwargs=run_kwargs,
                  rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                  construction={'stride': stride}, sweep_mode='conditional')
