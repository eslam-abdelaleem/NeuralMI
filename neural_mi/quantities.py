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

from neural_mi.run import run
from neural_mi.results import Results
from neural_mi.logger import call_verbosity, collect_repeats
from neural_mi.parallel import dispatch_tasks
from neural_mi.analysis.assemble import merge_results
from neural_mi.analysis.offsets import build_rows, one_step_series, resolve_stream_processing
from neural_mi.utils import _as_tensor, validate_stride
from neural_mi.embeddings_io import model_file, resolve_model_path, announce_saving_several


# The fit settings a Rigorous config carries into a difference quantity's own
# config (Transfer, Conditional, Interaction).
_FIT_FIELDS = ('gamma_range', 'curvature_t_threshold', 'min_gamma_points',
               'confidence_level', 'residual_threshold', 'leverage_threshold')


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
          show_progress: bool, transfer: Optional[Dict[str, Any]] = None, rows=None) -> Results:
    """One :func:`run` call on arrays a quantity has built.

    ``kind`` is ``'mi'`` for one MI estimate per repeat, or the mode of a
    difference quantity (``'transfer'``, ``'conditional'``, ``'interaction'``).
    A conditional quantity with no conditioning set (``c`` is None) is one MI
    estimate. ``rows`` is the geometry of the rows the quantity built in time
    order (:mod:`neural_mi.data.rows`), which :func:`run` records for the split's
    leak checks, the rigorous chunks and the permutation blocks.
    """
    from neural_mi.config import Conditional, Interaction, Transfer
    on, config = _rigorous_settings(rigorous)
    common = dict(n_workers=n_workers, show_progress=show_progress, _rows=rows, **run_kwargs)
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
    kind, a, b, c, transfer, rows, run_kwargs, rigorous, show_progress = task
    return _call(kind, a, b, c, run_kwargs=run_kwargs, rigorous=rigorous, n_workers=1,
                 show_progress=show_progress, transfer=transfer, rows=rows)


def _refuse_custom_split(run_kwargs: Dict[str, Any], quantity: str) -> None:
    """Refuse custom split indices for a quantity that builds its own rows."""
    from neural_mi.config import Split, as_config
    split = as_config(run_kwargs.get('split'), Split)
    if split is not None and (split.train_indices is not None or split.test_indices is not None):
        raise ValueError(
            f"{quantity}() trains on rows built from shifted copies of the data. "
            f"Split(train_indices=..., test_indices=...) would then address rows other than "
            f"the ones you passed. Split with Split(mode=...) and Split(train_fraction=...)."
        )


def _named(quantity: str, param: str, value, build, *, run_kwargs: Dict[str, Any], rigorous,
           n_workers: int, show_progress: bool, construction: Dict[str, Any],
           sweep_mode: str, rows_are_the_input: bool = False) -> Results:
    """Run a quantity at one value of its own parameter, or at every value of an
    iterable, merged into one result.

    ``build(v)`` returns ``(kind, a, b, c, transfer_kwargs, rows)`` for value ``v``,
    ``rows`` being the geometry of the rows it built (:mod:`neural_mi.data.rows`).
    ``construction`` holds the settings held fixed, recorded in ``params``.
    ``sweep_mode`` labels a merged result. ``rows_are_the_input`` says that the
    rows built are the rows passed, in order, so custom split indices still
    address them.
    """
    with call_verbosity(run_kwargs.get('verbose')), collect_repeats():
        return _named_run(quantity, param, value, build, run_kwargs=run_kwargs, rigorous=rigorous,
                          n_workers=n_workers, show_progress=show_progress,
                          construction=construction, sweep_mode=sweep_mode,
                          rows_are_the_input=rows_are_the_input)


def _named_run(quantity: str, param: str, value, build, *, run_kwargs: Dict[str, Any], rigorous,
               n_workers: int, show_progress: bool, construction: Dict[str, Any],
               sweep_mode: str, rows_are_the_input: bool = False) -> Results:
    if not rows_are_the_input:
        _refuse_custom_split(run_kwargs, quantity)
    grid = dict(run_kwargs.get('sweep_grid') or {})
    if param in grid:
        raise ValueError(
            f"{quantity}() takes '{param}' as its own argument. Pass the values as "
            f"{param}=[...] and leave '{param}' out of sweep_grid."
        )
    record = {**construction, 'quantity': quantity}
    run_kwargs = _with_save_path(run_kwargs, quantity)
    if param is None or not _is_sweep(value):
        kind, a, b, c, transfer, rows = build(value)
        result = _call(kind, a, b, c, run_kwargs=run_kwargs, rigorous=rigorous,
                       n_workers=n_workers, show_progress=show_progress, transfer=transfer,
                       rows=rows)
        result.params.update({**record, **({param: value} if param is not None else {})})
        return result
    values = list(value)
    save_path = _save_path(run_kwargs)
    if save_path:
        root, ext = os.path.splitext(save_path)
        announce_saving_several(f"{root}_{param}-<value>{ext}")
    tasks = [(*build(v), _with_save_path(run_kwargs, quantity, **{param: v}), rigorous, show_progress)
             for v in values]
    parts = dispatch_tasks(tasks, _task, n_workers=n_workers, show_progress=show_progress,
                           desc=f"{quantity} sweep")
    params = {**parts[0].params, **record}
    return merge_results([({param: v}, part) for v, part in zip(values, parts)],
                         {param: values, **grid}, params, mode=sweep_mode)


def _sweep_label(rigorous) -> str:
    return 'rigorous' if _rigorous_settings(rigorous)[0] else 'sweep'


def _series(streams: Dict[str, Any], processing, quantity: str) -> 'OrderedDict[str, Any]':
    """A quantity's streams as ``(T, C)`` series, the one way they reach the builder of rows by offset.

    Without ``processing`` each stream is a regularly sampled series already,
    ``(T, n_channels)`` or one-dimensional. With ``processing=`` every stream
    goes onto one grid of rows one step wide that tile time
    (:func:`neural_mi.analysis.offsets.one_step_series`), each read with its own
    processor from ``processing``: ``x``, ``y`` and ``w``, with Y and W falling
    back to X's.
    """
    if processing is not None:
        resolved = resolve_stream_processing(processing)
        specs = OrderedDict()
        for name, data in streams.items():
            proc, params, time = resolved[name]
            specs[name] = dict(data=data, time=time, processor_type=proc, processor_params=params)
        return one_step_series(specs, f"{quantity}()")
    series = OrderedDict()
    for name, data in streams.items():
        tensor = _as_tensor(data)
        if tensor.ndim not in (1, 2):
            raise ValueError(
                f"{quantity}() builds its rows from (T, n_channels) series, and {name}_data has "
                f"shape {tuple(tensor.shape)}. Pass the series unwindowed, or put spike times and "
                f"streams on separate clocks onto one grid with processing=. Rows built elsewhere "
                f"go to nmi.run()."
            )
        series[name] = tensor
    return series


def _steps(name: str, first: int, last: int) -> list:
    """``(name, offset)`` for every offset from ``first`` to ``last``, both included."""
    return [(name, s) for s in range(first, last + 1)]


def _by_offsets(quantity: str, streams: Dict[str, Any], spec_of, param: Optional[str], value, *,
                stride: int, rigorous, n_workers: int, show_progress: bool,
                run_kwargs: Dict[str, Any], construction: Dict[str, Any], sweep_mode: str) -> Results:
    """Run a quantity defined by time offsets: its streams through the one path to ``(T, C)`` series,
    its rows through the one builder, which gives their geometry too.

    ``spec_of(v)`` is the quantity's spec at value ``v`` of its own parameter
    (:func:`neural_mi.utils.build_offset_arrays`). A spec with a C group is a
    conditional quantity, a difference of two networks, and one without is one
    MI estimate.
    """
    stride = validate_stride(stride, f"{quantity}()")
    series = _series(streams, run_kwargs.pop('processing', None), quantity)

    def build(v):
        a, b, c, rows = build_rows(series, spec_of(v), stride=stride)
        return ('conditional' if c is not None else 'mi', a, b, c, None, rows)

    # Offsets all at zero, every row kept: the rows are the ones passed, in
    # order, so custom split indices still address them (instantaneous_mi).
    values = list(value) if _is_sweep(value) else [value]
    own = stride == 1 and all(offset == 0 for v in values for group in spec_of(v).values()
                              for _, offset in (group or []))
    return _named(quantity, param, value, build, run_kwargs=run_kwargs, rigorous=rigorous,
                  n_workers=n_workers, show_progress=show_progress, construction=construction,
                  sweep_mode=sweep_mode, rows_are_the_input=own)


def _spec_active_information_storage(k: int, future_k: int = 1) -> dict:
    return {'A': _steps('x', -k, -1), 'B': _steps('x', 0, future_k - 1)}


def _spec_predictive_information(k: int) -> dict:
    return {'A': _steps('x', -k, -1), 'B': _steps('x', 0, k - 1)}


def _spec_cross_predictive_information(k: int) -> dict:
    return {'A': _steps('x', -k, -1), 'B': _steps('y', 0, k - 1)}


def _spec_instantaneous_mi() -> dict:
    return {'A': [('x', 0)], 'B': [('y', 0)]}


def _spec_mi_rate(h: int, half_width: int) -> dict:
    """X_all, the steps from ``-half_width`` to ``half_width`` around Y_0, given Y's ``h`` steps before
    Y_0. Without a history (``h=0``) it is one MI estimate."""
    spec = {'A': _steps('x', -half_width, half_width), 'B': [('y', 0)]}
    if h > 0:
        spec['C'] = _steps('y', -h, -1)
    return spec


def _spec_instantaneous_exchange(k: int) -> dict:
    """X_0 and Y_0 given both pasts of ``k`` steps, X's channels first. At ``k=0`` it is
    instantaneous MI."""
    spec = _spec_instantaneous_mi()
    if k > 0:
        spec['C'] = _steps('x', -k, -1) + _steps('y', -k, -1)
    return spec


def _spec_directed_information_rate(k: int) -> dict:
    """X's past of ``k`` steps and X_0 against Y_0, given Y's past of ``k`` steps. At ``k=0`` it is
    instantaneous MI."""
    spec = {'A': _steps('x', -k, 0), 'B': [('y', 0)]}
    if k > 0:
        spec['C'] = _steps('y', -k, -1)
    return spec


# ---------------------------------------------------------------------------
# Unconditioned quantities
# ---------------------------------------------------------------------------

def active_information_storage(
    x_data, k: Union[int, list], future_k: int = 1, stride: int = 1, rigorous=False,
    n_workers: int = 1, show_progress: bool = True, **run_kwargs
) -> Results:
    r"""Active information storage :math:`I(X_{past}; X_0)`.

    How much a window of X's own past predicts its present: X's ``k`` steps
    before the present against its ``future_k`` steps from the present on.

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
        of rows one step wide first.

    Returns
    -------
    neural_mi.results.Results
    """
    return _by_offsets('active_information_storage', OrderedDict(x=x_data),
                       lambda kv: _spec_active_information_storage(kv, future_k), 'k', k,
                       stride=stride, rigorous=rigorous, n_workers=n_workers,
                       show_progress=show_progress, run_kwargs=run_kwargs,
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
    return _by_offsets('predictive_information', OrderedDict(x=x_data),
                       _spec_predictive_information, 'k', k, stride=stride, rigorous=rigorous,
                       n_workers=n_workers, show_progress=show_progress, run_kwargs=run_kwargs,
                       construction={'stride': stride}, sweep_mode=_sweep_label(rigorous))


def instantaneous_mi(x_data, y_data, stride: int = 1, rigorous=False, n_workers: int = 1,
                     show_progress: bool = True, **run_kwargs) -> Results:
    r"""Instantaneous mutual information :math:`I(X_0; Y_0)`.

    MI at matching time steps, built from the same rows as the ``k=0`` forms
    of :func:`instantaneous_exchange` and :func:`directed_information_rate`.
    ``mode='estimate'`` computes MI on rows built in any other way.

    Parameters
    ----------
    x_data, y_data : array-like, shape (T, n_channels)
        Raw time series, same leading dimension, or any streams ``processing``
        can put on a grid.
    stride, rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results
    """
    return _by_offsets('instantaneous_mi', OrderedDict(x=x_data, y=y_data),
                       lambda _: _spec_instantaneous_mi(), None, None, stride=stride,
                       rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                       run_kwargs=run_kwargs, construction={'stride': stride},
                       sweep_mode=_sweep_label(rigorous))


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
    return _by_offsets('cross_predictive_information', OrderedDict(x=x_data, y=y_data),
                       _spec_cross_predictive_information, 'k', k, stride=stride,
                       rigorous=rigorous, n_workers=n_workers, show_progress=show_progress,
                       run_kwargs=run_kwargs, construction={'stride': stride},
                       sweep_mode=_sweep_label(rigorous))


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

    How much of :math:`Y`'s present is predicted by :math:`X`'s past beyond what
    :math:`Y`'s own past already predicts. Runs ``mode='transfer'``.

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
        # mode='transfer' builds the histories and records their geometry.
        return ('transfer', x_data, y_data, None,
                {'history_window': hw, 'stride': stride, 'bidirectional': bidirectional}, None)

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
    which controls for how much of :math:`X`'s apparent influence on :math:`Y` a
    third process :math:`W` already explains. Runs ``mode='transfer'`` with ``w_data``.

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
                {'history_window': hw, 'stride': stride, 'bidirectional': bidirectional}, None)

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
# align='dual_branch': the joint term's X is the pair (A, C).

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
        fixed. Both windows change the answer. A curve that has flattened along
        one of them is therefore no evidence on its own. Too narrow a window on
        X omits signal and reads low. Too little conditioning history reads high
        when Y's memory comes from what it shares with X and low when Y has
        private memory its own past would cancel. Sweep one, fix it past its
        knee, then sweep the other.
        Passing iterables for both raises, since the grid multiplies training
        runs for a reading two sequential sweeps already give.
    stride : int, default=1
        See :func:`active_information_storage`.
    rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`.

    Returns
    -------
    neural_mi.results.Results
    """
    if _is_sweep(h) and _is_sweep(half_width):
        raise ValueError(
            "mi_rate takes an iterable for h or for half_width and refuses one for both. "
            "Both windows change the estimate. Sweep one, fix it "
            "past its knee and then sweep the other. A grid over both costs "
            "len(h) * len(half_width) training runs for the same conclusion."
        )
    if _is_sweep(half_width):
        param, value, construction = 'half_width', half_width, {'h': h, 'stride': stride}

        def spec_of(v):
            return _spec_mi_rate(h, v)
    else:
        param, value, construction = 'h', h, {'half_width': half_width, 'stride': stride}

        def spec_of(v):
            return _spec_mi_rate(v, half_width)
    return _by_offsets('mi_rate', OrderedDict(x=x_data, y=y_data), spec_of, param, value,
                       stride=stride, rigorous=rigorous, n_workers=n_workers,
                       show_progress=show_progress, run_kwargs=run_kwargs,
                       construction=construction, sweep_mode='conditional')


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
    return _by_offsets('instantaneous_exchange', OrderedDict(x=x_data, y=y_data),
                       _spec_instantaneous_exchange, 'k', k, stride=stride, rigorous=rigorous,
                       n_workers=n_workers, show_progress=show_progress, run_kwargs=run_kwargs,
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
    return _by_offsets('directed_information_rate', OrderedDict(x=x_data, y=y_data),
                       _spec_directed_information_rate, 'k', k, stride=stride, rigorous=rigorous,
                       n_workers=n_workers, show_progress=show_progress, run_kwargs=run_kwargs,
                       construction={'stride': stride}, sweep_mode='conditional')


# ---------------------------------------------------------------------------
# Any quantity defined by time offsets
# ---------------------------------------------------------------------------

def offset_quantity(data: Dict[str, Any], spec: Dict[str, Any], stride: int = 1, rigorous=False,
                    n_workers: int = 1, show_progress: bool = True, **run_kwargs) -> Results:
    r"""The quantity a spec of time offsets defines, :math:`I(A; B \mid C)`, for one with no
    function of its own.

    Every named quantity defined by time offsets is such a spec. This runs any
    other the way they run: the rows come from
    :func:`neural_mi.utils.build_offset_arrays`, and their geometry reaches the
    split's leak checks, the rigorous chunks and the permutation blocks, which
    rows built by hand and passed to :func:`neural_mi.run` do not carry. A spec
    with a C group runs ``mode='conditional'`` with ``align='dual_branch'``, a
    difference of two networks, and one without is one MI estimate.

    Parameters
    ----------
    data : dict of str to array-like
        The streams by name, each ``(T, n_channels)`` with the same ``T``. With
        ``processing=`` the names are ``'x'``, ``'y'`` and ``'w'``, which say
        which processor reads each stream, and the streams go onto one grid of
        rows one step wide first.
    spec : dict
        ``{'A': [...], 'B': [...], 'C': [...]}``, each group a list of
        ``(stream, offset)`` pairs, offsets counting steps with the past
        negative. C is optional. See :func:`neural_mi.utils.build_offset_arrays`.
    stride, rigorous, n_workers, show_progress, **run_kwargs
        See :func:`active_information_storage`. ``result.params['spec']``
        records the spec.

    Returns
    -------
    neural_mi.results.Results
    """
    if run_kwargs.get('processing') is not None and set(data) - {'x', 'y', 'w'}:
        raise ValueError(
            f"With processing=, offset_quantity() reads its streams with the processors of "
            f"'x', 'y' and 'w', and data names {sorted(set(data) - {'x', 'y', 'w'})}. Name the "
            f"streams 'x', 'y' and 'w', or pass regular (T, n_channels) series without processing=."
        )
    return _by_offsets('offset_quantity', OrderedDict(data), lambda _: spec, None, None,
                       stride=stride, rigorous=rigorous, n_workers=n_workers,
                       show_progress=show_progress, run_kwargs=run_kwargs,
                       construction={'spec': spec, 'stride': stride},
                       sweep_mode=_sweep_label(rigorous))
