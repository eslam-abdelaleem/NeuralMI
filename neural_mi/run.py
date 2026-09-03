# neural_mi/run.py
"""Provides the main `run` function, the primary entry point for the library.

This module orchestrates the entire analysis pipeline, from data validation
and preprocessing to model training and results aggregation. The `run` function
acts as a unified interface for all supported analysis modes.
"""
import warnings
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Union, Optional, Dict, Any, List
import random
from tqdm.auto import tqdm

from .analysis.rigorous import run_rigorous_analysis
from .analysis.dimensionality import run_dimensionality_analysis
from .analysis.precision import run_precision_analysis
from .analysis.lag import run_lag_analysis
from .analysis.conditional import run_conditional_mi
from .analysis.transfer import run_transfer_entropy
from .analysis.interaction import run_interaction_information
from .analysis.pairwise import run_pairwise_mi, _n_channels_of
from .data.handler import create_dataset
from .data.shift_windowing import shift_family, mixed_pair_sample_rate_ok
from .results import Results
from .validation import ParameterValidator, DataValidator
from .utils import get_device
from .logger import logger, worker_init_args
from .defaults import PROCESSOR_PARAMS_SCHEMA
import inspect as _inspect
from .config import (
    Model, Training, Split, Estimator, Output, Processing,
    Rigorous, Precision, Lag, Transfer, Dimensionality, Conditional,
    Interaction, Pairwise, Sweep, as_config,
)

# Mode name -> its dedicated config class (modes not listed take no mode config).
_MODE_CONFIG_CLASSES = {
    'rigorous': Rigorous, 'precision': Precision, 'lag': Lag,
    'transfer': Transfer, 'dimensionality': Dimensionality, 'conditional': Conditional,
    'interaction': Interaction, 'pairwise': Pairwise, 'sweep': Sweep,
}

# Modes that dispatch one or more independent training runs from raw,
# unwindowed data with no cross-run comparison for per-run shift randomness
# to disturb (or, for 'dimensionality'/'precision', a comparison/evaluation
# that already only reads a frozen, shared, pre-shift view) -- see
# shift-related gating/warnings below and in `_run_flat`.
_SHIFT_SAFE_MODES = ('estimate', 'sweep', 'pairwise', 'dimensionality', 'precision')
# 'rigorous' additionally supports shift_windows for the regular-grid family
# and shift_time for the spike family (each independently): both mechanisms'
# bias-correction ladder chunk boundaries are translated into a raw range
# before that chunk is windowed (see
# rigorous.py::AnalysisWorkflow._prepare_tasks) -- a raw *sample* range for
# shift_windows, a raw *time* range (searchsorted-sliced against a ragged
# per-neuron spike-time list) for shift_time. 'rigorous' is deliberately NOT
# in _SHIFT_SAFE_MODES itself, since that tuple also gates shift_time's
# 'mixed'-family reach (spike paired with continuous/categorical), which
# 'rigorous' does NOT support -- translating a chunk into a raw sample range
# on one side and a raw time range on the other simultaneously is real
# additional work, not attempted this pass.
# 'lag' also supports shift_windows for the regular-grid family: raw,
# unwindowed lag-shifted data already survives to task.py::run_training_task
# (mode='lag' forces the same "defer, don't window here" treatment as
# is_proc_sweep), which calls try_build_shift_windows_dataset unconditionally
# (mode-agnostic) -- confirmed by direct instrumentation that shift_windows
# already engages correctly for mode='lag' today. 'lag' was missing from
# this tuple purely as a reachability-warning bug (a user explicitly setting
# shift_windows=True for mode='lag' got a false "has no effect" warning even
# though it was already working), not because the mechanism needed building.
_SHIFT_WINDOWS_SAFE_MODES = _SHIFT_SAFE_MODES + ('rigorous', 'lag')
_SHIFT_TIME_RIGOROUS_SAFE_MODES = _SHIFT_SAFE_MODES + ('rigorous',)


# Per-epoch MI curves. These are list-valued, so a column-wise multiply cannot
# reach them; they need an elementwise map. Handled inside _convert_mi_units so
# every path gets it, rather than at one call site (which is how the sweep-family
# modes ended up reporting history in nats beside scalars in bits).
_MI_HISTORY_KEYS = ('test_mi_history', 'train_mi_history')


def _scale_history(seq, factor: float):
    """Apply `factor` elementwise to a per-epoch MI curve, preserving NaNs."""
    if seq is None:
        return seq
    try:
        return [v if (v is None or (isinstance(v, float) and np.isnan(v)))
                else v * factor for v in seq]
    except TypeError:
        return seq


def _convert_mi_units(results: Any, to_bits: bool) -> Any:
    """Recursively converts MI values in results from nats to bits.

    Every MI-valued field the library returns carries the unit set by
    `output_units`, whether it is a scalar, a DataFrame column, or a per-epoch
    history list, and whether it sits in `dataframe` or in `details`.
    """
    if not to_bits: return results
    NATS_TO_BITS = 1 / np.log(2)
    if isinstance(results, float): return results * NATS_TO_BITS
    elif isinstance(results, np.ndarray):
        # e.g. mode='pairwise''s mi_matrix. Handled here so callers do not
        # re-implement the nats->bits factor locally.
        return results * NATS_TO_BITS
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
        cols = [
            'test_mi', 'train_mi', 'raw_train_mi', 'train_mi_at_peak',
            'test_mi_std', 'train_mi_std',          # precision-mode std columns
            'test_mi_mean',                         # pairwise-mode held-out aggregate
            'mi_mean', 'mi_std', 'mi_corrected', 'mi_error', 'mi_error_pred', 'slope',
        ]
        for col in cols:
            if col in df.columns: df[col] *= NATS_TO_BITS
        for col in _MI_HISTORY_KEYS:
            if col in df.columns:
                df[col] = df[col].map(lambda v: _scale_history(v, NATS_TO_BITS))
        return df
    elif isinstance(results, list) and all(isinstance(r, dict) for r in results):
        keys = ['test_mi', 'train_mi', 'raw_train_mi', 'train_mi_at_peak',
                'mi_corrected', 'mi_error', 'mi_error_pred', 'slope']
        return [{**r,
                 **{k: r.get(k, 0) * NATS_TO_BITS for k in keys if r.get(k) is not None},
                 **{k: _scale_history(r[k], NATS_TO_BITS)
                    for k in _MI_HISTORY_KEYS if isinstance(r.get(k), list)}}
                for r in results]
    elif isinstance(results, dict):
        new_results = results.copy()
        # Scalar MI values stored by analysis modules (transfer entropy, CMI, etc.)
        # mi_corrected/mi_error/mi_error_pred/slope cover rigorous conditional/transfer's
        # flat scalar-rigorous result dict (run_rigorous_scalar_analysis's return value),
        # which stores them as top-level dict keys rather than nested inside a
        # 'corrected_results' list like plain mode='rigorous' does -- that nested case is
        # already handled below via the 'corrected_results' recursion into the list-of-dicts
        # branch, so adding these keys here doesn't double-convert it.
        _MI_SCALAR_KEYS = (
            'te_estimate', 'te_xy', 'te_yx',
            'i_xypast_yfuture', 'i_ypast_yfuture',
            'i_yxpast_xfuture', 'i_xpast_xfuture',
            'cmi_estimate',
            'interaction_info', 'mi_xw_y', 'mi_x_y', 'mi_w_y',
            'mi_corrected', 'mi_error', 'mi_error_pred', 'slope',
        )
        for k in _MI_SCALAR_KEYS:
            if k in new_results and isinstance(new_results[k], (int, float)):
                new_results[k] = new_results[k] * NATS_TO_BITS
        for k in _MI_HISTORY_KEYS:
            if isinstance(new_results.get(k), list):
                new_results[k] = _scale_history(new_results[k], NATS_TO_BITS)
        if 'corrected_results' in new_results:
            new_results['corrected_results'] = _convert_mi_units(new_results['corrected_results'], to_bits)
        if 'raw_results_df' in new_results:
            new_results['raw_results_df'] = _convert_mi_units(new_results['raw_results_df'], to_bits)
        return new_results
    return results

def _hashable_group_vars(df: pd.DataFrame, group_vars: List[str]) -> pd.DataFrame:
    """Return a copy of `df` where any list-valued columns in `group_vars` are
    converted to tuples so `groupby` can hash them.

    Swept parameters that are themselves lists (e.g. ``sweep_grid={'hidden_dim':
    [[64, 64], [128]]}`` for a per-layer width spec) otherwise crash
    ``DataFrame.groupby`` with ``TypeError: unhashable type: 'list'``. Values are
    preserved exactly, just as tuples instead of lists.
    """
    df = df.copy()
    for col in group_vars:
        if col in df.columns and df[col].map(lambda v: isinstance(v, list)).any():
            df[col] = df[col].map(lambda v: tuple(v) if isinstance(v, list) else v)
    return df


def _align_conditioning_windows(mode, x_run_data, y_run_data, w_run_data,
                                xy_window_times, w_dataset, base_params):
    """Subset X, Y and W to the windows all three retained.

    Window validity is decided per *pair*. X's windows are the ones where X and
    Y are both valid. W is built paired with Y, so its windows are the ones
    where W and Y are both valid. Those two sets coincide only when X and W
    impose comparable constraints, and diverge when they do not: a continuous X
    carrying ``min_coverage_fraction=0.9`` against a categorical W carrying no
    such rule differed by 1501 windows out of 3331 on a real recording.

    ``mode='conditional'`` (non-dual_branch) and ``mode='interaction'``
    concatenate the windowed X and W along the channel axis downstream, so the
    two must line up window for window. Neither two-way criterion delivers that.
    The three-way intersection does, and it is the only formulation that can
    also *shrink* X and Y, which is what is required whenever W is the binding
    constraint.

    Aligning here matters beyond the crash it prevents. The engine-level trim in
    conditional.py/interaction.py absorbs a one-window difference by truncating
    all three to the shared first ``min_n``, on the assumption that the odd
    window sits at a boundary. When it sits in the middle instead, every window
    after it shifts by one: measured on the spike-X/categorical-W/continuous-Y
    combination, a single extra window at index 2730 of 3332 left 601 of 3331
    pairs (18%) referring to different times, silently.

    Returns ``(x, y, w)``, subset when both sides expose window times and
    unchanged when they do not, in which case the existing trim and shape checks
    still apply.
    """
    w_times = getattr(getattr(w_dataset, 'window_manager', None), 'window_times', None)
    if xy_window_times is None or w_times is None:
        return x_run_data, y_run_data, w_run_data
    xy_times = np.asarray(xy_window_times)
    w_times = np.asarray(w_times)
    # intersect1d's returned indices are only meaningful for unique inputs, and
    # window start times are unique by construction; bail out rather than
    # mis-index if some processor ever breaks that.
    if (len(np.unique(xy_times)) != len(xy_times)
            or len(np.unique(w_times)) != len(w_times)):
        return x_run_data, y_run_data, w_run_data

    common, i_xy, i_w = np.intersect1d(xy_times, w_times, return_indices=True)
    if len(common) == len(xy_times) == len(w_times):
        return x_run_data, y_run_data, w_run_data      # already aligned

    if len(common) == 0:
        raise ValueError(
            f"mode='{mode}': X and the conditioning variable W have no windows in "
            f"common, so there is nothing to condition on. X paired with Y retained "
            f"{len(xy_times)} windows and W paired with Y retained {len(w_times)}, "
            f"with no overlapping window times. Window validity is decided per pair, "
            f"so this means X and W are being judged by different rules -- most often "
            f"a `min_coverage_fraction` on one side that the other has no equivalent "
            f"of, or window_size/step_size that differ between processing.x_params "
            f"and w_processor_params. This is not a data-coverage problem."
        )

    _min_common = 2
    if len(common) < _min_common:
        raise ValueError(
            f"mode='{mode}': only {len(common)} window(s) are valid for X, Y and W "
            f"simultaneously ({len(xy_times)} for X with Y, {len(w_times)} for W with "
            f"Y), which is too few to estimate anything. See the note above about "
            f"differing validity rules between X and W."
        )

    logger.info(
        f"mode='{mode}': aligning X/Y ({len(xy_times)} windows) and W "
        f"({len(w_times)}) to the {len(common)} windows valid for all three."
    )
    if len(common) < len(xy_times):
        # X and Y lose windows here, which changes what the estimate covers, so
        # say so rather than shrinking the sample silently.
        logger.warning(
            f"mode='{mode}': the conditioning variable W is valid on fewer windows "
            f"than X, so X and Y have been reduced from {len(xy_times)} to "
            f"{len(common)} windows ({len(common)/len(xy_times):.1%}) to match. The "
            f"estimate describes that shared subset. W's own coverage rules "
            f"(w_processor_params) decide this, so widen them if the reduction is "
            f"larger than you intend."
        )
        if base_params.get('_n_windows_retained'):
            base_params['_n_windows_retained'] = int(len(common))
            _built = base_params.get('_n_windows_built')
            if _built:
                base_params['_window_retention'] = len(common) / _built

    i_xy_t = torch.from_numpy(np.ascontiguousarray(i_xy))
    i_w_t = torch.from_numpy(np.ascontiguousarray(i_w))
    x_out = x_run_data[i_xy_t] if x_run_data is not None else None
    y_out = y_run_data[i_xy_t] if y_run_data is not None else None
    w_out = w_run_data[i_w_t] if w_run_data is not None else None
    return x_out, y_out, w_out


def _reshape_categorical_w_for_conditional(w_run_data, cat_dataset):
    """Re-lay-out a categorical-processor W tensor for ``mode='conditional'``.

    ``mode='conditional'`` builds XW by concatenating X and W along the
    channel axis, which requires both to share X's window-size axis. The
    categorical processor's encodings don't produce that layout natively:

    - ``'majority_vote'`` / ``'probability'`` collapse each window to a
      single per-category summary, shape ``(N, C, n_categories)`` — W has no
      temporal extent within a window by construction. Folded here into
      ``C * n_categories`` channels with a size-1 window axis; the caller
      broadcasts that axis against X's window size.
    - ``'full_trajectory'`` keeps full per-timepoint resolution but flattens
      ``n_categories * window_size`` onto the last axis, shape
      ``(N, C, n_categories * window_size)``. Un-flattened and folded here
      into ``C * n_categories`` channels with the real window axis restored,
      preserving the per-timepoint information.

    Only reshapes the tensor handed to this specific call; the categorical
    processor's own stored data and its behavior in every other mode are
    untouched.
    """
    encoding = cat_dataset.encoding
    n_cat = cat_dataset.n_categories
    n, c, last = w_run_data.shape
    if encoding in ('majority_vote', 'probability'):
        return w_run_data.reshape(n, c * n_cat, 1)
    elif encoding == 'full_trajectory':
        w = cat_dataset.max_samples_per_window
        # (N, C, n_cat*W) -> (N, C, W, n_cat) -> (N, C, n_cat, W) -> (N, C*n_cat, W)
        # The (W, n_cat) un-flatten order matches how _move_full_trajectory
        # wrote columns: col = timepoint_index * n_categories + category.
        return w_run_data.reshape(n, c, w, n_cat).permute(0, 1, 3, 2).reshape(n, c * n_cat, w)
    else:
        raise ValueError(
            f"Unknown categorical encoding '{encoding}' — cannot prepare it for "
            f"mode='conditional'."
        )


def run(
    x_data: Union[np.ndarray, torch.Tensor, List],
    y_data: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    *,
    mode: str = 'estimate',
    processing: Optional[Union[Processing, Dict[str, Any]]] = None,
    model: Optional[Union[Model, Dict[str, Any]]] = None,
    training: Optional[Union[Training, Dict[str, Any]]] = None,
    split: Optional[Union[Split, Dict[str, Any]]] = None,
    estimator: Optional[Union[Estimator, str, Dict[str, Any]]] = None,
    output: Optional[Union[Output, Dict[str, Any]]] = None,
    sweep_grid: Optional[Dict[str, list]] = None,
    rigorous: Optional[Union[Rigorous, Dict[str, Any]]] = None,
    precision: Optional[Union[Precision, Dict[str, Any]]] = None,
    lag: Optional[Union[Lag, Dict[str, Any]]] = None,
    transfer: Optional[Union[Transfer, Dict[str, Any]]] = None,
    dimensionality: Optional[Union[Dimensionality, Dict[str, Any]]] = None,
    conditional: Optional[Union[Conditional, Dict[str, Any]]] = None,
    interaction: Optional[Union[Interaction, Dict[str, Any]]] = None,
    pairwise: Optional[Union[Pairwise, Dict[str, Any]]] = None,
    sweep: Optional[Union[Sweep, Dict[str, Any]]] = None,
    n_workers: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
    device: Optional[str] = None,
    permutation_test: bool = False,
    n_permutations: int = 1,
    permutation_shuffle: str = 'circular',
    **_removed: Any,
) -> Results:
    """Unified entry point for all NeuralMI analyses (config-based API).

    Parameters are grouped into a small set of typed config objects (see
    :mod:`neural_mi.config`). Every config is optional -- omitted configs and
    unset fields fall back to the defaults in
    :data:`neural_mi.defaults.BASE_PARAMS_SCHEMA`. Anywhere a config is accepted
    a plain ``dict`` with the same keys works too, so importing the classes is
    optional.

    Parameters
    ----------
    x_data, y_data : array-like
        Input data for variables X and Y. ``y_data`` is required for all modes
        except ``'dimensionality'``/``'pairwise'`` (self-pairwise). With
        ``processing=Processing(x='continuous'|'categorical', ...)``, raw arrays
        are shape ``(n_timepoints, n_channels)`` (a 1-D array is treated as
        ``(n_timepoints, 1)``). With ``processing=Processing(x='spike', ...)``,
        pass a list of 1-D arrays of spike times, one per channel/neuron.
        Already-processed data (``processing=None``) is shape
        ``(n_samples, n_channels, window_size)`` (3-D) or ``(n_samples, n_channels)``
        (2-D, treated as a trailing window size of 1).
    mode : {'estimate','sweep','rigorous','dimensionality','lag','precision','conditional','interaction','transfer','pairwise'}
        The analysis to run.
    processing : Processing or dict, optional
        Raw-data processors, e.g. ``Processing(x='continuous', x_params={'window_size': 1})``.
    model : Model or dict, optional
        Architecture, e.g. ``Model(embedding_dim=16, hidden_dim=64, critic_type='separable')``.
    training : Training or dict, optional
        Optimization loop, e.g. ``Training(n_epochs=50, learning_rate=1e-3, batch_size=128)``.
    split : Split or dict, optional
        Splitting strategy, e.g. ``Split(mode='random')``.
    estimator : Estimator, str, or dict, optional
        MI estimator. Accepts a bare name (``estimator='smile'``) or
        ``Estimator(name='smile', params={'clip': 5.0})``.
    output : Output or dict, optional
        Units, spectral tracking, embedding returns, and display labels.
    sweep_grid : dict, optional
        Parameter grid for ``mode='sweep'``/``'dimensionality'``.
    rigorous, precision, lag, transfer, dimensionality, conditional, interaction, pairwise, sweep : mode config or dict, optional
        Mode-specific parameters; only the one matching ``mode`` is used. E.g.
        ``rigorous=Rigorous(confidence_level=0.68)``,
        ``precision=Precision(tau_grid=[...])``,
        ``transfer=Transfer(history_window=10)`` (or ``Transfer(history_window=10, w_data=w)`` for conditional TE),
        ``conditional=Conditional(w_data=w)``,
        ``interaction=Interaction(w_data=w)``,
        ``pairwise=Pairwise(pairs=[(0, 1), (0, 2)])``,
        ``sweep=Sweep(max_samples_per_task=1000)``.
    n_workers : int, default=1
        Worker processes for parallelizable modes.
    seed : int, optional
        Random seed (``random``/``numpy``/``torch``). **Reproducible at any
        ``n_workers``.** Each parallel task re-seeds inside its own worker from
        ``seed`` plus a deterministic per-task key
        (``analysis/task.py::run_training_task``), so which worker runs which
        task, and in what order, does not affect the result. Verified
        bit-identical between ``n_workers=1`` and ``n_workers=3`` for the shared
        task path, ``mode='dimensionality'``'s per-split dispatch and
        ``mode='pairwise'``'s per-pair dispatch.
    verbose, show_progress : bool
        Logging verbosity and progress bars.
    device : str, optional
        Compute device ('cpu'/'cuda'/'mps'); auto-detected if None.
    permutation_test : bool, default=False
        Run a label-permutation null test (supported modes only).
    n_permutations : int, default=1
        Number of permutations when ``permutation_test=True``.

    Returns
    -------
    neural_mi.results.Results

    Examples
    --------
    >>> import neural_mi as nmi
    >>> from neural_mi import Model, Training, Split, Processing, Rigorous
    >>> results = nmi.run(
    ...     x_raw, y_raw, mode='rigorous',
    ...     processing=Processing(x='continuous', x_params={'window_size': 1}),
    ...     model=Model(embedding_dim=16, hidden_dim=64),
    ...     training=Training(n_epochs=50, batch_size=128),
    ...     split=Split(mode='random'),
    ...     rigorous=Rigorous(confidence_level=0.68),
    ...     n_workers=4, seed=42,
    ... )
    """
    if _removed:
        raise TypeError(
            f"run() got unexpected keyword argument(s) {sorted(_removed)}. "
            f"Parameters are grouped into config objects: model=Model(...), "
            f"training=Training(...), split=Split(...), processing=Processing(...), "
            f"estimator=..., output=Output(...), and one per-mode config "
            f"(rigorous=/precision=/lag=/transfer=/dimensionality=/conditional=). "
            f"See help(neural_mi.run)."
        )

    # Coerce dict/str inputs to config instances.
    model = as_config(model, Model)
    training = as_config(training, Training)
    split = as_config(split, Split)
    output = as_config(output, Output)
    processing = as_config(processing, Processing)
    if isinstance(estimator, str):
        estimator = Estimator(name=estimator)
    else:
        estimator = as_config(estimator, Estimator)

    # Named engine parameters (computed once at import, see _ENGINE_PARAMS) decide
    # each lowered key's bucket: a named engine kwarg vs the base_params dict /
    # analysis_kwargs.
    _named = _ENGINE_PARAMS

    base_params: Dict[str, Any] = {}
    flat: Dict[str, Any] = {}
    analysis_kwargs: Dict[str, Any] = {}

    def _route_base(d):
        for k, v in d.items():
            (flat if k in _named else base_params)[k] = v

    def _route_analysis(d):
        for k, v in d.items():
            (flat if k in _named else analysis_kwargs)[k] = v

    if model is not None:
        _route_base(model.to_base_params())
    if training is not None:
        _route_base(training.to_base_params())
    if split is not None:
        _route_base(split.to_base_params())
    if output is not None:
        _route_base(output.to_base_params())
        flat.update(output.to_labels())
    if estimator is not None:
        if estimator.name is not None:
            flat['estimator'] = estimator.name
        if estimator.params is not None:
            flat['estimator_params'] = estimator.params
    if processing is not None:
        _route_analysis(processing.to_kwargs())

    # Mode-specific config: only the one matching `mode` is consulted.
    _provided = {'rigorous': rigorous, 'precision': precision, 'lag': lag,
                 'transfer': transfer, 'dimensionality': dimensionality,
                 'conditional': conditional, 'interaction': interaction,
                 'pairwise': pairwise, 'sweep': sweep}
    _stray = [name for name, cfg in _provided.items() if cfg is not None and name != mode]
    if _stray:
        warnings.warn(
            f"Mode config(s) {_stray} were provided but mode='{mode}'; they are ignored. "
            f"Only the config matching the active mode is used.",
            UserWarning, stacklevel=2,
        )
    if mode in _MODE_CONFIG_CLASSES:
        mode_cfg = as_config(_provided[mode], _MODE_CONFIG_CLASSES[mode])
        if mode_cfg is not None:
            if isinstance(mode_cfg, Transfer):
                flat.update(mode_cfg.to_w_kwargs())
                ak = mode_cfg.to_analysis_kwargs()
                if 'bidirectional' in ak:
                    flat['bidirectional_te'] = ak.pop('bidirectional')
                _route_analysis(ak)
            elif isinstance(mode_cfg, Conditional):
                flat.update(mode_cfg.to_w_kwargs())
                _route_analysis(mode_cfg.to_analysis_kwargs())
            elif isinstance(mode_cfg, Interaction):
                flat.update(mode_cfg.to_w_kwargs())
                _route_analysis(mode_cfg.to_analysis_kwargs())
            else:
                _route_analysis(mode_cfg.to_analysis_kwargs())

    # Runtime / dispatch args (always forwarded).
    flat['mode'] = mode
    flat['sweep_grid'] = sweep_grid
    flat['random_seed'] = seed
    flat['verbose'] = verbose
    flat['show_progress'] = show_progress
    flat['device'] = device
    flat['permutation_test'] = permutation_test
    flat['n_permutations'] = n_permutations
    flat['permutation_shuffle'] = permutation_shuffle
    analysis_kwargs['n_workers'] = n_workers

    if base_params:
        flat['base_params'] = base_params

    return _run_flat(x_data, y_data, **flat, **analysis_kwargs)


def _run_flat(
    x_data: Union[np.ndarray, torch.Tensor, List],
    y_data: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    x_time: Optional[np.ndarray] = None,
    y_time: Optional[np.ndarray] = None,
    mode: str = 'estimate',
    processor_type_x: Optional[str] = None,
    processor_params_x: Optional[Dict[str, Any]] = None,
    processor_type_y: Optional[str] = None,
    processor_params_y: Optional[Dict[str, Any]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    sweep_grid: Optional[Dict[str, list]] = None,
    output_units: str = 'bits',
    estimator: str = 'infonce',
    estimator_params: Optional[Dict[str, Any]] = None,
    custom_critic: Optional[torch.nn.Module] = None,
    custom_embedding_cls: Optional[type] = None,
    save_best_model_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
    device: Optional[str] = None,
    split_mode: str = 'blocked',
    train_fraction: float = 0.9,
    n_test_blocks: int = 5,
    split_gap_fraction: float = 0.5,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    curvature_t_threshold: float = 2.0,
    min_gamma_points: int = 5,
    confidence_level: float = 0.68,
    max_eval_samples: int = 5000,
    train_subset_size: Optional[int] = None,
    track_spectral_history: bool = False,
    max_index_reduction: float = 0.05,
    tau_grid: Optional[List[float]] = None,
    corrupt_target: str = 'x',
    corruption_method: str = 'rounding',
    n_noise_samples: int = 50,
    threshold_ratio: float = 0.9,
    permutation_test: bool = False,
    n_permutations: int = 1,
    permutation_shuffle: str = 'circular',
    history_window: Optional[int] = None,
    prediction_horizon: int = 1,
    bidirectional_te: bool = False,
    w_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    w_time: Optional[np.ndarray] = None,
    w_processor_type: Optional[str] = None,
    w_processor_params: Optional[Dict[str, Any]] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    shared_encoder: Optional[bool] = None,
    return_embeddings: bool = False,
    lag_range: Optional[List] = None,
    use_spectral_norm: bool = True,
    gradient_clip_val: Optional[float] = None,
    optimizer: Union[str, type] = 'adam',
    optimizer_params: Optional[Dict[str, Any]] = None,
    scheduler: Union[str, type, None] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    eval_train: Union[bool, float, int] = False,
    peak_fraction: float = 1.0,
    dropout: Optional[float] = None,
    norm_layer: Optional[str] = None,
    use_amp: Union[bool, str] = 'auto',
    track_embeddings: Optional[Union[bool, float, int, str]] = None,
    return_rotated_embeddings: Optional[bool] = None,
    rotated_embeddings_whitening: Optional[str] = None,
    rotated_embeddings_per_epoch: Optional[bool] = None,
    return_rotation_matrices: Optional[bool] = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    channel_names_x: Optional[List[str]] = None,
    channel_names_y: Optional[List[str]] = None,
    **analysis_kwargs
) -> Results:
    
    """Flat-kwarg engine behind :func:`run`; see that function's docstring for
    the public API and parameter semantics.
    """
    
    # Integrate run(verbose=) with the global logger for the duration of this call.
    # verbose=True → INFO level (informational messages shown)
    # verbose=False → WARNING level (only warnings and errors shown)
    import logging as _logging
    from .data.handler import reset_retention_warnings as _reset_retention
    _reset_retention()  # dedup is per run, not per process lifetime
    _prev_level = logger.level
    _prev_handler_levels = [h.level for h in logger.handlers]
    target_level = _logging.INFO if verbose else _logging.WARNING
    logger.setLevel(target_level)
    for h in logger.handlers:
        h.setLevel(target_level)
    try:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
        # No "reproducibility is not guaranteed with n_workers > 1" warning
        # here any more: it was false. run_training_task re-seeds random/numpy/
        # torch inside each worker from random_seed plus a deterministic
        # per-task key, which makes worker count and scheduling order
        # irrelevant. Measured bit-identical at n_workers=1 and 3 across the
        # shared task path, _dispatch_splits and _dispatch_pairs. The warning
        # was worse than noise: it pushed callers onto n_workers=1 to protect a
        # property they already had, which is a straight multiple on wall clock
        # for exactly the repeat-heavy runs (sweep_grid={'run_id': ...}) that
        # the amplification warning tells them to do.

        if base_params is None: base_params = {}
        # Copy so we never mutate the caller's dict across multiple calls
        base_params = dict(base_params)

        def _inject(bp: dict, key: str, val, source: str = "keyword argument") -> None:
            """Inject val into bp[key], warning if an existing value is overwritten."""
            if val is None:
                return
            if key in bp and bp[key] != val:
                logger.warning(
                    f"Parameter '{key}' is defined in base_params ({bp[key]!r}) but is "
                    f"being overridden by {source} value ({val!r}). The {source} value "
                    f"takes precedence. To silence this, remove '{key}' from base_params."
                )
            bp[key] = val

        # Populate base_params with explicit arguments to ensure they are validated
        # Time vectors travel with the params so that windowing deferred to the
        # task layer (shift_windows on conditional/interaction, where X and W are
        # merged and windowed later) sees the same real-time grid the eager path
        # gets via create_dataset(x_time=...). Without them a continuous X is
        # windowed in sample-index units while a spike Y is in seconds, and no
        # window can satisfy coverage: measured as 0 of 33080 windows retained.
        _inject(base_params, 'x_time', x_time)
        _inject(base_params, 'y_time', y_time)
        _inject(base_params, 'output_units', output_units)
        _inject(base_params, 'verbose', verbose)
        _inject(base_params, 'show_progress', show_progress)
        _inject(base_params, 'device', device)
        if 'device' not in base_params:
            base_params['device'] = get_device()
        _inject(base_params, 'estimator_name', estimator)
        # No `or {}` here: that would convert an un-passed (None) top-level kwarg
        # into a real value, defeating _inject's "leave base_params alone if not
        # explicitly given" guard and silently overwriting a caller-supplied
        # base_params['estimator_params'] with {}. apply_defaults() already backstops
        # the case where the key is absent from both.
        _inject(base_params, 'estimator_params', estimator_params)
        _inject(base_params, 'custom_critic', custom_critic)
        _inject(base_params, 'custom_embedding_cls', custom_embedding_cls)
        _inject(base_params, 'save_best_model_path', save_best_model_path)
        _inject(base_params, 'split_mode', split_mode)
        _inject(base_params, 'train_fraction', train_fraction)
        _inject(base_params, 'n_test_blocks', n_test_blocks)
        _inject(base_params, 'split_gap_fraction', split_gap_fraction)
        _inject(base_params, 'train_indices', train_indices)
        _inject(base_params, 'test_indices', test_indices)
        # Inject  Trainer pipeline arguments
        _inject(base_params, 'max_eval_samples', max_eval_samples)
        _inject(base_params, 'train_subset_size', train_subset_size)
        _inject(base_params, 'use_spectral_norm', use_spectral_norm)
        _inject(base_params, 'gradient_clip_val', gradient_clip_val)
        _inject(base_params, 'optimizer', optimizer)
        # See the estimator_params comment above -- same bug shape, same fix.
        _inject(base_params, 'optimizer_params', optimizer_params)
        _inject(base_params, 'scheduler', scheduler)
        _inject(base_params, 'scheduler_params', scheduler_params)
        _inject(base_params, 'eval_train', eval_train)
        _inject(base_params, 'peak_fraction', peak_fraction)
        _inject(base_params, 'dropout', dropout)
        _inject(base_params, 'norm_layer', norm_layer)
        _inject(base_params, 'use_amp', use_amp)

        _inject(base_params, 'track_spectral_history', track_spectral_history)
        _inject(base_params, 'max_index_reduction', max_index_reduction)

        _inject(base_params, 'processor_type_x', processor_type_x)
        _inject(base_params, 'processor_params_x', processor_params_x)
        _inject(base_params, 'processor_type_y', processor_type_y)
        _inject(base_params, 'processor_params_y', processor_params_y)
        if random_seed is not None:
            _inject(base_params, 'random_seed', random_seed)

        # Top-level shortcuts: inject into base_params
        _inject(base_params, 'n_epochs', n_epochs)
        _inject(base_params, 'batch_size', batch_size)
        _inject(base_params, 'shared_encoder', shared_encoder)
        if return_embeddings:
            base_params['return_embeddings'] = True
        _inject(base_params, 'track_embeddings', track_embeddings)
        _inject(base_params, 'return_rotated_embeddings', return_rotated_embeddings)
        _inject(base_params, 'rotated_embeddings_whitening', rotated_embeddings_whitening)
        _inject(base_params, 'rotated_embeddings_per_epoch', rotated_embeddings_per_epoch)
        _inject(base_params, 'return_rotation_matrices', return_rotation_matrices)

        if permutation_shuffle not in ('circular', 'block'):
            raise ValueError(
                f"permutation_shuffle must be 'circular' or 'block', got {permutation_shuffle!r}."
            )

        if permutation_test and n_permutations < 50:
            warnings.warn(
                f"permutation_test=True with n_permutations={n_permutations}. "
                f"This is insufficient to estimate a reliable p-value or null distribution. "
                f"Use n_permutations >= 100 for meaningful statistical inference.",
                UserWarning,
                stacklevel=2,
            )

        # Permutation test not supported for rigorous/precision modes
        if permutation_test and mode in ('rigorous', 'precision'):
            raise ValueError(
                f"permutation_test=True is not supported for mode='{mode}'. "
                f"This mode already produces an analytical error estimate. "
                f"Use mode='estimate', 'sweep', 'dimensionality', 'lag', "
                f"'conditional', 'interaction', or 'transfer' for permutation testing."
            )

        # Verify conditional-MI / interaction-information / conditional-TE input.
        # w_data is the shared "third variable" slot for all three modes
        # (Conditional/Interaction/Transfer all name it identically, see
        # config.py) -- one variable, three possible roles, dispatched by mode.
        if w_data is not None and mode not in ('conditional', 'interaction', 'transfer'):
            logger.warning(
                f"w_data was provided but mode='{mode}' does not use it. "
                f"w_data is only consumed by mode='conditional' (conditional MI), "
                f"mode='interaction' (interaction information), and mode='transfer' "
                f"(conditional transfer entropy)."
            )

        # Validate parameters and apply defaults to base_params
        _pre_default_keys = set(base_params.keys())
        param_validator = ParameterValidator(locals())
        param_validator.validate()
        param_validator.apply_defaults()

        # Warn about n_layers/hidden_dim list mismatch only when the user explicitly
        # set n_layers (i.e., it was in base_params before defaults were applied).
        _hd = base_params.get('hidden_dim')
        if isinstance(_hd, list) and 'n_layers' in _pre_default_keys:
            _nl = base_params.get('n_layers')
            if _nl != len(_hd):
                warnings.warn(
                    f"hidden_dim is a list of length {len(_hd)}, so n_layers={_nl} is "
                    f"ignored. The network will have {len(_hd)} hidden layer(s).",
                    UserWarning, stacklevel=3,
                )

        DataValidator(x_data, y_data, processor_type_x, processor_type_y).validate()
    
        _processor = base_params.get('processor_type_x', None)
        _embedding = base_params.get('embedding_model', 'mlp')
        # A 3-D array/tensor passed with processor_type=None is already
        # pre-windowed (N, C, W) sequential data, not a StaticDataset -- this
        # matches the same auto-detection ParameterSweep uses (`is_proc_sweep`)
        # to allow 'gru'/'lstm' on pre-processed data without re-running a
        # processor.
        _has_time_dim = hasattr(x_data, 'ndim') and x_data.ndim == 3
        # mode='transfer' builds its own (N, C, history_window) arrays from
        # raw 2-D (T, n_channels) input internally, via unfold
        # (analysis/transfer.py's _build_te_arrays) -- raw 2-D input there is
        # the intended, documented shape, not a mistake this check should
        # catch. Everywhere else (mode='estimate'/'conditional'/'interaction'/
        # etc.), the caller is expected to have already windowed the data
        # themselves before it reaches this validation.
        _mode_builds_own_windows = mode == 'transfer'
        if (_processor is None and str(_embedding).lower() in ('gru', 'lstm')
                and not _has_time_dim and not _mode_builds_own_windows):
            raise ValueError(
                f"embedding_model='{_embedding}' requires sequential input but "
                f"processor_type=None produces a StaticDataset with no time dimension. "
                f"Either set processor_type to a windowed processor (e.g. 'continuous_window', "
                f"'spike_window') or switch embedding_model to 'mlp' / 'linear'."
            )
    
        run_params = {"mode": mode, "processor_type_x": processor_type_x, "processor_params_x": processor_params_x,
                      "processor_type_y": processor_type_y, "processor_params_y": processor_params_y,
                      "base_params": base_params, "sweep_grid": sweep_grid, "output_units": output_units,
                      "estimator": estimator, "random_seed": random_seed, "curvature_t_threshold": curvature_t_threshold,
                      "min_gamma_points": min_gamma_points, "confidence_level": confidence_level,
                      **analysis_kwargs}
        if x_name is not None: run_params['x_name'] = x_name
        if y_name is not None: run_params['y_name'] = y_name
        if channel_names_x is not None: run_params['channel_names_x'] = channel_names_x
        if channel_names_y is not None: run_params['channel_names_y'] = channel_names_y

        # Build the complete set of processor-level keys from the schema so that
        # any schema addition automatically triggers the deferred-processing path.
        processor_param_keys = set().union(*PROCESSOR_PARAMS_SCHEMA.values())
        is_proc_sweep = mode == 'sweep' and any(key in (sweep_grid or {}) for key in processor_param_keys)
    
        def _to_tensor(arr):
            """Convert array-like to a float32 tensor; expand 2-D (N, C) to (N, C, 1)."""
            if torch.is_tensor(arr):
                t = arr.float()
            else:
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            if t.ndim == 2:
                t = t.unsqueeze(-1)
            return t

        # Both shift_windows (neural_mi/data/shift_windowing.py) and a
        # reachability extension for shift_time need the raw,
        # unwindowed arrays to survive to task.py::run_training_task -- same
        # "defer, don't window here" treatment as is_proc_sweep/mode='lag'.
        # See _SHIFT_SAFE_MODES/_SHIFT_WINDOWS_SAFE_MODES (module scope) for
        # which modes qualify for which mechanism and why. 'transfer' needs
        # real additional orchestration (past/future construction) and is
        # handled separately, with its own gating (not attempted this pass).
        # processor_type_y=None means "inherit X's type" (create_dataset's
        # own convention, handler.py: proc_type_y = processor_type_y or
        # processor_type_x).
        _effective_processor_type_y = processor_type_y if processor_type_y is not None else processor_type_x
        _shift_pair_family = shift_family(processor_type_x, _effective_processor_type_y)
        # shift_windows: the cheap reslice mechanism, for the 'regular'
        # family (continuous/categorical, either side, need not match).
        _defer_for_shift_windows = (mode in _SHIFT_WINDOWS_SAFE_MODES and base_params.get('shift_windows')
                                    and _shift_pair_family == 'regular')
        # shift_time: the general PairedTemporalDataset/time_shift
        # mechanism, reachable for 'spike' pairs (no cross-unit concerns,
        # both sides natively in seconds) and 'mixed' pairs *only* when the
        # regular-grid side has 'sample_rate' set (so a shift value means
        # the same real time on both sides -- otherwise the pairing's own
        # window alignment is already questionable, see
        # NEURALMI_REFERENCE.md). Deliberately excludes 'regular' pairs --
        # those already have the strictly better shift_windows. 'rigorous'
        # reaches only the 'spike' sub-case (_SHIFT_TIME_RIGOROUS_SAFE_MODES);
        # its 'mixed'-pair chunk translation isn't attempted this pass (see
        # the comment at _SHIFT_TIME_RIGOROUS_SAFE_MODES's definition).
        _defer_for_shift_time = (
            base_params.get('shift_time')
            and (
                (_shift_pair_family == 'spike' and mode in _SHIFT_TIME_RIGOROUS_SAFE_MODES)
                or (_shift_pair_family == 'mixed' and mode in _SHIFT_SAFE_MODES
                    and mixed_pair_sample_rate_ok(
                        processor_type_x, processor_params_x,
                        _effective_processor_type_y, processor_params_y))
            )
        )
        # 'conditional'/'interaction': shift_windows reachable when X and
        # the conditioning variable (w_data, shared by both modes) are both in
        # {'continuous', 'categorical'} -- any combination, including mixed
        # (X continuous + W categorical or vice versa), not just matching
        # types. A categorical side is relabeled and given its own
        # n_categories via conditional.py/interaction.py's raw_deferred
        # branch + shift_windowing.make_multi_categorical_encoder (which
        # passes a continuous block through unencoded and broadcasts a
        # categorical block's collapsed window axis up to match it), so
        # each side's channels are always encoded correctly regardless of
        # whether the other side matches its type. shift_time is similarly
        # reachable for a spike+spike pair specifically (matching family
        # only -- a mixed spike + regular-grid conditioning variable has no
        # raw sample axis to concatenate against and remains out of scope,
        # same as the plain X/Y case's own 'mixed' family exclusion from
        # this raw-concat mechanism). Both gated the same way: (for
        # conditional) align != 'dual_branch'. See
        # conditional.py/interaction.py's `raw_deferred` path for how the
        # raw concat + deferred windowing actually happens. Includes the
        # rigorous=True sub-path too -- run_rigorous_scalar_analysis's own
        # chunk-boundary translation (see its _is_raw_deferred/
        # _is_spike_deferred handling) mirrors AnalysisWorkflow._prepare_tasks's,
        # so the raw-concat scenario is now covered there as well.
        # W inherits X's processor when it declares none of its own.
        #
        # Both modes treat a w_processor_type of None as "W is already
        # processed" and hand it through untouched. That is right when the
        # caller really did pass a 3-D windowed W, and wrong whenever X is
        # being processed here: W is then a raw 2-D array that needs exactly
        # the treatment X is getting, and leaving it alone produces
        # (n_windows, C, w) against (T, C, 1) -- a shape error at best, and at
        # window_size=1 a pair of mismatches small enough for
        # interaction.py's trim tolerances to absorb, so the call returns a
        # number built from an unwindowed W.
        #
        # Inheriting is what the library already documents:
        # run_interaction_information's `w_processor_type` parameter says
        # "None (default) inherits X's own type", a promise its raw_deferred
        # branch keeps and this one did not. Resolving here rather than at
        # each use site also means _cond_var_type below sees the inherited
        # type, so the natural call lands on the same deferred, already-tested
        # route an explicit w_processor_type would have reached.
        #
        # Narrow by construction: only when W exists, declares nothing, X
        # declares something, and W is not already windowed. A pre-processed
        # 3-D W and the no-processor fast path are both untouched.
        if (mode in ('conditional', 'interaction') and w_data is not None
                and w_processor_type is None and processor_type_x is not None
                and getattr(w_data, 'ndim', None) != 3):
            w_processor_type = processor_type_x
            if w_processor_params is None:
                w_processor_params = processor_params_x
            logger.info(
                f"mode='{mode}': w_data has no processor type of its own, "
                f"inheriting X's ('{processor_type_x}') so W is windowed on the "
                f"same grid. Pass w_processor_type explicitly to override."
            )
        _cond_var_type = w_processor_type if mode in ('conditional', 'interaction') else None
        _regular_types = ('continuous', 'categorical')
        _not_dual_branch = (mode != 'conditional' or analysis_kwargs.get('align') != 'dual_branch')
        _defer_regular_conditional_interaction = (
            mode in ('conditional', 'interaction') and _not_dual_branch
            and base_params.get('shift_windows')
            and processor_type_x in _regular_types and _cond_var_type in _regular_types
        )
        # Unlike the regular-grid case above, this is NOT gated on
        # base_params.get('shift_time') -- it's a correctness requirement,
        # not an optional shift-reachability path. Windowing W separately
        # (even paired with Y) only guarantees "W has data AND Y has data",
        # which is a *different* random subset of windows than X's own
        # "X has data AND Y has data" whenever spike coverage is patchy
        # (confirmed empirically: two independently-drawn spike populations
        # sharing a Y can easily diverge by dozens of windows, well past
        # _SAMPLE_COUNT_TRIM_TOLERANCE). Merging X and W into one combined
        # population *before* windowing (below) guarantees both share
        # exactly the same window-validity decision, since it's now one
        # array being windowed once -- always correct, so always applied
        # for a spike+spike pair regardless of shift_time.
        _defer_spike_conditional_interaction = (
            mode in ('conditional', 'interaction') and _not_dual_branch
            and processor_type_x == 'spike' and _cond_var_type == 'spike'
        )
        _defer_for_conditional_interaction = (
            _defer_regular_conditional_interaction or _defer_spike_conditional_interaction
        )
        if _defer_for_conditional_interaction:
            # Companion correctness fix (applies to the already-shipped
            # continuous case too, not just categorical): the raw-deferred
            # path below windows the concatenated array using only
            # processor_params_x's window_size/step_size -- the
            # conditioning variable's own w_processor_params window_size/
            # step_size, if explicitly set to a *different* value, would
            # otherwise be silently ignored rather than validated.
            _cond_processor_params = w_processor_params
            _cond_var_label = 'w_processor_params'
            _x_window_size = (processor_params_x or {}).get('window_size')
            _x_step_size = (processor_params_x or {}).get('step_size')
            _cond_window_size = (_cond_processor_params or {}).get('window_size')
            _cond_step_size = (_cond_processor_params or {}).get('step_size')
            if _cond_window_size is not None and _cond_window_size != _x_window_size:
                raise ValueError(
                    f"shift_windows=True with mode='{mode}': {_cond_var_label}['window_size']="
                    f"{_cond_window_size} differs from processor_params_x['window_size']="
                    f"{_x_window_size}. The conditioning variable is concatenated onto X "
                    f"*before* windowing and shares X's window grid exactly, so both must "
                    f"use the same window_size. Remove window_size from {_cond_var_label} "
                    f"to inherit X's, or set them equal explicitly."
                )
            if _cond_step_size is not None and _cond_step_size != _x_step_size:
                raise ValueError(
                    f"shift_windows=True with mode='{mode}': {_cond_var_label}['step_size']="
                    f"{_cond_step_size} differs from processor_params_x['step_size']="
                    f"{_x_step_size}. The conditioning variable is concatenated onto X "
                    f"*before* windowing and shares X's window grid exactly, so both must "
                    f"use the same step_size. Remove step_size from {_cond_var_label} to "
                    f"inherit X's, or set them equal explicitly."
                )
            # Same class of silent-ignore as the two above: the concatenated
            # array is windowed on X's time vector, so a distinct w_time would
            # be dropped and W would be read on X's grid, giving a wrong answer
            # with nothing said. Equal vectors are the ordinary case (both
            # variables sampled together) and stay silent.
            if w_time is not None and x_time is not None:
                _wt, _xt = np.asarray(w_time), np.asarray(x_time)
                if _wt.shape != _xt.shape or not np.allclose(_wt, _xt):
                    raise ValueError(
                        f"shift_windows=True with mode='{mode}': w_time differs from "
                        f"x_time. The conditioning variable is concatenated onto X "
                        f"*before* windowing and is read on X's time grid exactly, so a "
                        f"separate w_time cannot be honoured here and would be silently "
                        f"ignored. Either resample W onto X's grid and drop w_time, or "
                        f"set shift_windows=False, which windows W separately and does "
                        f"honour w_time."
                    )
        # align='dual_branch': shift_windows reachable via a genuinely
        # different mechanism than the concat-based one above -- X and the
        # conditioning variable (C, still configured via
        # w_processor_type/w_processor_params, reused as-is) are never
        # concatenated for dual_branch (that's its entire premise: C keeps
        # its own, generally different, window geometry), so none of the
        # matching-window-size validation above applies here. Instead this
        # reuses PairedWindowShifter's own already-proven pattern (two
        # independently-shaped sides shifted in sync) via a new 3-way
        # DualBranchWindowShifter (X, C, Y) -- see
        # shift_windowing.try_build_shift_windows_dataset_dual_branch.
        _defer_for_dual_branch_shift_windows = (
            mode == 'conditional' and analysis_kwargs.get('align') == 'dual_branch'
            and base_params.get('shift_windows')
            and processor_type_x in _regular_types and _cond_var_type in _regular_types
            # Y's type too (unlike the concat-based gate above, which never
            # needed to check it since it never builds a dataset itself) --
            # try_build_shift_windows_dataset_dual_branch requires all three
            # sides in the regular-grid family, and returning None there
            # would otherwise fall back to create_dataset with a *tuple*
            # x_data, which it doesn't support (a crash, not a graceful
            # no-op) if only X and C, not Y, were checked here.
            and _effective_processor_type_y in _regular_types
        )
        # Set only on the windowing branch below; None everywhere else (the
        # pre-processed fast path and every deferred path, none of which build a
        # window grid here).
        _xy_window_times = None
        if (is_proc_sweep or mode == 'lag' or _defer_for_shift_windows or _defer_for_shift_time
                or _defer_for_conditional_interaction or _defer_for_dual_branch_shift_windows):
            logger.info("Detected sweep over processor or lag parameters. Deferring data processing to workers.")
            x_run_data, y_run_data = x_data, y_data
        elif processor_type_x is None and processor_type_y is None:
            # Fast path: data is already pre-processed. Convert to tensors inline and skip
            # the full create_dataset / PairedDataset allocation.
            x_run_data = _to_tensor(x_data)
            y_run_data = _to_tensor(y_data) if y_data is not None else None
            if y_run_data is not None and x_run_data.shape[0] != y_run_data.shape[0]:
                _min_n = min(x_run_data.shape[0], y_run_data.shape[0])
                logger.warning(
                    f"X ({x_run_data.shape[0]}) and Y ({y_run_data.shape[0]}) differ in sample count; "
                    f"truncating both to {_min_n}."
                )
                x_run_data = x_run_data[:_min_n]
                y_run_data = y_run_data[:_min_n]
            base_params['processor_type_x'] = None
            base_params['processor_type_y'] = None
            if base_params.get('processor_params_x') is None:
                base_params['processor_params_x'] = {}
            if base_params.get('processor_params_y') is None:
                base_params['processor_params_y'] = {}
            base_params['processor_params_x']['preprocessed'] = True
            base_params['processor_params_y']['preprocessed'] = True
            n_samples = x_run_data.shape[0]
            if n_samples < 200:
                warnings.warn(
                    f"Very few samples detected ({n_samples} samples). "
                    f"Neural MI estimators are prone to overfitting at this scale. "
                    f"Consider adding regularisation (Model(dropout=..., norm_layer=...)).",
                    UserWarning, stacklevel=4,
                )
            if mode not in ('dimensionality', 'pairwise') and y_run_data is None:
                raise ValueError(f"y_data must be provided for mode '{mode}'.")
        else:
            dataset = create_dataset(
                x_data=x_data,
                y_data=y_data if (mode != 'dimensionality' or y_data is not None) else None,
                x_time=x_time,
                y_time=y_time,
                processor_type_x=processor_type_x,
                processor_params_x=processor_params_x,
                processor_type_y=processor_type_y,
                processor_params_y=processor_params_y
            )

            base_params['processor_type_x'] = None
            base_params['processor_type_y'] = None

            if base_params.get('processor_params_x') is None: base_params['processor_params_x'] = {}
            if base_params.get('processor_params_y') is None: base_params['processor_params_y'] = {}
            base_params['processor_params_x']['preprocessed'] = True
            base_params['processor_params_y']['preprocessed'] = True

            # Windowing happens once, here -- the dataset that reaches the
            # Trainer downstream is a plain, already-windowed PairedDataset
            # with no window_manager of its own (processor_type_x/y were just
            # wiped to None above). Capture the window geometry now, while
            # dataset.window_manager is still live, so the blocked-split
            # leakage check has something to validate against.
            _wm = getattr(dataset, 'window_manager', None)
            if _wm is not None:
                # Kept for mode='conditional'/'interaction', which build W in a
                # separate create_dataset call and must line its windows up with
                # these before the two are concatenated channel-wise. See
                # _align_conditioning_windows.
                _xy_window_times = getattr(_wm, 'window_times', None)
                base_params['leak_check_window_size'] = _wm.window_size
                base_params['leak_check_step'] = _wm.resolve_step()

            # Retention is reported per task (see analysis/task.py), since it
            # varies between tasks and one run-level scalar would misdescribe
            # every row but one on a sweep. When windowing happens here the
            # tasks receive already-windowed tensors and never see this
            # dataset, so hand the value down through base_params for them to
            # report. Tasks that window for themselves prefer their own.
            if getattr(dataset, 'window_retention', None) is not None:
                base_params['_window_retention'] = dataset.window_retention
                base_params['_n_windows_built'] = dataset.n_windows_built
                base_params['_n_windows_retained'] = dataset.n_windows_retained

            _warn_small_sample(dataset, base_params)

            if mode in ('dimensionality', 'pairwise'):
                # dimensionality and pairwise can operate on x_data alone
                x_run_data = dataset.x_data
                y_run_data = dataset.y_data if y_data is not None else None
            else:
                if y_data is None: raise ValueError(f"y_data must be provided for mode '{mode}'.")
                x_run_data = dataset.x_data
                y_run_data = dataset.y_data

        _warn_if_shift_time_dead(base_params, mode, is_proc_sweep, processor_type_x,
                                 processor_params_x, _effective_processor_type_y, processor_params_y,
                                 user_set_keys=_pre_default_keys,
                                 extra_reachable=_defer_spike_conditional_interaction)
        _warn_if_shift_windows_dead(base_params, mode, processor_type_x, _effective_processor_type_y,
                                    user_set_keys=_pre_default_keys,
                                    extra_reachable=(_defer_regular_conditional_interaction
                                                    or _defer_for_dual_branch_shift_windows))

        from .analysis.sweep import ParameterSweep
        if mode == 'sweep':
            results_list = ParameterSweep(x_run_data, y_run_data, base_params).run(
                sweep_grid, is_proc_sweep=is_proc_sweep, **analysis_kwargs
            )
            # Strip embedding arrays from raw results before building the DataFrame.
            # In sweep mode every sweep config trains a different model and produces
            # its own embedding array; storing 2-D numpy arrays as DataFrame columns
            # would corrupt aggregation.  The embeddings from the last result are
            # surfaced in result.details instead.
            _sweep_embeddings = None
            for _r in reversed(results_list):
                if 'embeddings_x' in _r:
                    _sweep_embeddings = {'embeddings_x': _r.pop('embeddings_x'),
                                         'embeddings_y': _r.pop('embeddings_y', None)}
                    break
            for _r in results_list:
                _r.pop('embeddings_x', None)
                _r.pop('embeddings_y', None)
            df = pd.DataFrame(results_list)
            df = _convert_mi_units(df, output_units == 'bits')
            group_vars = [key for key in sweep_grid.keys() if key != 'run_id']
            agg_df = _hashable_group_vars(df, group_vars).groupby(group_vars)['train_mi'].agg(
                ['mean', 'std']).reset_index().rename(
                columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0) if group_vars else df
            primary_sweep_var = group_vars[0] if group_vars else None
            result = Results(mode=mode,
                             dataframe=agg_df,
                             params={**run_params, 'sweep_var': primary_sweep_var,
                                     'sweep_group_vars': group_vars},
                             details={'raw_results': df})
            if _sweep_embeddings is not None:
                result.details.update(_sweep_embeddings)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs, permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'estimate':
            results_list = ParameterSweep(x_run_data, y_run_data, base_params).run(
                sweep_grid or {}, **analysis_kwargs)
            if not results_list:
                return Results(mode=mode, mi_estimate=float('nan'), params=run_params)
            res_dict = results_list[0].copy()
            to_bits = output_units == 'bits'
            NATS_TO_BITS = 1 / np.log(2)

            # Report the train MI evaluated at the best-generalising checkpoint.
            # Model selection used test MI; if all test-MI values were non-positive,
            # the Trainer already zeroes train_mi — preserve that guard explicitly.
            mi = res_dict.pop('train_mi', float('nan'))
            if res_dict.get('all_mi_negative'):
                mi = 0.0
            mi = _convert_mi_units(mi, to_bits)

            # Keep test_mi, raw_train_mi, and train_mi_at_peak in details, converting units
            for _key in ('test_mi', 'raw_train_mi', 'train_mi_at_peak'):
                if _key in res_dict and isinstance(res_dict[_key], (int, float)):
                    res_dict[_key] = res_dict[_key] * NATS_TO_BITS if to_bits else res_dict[_key]

            # History lists: one shared implementation, so the estimate path
            # and the sweep-family paths cannot drift apart on units again.
            res_dict = _convert_mi_units(res_dict, to_bits)

            result = Results(mode=mode,
                             mi_estimate=mi,
                             params=run_params,
                             details=res_dict)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs, permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'dimensionality':
            df, _dim_embeddings = run_dimensionality_analysis(
                x_run_data, base_params, y_data=y_run_data,
                sweep_grid=sweep_grid,
                processor_type_x=processor_type_x, processor_type_y=processor_type_y,
                user_set_keys=_pre_default_keys,
                **analysis_kwargs)
            df = _convert_mi_units(df, output_units == 'bits')
            group_vars = [key for key in (sweep_grid or {}).keys() if key != 'run_id']
            metrics = ['train_mi', 'pr_eig', 'pr_singular']
            valid_metrics = [m for m in metrics if m in df.columns]
            if group_vars:
                agg_df = _hashable_group_vars(df, group_vars).groupby(group_vars)[valid_metrics].agg(
                    ['mean', 'std']).reset_index()
                agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns.values]
                rename_map = {f'{m}_mean': 'mi_mean' if m == 'train_mi' else f'{m}_mean' for m in valid_metrics}
                rename_map.update({f'{m}_std': 'mi_std' if m == 'train_mi' else f'{m}_std' for m in valid_metrics})
                agg_df = agg_df.rename(columns=rename_map).fillna(0)
            else:
                agg_data = {f'{m}_mean': df[m].mean() for m in valid_metrics}
                agg_data.update({f'{m}_std': df[m].std() for m in valid_metrics})
                if 'train_mi_mean' in agg_data:
                    agg_data['mi_mean'] = agg_data.pop('train_mi_mean')
                if 'train_mi_std' in agg_data:
                    agg_data['mi_std'] = agg_data.pop('train_mi_std')
                agg_df = pd.DataFrame([agg_data])
            result = Results(mode=mode, dataframe=agg_df, params={**run_params},
                             details={'raw_results': df})
            if _dim_embeddings is not None:
                result.details.update(_dim_embeddings)
            if permutation_test and y_run_data is not None:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs, permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'precision':
            if tau_grid is None:
                raise ValueError("`tau_grid` must be provided for mode='precision'.")
            prec_results = run_precision_analysis(
                x_run_data, y_run_data, base_params, tau_grid=tau_grid,
                corrupt_target=corrupt_target, corruption_method=corruption_method,
                n_noise_samples=n_noise_samples, threshold_ratio=threshold_ratio,
                **analysis_kwargs
            )
            df = prec_results['dataframe']
            df = _convert_mi_units(df, output_units == 'bits')
            details = prec_results['details']
            details['baseline_mi'] = _convert_mi_units(details['baseline_mi'], output_units == 'bits')
            details['threshold_value'] = _convert_mi_units(details['threshold_value'], output_units == 'bits')
            # Convert threshold_value inside each entry of the precision_thresholds dict
            if 'precision_thresholds' in details:
                for _ratio_dict in details['precision_thresholds'].values():
                    if 'threshold_value' in _ratio_dict and _ratio_dict['threshold_value'] is not None:
                        _ratio_dict['threshold_value'] = _convert_mi_units(
                            _ratio_dict['threshold_value'], output_units == 'bits'
                        )
            details['raw_results'] = df
            return Results(
                mode=mode,
                mi_estimate=details['baseline_mi'],  # baseline MI at zero corruption; precision_tau is in details
                dataframe=df,
                params={**run_params, 'tau_grid': tau_grid},
                details=details
            )

        elif mode == 'rigorous':
            analysis_kwargs.update({'curvature_t_threshold': curvature_t_threshold,
                                     'min_gamma_points': min_gamma_points,
                                     'confidence_level': confidence_level})
            results = run_rigorous_analysis(
                x_run_data, y_run_data, base_params,
                sweep_grid=sweep_grid, **analysis_kwargs)
            results = _convert_mi_units(results, output_units == 'bits')
            corrected_list = results.get('corrected_results', [])
            details = corrected_list[0] if corrected_list else {}
            return Results(mode=mode, mi_estimate=details.get('mi_corrected'),
                           dataframe=results.get('raw_results_df'), details=details,
                           params=run_params)

        elif mode == 'lag':
            # `lag_range` reaches _run_flat already unpacked from Lag(...) by
            # run(); the analysis_kwargs fallback covers direct _run_flat callers.
            lag_range_val = lag_range if lag_range is not None else analysis_kwargs.pop('lag_range', None)
            if lag_range_val is None:
                raise ValueError(
                    "`lag_range` must be provided for mode='lag'. "
                    "Pass it in the per-mode config: "
                    "nmi.run(..., mode='lag', lag=Lag(lag_range=range(-10, 11))). "
                    "A bare lag_range=... keyword is rejected by run()."
                )
            results_list = run_lag_analysis(x_run_data, y_run_data, base_params,
                                            lag_range=lag_range_val, sweep_grid=sweep_grid,
                                            **analysis_kwargs)
            df = pd.DataFrame(results_list)
            # Convert before aggregating, exactly as mode='sweep' does, so that
            # `dataframe` and `details['raw_results']` are in the same units.
            # Converting only the aggregate left raw_results in nats while the
            # dataframe was in bits -- the same numbers 1.443x apart.
            df = _convert_mi_units(df, output_units == 'bits')
            group_vars = ['lag']
            if sweep_grid:
                group_vars.extend([key for key in sweep_grid.keys() if key != 'run_id'])
            valid_group_vars = [var for var in group_vars if var in df.columns]
            if valid_group_vars:
                agg_df = _hashable_group_vars(df, valid_group_vars).groupby(valid_group_vars)['train_mi'].agg(
                    ['mean', 'std']).reset_index().rename(
                    columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0)
            else:
                # copy() so `dataframe` and details['raw_results'] cannot alias
                # the same object and have a caller's edit to one show up in the other.
                agg_df = df.copy()
            result = Results(mode=mode,
                             dataframe=agg_df,
                             params={**run_params, 'sweep_var': 'lag',
                                     'sweep_group_vars': valid_group_vars or group_vars},
                             details={'raw_results': df})
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs, lag_range=lag_range_val,
                    permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'conditional':
            if w_data is None:
                raise ValueError("`w_data` must be provided for mode='conditional'.")
            if analysis_kwargs.get('align') == 'dual_branch' and permutation_test:
                raise NotImplementedError(
                    "permutation_test=True is not supported with "
                    "Conditional(align='dual_branch'). The permutation baseline "
                    "would need the same dual-branch construction _run_single_permutation "
                    "doesn't have wired up; not needed for this pass."
                )
            # Process w_data if a processor type is given; otherwise assume pre-processed.
            # When _defer_for_conditional_interaction (or its dual_branch
            # sibling), keep w_data raw too -- same "defer, don't window
            # here" treatment x_run_data/y_run_data already got above -- so
            # run_conditional_mi's raw_deferred path can concatenate raw X
            # and raw W before windowing (or, for dual_branch, keep them as
            # a raw tuple, never concatenated). Left exactly as passed in
            # (not even converted to a tensor here, matching
            # x_run_data/y_run_data's own treatment on this path) --
            # run_conditional_mi's raw_deferred branch converts it.
            if _defer_for_conditional_interaction or _defer_for_dual_branch_shift_windows:
                w_run_data = w_data
            elif w_processor_type is not None:
                from .data.handler import create_dataset as _cds
                # Paired with Y (not built alone) so W's window-validity
                # criterion is "W has data AND Y has data" -- the same
                # invariant X's own windows (built above) already enforce.
                # Windowing W alone would only require "W has data",
                # silently admitting windows X/Y's own pairing rejects; for
                # patchy coverage (e.g. spike data) that gap can be large
                # enough to blow the trim tolerance in
                # analysis/conditional.py's sample-count check.
                w_dataset = _cds(
                    x_data=w_data, y_data=y_data,
                    x_time=w_time, y_time=y_time,
                    processor_type_x=w_processor_type,
                    processor_params_x=w_processor_params or {},
                    processor_type_y=_effective_processor_type_y,
                    processor_params_y=processor_params_y or {},
                )
                w_run_data = w_dataset.x_data
                if w_processor_type == 'categorical':
                    w_run_data = _reshape_categorical_w_for_conditional(
                        w_run_data, w_dataset.x_dataset
                    )
                x_run_data, y_run_data, w_run_data = _align_conditioning_windows(
                    mode, x_run_data, y_run_data, w_run_data,
                    _xy_window_times, w_dataset, base_params)
            else:
                w_run_data = w_data if torch.is_tensor(w_data) else torch.from_numpy(np.array(w_data)).float()
            n_workers = analysis_kwargs.get('n_workers', 1)
            use_rigorous = analysis_kwargs.pop('rigorous', False)
            _align = analysis_kwargs.pop('align', None)
            if use_rigorous and _defer_for_dual_branch_shift_windows:
                raise NotImplementedError(
                    "rigorous=True is not supported together with "
                    "Conditional(align='dual_branch') and shift_windows=True. "
                    "run_rigorous_scalar_analysis's chunk-to-raw-range translation "
                    "computes one raw-sample range from X's own window_size and "
                    "applies it uniformly to every extra_data array -- since "
                    "dual_branch's C genuinely has its own, different window "
                    "geometry, reusing X's chunk boundaries for C would silently "
                    "misalign it per gamma-chunk rather than just being unsupported. "
                    "Pass shift_windows=False (or drop rigorous=True) to proceed."
                )
            if use_rigorous:
                from .analysis.rigorous import run_rigorous_scalar_analysis
                from .analysis.conditional import _cmi_rigorous_scalar
                _gamma_range = analysis_kwargs.pop('gamma_range', None) or range(1, 11)
                _rig_kwargs = {
                    'gamma_range': _gamma_range,
                    'curvature_t_threshold': analysis_kwargs.pop('curvature_t_threshold', curvature_t_threshold),
                    'min_gamma_points': analysis_kwargs.pop('min_gamma_points', min_gamma_points),
                    'confidence_level': analysis_kwargs.pop('confidence_level', confidence_level),
                    'residual_threshold': analysis_kwargs.pop('residual_threshold', 2.5),
                    'r2_threshold': analysis_kwargs.pop('r2_threshold', 0.90),
                    'leverage_threshold': analysis_kwargs.pop('leverage_threshold', 0.20),
                }
                # Parallelised across gamma-chunk tasks (like plain mode='rigorous');
                # each individual chunk's CMI call runs with n_workers=1 internally
                # (see _cmi_rigorous_scalar) to avoid nested multiprocessing pools.
                # w_run_data is passed as both 'w_data' and 'c_data' -- x_data
                # stays a plain tensor throughout this call (align='dual_branch'
                # never touches rigorous.py's own chunking), _cmi_rigorous_scalar
                # picks whichever of the two it needs based on 'align' and
                # assembles the tuple only at that boundary.
                rig_details = run_rigorous_scalar_analysis(
                    scalar_fn=_cmi_rigorous_scalar,
                    x_data=x_run_data, y_data=y_run_data, base_params=base_params,
                    extra_data={'w_data': w_run_data, 'c_data': w_run_data},
                    extra_kwargs={'sweep_grid': sweep_grid, 'align': _align,
                                 'raw_deferred': _defer_for_conditional_interaction,
                                 'w_processor_type': w_processor_type},
                    n_workers=n_workers,
                    raw_deferred=_defer_for_conditional_interaction,
                    **_rig_kwargs,
                )
                # _convert_mi_units already recurses into rig_details['raw_results_df']
                # (dict branch, 'raw_results_df' key) -- popping it AFTER this call and
                # converting it again would double-scale its 'train_mi' column.
                rig_details = _convert_mi_units(rig_details, output_units == 'bits')
                raw_df = rig_details.pop('raw_results_df', pd.DataFrame())
                return Results(
                    mode=mode,
                    mi_estimate=rig_details.get('mi_corrected'),
                    dataframe=raw_df,
                    params={**run_params, 'rigorous': True},
                    details=rig_details,
                )
            # Standard (non-rigorous) path. w_run_data is passed as both
            # w_data and c_data -- run_conditional_mi uses whichever it
            # needs based on align. c_processor_type/c_processor_params are
            # only read by the align='dual_branch' + raw_deferred sub-path
            # (harmless to always pass -- w_processor_type/w_processor_params
            # already are C's config by convention, see the dual_branch gate
            # above).
            raw = run_conditional_mi(x_run_data, y_run_data, w_run_data, base_params,
                                     sweep_grid=sweep_grid, n_workers=n_workers,
                                     align=_align, c_data=w_run_data,
                                     raw_deferred=_defer_for_conditional_interaction
                                                 or _defer_for_dual_branch_shift_windows,
                                     w_processor_type=w_processor_type,
                                     c_processor_type=w_processor_type,
                                     c_processor_params=w_processor_params)
            raw = _convert_mi_units(raw, output_units == 'bits')
            cmi = raw['cmi_estimate']
            result = Results(mode=mode, mi_estimate=cmi, params=run_params, details=raw)
            if permutation_test:
                # x_run_data/y_run_data/w_run_data are raw (unwindowed) here
                # whenever _defer_for_conditional_interaction fired above (a
                # mixed-type or spike conditioning pair) -- run_conditional_mi
                # needs raw_deferred=True (and w_processor_type, since a
                # mixed pair's W may not share X's type) to window them
                # correctly instead of treating them as already-windowed
                # tensors. align='dual_branch' already raises a clear error
                # earlier for permutation_test, so raw_deferred here only
                # ever means the concat-based (non-dual_branch) case.
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, 'conditional', sweep_grid,
                    n_permutations, analysis_kwargs, w_data=w_run_data,
                    raw_deferred=_defer_for_conditional_interaction,
                    w_processor_type=w_processor_type,
                    permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'interaction':
            if w_data is None:
                raise ValueError("`w_data` must be provided for mode='interaction'.")
            # Process w_data if a processor type is given; otherwise assume pre-processed
            # (same pattern as mode='conditional' -- interaction information's
            # W is a third population, not a growing-history conditioning array, so it
            # needs no special-casing beyond that). When
            # _defer_for_conditional_interaction, keep w_data raw too, left
            # exactly as passed in -- run_interaction_information's
            # raw_deferred branch converts it.
            if _defer_for_conditional_interaction:
                w_run_data = w_data
            elif w_processor_type is not None:
                from .data.handler import create_dataset as _cds
                # Paired with Y (not built alone) so W's window-validity
                # criterion is "W has data AND Y has data" -- the same
                # invariant X's own windows (built above) already enforce.
                # See mode='conditional''s identical fix above for why.
                w_dataset = _cds(
                    x_data=w_data, y_data=y_data,
                    x_time=w_time, y_time=y_time,
                    processor_type_x=w_processor_type,
                    processor_params_x=w_processor_params or {},
                    processor_type_y=_effective_processor_type_y,
                    processor_params_y=processor_params_y or {},
                )
                w_run_data = w_dataset.x_data
                if w_processor_type == 'categorical':
                    # The same re-layout mode='conditional' applies above. Both
                    # modes concatenate W onto X along the channel axis, so both
                    # need W on X's window-size axis. interaction handled only
                    # the collapsed size-1 case, by broadcasting; a
                    # 'full_trajectory' categorical W (window axis 2 against X's
                    # 21) reached the concat and raised.
                    w_run_data = _reshape_categorical_w_for_conditional(
                        w_run_data, w_dataset.x_dataset
                    )
                x_run_data, y_run_data, w_run_data = _align_conditioning_windows(
                    mode, x_run_data, y_run_data, w_run_data,
                    _xy_window_times, w_dataset, base_params)
            else:
                w_run_data = w_data if torch.is_tensor(w_data) else torch.from_numpy(np.array(w_data)).float()
            n_workers = analysis_kwargs.get('n_workers', 1)
            use_rigorous = analysis_kwargs.pop('rigorous', False)
            if use_rigorous:
                from .analysis.rigorous import run_rigorous_scalar_analysis
                from .analysis.interaction import _ii_rigorous_scalar
                _gamma_range = analysis_kwargs.pop('gamma_range', None) or range(1, 11)
                _rig_kwargs = {
                    'gamma_range': _gamma_range,
                    'curvature_t_threshold': analysis_kwargs.pop('curvature_t_threshold', curvature_t_threshold),
                    'min_gamma_points': analysis_kwargs.pop('min_gamma_points', min_gamma_points),
                    'confidence_level': analysis_kwargs.pop('confidence_level', confidence_level),
                    'residual_threshold': analysis_kwargs.pop('residual_threshold', 2.5),
                    'r2_threshold': analysis_kwargs.pop('r2_threshold', 0.90),
                    'leverage_threshold': analysis_kwargs.pop('leverage_threshold', 0.20),
                }
                # Parallelised across gamma-chunk tasks (like plain mode='rigorous');
                # each individual chunk's II call runs with n_workers=1 internally
                # (see _ii_rigorous_scalar) to avoid nested multiprocessing pools.
                rig_details = run_rigorous_scalar_analysis(
                    scalar_fn=_ii_rigorous_scalar,
                    x_data=x_run_data, y_data=y_run_data, base_params=base_params,
                    extra_data={'w_data': w_run_data},
                    extra_kwargs={'sweep_grid': sweep_grid,
                                 'raw_deferred': _defer_for_conditional_interaction,
                                 'w_processor_type': w_processor_type},
                    n_workers=n_workers,
                    raw_deferred=_defer_for_conditional_interaction,
                    **_rig_kwargs,
                )
                rig_details = _convert_mi_units(rig_details, output_units == 'bits')
                raw_df = rig_details.pop('raw_results_df', pd.DataFrame())
                return Results(
                    mode=mode,
                    mi_estimate=rig_details.get('mi_corrected'),
                    dataframe=raw_df,
                    params={**run_params, 'rigorous': True},
                    details=rig_details,
                )
            # Standard (non-rigorous) path
            raw = run_interaction_information(x_run_data, y_run_data, w_run_data, base_params,
                                              sweep_grid=sweep_grid, n_workers=n_workers,
                                              raw_deferred=_defer_for_conditional_interaction,
                                              w_processor_type=w_processor_type)
            raw = _convert_mi_units(raw, output_units == 'bits')
            ii = raw['interaction_info']
            result = Results(mode=mode, mi_estimate=ii, params=run_params, details=raw)
            if permutation_test:
                # See the identical comment at mode='conditional''s
                # permutation_test call site: x_run_data/y_run_data/w_run_data
                # are raw here whenever _defer_for_conditional_interaction
                # fired, and run_interaction_information needs raw_deferred/
                # w_processor_type to window them correctly.
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, 'interaction', sweep_grid,
                    n_permutations, analysis_kwargs, w_data=w_run_data,
                    raw_deferred=_defer_for_conditional_interaction,
                    w_processor_type=w_processor_type,
                    permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'transfer':
            if history_window is None:
                raise ValueError("`history_window` must be provided for mode='transfer'.")
            # For transfer entropy, expect 2-D inputs (T, channels).
            # StaticDataset wraps (T, C) → (T, C, 1); squeeze the trailing 1 back out.
            def _to_2d(t):
                if hasattr(t, 'ndim') and t.ndim == 3 and t.shape[-1] == 1:
                    return t.reshape(t.shape[0], t.shape[1]).contiguous()
                return t
            _x_te = _to_2d(x_run_data)
            _y_te = _to_2d(y_run_data)
            if _x_te.ndim == 3:
                raise ValueError(
                    "mode='transfer' requires 2-D input data of shape (n_timepoints, n_channels), "
                    f"but received a 3-D array of shape {tuple(_x_te.shape)}. "
                    "This typically happens when a windowed processor_type_x is used, which "
                    "collapses the temporal structure that transfer entropy relies on. "
                    "Pass the raw time-series directly (without a windowed processor) and let "
                    "mode='transfer' build its own history/prediction arrays internally."
                )
            # Optional third conditioning signal for conditional transfer entropy
            # (TE(X->Y|W) instead of plain TE(X->Y)) -- same raw 2-D convention as
            # x_data/y_data, since it feeds the same internal unfold-based W_past
            # construction as run_transfer_entropy already does for Y_past.
            w_run_data = None
            if w_data is not None:
                if w_processor_type is not None:
                    from .data.handler import create_dataset as _cds
                    w_dataset = _cds(
                        x_data=w_data, y_data=None,
                        x_time=w_time,
                        processor_type_x=w_processor_type,
                        processor_params_x=w_processor_params or {}
                    )
                    w_run_data = w_dataset.x_data
                else:
                    w_run_data = w_data if torch.is_tensor(w_data) else torch.from_numpy(np.array(w_data)).float()
                w_run_data = _to_2d(w_run_data)
                if w_run_data.ndim == 3:
                    raise ValueError(
                        "mode='transfer' requires w_data of shape (n_timepoints, n_channels) "
                        f"(2-D), but received a 3-D array of shape {tuple(w_run_data.shape)}. "
                        "Pass the raw time-series directly and let mode='transfer' build its "
                        "own W_past history array internally, the same as x_data/y_data."
                    )
            n_workers = analysis_kwargs.get('n_workers', 1)
            use_rigorous = analysis_kwargs.pop('rigorous', False)
            if use_rigorous:
                from .analysis.rigorous import run_rigorous_scalar_analysis
                from .analysis.transfer import _te_rigorous_scalar
                _gamma_range = analysis_kwargs.pop('gamma_range', None) or range(1, 11)
                _rig_kwargs = {
                    'gamma_range': _gamma_range,
                    'curvature_t_threshold': analysis_kwargs.pop('curvature_t_threshold', curvature_t_threshold),
                    'min_gamma_points': analysis_kwargs.pop('min_gamma_points', min_gamma_points),
                    'confidence_level': analysis_kwargs.pop('confidence_level', confidence_level),
                    'residual_threshold': analysis_kwargs.pop('residual_threshold', 2.5),
                    'r2_threshold': analysis_kwargs.pop('r2_threshold', 0.90),
                    'leverage_threshold': analysis_kwargs.pop('leverage_threshold', 0.20),
                }
                # Parallelised across gamma-chunk tasks (like plain mode='rigorous');
                # each individual chunk's TE call runs with n_workers=1 internally
                # (see _te_rigorous_scalar) to avoid nested multiprocessing pools.
                rig_details = run_rigorous_scalar_analysis(
                    scalar_fn=_te_rigorous_scalar,
                    x_data=_x_te, y_data=_y_te, base_params=base_params,
                    # w_data must be chunked identically to x_data/y_data for every
                    # gamma-chunk, so it goes through extra_data (already correct for
                    # this, proven by w_data's use of the same mechanism in
                    # mode='conditional' above), not extra_kwargs, which is passed
                    # unsplit to every chunk and would be wrong for per-chunk-varying data.
                    extra_data={'w_data': w_run_data} if w_run_data is not None else None,
                    extra_kwargs={'sweep_grid': sweep_grid, 'history_window': history_window,
                                  'prediction_horizon': prediction_horizon,
                                  'bidirectional': bidirectional_te},
                    n_workers=n_workers,
                    # Transfer entropy is unconditionally temporal (built from
                    # time-ordered history windows via unfold) -- forced True
                    # rather than auto-detected, since base_params here
                    # predates run_transfer_entropy's own leak_check_window_size
                    # injection (set per-chunk, inside _te_rigorous_scalar,
                    # after this call's chunking has already happened).
                    temporal_chunking=True,
                    **_rig_kwargs,
                )
                # _convert_mi_units already recurses into rig_details['raw_results_df']
                # (dict branch, 'raw_results_df' key) -- popping it AFTER this call and
                # converting it again would double-scale its 'train_mi' column.
                rig_details = _convert_mi_units(rig_details, output_units == 'bits')
                raw_df = rig_details.pop('raw_results_df', pd.DataFrame())
                return Results(
                    mode=mode,
                    mi_estimate=rig_details.get('mi_corrected'),
                    dataframe=raw_df,
                    params={**run_params, 'rigorous': True},
                    details=rig_details,
                )
            # Standard (non-rigorous) path
            raw = run_transfer_entropy(_x_te, _y_te, base_params,
                                       history_window=history_window,
                                       prediction_horizon=prediction_horizon,
                                       sweep_grid=sweep_grid, n_workers=n_workers,
                                       bidirectional=bidirectional_te,
                                       w_data=w_run_data)
            raw = _convert_mi_units(raw, output_units == 'bits')
            te = raw['te_estimate']
            result = Results(mode=mode, mi_estimate=te, params=run_params, details=raw)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    _x_te, _y_te, base_params, 'transfer', sweep_grid,
                    n_permutations, analysis_kwargs,
                    history_window=history_window, prediction_horizon=prediction_horizon,
                    bidirectional_te=bidirectional_te,
                    permutation_shuffle=permutation_shuffle,
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'pairwise':
            n_workers = analysis_kwargs.get('n_workers', 1)
            pairs = analysis_kwargs.get('pairs', None)
            # Pass y_run_data when provided to enable cross-pairwise mode.
            pairwise_y = y_run_data if y_data is not None else None
            if permutation_test:
                n_ch_x = _n_channels_of(x_run_data)
                if pairwise_y is not None:
                    n_pairs_est = n_ch_x * _n_channels_of(pairwise_y)
                else:
                    n_pairs_est = n_ch_x * (n_ch_x - 1) // 2
                warnings.warn(
                    f"Permutation test requested for mode='pairwise'. This will run the full "
                    f"pairwise matrix estimation {n_permutations} time(s), which is computationally "
                    f"expensive ({n_pairs_est} pairs × {n_permutations} permutation(s) = "
                    f"{n_pairs_est * n_permutations} MI estimations total). "
                    f"Allow additional time or reduce n_permutations.",
                    UserWarning,
                    stacklevel=2,
                )
            raw = run_pairwise_mi(x_run_data, base_params, y_data=pairwise_y,
                                  sweep_grid=sweep_grid, n_workers=n_workers, pairs=pairs)
            _to_bits = output_units == 'bits'
            raw['mi_matrix'] = _convert_mi_units(raw['mi_matrix'], _to_bits)
            # `df` stays bound: the Results construction further down uses it.
            df = _convert_mi_units(raw['dataframe'].copy(), _to_bits)
            raw['dataframe'] = df
            # Inject channel names for heatmap axis labels when provided.
            n_ch = raw.get('n_channels')
            if isinstance(n_ch, tuple):
                # Cross-pairwise: rows = x channels, cols = y channels
                if channel_names_x is not None:
                    raw['variable_names_y'] = list(channel_names_x)[:n_ch[0]]
                if channel_names_y is not None:
                    raw['variable_names_x'] = list(channel_names_y)[:n_ch[1]]
            elif isinstance(n_ch, int) and channel_names_x is not None:
                # Self-pairwise: same channel set for both axes
                raw['variable_names_x'] = list(channel_names_x)[:n_ch]
                raw['variable_names_y'] = list(channel_names_x)[:n_ch]
            result = Results(mode=mode, params=run_params, details=raw,
                             dataframe=df)
            if permutation_test:
                if pairwise_y is not None:
                    _null_clipped, _null_raw = _run_permutation_test(
                        x_run_data, pairwise_y, base_params, 'pairwise', sweep_grid,
                        n_permutations, analysis_kwargs, pairs=pairs,
                        permutation_shuffle=permutation_shuffle,
                    )
                    result.details['null_distribution'] = _null_clipped
                    result.details['null_distribution_raw'] = _null_raw
                else:
                    logger.warning(
                        "permutation_test=True has no effect for self-pairwise mode "
                        "(mode='pairwise' without y_data): there is no second variable "
                        "to shuffle against, so no null distribution is computed. "
                        "Cross-pairwise mode (pass y_data) supports permutation testing."
                    )
            return result

        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. "
                f"Expected one of: 'estimate', 'sweep', 'dimensionality', 'rigorous', "
                f"'lag', 'precision', 'conditional', 'interaction', 'transfer', 'pairwise'."
            )
    finally:
        logger.setLevel(_prev_level)
        for h, lv in zip(logger.handlers, _prev_handler_levels):
            h.setLevel(lv)


# Named parameters of the engine, computed once at import. run() consults this
# (via `_named`) to route each lowered config key to the correct bucket. Defined
# here because it depends on _run_flat's signature.
_ENGINE_PARAMS = frozenset(
    n for n, p in _inspect.signature(_run_flat).parameters.items()
    if p.kind in (_inspect.Parameter.POSITIONAL_OR_KEYWORD,
                  _inspect.Parameter.KEYWORD_ONLY)
) - {'x_data', 'y_data'}


def _warn_small_sample(dataset, base_params: dict) -> None:
    """Emit guidance when the processed dataset has very few samples."""
    try:
        n_samples = dataset.x_data.shape[0] if dataset.x_data is not None else 0
    except AttributeError:
        return
    if n_samples <= 0:
        return

    user_dropout = base_params.get('dropout', 0.0)
    user_norm = base_params.get('norm_layer', None)
    user_hidden = base_params.get('hidden_dim', 64)
    user_embed = base_params.get('embedding_dim', 64)

    if n_samples < 200:
        tips = []
        if user_dropout == 0.0:
            tips.append("dropout=0.2 (adds regularisation)")
        if user_norm is None:
            tips.append("norm_layer='layer' (LayerNorm stabilises small-batch training)")
        if user_hidden > 32:
            tips.append(f"hidden_dim=32 (current: {user_hidden})")
        if user_embed > 32:
            tips.append(f"embedding_dim=32 (current: {user_embed})")
        tips.append("optimizer='adamw' with optimizer_params={'weight_decay': 1e-3}")
        hint = "; ".join(tips)
        warnings.warn(
            f"Very few samples detected ({n_samples} windows after processing). "
            f"Neural MI estimators are prone to overfitting and high-variance estimates "
            f"at this scale. Consider adding these to your Model/Training configs: {hint}. "
            f"See the NeuralMI documentation for small-sample guidance.",
            UserWarning,
            stacklevel=4,
        )
    elif n_samples < 500:
        tips = []
        if user_dropout == 0.0:
            tips.append("dropout=0.1")
        if user_norm is None:
            tips.append("norm_layer='layer'")
        if tips:
            warnings.warn(
                f"Small dataset detected ({n_samples} windows). Regularisation may help: "
                f"consider adding {' and '.join(tips)} to your Model config.",
                UserWarning,
                stacklevel=4,
            )


# Modes/paths where windowing is deferred to the worker that trains the model,
# so the dataset it builds is still a genuine PairedTemporalDataset with a
# live WindowManager -- shift_time (and anything else gated on
# is_temporal at the Trainer) is actually reachable there. Everywhere else,
# run.py windows the data once, up front, and hands the Trainer an
# already-windowed static PairedDataset that has never heard of time.
def _shift_time_is_reachable(mode: str, is_proc_sweep: bool,
                             processor_type_x: Optional[str] = None,
                             effective_processor_type_y: Optional[str] = None,
                             processor_params_x: Optional[dict] = None,
                             processor_params_y: Optional[dict] = None) -> bool:
    if is_proc_sweep or mode == 'lag':
        return True
    _family = shift_family(processor_type_x, effective_processor_type_y)
    if mode == 'rigorous':
        # Narrower than the other modes below: 'rigorous' only supports the
        # 'spike' family's chunk-to-raw-time-range translation, not 'mixed'.
        return _family == 'spike'
    if mode not in _SHIFT_SAFE_MODES:
        return False
    if _family == 'spike':
        return True
    if _family == 'mixed':
        return mixed_pair_sample_rate_ok(processor_type_x, processor_params_x,
                                         effective_processor_type_y, processor_params_y)
    return False


def _warn_if_shift_time_dead(base_params: dict, mode: str, is_proc_sweep: bool,
                             processor_type_x: Optional[str] = None,
                             processor_params_x: Optional[dict] = None,
                             effective_processor_type_y: Optional[str] = None,
                             processor_params_y: Optional[dict] = None,
                             user_set_keys: Optional[set] = None,
                             extra_reachable: bool = False) -> None:
    """Warn once, here, if shift_time was requested but cannot take effect
    on this mode/processing path.

    Placed in run.py (not deep in the training loop) because this is the
    earliest point where both the resolved mode and whether processing will
    be deferred to the worker are known. Only fires when the user explicitly
    set `shift_time=True` (i.e. the key was present in `base_params` before
    `apply_defaults()` ran, per `user_set_keys`) -- a value that only came
    from the schema default stays silent even when it has no effect, since
    the user never asked for it on this pair.

    Beyond `mode='lag'`, this is also reachable at every mode in
    `_SHIFT_SAFE_MODES` (module scope: `estimate`, plain `sweep`,
    `pairwise`, `dimensionality`, `precision` -- each dispatches
    independent training run(s), or a comparison/evaluation that only
    reads a frozen, pre-shift view, with nothing for per-run shift
    randomness to corrupt) for
    `shift_family(...) == 'spike'` (spike+spike -- both sides natively in
    seconds, no cross-unit concern) and for `'mixed'` pairs (one side
    spike, the other continuous/categorical) *only when* the regular-grid
    side has `sample_rate` set -- see `shift_windowing.mixed_pair_sample_rate_ok`
    and `NEURALMI_REFERENCE.md`'s shift-mechanisms section for why that's
    required rather than optional. `'regular'` pairs (continuous/
    categorical on both sides) are deliberately excluded even there -- they
    already have the strictly cheaper, bug-free `shift_windows`
    mechanism; routing them through this one too would just reopen its
    `SubsetView` risk for no benefit. Processor-swept `mode='sweep'` is
    reachable regardless of processor type (`is_proc_sweep` alone triggers
    it), unrelated to this pair-based logic.

    extra_reachable : bool, optional
        Set by the caller for a reachability path this function doesn't
        derive from `mode`/`processor_type_x`/`effective_processor_type_y`
        alone -- currently the spike sub-case of
        `_defer_spike_conditional_interaction` (`mode` in
        `('conditional', 'interaction')` with a 'spike' conditioning
        variable matching X's 'spike' type). `True` silences the warning
        the same as the mode/family check below. Sibling to
        `_warn_if_shift_windows_dead`'s own `extra_reachable` -- kept
        separate (not one shared flag) since a case reachable via one
        mechanism is not necessarily reachable via the other (e.g. the
        *regular*-family sub-case of conditional/interaction reachability
        makes shift_windows reachable but not shift_time).
    """
    if user_set_keys is not None and 'shift_time' not in user_set_keys:
        return
    if base_params.get('shift_time') is not True:
        return
    if extra_reachable:
        return
    if _shift_time_is_reachable(mode, is_proc_sweep, processor_type_x,
                                effective_processor_type_y, processor_params_x, processor_params_y):
        return
    _family = shift_family(processor_type_x, effective_processor_type_y)
    _mixed_hint = ""
    if mode in _SHIFT_SAFE_MODES and _family == 'mixed':
        _mixed_hint = (
            " This pair mixes 'spike' with 'continuous'/'categorical': a shift "
            "value means seconds for spike but raw sample-index units for the "
            "other side unless it has a 'sample_rate'. Pass "
            "processor_params_x={'sample_rate': ...} (or processor_params_y) "
            "on whichever side is 'continuous'/'categorical' to enable shifting "
            "for this pair."
        )
    warnings.warn(
        f"shift_time=True was requested but has no effect for "
        f"mode='{mode}' with this configuration: windowing is applied once, "
        f"eagerly, before training starts, so the Trainer receives an "
        f"already-windowed static dataset with no notion of time left to "
        f"shift. This option currently only takes effect when windowing is "
        f"deferred to the training worker -- mode='lag'; mode='sweep' when a "
        f"processor parameter (e.g. window_size) is itself part of the sweep "
        f"grid; or mode in {list(_SHIFT_SAFE_MODES)} with a "
        f"'spike'+'spike' pair (or a mixed spike/continuous(-categorical) pair "
        f"with 'sample_rate' set on the non-spike side); mode='rigorous' also "
        f"works for a 'spike'+'spike' pair specifically (not mixed)."
        f"{_mixed_hint} For "
        f"continuous/categorical pairs, prefer Training(shift_windows=True) "
        f"instead -- cheaper and without this mechanism's SubsetView caveat. "
        f"Pass shift_time=False to silence this warning.",
        UserWarning,
        stacklevel=4,
    )


def _warn_if_shift_windows_dead(base_params: dict, mode: str, processor_type_x: Optional[str],
                                effective_processor_type_y: Optional[str],
                                user_set_keys: Optional[set] = None,
                                extra_reachable: bool = False) -> None:
    """Sibling to `_warn_if_shift_time_dead`: warn if `shift_windows=True`
    was requested but this pass only wires it up for modes in
    `_SHIFT_WINDOWS_SAFE_MODES` (module scope) with `shift_family(...) ==
    'regular'` -- both `processor_type_x` and the effective
    `processor_type_y` (after the `None` -> "inherit X" convention) in
    `{'continuous', 'categorical'}` (need not match each other --
    continuous+categorical is fine), not 'spike' on either side (see the
    matching comment at the `_defer_for_shift_windows` call site in
    `_run_flat` for why a mismatched pair would silently misbehave rather
    than just be inert), and not `None` on either side
    (`neural_mi/data/shift_windowing.py` needs the raw array +
    window_size/step_size; there's nothing to reslice without them). Same
    explicit-vs-defaulted convention as the sibling check.

    extra_reachable : bool, optional
        Set by the caller for a reachability path this function doesn't
        derive from `mode`/`processor_type_x`/`effective_processor_type_y`
        alone -- currently `_defer_for_conditional_interaction` (`mode` in
        `('conditional', 'interaction')` with a 'continuous' conditioning
        variable matching X's family). `True` silences the warning the same
        as the mode/family check below.
    """
    if user_set_keys is not None and 'shift_windows' not in user_set_keys:
        return
    if base_params.get('shift_windows') is not True:
        return
    if extra_reachable:
        return
    if mode in _SHIFT_WINDOWS_SAFE_MODES and shift_family(processor_type_x, effective_processor_type_y) == 'regular':
        return
    warnings.warn(
        f"shift_windows=True was requested but has no effect for "
        f"mode='{mode}' with processor_type_x={processor_type_x!r}, "
        f"processor_type_y={effective_processor_type_y!r} (effective). "
        f"This is currently wired up only for mode in "
        f"{list(_SHIFT_WINDOWS_SAFE_MODES)} with processor_type_x and "
        f"processor_type_y both in "
        f"{{'continuous', 'categorical'}} (e.g. Processing(x='continuous', "
        f"x_params={{'window_size': ...}}, y='categorical', "
        f"y_params={{'window_size': ...}})) -- 'spike' is not supported by "
        f"this mechanism on either side (no regular sampling grid to "
        f"reslice; use Training(shift_time=True) for spike data "
        f"instead). Pass shift_windows=False to silence this warning.",
        UserWarning,
        stacklevel=4,
    )


def _spike_population_extent(y_data: list, base_params: dict) -> tuple:
    """``(t_start, t_end)`` for a raw spike-time population -- mirrors
    ``SpikeDataset.get_temporal_extent()``'s exact convention (the
    already-established one for spike data elsewhere in this library, not a
    new one invented here): ``t_start`` is the earliest spike across
    neurons; ``t_end`` is ``processor_params_y['n_seconds']`` if the caller
    explicitly set it (letting a recording extend past its last spike),
    else the latest spike across neurons.
    """
    valid = [np.asarray(st) for st in y_data if len(st) > 0]
    t_start = min((st[0] for st in valid), default=0.0)
    n_seconds = (base_params.get('processor_params_y') or {}).get('n_seconds')
    if n_seconds is not None:
        t_end = float(n_seconds)
    else:
        t_end = max((st[-1] for st in valid), default=0.0)
    return float(t_start), float(t_end)


def _circular_shift_spike_population(y_data: list, t_start: float, t_end: float) -> list:
    """Shift every neuron's spike train by one shared random offset, wrapping
    at the recording boundary -- the default permutation-test null for
    spike-type Y.

    One shared offset (not an independent one per neuron) preserves Y's own
    internal cross-neuron structure (e.g. any real synchrony within the Y
    population itself), only breaking Y's temporal correspondence with X --
    the same reasoning that already rules out per-neuron independent
    reordering. The offset is drawn uniformly across the *entire* valid
    range rather than a narrow band, deliberately: the true coupling
    timescale (if any) is unknown, and a uniform draw is agnostic to it,
    whether the real dependency (if any) is a fast, precise-timing effect
    or a slow, shared-drift one.
    """
    duration = t_end - t_start
    if duration <= 0:
        return [np.asarray(st).copy() for st in y_data]
    delta = np.random.uniform(0.0, duration)
    shifted = []
    for st in y_data:
        st = np.asarray(st, dtype=float)
        if st.size == 0:
            shifted.append(st.copy())
            continue
        new_st = t_start + np.mod((st - t_start) + delta, duration)
        shifted.append(np.sort(new_st))
    return shifted


def _block_shuffle_spike_population(y_data: list, t_start: float, t_end: float,
                                    block_size: float) -> list:
    """Cut the recording into fixed-size contiguous blocks and reorder them --
    the opt-in alternative null for spike-type Y (``permutation_shuffle='block'``).

    Preserves every spike's position *within* its block exactly (unlike
    circular shift, which preserves the whole train's global statistics but
    not any fixed block-boundary structure); breaks block-to-block
    correspondence with X. Reassembled block-by-block into a new spike-time
    list of the same total duration, so it flows through the same
    downstream (raw, unwindowed) pipeline as circular shift or the
    unpermuted data -- no separate windowing/rebuilding needed.
    """
    duration = t_end - t_start
    if duration <= 0 or block_size <= 0:
        return [np.asarray(st).copy() for st in y_data]
    n_blocks = max(1, int(round(duration / block_size)))
    edges = np.linspace(t_start, t_end, n_blocks + 1)
    order = np.random.permutation(n_blocks)
    shifted = []
    for st in y_data:
        st = np.asarray(st, dtype=float)
        pieces = []
        cursor = t_start
        for b in order:
            lo, hi = edges[b], edges[b + 1]
            # Half-open [lo, hi) for every block except the recording's own
            # last one, which must be closed on the right -- otherwise a
            # spike landing exactly at t_end (the final edge) matches no
            # block at all and silently vanishes.
            in_block = (st >= lo) & (st < hi if b < n_blocks - 1 else st <= hi)
            pieces.append(st[in_block] - lo + cursor)
            cursor += (hi - lo)
        shifted.append(np.sort(np.concatenate(pieces)) if pieces else np.array([]))
    return shifted


def _run_single_permutation(args):
    """Top-level picklable function for one permutation trial.

    Parameters
    ----------
    args : tuple
        ``(x_data, y_data, base_params, mode, sweep_grid, perm_seed,
        mode_kwargs, permutation_shuffle)``

    Returns
    -------
    tuple[float, float]
        ``(mi_clipped, mi_raw)`` where *mi_clipped* matches the main-run
        convention (negatives zeroed by the trainer's ``all_mi_negative`` guard)
        and *mi_raw* retains the actual value including negatives.
    """
    import numpy as _np
    import torch as _torch
    x_data, y_data, base_params, mode, sweep_grid, perm_seed, mode_kwargs, permutation_shuffle = args
    _np.random.seed(perm_seed)
    if isinstance(y_data, list):
        # Raw spike-type population (list of per-neuron spike-time arrays).
        # Index-permuting the list (this function's convention for every
        # other data type) only reorders which array sits at which list
        # position -- every neuron's own, complete spike train is untouched,
        # so the population's joint activity across time is byte-for-byte
        # identical and no X<->Y temporal correspondence is actually broken.
        t_start, t_end = _spike_population_extent(y_data, base_params)
        if permutation_shuffle == 'block':
            block_size = (
                (base_params.get('processor_params_y') or {}).get('window_size')
                or (base_params.get('processor_params_x') or {}).get('window_size')
                or (t_end - t_start) / 10.0
            )
            y_perm = _block_shuffle_spike_population(y_data, t_start, t_end, block_size)
        else:
            y_perm = _circular_shift_spike_population(y_data, t_start, t_end)
    else:
        n = y_data.shape[0] if hasattr(y_data, 'shape') else len(y_data)
        shuffle_idx = _np.random.permutation(n)
        if _torch.is_tensor(y_data):
            y_perm = y_data[shuffle_idx]
        else:
            y_perm = [y_data[i] for i in shuffle_idx]

    _nan = float('nan')
    try:
        if mode in ('estimate', 'sweep', 'dimensionality'):
            from neural_mi.analysis.sweep import ParameterSweep
            res = ParameterSweep(x_data, y_perm, base_params.copy()).run(
                sweep_grid or {}, n_workers=1, is_proc_sweep=False
            )
            mi_clipped = float(_np.nanmean([r.get('train_mi', _nan) for r in res]))
            mi_raw = float(_np.nanmean([r.get('raw_train_mi', _nan) for r in res]))
            return mi_clipped, mi_raw

        elif mode == 'lag':
            from neural_mi.analysis.lag import run_lag_analysis as _rla
            lag_range = mode_kwargs.get('lag_range')
            res = _rla(x_data, y_perm, base_params.copy(),
                       lag_range=lag_range, n_workers=1)
            # Each task result dict contains both train_mi (zeroed for all-neg runs) and
            # raw_train_mi (actual value), matching the estimate/sweep convention.
            mi_clipped = float(_np.nanmean([r.get('train_mi', _nan) for r in res]))
            mi_raw = float(_np.nanmean([r.get('raw_train_mi', _nan) for r in res]))
            return mi_clipped, mi_raw

        elif mode == 'conditional':
            from neural_mi.analysis.conditional import run_conditional_mi as _rcmi
            w_data = mode_kwargs.get('w_data')
            raw = _rcmi(x_data, y_perm, w_data, base_params.copy(), n_workers=1,
                       raw_deferred=mode_kwargs.get('raw_deferred', False),
                       w_processor_type=mode_kwargs.get('w_processor_type'))
            mi_clipped = raw['cmi_estimate']
            # Raw CMI = mean(raw_train_mi of XW→Y sweep) − mean(raw_train_mi of W→Y sweep)
            _rxw = [r.get('raw_train_mi', _nan) for r in raw.get('raw_xw_y', [])]
            _rw = [r.get('raw_train_mi', _nan) for r in raw.get('raw_w_y', [])]
            mi_raw = (float(_np.nanmean(_rxw)) - float(_np.nanmean(_rw))
                      if _rxw and _rw else mi_clipped)
            return mi_clipped, mi_raw

        elif mode == 'interaction':
            from neural_mi.analysis.interaction import run_interaction_information as _rii
            w_data = mode_kwargs.get('w_data')
            raw = _rii(x_data, y_perm, w_data, base_params.copy(), n_workers=1,
                      raw_deferred=mode_kwargs.get('raw_deferred', False),
                      w_processor_type=mode_kwargs.get('w_processor_type'))
            mi_clipped = raw['interaction_info']
            # Raw II has no single joint/marginal pair (it's a 3-term combination),
            # so there's no equally cheap "raw" counterpart -- reuse the clipped value.
            return mi_clipped, mi_clipped

        elif mode == 'transfer':
            from neural_mi.analysis.transfer import run_transfer_entropy as _rte
            raw = _rte(
                x_data, y_perm, base_params.copy(),
                history_window=mode_kwargs.get('history_window'),
                prediction_horizon=mode_kwargs.get('prediction_horizon', 1),
                bidirectional=mode_kwargs.get('bidirectional_te', False),
                n_workers=1,
            )
            mi_clipped = raw['te_estimate']
            # Raw TE = mean(raw_train_mi of joint sweep) − mean(raw_train_mi of marginal sweep)
            _rjoint = [r.get('raw_train_mi', _nan) for r in raw.get('raw_xypast_yfuture', [])]
            _rmarg = [r.get('raw_train_mi', _nan) for r in raw.get('raw_ypast_yfuture', [])]
            mi_raw = (float(_np.nanmean(_rjoint)) - float(_np.nanmean(_rmarg))
                      if _rjoint and _rmarg else mi_clipped)
            return mi_clipped, mi_raw

        elif mode == 'pairwise':
            from neural_mi.analysis.pairwise import run_pairwise_mi as _rpm
            raw = _rpm(x_data, base_params.copy(), y_data=y_perm,
                       sweep_grid=sweep_grid, n_workers=1, pairs=mode_kwargs.get('pairs'))
            mi_vals = raw['dataframe']['mi_mean']
            mi_clipped = float(_np.nanmean(mi_vals)) if len(mi_vals) else _nan
            # Pairwise doesn't track a separate unclipped value per pair.
            return mi_clipped, mi_clipped

        else:
            return _nan, _nan

    except Exception as exc:
        logger.warning(f"Permutation trial failed: {exc}")
        return _nan, _nan


def _run_permutation_test(x_data, y_data, base_params, mode, sweep_grid,
                          n_permutations, analysis_kwargs, permutation_shuffle='circular',
                          **mode_kwargs):
    """Run the permutation null test by shuffling y_data *n_permutations* times.

    permutation_shuffle : {'circular', 'block'}, default='circular'
        Only affects raw spike-type y_data (a list of per-neuron spike-time
        arrays) -- non-spike y_data (already an array/tensor, whether raw
        2-D or already windowed) is unaffected, see
        ``_run_single_permutation``. ``'circular'``: shift the whole
        population by one shared random offset, wrapping at the recording
        boundary -- preserves every spike-timing detail and Y's own
        internal cross-neuron structure, only breaks temporal alignment
        with X. ``'block'``: cut the recording into fixed-size contiguous
        blocks (sized from ``processor_params_y``/``processor_params_x``'s
        ``window_size``) and reorder them -- preserves exact within-block
        spike patterns, at the cost of leaving block-boundary structure
        intact. See CHANGELOG for why a plain index permutation (this
        function's convention for every other data type) is wrong for a
        spike-time list.

    When ``analysis_kwargs['n_workers'] > 1`` the permutation trials are
    dispatched to a multiprocessing pool so they run in parallel (each
    individual trial uses a single worker internally to avoid nested pools).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(null_distribution, null_distribution_raw)``

        *null_distribution* — per-permutation mean MI with negatives clipped to
        zero (matching the library's main-run reporting convention).

        *null_distribution_raw* — per-permutation mean MI retaining actual
        values (including negatives), mirroring ``details['raw_train_mi']``.
    """
    n_workers = analysis_kwargs.get('n_workers', 1)
    show_progress = base_params.get('show_progress', True)
    logger.info(
        f"Permutation test: running {n_permutations} permutations for "
        f"mode='{mode}' (n_workers={n_workers})..."
    )

    # Generate independent seeds so parallel workers produce different shuffles
    perm_seeds = [int(np.random.randint(0, 2**31)) for _ in range(n_permutations)]
    perm_args = [
        (x_data, y_data, base_params.copy(), mode, sweep_grid, seed, dict(mode_kwargs),
         permutation_shuffle)
        for seed in perm_seeds
    ]

    if n_workers > 1:
        _log_init, _log_args = worker_init_args()
        with mp.get_context("spawn").Pool(processes=n_workers,
                                          initializer=_log_init, initargs=_log_args) as pool:
            raw_results = list(tqdm(
                pool.imap(_run_single_permutation, perm_args),
                total=n_permutations,
                desc="Permutation test",
                leave=False,
                disable=not show_progress,
            ))
    else:
        raw_results = [
            _run_single_permutation(args)
            for args in tqdm(perm_args, desc="Permutation test", leave=False,
                             disable=not show_progress)
        ]

    null_distribution = [r[0] for r in raw_results]
    null_distribution_raw = [r[1] for r in raw_results]

    if null_distribution and all(np.isnan(v) for v in null_distribution):
        warnings.warn(
            f"All {n_permutations} permutation trial(s) for mode='{mode}' failed or "
            f"returned NaN; the null distribution is entirely NaN. Check the log for "
            f"'Permutation trial failed' messages to see why, or verify your "
            f"configuration is valid for this mode.",
            UserWarning,
            stacklevel=3,
        )

    logger.info(
        f"Permutation test complete. "
        f"Null MI (clipped): mean={np.nanmean(null_distribution):.4f}, "
        f"std={np.nanstd(null_distribution):.4f}"
    )
    return null_distribution, null_distribution_raw







