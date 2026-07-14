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
from .analysis.pairwise import run_pairwise_mi
from .data.handler import create_dataset
from .results import Results
from .validation import ParameterValidator, DataValidator
from .utils import get_device
from .logger import logger
from .defaults import PROCESSOR_PARAMS_SCHEMA
import inspect as _inspect
from .config import (
    Model, Training, Split, Estimator, Output, Processing,
    Rigorous, Precision, Lag, Transfer, Dimensionality, Conditional,
    Pairwise, Sweep, as_config,
)

# Mode name -> its dedicated config class (modes not listed take no mode config).
_MODE_CONFIG_CLASSES = {
    'rigorous': Rigorous, 'precision': Precision, 'lag': Lag,
    'transfer': Transfer, 'dimensionality': Dimensionality, 'conditional': Conditional,
    'pairwise': Pairwise, 'sweep': Sweep,
}


def _convert_mi_units(results: Any, to_bits: bool) -> Any:
    """Recursively converts MI values in results from nats to bits."""
    if not to_bits: return results
    NATS_TO_BITS = 1 / np.log(2)
    if isinstance(results, float): return results * NATS_TO_BITS
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
        cols = [
            'test_mi', 'train_mi', 'raw_train_mi', 'train_mi_at_peak',
            'test_mi_std', 'train_mi_std',          # precision-mode std columns
            'mi_mean', 'mi_std', 'mi_corrected', 'mi_error', 'mi_error_pred', 'slope',
        ]
        for col in cols:
            if col in df.columns: df[col] *= NATS_TO_BITS
        return df
    elif isinstance(results, list) and all(isinstance(r, dict) for r in results):
        keys = ['test_mi', 'train_mi', 'raw_train_mi', 'train_mi_at_peak',
                'mi_corrected', 'mi_error', 'mi_error_pred', 'slope']
        return [{**r, **{k: r.get(k, 0) * NATS_TO_BITS for k in keys if r.get(k) is not None}} for r in results]
    elif isinstance(results, dict):
        new_results = results.copy()
        # Scalar MI values stored by analysis modules (transfer entropy, CMI, etc.)
        _MI_SCALAR_KEYS = (
            'te_estimate', 'te_xy', 'te_yx',
            'i_xypast_yfuture', 'i_ypast_yfuture',
            'i_yxpast_xfuture', 'i_xpast_xfuture',
            'cmi_estimate', 'mi_xz_y', 'mi_z_y',
        )
        for k in _MI_SCALAR_KEYS:
            if k in new_results and isinstance(new_results[k], (int, float)):
                new_results[k] = new_results[k] * NATS_TO_BITS
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
    pairwise: Optional[Union[Pairwise, Dict[str, Any]]] = None,
    sweep: Optional[Union[Sweep, Dict[str, Any]]] = None,
    n_workers: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False,
    show_progress: bool = True,
    device: Optional[str] = None,
    permutation_test: bool = False,
    n_permutations: int = 1,
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
    mode : {'estimate','sweep','rigorous','dimensionality','lag','precision','conditional','transfer','pairwise'}
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
    rigorous, precision, lag, transfer, dimensionality, conditional, pairwise, sweep : mode config or dict, optional
        Mode-specific parameters; only the one matching ``mode`` is used. E.g.
        ``rigorous=Rigorous(confidence_level=0.68)``,
        ``precision=Precision(tau_grid=[...])``,
        ``transfer=Transfer(history_window=10)``,
        ``conditional=Conditional(z_data=z)``,
        ``pairwise=Pairwise(pairs=[(0, 1), (0, 2)])``,
        ``sweep=Sweep(max_samples_per_task=1000)``.
    n_workers : int, default=1
        Worker processes for parallelizable modes.
    seed : int, optional
        Random seed (``random``/``numpy``/``torch``). Full reproducibility only
        with ``n_workers=1``.
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
                 'conditional': conditional, 'pairwise': pairwise, 'sweep': sweep}
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
                ak = mode_cfg.to_analysis_kwargs()
                if 'bidirectional' in ak:
                    flat['bidirectional_te'] = ak.pop('bidirectional')
                _route_analysis(ak)
            elif isinstance(mode_cfg, Conditional):
                flat.update(mode_cfg.to_z_kwargs())
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
    delta_threshold: float = 0.1,
    min_gamma_points: int = 5,
    confidence_level: float = 0.68,
    max_eval_samples: int = 5000,
    train_subset_size: Optional[int] = None,
    spectral_mode: str = 'none',
    max_index_reduction: float = 0.05,
    tau_grid: Optional[List[float]] = None,
    corrupt_target: str = 'x',
    corruption_method: str = 'rounding',
    n_noise_samples: int = 50,
    threshold_ratio: float = 0.9,
    permutation_test: bool = False,
    n_permutations: int = 1,
    z_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    z_time: Optional[np.ndarray] = None,
    z_processor_type: Optional[str] = None,
    z_processor_params: Optional[Dict[str, Any]] = None,
    history_window: Optional[int] = None,
    prediction_horizon: int = 1,
    bidirectional_te: bool = False,
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
    
        if random_seed is not None and analysis_kwargs.get('n_workers', 1) is not None and analysis_kwargs.get('n_workers', 1) > 1:
            logger.warning("Reproducibility with random_seed is not guaranteed with n_workers > 1.")

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

        # Validate and apply spectral_mode
        _SPECTRAL_MODES = {'none', 'summary', 'full'}
        if spectral_mode not in _SPECTRAL_MODES:
            raise ValueError(
                f"spectral_mode='{spectral_mode}' is not valid. "
                f"Choose from {sorted(_SPECTRAL_MODES)}."
            )
        if spectral_mode == 'summary':
            _inject(base_params, 'track_spectral_metrics', True)
            _inject(base_params, 'spectral_output', 'default')
            _inject(base_params, 'return_spectrum', False)
        elif spectral_mode == 'full':
            _inject(base_params, 'track_spectral_metrics', True)
            _inject(base_params, 'spectral_output', 'all')
            _inject(base_params, 'return_spectrum', True)
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
                f"'conditional', or 'transfer' for permutation testing."
            )

        # Verify conditional MI input
        if z_data is not None and mode != 'conditional':
            logger.warning(
                f"z_data was provided but mode='{mode}' does not use it. "
                f"z_data is only consumed by mode='conditional'. "
                f"If you intended to compute conditional MI, set mode='conditional'."
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
        if _processor is None and str(_embedding).lower() in ('gru', 'lstm') and not _has_time_dim:
            raise ValueError(
                f"embedding_model='{_embedding}' requires sequential input but "
                f"processor_type=None produces a StaticDataset with no time dimension. "
                f"Either set processor_type to a windowed processor (e.g. 'continuous_window', "
                f"'spike_window') or switch embedding_model to 'mlp' / 'linear'."
            )
    
        run_params = {"mode": mode, "processor_type_x": processor_type_x, "processor_params_x": processor_params_x,
                      "processor_type_y": processor_type_y, "processor_params_y": processor_params_y,
                      "base_params": base_params, "sweep_grid": sweep_grid, "output_units": output_units,
                      "estimator": estimator, "random_seed": random_seed, "delta_threshold": delta_threshold,
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

        if is_proc_sweep or mode == 'lag':
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

            _warn_small_sample(dataset, base_params)

            if mode in ('dimensionality', 'pairwise'):
                # dimensionality and pairwise can operate on x_data alone
                x_run_data = dataset.x_data
                y_run_data = dataset.y_data if y_data is not None else None
            else:
                if y_data is None: raise ValueError(f"y_data must be provided for mode '{mode}'.")
                x_run_data = dataset.x_data
                y_run_data = dataset.y_data

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
                             params={**run_params, 'sweep_var': primary_sweep_var},
                             details={'raw_results': df})
            if _sweep_embeddings is not None:
                result.details.update(_sweep_embeddings)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs
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

            # Convert MI history lists to the requested units
            for _key in ('test_mi_history', 'train_mi_history'):
                if _key in res_dict and isinstance(res_dict[_key], list):
                    res_dict[_key] = [
                        v * NATS_TO_BITS if (to_bits and not np.isnan(v)) else v
                        for v in res_dict[_key]
                    ]

            result = Results(mode=mode,
                             mi_estimate=mi,
                             params=run_params,
                             details=res_dict)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'dimensionality':
            df, _dim_embeddings = run_dimensionality_analysis(
                x_run_data, base_params, y_data=y_run_data,
                sweep_grid=sweep_grid,
                processor_type_x=processor_type_x, processor_type_y=processor_type_y,
                **analysis_kwargs)
            df = _convert_mi_units(df, output_units == 'bits')
            group_vars = [key for key in (sweep_grid or {}).keys() if key != 'run_id']
            if 'sigma_add' in df.columns and 'sigma_add' not in group_vars:
                # Noise-injection ladder: group per rung so mi_mean/pr_*_mean reflect
                # the across-split spread at each sigma_add level, not a mix of levels.
                group_vars.append('sigma_add')
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
                if 'sigma_add_ladder' in _dim_embeddings:
                    # Keep the ladder's mi_mean/mi_std in the same units as
                    # result.dataframe (ceiling_nats is always nats, by name).
                    _dim_embeddings['sigma_add_ladder'] = _convert_mi_units(
                        _dim_embeddings['sigma_add_ladder'], output_units == 'bits')
                result.details.update(_dim_embeddings)
            if permutation_test and y_run_data is not None:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs
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
            analysis_kwargs.update({'delta_threshold': delta_threshold,
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
            # Accept lag_range as a top-level argument or from **analysis_kwargs
            lag_range_val = lag_range if lag_range is not None else analysis_kwargs.pop('lag_range', None)
            if lag_range_val is None:
                raise ValueError(
                    "`lag_range` must be provided for mode='lag'. "
                    "Pass it as a top-level argument: "
                    "nmi.run(..., mode='lag', lag_range=range(-10, 11))."
                )
            results_list = run_lag_analysis(x_run_data, y_run_data, base_params,
                                            lag_range=lag_range_val, sweep_grid=sweep_grid,
                                            **analysis_kwargs)
            df = pd.DataFrame(results_list)
            group_vars = ['lag']
            if sweep_grid:
                group_vars.extend([key for key in sweep_grid.keys() if key != 'run_id'])
            valid_group_vars = [var for var in group_vars if var in df.columns]
            if valid_group_vars:
                agg_df = _hashable_group_vars(df, valid_group_vars).groupby(valid_group_vars)['train_mi'].agg(
                    ['mean', 'std']).reset_index().rename(
                    columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0)
            else:
                agg_df = df
            result = Results(mode=mode,
                             dataframe=_convert_mi_units(agg_df, output_units == 'bits'),
                             params={**run_params, 'sweep_var': 'lag'},
                             details={'raw_results': df})
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, mode, sweep_grid,
                    n_permutations, analysis_kwargs, lag_range=lag_range_val
                )
                result.details['null_distribution'] = _null_clipped
                result.details['null_distribution_raw'] = _null_raw
            return result

        elif mode == 'conditional':
            if z_data is None:
                raise ValueError("`z_data` must be provided for mode='conditional'.")
            # Process z_data if a processor type is given; otherwise assume pre-processed
            if z_processor_type is not None:
                from .data.handler import create_dataset as _cds
                z_dataset = _cds(
                    x_data=z_data, y_data=None,
                    x_time=z_time,
                    processor_type_x=z_processor_type,
                    processor_params_x=z_processor_params or {}
                )
                z_run_data = z_dataset.x_data
            else:
                z_run_data = z_data if torch.is_tensor(z_data) else torch.from_numpy(np.array(z_data)).float()
            n_workers = analysis_kwargs.get('n_workers', 1)
            use_rigorous = analysis_kwargs.pop('rigorous', False)
            if use_rigorous:
                from .analysis.rigorous import run_rigorous_scalar_analysis
                _gamma_range = analysis_kwargs.pop('gamma_range', None) or range(1, 11)
                _rig_kwargs = {
                    'gamma_range': _gamma_range,
                    'delta_threshold': analysis_kwargs.pop('delta_threshold', delta_threshold),
                    'min_gamma_points': analysis_kwargs.pop('min_gamma_points', min_gamma_points),
                    'confidence_level': analysis_kwargs.pop('confidence_level', confidence_level),
                    'residual_threshold': analysis_kwargs.pop('residual_threshold', 2.5),
                    'r2_threshold': analysis_kwargs.pop('r2_threshold', 0.90),
                    'leverage_threshold': analysis_kwargs.pop('leverage_threshold', 0.20),
                }
                def _cmi_scalar(x_s, y_s, bp, z_data=None, **kw):
                    raw = run_conditional_mi(x_s, y_s, z_data, bp,
                                             sweep_grid=sweep_grid,
                                             n_workers=kw.get('n_workers', 1))
                    return raw['cmi_estimate']
                rig_details = run_rigorous_scalar_analysis(
                    scalar_fn=_cmi_scalar,
                    x_data=x_run_data, y_data=y_run_data, base_params=base_params,
                    extra_data={'z_data': z_run_data},
                    extra_kwargs={'n_workers': n_workers},
                    **_rig_kwargs,
                )
                rig_details = _convert_mi_units(rig_details, output_units == 'bits')
                raw_df = rig_details.pop('raw_results_df', pd.DataFrame())
                raw_df = _convert_mi_units(raw_df, output_units == 'bits')
                return Results(
                    mode=mode,
                    mi_estimate=rig_details.get('mi_corrected'),
                    dataframe=raw_df,
                    params={**run_params, 'rigorous': True},
                    details=rig_details,
                )
            # Standard (non-rigorous) path
            raw = run_conditional_mi(x_run_data, y_run_data, z_run_data, base_params,
                                     sweep_grid=sweep_grid, n_workers=n_workers)
            raw = _convert_mi_units(raw, output_units == 'bits')
            cmi = raw['cmi_estimate']
            result = Results(mode=mode, mi_estimate=cmi, params=run_params, details=raw)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    x_run_data, y_run_data, base_params, 'conditional', sweep_grid,
                    n_permutations, analysis_kwargs, z_data=z_run_data
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
            n_workers = analysis_kwargs.get('n_workers', 1)
            use_rigorous = analysis_kwargs.pop('rigorous', False)
            if use_rigorous:
                from .analysis.rigorous import run_rigorous_scalar_analysis
                _gamma_range = analysis_kwargs.pop('gamma_range', None) or range(1, 11)
                _rig_kwargs = {
                    'gamma_range': _gamma_range,
                    'delta_threshold': analysis_kwargs.pop('delta_threshold', delta_threshold),
                    'min_gamma_points': analysis_kwargs.pop('min_gamma_points', min_gamma_points),
                    'confidence_level': analysis_kwargs.pop('confidence_level', confidence_level),
                    'residual_threshold': analysis_kwargs.pop('residual_threshold', 2.5),
                    'r2_threshold': analysis_kwargs.pop('r2_threshold', 0.90),
                    'leverage_threshold': analysis_kwargs.pop('leverage_threshold', 0.20),
                }
                def _te_scalar(x_s, y_s, bp, **kw):
                    raw = run_transfer_entropy(
                        x_s, y_s, bp,
                        history_window=history_window,
                        prediction_horizon=prediction_horizon,
                        sweep_grid=sweep_grid,
                        n_workers=kw.get('n_workers', 1),
                        bidirectional=bidirectional_te,
                    )
                    return raw['te_estimate']
                rig_details = run_rigorous_scalar_analysis(
                    scalar_fn=_te_scalar,
                    x_data=_x_te, y_data=_y_te, base_params=base_params,
                    extra_data=None,
                    extra_kwargs={'n_workers': n_workers},
                    **_rig_kwargs,
                )
                rig_details = _convert_mi_units(rig_details, output_units == 'bits')
                raw_df = rig_details.pop('raw_results_df', pd.DataFrame())
                raw_df = _convert_mi_units(raw_df, output_units == 'bits')
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
                                       bidirectional=bidirectional_te)
            raw = _convert_mi_units(raw, output_units == 'bits')
            te = raw['te_estimate']
            result = Results(mode=mode, mi_estimate=te, params=run_params, details=raw)
            if permutation_test:
                _null_clipped, _null_raw = _run_permutation_test(
                    _x_te, _y_te, base_params, 'transfer', sweep_grid,
                    n_permutations, analysis_kwargs,
                    history_window=history_window, prediction_horizon=prediction_horizon,
                    bidirectional_te=bidirectional_te,
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
                n_ch_x = x_run_data.shape[1] if x_run_data.ndim >= 2 else 1
                if pairwise_y is not None:
                    n_pairs_est = n_ch_x * pairwise_y.shape[1]
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
            _scale = 1 / np.log(2) if output_units == 'bits' else 1.0
            raw['mi_matrix'] = raw['mi_matrix'] * _scale
            df = raw['dataframe'].copy()
            for _col in ('mi_mean', 'mi_std', 'mi_estimate'):
                if _col in df.columns:
                    df[_col] = df[_col] * _scale
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
                        n_permutations, analysis_kwargs, pairs=pairs
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
                f"'lag', 'precision', 'conditional', 'transfer', 'pairwise'."
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


def _run_single_permutation(args):
    """Top-level picklable function for one permutation trial.

    Parameters
    ----------
    args : tuple
        ``(x_data, y_data, base_params, mode, sweep_grid, perm_seed, mode_kwargs)``

    Returns
    -------
    tuple[float, float]
        ``(mi_clipped, mi_raw)`` where *mi_clipped* matches the main-run
        convention (negatives zeroed by the trainer's ``all_mi_negative`` guard)
        and *mi_raw* retains the actual value including negatives.
    """
    import numpy as _np
    import torch as _torch
    x_data, y_data, base_params, mode, sweep_grid, perm_seed, mode_kwargs = args
    _np.random.seed(perm_seed)
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
            z_data = mode_kwargs.get('z_data')
            raw = _rcmi(x_data, y_perm, z_data, base_params.copy(), n_workers=1)
            mi_clipped = raw['cmi_estimate']
            # Raw CMI = mean(raw_train_mi of XZ→Y sweep) − mean(raw_train_mi of Z→Y sweep)
            _rxz = [r.get('raw_train_mi', _nan) for r in raw.get('raw_xz_y', [])]
            _rz = [r.get('raw_train_mi', _nan) for r in raw.get('raw_z_y', [])]
            mi_raw = (float(_np.nanmean(_rxz)) - float(_np.nanmean(_rz))
                      if _rxz and _rz else mi_clipped)
            return mi_clipped, mi_raw

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
                          n_permutations, analysis_kwargs, **mode_kwargs):
    """Run the permutation null test by shuffling y_data *n_permutations* times.

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
    logger.info(
        f"Permutation test: running {n_permutations} permutations for "
        f"mode='{mode}' (n_workers={n_workers})..."
    )

    # Generate independent seeds so parallel workers produce different shuffles
    perm_seeds = [int(np.random.randint(0, 2**31)) for _ in range(n_permutations)]
    perm_args = [
        (x_data, y_data, base_params.copy(), mode, sweep_grid, seed, dict(mode_kwargs))
        for seed in perm_seeds
    ]

    if n_workers > 1:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            raw_results = list(tqdm(
                pool.imap(_run_single_permutation, perm_args),
                total=n_permutations,
                desc="Permutation test",
                leave=False,
            ))
    else:
        raw_results = [
            _run_single_permutation(args)
            for args in tqdm(perm_args, desc="Permutation test", leave=False)
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







