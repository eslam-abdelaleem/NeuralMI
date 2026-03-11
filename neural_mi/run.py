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
import platform
import os
import tempfile
import torch.multiprocessing as mp
from typing import Union, Optional, Dict, Any, List
import random

from .analysis.rigorous import run_rigorous_analysis
from .analysis.dimensionality import run_dimensionality_analysis
from .analysis.precision import run_precision_analysis
from .analysis.lag import run_lag_analysis
from .analysis.conditional import run_conditional_mi
from .analysis.transfer import run_transfer_entropy
from .analysis.pairwise import run_pairwise_mi
from .data.handler import create_dataset
from .estimators import ESTIMATORS
from .results import Results
from .validation import ParameterValidator, DataValidator
from .utils import get_device
from .logger import logger, set_verbosity


def _convert_mi_units(results: Any, to_bits: bool) -> Any:
    """Recursively converts MI values in results from nats to bits."""
    if not to_bits: return results
    NATS_TO_BITS = 1 / np.log(2)
    if isinstance(results, float): return results * NATS_TO_BITS
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
        cols = ['test_mi', 'train_mi', 'mi_mean', 'mi_std', 'mi_corrected', 'mi_error', 'slope']
        for col in cols:
            if col in df.columns: df[col] *= NATS_TO_BITS
        return df
    elif isinstance(results, list) and all(isinstance(r, dict) for r in results):
        keys = ['test_mi', 'train_mi', 'mi_corrected', 'mi_error', 'slope']
        return [{**r, **{k: r.get(k, 0) * NATS_TO_BITS for k in keys if r.get(k) is not None}} for r in results]
    elif isinstance(results, dict):
        new_results = results.copy()
        # Scalar MI values stored by analysis modules (transfer entropy, CMI, etc.)
        _MI_SCALAR_KEYS = (
            'te_estimate', 'i_xypast_yfuture', 'i_ypast_yfuture',
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

def run(
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
    track_spectral_metrics: bool = False,
    spectral_output: str = 'default',
    return_spectrum: bool = False,
    max_index_reduction: float = 0.05,
    tau_grid: Optional[List[float]] = None,
    corrupt_target: str = 'x',
    corruption_method: str = 'rounding',
    n_noise_samples: int = 50,
    threshold_ratio: float = 0.9,
    permutation_test: bool = False,
    n_permutations: int = 1,
    z_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    **analysis_kwargs
) -> Results:
    
    """The unified entry point for all analyses in the NeuralMI library.
    
    This function provides a single, consistent interface for various mutual
    information estimation workflows. It handles data validation, processing,
    model training, and results aggregation, returning a standardized
    :class:`~neural_mi.results.Results` object that can be easily
    inspected and plotted.
    
    Parameters
    ----------
    x_data : np.ndarray, torch.Tensor, or list
        The data for variable X. The required format depends on ``processor_type``:
        
        - 'continuous' or 'categorical': A 2D array, typically of shape
          (n_channels, n_timepoints). Data of shape (n_timepoints, n_channels)
          is also supported and will be transposed automatically.
        - 'spike': A list of 1D NumPy arrays, where each array contains
          spike times for a single channel/neuron.
    y_data : np.ndarray, torch.Tensor, or list, optional
        The data for variable Y. Required for all modes except 'dimensionality'.
        Should have the same format as ``x_data``. Defaults to None.
    x_time : np.ndarray, optional
        Time vector for `x_data`. Required for temporal datasets. Defaults to None.
    y_time : np.ndarray, optional
        Time vector for `y_data`. Required for temporal datasets. Defaults to None.
    mode : {'estimate', 'sweep', 'dimensionality', 'rigorous', 'lag', 'precision', 'conditional', 'transfer', 'pairwise'}, default='estimate'
        The analysis mode to run.
    processor_type_x : {'continuous', 'spike', 'categorical'}, optional
        The type of processing to apply to raw X data. If None, data is assumed
        to be pre-processed. Defaults to None.
    processor_params_x : dict, optional
        Parameters for the X data processor. The key ``window_size`` sets the
        width of each analysis window **in the same time units as** ``x_time``
        (e.g., seconds if ``x_time`` is in seconds; a value of 0.05 gives 50 ms
        windows when time is in seconds). ``window_size`` is not applicable when
        ``processor_type_x`` is None, i.e., when data are already pre-processed
        and treated as IID samples. Example: ``{'window_size': 0.05}``.
        Defaults to None.
    processor_type_y : {'continuous', 'spike', 'categorical'}, optional
        The type of processing to apply to raw Y data. If None, data is assumed
        to be pre-processed. Defaults to None.
    processor_params_y : dict, optional
        Parameters for the Y data processor, e.g., ``{'window_size': 10}``.
        Defaults to None.
    base_params : dict, optional
        A dictionary of fixed parameters for the MI estimator's trainer. These
        are used for all runs. Common parameters include ``n_epochs``,
        ``learning_rate``, ``batch_size``, ``embedding_dim``, etc. Defaults to {}.
    sweep_grid : dict, optional
        A dictionary defining the parameter grid for 'sweep' and
        'dimensionality' modes. Keys are parameter names and values are lists
        of values to test, e.g., ``{'embedding_dim': [8, 16, 32]}``.
        Defaults to None.
    output_units : {'bits', 'nats'}, default='bits'
        The units for the final MI estimate.
    estimator : {'infonce', 'smile', 'nwj', 'tuba', 'js'}, default='infonce'
        The MI lower bound estimator to use. Recommended choices:

        - **'infonce'** *(default)* — InfoNCE lower bound. Best all-around
          default. Low variance, stable training. Theoretical ceiling at
          ``log(batch_size)`` nats (≈ 5 bits for batch_size=128, ≈ 6.6 bits
          for batch_size=512). If your true MI is near this ceiling, estimates
          will be systematically biased downward — switch to 'smile'.
        - **'smile'** — SMILE estimator. No hard ceiling on the MI estimate.
          Recommended when you expect MI > 3–4 bits or are not sure about the
          scale. Use with ``estimator_params={'clip': 5.0}`` to stabilise
          training: ``nmi.run(..., estimator='smile', estimator_params={'clip': 5.0})``.
        - 'nwj', 'tuba', 'js' — Other valid lower bounds, included for
          completeness. Higher variance than InfoNCE and generally not
          recommended for practical use.
    estimator_params : dict, optional
        Additional keyword arguments for the selected estimator function.
        For example, ``{'clip': 5.0}`` for the 'smile' estimator. Defaults to None.
    custom_critic : torch.nn.Module, optional
        A pre-initialized custom critic model. If provided, all internal model
        building is skipped. ``base_params`` related to model architecture will be
        ignored. Defaults to None.
    custom_embedding_cls : type, optional
        A user-defined embedding model class (not an instance) to be used
        within the library's standard critic structures. Defaults to None.
    save_best_model_path : str, optional
        If provided, the file path where the state dictionary of the
        best-performing trained critic model will be saved. Defaults to None.
    random_seed : int, optional
        A seed for ``random``, ``numpy``, and ``torch`` to ensure reproducibility.
        Note: Full reproducibility is only guaranteed for ``n_workers=1``.
        Defaults to None.
    verbose : bool, default=False
        If True, informational logs and defaults will be displayed.
    show_progress : bool, default=True
        If True, progress bars will be displayed during training.
    device : str, optional
        The compute device to use (e.g., 'cpu', 'cuda', 'mps'). If None, it
        is auto-detected. Defaults to None.
    split_mode : {'blocked', 'random'}, default='blocked'
        Method for splitting data. 'blocked' is for time-series, 'random' for IID.
        Ignored if train/test indices are provided.
    train_fraction : float, default=0.9
        The fraction of data to use for training when creating splits. Ignored if train/test indices are provided.
    n_test_blocks : int, default=5
        For 'blocked' split mode, the number of contiguous blocks to use for testing. Ignored if train/test indices are provided.
    split_gap_fraction : float, default=0.5
        When using 'blocked' split_mode, this fraction of each test-block length
        is excluded from **both sides** of every test block and removed from the
        training set. This creates a buffer that prevents temporal leakage between
        adjacent train and test windows. For example, with block_size=100 samples
        and split_gap_fraction=0.5, 50 samples on each side of every test block
        are withheld from training. Set to 0.0 to disable the gap.
    train_indices : np.ndarray, optional
        Specific indices for the training set.
    test_indices : np.ndarray, optional
        Specific indices for the test set.
    delta_threshold : float, default=0.1
        For ``mode='rigorous'``, the curvature threshold for determining the
        linear region of the MI vs. 1/gamma plot. Lower values enforce
        stricter linearity.
    min_gamma_points : int, default=5
        For ``mode='rigorous'``, the minimum number of gamma values required to
        perform a reliable extrapolation fit after pruning non-linear points.
    confidence_level : float, default=0.68
        For ``mode='rigorous'``, the confidence level (e.g., 0.68 for ~1 std
        dev) used for the final MI estimate's error bars.
    n_epochs : int, optional
        Top-level shortcut for ``base_params['n_epochs']``. If provided and
        ``base_params`` also contains ``'n_epochs'``, this value takes
        precedence. Defaults to None (use the value in ``base_params``).
    batch_size : int, optional
        Top-level shortcut for ``base_params['batch_size']``. If provided and
        ``base_params`` also contains ``'batch_size'``, this value takes
        precedence. Defaults to None (use the value in ``base_params``).
    lag_range : iterable of int or float, optional
        For ``mode='lag'``, the range of time-lags to evaluate, e.g.
        ``range(-10, 11)`` or ``[-0.1, 0.0, 0.1]``. Can also be passed via
        ``**analysis_kwargs`` for backward compatibility. Required when
        ``mode='lag'``.
    **analysis_kwargs
        Additional keyword arguments passed to the specific analysis engine.
        Common examples include ``n_workers``, ``n_splits``, or ``gamma_range``.
    max_eval_samples : int, default=5000
        The maximum number of samples to use for evaluation during training.
        This is a computational safeguard and does not affect the training data size.
    train_subset_size : int, optional
        If provided, the number of training samples to use when evaluating at the end of each epoch. This can speed up training on large datasets. Defaults to None (use all training data).
    spectral_mode : {'none', 'summary', 'full'}, default='none'
        Consolidated control for spectral metric tracking. Replaces the
        deprecated ``track_spectral_metrics`` / ``spectral_output`` /
        ``return_spectrum`` trio.

        - ``'none'`` — no spectral metrics computed (default, no overhead).
        - ``'summary'`` — compute participation ratio at the end of training;
          equivalent to ``track_spectral_metrics=True, spectral_output='default',
          return_spectrum=False``.
        - ``'full'`` — compute all spectral metrics (participation ratio, all
          singular values) and include the raw spectrum in results; equivalent to
          ``track_spectral_metrics=True, spectral_output='all',
          return_spectrum=True``.
    track_spectral_metrics : bool, default=False
        *Deprecated* — use ``spectral_mode='summary'`` instead. If True,
        spectral metrics will be computed during training.
    spectral_output : str, default='default'
        *Deprecated* — use ``spectral_mode`` instead.
    return_spectrum : bool, default=False
        *Deprecated* — use ``spectral_mode='full'`` instead.
    max_index_reduction : float, default=0.05
        When using temporal datasets with windowing, random time shifting can reduce the number of valid windows
        due to edge effects. This parameter sets a threshold for acceptable reduction in valid windows after shifting.
        If the reduction exceeds this threshold, a warning is logged. Defaults to 0.05 (5%).
    tau_grid : list of float, optional
        For ``mode='precision'``, a list of corruption levels to sweep over. Each value is
        a precision parameter *tau* applied to the target variable. With
        ``corruption_method='rounding'`` (default), values are quantized to the nearest
        multiple of *tau*. With ``corruption_method='noise'``, additive uniform noise drawn
        from U(-tau/2, tau/2) is applied. Defaults to None.
    corrupt_target : {'x', 'y'}, default='x'
        For ``mode='precision'``, which variable to apply noise corruption to during the precision sweep. Defaults to 'x'.
    corruption_method : {'rounding', 'noise'}, default='rounding'
        The method for corrupting the target variable in the precision sweep. 'rounding' rounds values to the nearest multiple of tau, while 'noise' adds uniform noise in the range [-tau/2, tau/2]. Defaults to 'rounding'.
    n_noise_samples : int, default=50
        For ``mode='precision'`` with ``corruption_method='noise'``, the number of noise realizations to average over for each tau value to stabilize the MI estimates. Defaults to 50.
    threshold_ratio : float, default=0.9
        For ``mode='precision'``, the ratio of the baseline MI used to determine the precision threshold. For example, a value of 0.9 means the precision threshold is the tau value at which the MI drops to 90% of the baseline MI. Defaults to 0.9

    Returns
    -------
    neural_mi.results.Results
        A standardized object containing the analysis results, which can be
        inspected as a dataframe or plotted directly via its ``.plot()`` method.
    
    Examples
    --------
    Perform a rigorous, bias-corrected MI estimation between two variables.
    
    >>> import neural_mi as nmi
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> x_raw, y_raw = nmi.generators.generate_nonlinear_from_latent(
    ...     n_samples=2500, latent_dim=10, observed_dim=100, mi=3.0
    ... )
    >>> # Define model and training parameters
    >>> base_params = {
    ...     'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    ...     'embedding_dim': 16, 'hidden_dim': 64
    ... }
    >>> # Run the analysis
    >>> results = nmi.run(
    ...     x_data=x_raw.T, y_data=y_raw.T,
    ...     mode='rigorous',
    ...     processor_type_x='continuous',
    ...     processor_params_x={'window_size': 1},
    ...     base_params=base_params,
    ...     n_workers=4,
    ...     random_seed=42
    ... )
    >>> mi_est = results.mi_estimate
    >>> mi_err = results.details.get('mi_error', 0.0)
    >>> print(f"Corrected MI: {mi_est:.3f} ± {mi_err:.3f} bits")
    Corrected MI: 2.953 ± 0.081 bits
    
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
        return _run_inner(
            x_data=x_data, y_data=y_data, x_time=x_time, y_time=y_time,
            mode=mode, processor_type_x=processor_type_x, processor_params_x=processor_params_x,
            processor_type_y=processor_type_y, processor_params_y=processor_params_y,
            base_params=base_params, sweep_grid=sweep_grid, output_units=output_units,
            estimator=estimator, estimator_params=estimator_params,
            custom_critic=custom_critic, custom_embedding_cls=custom_embedding_cls,
            save_best_model_path=save_best_model_path, random_seed=random_seed,
            verbose=verbose, show_progress=show_progress, device=device,
            split_mode=split_mode, train_fraction=train_fraction, n_test_blocks=n_test_blocks,
            split_gap_fraction=split_gap_fraction, train_indices=train_indices,
            test_indices=test_indices, delta_threshold=delta_threshold,
            min_gamma_points=min_gamma_points, confidence_level=confidence_level,
            max_eval_samples=max_eval_samples, train_subset_size=train_subset_size,
            spectral_mode=spectral_mode,
            track_spectral_metrics=track_spectral_metrics, spectral_output=spectral_output,
            return_spectrum=return_spectrum, max_index_reduction=max_index_reduction,
            tau_grid=tau_grid, corrupt_target=corrupt_target,
            corruption_method=corruption_method, n_noise_samples=n_noise_samples,
            threshold_ratio=threshold_ratio, permutation_test=permutation_test,
            n_permutations=n_permutations, z_data=z_data, z_processor_type=z_processor_type,
            z_processor_params=z_processor_params, history_window=history_window,
            prediction_horizon=prediction_horizon, bidirectional_te=bidirectional_te,
            n_epochs=n_epochs, batch_size=batch_size, shared_encoder=shared_encoder,
            return_embeddings=return_embeddings, lag_range=lag_range,
            **analysis_kwargs
        )
    finally:
        logger.setLevel(_prev_level)
        for h, lv in zip(logger.handlers, _prev_handler_levels):
            h.setLevel(lv)


def _run_inner(
    x_data, y_data, x_time, y_time, mode, processor_type_x, processor_params_x,
    processor_type_y, processor_params_y, base_params, sweep_grid, output_units,
    estimator, estimator_params, custom_critic, custom_embedding_cls,
    save_best_model_path, random_seed, verbose, show_progress, device,
    split_mode, train_fraction, n_test_blocks, split_gap_fraction, train_indices,
    test_indices, delta_threshold, min_gamma_points, confidence_level,
    max_eval_samples, train_subset_size, track_spectral_metrics, spectral_output,
    return_spectrum, max_index_reduction, tau_grid, corrupt_target,
    corruption_method, n_noise_samples, threshold_ratio, permutation_test,
    n_permutations, z_data, z_processor_type, z_processor_params,
    history_window, prediction_horizon, bidirectional_te=False,
    n_epochs=None, batch_size=None, shared_encoder=None,
    return_embeddings=False, lag_range=None,
    spectral_mode='none',
    **analysis_kwargs
):
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
    _inject(base_params, 'estimator_params', estimator_params or {})
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

    # Validate spectral_mode
    _SPECTRAL_MODES = {'none', 'summary', 'full'}
    if spectral_mode not in _SPECTRAL_MODES:
        raise ValueError(
            f"spectral_mode='{spectral_mode}' is not valid. "
            f"Choose from {sorted(_SPECTRAL_MODES)}."
        )
    if spectral_mode != 'none':
        if track_spectral_metrics or spectral_output != 'default' or return_spectrum:
            logger.warning(
                "Both `spectral_mode` and individual spectral parameters were specified. "
                "`spectral_mode` takes precedence."
            )
        if spectral_mode == 'summary':
            track_spectral_metrics, spectral_output, return_spectrum = True, 'default', False
        else:  # 'full'
            track_spectral_metrics, spectral_output, return_spectrum = True, 'all', True
    elif track_spectral_metrics or spectral_output != 'default' or return_spectrum:
        logger.warning(
            "The `track_spectral_metrics`, `spectral_output`, and `return_spectrum` "
            "parameters are deprecated. Use spectral_mode='summary' or "
            "spectral_mode='full' instead. These parameters will be removed in a "
            "future release."
        )

    _inject(base_params, 'track_spectral_metrics', track_spectral_metrics)
    _inject(base_params, 'spectral_output', spectral_output)
    _inject(base_params, 'return_spectrum', return_spectrum)
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
    param_validator = ParameterValidator(locals())
    param_validator.validate()
    param_validator.apply_defaults()

    DataValidator(x_data, y_data, processor_type_x, processor_type_y).validate()
    
    _processor = base_params.get('processor_type_x', None)
    _embedding = base_params.get('embedding_model', 'mlp')
    if _processor is None and str(_embedding).lower() in ('gru', 'lstm'):
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

    processor_param_keys = ['window_size', 'n_seconds', 'max_spikes_per_window', 'data_format']
    is_proc_sweep = mode == 'sweep' and any(key in (sweep_grid or {}) for key in processor_param_keys)
    
    if is_proc_sweep or mode == 'lag':
        logger.info("Detected sweep over processor or lag parameters. Deferring data processing to workers.")
        x_run_data, y_run_data = x_data, y_data
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
        df = pd.DataFrame(results_list)
        group_vars = [key for key in sweep_grid.keys() if key != 'run_id']
        agg_df = df.groupby(group_vars)['test_mi'].agg(['mean', 'std']).reset_index().rename(
            columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0) if group_vars else df
        primary_sweep_var = group_vars[0] if group_vars else None
        result = Results(mode=mode,
                         dataframe=_convert_mi_units(agg_df, output_units == 'bits'),
                         params={**run_params, 'sweep_var': primary_sweep_var},
                         details={'raw_results': df})
        if permutation_test:
            result.details['null_distribution'] = _run_permutation_test(
                x_run_data, y_run_data, base_params, mode, sweep_grid,
                n_permutations, analysis_kwargs
            )
        return result

    elif mode == 'estimate':
        results_list = ParameterSweep(x_run_data, y_run_data, base_params).run(
            sweep_grid or {}, **analysis_kwargs)
        if not results_list:
            return Results(mode=mode, mi_estimate=float('nan'), params=run_params)
        res_dict = results_list[0].copy()
        mi = res_dict.pop('test_mi', float('nan'))
        result = Results(mode=mode,
                         mi_estimate=_convert_mi_units(mi, output_units == 'bits'),
                         params=run_params,
                         details=res_dict)
        if permutation_test:
            result.details['null_distribution'] = _run_permutation_test(
                x_run_data, y_run_data, base_params, mode, sweep_grid,
                n_permutations, analysis_kwargs
            )
        return result

    elif mode == 'dimensionality':
        df = run_dimensionality_analysis(x_run_data, base_params, y_data=y_run_data,
                                         sweep_grid=sweep_grid, **analysis_kwargs)
        df = _convert_mi_units(df, output_units == 'bits')
        group_vars = [key for key in (sweep_grid or {}).keys() if key != 'run_id']
        metrics = ['test_mi', 'participation_ratio']
        valid_metrics = [m for m in metrics if m in df.columns]
        if group_vars:
            agg_df = df.groupby(group_vars)[valid_metrics].agg(['mean', 'std']).reset_index()
            agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns.values]
            rename_map = {f'{m}_mean': 'mi_mean' if m == 'test_mi' else f'{m}_mean' for m in valid_metrics}
            rename_map.update({f'{m}_std': 'mi_std' if m == 'test_mi' else f'{m}_std' for m in valid_metrics})
            agg_df = agg_df.rename(columns=rename_map).fillna(0)
        else:
            agg_data = {f'{m}_mean': df[m].mean() for m in valid_metrics}
            agg_data.update({f'{m}_std': df[m].std() for m in valid_metrics})
            if 'test_mi_mean' in agg_data:
                agg_data['mi_mean'] = agg_data.pop('test_mi_mean')
            if 'test_mi_std' in agg_data:
                agg_data['mi_std'] = agg_data.pop('test_mi_std')
            agg_df = pd.DataFrame([agg_data])
        result = Results(mode=mode, dataframe=agg_df, params={**run_params},
                         details={'raw_results': df})
        if permutation_test and y_run_data is not None:
            result.details['null_distribution'] = _run_permutation_test(
                x_run_data, y_run_data, base_params, mode, sweep_grid,
                n_permutations, analysis_kwargs
            )
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
        details['raw_results'] = df
        return Results(
            mode=mode,
            mi_estimate=details['precision_tau'],
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
        # Accept lag_range as explicit param (preferred) or via **analysis_kwargs (legacy)
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
            agg_df = df.groupby(valid_group_vars)['test_mi'].agg(['mean', 'std']).reset_index().rename(
                columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0)
        else:
            agg_df = df
        result = Results(mode=mode,
                         dataframe=_convert_mi_units(agg_df, output_units == 'bits'),
                         params={**run_params, 'sweep_var': 'lag'},
                         details={'raw_results': df})
        if permutation_test:
            result.details['null_distribution'] = _run_permutation_test(
                x_run_data, y_run_data, base_params, mode, sweep_grid,
                n_permutations, analysis_kwargs, lag_range=lag_range_val
            )
        return result

    elif mode == 'conditional':
        if z_data is None:
            raise ValueError("`z_data` must be provided for mode='conditional'.")
        # Process z_data if a processor type is given; otherwise assume pre-processed
        if z_processor_type is not None:
            from .data.handler import create_dataset as _cds
            z_dataset = _cds(
                x_data=z_data, y_data=None,
                processor_type_x=z_processor_type,
                processor_params_x=z_processor_params or {}
            )
            z_run_data = z_dataset.x_data
        else:
            z_run_data = z_data if torch.is_tensor(z_data) else torch.from_numpy(np.array(z_data)).float()
        n_workers = analysis_kwargs.get('n_workers', 1)
        raw = run_conditional_mi(x_run_data, y_run_data, z_run_data, base_params,
                                 sweep_grid=sweep_grid, n_workers=n_workers)
        raw = _convert_mi_units(raw, output_units == 'bits')  # convert all MI scalars at once
        cmi = raw['cmi_estimate']
        result = Results(mode=mode, mi_estimate=cmi, params=run_params, details=raw)
        if permutation_test:
            result.details['null_distribution'] = _run_permutation_test(
                x_run_data, y_run_data, base_params, 'conditional', sweep_grid,
                n_permutations, analysis_kwargs, z_data=z_run_data
            )
        return result

    elif mode == 'transfer':
        if history_window is None:
            raise ValueError("`history_window` must be provided for mode='transfer'.")
        # For transfer entropy, expect 2-D inputs (T, channels).
        # StaticDataset wraps (T, C) → (T, C, 1); squeeze the trailing 1 back out.
        # Use .reshape().contiguous() — StaticDataset may produce non-contiguous
        # views and downstream code uses .view(), which requires contiguous memory.
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
        raw = run_transfer_entropy(_x_te, _y_te, base_params,
                                   history_window=history_window,
                                   prediction_horizon=prediction_horizon,
                                   sweep_grid=sweep_grid, n_workers=n_workers,
                                   bidirectional=analysis_kwargs.pop('bidirectional_te', bidirectional_te))
        raw = _convert_mi_units(raw, output_units == 'bits')  # convert all MI scalars at once
        te = raw['te_estimate']
        result = Results(mode=mode, mi_estimate=te, params=run_params, details=raw)
        if permutation_test:
            result.details['null_distribution'] = _run_permutation_test(
                _x_te, _y_te, base_params, 'transfer', sweep_grid,
                n_permutations, analysis_kwargs,
                history_window=history_window, prediction_horizon=prediction_horizon
            )
        return result

    elif mode == 'pairwise':
        n_workers = analysis_kwargs.get('n_workers', 1)
        pairs = analysis_kwargs.get('pairs', None)
        # Pass y_run_data when provided to enable cross-pairwise mode.
        pairwise_y = y_run_data if y_data is not None else None
        raw = run_pairwise_mi(x_run_data, base_params, y_data=pairwise_y,
                              sweep_grid=sweep_grid, n_workers=n_workers, pairs=pairs)
        # Convert mi_matrix and dataframe
        raw['mi_matrix'] = raw['mi_matrix'] * (1 / np.log(2) if output_units == 'bits' else 1.0)
        raw['dataframe']['mi_estimate'] *= (1 / np.log(2) if output_units == 'bits' else 1.0)
        return Results(mode=mode, params=run_params, details=raw,
                       dataframe=raw['dataframe'])

    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            f"Expected one of: 'estimate', 'sweep', 'dimensionality', 'rigorous', "
            f"'lag', 'precision', 'conditional', 'transfer', 'pairwise'."
        )


def _run_permutation_test(x_data, y_data, base_params, mode, sweep_grid,
                          n_permutations, analysis_kwargs, **mode_kwargs):
    """Runs the permutation test by shuffling y_data n_permutations times.

    Returns a list of null MI estimates (one per permutation).
    """
    from .analysis.sweep import ParameterSweep
    null_mis = []
    logger.info(f"Permutation test: running {n_permutations} permutations for mode='{mode}'...")
    n = y_data.shape[0] if hasattr(y_data, 'shape') else len(y_data)

    for perm_idx in range(n_permutations):
        shuffle_idx = np.random.permutation(n)
        if torch.is_tensor(y_data):
            y_perm = y_data[shuffle_idx]
        else:
            y_perm = [y_data[i] for i in shuffle_idx]

        try:
            if mode in ('estimate', 'sweep', 'dimensionality'):
                res = ParameterSweep(x_data, y_perm, base_params.copy()).run(
                    sweep_grid or {}, n_workers=analysis_kwargs.get('n_workers', 1),
                    is_proc_sweep=False
                )
                null_mis.append(float(np.mean([r.get('test_mi', float('nan')) for r in res])))

            elif mode == 'lag':
                from .analysis.lag import run_lag_analysis as _rla
                lag_range = mode_kwargs.get('lag_range')
                res = _rla(x_data, y_perm, base_params.copy(), lag_range=lag_range,
                           n_workers=analysis_kwargs.get('n_workers', 1))
                null_mis.append(float(np.mean([r.get('test_mi', float('nan')) for r in res])))

            elif mode == 'conditional':
                z_data = mode_kwargs.get('z_data')
                raw = run_conditional_mi(x_data, y_perm, z_data, base_params.copy(),
                                         n_workers=analysis_kwargs.get('n_workers', 1))
                null_mis.append(raw['cmi_estimate'])

            elif mode == 'transfer':
                hw = mode_kwargs.get('history_window')
                ph = mode_kwargs.get('prediction_horizon', 1)
                raw = run_transfer_entropy(x_data, y_perm, base_params.copy(),
                                           history_window=hw, prediction_horizon=ph,
                                           n_workers=analysis_kwargs.get('n_workers', 1))
                null_mis.append(raw['te_estimate'])

            else:
                null_mis.append(float('nan'))

        except Exception as e:
            logger.warning(f"Permutation {perm_idx + 1} failed: {e}")
            null_mis.append(float('nan'))

    logger.info(f"Permutation test complete. Null MI: mean={np.nanmean(null_mis):.4f}, "
                f"std={np.nanstd(null_mis):.4f}")
    return null_mis







