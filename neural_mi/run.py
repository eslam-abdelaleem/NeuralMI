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
from tqdm.auto import tqdm

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
from .defaults import PROCESSOR_PARAMS_SCHEMA


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
    estimator : {'infonce', 'smile'}, default='infonce'
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
        linear region of the MI vs. gamma plot. Lower values enforce
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
        ``range(-10, 11)`` or ``[-0.1, 0.0, 0.1]``. Required when
        ``mode='lag'``.
    use_spectral_norm : bool, default=True
        Apply spectral normalisation to the hidden linear layers of MLP
        embedding networks.  Spectral norm constrains the Lipschitz constant of
        each hidden layer (largest singular value = 1), improving training
        stability.  Has no effect on non-MLP architectures (CNN, GRU, LSTM,
        TCN, Transformer).
    gradient_clip_val : float, optional
        Maximum norm for gradient clipping (``torch.nn.utils.clip_grad_norm_``).
        Applied between ``loss.backward()`` and ``optimizer.step()`` each
        training step.  ``None`` (default) disables clipping.  A value of 1.0
        is a common starting point for preventing gradient explosions.
    optimizer : str or torch.optim.Optimizer subclass, default='adam'
        The optimizer to use for training. Can be a string name or a
        ``torch.optim.Optimizer`` subclass directly.

        Supported names: ``'adam'``, ``'adamw'``, ``'sgd'``, ``'rmsprop'``,
        ``'adagrad'``.

        - ``'adam'`` *(default)* — good general-purpose choice.
        - ``'adamw'`` — Adam with decoupled weight decay; preferred when using
          ``optimizer_params={'weight_decay': 1e-4}`` for regularisation.
        - ``'sgd'`` — requires setting ``optimizer_params={'momentum': 0.9}``
          for practical use; may generalise better on large datasets.
    optimizer_params : dict, optional
        Additional keyword arguments forwarded to the optimizer constructor
        (beyond ``lr``, which is always taken from ``base_params['learning_rate']``).
        For example, ``{'weight_decay': 1e-4}`` or ``{'momentum': 0.9}``.
        Defaults to ``None`` (treated as ``{}``).
    scheduler : str, torch.optim.lr_scheduler class, or None, default=None
        A learning-rate scheduler to apply at the end of each training epoch.
        ``None`` (default) disables scheduling. Supported string names:

        - ``'cosine'`` — ``CosineAnnealingLR`` completing one full cycle over
          ``n_epochs``. Best general-purpose choice; smoothly decays lr to near
          zero by the end of training.
        - ``'cosine_warmup'`` — linear warm-up for the first 10 % of epochs,
          then cosine annealing for the remainder. Recommended when using a
          large learning rate or when training is unstable in early epochs.
        - ``'step'`` — ``StepLR`` decaying lr by ``gamma=0.1`` every
          ``n_epochs // 3`` epochs (two decays total by default). Override via
          ``scheduler_params``.
        - ``'plateau'`` — ``ReduceLROnPlateau`` monitoring test MI (maximize).
          Reduces lr when MI stops improving; useful when the optimal number of
          epochs is unknown.

        You can also pass a ``torch.optim.lr_scheduler`` subclass directly for
        full control; it will be instantiated with the optimizer and any
        ``scheduler_params`` kwargs.
    scheduler_params : dict, optional
        Additional keyword arguments forwarded to the scheduler constructor.
        For example, ``{'T_max': 100}`` to override the period for ``'cosine'``,
        or ``{'step_size': 20, 'gamma': 0.5}`` for ``'step'``.
        Defaults to ``None`` (treated as ``{}``).
    eval_train : bool, float, or int, default=False
        Controls whether train-set MI is evaluated at every epoch, producing a
        ``'train_mi_history'`` alongside ``'test_mi_history'`` in the results.

        - ``False`` (default) — no per-epoch train evaluation.
        - ``True`` — evaluate on the full training subset (capped by
          ``max_eval_samples``).
        - ``float`` in ``(0, 1)`` — use that fraction of training samples.
        - ``int >= 1`` — use exactly that many training samples.
    peak_fraction : float, default=1.0
        Controls which epoch's train MI is reported as the final estimate.

        - ``1.0`` (default) — report train MI at the epoch where smoothed test
          MI is maximised.  Fully backward-compatible.
        - ``< 1.0`` — report train MI at the *first improvement checkpoint*
          where smoothed test MI reaches ``peak_fraction × max_test_mi``.
          Both the conservative and best-epoch estimates are fresh full
          evaluations performed at the end of training (no per-epoch train
          tracking is required; ``eval_train`` need not be set).
          ``result.details`` will contain ``'conservative_epoch'`` (the epoch
          used) and ``'train_mi_at_peak'`` (train MI at the actual peak epoch
          for comparison).
    use_amp : bool or str, default='auto'
        Mixed-precision (AMP) training.  ``'auto'`` (default) enables AMP when
        a CUDA GPU is detected and is a silent no-op on CPU / MPS.  ``True``
        enables explicitly (CUDA only; ignored on other devices).  ``False``
        disables entirely.  AMP typically gives 1.5–2× speed-up on modern
        NVIDIA GPUs with negligible impact on MI estimates.
    x_name : str, optional
        Short human-readable name for variable X (e.g. ``'LFP'``).  Stored in
        ``result.params['x_name']`` and used as a plot-label hint for custom
        visualisations.  Defaults to None (no label).
    y_name : str, optional
        Short human-readable name for variable Y (e.g. ``'spikes'``).  Same
        semantics as ``x_name``.  Defaults to None.
    channel_names_x : list of str, optional
        Names for each channel of ``x_data``.  For ``mode='pairwise'``, these
        appear as axis tick labels on the MI heatmap.  Length must match the
        number of channels; if shorter, only the first *k* channels are named.
        Defaults to None (integer indices used as labels).
    channel_names_y : list of str, optional
        Names for each channel of ``y_data`` in cross-pairwise mode.  Same
        semantics as ``channel_names_x``.  Defaults to None.
    dropout : float, optional
        Dropout probability applied after each hidden layer of MLP
        embedding networks.  ``0.0`` disables dropout (default).  Values in
        ``(0, 1)`` add regularisation, which is particularly helpful for small
        datasets.  Has no effect on CNN, GRU, LSTM, TCN, or Transformer
        architectures.  Defaults to ``None`` (use ``base_params`` value or
        apply the schema default of ``0.0``).
    norm_layer : {'layer', 'batch'} or None, optional
        Normalisation layer to insert after each hidden layer of MLP
        embedding networks.  ``None`` disables normalisation (default).
        ``'layer'`` inserts ``LayerNorm`` (recommended for small batches);
        ``'batch'`` inserts ``BatchNorm1d``.  Has no effect on other
        architectures.  Defaults to ``None`` (use ``base_params`` value or
        apply the schema default of ``None``).
    history_window : int, optional
        For ``mode='transfer'``, the number of past samples used as history
        context.  **Required** when ``mode='transfer'``.  The raw inputs
        ``x_data`` and ``y_data`` must be 2-D arrays of shape
        ``(T, n_channels)`` — no prior windowing should be applied, as the
        transfer-entropy module builds all sliding windows internally.
    prediction_horizon : int, default=1
        For ``mode='transfer'``, how many future samples to predict.
    bidirectional_te : bool, default=False
        For ``mode='transfer'``, if True, also compute TE(Y→X) and return a
        directionality index ``(TE_xy - TE_yx) / (|TE_xy| + |TE_yx|)``.
    **analysis_kwargs
        Mode-specific keyword arguments forwarded to the corresponding analysis
        engine.  These differ from ``base_params`` in scope:

        - **``base_params``** controls model architecture and training
          (``n_epochs``, ``learning_rate``, ``embedding_dim``, etc.) and is
          shared across all runs inside a single call.
        - **``analysis_kwargs``** controls the analysis logic itself (e.g.,
          ``n_workers``, ``n_splits``, ``gamma_range``).  Accepted keys for
          each mode are listed in ``neural_mi.defaults.MODE_KWARGS_SCHEMA``.
    max_eval_samples : int, default=5000
        The maximum number of samples to use for evaluation during training.
        This is a computational safeguard and does not affect the training data size.
    train_subset_size : int, optional
        If provided, the number of training samples to use when evaluating at the end of each epoch. This can speed up training on large datasets. Defaults to None (use all training data).
    spectral_mode : {'none', 'summary', 'full'}, default='none'
        Controls spectral metric tracking during training.

        - ``'none'`` — no spectral metrics computed (default, no overhead).
        - ``'summary'`` — compute the participation ratio of the joint embedding
          cross-covariance at the end of training.
        - ``'full'`` — compute all spectral metrics (participation ratio, effective
          rank, all singular values) and include the raw spectrum in results.
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
    corrupt_target : {'x', 'y', 'both'}, default='x'
        For ``mode='precision'``, which variable to apply corruption to during
        the precision sweep.  ``'x'`` corrupts only X, ``'y'`` only Y, and
        ``'both'`` applies the same corruption level simultaneously to X and Y
        (useful for measuring shared spike-timing precision).
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
            spectral_mode=spectral_mode, max_index_reduction=max_index_reduction,
            tau_grid=tau_grid, corrupt_target=corrupt_target,
            corruption_method=corruption_method, n_noise_samples=n_noise_samples,
            threshold_ratio=threshold_ratio, permutation_test=permutation_test,
            n_permutations=n_permutations, z_data=z_data, z_processor_type=z_processor_type,
            z_processor_params=z_processor_params, z_time=z_time, history_window=history_window,
            prediction_horizon=prediction_horizon, bidirectional_te=bidirectional_te,
            n_epochs=n_epochs, batch_size=batch_size, shared_encoder=shared_encoder,
            return_embeddings=return_embeddings, lag_range=lag_range,
            use_spectral_norm=use_spectral_norm, gradient_clip_val=gradient_clip_val,
            optimizer=optimizer, optimizer_params=optimizer_params,
            scheduler=scheduler, scheduler_params=scheduler_params,
            eval_train=eval_train,
            peak_fraction=peak_fraction,
            dropout=dropout, norm_layer=norm_layer,
            use_amp=use_amp,
            track_embeddings=track_embeddings,
            return_rotated_embeddings=return_rotated_embeddings,
            rotated_embeddings_whitening=rotated_embeddings_whitening,
            rotated_embeddings_per_epoch=rotated_embeddings_per_epoch,
            return_rotation_matrices=return_rotation_matrices,
            x_name=x_name, y_name=y_name,
            channel_names_x=channel_names_x, channel_names_y=channel_names_y,
            **analysis_kwargs
        )
    finally:
        logger.setLevel(_prev_level)
        for h, lv in zip(logger.handlers, _prev_handler_levels):
            h.setLevel(lv)


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
            f"at this scale. Consider the following additions in base_params: {hint}. "
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
                f"consider adding {' and '.join(tips)} in base_params.",
                UserWarning,
                stacklevel=4,
            )


def _run_inner(
    x_data, y_data, x_time, y_time, mode, processor_type_x, processor_params_x,
    processor_type_y, processor_params_y, base_params, sweep_grid, output_units,
    estimator, estimator_params, custom_critic, custom_embedding_cls,
    save_best_model_path, random_seed, verbose, show_progress, device,
    split_mode, train_fraction, n_test_blocks, split_gap_fraction, train_indices,
    test_indices, delta_threshold, min_gamma_points, confidence_level,
    max_eval_samples, train_subset_size, spectral_mode, max_index_reduction,
    tau_grid, corrupt_target, corruption_method, n_noise_samples, threshold_ratio,
    permutation_test, n_permutations, z_data, z_processor_type, z_processor_params,
    z_time=None,
    history_window=None, prediction_horizon=1, bidirectional_te=False,
    n_epochs=None, batch_size=None, shared_encoder=None,
    return_embeddings=False, lag_range=None,
    use_spectral_norm=True, gradient_clip_val=None,
    optimizer='adam', optimizer_params=None,
    scheduler=None, scheduler_params=None,
    eval_train=False,
    peak_fraction=1.0,
    dropout=None, norm_layer=None,
    use_amp='auto',
    track_embeddings=None,
    return_rotated_embeddings=None,
    rotated_embeddings_whitening=None,
    rotated_embeddings_per_epoch=None,
    return_rotation_matrices=None,
    x_name=None, y_name=None,
    channel_names_x=None, channel_names_y=None,
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
                f"Consider adding regularisation (dropout, norm_layer) in base_params.",
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
        agg_df = df.groupby(group_vars)['train_mi'].agg(['mean', 'std']).reset_index().rename(
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
            agg_df = df.groupby(group_vars)[valid_metrics].agg(['mean', 'std']).reset_index()
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
            agg_df = df.groupby(valid_group_vars)['train_mi'].agg(['mean', 'std']).reset_index().rename(
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
        if permutation_test:
            n_ch_x = x_run_data.shape[1] if x_run_data.ndim >= 2 else 1
            pairwise_y_for_warn = y_run_data if y_data is not None else None
            if pairwise_y_for_warn is not None:
                n_pairs_est = n_ch_x * pairwise_y_for_warn.shape[1]
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
        # Pass y_run_data when provided to enable cross-pairwise mode.
        pairwise_y = y_run_data if y_data is not None else None
        if permutation_test:
            n_pairs_estimate = x_run_data.shape[1] * (x_run_data.shape[1] - 1) // 2
            logger.warning(
                f"Permutation test requested for mode='pairwise'. This will run the full pairwise "
                f"matrix n_permutations times, which is computationally expensive "
                f"(estimated_pairs × n_permutations ≈ {n_pairs_estimate} × {n_permutations} MI "
                f"estimations). Allow additional time or reduce n_permutations."
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
            _null_clipped, _null_raw = _run_permutation_test(
                x_run_data, y_run_data, base_params, 'pairwise', sweep_grid,
                n_permutations, analysis_kwargs
            )
            result.details['null_distribution'] = _null_clipped
            result.details['null_distribution_raw'] = _null_raw
        return result

    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            f"Expected one of: 'estimate', 'sweep', 'dimensionality', 'rigorous', "
            f"'lag', 'precision', 'conditional', 'transfer', 'pairwise'."
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

        else:
            return _nan, _nan

    except Exception as exc:
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

    logger.info(
        f"Permutation test complete. "
        f"Null MI (clipped): mean={np.nanmean(null_distribution):.4f}, "
        f"std={np.nanstd(null_distribution):.4f}"
    )
    return null_distribution, null_distribution_raw







