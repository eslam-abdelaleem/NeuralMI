# neural_mi/analysis/precision.py
"""Estimates spike-timing precision of a representation relative to a target.

This module trains a baseline mutual information estimator, then freezes the
network and repeatedly evaluates the *train* partition across a grid of precision
levels (tau). It corrupts the data using deterministic rounding or additive noise
to find the precision threshold where mutual information degrades.
"""
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as _lr_sched
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from neural_mi.training.trainer import Trainer
from neural_mi.estimators import ESTIMATORS
from neural_mi.utils import build_critic, get_device
from neural_mi.data.handler import create_dataset
from neural_mi.logger import logger


def _build_optimizer_and_scheduler(params: Dict[str, Any], critic):
    """Build optimizer and optional LR scheduler from base_params, mirroring task.py."""
    _OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
    }
    _opt_val = params.get('optimizer', 'adam')
    if isinstance(_opt_val, type):
        OptCls = _opt_val
    else:
        OptCls = _OPTIMIZERS.get(str(_opt_val).lower())
        if OptCls is None:
            raise ValueError(
                f"Unknown optimizer '{_opt_val}'. "
                f"Supported names: {list(_OPTIMIZERS.keys())}."
            )
    optimizer = OptCls(
        critic.parameters(),
        lr=params.get('learning_rate', 0.001),
        **params.get('optimizer_params', {}),
    )

    _sched_val = params.get('scheduler', None)
    scheduler = None
    if _sched_val is not None:
        _sched_params = params.get('scheduler_params', {})
        n_epochs = params.get('n_epochs', 50)
        if isinstance(_sched_val, type):
            scheduler = _sched_val(optimizer, **_sched_params)
        elif _sched_val == 'cosine':
            scheduler = _lr_sched.CosineAnnealingLR(optimizer, T_max=n_epochs, **_sched_params)
        elif _sched_val == 'step':
            scheduler = _lr_sched.StepLR(optimizer, step_size=max(1, n_epochs // 3), **_sched_params)
        elif _sched_val == 'plateau':
            scheduler = _lr_sched.ReduceLROnPlateau(optimizer, mode='max', **_sched_params)
        elif _sched_val == 'cosine_warmup':
            warmup = max(1, int(n_epochs * 0.1))
            warmup_sched = _lr_sched.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
            cosine_sched = _lr_sched.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup))
            scheduler = _lr_sched.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
            )
        else:
            raise ValueError(f"Unknown scheduler '{_sched_val}'.")
    return optimizer, scheduler

def apply_corruption(data: torch.Tensor, tau: float, method: str) -> torch.Tensor:
    """Applies precision corruption to a tensor."""
    if tau == 0.0:
        return data
        
    if method == 'rounding':
        # Deterministic rounding to the nearest integer grid of tau
        return tau * torch.round(data / tau)
    elif method == 'noise':
        # Additive uniform noise centered around 0 (interval [-tau/2, tau/2])
        noise = (torch.rand_like(data) - 0.5) * tau
        return data + noise
    else:
        raise ValueError(f"Unknown corruption method: {method}")

def run_precision_analysis(
    x_data: Any, y_data: Any, base_params: Dict[str, Any],
    tau_grid: List[float], corrupt_target: str = 'x',
    corruption_method: str = 'rounding', n_noise_samples: int = 50,
    threshold_ratio: Union[float, List[float]] = 0.9, **kwargs
) -> Dict[str, Any]:
    """Estimate spike-timing precision via a "Train Once, Evaluate Many" sweep.

    Trains a single baseline MI estimator at full precision (zero corruption),
    then freezes the network and evaluates it across a grid of corruption levels
    (*tau*).  The precision threshold is defined as the smallest *tau* at which
    the MI drops below ``threshold_ratio`` × baseline MI.

    Parameters
    ----------
    x_data : array-like
        Preprocessed data for variable X, shape ``(n_samples, n_channels, window_size)``.
    y_data : array-like
        Preprocessed data for variable Y, same leading dimension as *x_data*.
    base_params : Dict[str, Any]
        Parameters for the MI estimator (model architecture, training schedule, etc.).
        See ``run()`` documentation for the full list of accepted keys.
    tau_grid : list of float
        Corruption levels to sweep over (ascending order recommended).  Each value
        is applied to the target variable according to *corruption_method*.
    corrupt_target : {'x', 'y'}, default='x'
        Which variable to corrupt during the sweep.
    corruption_method : {'rounding', 'noise'}, default='rounding'
        How corruption is applied.  ``'rounding'`` quantizes values to the nearest
        multiple of *tau* (deterministic).  ``'noise'`` adds uniform noise drawn
        from U(-tau/2, tau/2).
    n_noise_samples : int, default=50
        Number of independent noise realizations to average when
        ``corruption_method='noise'``.  Ignored for ``'rounding'``.
    threshold_ratio : float or list of float, default=0.9
        The precision threshold is the smallest *tau* at which MI falls below
        ``threshold_ratio × baseline_MI``.  Each value must be in (0, 1].
        If a list is provided, thresholds are computed for all ratios and
        returned in the ``precision_thresholds`` dict; the first ratio is
        used as the primary result reported in ``details['precision_tau']``.
    **kwargs
        Additional keyword arguments forwarded to the trainer
        (e.g., ``n_test_blocks``).

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:

        - ``'dataframe'`` : pd.DataFrame with columns ``tau``, ``train_mi``, and
          ``train_mi_std`` (one row per *tau* value).
        - ``'details'`` : dict containing:

          - ``'baseline_mi'`` — MI at zero corruption (float, nats).
          - ``'precision_tau'`` — the primary estimated precision threshold (float).
          - ``'threshold_ratio'`` — the original input (scalar or list).
          - ``'threshold_value'`` — MI value at the primary threshold (float, nats).
          - ``'precision_thresholds'`` — dict mapping each ratio to its
            ``{'precision_tau', 'threshold_value'}`` result.
          - ``'raw_results'`` — same DataFrame as ``'dataframe'``.
    """
    logger.info("Initializing Precision Analysis...")

    # 1. Prepare Data & Model
    # Precision analysis trains once then runs many forward passes on the same
    # dataset at different corruption levels.  Keeping data on the compute
    # device ('auto') avoids repeated host→device transfers and is therefore
    # the default here.  Users can override by setting dataset_device='cpu'
    # in base_params if memory is a concern.
    device = get_device(base_params.get('device'))
    _data_device_raw = base_params.get('dataset_device', 'auto')
    _data_device = str(device) if _data_device_raw == 'auto' else (_data_device_raw or 'cpu')

    dataset = create_dataset(
        x_data, y_data,
        processor_type_x=base_params.get('processor_type_x'),
        processor_type_y=base_params.get('processor_type_y'),
        processor_params_x=base_params.get('processor_params_x'),
        processor_params_y=base_params.get('processor_params_y'),
        data_device=_data_device,
    )
    
    # Update dimensions in base_params based on the dataset
    if dataset.x_data is not None and hasattr(dataset.x_data, 'shape'):
        base_params['input_dim_x'] = dataset.x_data.shape[1] * dataset.x_data.shape[2]
        base_params['n_channels_x'] = dataset.x_data.shape[1]
    if dataset.y_data is not None and hasattr(dataset.y_data, 'shape'):
        base_params['input_dim_y'] = dataset.y_data.shape[1] * dataset.y_data.shape[2]
        base_params['n_channels_y'] = dataset.y_data.shape[1]

    if base_params.get('custom_critic') is not None:
        critic = base_params['custom_critic']
        logger.debug("Using provided custom critic.")
    else:
        critic = build_critic(base_params.get('critic_type', 'separable'), base_params, base_params.get('custom_embedding_cls'))

    optimizer, scheduler = _build_optimizer_and_scheduler(base_params, critic)

    trainer = Trainer(
        model=critic.to(device),
        estimator_fn=ESTIMATORS[base_params.get('estimator_name', 'infonce')],
        optimizer=optimizer,
        device=device,
        use_variational=base_params.get('use_variational', False),
        beta=base_params.get('beta', 1024.0),
        estimator_params=base_params.get('estimator_params'),
        gradient_clip_val=base_params.get('gradient_clip_val', None),
    )
    
    # Determine train/test splits explicitly so we can reuse the exact test set for evaluation
    n_samples = len(dataset)
    train_frac = base_params.get('train_fraction', 0.9)
    split_mode = base_params.get('split_mode', 'blocked')
    if split_mode == 'random':
        train_idx, test_idx = trainer._create_random_split(n_samples, train_frac)
    else:
        train_idx, test_idx = trainer._create_blocked_split(n_samples, train_frac, base_params.get('n_test_blocks', 5))
        
    # 2. Train the Baseline Model (Zero-Noise)
    logger.info("Training baseline model at maximum precision...")
    baseline_results = trainer.train(
        dataset,
        n_epochs=base_params.get('n_epochs', 50),
        batch_size=base_params.get('batch_size', 256),
        patience=base_params.get('patience', 1000),
        train_indices=train_idx,
        test_indices=test_idx,
        verbose=base_params.get('verbose', False),
        show_progress=base_params.get('show_progress', True),
        max_eval_samples=base_params.get('max_eval_samples', 5000),
        track_spectral_metrics=False,  # Skip dimensionality math to save time
        scheduler=scheduler,
    )
    
    baseline_mi = baseline_results['train_mi']
    logger.info(f"Baseline MI established: {baseline_mi:.3f} nats")

    # 3. The Precision Sweep (Inference Only)
    # We evaluate on the *train* partition (the larger 90 % slice) to keep
    # the reported MI consistent with every other mode, which also uses train_mi.
    logger.info(f"Starting precision sweep using '{corruption_method}' on target '{corrupt_target}'...")
    trainer.model.eval()
    x_train_raw = dataset.x_dataset[train_idx, ...]
    y_train_raw = dataset.y_dataset[train_idx, ...]
    max_eval = base_params.get('max_eval_samples', 5000)

    results_list = []

    # Force 0.0 into the grid to log the exact baseline
    sorted_tau = sorted(list(set([0.0] + tau_grid)))

    with torch.no_grad():
        for tau in sorted_tau:
            if corruption_method == 'rounding':
                x_c = apply_corruption(x_train_raw, tau, 'rounding') if corrupt_target in ['x', 'both'] else x_train_raw
                y_c = apply_corruption(y_train_raw, tau, 'rounding') if corrupt_target in ['y', 'both'] else y_train_raw
                mi = trainer._safe_eval_mi(x_c.to(device), y_c.to(device), max_eval)
                results_list.append({'tau': tau, 'train_mi': mi, 'train_mi_std': 0.0})

            elif corruption_method == 'noise':
                # Average over multiple forward passes to stabilize stochastic noise bounds
                mis = []
                for _ in range(n_noise_samples if tau > 0 else 1):
                    x_c = apply_corruption(x_train_raw, tau, 'noise') if corrupt_target in ['x', 'both'] else x_train_raw
                    y_c = apply_corruption(y_train_raw, tau, 'noise') if corrupt_target in ['y', 'both'] else y_train_raw
                    mis.append(trainer._safe_eval_mi(x_c.to(device), y_c.to(device), max_eval))
                results_list.append({'tau': tau, 'train_mi': np.mean(mis), 'train_mi_std': np.std(mis)})

    df = pd.DataFrame(results_list)
    
    # 4. Find Precision Threshold(s)
    # Normalise threshold_ratio to a list for uniform handling
    if isinstance(threshold_ratio, (int, float)):
        ratio_list = [float(threshold_ratio)]
        scalar_input = True
    else:
        ratio_list = sorted([float(r) for r in threshold_ratio], reverse=True)
        scalar_input = False

    precision_thresholds = {}
    for ratio in ratio_list:
        threshold_value_i = baseline_mi * ratio
        tau_i = None
        for _, row in df.iterrows():
            if row['tau'] > 0 and row['train_mi'] < threshold_value_i:
                tau_i = row['tau']
                break
        if tau_i is None:
            logger.warning(
                f"MI never dropped below {ratio*100:.0f}% of baseline. "
                f"Consider extending the tau_grid."
            )
        precision_thresholds[ratio] = {
            'precision_tau': tau_i,
            'threshold_value': threshold_value_i,
        }

    # Backward-compatible scalar output
    primary_ratio = ratio_list[0]
    precision_tau = precision_thresholds[primary_ratio]['precision_tau']
    threshold_value = precision_thresholds[primary_ratio]['threshold_value']
    logger.info(f"Precision Threshold ({primary_ratio*100:.0f}%) estimated at tau = {precision_tau}")

    return {
        'dataframe': df,
        'details': {
            'baseline_mi': baseline_mi,
            'precision_tau': precision_tau,
            'threshold_ratio': threshold_ratio,        # original input (scalar or list)
            'threshold_value': threshold_value,        # primary threshold value
            'precision_thresholds': precision_thresholds,  # full multi-threshold dict
            'corruption_method': corruption_method,
            'corrupt_target': corrupt_target,
        }
    }