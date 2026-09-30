# neural_mi/analysis/precision.py
"""Estimates spike-timing precision of a representation relative to a target.

This module trains a baseline mutual information estimator, then freezes the
network and repeatedly evaluates the *train* partition across a grid of precision
levels (tau). It degrades the data by rounding to bins of width tau or by
uniform jitter of width tau, through :func:`neural_mi.data.corruption.corrupt`,
to find the precision at which mutual information falls. Entries that hold no
measurement, such as unused spike-time slots, are never corrupted.
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

from neural_mi.analysis.task import run_training_task
from neural_mi.data.corruption import corrupt, METHODS
from neural_mi.logger import logger
from neural_mi.validation import ParameterValidator


def _empty_value(base_params: Dict[str, Any], side: str) -> float:
    """The value of an entry that holds no measurement on one side.

    Spike times are windowed into fixed slots padded with ``no_spike_value``.
    Every other representation (binned spikes, continuous samples with
    zero-padded gaps, unwindowed arrays) marks an empty entry with zero.
    """
    proc = base_params.get(f'processor_type_{side}')
    params = base_params.get(f'processor_params_{side}')
    if side == 'y' and proc is None:
        proc = base_params.get('processor_type_x')
        params = params if params is not None else base_params.get('processor_params_x')
    params = params or {}
    if proc == 'spike' and params.get('bin_size') is None:
        return float(params.get('no_spike_value', 0.0))
    return 0.0

def run_precision_analysis(
    x_data: Any, y_data: Any, base_params: Dict[str, Any],
    tau_grid: List[float], corrupt_target: str = 'x',
    corruption_method: str = 'rounding', n_noise_samples: int = 50,
    threshold_ratio: Union[float, List[float]] = 0.9,
    n_workers: int = 1,
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
    corrupt_target : {'x', 'y', 'both'}, default='x'
        Which variable to corrupt during the sweep.  Use ``'both'`` to apply
        the same corruption level simultaneously to X and Y (e.g. to measure
        shared temporal precision).
    corruption_method : {'rounding', 'noise'}, default='rounding'
        How corruption is applied. ``'rounding'`` moves every value to the
        centre of its bin of width *tau* (deterministic). ``'noise'`` adds
        uniform noise drawn from U(-tau/2, tau/2). Either way, entries that
        hold no measurement (unused spike-time slots, empty bins, zero-padded
        gaps) are left as they are, so no spike is created or deleted.
    n_noise_samples : int, default=50
        Number of independent noise realizations to average when
        ``corruption_method='noise'``.  Ignored for ``'rounding'``.
    threshold_ratio : float or list of float, default=0.9
        The precision threshold is the smallest *tau* at which MI falls below
        ``threshold_ratio × baseline_MI``.  Each value must be in (0, 1].
        If a list is provided, thresholds are computed for all ratios and
        returned in the ``precision_thresholds`` dict; the first ratio is
        used as the primary result reported in ``details['precision_tau']``.
    n_workers : int, default=1
        Unused: precision analysis is inherently single-process. It trains
        one baseline model, then evaluates it repeatedly (inference only)
        across the tau grid. There is no independent work to parallelize.
        Accepted so the top-level ``run(..., n_workers=...)`` argument, forwarded uniformly to every mode, doesn't raise a ``TypeError``.


    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:

        - ``'dataframe'`` : pd.DataFrame with columns ``tau``, ``train_mi``, and
          ``train_mi_std`` (one row per *tau* value).
        - ``'details'`` : dict containing:

          - ``'baseline_mi'``: MI at zero corruption (float, nats).
          - ``'precision_tau'``: the primary estimated precision threshold
            (float), or ``None`` if MI never dropped below the threshold
            (check with ``is None``, not ``np.isnan``).
          - ``'threshold_ratio'``: the original input (scalar or list).
          - ``'threshold_value'``: MI value at the primary threshold (float, nats).
          - ``'precision_thresholds'``: dict mapping each ratio to its
            ``{'precision_tau', 'threshold_value'}`` result (``precision_tau``
            is ``None`` per-ratio under the same not-found condition).
          - ``'raw_results'``: same DataFrame as ``'dataframe'``.
    """
    logger.info("Initializing Precision Analysis...")

    # 1. Train the baseline at full precision. It goes through the same task as
    # every other mode, so every Model, Training and Split setting applies to it,
    # and the task hands back the trainer and the dataset for the sweep below.
    logger.info("Training baseline model at maximum precision...")
    # A direct caller can pass a partial base_params; fill it from the schema as
    # run() does, so a direct call trains exactly what run() would.
    params = dict(base_params)
    ParameterValidator({'base_params': params}).apply_defaults()
    baseline_results = run_training_task((x_data, y_data, {**params, '_keep_trained': True}, 'precision'))
    trainer = baseline_results.pop('_trainer')
    dataset = baseline_results.pop('_dataset')
    train_idx = trainer.last_train_indices
    device = trainer.device
    baseline_mi = baseline_results['train_mi']
    model_path = baseline_results.get('model_path')
    logger.info(f"Baseline MI established: {baseline_mi:.3f} nats")

    # 3. The Precision Sweep (Inference Only)
    # We evaluate on the *train* partition (the larger 90 % slice) to keep
    # the reported MI consistent with every other mode, which also uses train_mi.
    logger.info(f"Starting precision sweep using '{corruption_method}' on target '{corrupt_target}'...")
    trainer.model.eval()
    # Read through the frozen pre-shift snapshot when shifting was active
    # (dataset.x_data/.y_data would otherwise reflect whichever shift was
    # last applied during training, not the canonical original data this
    # sweep corrupts).
    _corrupt_x_source = baseline_results.get('_frozen_eval_x', dataset.x_data)
    _corrupt_y_source = baseline_results.get('_frozen_eval_y', dataset.y_data)
    x_train_raw = _corrupt_x_source[train_idx, ...]
    y_train_raw = _corrupt_y_source[train_idx, ...]
    max_eval = base_params.get('max_eval_samples', 5000)

    results_list = []
    # Every individual evaluation, one per tau for rounding and one per noise
    # draw for noise, so the caller keeps the values the per-tau mean is made of.
    samples = []

    # Force 0.0 into the grid to log the exact baseline
    sorted_tau = sorted(list(set([0.0] + tau_grid)))

    if corruption_method not in METHODS:
        raise ValueError(f"corruption_method must be one of {list(METHODS)}, got {corruption_method!r}.")
    # Entries that hold no measurement (unused spike-time slots, empty bins,
    # zero-padded gaps) are never corrupted, so neither method invents activity.
    empty_x, empty_y = _empty_value(base_params, 'x'), _empty_value(base_params, 'y')
    noise = corruption_method == 'noise'

    with torch.no_grad():
        for tau in sorted_tau:
            # Noise is averaged over several draws; rounding is deterministic.
            mis = []
            for _ in range(n_noise_samples if noise and tau > 0 else 1):
                x_c = (corrupt(x_train_raw, tau, corruption_method, empty_x)
                       if corrupt_target in ('x', 'both') else x_train_raw)
                y_c = (corrupt(y_train_raw, tau, corruption_method, empty_y)
                       if corrupt_target in ('y', 'both') else y_train_raw)
                mis.append(trainer._safe_eval_mi(x_c.to(device), y_c.to(device), max_eval))
                sample = {'tau': tau, 'mi': mis[-1]}
                if noise:
                    sample['noise_sample'] = len(mis) - 1
                samples.append(sample)
            results_list.append({'tau': tau, 'train_mi': float(np.mean(mis)),
                                 'train_mi_std': float(np.std(mis)) if noise else float('nan'),
                                 'eval_size': max_eval})

    df = pd.DataFrame(results_list)

    # A negative swept value means the estimator has left the region where its
    # output is interpretable, not that information went below zero. The critic
    # is frozen on clean data and then scored on corrupted inputs, and InfoNCE's
    # bound is unbounded below, so the magnitude reports how badly the bound has
    # broken instead of how much information the corruption destroyed. Warn on
    # the first one, because the threshold crossing above it is still readable
    # and a reader plotting the tail otherwise has nothing telling them so.
    _neg = df[(df['tau'] > 0) & (df['train_mi'] < 0)]
    if not _neg.empty:
        _first = _neg.iloc[0]
        _worst = df['train_mi'].min()
        # Reported as a multiple of baseline instead of as an absolute value:
        # this runs in nats while the returned frame is converted to the caller's
        # output_units, so an absolute figure here would not match the curve the
        # caller prints.
        _depth = abs(_worst) / baseline_mi if baseline_mi else float('inf')
        logger.warning(
            f"mode='precision': MI goes negative from tau={_first['tau']:g} onward and "
            f"reaches roughly {_depth:.0f}x the baseline in the opposite direction. "
            f"Past this point the frozen critic is evaluated on inputs it was never "
            f"trained for. The estimator's lower bound is unbounded below there. The "
            f"depth of the fall measures how badly the bound has broken. MI cannot be "
            f"negative. Those values are reported as 0 and the measured ones are kept "
            f"as mi_raw in result.runs. The threshold crossing is read before that "
            f"point and is unaffected."
        )

    # 4. Find Precision Threshold(s)
    # Normalise threshold_ratio to a list for uniform handling
    if isinstance(threshold_ratio, (int, float)):
        ratio_list = [float(threshold_ratio)]
    else:
        ratio_list = sorted([float(r) for r in threshold_ratio], reverse=True)

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

    # The first ratio is the primary one, reported on its own as well.
    primary_ratio = ratio_list[0]
    precision_tau = precision_thresholds[primary_ratio]['precision_tau']
    threshold_value = precision_thresholds[primary_ratio]['threshold_value']
    logger.info(f"Precision Threshold ({primary_ratio*100:.0f}%) estimated at tau = {precision_tau}")

    return {
        'samples': samples,
        'dataframe': df,
        'details': {
            'baseline_mi': baseline_mi,
            'precision_tau': precision_tau,
            'threshold_ratio': threshold_ratio,        # original input (scalar or list)
            'threshold_value': threshold_value,        # primary threshold value
            'precision_thresholds': precision_thresholds,  # full multi-threshold dict
            'corruption_method': corruption_method,
            'corrupt_target': corrupt_target,
            **({'model_path': model_path} if model_path else {}),
        }
    }