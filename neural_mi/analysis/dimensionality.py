# neural_mi/analysis/dimensionality.py
"""Estimates the latent dimensionality of a dataset using spectral metrics.

This module forces the use of a Hybrid critic with a large bottleneck and
analyzes the cross-covariance spectrum of the resulting embeddings to
determine Intrinsic or Interaction Dimensionality.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from .sweep import ParameterSweep
from neural_mi.logger import logger
from neural_mi.utils import _configure_multiprocessing


# ---------------------------------------------------------------------------
# Module-level picklable wrapper — must be defined at module scope so that
# multiprocessing can serialise it via its qualified name.
# ---------------------------------------------------------------------------

def _run_single_split_task(args):
    """Top-level wrapper for Pool.map — must be module-level for pickling.

    Each split is executed with ``n_workers=1`` internally to avoid nested
    multiprocessing pools.
    """
    x_a, x_b, analysis_params, sweep_grid, split_id = args
    return _run_single_split(x_a, x_b, analysis_params, sweep_grid,
                             n_workers=1, split_id=split_id)


def _dispatch_splits(split_tasks, n_workers, show_progress):
    """Execute split tasks, parallelising *across splits* when ``n_workers > 1``.

    Strategy
    --------
    * **Single split** (``len(split_tasks) == 1``): run sequentially and
      forward ``n_workers`` into the inner ``ParameterSweep`` so that any
      sweep-grid parallelism still uses the available workers.
    * **Multiple splits, ``n_workers > 1``**: dispatch splits to a
      ``Pool(n_workers)`` — each split's inner ``ParameterSweep`` gets
      ``n_workers=1`` to prevent nested pools.
    * **``n_workers <= 1``**: fully sequential.
    """
    n_tasks = len(split_tasks)

    if n_workers <= 1 or n_tasks <= 1:
        # Sequential path.
        # When there is only one split, pass n_workers through so the inner
        # ParameterSweep can use them for sweep-grid parallelism.
        inner_workers = n_workers if n_tasks == 1 else 1
        all_results = []
        for args in tqdm(split_tasks, desc="Dimensionality Splits",
                         disable=not show_progress or n_tasks == 1):
            x_a, x_b, analysis_params, sweep_grid, split_id = args
            rows = _run_single_split(x_a, x_b, analysis_params, sweep_grid,
                                     n_workers=inner_workers, split_id=split_id)
            all_results.extend(rows)
        return all_results

    # Parallel path — splits dispatched to a Pool, inner sweeps sequential.
    logger.info(f"Parallelising {n_tasks} dimensionality splits across {n_workers} workers...")
    _configure_multiprocessing()
    with mp.get_context('spawn').Pool(processes=n_workers) as pool:
        results_per_split = list(tqdm(
            pool.imap(_run_single_split_task, split_tasks),
            total=n_tasks,
            desc="Dimensionality Splits",
            disable=not show_progress,
        ))

    all_results = []
    for rows in results_per_split:
        all_results.extend(rows)
    return all_results


def run_dimensionality_analysis(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    y_data: Optional[torch.Tensor] = None,
    sweep_grid: Optional[Dict[str, Any]] = None,
    split_method: str = 'random',
    n_splits: int = 5,
    spectral_mode: str = 'summary',
    n_workers: int = 1,
    **kwargs
) -> pd.DataFrame:
    """Estimates dimensionality via embedding cross-covariance.

    Parameters
    ----------
    x_data : torch.Tensor
        Input data for variable X.
    base_params : Dict[str, Any]
        Dictionary of fixed parameters for the MI estimator's trainer.
        If train_indices and test_indices are present in base_params, they
        are passed through to the trainer for each split. These are indices into
        the temporal/sample dimension of x_data (not the channel dimension), so
        they remain valid after any spatial or random channel split.
    y_data : torch.Tensor, optional
        If provided, computes Interaction Dimensionality between X and Y directly.
        If None, computes Intrinsic Dimensionality by splitting x_data channels.
    split_method : {'random', 'spatial', 'temporal'}, optional
        How to split x_data when y_data is None.

        - ``'random'``: randomly shuffles channels into two halves, repeated
          ``n_splits`` times so the result averages over different channel
          assignments.
        - ``'spatial'``: splits channels at the midpoint (first vs second half).
          Use when channels have a meaningful spatial ordering (e.g. electrode
          array).
        - ``'temporal'``: correlates x_data with a lag-shifted copy of itself.
          Pass ``lag=<int>`` (in samples) as a kwarg. Measures autocorrelation
          structure rather than cross-channel shared information.

        Defaults to ``'random'``.
    n_splits : int, optional
        Number of independent runs.  For intrinsic dimensionality with
        ``split_method='random'`` this controls how many distinct random
        channel-split assignments are evaluated.  For interaction
        dimensionality (``y_data`` provided) there is no channel split, so
        ``n_splits`` instead controls how many independent model fits are
        performed — each starting from a different random weight
        initialisation — giving a proper mean and standard deviation in the
        output.  Defaults to 5.
    spectral_mode : {'summary', 'full'}, optional
        Controls which spectral metrics are returned.

        - ``'summary'`` *(default)* — compute the participation ratio only.
        - ``'full'`` — compute all spectral metrics and include the raw
          singular values array in each result row.
    n_workers : int, optional
        Number of parallel workers.  When ``n_splits > 1`` the workers are
        distributed *across splits* (each split's inner sweep runs
        sequentially to avoid nested pools).  When ``n_splits == 1`` the
        workers are forwarded into the inner ``ParameterSweep`` to
        parallelise any sweep-grid combinations.  Defaults to 1.

    Returns
    -------
    pd.DataFrame
        One row per split (and per sweep combination). Columns include split_id,
        train_mi, participation_ratio, and any additional spectral metrics.

    """

    # 1. Force correct configuration for dimensionality
    analysis_params = base_params.copy()
    analysis_params['critic_type'] = 'hybrid'
    analysis_params['track_spectral_metrics'] = True
    if spectral_mode == 'full':
        analysis_params['spectral_output'] = 'all'
        analysis_params['return_spectrum'] = True
    else:  # 'summary' or any unrecognised value defaults to summary
        analysis_params['spectral_output'] = 'default'
        analysis_params['return_spectrum'] = False

    # Default shared_encoder=True: X and Y are always split halves of the same
    # distribution in dimensionality mode, so tying their embedding weights is
    # both theoretically justified and reduces the parameter count by half.
    # Users who want independent encoders can override via base_params.
    if 'shared_encoder' not in analysis_params:
        analysis_params['shared_encoder'] = True

    if 'embedding_dim' not in analysis_params and 'embedding_dim' not in (sweep_grid or {}):
        logger.info("No embedding_dim specified. Defaulting to 64 for robust dimensionality capacity.")
        analysis_params['embedding_dim'] = 64

    # n_workers=None would crash the pool; default to 1
    if n_workers is None:
        n_workers = 1

    show_progress = analysis_params.get('show_progress', True)

    # 2. Interaction Dimensionality (X and Y both provided)
    if y_data is not None:
        logger.info(
            f"y_data provided. Computing Interaction Dimensionality "
            f"({n_splits} independent run{'s' if n_splits != 1 else ''})."
        )
        split_tasks = [
            (x_data, y_data, analysis_params, sweep_grid, i)
            for i in range(n_splits)
        ]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)
        return pd.DataFrame(all_results)

    # 3. Intrinsic Dimensionality (only X provided — channel split)
    logger.info(f"Computing Intrinsic Dimensionality using '{split_method}' splits.")

    # shape is (n_windows, n_channels, window_size) — channels are at dim 1, not dim -1
    n_channels = x_data.shape[1]

    if split_method == 'temporal':
        lag = kwargs.get('lag', None)
        if lag is None:
            raise ValueError(
                "split_method='temporal' requires a 'lag' kwarg (in samples). "
                "Example: run(..., lag=1)"
            )
        if not isinstance(lag, int) or lag < 1:
            raise ValueError(f"'lag' must be a positive integer, got {lag!r}.")
        x_a = x_data[:-lag, ...]
        x_b = x_data[lag:, ...]
        logger.info(
            f"Temporal split at lag={lag} samples: {x_a.shape[0]} aligned sample pairs."
        )
        # Only one temporal split — forward n_workers into the inner ParameterSweep
        split_tasks = [(x_a, x_b, analysis_params, sweep_grid, 0)]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)

    elif split_method in ('random', 'spatial'):
        if n_channels < 2:
            raise ValueError(
                f"Cannot perform '{split_method}' channel split with fewer than 2 channels. "
                f"x_data has shape {tuple(x_data.shape)}."
            )
        loops = n_splits if split_method == 'random' else 1

        # Pre-compute all (x_a, x_b) pairs before dispatching.
        split_tasks = []
        for i in range(loops):
            if split_method == 'random':
                indices = np.random.permutation(n_channels)
                half = n_channels // 2
                if x_data.ndim == 2:
                    x_a = x_data[:, indices[:half]]
                    x_b = x_data[:, indices[half:]]
                else:  # 3D (N, C, W)
                    x_a = x_data[:, indices[:half], :]
                    x_b = x_data[:, indices[half:], :]
            else:  # spatial
                half = n_channels // 2
                if x_data.ndim == 2:
                    x_a = x_data[:, :half]
                    x_b = x_data[:, half:]
                else:  # 3D (N, C, W)
                    x_a = x_data[:, :half, :]
                    x_b = x_data[:, half:, :]
            split_tasks.append((x_a, x_b, analysis_params, sweep_grid, i))

        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)

    else:
        raise ValueError(
            f"Unknown split_method: '{split_method}'. "
            f"Expected one of: 'random', 'spatial', 'temporal'."
        )

    df = pd.DataFrame(all_results)
    logger.info("--- Dimensionality Analysis Complete ---")
    return df


def _run_single_split(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    analysis_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]],
    n_workers: int,
    split_id: int,
) -> list:
    """Run one channel-split and return result dicts with split_id attached."""
    sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=analysis_params)
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers,
                        is_proc_sweep=False)
    for res in results:
        res['split_id'] = split_id
    return results
