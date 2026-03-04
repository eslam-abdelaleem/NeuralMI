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

from .sweep import ParameterSweep
from neural_mi.logger import logger


def run_dimensionality_analysis(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    y_data: Optional[torch.Tensor] = None,
    sweep_grid: Optional[Dict[str, Any]] = None,
    split_method: str = 'random',
    n_splits: int = 5,
    spectral_output: str = 'default',
    return_spectrum: bool = False,
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
        - 'random': randomly shuffles channels into two halves, repeated n_splits
          times so the result averages over different channel assignments.
        - 'spatial': splits channels at the midpoint (first vs second half).
          Use when channels have a meaningful spatial ordering (e.g. electrode array).
        - 'temporal': correlates x_data with a lag-shifted copy of itself.
          Pass lag=<int> (in samples) as a kwarg. Measures autocorrelation
          structure rather than cross-channel shared information.
        Defaults to 'random'.
    n_splits : int, optional
        Number of random channel splits (only applies when split_method='random').
    spectral_output : {'default', 'all'}, optional
        'default' returns only participation_ratio; 'all' returns all spectral metrics.
    return_spectrum : bool, optional
        If True, includes the raw singular values array in each result row.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.

    Returns
    -------
    pd.DataFrame
        One row per split (and per sweep combination). Columns include split_id,
        test_mi, participation_ratio, and any additional spectral metrics.

    """

    # 1. Force correct configuration for dimensionality
    analysis_params = base_params.copy()
    analysis_params['critic_type'] = 'hybrid'
    analysis_params['track_spectral_metrics'] = True
    analysis_params['spectral_output'] = spectral_output
    analysis_params['return_spectrum'] = return_spectrum

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

    # 2. Interaction Dimensionality (X and Y both provided)
    if y_data is not None:
        logger.info("y_data provided. Computing Interaction Dimensionality.")
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=analysis_params)
        results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers,
                            is_proc_sweep=False)
        df = pd.DataFrame(results)
        df['split_id'] = 0
        return df

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
        all_results = _run_single_split(x_a, x_b, analysis_params,
                                        sweep_grid, n_workers, split_id=0)

    elif split_method in ('random', 'spatial'):
        if n_channels < 2:
            raise ValueError(
                f"Cannot perform '{split_method}' channel split with fewer than 2 channels. "
                f"x_data has shape {tuple(x_data.shape)}."
            )
        loops = n_splits if split_method == 'random' else 1
        all_results = []

        for i in range(loops):
            logger.info(f"--- Running Split {i + 1}/{loops} ---")

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

            split_rows = _run_single_split(x_a, x_b, analysis_params,
                                           sweep_grid, n_workers, split_id=i)
            all_results.extend(split_rows)

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