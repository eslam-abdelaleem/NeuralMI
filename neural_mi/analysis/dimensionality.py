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
    n_workers: int = None,
    **kwargs
) -> pd.DataFrame:
    """Estimates dimensionality via embedding cross-covariance.

    Parameters
    ----------
    x_data : torch.Tensor
        Input data for variable X.
    base_params : Dict[str, Any]
        Dictionary of fixed parameters for the MI estimator's trainer.
    y_data : torch.Tensor, optional
        Input data for variable Y. If provided, computes Interaction Dimensionality.
        If None, computes Intrinsic Dimensionality using `split_method`.
    split_method : {'random', 'spatial', 'temporal'}, optional
        How to split `x_data` when `y_data` is None. Defaults to 'random'.
    n_splits : int, optional
        Number of random splits to average over (only applies to 'random'). Defaults to 5.
    spectral_output : {'default', 'all'}, optional
        Which spectral metrics to return. 'default' returns only pr_singular.
    return_spectrum : bool, optional
        If True, includes the raw singular values array in the results.
    """
    
    # 1. Force the correct configuration for dimensionality
    analysis_params = base_params.copy()
    analysis_params['critic_type'] = 'hybrid'
    analysis_params['track_spectral_metrics'] = True
    analysis_params['spectral_output'] = spectral_output
    analysis_params['return_spectrum'] = return_spectrum

    if 'embedding_dim' not in analysis_params and 'embedding_dim' not in (sweep_grid or {}):
        logger.info("No embedding_dim specified. Defaulting to 64 for robust dimensionality capacity.")
        analysis_params['embedding_dim'] = 64

    # 2. Interaction Dimensionality (X and Y provided)
    if y_data is not None:
        logger.info("y_data provided. Computing Interaction Dimensionality.")
        sweep = ParameterSweep(x_data=x_data, y_data=y_data, base_params=analysis_params)
        # We run a single pass (empty sweep grid)
        results = sweep.run(sweep_grid={}, n_workers=n_workers, is_proc_sweep=False)
        return pd.DataFrame(results)

    # 3. Intrinsic Dimensionality (Only X provided)
    logger.info(f"Computing Intrinsic Dimensionality using '{split_method}' splits.")
    
    # Handle different tensor shapes (Time x Channels vs Samples x Features x Channels)
    n_channels = x_data.shape[-1] 
    if n_channels < 2 and split_method in ['random', 'spatial']:
        raise ValueError(f"Cannot perform '{split_method}' split with fewer than 2 channels.")

    all_results = []
    loops = n_splits if split_method == 'random' else 1

    for i in range(loops):
        logger.info(f"--- Running Split {i+1}/{loops} ---")
        
        if split_method == 'random':
            indices = np.random.permutation(n_channels)
            half = n_channels // 2
            x_a = x_data[..., indices[:half]]
            x_b = x_data[..., indices[half:]]
        elif split_method == 'spatial':
            half = n_channels // 2
            x_a = x_data[..., :half]
            x_b = x_data[..., half:]
        elif split_method == 'temporal':
            lag = kwargs.get('lag', 1)
            x_a = x_data[:-lag, ...]
            x_b = x_data[lag:, ...]
        else:
            raise ValueError(f"Unknown split_method: '{split_method}'")

        sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=analysis_params)
        split_results = sweep.run(sweep_grid={}, n_workers=n_workers, is_proc_sweep=False)
        
        for res in split_results:
            res['split_id'] = i
        all_results.extend(split_results)
        
    df = pd.DataFrame(all_results)
    logger.info("--- Dimensionality Analysis Complete ---")
    return df