# neural_mi/analysis/dimensionality.py

import numpy as np
import pandas as pd
from typing import Dict, Any
import torch

from .sweep import ParameterSweep
from neural_mi.logger import logger

def run_dimensionality_analysis(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Dict[str, Any],
    n_splits: int = 5,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Estimates the latent dimensionality of a neural population using internal information.
    """
    if 'embedding_dim' not in sweep_grid:
        raise ValueError("'embedding_dim' must be in the sweep_grid for dimensionality analysis.")
    
    n_channels = x_data.shape[1]
    if n_channels < 2:
        raise ValueError("Cannot split channels; input data has fewer than 2 channels.")
    if n_channels % 2 != 0:
        logger.warning(
            f"Number of channels ({n_channels}) is odd. "
            f"Using {n_channels // 2} channels for one split and {n_channels - (n_channels // 2)} for the other. "
            "Consider using an even number of channels for more balanced splits."
        )

    all_results = []
    for i in range(n_splits):
        logger.info(f"--- Running Split {i+1}/{n_splits} ---")
        
        indices = np.random.permutation(n_channels)
        indices_a = indices[:n_channels // 2]
        indices_b = indices[n_channels // 2:]

        x_a = x_data[:, indices_a, :]
        x_b = x_data[:, indices_b, :]
        
        sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=base_params, critic_type='separable')
        split_results = sweep.run(sweep_grid=sweep_grid, n_workers=n_workers)
        
        for res in split_results:
            res['split_id'] = i
        all_results.extend(split_results)
        
    df = pd.DataFrame(all_results)
    summary_df = df.groupby('embedding_dim')['test_mi'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.rename(columns={'mean': 'mi_mean', 'std': 'mi_std'})
    
    logger.info("--- Dimensionality Analysis Complete ---")
    return summary_df