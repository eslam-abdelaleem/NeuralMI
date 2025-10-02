# neural_mi/analysis/dimensionality.py

import numpy as np
import pandas as pd
import warnings
import torch
from typing import Dict, Any, Optional, List

from .sweep import ParameterSweep
from ..logger import logger

        
def run_dimensionality_analysis(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Dict[str, List[Any]],
    n_splits: int = 5,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    Estimates the latent dimensionality of a neural population using internal information.

    This function repeatedly splits the channels of x_data into two random, non-overlapping
    halves (X_A and X_B) and then uses the ParameterSweep engine to calculate the
    mutual information I(X_A; X_B) across the embedding dimensions specified in the sweep_grid.

    Args:
        x_data (torch.Tensor): The processed neural data of shape 
            [num_windows, num_channels, features].
        base_params (dict): Base parameters for the Trainer (e.g., learning_rate, n_epochs).
        sweep_grid (dict): The parameter grid to sweep over. Must contain 'embedding_dim'.
        n_splits (int): The number of random channel splits to average over for robustness.
        n_workers (int, optional): The number of parallel processes to use. 
            Defaults to cpu_count().

    Returns:
        pd.DataFrame: A dataframe summarizing the MI results for each embedding dimension,
                      averaged over the random splits.
    """
    if 'embedding_dim' not in sweep_grid:
        raise ValueError("'embedding_dim' must be in the sweep_grid for dimensionality analysis.")
    
    n_channels = x_data.shape[1]
    if n_channels < 2:
        raise ValueError("Cannot split channels for dimensionality analysis; input data has fewer than 2 channels.")

    all_results = []
    for i in range(n_splits):
        logger.info(f"--- Running Split {i+1}/{n_splits} ---")
        
        # 1. Create a random split of channel indices
        if n_channels % 2 != 0:
            warnings.warn(
                f"Number of channels ({n_channels}) is odd. "
                f"Using {n_channels // 2} channels for X_A and {n_channels // 2 + 1} for X_B. "
                f"This may introduce a slight bias. Consider using an even number of channels."
            )

        indices = np.random.permutation(n_channels)
        indices_a = indices[:n_channels // 2]
        indices_b = indices[n_channels // 2:] # Use all remaining channels

        x_a = x_data[:, indices_a, :]
        x_b = x_data[:, indices_b, :]
        
        # 2. Use the ParameterSweep engine to run the analysis for this split
        sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=base_params, critic_type='separable')
        
        split_results = sweep.run(sweep_grid=sweep_grid, n_workers=n_workers)
        
        for res in split_results:
            res['split_id'] = i
        all_results.extend(split_results)
        
    # 3. Aggregate the results
    df = pd.DataFrame(all_results)
    summary_df = df.groupby('embedding_dim')['test_mi'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.rename(columns={'mean': 'mi_mean', 'std': 'mi_std'})
    
    logger.info("--- Dimensionality Analysis Complete ---")
    return summary_df