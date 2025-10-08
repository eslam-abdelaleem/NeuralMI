# neural_mi/analysis/dimensionality.py
"""Estimates the latent dimensionality of a dataset using internal information.

This module provides a function to perform a dimensionality analysis by
repeatedly splitting the channels of a dataset in half and measuring the
mutual information between the two splits across a range of embedding
dimensions. The MI is expected to saturate when the model's embedding capacity
matches the data's intrinsic dimensionality.
"""
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
    """Estimates latent dimensionality via split-half internal information analysis.

    This function measures the mutual information between two random, non-overlapping
    halves of the input channels (`x_data`). This process is repeated `n_splits`
    times with different random splits, and the results are averaged. The
    analysis is performed for each `embedding_dim` specified in the `sweep_grid`.

    The resulting dataframe can be used to find the "saturation point" where
    increasing the model's embedding dimension no longer increases the measured
    mutual information, providing an estimate of the data's intrinsic dimensionality.

    Parameters
    ----------
    x_data : torch.Tensor
        A 3D tensor of shape `(n_samples, n_channels, n_features)` representing
        the preprocessed input data.
    base_params : Dict[str, Any]
        A dictionary of fixed parameters for the MI estimator's trainer, such as
        `n_epochs`, `learning_rate`, etc.
    sweep_grid : Dict[str, Any]
        A dictionary defining the parameter grid for the sweep. It *must*
        contain the key 'embedding_dim' with a list of dimensions to test.
    n_splits : int, optional
        The number of random channel splits to perform. Results are averaged
        across these splits for robustness. Defaults to 5.
    n_workers : int, optional
        The number of worker processes to use for parallelizing the parameter
        sweep. If None, it runs sequentially. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the results, with columns 'embedding_dim',
        'mi_mean', and 'mi_std', showing the mean and standard deviation of
        the MI estimate for each tested embedding dimension.

    Raises
    ------
    ValueError
        If 'embedding_dim' is not in `sweep_grid` or if `x_data` has fewer
        than 2 channels.
    """
    if 'embedding_dim' not in sweep_grid:
        raise ValueError("'embedding_dim' must be in the sweep_grid for dimensionality analysis.")
    
    # For dimensionality, SMILE is often better. Default to it and warn the user.
    if 'estimator_name' not in base_params:
        logger.info("Defaulting to 'smile' estimator for dimensionality analysis, as it is less biased for this task.")
        base_params['estimator_name'] = 'smile'
        base_params.setdefault('estimator_params', {}).setdefault('clip', 5.0)
    elif base_params['estimator_name'] != 'smile':
        logger.warning(
            f"Using '{base_params['estimator_name']}' estimator for dimensionality analysis. "
            "For this specific mode, consider using the 'smile' estimator, as its lower "
            "bias may reveal the saturation point more clearly."
        )
    
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
        
        # Data is already processed. We pass it directly to the sweep.
        # is_proc_sweep=False ensures the worker doesn't try to re-process it.
        sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=base_params, critic_type='separable')
        split_results = sweep.run(sweep_grid=sweep_grid, n_workers=n_workers, is_proc_sweep=False)
        
        for res in split_results:
            res['split_id'] = i
        all_results.extend(split_results)
        
    df = pd.DataFrame(all_results)
    if df.empty:
        return pd.DataFrame(columns=['embedding_dim', 'mi_mean', 'mi_std'])

    summary_df = df.groupby('embedding_dim')['test_mi'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.rename(columns={'mean': 'mi_mean', 'std': 'mi_std'}).fillna(0)
    
    logger.info("--- Dimensionality Analysis Complete ---")
    return summary_df