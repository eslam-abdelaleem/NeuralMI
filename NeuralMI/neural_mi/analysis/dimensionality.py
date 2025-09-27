# neural_mi/analysis/dimensionality.py

import numpy as np
import pandas as pd
from .sweep import ParameterSweep

def find_saturation_point(summary_df, strictness=0.5):
    """
    Finds the saturation point of a dimensionality curve.

    The saturation point is defined as the first embedding dimension k where the
    increase in MI from k-1 to k is less than a fraction of the standard
    deviation at k. (i.e., MI(k) - MI(k-1) < strictness * STD(k))

    Args:
        summary_df (pd.DataFrame): The output from run_dimensionality_analysis.
        strictness (float): A factor controlling how strict the saturation
            criterion is. Lower values are stricter. Defaults to 0.5.

    Returns:
        int: The estimated latent dimensionality.
    """
    df = summary_df.sort_values('embedding_dim').reset_index()
    
    # Calculate the difference in MI between consecutive dimensions
    df['mi_diff'] = df['mi_mean'].diff().fillna(float('inf')) # First diff is infinite
    
    # The saturation point is where the increase is less than strictness*std
    saturation_candidates = df[df['mi_diff'] < strictness * df['mi_std']]
    
    if not saturation_candidates.empty:
        # Return the first dimension that meets the criterion
        return saturation_candidates['embedding_dim'].iloc[0]
    else:
        # If no saturation is found, return the dimension with the highest MI
        return df.loc[df['mi_mean'].idxmax()]['embedding_dim']

        
def run_dimensionality_analysis(
    x_data,
    base_params,
    sweep_grid,
    n_splits=5,
    n_workers=None
):
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
        print(f"\n--- Running Split {i+1}/{n_splits} ---")
        
        # 1. Create a random split of channel indices
        indices = np.random.permutation(n_channels)
        indices_a = indices[:n_channels // 2]
        indices_b = indices[n_channels // 2 : 2 * (n_channels // 2)] # Ensures equal size

        x_a = x_data[:, indices_a, :]
        x_b = x_data[:, indices_b, :]
        
        # 2. Use the ParameterSweep engine to run the analysis for this split
        # We assume a separable critic, as it's required for varying embedding_dim
        sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=base_params, critic_type='separable')
        
        split_results = sweep.run(sweep_grid=sweep_grid, n_workers=n_workers)
        
        for res in split_results:
            res['split_id'] = i
        all_results.extend(split_results)
        
    # 3. Aggregate the results
    df = pd.DataFrame(all_results)
    # Group by embedding dimension and calculate mean and std of the MI
    summary_df = df.groupby('embedding_dim')['test_mi'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.rename(columns={'mean': 'mi_mean', 'std': 'mi_std'})
    
    print("\n--- Dimensionality Analysis Complete ---")
    return summary_df