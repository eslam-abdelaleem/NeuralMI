# neural_mi/utils/helpers.py

import torch

def get_device():
    """
    Determines the appropriate device for PyTorch computations (CUDA, MPS, or CPU).

    Returns:
        str: The device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def find_saturation_point(
    summary_df, 
    param_col='embedding_dim', 
    mean_col='mi_mean', 
    std_col='mi_std', 
    strictness=None
):
    """
    Finds the saturation point of a curve for given strictness levels.

    The saturation point is defined as the first point 'k' where the increase 
    in the mean value is less than a fraction of the standard deviation.

    Args:
        summary_df (pd.DataFrame): A DataFrame with aggregated MI results.
        param_col (str): The name of the column to treat as the independent variable 
                         (e.g., 'embedding_dim', 'window_size').
        mean_col (str): The name of the column containing the mean MI values. 
                        Defaults to 'mi_mean'.
        std_col (str): The name of the column containing the standard deviation of MI. 
                       Defaults to 'mi_std'.
        strictness (float or list, optional): A factor controlling the saturation criterion.
            If None, defaults to [0.1, 5].

    Returns:
        float or dict: The estimated saturation point(s).
    """
    if strictness is None:
        strictness = [0.1, 5]

    if isinstance(strictness, (list, tuple)):
        results = {}
        for s in strictness:
            results[s] = find_saturation_point(
                summary_df, 
                param_col=param_col, 
                mean_col=mean_col, 
                std_col=std_col, 
                strictness=s
            )
        return results

    for col in [param_col, mean_col, std_col]:
        if col not in summary_df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
        
    df = summary_df.sort_values(param_col).reset_index()
    df['mi_diff'] = df[mean_col].diff().fillna(float('inf'))
    
    saturation_candidates = df[df['mi_diff'] < strictness * df[std_col]]
    
    if not saturation_candidates.empty:
        return saturation_candidates[param_col].iloc[0]
    else:
        return df.loc[df[mean_col].idxmax()][param_col]