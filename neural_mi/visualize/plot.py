# neural_mi/visualize/plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

def set_publication_style():
    """Applies a professional, publication-ready style to matplotlib plots."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14})

def plot_sweep_curve(
    summary_df,
    param_col='embedding_dim',
    mean_col='mi_mean',
    std_col='mi_std',
    true_value=None,
    estimated_values=None,
    ax=None,
    units='bits',
    **kwargs
):
    """
    Plots a generic curve from sweep results with a shaded error band.

    Args:
        summary_df (pd.DataFrame): DataFrame with aggregated sweep results.
        param_col (str): Column for the x-axis parameter (e.g., 'embedding_dim', 'window_size').
        mean_col (str): Column for the y-axis mean values.
        std_col (str): Column for the y-axis standard deviation.
        true_value (float, optional): Plots a vertical line for a known ground truth.
        estimated_values (float or dict, optional): Plots vertical line(s) for estimated points.
        ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.
        units (str): The units for the y-axis label.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Use the generic column names
    ax.plot(summary_df[param_col], summary_df[mean_col], 'o-', label='Mean MI', **kwargs)
    ax.fill_between(
        summary_df[param_col],
        summary_df[mean_col] - summary_df[std_col],
        summary_df[mean_col] + summary_df[std_col],
        alpha=0.2,
        label='±1 Std Dev'
    )
    
    if true_value is not None:
        ax.axvline(x=true_value, color='r', linestyle='--', label=f'True Value = {true_value}')
    
    if isinstance(estimated_values, dict):
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(estimated_values)))
        for i, (prefix, val) in enumerate(estimated_values.items()):
             ax.axvline(x=val, color=colors[i], linestyle=':', linewidth=3,
                        label=f'Est. ({prefix}) = {val}')
    elif estimated_values is not None:
         ax.axvline(x=estimated_values, color='g', linestyle=':', linewidth=3,
                    label=f'Estimated Value = {estimated_values}')
        
    # Generic Labels
    ax.set_xlabel(param_col.replace('_', ' ').title())
    ax.set_ylabel(f"MI ({units})")
    ax.set_title(f"MI vs. {param_col.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, linestyle=':')
    sns.despine(ax=ax)
    
    # Check if the parameter column is numeric before forcing integer ticks
    if pd.api.types.is_numeric_dtype(summary_df[param_col]):
        # If all values in param_col are integers, force integer ticks
        if all(summary_df[param_col] == np.floor(summary_df[param_col])):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if 'fig' in locals():
        plt.tight_layout()
    return ax

def plot_bias_correction_fit(
    raw_results_df,
    corrected_result,
    ax=None,
    units='bits',
    **kwargs
):
    """
    Visualizes the bias correction WLS fit.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.stripplot(x='gamma', y='test_mi', data=raw_results_df, ax=ax, color='gray', alpha=0.5)
    agg = raw_results_df.groupby('gamma')['test_mi'].mean().reset_index()
    ax.plot(agg['gamma'] - 1, agg['test_mi'], 'o-', color='black', label='Mean MI per Gamma')

    slope = corrected_result['slope']
    intercept = corrected_result['mi_corrected']
    mi_error = corrected_result.get('mi_error', 0) # Get error, default to 0 if not present
    gammas_used = corrected_result['gammas_used']
    
    fit_x = np.array([0] + gammas_used)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x - 1, fit_y, 'r--', linewidth=2, label='WLS Extrapolation')
    
    # --- Use ax.errorbar to plot the point and its error range ---
    ax.errorbar(
        x=0 - 1,
        y=intercept,
        yerr=mi_error,
        fmt='r*',          # Format for the marker (red star)
        markersize=15,
        capsize=5,         # Adds caps to the error bar
        label=f'Corrected MI = {intercept:.2f} ± {mi_error:.2f} {units}'
    )
    
    ax.set_xticks(np.unique(raw_results_df['gamma']) - 1)
    ax.set_xticklabels(np.unique(raw_results_df['gamma']))
    ax.set_xlabel(r"Number of Subsets ($\gamma$)")
    ax.set_ylabel(f"MI Estimate ({units})")
    ax.set_title("Bias Correction via Extrapolation")
    ax.legend()
    ax.grid(True, linestyle=':')
    sns.despine(ax=ax)
    
    if 'fig' in locals():
        plt.tight_layout()
    return ax