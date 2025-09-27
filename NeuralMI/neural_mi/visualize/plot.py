# neural_mi/visualize/plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_publication_style():
    """Applies a professional, publication-ready style to matplotlib plots."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14})

def plot_dimensionality_curve(
    summary_df,
    true_dimensionality=None,
    estimated_dimensionality=None,
    ax=None,
    **kwargs
):
    """
    Plots the MI saturation curve from a dimensionality analysis.

    Args:
        summary_df (pd.DataFrame): The output from run_dimensionality_analysis.
        true_dimensionality (int, optional): Plots a vertical line for ground truth.
        estimated_dimensionality (int, optional): Plots a vertical line for the
            estimated saturation point.
        ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(summary_df['embedding_dim'], summary_df['mi_mean'], 'o-', label='Mean Internal MI', **kwargs)
    ax.fill_between(
        summary_df['embedding_dim'],
        summary_df['mi_mean'] - summary_df['mi_std'],
        summary_df['mi_mean'] + summary_df['mi_std'],
        alpha=0.2,
        label='±1 Std Dev'
    )
    
    if true_dimensionality is not None:
        ax.axvline(x=true_dimensionality, color='r', linestyle='--', label=f'True Dim = {true_dimensionality}')
    
    if estimated_dimensionality is not None:
        ax.axvline(x=estimated_dimensionality, color='g', linestyle=':', linewidth=3,
                    label=f'Estimated Dim = {estimated_dimensionality}')
        
    ax.set_xlabel(r"Embedding Dimension ($k_Z$)")
    ax.set_ylabel("Internal Information (nats)")
    ax.set_title("Dimensionality Saturation Curve")
    ax.legend()
    ax.grid(True, linestyle=':')
    sns.despine(ax=ax)
    
    if 'fig' in locals():
        plt.tight_layout()
    return ax

def plot_bias_correction_fit(
    raw_results_df,
    corrected_result,
    ax=None,
    **kwargs
):
    """
    Visualizes the bias correction WLS fit.
    (This function remains the same as before)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.stripplot(x='gamma', y='test_mi', data=raw_results_df, ax=ax, color='gray', alpha=0.5)#, label='Individual Runs')
    agg = raw_results_df.groupby('gamma')['test_mi'].mean().reset_index()
    ax.plot(agg['gamma'] - 1, agg['test_mi'], 'o-', color='black', label='Mean MI per Gamma')

    slope = corrected_result['slope']
    intercept = corrected_result['mi_corrected']
    gammas_used = corrected_result['gammas_used']
    fit_x = np.array([0] + gammas_used)
    fit_y = slope * fit_x + intercept
    
    ax.plot(fit_x - 1, fit_y, 'r--', linewidth=2, label='WLS Extrapolation')
    ax.plot(0 - 1, intercept, 'r*', markersize=15, label=f'Corrected MI = {intercept:.3f}')
    
    ax.set_xticks(np.unique(raw_results_df['gamma']) - 1)
    ax.set_xticklabels(np.unique(raw_results_df['gamma']))
    ax.set_xlabel(r"Number of Subsets ($\gamma$)")
    ax.set_ylabel("MI Estimate (nats)")
    ax.set_title("Bias Correction via Extrapolation")
    ax.legend()
    ax.grid(True, linestyle=':')
    sns.despine(ax=ax)
    
    if 'fig' in locals():
        plt.tight_layout()
    return ax