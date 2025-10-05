# neural_mi/visualize/plot.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from typing import Optional, Dict, Any

def set_publication_style():
    """Applies a professional, publication-ready style to matplotlib plots."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": "Times New Roman", "mathtext.fontset": "cm",
        'figure.dpi': 100, 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14
    })

def plot_sweep_curve(summary_df: pd.DataFrame, param_col: str, mean_col: str = 'mi_mean',
                     std_col: str = 'mi_std', true_value: Optional[float] = None,
                     estimated_values: Optional[Any] = None, ax: Optional[plt.Axes] = None,
                     units: str = 'bits', **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(summary_df[param_col], summary_df[mean_col], 'o-', label='Mean MI', **kwargs)
    ax.fill_between(summary_df[param_col], summary_df[mean_col] - summary_df[std_col],
                    summary_df[mean_col] + summary_df[std_col], alpha=0.2, label='±1 Std Dev')
    
    if true_value is not None:
        ax.axvline(x=true_value, color='r', linestyle='--', label=f'True Value = {true_value}')
    
    if isinstance(estimated_values, dict):
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(estimated_values)))
        for i, (prefix, val) in enumerate(estimated_values.items()):
             ax.axvline(x=val, color=colors[i], linestyle=':', linewidth=3, label=f'Est. ({prefix}) = {val}')
    elif estimated_values is not None:
         ax.axvline(x=estimated_values, color='g', linestyle=':', linewidth=3, label=f'Estimated = {estimated_values}')
        
    ax.set_xlabel(param_col.replace('_', ' ').title()); ax.set_ylabel(f"MI ({units})")
    ax.set_title(f"MI vs. {param_col.replace('_', ' ').title()}"); ax.legend()
    ax.grid(True, linestyle=':'); sns.despine(ax=ax)
    
    if pd.api.types.is_numeric_dtype(summary_df[param_col]) and all(summary_df[param_col] == np.floor(summary_df[param_col])):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if 'fig' in locals(): plt.tight_layout()
    return ax

def plot_bias_correction_fit(raw_results_df: pd.DataFrame, corrected_result: Dict[str, Any],
                             ax: Optional[plt.Axes] = None, units: str = 'bits', **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.stripplot(x='gamma', y='test_mi', data=raw_results_df, ax=ax, color='gray', alpha=0.5)
    agg = raw_results_df.groupby('gamma')['test_mi'].mean().reset_index()
    ax.plot(agg['gamma'] - 1, agg['test_mi'], 'o-', color='black', label='Mean MI per Gamma')

    slope, intercept = corrected_result['slope'], corrected_result['mi_corrected']
    mi_error, gammas_used = corrected_result.get('mi_error', 0), corrected_result['gammas_used']
    
    fit_x = np.array([0] + gammas_used)
    ax.plot(fit_x - 1, slope * fit_x + intercept, 'r--', linewidth=2, label='WLS Extrapolation')
    
    ax.errorbar(x=-1, y=intercept, yerr=mi_error, fmt='r*', markersize=15, capsize=5,
                label=f'Corrected MI = {intercept:.2f} ± {mi_error:.2f} {units}')
    
    ax.set_xticks(np.unique(raw_results_df['gamma']) - 1)
    ax.set_xticklabels(np.unique(raw_results_df['gamma']))
    ax.set_xlabel(r"Number of Subsets ($\gamma$)"); ax.set_ylabel(f"MI Estimate ({units})")
    ax.set_title("Bias Correction via Extrapolation"); ax.legend()
    ax.grid(True, linestyle=':'); sns.despine(ax=ax)
    
    if 'fig' in locals(): plt.tight_layout()
    return ax