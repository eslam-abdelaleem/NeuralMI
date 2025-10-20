# neural_mi/visualize/plot.py
"""Provides plotting functions for visualizing analysis results.

This module contains functions to generate plots for different analysis modes,
such as hyperparameter sweeps and bias correction fits. These are typically
called via the `.plot()` method of the `Results` object.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.stats import zscore
from matplotlib.ticker import MaxNLocator
from typing import Optional, Dict, Any
from scipy.spatial.distance import cdist
from matplotlib.lines import Line2D

def set_publication_style():
    """Applies a professional, publication-ready style to matplotlib plots.
    
    This function updates matplotlib's rcParams to create plots with a
    serif font (Times New Roman), appropriate font sizes for labels and
    titles, and a clean layout suitable for academic papers or reports.
    """
    plt.rcParams.update({
        "font.family": "serif", "font.serif": "Times New Roman", "mathtext.fontset": "cm",
        'figure.dpi': 100, 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14
    })

def plot_sweep_curve(summary_df: pd.DataFrame, param_col: str, mean_col: str = 'mi_mean',
                     std_col: str = 'mi_std', true_value: Optional[float] = None,
                     estimated_values: Optional[Any] = None, ax: Optional[plt.Axes] = None,
                     units: str = 'bits', **kwargs):
    """Plots the results of a hyperparameter sweep.

    This function creates a curve of the mean MI estimate against the values
    of the swept hyperparameter, with a shaded region representing the
    standard deviation. It can also display true and estimated values as
    vertical lines for comparison.

    Parameters
    ----------
    summary_df : pd.DataFrame
        A DataFrame containing the summarized results of the sweep. Must
        contain columns for the parameter, mean MI, and std dev of MI.
    param_col : str
        The name of the column in `summary_df` that contains the swept
        hyperparameter values.
    mean_col : str, optional
        The name of the column for the mean MI estimate. Defaults to 'mi_mean'.
    std_col : str, optional
        The name of the column for the standard deviation of the MI estimate.
        Defaults to 'mi_std'.
    true_value : float, optional
        If known, the true value of the parameter, to be plotted as a
        vertical dashed line. Defaults to None.
    estimated_values : Any, optional
        An estimated value or a dictionary of estimated values to plot as
        vertical dotted lines. Defaults to None.
    ax : plt.Axes, optional
        A matplotlib Axes object to plot on. If None, a new figure and axes
        are created. Defaults to None.
    units : str, optional
        The units of the MI estimate (e.g., 'bits' or 'nats') for axis labels.
        Defaults to 'bits'.
    **kwargs : dict
        Additional keyword arguments passed to `ax.plot`.
    """
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
    """Plots the results of a rigorous, bias-corrected analysis.

    This function visualizes the extrapolation fit used for bias correction.
    It shows the raw MI estimates for each data subset size (gamma), the mean
    MI at each gamma, and the final linear fit extrapolated to an infinite
    dataset size (gamma=0).

    Parameters
    ----------
    raw_results_df : pd.DataFrame
        A DataFrame containing the raw results from all training runs in the
        rigorous analysis. Must contain 'gamma' and 'test_mi' columns.
    corrected_result : Dict[str, Any]
        A dictionary containing the results of the bias correction, including
        the 'slope', 'mi_corrected', 'mi_error', and 'gammas_used'.
    ax : plt.Axes, optional
        A matplotlib Axes object to plot on. If None, a new figure and axes
        are created. Defaults to None.
    units : str, optional
        The units of the MI estimate (e.g., 'bits' or 'nats') for labels.
        Defaults to 'bits'.
    """
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
    
    if 'fig' in locals():
        plt.tight_layout()

def plot_cross_correlation(x, y, true_lag):
    """Plotting function for cross-correlation."""
    lags = np.arange(-len(x[0]) // 2 + 1, len(x[0]) // 2 + 1)
    corr = correlate(zscore(y[0]), zscore(x[0]), mode='same') / len(x[0])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, corr)
    ax.axvline(true_lag + 1, color='r', linestyle='-.', label=f'True Lag ({true_lag})')
    ax.axvline(lags[np.argmax(corr)], color='g', linestyle=':', label=f'Found Lag ({lags[np.argmax(corr)]})')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Cross-Correlation')
    ax.set_title('Linear Correlation vs Lag')
    ax.set_xlim(-100, 100)
    ax.legend()
    plt.show()

def analyze_mi_heatmap(results_df, absolute_mi_threshold=0.2, contour_rise_fraction=0.1, radius_multiplier=1.2, 
                                 true_lag=None, history_duration=None):
    """
    Performs the ultimate topological analysis of a 2D MI heatmap.
    - Finds the Causal Contour.
    - Finds the shortest "bridge" between the Causal and Significant MI Contours.
    - Draws a "Parsimonious Circle" centered on this bridge to highlight the optimal region.

    Args:
        results_df (pd.DataFrame): DataFrame with 'lag', 'window_size', and 'mi' columns.
        absolute_mi_threshold (float): The absolute MI value to consider "significant".
        contour_rise_fraction (float): Heuristic for finding the Causal Contour rise point.
        radius_multiplier (float): Factor to scale the Parsimonious Circle's radius.
        true_lag (float, optional): The true lag value to mark on the plot.
        history_duration (float, optional): The true history/window duration to mark on the plot.
    """
    # --- 1. Data Preparation ---
    heatmap_data = results_df.pivot(index='window_size', columns='lag', values='mi')
    lags = heatmap_data.columns.values
    windows = heatmap_data.index.values
    
    # --- 2. Causal Contour Analysis ---
    causal_contour_c = None
    if 0 in lags:
        lag0_data = heatmap_data[0]
        noise_floor = lag0_data.iloc[:3].median()
        peak_mi = lag0_data.max()
        rise_threshold = noise_floor + (peak_mi - noise_floor) * contour_rise_fraction
        significant_windows = lag0_data[lag0_data > rise_threshold]
        if not significant_windows.empty:
            causal_contour_c = significant_windows.index[0]
            print(f"--- Causal Contour Analysis ---")
            print(f"MI at lag=0 rises at window_size = {causal_contour_c} (implies lag_true + history_true ≈ {causal_contour_c})\n")
    else:
        print("--- Causal Contour Analysis ---\nLag=0 not found. Skipping Causal Contour estimation.\n")

    # --- 3. Create the main figure for all analysis ---
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Use pcolormesh instead of seaborn heatmap for consistent coordinates
    # Need to create mesh grid edges for pcolormesh
    lag_edges = np.concatenate([lags - (lags[1] - lags[0])/2, [lags[-1] + (lags[1] - lags[0])/2]])
    window_edges = np.concatenate([windows - (windows[1] - windows[0])/2, [windows[-1] + (windows[1] - windows[0])/2]])
    
    mesh = ax.pcolormesh(lag_edges, window_edges, heatmap_data.values, cmap='viridis', shading='flat')
    cbar = plt.colorbar(mesh, ax=ax, label='Mutual Information')
    
    # --- 4. Significant Zone & Parsimony Analysis ---
    print(f"--- Parsimony Analysis (Significant MI > {absolute_mi_threshold}) ---")
    
    # Create contour on the same axes
    cs = ax.contour(lags, windows, heatmap_data.values, levels=[absolute_mi_threshold], 
                    colors='red', linewidths=2.5, linestyles='-')
    
    if not cs.allsegs[0]:
        print("Warning: No significant MI contour found. Try a lower threshold.")
        ax.set_title('Parsimony-Informed Topological Analysis (No Significant Contour Found)')
        ax.set_xlabel('Lag (Timepoints)')
        ax.set_ylabel('Window Size (Timepoints)')
        plt.show()
        return

    # Extract the largest continuous contour segment
    significant_contour_points = np.array(max(cs.allsegs[0], key=len))

    midpoint, radius = None, None
    if causal_contour_c is not None:
        # Define the Causal Contour line *only within the plot's window range*
        causal_lags = lags[(causal_contour_c - lags >= windows.min()) & (causal_contour_c - lags <= windows.max())]
        causal_contour_line = np.array([[lg, causal_contour_c - lg] for lg in causal_lags])
        
        if causal_contour_line.size > 0:
            # Draw the Causal Contour line
            ax.plot(causal_contour_line[:, 0], causal_contour_line[:, 1], 
                   color='cyan', linestyle='--', linewidth=3, label=f'Causal Contour (C≈{causal_contour_c})')
            
            # Find the shortest distance between the two contours
            distances = cdist(significant_contour_points, causal_contour_line)
            min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
            point_on_mi_contour = significant_contour_points[min_dist_idx[0]]
            point_on_causal_contour = causal_contour_line[min_dist_idx[1]]
            
            midpoint = (point_on_mi_contour + point_on_causal_contour) / 2
            bridge_length = np.linalg.norm(point_on_mi_contour - point_on_causal_contour)
            radius = (bridge_length / 2) * radius_multiplier
            
            print(f"Shortest bridge is between {point_on_causal_contour} on Causal Contour")
            print(f"and {point_on_mi_contour} on Significant MI Contour.")
            print(f"Bridge length: {bridge_length:.2f}")
            print(f"Parsimonious Center: (lag={midpoint[0]:.1f}, window={midpoint[1]:.1f})")
            print(f"Parsimonious Radius: {radius:.2f}")
            
            # Draw the bridge line
            ax.plot([point_on_causal_contour[0], point_on_mi_contour[0]], 
                   [point_on_causal_contour[1], point_on_mi_contour[1]], 
                   'orange', linewidth=2, linestyle='-', alpha=0.7)
            
            # Draw the Parsimonious Circle
            circle = patches.Circle(midpoint, radius, linewidth=2.5, edgecolor='white', 
                                   facecolor='none', linestyle=':', label='Parsimonious Region')
            ax.add_patch(circle)
            
            # Mark the center
            ax.plot(midpoint[0], midpoint[1], 'w+', markersize=15, mew=3, 
                   label='Parsimonious Center')
    
    # --- 5. Mark True Parameter Box ---
    if true_lag is not None and history_duration is not None:
        # Calculate the box edges (half a step in each direction)
        lag_step = lags[1] - lags[0] if len(lags) > 1 else 1
        window_step = windows[1] - windows[0] if len(windows) > 1 else 1
        
        # Create rectangle centered on the true values
        true_rect = patches.Rectangle(
            (true_lag - lag_step/2, history_duration - window_step/2),
            lag_step, window_step,
            linewidth=3, edgecolor='lime', facecolor='none', linestyle='-'
        )
        ax.add_patch(true_rect)
        print(f"\n--- True Parameters ---")
        print(f"True lag: {true_lag}, True history: {history_duration}")
    
    # Manually create legend handles for a clean legend
    legend_elements = []
    if causal_contour_c is not None:
        legend_elements.append(Line2D([0], [0], color='cyan', lw=3, ls='--', 
                                     label=f'Causal Contour (C≈{causal_contour_c})'))
    legend_elements.append(Line2D([0], [0], color='red', lw=2.5, 
                                 label=f'Significant MI Contour (>{absolute_mi_threshold})'))
    if midpoint is not None:
        legend_elements.append(Line2D([0], [0], color='orange', lw=2, 
                                     label='Bridge (shortest distance)', alpha=0.7))
        legend_elements.append(Line2D([0], [0], color='white', lw=2.5, ls=':', 
                                     label='Parsimonious Region'))
        legend_elements.append(Line2D([0], [0], marker='+', color='w', 
                                     label='Parsimonious Center', ls='none', mew=3, markersize=12))
    if true_lag is not None and history_duration is not None:
        legend_elements.append(patches.Rectangle((0, 0), 1, 1, linewidth=3, 
                                                 edgecolor='lime', facecolor='none', 
                                                 label='True Parameters'))
        
    ax.set_title('Parsimony-Informed Topological Analysis')
    ax.set_xlabel('Lag (Timepoints)')
    ax.set_ylabel('Window Size (Timepoints)')
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Fix axis limits to heatmap data range to prevent rescaling
    ax.set_xlim(lags.min(), lags.max())
    ax.set_ylim(windows.min(), windows.max())
    
    # Show all tick values if we have fewer than 20 of them
    if len(lags) < 25:
        ax.set_xticks(lags)
    if len(windows) < 25:
        ax.set_yticks(windows)
    
    plt.tight_layout()
    plt.show()