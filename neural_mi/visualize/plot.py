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
from neural_mi.logger import logger as _logger

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
                     units: str = 'bits', show: bool = True, **kwargs):
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
    show : bool, optional
        Whether to call ``plt.show()`` at the end.  Set to ``False`` when
        embedding this plot in a larger figure.  Defaults to ``True``.
    **kwargs : dict
        Additional keyword arguments passed to `ax.plot`.

    Returns
    -------
    plt.Axes
        The axes containing the plot.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

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

    if created_fig: plt.tight_layout()
    if show: plt.show()
    return ax


def plot_sweep_heatmap(summary_df: pd.DataFrame, param_x: str, param_y: str,
                       mean_col: str = 'mi_mean', ax: Optional[plt.Axes] = None,
                       units: str = 'bits', show: bool = True, **kwargs) -> plt.Axes:
    """Plots a 2-parameter sweep as a heatmap (``param_x`` × ``param_y`` → MI).

    Use this instead of :func:`plot_sweep_curve` when exactly two parameters
    were swept together — a 1-D line/scatter would otherwise have to collapse
    one of the two parameters, hiding its effect on MI.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``param_x``, ``param_y``, and ``mean_col`` columns, with
        one row per ``(param_x, param_y)`` combination (as produced by
        ``mode='sweep'`` / ``mode='lag'`` with two swept parameters).
    param_x, param_y : str
        Column names for the two swept parameters. ``param_x`` becomes the
        heatmap's columns (horizontal axis), ``param_y`` its rows (vertical
        axis).
    mean_col : str, optional
        Column holding the value to colour by. Defaults to ``'mi_mean'``.
    ax : plt.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    units : str, optional
        MI units for the colourbar label. Defaults to ``'bits'``.
    show : bool, optional
        Whether to call ``plt.show()`` at the end. Defaults to ``True``.
    **kwargs
        Additional keyword arguments forwarded to ``sns.heatmap`` (e.g.
        ``cmap``, ``fmt``, ``annot``).

    Returns
    -------
    plt.Axes
    """
    pivot = summary_df.pivot(index=param_y, columns=param_x, values=mean_col)
    # Row/column order follows the swept values' natural sort order rather
    # than groupby's first-seen order, so the grid reads left-to-right /
    # bottom-to-top as ascending parameter values.
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    created_fig = ax is None
    if created_fig:
        figsize = kwargs.pop('figsize', (max(4, pivot.shape[1] * 0.9 + 1.5),
                                         max(3, pivot.shape[0] * 0.7 + 1.0)))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        kwargs.pop('figsize', None)

    annot = kwargs.pop('annot', True)
    fmt = kwargs.pop('fmt', '.3f')
    cmap = kwargs.pop('cmap', 'viridis')

    sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap,
               cbar_kws={'label': f'MI ({units})'}, ax=ax, **kwargs)
    ax.set_xlabel(param_x.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(param_y.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"MI vs. {param_x.replace('_', ' ').title()} × "
                f"{param_y.replace('_', ' ').title()}", fontsize=13)

    if created_fig: plt.tight_layout()
    if show: plt.show()
    return ax


def plot_sweep_bar(summary_df: pd.DataFrame, param_cols: list, mean_col: str = 'mi_mean',
                   std_col: str = 'mi_std', ax: Optional[plt.Axes] = None,
                   units: str = 'bits', show: bool = True, **kwargs) -> plt.Axes:
    """Plots a 3-or-more-parameter sweep as a bar chart over parameter combinations.

    Each bar is one combination of ``param_cols``, labelled on the x-axis as
    ``"p1=v1, p2=v2, ..."`` and coloured by ``mean_col`` with error bars from
    ``std_col``. Intended for sweeps with too many swept parameters for a
    heatmap (which only has two axes); see :func:`plot_sweep_heatmap` for the
    2-parameter case.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain every column in ``param_cols`` plus ``mean_col`` (and
        ``std_col`` if present), one row per parameter combination.
    param_cols : list of str
        Column names of the swept parameters to combine into each bar's label.
    mean_col : str, optional
        Column holding the bar height. Defaults to ``'mi_mean'``.
    std_col : str, optional
        Column holding the error-bar half-height. Defaults to ``'mi_std'``;
        silently skipped if absent.
    ax : plt.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    units : str, optional
        MI units for the y-axis label. Defaults to ``'bits'``.
    show : bool, optional
        Whether to call ``plt.show()`` at the end. Defaults to ``True``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.bar``.

    Returns
    -------
    plt.Axes
    """
    df = summary_df.sort_values(param_cols).reset_index(drop=True)
    # Build labels from each column's own native values (via .tolist()) rather
    # than a row-wise .apply(axis=1), which would construct a per-row Series
    # spanning all param_cols and upcast integer columns to float wherever
    # another swept column in the same row is a float (e.g. embedding_dim=4
    # rendering as "4.0" because a sibling column is a float dropout value).
    column_values = [df[col].tolist() for col in param_cols]
    labels = [
        ', '.join(f'{col}={val}' for col, val in zip(param_cols, combo))
        for combo in zip(*column_values)
    ]

    created_fig = ax is None
    if created_fig:
        figsize = kwargs.pop('figsize', (max(6, len(df) * 0.6 + 1.5), 5))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        kwargs.pop('figsize', None)

    yerr = df[std_col] if std_col in df.columns else None
    color = kwargs.pop('color', 'steelblue')
    ax.bar(labels, df[mean_col], yerr=yerr, capsize=4, color=color,
          edgecolor='white', **kwargs)
    ax.set_ylabel(f'MI ({units})', fontsize=12)
    ax.set_title(f"MI vs. {', '.join(c.replace('_', ' ').title() for c in param_cols)}",
                fontsize=13)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.grid(True, axis='y', linestyle=':')
    sns.despine(ax=ax)

    if created_fig: plt.tight_layout()
    if show: plt.show()
    return ax


def plot_dimensionality_curve(
    details: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Per-rank chart of which directions of shared structure are trustworthy.

    Visualizes ``result.details['stability_per_rank']`` (plus
    ``'stable_directions'`` and ``'stable_but_degenerate_groups'``): one bar
    per embedding rank, height = mean singular-value strength across splits
    (log scale), colored by status --

    - **stable** (green): reproducible across every split/rerun, above the
      noise floor, individually trustworthy.
    - **stable, degenerate group** (amber, hatched): reproducible and above
      the noise floor, but too close in strength to an adjacent rank to
      individually order -- existence confirmed, identity not claimed. Ranks
      in the same group share a bracket above their bars.
    - **not stable / below noise floor** (gray): not reported as trustworthy,
      either because it didn't reproduce across splits or its strength is
      indistinguishable from noise.

    This does not plot ``pr_eig``/``pr_singular`` (see
    ``result.dataframe`` for those, kept as a secondary, non-headline
    diagnostic) or an MI-vs-embedding-dim curve -- this mode does not sweep
    embedding_dim or claim a saturation point.

    Parameters
    ----------
    details : dict
        ``result.details`` from a ``mode='dimensionality'`` run. Must contain
        ``'stability_per_rank'`` (a dict as produced by
        ``_compute_stability_report`` in ``analysis/dimensionality.py``).
    ax : plt.Axes, optional
        Axes to plot on. Creates a new figure if ``None``.
    show : bool, optional
        Whether to call ``plt.show()`` at the end. Defaults to True.
    **kwargs
        Additional keyword arguments forwarded to ``ax.bar``.

    Returns
    -------
    plt.Axes
    """
    per_rank = details.get('stability_per_rank')
    if not per_rank:
        raise ValueError(
            "Cannot plot: details does not contain 'stability_per_rank' (need at "
            "least 2 splits/reruns to compute cross-run stability -- see the "
            "warning emitted by run_dimensionality_analysis if this is missing)."
        )

    ranks = sorted(per_rank.keys())
    stable_set = set(details.get('stable_directions', []))
    degenerate_groups = details.get('stable_but_degenerate_groups', [])
    degenerate_set = {r for group in degenerate_groups for r in group}

    strengths = [per_rank[r].get('mean_strength') or 0.0 for r in ranks]
    colors, hatches = [], []
    for r in ranks:
        if r in stable_set:
            colors.append('#2ca02c'); hatches.append(None)
        elif r in degenerate_set:
            colors.append('#d4a017'); hatches.append('//')
        else:
            colors.append('#b0b0b0'); hatches.append(None)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (8, 5)))

    bars = ax.bar([str(r) for r in ranks], strengths, color=colors,
                  edgecolor='white', linewidth=0.5, **kwargs)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    if any(s > 0 for s in strengths):
        ax.set_yscale('log')
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Mean singular-value strength (across splits)', fontsize=11)
    ax.set_title('Dimensionality: cross-run-stable directions', fontsize=12)

    # Bracket contiguous degenerate groups above their bars.
    if strengths:
        y_top = max(strengths) * 1.15
        for group in degenerate_groups:
            xs = [ranks.index(r) for r in group]
            ax.plot([min(xs), max(xs)], [y_top, y_top], color='#d4a017', lw=1.5)
            ax.text(np.mean(xs), y_top * 1.05, 'grouped', ha='center', fontsize=8, color='#d4a017')

    legend_handles = [
        patches.Patch(facecolor='#2ca02c', label='Stable (individually trustworthy)'),
        patches.Patch(facecolor='#d4a017', hatch='//', label='Stable, degenerate group'),
        patches.Patch(facecolor='#b0b0b0', label='Not stable / below noise floor'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    ax.grid(True, axis='y', linestyle=':')
    sns.despine(ax=ax)

    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_bias_correction_fit(raw_results_df: pd.DataFrame, corrected_result: Dict[str, Any],
                             ax: Optional[plt.Axes] = None, units: str = 'bits',
                             show: bool = True, label: Optional[str] = None,
                             color: Optional[str] = None, **kwargs):
    """Plots the results of a rigorous, bias-corrected analysis.

    This function visualizes the extrapolation fit used for bias correction.
    It shows the raw MI estimates for each data subset size (gamma), the mean
    MI at each gamma, and the final linear fit extrapolated to an infinite
    dataset size (gamma=0).

    Parameters
    ----------
    raw_results_df : pd.DataFrame
        A DataFrame containing the raw results from all training runs in the
        rigorous analysis. Must contain 'gamma' and 'train_mi' columns.
    corrected_result : Dict[str, Any]
        A dictionary containing the results of the bias correction, including
        the 'slope', 'mi_corrected', 'mi_error', and 'gammas_used'.
    ax : plt.Axes, optional
        A matplotlib Axes object to plot on. If None, a new figure and axes
        are created. Defaults to None.
    units : str, optional
        The units of the MI estimate (e.g., 'bits' or 'nats') for labels.
        Defaults to 'bits'.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.  Set to ``False`` when
        embedding this plot in a larger figure.  Defaults to ``True``.
    label : str, optional
        Name for this result, used when overlaying several fits (e.g. from
        ``Results.compare()``). If given, the raw points/mean line/fit
        line/corrected-MI marker collapse to a single legend entry under this
        label instead of each carrying its own generic description. Defaults
        to None (single-result appearance: three descriptive legend entries).
    color : str, optional
        Color for all of this result's plotted elements (raw points, mean
        line, fit line, corrected-MI marker). If None, uses the original
        single-result scheme (gray points, black mean line, red fit/marker).
        Defaults to None.

    Returns
    -------
    plt.Axes
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    raw_color = 'gray' if color is None else color
    mean_color = 'black' if color is None else color
    fit_color = 'red' if color is None else color

    sns.stripplot(x='gamma', y='train_mi', data=raw_results_df, ax=ax, color=raw_color, alpha=0.5)
    agg = raw_results_df.groupby('gamma')['train_mi'].mean().reset_index()

    # A single label collapses the three elements below to one legend entry
    # (via the proxy artist added after them) rather than three near-duplicate
    # ones -- readable when compare() overlays several results on one ax.
    element_label = '_nolegend_' if label is not None else 'Mean MI per Gamma'
    ax.plot(agg['gamma'] - 1, agg['train_mi'], 'o-', color=mean_color, label=element_label)

    slope, intercept = corrected_result['slope'], corrected_result['mi_corrected']
    mi_error, gammas_used = corrected_result.get('mi_error', 0), corrected_result['gammas_used']

    fit_x = np.array([0] + gammas_used)
    fit_label = '_nolegend_' if label is not None else 'WLS Extrapolation'
    ax.plot(fit_x - 1, slope * fit_x + intercept, linestyle='--', color=fit_color,
             linewidth=2, label=fit_label)

    errorbar_label = ('_nolegend_' if label is not None
                       else f'Corrected MI = {intercept:.2f} ± {mi_error:.2f} {units}')
    ax.errorbar(x=-1, y=intercept, yerr=mi_error, marker='*', linestyle='None',
                color=fit_color, markersize=15, capsize=5, label=errorbar_label)

    if label is not None:
        # Zero-length proxy artist purely to give this result one legend entry.
        ax.plot([], [], marker='*', linestyle='--', color=fit_color, markersize=10, label=label)

    ax.set_xticks(np.unique(raw_results_df['gamma']) - 1)
    ax.set_xticklabels(np.unique(raw_results_df['gamma']))
    ax.set_xlabel(r"Number of Subsets ($\gamma$)"); ax.set_ylabel(f"MI Estimate ({units})")
    ax.set_title("Bias Correction via Extrapolation"); ax.legend()
    ax.grid(True, linestyle=':'); sns.despine(ax=ax)

    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()
    return ax

def plot_embeddings(
    z: np.ndarray,
    color: Optional[np.ndarray] = None,
    method: str = 'auto',
    dim: int = 2,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Visualize learned embeddings in 2-D or 3-D.

    Parameters
    ----------
    z : np.ndarray
        Embedding array of shape ``(n_samples, embed_dim)``.
    color : np.ndarray, optional
        Length-n array of values used for colouring points.  Continuous arrays
        produce a colormap; integer / string arrays produce a discrete palette
        with a legend.  Defaults to None (uniform colour).
    method : {'auto', 'none', 'pca', 'tsne', 'umap'}, default='auto'
        Dimensionality-reduction method applied before plotting:

        - ``'none'`` — use the first ``dim`` dimensions directly (requires
          ``embed_dim >= dim``).
        - ``'pca'`` — sklearn PCA (always available).
        - ``'tsne'`` — sklearn t-SNE.
        - ``'umap'`` — UMAP (requires the ``umap-learn`` package).
        - ``'auto'`` — uses ``'none'`` if ``embed_dim <= dim``, else tries
          ``'umap'``, falls back to ``'pca'``.
    dim : {2, 3}, default=2
        Output dimensionality: 2 → 2-D scatter, 3 → 3-D scatter.
    title : str, optional
        Plot title.  Defaults to None.
    ax : plt.Axes, optional
        Axes to plot on.  Created automatically if None.  For dim=3 the axes
        must be a 3-D axes (``projection='3d'``).
    **kwargs
        Additional keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    plt.Axes
        The axes containing the plot.

    Examples
    --------
    >>> zx, zy = nmi.extract_embeddings('model.pt', x_test, y_test)
    >>> ax = nmi.visualize.plot_embeddings(zx, color=labels, method='pca')
    >>> plt.show()
    """
    from neural_mi.logger import logger as _logger

    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError(f"z must be 2-D (n_samples, embed_dim), got shape {z.shape}.")
    n_samples, embed_dim = z.shape

    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}.")

    # --- Resolve method ---
    if method == 'auto':
        if embed_dim <= dim:
            method = 'none'
        else:
            import importlib.util
            # Check availability without importing yet -- the 'umap' branch
            # below does the real import when it's actually used.
            method = 'umap' if importlib.util.find_spec('umap') is not None else 'pca'

    # --- Apply dimensionality reduction ---
    if method == 'none':
        if embed_dim < dim:
            raise ValueError(
                f"method='none' requires embed_dim >= dim, but embed_dim={embed_dim} < dim={dim}."
            )
        z_plot = z[:, :dim]
    elif method == 'pca':
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                'PCA requires scikit-learn. Install it with: pip install "neural_mi[viz]"'
            )
        z_plot = PCA(n_components=dim).fit_transform(z)
    elif method == 'tsne':
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError(
                't-SNE requires scikit-learn. Install it with: pip install "neural_mi[viz]"'
            )
        z_plot = TSNE(n_components=dim, **{k: v for k, v in kwargs.items()
                                            if k in ('perplexity', 'learning_rate', 'n_iter',
                                                      'random_state', 'init')}).fit_transform(z)
        # Remove t-SNE-specific kwargs so they don't reach ax.scatter
        for _k in ('perplexity', 'learning_rate', 'n_iter', 'random_state', 'init'):
            kwargs.pop(_k, None)
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError(
                "method='umap' requires the umap-learn package. "
                "Install it with: pip install umap-learn"
            )
        reducer = umap.UMAP(n_components=dim, **{k: v for k, v in kwargs.items()
                                                  if k in ('n_neighbors', 'min_dist',
                                                            'metric', 'random_state')})
        z_plot = reducer.fit_transform(z)
        for _k in ('n_neighbors', 'min_dist', 'metric', 'random_state'):
            kwargs.pop(_k, None)
    else:
        raise ValueError(
            f"method='{method}' is not recognised. "
            f"Choose from 'auto', 'none', 'pca', 'tsne', 'umap'."
        )

    _logger.debug(f"plot_embeddings: {method} → {z_plot.shape}")

    # --- Resolve colour ---
    if color is None:
        c_arr = None
        cmap = kwargs.pop('cmap', 'viridis')
        scatter_kwargs = {'c': None, 'cmap': cmap, **kwargs}
        legend_handles = None
    else:
        color = np.asarray(color)
        # Detect categorical vs continuous
        is_categorical = not np.issubdtype(color.dtype, np.floating)
        if is_categorical:
            unique_vals = np.unique(color)
            palette = plt.colormaps.get_cmap('tab10').resampled(len(unique_vals))
            c_arr = np.array([np.where(unique_vals == v)[0][0] for v in color])
            scatter_kwargs = {'c': c_arr, 'cmap': palette,
                              'vmin': -0.5, 'vmax': len(unique_vals) - 0.5, **kwargs}
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=palette(i / max(len(unique_vals) - 1, 1)),
                           label=str(v), markersize=8)
                for i, v in enumerate(unique_vals)
            ]
        else:
            scatter_kwargs = {'c': color, 'cmap': kwargs.pop('cmap', 'viridis'), **kwargs}
            legend_handles = None

    # --- Create axes ---
    created_fig = ax is None
    if ax is None:
        if dim == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(7, 6))

    # --- Plot ---
    if dim == 2:
        sc = ax.scatter(z_plot[:, 0], z_plot[:, 1], **scatter_kwargs)
        ax.set_xlabel(f'{method.upper()}-1')
        ax.set_ylabel(f'{method.upper()}-2')
    else:
        sc = ax.scatter(z_plot[:, 0], z_plot[:, 1], z_plot[:, 2], **scatter_kwargs)
        ax.set_xlabel(f'{method.upper()}-1')
        ax.set_ylabel(f'{method.upper()}-2')
        ax.set_zlabel(f'{method.upper()}-3')

    if legend_handles:
        ax.legend(handles=legend_handles, title='Class', loc='best')
    elif color is not None and not is_categorical:
        plt.colorbar(sc, ax=ax, label='Value')

    if title:
        ax.set_title(title)

    sns.despine(ax=ax)
    if created_fig:
        plt.tight_layout()

    return ax


def plot_cross_correlation(
    x,
    y,
    true_lag: int,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    xlim=None,
) -> plt.Axes:
    """Plot the cross-correlation between two signals against lag.

    Parameters
    ----------
    x : array-like
        First signal (iterable of samples; uses ``x[0]``).
    y : array-like
        Second signal (iterable of samples; uses ``y[0]``).
    true_lag : int
        Known true lag between the signals, plotted as a reference line.
    ax : plt.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.  Set to ``False`` when
        embedding this plot in a larger figure.  Defaults to ``True``.
    xlim : tuple of (float, float), optional
        X-axis limits ``(left, right)``.  When ``None`` the full lag range is
        shown (previously hard-coded as ``(-100, 100)``).

    Returns
    -------
    plt.Axes
        The axes containing the plot.
    """
    lags = np.arange(-len(x[0]) // 2 + 1, len(x[0]) // 2 + 1)
    corr = correlate(zscore(y[0]), zscore(x[0]), mode='same') / len(x[0])

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, corr)
    ax.axvline(true_lag, color='r', linestyle='-.', label=f'True Lag ({true_lag})')
    ax.axvline(lags[np.argmax(corr)], color='g', linestyle=':',
               label=f'Found Lag ({lags[np.argmax(corr)]})')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Cross-Correlation')
    ax.set_title('Linear Correlation vs Lag')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend()
    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()
    return ax

def analyze_mi_heatmap(
    results_df,
    mi_col: str = 'mi_mean',
    absolute_mi_threshold=0.2,
    contour_rise_fraction=0.1,
    radius_multiplier=1.2,
    true_lag=None,
    history_duration=None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """Topological analysis of a 2-D MI heatmap (lag × window_size).

    Finds the Causal Contour, the shortest bridge to the Significant MI
    Contour, and draws a Parsimonious Circle highlighting the optimal
    (lag, window) region.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns ``'lag'``, ``'window_size'``, and ``mi_col``.
        ``mode='lag'`` sweeps ``lag_range`` internally but does not itself
        sweep ``window_size`` (it is a processor parameter, not one
        ``mode='lag'``'s own ``sweep_grid`` forwards). Build this shape by
        calling ``mode='lag'`` once per ``window_size``, tagging each
        resulting ``result.dataframe`` with that value, and concatenating.
    mi_col : str, optional
        Name of the MI column in ``results_df``. Defaults to ``'mi_mean'``,
        the column produced by sweep-style aggregation. Pass ``'mi'`` (or
        another column name) if plotting a differently-shaped DataFrame.
    absolute_mi_threshold : float, optional
        Absolute MI value for the "significant" contour.  Defaults to 0.2.
    contour_rise_fraction : float, optional
        Heuristic fraction of the MI rise at lag=0 used to find the Causal
        Contour.  Defaults to 0.1.
    radius_multiplier : float, optional
        Scale factor for the Parsimonious Circle radius.  Defaults to 1.2.
    true_lag : float, optional
        Known true lag — drawn as a reference box when provided together with
        ``history_duration``.
    history_duration : float, optional
        Known true history duration — drawn as a reference box when provided
        together with ``true_lag``.
    ax : plt.Axes, optional
        Axes to draw on.  When ``None`` a new figure is created internally.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.  Set to ``False`` when
        embedding this plot in a larger figure.  Defaults to ``True``.

    Returns
    -------
    plt.Axes or None
        The axes containing the heatmap, or ``None`` when no significant
        contour is found and the function exits early.
    """
    # --- 1. Data Preparation ---
    heatmap_data = results_df.pivot(index='window_size', columns='lag', values=mi_col)
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
            _logger.info(
                "Causal Contour Analysis: MI at lag=0 rises at window_size=%s "
                "(implies lag_true + history_true ≈ %s)",
                causal_contour_c, causal_contour_c,
            )
    else:
        _logger.info("Causal Contour Analysis: lag=0 not found — skipping Causal Contour estimation.")

    # --- 3. Create the main figure for all analysis ---
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(11, 8))

    lag_edges = np.concatenate([lags - (lags[1] - lags[0])/2, [lags[-1] + (lags[1] - lags[0])/2]])
    window_edges = np.concatenate([windows - (windows[1] - windows[0])/2, [windows[-1] + (windows[1] - windows[0])/2]])

    mesh = ax.pcolormesh(lag_edges, window_edges, heatmap_data.values, cmap='viridis', shading='flat')
    plt.colorbar(mesh, ax=ax, label='Mutual Information')

    # --- 4. Significant Zone & Parsimony Analysis ---
    _logger.info("Parsimony Analysis: Significant MI threshold = %s", absolute_mi_threshold)
    cs = ax.contour(lags, windows, heatmap_data.values, levels=[absolute_mi_threshold],
                    colors='red', linewidths=2.5, linestyles='-')

    # A non-empty allsegs[0] list can still contain only degenerate (empty or
    # single-point) segments -- e.g. a threshold that only grazes the grid at
    # isolated points -- so check the picked segment's own size too, not just
    # whether the list itself is non-empty.
    significant_contour_points = (
        np.array(max(cs.allsegs[0], key=len)) if cs.allsegs[0] else np.empty((0, 2))
    )

    if significant_contour_points.size == 0:
        _logger.warning("No significant MI contour found at threshold %.3f — try a lower value.",
                        absolute_mi_threshold)
        ax.set_title('Parsimony-Informed Topological Analysis (No Significant Contour Found)')
        ax.set_xlabel('Lag (Timepoints)')
        ax.set_ylabel('Window Size (Timepoints)')
        if show:
            if created_fig:
                plt.tight_layout()
            plt.show()
        return ax

    midpoint, radius = None, None
    if causal_contour_c is not None:
        causal_lags = lags[
            (causal_contour_c - lags >= windows.min()) & (causal_contour_c - lags <= windows.max())
        ]
        causal_contour_line = np.array([[lg, causal_contour_c - lg] for lg in causal_lags])

        if causal_contour_line.size > 0:
            ax.plot(causal_contour_line[:, 0], causal_contour_line[:, 1],
                    color='cyan', linestyle='--', linewidth=3,
                    label=f'Causal Contour (C≈{causal_contour_c})')

            distances = cdist(significant_contour_points, causal_contour_line)
            min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
            point_on_mi_contour = significant_contour_points[min_dist_idx[0]]
            point_on_causal_contour = causal_contour_line[min_dist_idx[1]]
            midpoint = (point_on_mi_contour + point_on_causal_contour) / 2
            bridge_length = np.linalg.norm(point_on_mi_contour - point_on_causal_contour)
            radius = (bridge_length / 2) * radius_multiplier

            _logger.info(
                "Bridge: causal contour point %s → MI contour point %s  "
                "(length=%.2f, parsimonious center=(%.1f, %.1f), radius=%.2f)",
                point_on_causal_contour, point_on_mi_contour,
                bridge_length, midpoint[0], midpoint[1], radius,
            )

            ax.plot(
                [point_on_causal_contour[0], point_on_mi_contour[0]],
                [point_on_causal_contour[1], point_on_mi_contour[1]],
                'orange', linewidth=2, linestyle='-', alpha=0.7,
            )
            circle = patches.Circle(midpoint, radius, linewidth=2.5, edgecolor='white',
                                    facecolor='none', linestyle=':', label='Parsimonious Region')
            ax.add_patch(circle)
            ax.plot(midpoint[0], midpoint[1], 'w+', markersize=15, mew=3,
                    label='Parsimonious Center')

    # --- 5. Mark True Parameter Box ---
    if true_lag is not None and history_duration is not None:
        lag_step = lags[1] - lags[0] if len(lags) > 1 else 1
        window_step = windows[1] - windows[0] if len(windows) > 1 else 1
        true_rect = patches.Rectangle(
            (true_lag - lag_step/2, history_duration - window_step/2),
            lag_step, window_step,
            linewidth=3, edgecolor='lime', facecolor='none', linestyle='-',
        )
        ax.add_patch(true_rect)
        _logger.info("True parameters: lag=%s, history=%s", true_lag, history_duration)

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
    ax.set_xlim(lags.min(), lags.max())
    ax.set_ylim(windows.min(), windows.max())
    if len(lags) < 25:
        ax.set_xticks(lags)
    if len(windows) < 25:
        ax.set_yticks(windows)

    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()
    return ax