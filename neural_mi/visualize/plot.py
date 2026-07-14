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


def plot_dimensionality_curve(
    summary_df: pd.DataFrame,
    sweep_var: Optional[str] = None,
    units: str = 'bits',
    axes=None,
    show: bool = True,
    **kwargs,
) -> plt.Axes:
    """Two-panel plot for dimensionality analysis: MI (top) and Participation Ratio (bottom).

    The PR panel plots ``pr_singular`` (the singular-spectrum variant). The
    eigenvalue/covariance-spectrum variant (``pr_eig``) is also present in
    ``result.dataframe`` but is not plotted by this function.

    When a sweep variable is present and participation ratio data is available,
    this function creates a two-panel figure so both metrics are visible side by
    side.  When no sweep variable is present (scalar result), the values are
    displayed as annotated single-point plots.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregated dimensionality results.  Expected columns depend on context:

        - With sweep: ``sweep_var``, ``mi_mean``, ``mi_std``,
          ``pr_singular_mean``, ``pr_singular_std``.
        - Without sweep (single row): same column names, single row.
    sweep_var : str, optional
        Name of the sweep variable column (e.g. ``'embedding_dim'``).  When
        ``None`` or absent from the DataFrame, a single-point display is used.
    units : str, optional
        MI units for axis labels (e.g. ``'bits'``).  Defaults to ``'bits'``.
    axes : None, plt.Axes, or (plt.Axes, plt.Axes), optional
        Axes to plot on.

        - ``None`` (default) — create a new figure with one or two subplots.
        - Single ``plt.Axes`` — use for the MI panel only; PR panel is skipped.
        - 2-tuple of ``plt.Axes`` — use as ``(ax_mi, ax_pr)``.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.  Set to ``False`` when
        embedding this plot in a larger figure.  Defaults to ``True``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.plot`` for the MI curve.

    Returns
    -------
    plt.Axes
        The MI (top) axes.  When a two-panel figure is created internally, the
        PR axes is accessible via ``ax_mi.figure.axes[1]``.
    """
    has_pr = 'pr_singular_mean' in summary_df.columns
    has_sweep = (
        sweep_var is not None
        and sweep_var in summary_df.columns
        and len(summary_df) > 1
    )
    figsize = kwargs.pop('figsize', (8, 8 if has_pr else 5))

    # --- Resolve axes ---
    created_fig = axes is None
    if axes is None:
        if has_pr:
            fig, (ax_mi, ax_pr) = plt.subplots(
                2, 1, figsize=figsize,
                gridspec_kw={'hspace': 0.35},
                sharex=True if has_sweep else False,
            )
        else:
            fig, ax_mi = plt.subplots(1, 1, figsize=figsize)
            ax_pr = None
    elif isinstance(axes, (list, tuple)) and len(axes) == 2:
        ax_mi, ax_pr = axes
    else:
        ax_mi = axes
        ax_pr = None
        has_pr = False

    # --- MI panel ---
    if has_sweep:
        ax_mi.plot(summary_df[sweep_var], summary_df['mi_mean'],
                   'o-', label='Mean MI', **kwargs)
        if 'mi_std' in summary_df.columns:
            ax_mi.fill_between(
                summary_df[sweep_var],
                summary_df['mi_mean'] - summary_df['mi_std'],
                summary_df['mi_mean'] + summary_df['mi_std'],
                alpha=0.2, label='±1 Std Dev',
            )
        if pd.api.types.is_numeric_dtype(summary_df[sweep_var]) and all(
            summary_df[sweep_var] == np.floor(summary_df[sweep_var])
        ):
            ax_mi.xaxis.set_major_locator(MaxNLocator(integer=True))
        if not has_pr:
            ax_mi.set_xlabel(sweep_var.replace('_', ' ').title(), fontsize=11)
    else:
        mi_mean = float(summary_df['mi_mean'].iloc[0])
        mi_std = float(summary_df['mi_std'].iloc[0]) if 'mi_std' in summary_df.columns else 0.0
        ax_mi.errorbar([0], [mi_mean], yerr=[mi_std], fmt='o', capsize=6,
                       color='steelblue', markersize=8)
        ax_mi.annotate(
            f'{mi_mean:.3f} ± {mi_std:.3f} {units}',
            xy=(0, mi_mean), xytext=(0.05, 0.80),
            textcoords='axes fraction', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        )
        ax_mi.set_xticks([])

    ax_mi.set_ylabel(f'MI ({units})', fontsize=11)
    ax_mi.set_title('Dimensionality Analysis — MI', fontsize=12)
    if has_sweep:
        ax_mi.legend(fontsize=9)
    ax_mi.grid(True, linestyle=':')
    sns.despine(ax=ax_mi)

    # --- Participation Ratio panel ---
    if has_pr and ax_pr is not None:
        if has_sweep:
            ax_pr.plot(summary_df[sweep_var], summary_df['pr_singular_mean'],
                       's-', color='teal', label='Mean PR')
            if 'pr_singular_std' in summary_df.columns:
                ax_pr.fill_between(
                    summary_df[sweep_var],
                    summary_df['pr_singular_mean'] - summary_df['pr_singular_std'],
                    summary_df['pr_singular_mean'] + summary_df['pr_singular_std'],
                    alpha=0.2, color='teal', label='±1 Std Dev',
                )
            if pd.api.types.is_numeric_dtype(summary_df[sweep_var]) and all(
                summary_df[sweep_var] == np.floor(summary_df[sweep_var])
            ):
                ax_pr.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_pr.set_xlabel(sweep_var.replace('_', ' ').title(), fontsize=11)
        else:
            pr_mean = float(summary_df['pr_singular_mean'].iloc[0])
            pr_std = (
                float(summary_df['pr_singular_std'].iloc[0])
                if 'pr_singular_std' in summary_df.columns else 0.0
            )
            ax_pr.errorbar([0], [pr_mean], yerr=[pr_std], fmt='s', capsize=6,
                           color='teal', markersize=8)
            ax_pr.annotate(
                f'{pr_mean:.2f} ± {pr_std:.2f}',
                xy=(0, pr_mean), xytext=(0.05, 0.80),
                textcoords='axes fraction', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            )
            ax_pr.set_xticks([])

        ax_pr.set_ylabel('Participation Ratio (Singular)', fontsize=11)
        ax_pr.set_title('Participation Ratio — pr_singular (Effective Dimensions)', fontsize=12)
        if has_sweep:
            ax_pr.legend(fontsize=9)
        ax_pr.grid(True, linestyle=':')
        sns.despine(ax=ax_pr)

    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()
    return ax_mi


def plot_noise_ladder(ladder_df: pd.DataFrame, ax: Optional[plt.Axes] = None,
                      overlay_mi: bool = False, show: bool = True) -> plt.Axes:
    """Plots the ceiling-escape noise-injection ladder (both PR variants vs. log(sigma_add)).

    Plots the estimated dimension ``d_hat`` from both ``pr_eig`` and
    ``pr_singular`` against ``log(sigma_add)``, with the detached band shaded.
    Not the per-``k_z`` MI curve.

    Parameters
    ----------
    ladder_df : pd.DataFrame
        The per-rung table from ``result.details['sigma_add_ladder']``. Expected
        columns: ``sigma_add``, ``pr_eig_mean``, ``pr_singular_mean``, ``regime``,
        and (if ``overlay_mi=True``) ``mi_mean``.
    ax : plt.Axes, optional
        Axes to plot on. Creates a new figure if ``None``.
    overlay_mi : bool, optional
        If True, overlays MI vs sigma_add on a secondary y-axis. Defaults to False.
    show : bool, optional
        Whether to call ``plt.show()``. Defaults to True.

    Returns
    -------
    plt.Axes
    """
    df = ladder_df.sort_values('sigma_add').reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Shade the contiguous detached band.
    detached_mask = (df['regime'] == 'detached').to_numpy()
    if detached_mask.any():
        x = df['sigma_add'].to_numpy()
        idx = np.where(detached_mask)[0]
        lo, hi = x[idx.min()], x[idx.max()]
        ax.axvspan(lo, hi, color='mediumseagreen', alpha=0.15, label='Detached band', zorder=0)

    if 'pr_eig_mean' in df.columns:
        ax.plot(df['sigma_add'], df['pr_eig_mean'], 'o-', color='tab:blue', label=r'$\hat{d}$ (pr_eig)')
        if 'pr_eig_std' in df.columns:
            ax.fill_between(df['sigma_add'], df['pr_eig_mean'] - df['pr_eig_std'],
                            df['pr_eig_mean'] + df['pr_eig_std'], alpha=0.15, color='tab:blue')
    if 'pr_singular_mean' in df.columns:
        ax.plot(df['sigma_add'], df['pr_singular_mean'], 's-', color='tab:orange', label=r'$\hat{d}$ (pr_singular)')
        if 'pr_singular_std' in df.columns:
            ax.fill_between(df['sigma_add'], df['pr_singular_mean'] - df['pr_singular_std'],
                            df['pr_singular_mean'] + df['pr_singular_std'], alpha=0.15, color='tab:orange')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma_{add}$ (log scale, per-channel std units)')
    ax.set_ylabel(r'Estimated dimension $\hat{d}$')
    ax.set_title('Noise-Injection Ladder')
    ax.grid(True, linestyle=':')
    sns.despine(ax=ax)

    if overlay_mi and 'mi_mean' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df['sigma_add'], df['mi_mean'], 'd--', color='gray', alpha=0.7, label='MI')
        ax2.set_ylabel('MI')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    else:
        ax.legend(fontsize=9)

    if show:
        plt.show()
    return ax


def plot_bias_correction_fit(raw_results_df: pd.DataFrame, corrected_result: Dict[str, Any],
                             ax: Optional[plt.Axes] = None, units: str = 'bits',
                             show: bool = True, **kwargs):
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
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.stripplot(x='gamma', y='train_mi', data=raw_results_df, ax=ax, color='gray', alpha=0.5)
    agg = raw_results_df.groupby('gamma')['train_mi'].mean().reset_index()
    ax.plot(agg['gamma'] - 1, agg['train_mi'], 'o-', color='black', label='Mean MI per Gamma')

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
        resulting ``result.dataframe`` with that value, and concatenating —
        see Tutorial 6 for a worked example.
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