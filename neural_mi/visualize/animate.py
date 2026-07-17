# neural_mi/visualize/animate.py
"""Training-history animation utilities.

Produces animated GIFs (or MP4s) that show how MI, spectral metrics, and
learned embeddings evolve over training epochs.

Typical usage::

    result = nmi.run(x, mode='dimensionality', ...)
    result.animate(output_path='training.gif')

    # With per-split labels for embedding colouring:
    result.animate(
        output_path='training.gif',
        embedding_labels={'stimulus': stim_labels, 'position': pos_values},
    )
"""
import warnings
from typing import Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim

from neural_mi.logger import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_training(
    result,
    panels: Optional[List[str]] = None,
    fps: int = 10,
    output_path: Optional[str] = None,
    show: bool = True,
    n_components: int = 2,
    reduction: str = 'pca',
    embedding_labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    **kwargs,
) -> manim.FuncAnimation:
    """Animate the training history as a GIF or MP4.

    Creates a frame-by-frame animation of training history stored in
    ``result.details``.  Panels are auto-detected from the available data or
    specified explicitly.

    Parameters
    ----------
    result : Results
        A ``Results`` object containing training history.  The following keys
        in ``result.details`` drive the panels:

        - ``'test_mi_history'`` — always required (drives the MI panel).
        - ``'train_mi_history'`` — overlaid on the MI panel when present.
        - ``'spectral_metrics_history'`` — drives the spectral-metrics panel.
        - ``'embedding_history_x'`` / ``'embedding_history_y'`` — drive the
          embedding panel (populated when ``track_embeddings != False``).

    panels : list of str, optional
        Which panels to include. When ``None`` (default) panels are
        auto-detected from available data. Valid values:

        - ``'mi'`` — MI vs epoch line plot (test MI + optional train MI).
        - ``'spectral_metrics'`` — participation ratio vs epoch.
        - ``'spectrum'`` — bar chart of singular values at each epoch
          (requires ``track_spectral_history=True``).
        - ``'embeddings'`` — 2-D or 3-D scatter of learned embeddings.

    fps : int, optional
        Frames per second.  Defaults to 10.
    output_path : str, optional
        Path to save the animation.  The extension determines the format:
        ``.gif`` uses ``PillowWriter``; ``.mp4`` uses ``FFMpegWriter``.
        When ``None`` the animation is returned without saving.
    show : bool, optional
        Whether to call ``plt.show()`` after creating the animation.
        Defaults to ``True``.
    n_components : {2, 3}, optional
        Dimensionality for the embedding scatter plots.  Defaults to 2.
    reduction : {'pca', 'umap', 'none'}, optional
        Dimensionality-reduction method for embedding panels.  ``'none'``
        plots the first ``n_components`` dimensions directly.  Defaults to
        ``'pca'``.
    embedding_labels : array-like or dict, optional
        Labels for colouring embedding scatter points.

        - ``None`` — uniform colour.
        - 1-D array — single label set; one embedding subplot.
        - dict mapping name → array — multiple label sets; one subplot
          per entry.

        Continuous arrays produce a gradient colormap; integer / string
        arrays produce a discrete palette with a legend.

    **kwargs
        ``figsize`` : tuple, forwarded to ``plt.figure``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.  In Jupyter notebooks use
        ``HTML(anim.to_jshtml())`` to display inline.

    Examples
    --------
    >>> result = nmi.run(x, mode='dimensionality', model=nmi.Model(...), training=nmi.Training(...))
    >>> anim = result.animate(output_path='training.gif', fps=8)

    >>> # With embedding labels
    >>> anim = result.animate(
    ...     embedding_labels={'class': class_labels},
    ...     reduction='umap',
    ... )
    """
    details = result.details

    # ---- collect history arrays ----
    test_history = list(details.get('test_mi_history', []))
    train_history = list(details.get('train_mi_history', []))
    spectral_history = list(details.get('spectral_metrics_history', []))
    embed_history_x = list(details.get('embedding_history_x', []))
    embed_history_y = list(details.get('embedding_history_y', []))
    best_epoch = details.get('best_epoch')

    n_frames = len(test_history)
    if n_frames == 0:
        raise ValueError(
            "result.details does not contain 'test_mi_history'. "
            "Cannot create animation. Ensure the result was produced by "
            "a training run (e.g. mode='estimate' or mode='dimensionality')."
        )

    # ---- resolve panels ----
    if panels is None:
        panels = _auto_panels(details)

    # ---- normalize embedding_labels to dict ----
    if embedding_labels is not None:
        if not isinstance(embedding_labels, dict):
            embedding_labels = {'': np.asarray(embedding_labels)}
        else:
            embedding_labels = {k: np.asarray(v) for k, v in embedding_labels.items()}

    # ---- pre-fit reducer for consistent embedding coordinates ----
    _reduced_x: List[np.ndarray] = []
    _reduced_y: List[np.ndarray] = []
    method_label = reduction.upper() if reduction != 'none' else ''

    if 'embeddings' in panels:
        if embed_history_x:
            _, _reduced_x = _fit_reducer(embed_history_x, n_components, reduction)
        if embed_history_y:
            _, _reduced_y = _fit_reducer(embed_history_y, n_components, reduction)
        if not _reduced_x and not _reduced_y:
            warnings.warn(
                "panels includes 'embeddings' but result.details does not contain "
                "'embedding_history_x' / 'embedding_history_y'. "
                "Set track_embeddings=512 (or any positive value) in base_params to "
                "enable per-epoch embedding tracking. Removing 'embeddings' panel.",
                UserWarning,
                stacklevel=2,
            )
            panels = [p for p in panels if p != 'embeddings']

    # ---- determine column layout ----
    n_embed_label_keys = len(embedding_labels) if embedding_labels else 1
    col_spec: List[str] = []  # e.g. ['mi', 'spectral_metrics', 'embed', 'embed']
    if 'mi' in panels:
        col_spec.append('mi')
    if 'spectral_metrics' in panels and spectral_history:
        col_spec.append('spectral_metrics')
    if 'spectrum' in panels and spectral_history and 'spectrum' in spectral_history[0]:
        col_spec.append('spectrum')
    if 'embeddings' in panels:
        for _ in range(n_embed_label_keys):
            col_spec.append('embed')

    if not col_spec:
        raise ValueError("No panels could be created. Check panels= and result.details content.")

    ncols = len(col_spec)
    units = result.params.get('output_units', 'bits')
    figsize = kwargs.pop('figsize', (max(5 * ncols, 6), 4))

    # ---- build figure with per-column projections ----
    fig = plt.figure(figsize=figsize)
    axes: List[plt.Axes] = []
    for i, ptype in enumerate(col_spec):
        proj = '3d' if (ptype == 'embed' and n_components == 3) else None
        ax = fig.add_subplot(1, ncols, i + 1, projection=proj)
        axes.append(ax)

    col_ax = {ptype: ax for ptype, ax in zip(col_spec, axes)}
    embed_axes = [ax for ptype, ax in zip(col_spec, axes) if ptype == 'embed']
    embed_label_keys = list(embedding_labels.keys()) if embedding_labels else [None] * len(embed_axes)

    # ---- MI panel ----
    line_test = line_train = dot_test = None
    if 'mi' in col_ax:
        ax_mi = col_ax['mi']
        (line_test,) = ax_mi.plot([], [], color='steelblue', linewidth=1.5, label='Test MI')
        dot_test = ax_mi.scatter([], [], color='steelblue', zorder=5, s=40)
        if train_history:
            (line_train,) = ax_mi.plot([], [], color='darkorange', linewidth=1.5,
                                        linestyle='--', alpha=0.8, label='Train MI')
        if best_epoch is not None:
            ax_mi.axvline(best_epoch, color='tomato', linestyle='--',
                          linewidth=1.0, alpha=0.5, label=f'Best ({best_epoch})')
        ax_mi.set_xlim(-0.5, n_frames - 0.5)
        _mi_vals = [v for v in test_history + train_history if not np.isnan(v)]
        _mi_lo = min(_mi_vals) if _mi_vals else 0.0
        _mi_hi = max(_mi_vals) if _mi_vals else 1.0
        _mi_pad = max(abs(_mi_hi - _mi_lo) * 0.12, 0.02)
        ax_mi.set_ylim(_mi_lo - _mi_pad, _mi_hi + _mi_pad)
        ax_mi.set_xlabel('Epoch')
        ax_mi.set_ylabel(f'MI ({units})')
        ax_mi.set_title('Training History')
        ax_mi.legend(fontsize=8)
        ax_mi.grid(True, alpha=0.3)

    # ---- spectral-metrics panel ----
    line_pr = dot_pr = None
    pr_vals: List[float] = []
    if 'spectral_metrics' in col_ax and spectral_history:
        ax_sp = col_ax['spectral_metrics']
        pr_vals = [float(m.get('pr_singular', np.nan)) for m in spectral_history]
        (line_pr,) = ax_sp.plot([], [], color='mediumseagreen', linewidth=1.5,
                                 label='Participation Ratio (Singular)')
        dot_pr = ax_sp.scatter([], [], color='mediumseagreen', zorder=5, s=40)
        ax_sp.set_xlim(-0.5, len(pr_vals) - 0.5)
        _valid_pr = [v for v in pr_vals if not np.isnan(v)]
        _pr_hi = max(_valid_pr) if _valid_pr else 1.0
        ax_sp.set_ylim(0, _pr_hi * 1.15)
        ax_sp.set_xlabel('Epoch')
        ax_sp.set_ylabel('Participation Ratio (pr_singular)')
        ax_sp.set_title('Spectral Dimensionality')
        ax_sp.grid(True, alpha=0.3)

    # ---- spectrum panel ----
    bar_container = None
    if 'spectrum' in col_ax and spectral_history and 'spectrum' in spectral_history[0]:
        ax_spec = col_ax['spectrum']
        _first_spec = np.asarray(spectral_history[0]['spectrum'])
        bar_container = ax_spec.bar(range(len(_first_spec)), _first_spec,
                                     color='mediumpurple', alpha=0.8)
        _spec_max = max(
            float(np.asarray(m['spectrum']).max())
            for m in spectral_history if 'spectrum' in m
        )
        ax_spec.set_ylim(0, _spec_max * 1.12)
        ax_spec.set_xlabel('Singular value index')
        ax_spec.set_ylabel('Magnitude')
        ax_spec.set_title('Spectrum (epoch 0)')
        ax_spec.grid(True, alpha=0.3, axis='y')

    # ---- embedding panels ----
    # Fix axis limits across all frames for stable animation.
    embed_scatters: List = []
    if embed_axes and (_reduced_x or _reduced_y):
        _src = _reduced_x if _reduced_x else _reduced_y
        _all_flat = np.concatenate(_src, axis=0)  # (n_frames * n_tracked, n_components)
        _pad_frac = 0.06

        for i_ax, (ax_emb, lk) in enumerate(zip(embed_axes, embed_label_keys)):
            color_arr = embedding_labels[lk] if (embedding_labels and lk in embedding_labels) else None
            c, cmap, vmin, vmax = _resolve_scatter_color(color_arr)

            z0 = _src[0]
            if n_components == 2:
                sc = ax_emb.scatter(z0[:, 0], z0[:, 1],
                                    c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                                    s=20, alpha=0.7)
                # Set fixed limits
                for dim_i, setter in enumerate([ax_emb.set_xlim, ax_emb.set_ylim]):
                    lo, hi = _all_flat[:, dim_i].min(), _all_flat[:, dim_i].max()
                    pad = (hi - lo) * _pad_frac or 0.1
                    setter(lo - pad, hi + pad)
                ax_emb.set_xlabel(f'{method_label}-1' if method_label else 'Dim 1')
                ax_emb.set_ylabel(f'{method_label}-2' if method_label else 'Dim 2')
            else:  # 3D
                sc = ax_emb.scatter(z0[:, 0], z0[:, 1], z0[:, 2],
                                    c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                                    s=20, alpha=0.7)
                for dim_i, setter in enumerate([ax_emb.set_xlim, ax_emb.set_ylim, ax_emb.set_zlim]):
                    lo, hi = _all_flat[:, dim_i].min(), _all_flat[:, dim_i].max()
                    pad = (hi - lo) * _pad_frac or 0.1
                    setter(lo - pad, hi + pad)
                ax_emb.set_xlabel(f'{method_label}-1' if method_label else 'Dim 1')
                ax_emb.set_ylabel(f'{method_label}-2' if method_label else 'Dim 2')
                ax_emb.set_zlabel(f'{method_label}-3' if method_label else 'Dim 3')

            label_suffix = f' ({lk})' if lk else ''
            ax_emb.set_title(f'Embeddings X{label_suffix}')

            # Add colorbar or legend
            if c is not None and np.issubdtype(np.asarray(c).dtype, np.floating):
                fig.colorbar(sc, ax=ax_emb, fraction=0.04, pad=0.04)
            elif color_arr is not None:
                _add_categorical_legend(ax_emb, sc, color_arr)

            embed_scatters.append(sc)

    plt.tight_layout()

    # ---- update function ----
    def update(frame: int):
        artists = []

        # MI panel
        if line_test is not None:
            epochs_so_far = list(range(frame + 1))
            line_test.set_data(epochs_so_far, test_history[:frame + 1])
            artists.append(line_test)
            if dot_test is not None:
                dot_test.set_offsets([[frame, test_history[frame]]])
                artists.append(dot_test)
        if line_train is not None and frame < len(train_history):
            line_train.set_data(list(range(frame + 1)), train_history[:frame + 1])
            artists.append(line_train)

        # Spectral metrics panel
        if line_pr is not None and frame < len(pr_vals):
            line_pr.set_data(list(range(frame + 1)), pr_vals[:frame + 1])
            artists.append(line_pr)
            if dot_pr is not None:
                dot_pr.set_offsets([[frame, pr_vals[frame]]])
                artists.append(dot_pr)

        # Spectrum panel
        if bar_container is not None and frame < len(spectral_history):
            spec = spectral_history[frame].get('spectrum')
            if spec is not None:
                spec = np.asarray(spec)
                for rect, val in zip(bar_container, spec):
                    rect.set_height(val)
                col_ax['spectrum'].set_title(f'Spectrum (epoch {frame})')
            artists.extend(list(bar_container))

        # Embedding panels
        _src = _reduced_x if _reduced_x else _reduced_y
        if embed_scatters and _src and frame < len(_src):
            z = _src[frame]
            for sc in embed_scatters:
                if n_components == 2:
                    sc.set_offsets(z[:, :2])
                else:
                    sc._offsets3d = (z[:, 0], z[:, 1], z[:, 2])
                artists.append(sc)
            for ax_emb, lk in zip(embed_axes, embed_label_keys):
                lsuffix = f' ({lk})' if lk else ''
                ax_emb.set_title(f'Embeddings X{lsuffix} (epoch {frame})')

        return artists

    anim = manim.FuncAnimation(
        fig, update, frames=n_frames, interval=max(1, 1000 // fps), blit=False,
    )

    if output_path is not None:
        ext = output_path.rsplit('.', 1)[-1].lower() if '.' in output_path else 'gif'
        if ext == 'gif':
            writer = manim.PillowWriter(fps=fps)
        else:
            try:
                writer = manim.FFMpegWriter(fps=fps)
            except Exception:
                warnings.warn(
                    "FFMpeg not found; falling back to PillowWriter (GIF). "
                    "Install ffmpeg to export MP4.",
                    UserWarning,
                    stacklevel=2,
                )
                writer = manim.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        logger.info(f"Animation saved to '{output_path}'.")

    if show:
        plt.show()

    return anim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_panels(details: dict) -> List[str]:
    """Infer which panels to include from result.details keys."""
    panels = ['mi']
    spectral = details.get('spectral_metrics_history', [])
    if spectral:
        first = spectral[0]
        if 'spectrum' in first:
            panels.append('spectrum')
        # Always show PR line if it exists (even alongside spectrum)
        if 'pr_singular' in first or 'pr_eig' in first:
            panels.append('spectral_metrics')
    if details.get('embedding_history_x') or details.get('embedding_history_y'):
        panels.append('embeddings')
    return panels


def _fit_reducer(
    embed_history: List[np.ndarray],
    n_components: int,
    reduction: str,
):
    """Fit a reducer on all frames' embeddings and return per-frame projections.

    Fitting on the concatenation of all frames gives consistent coordinates
    across the animation.

    Returns
    -------
    reducer : fitted reducer object or None
    reduced_per_frame : list of (n_tracked, n_components) arrays
    """
    if not embed_history:
        return None, []

    embed_dim = embed_history[0].shape[1]
    if embed_dim <= n_components or reduction == 'none':
        return None, [z[:, :n_components] for z in embed_history]

    all_embeds = np.concatenate(embed_history, axis=0)  # (n_frames * n_tracked, embed_dim)

    if reduction == 'pca':
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "PCA requires scikit-learn: pip install scikit-learn"
            )
        reducer = PCA(n_components=n_components)
        all_reduced = reducer.fit_transform(all_embeds)
    elif reduction == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP requires umap-learn: pip install umap-learn"
            )
        reducer = umap.UMAP(n_components=n_components)
        all_reduced = reducer.fit_transform(all_embeds)
    else:
        raise ValueError(
            f"reduction='{reduction}' not recognised. Choose 'pca', 'umap', or 'none'."
        )

    n_tracked = embed_history[0].shape[0]
    reduced_per_frame = [
        all_reduced[i * n_tracked: (i + 1) * n_tracked]
        for i in range(len(embed_history))
    ]
    return reducer, reduced_per_frame


def _resolve_scatter_color(color_arr):
    """Return (c, cmap, vmin, vmax) for ax.scatter."""
    if color_arr is None:
        return None, 'viridis', None, None
    color_arr = np.asarray(color_arr)
    is_categorical = not np.issubdtype(color_arr.dtype, np.floating)
    if is_categorical:
        unique_vals = np.unique(color_arr)
        c = np.array([int(np.where(unique_vals == v)[0][0]) for v in color_arr],
                     dtype=float)
        n_cats = len(unique_vals)
        cmap = plt.colormaps.get_cmap('tab10').resampled(n_cats)
        return c, cmap, -0.5, n_cats - 0.5
    return color_arr.astype(float), 'viridis', None, None


def _add_categorical_legend(ax, sc, color_arr):
    """Add a discrete legend for categorical colour arrays."""
    color_arr = np.asarray(color_arr)
    unique_vals = np.unique(color_arr)
    n_cats = len(unique_vals)
    cmap = plt.colormaps.get_cmap('tab10').resampled(n_cats)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(i / max(n_cats - 1, 1)),
                   label=str(v), markersize=7)
        for i, v in enumerate(unique_vals)
    ]
    ax.legend(handles=handles, fontsize=7, loc='best')
