# neural_mi/results.py
"""Defines the `Results` class for storing and interacting with analysis outcomes.

This module provides a standardized data structure for holding the results of
different analysis modes from the `run` function. The `Results` class acts as
a container for MI estimates, dataframes, and detailed metadata, and also
provides a convenient `.plot()` method for visualizing the results.
"""
import os
import datetime
import pickle
import json
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from neural_mi.logger import logger

# Module-level constant: columns produced by the MI estimator that are NOT
# sweep/hyperparameter variables. Used when inferring the x-axis of a sweep plot.
_RESULT_COLS: frozenset = frozenset({
    'mi_mean', 'mi_std', 'test_mi', 'train_mi', 'mi_corrected',
    'mi_error', 'mi_error_pred', 'slope', 'run_id', 'is_reliable', 'gammas_used',
    'n_windows', 'lag',
    # Dimensionality-specific columns
    'participation_ratio', 'participation_ratio_mean', 'participation_ratio_std',
    'participation_ratio_singular', 'split_id',
})


@dataclass
class Results:
    """A data class to store and interact with analysis results.

    This class provides a structured way to access the outputs of the `run`
    function. Depending on the analysis `mode`, different attributes will be
    populated.

    Attributes
    ----------
    mode : str
        The analysis mode that was run (e.g., 'estimate', 'sweep').
    params : Dict[str, Any]
        A dictionary of the parameters used for the analysis run.
    mi_estimate : float, optional
        The final point estimate of mutual information. Populated in 'estimate'
        and 'rigorous' modes.
    dataframe : pd.DataFrame, optional
        A DataFrame containing detailed results. Populated in 'sweep',
        'dimensionality', 'rigorous', 'lag', 'precision', and 'pairwise' modes.
    details : Dict[str, Any]
        A dictionary containing additional metadata or detailed results, such
        as raw run data or estimated latent dimensions.

    Methods
    -------
    summary()
        Print a human-readable summary to stdout.
    plot(ax=None, **kwargs)
        Generate a mode-appropriate figure.
    animate(**kwargs)
        Animate the training history as a GIF or MP4.
    compare(results_list, labels=None, ax=None, **kwargs)
        Static method; overlay multiple Results on a shared axis.
    to_dict()
        Return a fully serialisable dict (arrays as nested lists).
    to_json(path=None)
        Export to a JSON file; arrays serialised as nested lists.
    """
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)
    mi_estimate: Optional[float] = None
    dataframe: Optional[pd.DataFrame] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Provides a concise representation of the Results object."""
        rep = f"Results(mode='{self.mode}'"
        if self.mi_estimate is not None: rep += f", mi_estimate={self.mi_estimate:.4f}"
        if self.dataframe is not None: rep += f", dataframe_shape={self.dataframe.shape}"
        if self.details: rep += f", details_keys={list(self.details.keys())}"
        return rep + ")"

    def summary(self) -> None:
        """Print a human-readable summary of the analysis results to stdout.

        Prints the analysis mode, MI estimate (if available), confidence
        interval (for ``mode='rigorous'``), reliability flag (for
        ``mode='rigorous'``), mode-specific component values, and DataFrame
        shape (if a DataFrame is present).
        """
        SEP = "─" * 50
        units = self.params.get('output_units', 'bits')
        print(SEP)
        print(f"  NeuralMI Results  |  mode = '{self.mode}'")
        print(SEP)

        if self.mode == 'precision':
            baseline_mi   = self.details.get('baseline_mi')
            precision_tau = self.details.get('precision_tau')
            threshold_val = self.details.get('threshold_value')
            if baseline_mi is not None:
                print(f"  Baseline MI       : {baseline_mi:.4f} {units}")
            if precision_tau is not None:
                print(f"  Precision τ       : {precision_tau:.4g}")
            if threshold_val is not None:
                print(f"  Threshold MI      : {threshold_val:.4f} {units}")

        elif self.mode == 'pairwise':
            mi_matrix = self.details.get('mi_matrix')
            df = self.dataframe
            if mi_matrix is not None:
                import numpy as _np
                _finite = mi_matrix[_np.isfinite(mi_matrix)]
                n_ch = self.details.get('n_channels', '?')
                n_pairs = len(df) if df is not None else '?'
                shape_str = (f"{n_ch[0]} × {n_ch[1]}" if isinstance(n_ch, tuple) else f"{n_ch} × {n_ch}")
                print(f"  MI matrix         : {shape_str} channels  ({n_pairs} pairs)")
                if len(_finite) > 0:
                    print(f"  MI range          : {_finite.min():.4f} – {_finite.max():.4f} {units}")

        elif self.mode == 'conditional':
            mi_xz_y = self.details.get('mi_xz_y')
            mi_z_y  = self.details.get('mi_z_y')
            cmi     = self.details.get('cmi_estimate')
            if cmi is not None:
                print(f"  CMI I(X;Y|Z)      : {cmi:.4f} {units}")
            if mi_xz_y is not None:
                print(f"  I(XZ;Y)           : {mi_xz_y:.4f} {units}")
            if mi_z_y is not None:
                print(f"  I(Z;Y)            : {mi_z_y:.4f} {units}")

        elif self.mode == 'transfer':
            te_xy = self.details.get('te_xy')
            te_yx = self.details.get('te_yx')
            di    = self.details.get('directionality_index')
            if te_xy is not None:
                print(f"  TE(X→Y)           : {te_xy:.4f} {units}")
            if te_yx is not None:
                print(f"  TE(Y→X)           : {te_yx:.4f} {units}")
            if di is not None:
                print(f"  Directionality    : {di:.4f}  (+1 = X→Y, -1 = Y→X, 0 = symmetric)")

        else:
            # Generic display for all other modes (estimate, sweep, rigorous, lag, dimensionality)
            if self.mi_estimate is not None:
                print(f"  MI estimate : {self.mi_estimate:.4f} {units}")
            else:
                print("  MI estimate : (none — see result.dataframe or result.details)")
            if self.mode == 'rigorous':
                mi_err = self.details.get('mi_error')
                mi_err_pred = self.details.get('mi_error_pred')
                is_reliable = self.details.get('is_reliable')
                fit_quality_warning = self.details.get('fit_quality_warning')
                leverage_warning = self.details.get('leverage_warning')
                r_squared = self.details.get('r_squared')
                max_abs_residual = self.details.get('max_abs_residual')
                loo_shift = self.details.get('loo_intercept_shift')
                if mi_err is not None:
                    print(f"  CI half-width : {mi_err:.4f} {units}  [confidence interval on the fitted mean]")
                if mi_err_pred is not None:
                    print(f"  PI half-width : {mi_err_pred:.4f} {units}  [prediction interval, more conservative]")
                if is_reliable is False:
                    print("  ⚠  is_reliable = False — extrapolation is unreliable.")
                    _reasons = []
                    if fit_quality_warning:
                        _r2_str = f", R²={r_squared:.3f}" if r_squared is not None and r_squared == r_squared else ""
                        _res_str = f", max|residual|={max_abs_residual:.2f}" if max_abs_residual is not None and max_abs_residual == max_abs_residual else ""
                        _reasons.append(f"fit quality (fit_quality_warning=True{_r2_str}{_res_str})")
                    if leverage_warning:
                        _loo_str = f"={loo_shift:.3f}" if loo_shift is not None and loo_shift == loo_shift else ""
                        _reasons.append(f"gamma=1 leverage (leverage_warning=True, LOO shift{_loo_str})")
                    if _reasons:
                        print(f"     Reason(s): {'; '.join(_reasons)}")
                elif is_reliable is True:
                    print("  ✓  is_reliable = True")
                    if r_squared is not None and r_squared == r_squared:
                        print(f"     R² = {r_squared:.3f}")
            elif self.mode in ('conditional', 'transfer') and self.params.get('rigorous'):
                mi_err = self.details.get('mi_error')
                mi_err_pred = self.details.get('mi_error_pred')
                is_reliable = self.details.get('is_reliable')
                fit_quality_warning = self.details.get('fit_quality_warning')
                leverage_warning = self.details.get('leverage_warning')
                if mi_err is not None:
                    print(f"  CI half-width : {mi_err:.4f} {units}  [bias-corrected, confidence interval]")
                if mi_err_pred is not None:
                    print(f"  PI half-width : {mi_err_pred:.4f} {units}  [prediction interval, more conservative]")
                if is_reliable is False:
                    print("  ⚠  is_reliable = False — rigorous extrapolation flagged issues.")
                    if fit_quality_warning:
                        print("     ↳ fit_quality_warning=True (check residuals / R²)")
                    if leverage_warning:
                        print("     ↳ leverage_warning=True (gamma=1 point has high leverage)")
                elif is_reliable is True:
                    print("  ✓  is_reliable = True  [rigorous bias-corrected estimate]")
        if self.details.get('decoder_recon_loss') is not None:
            print(f"  Decoder MSE : {self.details['decoder_recon_loss']:.6f}  (weighted reconstruction loss)")
        if self.dataframe is not None:
            rows, cols = self.dataframe.shape
            col_names = list(self.dataframe.columns)
            print(f"  DataFrame   : {rows} rows × {cols} cols  {col_names}")
        print(SEP)

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Visualizes the results of the analysis.

        This method dispatches to the appropriate plotting function based on the
        analysis `mode`.

        - For 'sweep' and 'dimensionality' modes, it plots the MI estimate
          against the swept hyperparameter.
        - For 'rigorous' mode, it plots the bias correction fit.

        Parameters
        ----------
        ax : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and
            axes are created. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the underlying plotting
            function (e.g., `figsize`, `show`, `title`).

        Returns
        -------
        plt.Axes
            The matplotlib Axes object containing the plot.

        Raises
        ------
        ValueError
            If the Results object does not contain the necessary data
            (e.g., a DataFrame) to create the plot for the given mode.
        NotImplementedError
            If plotting is not supported for the analysis mode.
        """
        from neural_mi.visualize.plot import (
            plot_sweep_curve, plot_bias_correction_fit, plot_dimensionality_curve,
        )

        show = kwargs.pop('show', True)

        units = kwargs.pop('units', self.params.get('output_units', 'bits'))

        # For modes that create their own multi-panel figure, skip creating a
        # top-level axes here (dimensionality creates two panels internally).
        _multi_panel_modes = ('dimensionality',)
        if ax is None and self.mode not in _multi_panel_modes:
            fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (10, 6)))
        elif self.mode in _multi_panel_modes:
            # Consume figsize so it can be forwarded to the multi-panel function
            pass  # figsize stays in kwargs and is popped inside plot_dimensionality_curve

        if self.mode == 'estimate':
            # Training curve: test MI vs epoch, with optional train MI overlay.
            history = self.details.get('test_mi_history')
            train_history = self.details.get('train_mi_history')
            best_epoch = self.details.get('best_epoch')
            conservative_epoch = self.details.get('conservative_epoch')
            if history is None:
                raise ValueError(
                    "Results.plot() for mode='estimate' requires 'test_mi_history' "
                    f"in result.details, but only found: {list(self.details.keys())}. "
                    "This key is populated automatically during training."
                )
            import numpy as np
            epochs = list(range(len(history)))
            ax.plot(epochs, history, color='steelblue', linewidth=1.5,
                    label='Test MI')
            if train_history is not None and len(train_history) > 0:
                train_epochs = list(range(len(train_history)))
                ax.plot(train_epochs, train_history, color='darkorange',
                        linewidth=1.5, linestyle='--', alpha=0.8, label='Train MI')
            if best_epoch is not None and 0 <= best_epoch < len(history):
                ax.axvline(best_epoch, color='tomato', linestyle='--',
                           linewidth=1.5, label=f'Best epoch ({best_epoch})')
                ax.scatter([best_epoch], [history[best_epoch]],
                           color='tomato', zorder=5, s=60)
            if conservative_epoch is not None and 0 <= conservative_epoch < len(history):
                ax.axvline(conservative_epoch, color='mediumseagreen', linestyle=':',
                           linewidth=1.5,
                           label=f'Conservative epoch ({conservative_epoch}) — used for estimate')
                ax.scatter([conservative_epoch], [history[conservative_epoch]],
                           color='mediumseagreen', zorder=5, s=60, marker='D')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'MI ({units})', fontsize=12)
            title = kwargs.pop('title', 'Training curve')
            ax.set_title(title, fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif self.mode in ('sweep', 'lag'):
            if self.dataframe is None:
                raise ValueError("Cannot plot: results do not contain a DataFrame.")

            # Infer sweep_var more robustly by excluding all known result columns
            sweep_var = self.params.get('sweep_var')
            if not sweep_var:
                possible = [c for c in self.dataframe.columns if c not in _RESULT_COLS]
                if len(possible) == 1:
                    sweep_var = possible[0]
                    logger.warning(f"Inferring sweep_var='{sweep_var}' from DataFrame.")
                elif len(possible) > 1:
                    raise ValueError(
                        f"Cannot determine sweep variable. Multiple candidates found: {possible}. "
                        f"Pass sweep_var=... explicitly."
                    )
                else:
                    raise ValueError(
                        f"Cannot determine sweep variable. DataFrame columns: "
                        f"{list(self.dataframe.columns)}. Pass sweep_var=... explicitly."
                    )
            plot_sweep_curve(self.dataframe, param_col=sweep_var, units=units, ax=ax, **kwargs)

        elif self.mode == 'dimensionality':
            if self.dataframe is None:
                raise ValueError("Cannot plot: results do not contain a DataFrame.")
            sweep_var = self.params.get('sweep_var')
            if not sweep_var:
                possible = [c for c in self.dataframe.columns if c not in _RESULT_COLS]
                sweep_var = possible[0] if len(possible) == 1 else None
                if sweep_var:
                    logger.warning(f"Inferring sweep_var='{sweep_var}' from DataFrame.")
            ax = plot_dimensionality_curve(
                self.dataframe, sweep_var=sweep_var, units=units, axes=ax, show=show, **kwargs,
            )

        elif self.mode == 'rigorous':
            if self.dataframe is None or not self.details:
                raise ValueError("Rigorous results are incomplete and cannot be plotted.")

            # Validate required keys before entering the plotter so
            # missing keys raise a clear ValueError rather than a KeyError deep
            # inside plot_bias_correction_fit.
            _REQUIRED = {'slope', 'mi_corrected', 'mi_error', 'gammas_used'}
            _missing = _REQUIRED - set(self.details.keys())
            if _missing:
                raise ValueError(
                    f"Results.plot() for mode='rigorous' is missing required keys "
                    f"in details: {sorted(_missing)}. Present: {sorted(self.details.keys())}. "
                    f"This may indicate the rigorous run failed or produced only partial results."
                )
            plot_bias_correction_fit(self.dataframe, self.details, units=units, ax=ax, **kwargs)
            # Annotate when extrapolation is flagged as unreliable
            if self.details.get('is_reliable') is False:
                ax.text(
                    0.02, 0.98, '⚠ Extrapolation unreliable (leverage_warning)',
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    color='firebrick',
                    bbox=dict(facecolor='lightyellow', edgecolor='firebrick',
                              alpha=0.85, boxstyle='round,pad=0.3'),
                )

        elif self.mode == 'conditional':
            # Bar chart showing the three CMI components.
            cmi = self.details.get('cmi_estimate')
            mi_xz_y = self.details.get('mi_xz_y')
            mi_z_y = self.details.get('mi_z_y')
            if cmi is None and mi_xz_y is None:
                raise ValueError(
                    "Cannot plot conditional results: 'cmi_estimate' and 'mi_xz_y' "
                    "are missing from result.details. "
                    f"Present keys: {sorted(self.details.keys())}."
                )
            _labels = ['I(XZ;Y)', 'I(Z;Y)', 'CMI  I(X;Y|Z)']
            _values = [mi_xz_y, mi_z_y, cmi]
            _colors = ['steelblue', 'darkorange', 'mediumseagreen']
            valid = [(l, v, c) for l, v, c in zip(_labels, _values, _colors) if v is not None]
            _labels, _values, _colors = (list(x) for x in zip(*valid))
            bars = ax.bar(_labels, _values, color=_colors, width=0.45, edgecolor='white')
            ax.set_ylabel(f'Mutual Information ({units})', fontsize=12)
            ax.set_title('Conditional MI Components', fontsize=13)
            ax.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, _values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(_values) * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                )
            import seaborn as _sns
            _sns.despine(ax=ax)

        elif self.mode == 'transfer':
            # Bar chart showing TE(X→Y), TE(Y→X) and the directionality index.
            te_xy = self.details.get('te_xy')
            te_yx = self.details.get('te_yx')
            di = self.details.get('directionality_index')
            if te_xy is None:
                raise ValueError(
                    "Cannot plot transfer results: 'te_xy' is missing from result.details. "
                    f"Present keys: {sorted(self.details.keys())}."
                )
            _labels = ['TE(X→Y)']
            _values = [te_xy]
            _colors = ['steelblue']
            if te_yx is not None:
                _labels.append('TE(Y→X)')
                _values.append(te_yx)
                _colors.append('darkorange')
            bars = ax.bar(_labels, _values, color=_colors, width=0.35, edgecolor='white')
            ax.set_ylabel(f'Transfer Entropy ({units})', fontsize=12)
            _title = 'Transfer Entropy'
            if di is not None:
                _dir_str = (
                    'X → Y dominates' if di > 0.1 else
                    'Y → X dominates' if di < -0.1 else
                    '≈ symmetric'
                )
                _title += f'\nDirectionality Index = {di:.3f}  ({_dir_str})'
            ax.set_title(_title, fontsize=13)
            ax.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, _values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(_values) * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                )
            import seaborn as _sns
            _sns.despine(ax=ax)

        elif self.mode == 'precision':
            # Precision mode produces a MI-vs-tau curve. The curve shows MI as a function of
            # corruption level (tau), with horizontal and vertical dashed lines
            # marking the threshold MI and precision tau respectively.
            if self.dataframe is None or self.dataframe.empty:
                raise ValueError(
                    "Cannot plot precision results: dataframe is missing or empty. "
                    "Expected columns: 'tau' and 'train_mi'."
                )
            df = self.dataframe.copy()
            if 'mi_mean' not in df.columns:
                if 'train_mi' in df.columns:
                    df = df.rename(columns={'train_mi': 'mi_mean'})

            precision_tau   = self.details.get('precision_tau')
            baseline_mi     = self.details.get('baseline_mi')
            threshold_value = self.details.get('threshold_value')

            tau_col = 'tau'
            mi_col  = 'mi_mean'
            if df.duplicated(subset=[tau_col]).any():
                df = (df.groupby(tau_col)[mi_col]
                        .agg(['mean', 'std'])
                        .reset_index()
                        .rename(columns={'mean': mi_col, 'std': 'mi_std'}))

            ax.plot(df[tau_col], df[mi_col], 'o-', color='steelblue',
                    linewidth=2, markersize=5, label='MI vs corruption')
            if 'mi_std' in df.columns and (df['mi_std'] > 0).any():
                ax.fill_between(df[tau_col],
                                df[mi_col] - df['mi_std'],
                                df[mi_col] + df['mi_std'],
                                alpha=0.2, color='steelblue')
            if threshold_value is not None:
                ax.axhline(threshold_value, color='tomato', linestyle='--',
                           linewidth=1.5,
                           label=f'Threshold ({threshold_value:.3f} {units})')
            if precision_tau is not None:
                ax.axvline(precision_tau, color='darkorange', linestyle='--',
                           linewidth=1.5,
                           label=f'Precision τ = {precision_tau:.4g}')
            if baseline_mi is not None:
                ax.annotate(f'Baseline MI = {baseline_mi:.3f} {units}',
                            xy=(df[tau_col].iloc[0], baseline_mi),
                            xytext=(0.05, 0.92), textcoords='axes fraction',
                            fontsize=9, color='gray',
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
            ax.set_xlabel('Corruption level (τ)', fontsize=12)
            ax.set_ylabel(f'Mutual Information ({units})', fontsize=12)
            ax.set_title('Precision Analysis: MI vs Corruption', fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif self.mode == 'pairwise':
            # Pairwise mode: render the MI matrix as a heatmap.
            mi_matrix = self.details.get('mi_matrix')
            if mi_matrix is None:
                raise ValueError(
                    "Cannot plot pairwise results: 'mi_matrix' key is missing from result.details. "
                    "Expected a 2-D numpy array."
                )
            import numpy as np
            import seaborn as sns
            units = kwargs.pop('units', self.params.get('output_units', 'bits'))
            title = kwargs.pop('title', 'Pairwise MI Matrix')
            fmt = kwargs.pop('fmt', '.3f')
            cmap = kwargs.pop('cmap', 'viridis')
            figsize = kwargs.pop('figsize', None)

            n_rows, n_cols = mi_matrix.shape
            # Default figure sizing: ~0.6 in per cell, minimum 4 × 3
            if figsize is None:
                figsize = (max(4, n_cols * 0.65 + 1.2), max(3, n_rows * 0.65 + 1.0))

            # Only create a new figure if no axes were provided; otherwise, draw into the given axes.
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize)

            # Build annotation mask: hide zero entries on diagonal (self-pairs) when symmetric
            _is_symmetric = (n_rows == n_cols and np.allclose(mi_matrix, mi_matrix.T, equal_nan=True))
            mask = np.zeros_like(mi_matrix, dtype=bool)
            if _is_symmetric:
                np.fill_diagonal(mask, True)  # mask out self-pairs

            # Axis labels: use channel indices or user-supplied variable_names
            var_names_x = self.details.get('variable_names_x') or [str(i) for i in range(n_cols)]
            var_names_y = self.details.get('variable_names_y') or [str(i) for i in range(n_rows)]

            sns.heatmap(
                mi_matrix,
                mask=mask,
                annot=True, fmt=fmt, cmap=cmap,
                xticklabels=var_names_x,
                yticklabels=var_names_y,
                cbar_kws={'label': f'MI ({units})'},
                ax=ax,
                **kwargs,
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Channel Y', fontsize=11)
            ax.set_ylabel('Channel X', fontsize=11)

        else:
            raise NotImplementedError(f"Plotting is not implemented for mode: '{self.mode}'")

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    @staticmethod
    def compare(
        results_list: List['Results'],
        labels: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Axes:
        """Overlay-plots multiple Results objects on a shared axis for comparison.

        All Results objects in the list must share the same analysis mode.
        For ``'estimate'`` mode, the test-MI training curves are overlaid with
        distinct colours; best-epoch markers are shown as dashed vertical lines.
        For ``'sweep'``, ``'lag'``, and ``'dimensionality'`` modes, the sweep
        curves are overlaid with distinct colours and a legend.  For
        ``'rigorous'`` mode, the bias-correction fits are overlaid.

        Parameters
        ----------
        results_list : list of Results
            Two or more Results objects to compare.  All must have the same mode.
        labels : list of str, optional
            Legend labels for each result.  Defaults to ``'Result 0'``,
            ``'Result 1'``, etc.
        ax : plt.Axes, optional
            An existing matplotlib Axes to plot on.  If ``None``, a new figure
            and Axes are created.
        **kwargs
            Extra keyword arguments forwarded to the underlying plot call
            (e.g., ``figsize``, ``units``).

        Returns
        -------
        plt.Axes
            The shared Axes object containing all overlaid plots.

        Raises
        ------
        ValueError
            If ``results_list`` is empty, contains only one element, or the
            results do not all share the same mode.
        NotImplementedError
            If ``compare`` is not supported for the shared mode.
        """
        if not results_list:
            raise ValueError("results_list is empty.")
        if len(results_list) < 2:
            raise ValueError(
                "results_list must contain at least two Results objects to compare."
            )

        modes = [r.mode for r in results_list]
        if len(set(modes)) > 1:
            raise ValueError(
                f"All Results objects must share the same mode. Found: {modes}."
            )
        mode = modes[0]

        if labels is None:
            labels = [f"Result {i}" for i in range(len(results_list))]
        if len(labels) != len(results_list):
            raise ValueError(
                f"labels length ({len(labels)}) must match results_list length "
                f"({len(results_list)})."
            )

        from neural_mi.visualize.plot import plot_sweep_curve, plot_bias_correction_fit

        show = kwargs.pop('show', True)
        figsize = kwargs.pop('figsize', (10, 6))
        units = kwargs.pop('units', results_list[0].params.get('output_units', 'bits'))

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)

        # Use matplotlib's default colour cycle
        colours = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        if mode == 'estimate':
            for i, (res, label) in enumerate(zip(results_list, labels)):
                history = res.details.get('test_mi_history')
                if history is None:
                    raise ValueError(
                        f"Result '{label}' (index {i}) is missing 'test_mi_history' "
                        f"in details. This key is populated automatically during training."
                    )
                ax.plot(
                    range(len(history)), history,
                    color=colours[i % len(colours)], linewidth=1.5, label=label,
                )
                best_epoch = res.details.get('best_epoch')
                if best_epoch is not None and 0 <= best_epoch < len(history):
                    ax.axvline(best_epoch, color=colours[i % len(colours)],
                               linestyle='--', linewidth=1, alpha=0.6)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'Test MI ({units})', fontsize=12)
            ax.set_title('Training Curves Comparison', fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif mode in ('sweep', 'lag', 'dimensionality'):
            for i, (res, label) in enumerate(zip(results_list, labels)):
                if res.dataframe is None:
                    raise ValueError(
                        f"Result '{label}' (index {i}) does not contain a DataFrame."
                    )
                sweep_var = res.params.get(
                    'sweep_var',
                    'embedding_dim' if mode == 'dimensionality' else None,
                )
                if not sweep_var:
                    possible = [
                        c for c in res.dataframe.columns if c not in _RESULT_COLS
                    ]
                    sweep_var = possible[0] if possible else None
                    if sweep_var is None:
                        raise ValueError(
                            f"Cannot determine sweep variable for result '{label}'. "
                            f"Set sweep_var=... in the result's params."
                        )
                plot_sweep_curve(
                    res.dataframe,
                    param_col=sweep_var,
                    units=units,
                    ax=ax,
                    label=label,
                    color=colours[i % len(colours)],
                    **kwargs,
                )
            ax.legend(fontsize=9)

        elif mode == 'rigorous':
            for i, (res, label) in enumerate(zip(results_list, labels)):
                if res.dataframe is None or not res.details:
                    raise ValueError(
                        f"Rigorous result '{label}' (index {i}) is missing "
                        f"dataframe or details."
                    )
                plot_bias_correction_fit(
                    res.dataframe,
                    res.details,
                    units=units,
                    ax=ax,
                    label=label,
                    color=colours[i % len(colours)],
                    **kwargs,
                )
            ax.legend(fontsize=9)

        else:
            raise NotImplementedError(
                f"Results.compare() is not supported for mode='{mode}'. "
                f"Supported modes: 'estimate', 'sweep', 'lag', 'dimensionality', 'rigorous'."
            )

        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def animate(self, **kwargs):
        """Animate the training history as a GIF or MP4.

        Convenience wrapper around :func:`neural_mi.visualize.animate_training`.
        All keyword arguments are forwarded unchanged.

        Common parameters
        -----------------
        panels : list of str, optional
            Which panels to include (auto-detected when omitted).
            Options: ``'mi'``, ``'spectral_metrics'``, ``'spectrum'``,
            ``'embeddings'``.
        fps : int
            Frames per second (default 10).
        output_path : str, optional
            Path to save the animation (``.gif`` or ``.mp4``).
            When ``None`` the animation is returned without saving.
        show : bool
            Whether to call ``plt.show()`` (default ``True``).
        n_components : {2, 3}
            Dimensionality for embedding scatter plots (default 2).
        reduction : {'pca', 'umap', 'none'}
            Dimensionality-reduction method for embedding panels (default ``'pca'``).
        embedding_labels : array-like or dict, optional
            Labels for colouring embedding scatter points.  Either a 1-D array
            (one subplot) or a dict ``{name: array}`` (one subplot per entry).

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        from neural_mi.visualize.animate import animate_training
        return animate_training(self, **kwargs)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Serialise this Results object to a pickle file.

        Parameters
        ----------
        path : str, optional
            Target file path or directory.

            - If ``None`` or a directory path, a filename is generated
              automatically as ``neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl``
              and placed there (defaults to the current working directory).
            - If a full file path is given, it is used directly.

            Existing files are never overwritten; a numeric suffix
            (``_1``, ``_2``, …) is appended automatically when needed.

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"neuralmi_{self.mode}_{timestamp}.pkl"

        if path is None:
            filepath = os.path.join(os.getcwd(), base_name)
        elif os.path.isdir(path):
            filepath = os.path.join(path, base_name)
        else:
            filepath = path

        if os.path.exists(filepath):
            root, ext = os.path.splitext(filepath)
            counter = 1
            while os.path.exists(f"{root}_{counter}{ext}"):
                counter += 1
            filepath = f"{root}_{counter}{ext}"

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Results saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, path: str) -> 'Results':
        """Load a Results object previously saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to a ``.pkl`` file created by :meth:`save`.

        Returns
        -------
        Results
        """
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a Results object in '{path}', got {type(obj).__name__}."
            )
        return obj

    def to_dict(self) -> dict:
        """Return a fully serialisable dictionary representation.

        All numpy arrays and torch tensors are converted to nested Python
        lists via ``.tolist()``.  DataFrames are converted to
        ``orient='records'`` lists of dicts.  Suitable for JSON export,
        logging, or downstream inspection.

        Returns
        -------
        dict
            Keys: ``'mode'``, ``'mi_estimate'``, ``'params'``, ``'details'``,
            ``'dataframe'``.  Training history lists (``'test_mi_history'``,
            ``'train_mi_history'``, etc.) are included in full under
            ``'details'``.
        """
        def _cvt(obj):
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            if isinstance(obj, dict):
                return {k: _cvt(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_cvt(v) for v in obj]
            if hasattr(obj, 'tolist'):
                return obj.tolist()  # numpy array / torch tensor → nested list
            if hasattr(obj, 'to_dict'):
                return obj.to_dict(orient='records')  # DataFrame
            return f"<{type(obj).__name__}>"

        return {
            'mode':        self.mode,
            'mi_estimate': self.mi_estimate,
            'params':      _cvt(self.params or {}),
            'details':     _cvt(self.details or {}),
            'dataframe':   (self.dataframe.to_dict(orient='records')
                            if self.dataframe is not None else None),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Export a human-readable JSON snapshot of all results.

        All numpy arrays are serialised as nested Python lists (not shape
        summaries).  Training history lists (``'test_mi_history'``,
        ``'train_mi_history'``, etc.) are included in full.  For binary
        round-trip fidelity, use :meth:`save` / :meth:`load`.

        Parameters
        ----------
        path : str, optional
            Target ``.json`` file path or directory. Auto-naming follows the
            same convention as :meth:`save` but uses a ``.json`` extension.

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        payload = self.to_dict()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"neuralmi_{self.mode}_{timestamp}.json"

        if path is None:
            filepath = os.path.join(os.getcwd(), base_name)
        elif os.path.isdir(path):
            filepath = os.path.join(path, base_name)
        else:
            filepath = path

        if os.path.exists(filepath):
            root, ext = os.path.splitext(filepath)
            counter = 1
            while os.path.exists(f"{root}_{counter}{ext}"):
                counter += 1
            filepath = f"{root}_{counter}{ext}"

        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Results exported to {filepath}")
        return filepath