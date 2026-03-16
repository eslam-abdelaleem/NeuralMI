# neural_mi/results.py
"""Defines the `Results` class for storing and interacting with analysis outcomes.

This module provides a standardized data structure for holding the results of
different analysis modes from the `run` function. The `Results` class acts as
a container for MI estimates, dataframes, and detailed metadata, and also
provides a convenient `.plot()` method for visualizing the results.
"""
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from neural_mi.logger import logger

# Module-level constant: columns produced by the MI estimator that are NOT
# sweep/hyperparameter variables. Used when inferring the x-axis of a sweep plot.
_RESULT_COLS: frozenset = frozenset({
    'mi_mean', 'mi_std', 'test_mi', 'train_mi', 'mi_corrected',
    'mi_error', 'slope', 'run_id', 'is_reliable', 'gammas_used',
    'n_windows', 'lag',
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
    compare(results_list, labels=None, ax=None, **kwargs)
        Static method; overlay multiple Results on a shared axis.
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
        ``mode='rigorous'``), and DataFrame shape (if a DataFrame is present).
        """
        SEP = "─" * 50
        units = self.params.get('output_units', 'bits')
        print(SEP)
        print(f"  NeuralMI Results  |  mode = '{self.mode}'")
        print(SEP)
        if self.mi_estimate is not None:
            print(f"  MI estimate : {self.mi_estimate:.4f} {units}")
            if self.mode == 'rigorous':
                mi_err = self.details.get('mi_error')
                is_reliable = self.details.get('is_reliable')
                if mi_err is not None:
                    print(f"  ± (half CI)   : {mi_err:.4f} {units}")
                if is_reliable is False:
                    print("  ⚠  is_reliable = False — extrapolation is unreliable; "
                          "collect more data or simplify the model.")
                elif is_reliable is True:
                    print("  ✓  is_reliable = True")
        else:
            print("  MI estimate : (none — see result.dataframe or result.details)")
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
        from neural_mi.visualize.plot import plot_sweep_curve, plot_bias_correction_fit
        
        show = kwargs.pop('show', True)

        units = kwargs.pop('units', self.params.get('output_units', 'bits'))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (10, 6)))

        if self.mode == 'estimate':
            # Training curve: test MI vs epoch, with optional train MI overlay.
            history = self.details.get('test_mi_history')
            train_history = self.details.get('train_mi_history')
            best_epoch = self.details.get('best_epoch')
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
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'MI ({units})', fontsize=12)
            title = kwargs.pop('title', 'Training curve')
            ax.set_title(title, fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        elif self.mode in ['sweep', 'dimensionality', 'lag']:
            if self.dataframe is None:
                raise ValueError("Cannot plot: results do not contain a DataFrame.")

            # Infer sweep_var more robustly by excluding all known result columns
            sweep_var = self.params.get('sweep_var', 'embedding_dim' if self.mode == 'dimensionality' else None)
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

        elif self.mode == 'precision':
            # Precision mode produces a MI-vs-tau curve. The curve shows MI as a function of
            # corruption level (tau), with horizontal and vertical dashed lines
            # marking the threshold MI and precision tau respectively.
            if self.dataframe is None or self.dataframe.empty:
                raise ValueError(
                    "Cannot plot precision results: dataframe is missing or empty. "
                    "Expected columns: 'tau' and 'mi_mean' (or 'test_mi')."
                )
            df = self.dataframe.copy()
            if 'mi_mean' not in df.columns and 'test_mi' in df.columns:
                df = df.rename(columns={'test_mi': 'mi_mean'})

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

        if mode in ('sweep', 'lag', 'dimensionality'):
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
                f"Supported modes: 'sweep', 'lag', 'dimensionality', 'rigorous'."
            )

        if show:
            plt.tight_layout()
            plt.show()
        return ax