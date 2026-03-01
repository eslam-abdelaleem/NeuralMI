# neural_mi/results.py
"""Defines the `Results` class for storing and interacting with analysis outcomes.

This module provides a standardized data structure for holding the results of
different analysis modes from the `run` function. The `Results` class acts as
a container for MI estimates, dataframes, and detailed metadata, and also
provides a convenient `.plot()` method for visualizing the results.
"""
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
from neural_mi.logger import logger

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
        'dimensionality', and 'rigorous' modes.
    details : Dict[str, Any]
        A dictionary containing additional metadata or detailed results, such
        as raw run data or estimated latent dimensions.
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

        if self.mode in ['sweep', 'dimensionality', 'lag']:
            if self.dataframe is None:
                raise ValueError("Cannot plot: results do not contain a DataFrame.")

            # Infer sweep_var more robustly by excluding all known result columns 
            _RESULT_COLS = {
                'mi_mean', 'mi_std', 'test_mi', 'train_mi', 'mi_corrected',
                'mi_error', 'slope', 'run_id', 'is_reliable', 'gammas_used',
            }
            sweep_var = self.params.get('sweep_var', 'embedding_dim' if self.mode == 'dimensionality' else None)
            if not sweep_var:
                possible = [c for c in self.dataframe.columns if c not in _RESULT_COLS]
                if len(possible) == 1:
                    sweep_var = possible[0]
                    logger.warning(f"Inferring sweep_var='{sweep_var}' from DataFrame.")
                elif len(possible) > 1:
                    sweep_var = possible[0]
                    logger.warning(
                        f"Multiple candidate sweep variables found: {possible}. "
                        f"Using '{sweep_var}'. Pass sweep_var=... explicitly to suppress."
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