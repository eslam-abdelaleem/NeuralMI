from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import pandas as pd
from matplotlib.axes import Axes

@dataclass
class Results:
    """A standardized container for the results of an analysis run.

    This object provides a consistent interface to the outputs of the
    `nmi.run()` function, regardless of the mode used.

    Parameters
    ----------
    mode : str
        The analysis mode that generated these results (e.g., 'estimate').
    params : dict, optional
        The parameters used for the analysis run.
    mi_estimate : float, optional
        The final point estimate of mutual information.
    dataframe : pd.DataFrame, optional
        A DataFrame containing detailed, row-by-row results.
    details : dict, optional
        A dictionary for any mode-specific details or metadata.

    Attributes
    ----------
    mi_estimate : float or None
        The primary output for 'estimate' and 'rigorous' modes.
    dataframe : pd.DataFrame or None
        The primary output for 'sweep' and 'dimensionality' modes. For
        'rigorous' mode, this contains the raw, uncorrected MI estimates.
    details : dict
        For 'rigorous' mode, this includes fit parameters like `slope`,
        `mi_error`, `is_reliable`, etc.
    mode : str
    params : dict

    Examples
    --------
    >>> results = nmi.run(...)
    >>> if results.mi_estimate is not None:
    ...     print(f"MI Estimate: {results.mi_estimate:.3f}")
    >>> if results.dataframe is not None:
    ...     print(results.dataframe.head())
    >>> results.plot()

    """
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)
    mi_estimate: Optional[float] = None
    dataframe: Optional[pd.DataFrame] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        rep_str = f"Results(mode='{self.mode}'"
        if self.mi_estimate is not None:
            rep_str += f", mi_estimate={self.mi_estimate:.4f}"
        if self.dataframe is not None:
            rep_str += f", dataframe_shape={self.dataframe.shape}"
        if self.details:
            rep_str += f", details={list(self.details.keys())}"
        rep_str += ")"
        return rep_str

    def plot(self, ax: Optional[Axes] = None, **kwargs: Any) -> Axes:
        """
        Generates a plot suitable for the analysis mode.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing axes object to plot on. If None, creates a new figure.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.
            Common options:
            - figsize : tuple, size of figure if creating new (default: (8, 6))
            - title : str, custom title for the plot
            - show : bool, whether to call plt.show() (default: True)

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        import matplotlib.pyplot as plt
        import warnings
        from neural_mi.visualize.plot import (
            plot_sweep_curve,
            plot_bias_correction_fit,
        )

        show_plot = kwargs.pop('show', True)
        figsize = kwargs.pop('figsize', (8, 6))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if self.mode in ['sweep', 'dimensionality']:
            if self.dataframe is None:
                raise ValueError("Cannot plot, results do not contain a DataFrame.")

            sweep_var = self.params.get('sweep_var')
            if not sweep_var:
                if self.mode == 'dimensionality':
                    sweep_var = 'embedding_dim'
                else:
                    # Try to infer from dataframe columns
                    possible_vars = [
                        col for col in self.dataframe.columns
                        if col not in ['mi_mean', 'mi_std', 'test_mi', 'train_mi', 'best_epoch']
                    ]
                    if len(possible_vars) == 1:
                        sweep_var = possible_vars[0]
                        warnings.warn(f"Sweep variable not specified, inferring '{sweep_var}'")
                    else:
                        raise ValueError(f"Cannot determine sweep variable. Please specify 'sweep_var' in run() params.")

            plot_sweep_curve(self.dataframe, param_col=sweep_var, ax=ax, **kwargs)

        elif self.mode == 'rigorous':
            if self.dataframe is None or self.details is None:
                 raise ValueError("Rigorous results are incomplete and cannot be plotted.")

            plot_bias_correction_fit(
                raw_results_df=self.dataframe,
                corrected_result=self.details,
                ax=ax,
                **kwargs
            )
        else:
            raise ValueError(f"Plotting is not implemented for mode: '{self.mode}'")

        if show_plot:
            plt.tight_layout()
            plt.show()

        return ax