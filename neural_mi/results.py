from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import pandas as pd

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

    def plot(self):
        """
        Generates a plot suitable for the analysis mode.

        - For 'sweep' or 'dimensionality', plots the MI curve.
        - For 'rigorous', plots the bias correction fit.
        """
        # We need to import here to avoid circular dependencies
        from neural_mi.visualize.plot import (
            plot_sweep_curve,
            plot_bias_correction_fit,
        )

        if self.mode in ['sweep', 'dimensionality']:
            if self.dataframe is None:
                raise ValueError("Cannot plot, results do not contain a DataFrame.")

            sweep_var = self.params.get('sweep_var')
            if not sweep_var and self.mode == 'dimensionality':
                sweep_var = 'embedding_dim' # Default for dimensionality

            if not sweep_var:
                raise ValueError("Cannot determine sweep variable for plotting. Was this a single run?")

            plot_sweep_curve(self.dataframe, sweep_var=sweep_var)

        elif self.mode == 'rigorous':
            if self.dataframe is None or self.details is None:
                 raise ValueError("Rigorous results are incomplete and cannot be plotted.")

            plot_bias_correction_fit(
                self.dataframe,
                mi_corrected=self.mi_estimate,
                mi_error=self.details.get('mi_error'),
                slope=self.details.get('slope')
            )
        else:
            print(f"Plotting is not implemented for mode: '{self.mode}'")