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
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (10, 6)))

        if self.mode in ['sweep', 'dimensionality', 'lag']:
            if self.dataframe is None:
                raise ValueError("Cannot plot: results do not contain a DataFrame.")
            sweep_var = self.params.get('sweep_var', 'embedding_dim' if self.mode == 'dimensionality' else None)
            if not sweep_var:
                possible = [c for c in self.dataframe.columns if c not in ['mi_mean', 'mi_std']]
                if len(possible) == 1:
                    sweep_var = possible[0]
                    logger.warning(f"Inferring sweep_var='{sweep_var}' from DataFrame.")
                else:
                    raise ValueError(f"Cannot determine sweep variable from {possible}.")
            plot_sweep_curve(self.dataframe, param_col=sweep_var, ax=ax, **kwargs)
        elif self.mode == 'rigorous':
            if self.dataframe is None or not self.details:
                raise ValueError("Rigorous results are incomplete and cannot be plotted.")
            plot_bias_correction_fit(self.dataframe, self.details, ax=ax, **kwargs)
        else:
            raise NotImplementedError(f"Plotting is not implemented for mode: '{self.mode}'")

        if show:
            plt.tight_layout()
            plt.show()
        return ax