# neural_mi/results.py

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
from neural_mi.logger import logger

@dataclass
class Results:
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)
    mi_estimate: Optional[float] = None
    dataframe: Optional[pd.DataFrame] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        rep = f"Results(mode='{self.mode}'"
        if self.mi_estimate is not None: rep += f", mi_estimate={self.mi_estimate:.4f}"
        if self.dataframe is not None: rep += f", dataframe_shape={self.dataframe.shape}"
        if self.details: rep += f", details_keys={list(self.details.keys())}"
        return rep + ")"

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        from neural_mi.visualize.plot import plot_sweep_curve, plot_bias_correction_fit
        
        show = kwargs.pop('show', True)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (10, 6)))

        if self.mode in ['sweep', 'dimensionality']:
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