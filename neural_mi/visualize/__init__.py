# neural_mi/visualize/__init__.py
"""This package contains modules for visualizing analysis results."""
from .plot import (
    plot_sweep_curve,
    plot_dimensionality_curve,
    plot_bias_correction_fit,
    plot_cross_correlation,
    analyze_mi_heatmap,
    set_publication_style,
    plot_embeddings,
)
from .animate import animate_training

__all__ = [
    'plot_sweep_curve',
    'plot_dimensionality_curve',
    'plot_bias_correction_fit',
    'plot_cross_correlation',
    'analyze_mi_heatmap',
    'set_publication_style',
    'plot_embeddings',
    'animate_training',
]