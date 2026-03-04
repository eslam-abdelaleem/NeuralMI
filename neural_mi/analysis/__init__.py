# neural_mi/analysis/__init__.py
"""This package contains modules for running different analysis workflows."""
from .workflow import AnalysisWorkflow          # kept for backward compat
from .rigorous import run_rigorous_analysis
from .sweep import ParameterSweep
from .lag import run_lag_analysis
from .dimensionality import run_dimensionality_analysis
from .precision import run_precision_analysis
from .conditional import run_conditional_mi
from .transfer import run_transfer_entropy
from .pairwise import run_pairwise_mi

__all__ = [
    'AnalysisWorkflow',          # backward compat
    'run_rigorous_analysis',
    'ParameterSweep',
    'run_lag_analysis',
    'run_dimensionality_analysis',
    'run_precision_analysis',
    'run_conditional_mi',
    'run_transfer_entropy',
    'run_pairwise_mi',
]