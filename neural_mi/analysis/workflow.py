# neural_mi/analysis/workflow.py
"""Re-exports from rigorous.py for convenience.

All rigorous analysis logic lives in ``neural_mi.analysis.rigorous``.
"""
from .rigorous import (  # noqa: F401
    AnalysisWorkflow,
    _find_linear_region,
    _extrapolate_mi,
    _post_process_and_correct,
)
