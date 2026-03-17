# neural_mi/analysis/workflow.py
"""Backward-compatibility shim — all content has moved to rigorous.py.

Import from ``neural_mi.analysis.rigorous`` directly.
"""
from .rigorous import (  # noqa: F401  (re-exported for backward compat)
    AnalysisWorkflow,
    _find_linear_region,
    _extrapolate_mi,
    _post_process_and_correct,
)
