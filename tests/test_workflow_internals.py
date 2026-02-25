# tests/test_workflow_internals.py
import pytest
import pandas as pd
import numpy as np
from neural_mi.analysis.workflow import __find_linear_region as find_linear_region
from neural_mi.analysis.workflow import __extrapolate_mi as extrapolate_mi
from neural_mi.analysis.workflow import _post_process_and_correct

class TestWorkflowInternals:
    def test_find_linear_region(self):
        # Create perfectly linear data
        df = pd.DataFrame({
            'gamma': [1, 2, 3, 4, 5],
            'test_mi': [1.0, 0.5, 0.33, 0.25, 0.2]
        })

        # If we provide linear data y = 2x + 1
        df['test_mi'] = 2 * df['gamma'] + 1
        gammas = find_linear_region(df, delta_threshold=0.1, min_gamma_points=3, verbose=False)
        assert len(gammas) == 5

    def test_extrapolate_mi(self):
        df = pd.DataFrame({
            'gamma': [1, 2, 3, 4, 5],
            'test_mi': [5, 4, 3, 2, 1] # y = 6 - gamma. Intercept 6.
        })
        intercept, error, slope = extrapolate_mi(df, [1, 2, 3, 4, 5], confidence_level=0.95)
        assert np.isclose(intercept, 6.0)
        assert np.isclose(slope, -1.0)

    def test_post_process_and_correct(self):
        df = pd.DataFrame({
            'gamma': [1, 2, 3, 4, 5] * 2,
            'test_mi': [5, 4, 3, 2, 1] * 2, # Two identical runs
            'param': ['a'] * 5 + ['b'] * 5
        })

        results = _post_process_and_correct(
            df, sweep_grid={'param': ['a', 'b']}, delta_threshold=0.1,
            min_gamma_points=3, confidence_level=0.95, verbose=False
        )
        assert len(results) == 2
        assert results[0]['mi_corrected'] is not None
