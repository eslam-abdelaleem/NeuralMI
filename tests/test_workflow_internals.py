# tests/test_workflow_internals.py
import pytest
import pandas as pd
import numpy as np
from neural_mi.analysis.rigorous import _find_linear_region as find_linear_region
from neural_mi.analysis.rigorous import _extrapolate_mi as extrapolate_mi
from neural_mi.analysis.rigorous import _post_process_and_correct

class TestWorkflowInternals:
    def test_find_linear_region(self):
        # _find_linear_region works in the space of (1/gamma, train_mi).
        # For all 5 gammas to be retained, the data must be linear in 1/gamma,
        # i.e. train_mi = a + b*(1/gamma).
        gammas = [1, 2, 3, 4, 5]
        df = pd.DataFrame({
            'gamma': gammas,
            # train_mi = 2*(1/gamma) + 1  — perfectly linear in 1/gamma
            'train_mi': [2.0 / g + 1.0 for g in gammas],
        })
        gammas_kept = find_linear_region(df, delta_threshold=0.1, min_gamma_points=3, verbose=False)
        assert len(gammas_kept) == 5

    def test_extrapolate_mi(self):
        # _extrapolate_mi fits train_mi = intercept + slope * (1/gamma) and
        # now returns (intercept, mi_error, mi_error_pred, slope).
        # mi_error  = confidence-interval half-width on the fitted mean.
        # mi_error_pred = prediction-interval half-width (more conservative).
        # Data: train_mi = 6 - 1/gamma  →  intercept = 6, slope = -1.
        gammas = [1, 2, 3, 4, 5]
        df = pd.DataFrame({
            'gamma': gammas,
            'train_mi': [6.0 - 1.0 / g for g in gammas],
        })
        intercept, mi_error, mi_error_pred, slope = extrapolate_mi(
            df, [1, 2, 3, 4, 5], confidence_level=0.95
        )
        assert np.isclose(intercept, 6.0)
        assert np.isclose(slope, -1.0)
        # prediction interval must be at least as wide as the confidence interval
        assert mi_error_pred >= mi_error - 1e-9

    def test_post_process_and_correct(self):
        df = pd.DataFrame({
            'gamma': [1, 2, 3, 4, 5] * 2,
            'train_mi': [5, 4, 3, 2, 1] * 2, # Two identical runs
            'param': ['a'] * 5 + ['b'] * 5
        })

        results = _post_process_and_correct(
            df, sweep_grid={'param': ['a', 'b']}, delta_threshold=0.1,
            min_gamma_points=3, confidence_level=0.95, verbose=False
        )
        assert len(results) == 2
        assert results[0]['mi_corrected'] is not None
