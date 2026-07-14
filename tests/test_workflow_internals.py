# tests/test_workflow_internals.py
import pytest
import pandas as pd
import numpy as np
import torch
from neural_mi.analysis.rigorous import _find_linear_region as find_linear_region
from neural_mi.analysis.rigorous import _extrapolate_mi as extrapolate_mi
from neural_mi.analysis.rigorous import _post_process_and_correct
from neural_mi.analysis.rigorous import AnalysisWorkflow

class TestWorkflowInternals:
    def test_find_linear_region(self):
        # _find_linear_region works in the space of (gamma, train_mi).
        # For all 5 gammas to be retained, the data must be linear in gamma,
        # i.e. train_mi = I_true + c * gamma.
        gammas = [1, 2, 3, 4, 5]
        df = pd.DataFrame({
            'gamma': gammas,
            # train_mi = 1.0 + 0.5 * gamma  — perfectly linear in gamma
            'train_mi': [1.0 + 0.5 * g for g in gammas],
        })
        gammas_kept = find_linear_region(df, delta_threshold=0.1, min_gamma_points=3)
        assert len(gammas_kept) == 5

    def test_extrapolate_mi(self):
        # _extrapolate_mi fits train_mi = intercept + slope * gamma and
        # returns (intercept, mi_error, mi_error_pred, slope).
        # intercept = I_true (extrapolated at gamma=0 = infinite data).
        # mi_error  = confidence-interval half-width on the fitted mean.
        # mi_error_pred = prediction-interval half-width (more conservative).
        # Data: train_mi = 2.0 + 0.5 * gamma  →  I_true = 2.0, slope = 0.5.
        gammas = [1, 2, 3, 4, 5]
        df = pd.DataFrame({
            'gamma': gammas,
            'train_mi': [2.0 + 0.5 * g for g in gammas],
        })
        intercept, mi_error, mi_error_pred, slope = extrapolate_mi(
            df, [1, 2, 3, 4, 5], confidence_level=0.95
        )
        assert np.isclose(intercept, 2.0)
        assert np.isclose(slope, 0.5)
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
            min_gamma_points=3, confidence_level=0.95
        )
        assert len(results) == 2
        assert results[0]['mi_corrected'] is not None

    def test_input_dim_uses_full_flattened_shape_for_4d_data(self):
        """input_dim_x/y must be the product of ALL trailing dims (C*H*W for
        4-D cnn2d-shaped input), not just shape[1]*shape[2] which silently
        drops the width axis for 4-D data."""
        x_4d = torch.randn(20, 3, 8, 8)   # (N, C, H, W)
        y_4d = torch.randn(20, 3, 8, 8)
        workflow = AnalysisWorkflow(x_4d, y_4d, base_params={})
        assert workflow.base_params['input_dim_x'] == 3 * 8 * 8
        assert workflow.base_params['input_dim_y'] == 3 * 8 * 8

        # 3-D data must be unaffected (same value as the old shape[1]*shape[2]).
        x_3d = torch.randn(20, 4, 16)
        y_3d = torch.randn(20, 4, 16)
        workflow_3d = AnalysisWorkflow(x_3d, y_3d, base_params={})
        assert workflow_3d.base_params['input_dim_x'] == 4 * 16
        assert workflow_3d.base_params['input_dim_y'] == 4 * 16
