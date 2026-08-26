# tests/test_workflow_internals.py
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
        gammas_kept, converged, stats = find_linear_region(
            df, curvature_t_threshold=2.0, min_gamma_points=3)
        assert len(gammas_kept) == 5
        # Perfectly linear data satisfies the curvature criterion outright.
        assert converged is True

    def test_find_linear_region_never_shrinks_below_floor(self):
        # Strongly curved data never satisfies the curvature criterion. The
        # search must stop AT min_gamma_points rather than stepping past it,
        # and must report that no linear region was found rather than leaving
        # the caller to infer it from an undersized list.
        gammas = [1, 2, 3, 4, 5, 6, 7]
        df = pd.DataFrame({
            'gamma': [g for g in gammas for _ in range(3)],
            'train_mi': [2.0 - 0.02 * g - 0.03 * g ** 2 for g in gammas for _ in range(3)],
        })
        for floor in (3, 4, 5):
            kept, converged, _stats = find_linear_region(
                df, curvature_t_threshold=2.0, min_gamma_points=floor)
            assert len(kept) >= floor, (
                f"search shrank to {len(kept)} gammas, below the floor of {floor}")
            assert converged is False

    def test_linearity_verdict_is_independent_of_bias_magnitude(self):
        """The whole point of the t-test criterion.

        The old rule tested |a2/a1|, dividing curvature by the bias slope. On
        genuinely linear data that made the verdict depend on how much bias
        happened to be present, since a1 -> 0 inflates the ratio. The verdict
        must depend on how linear the data is, not on how steep it is.

        Note this is a significance test, so on genuinely linear data it
        rejects at roughly its nominal rate (about 5% at t=2) no matter how
        clean the data is. That is the criterion working, not failing, so the
        assertion is on the *rate* and on its *stability across slopes*, never
        on any single draw.
        """
        gammas = list(range(1, 11))
        slopes = (-0.5, -0.05, -0.005, -0.0005, 0.0)
        rates = {}
        for slope in slopes:
            rng = np.random.default_rng(abs(hash(slope)) % 2**31)
            hits = 0
            for _ in range(60):
                df = pd.DataFrame({
                    'gamma': [g for g in gammas for _ in range(3)],
                    'train_mi': [2.0 + slope * g + rng.normal(0, 0.02)
                                 for g in gammas for _ in range(3)],
                })
                _kept, converged, _stats = find_linear_region(
                    df, curvature_t_threshold=2.0, min_gamma_points=5)
                hits += converged
            rates[slope] = hits / 60

        for slope, rate in rates.items():
            assert rate >= 0.85, (
                f"linear data with slope {slope} accepted only {rate:.0%} of the "
                f"time; expected roughly the nominal 95%")
        spread = max(rates.values()) - min(rates.values())
        assert spread <= 0.15, (
            f"acceptance rate varies by {spread:.0%} across bias magnitude "
            f"({rates}); the verdict must not track how steep the trend is")

    def test_curvature_statistics_are_reported(self):
        gammas = list(range(1, 11))
        df = pd.DataFrame({
            'gamma': [g for g in gammas for _ in range(3)],
            'train_mi': [2.0 - 0.02 * g - 0.03 * g ** 2 for g in gammas for _ in range(3)],
        })
        _kept, _conv, stats = find_linear_region(
            df, curvature_t_threshold=2.0, min_gamma_points=5)
        for key in ('curvature_coefficient', 'curvature_se',
                    'curvature_t', 'curvature_slope'):
            assert key in stats and np.isfinite(stats[key])
        # the reported t must be the ratio the decision was actually made on
        assert np.isclose(stats['curvature_t'],
                          abs(stats['curvature_coefficient'] / stats['curvature_se']))

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
            df, sweep_grid={'param': ['a', 'b']}, curvature_t_threshold=2.0,
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
