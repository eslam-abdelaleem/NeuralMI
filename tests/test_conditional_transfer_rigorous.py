"""Tests for rigorous=True in conditional and transfer modes."""
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_3d(N, C, W, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, C, W)).astype(np.float32)


from neural_mi import Model, Training, Conditional, Transfer

_MODEL = Model(embedding_dim=8, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, batch_size=64, patience=1000, learning_rate=1e-3)


# ---------------------------------------------------------------------------
# Conditional rigorous mode
# ---------------------------------------------------------------------------

class TestConditionalRigorous:
    """Tests for rigorous bias correction in conditional MI estimation."""

    def test_conditional_rigorous_returns_mi_estimate(self):
        """run() with mode='conditional' and rigorous=True should return a bias-corrected MI."""
        import neural_mi as nmi
        N = 400
        x = _make_random_3d(N, 2, 8, seed=1)
        y = _make_random_3d(N, 2, 8, seed=2)
        w = _make_random_3d(N, 2, 8, seed=3)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None, "rigorous conditional should set mi_estimate"
        assert result.dataframe is not None, "rigorous conditional should set dataframe"

    def test_conditional_rigorous_details_contain_required_keys(self):
        """Rigorous conditional result must have the standard rigorous details dict."""
        import neural_mi as nmi
        N = 400
        x = _make_random_3d(N, 2, 8, seed=4)
        y = _make_random_3d(N, 2, 8, seed=5)
        w = _make_random_3d(N, 2, 8, seed=6)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        for key in ('is_reliable', 'slope', 'mi_error',
                    'gammas_used', 'fit_quality_warning', 'leverage_warning'):
            assert key in result.details, (
                f"Missing key '{key}' in result.details; "
                f"available: {sorted(result.details.keys())}"
            )

    def test_conditional_rigorous_params_flag_set(self):
        """result.params should record rigorous=True for conditional rigorous runs."""
        import neural_mi as nmi
        N = 300
        x = _make_random_3d(N, 1, 5, seed=7)
        y = _make_random_3d(N, 1, 5, seed=8)
        w = _make_random_3d(N, 1, 5, seed=9)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w, rigorous=True,
                                    gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.params.get('rigorous'), (
            "result.params should have rigorous=True for rigorous conditional"
        )

    def test_conditional_standard_path_unaffected(self):
        """Ensure rigorous=False (default) still works for conditional mode."""
        import neural_mi as nmi
        N = 300
        x = _make_random_3d(N, 2, 6, seed=10)
        y = _make_random_3d(N, 2, 6, seed=11)
        w = _make_random_3d(N, 2, 6, seed=12)
        result = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None
        # Standard path should NOT have rigorous keys in details
        assert 'gammas_used' not in result.details


# ---------------------------------------------------------------------------
# Transfer rigorous mode
# ---------------------------------------------------------------------------

class TestTransferRigorous:
    """Tests for rigorous bias correction in transfer entropy estimation."""

    def test_transfer_rigorous_returns_mi_estimate(self):
        """run() with mode='transfer' and rigorous=True should return a bias-corrected TE."""
        import neural_mi as nmi
        T = 500
        rng = np.random.default_rng(13)
        x = rng.normal(size=(T, 2)).astype(np.float32)
        y = rng.normal(size=(T, 2)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=5, rigorous=True,
                              gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None, "rigorous transfer should set mi_estimate"
        assert result.dataframe is not None, "rigorous transfer should set dataframe"

    def test_transfer_rigorous_details_contain_required_keys(self):
        """Rigorous transfer result must have the standard rigorous details dict."""
        import neural_mi as nmi
        T = 500
        rng = np.random.default_rng(14)
        x = rng.normal(size=(T, 1)).astype(np.float32)
        y = rng.normal(size=(T, 1)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=4, rigorous=True,
                              gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        for key in ('is_reliable', 'slope', 'fit_quality_warning', 'leverage_warning'):
            assert key in result.details, (
                f"Missing key '{key}' in result.details"
            )

    def test_transfer_standard_path_unaffected(self):
        """Ensure rigorous=False (default) still works for transfer mode."""
        import neural_mi as nmi
        T = 300
        rng = np.random.default_rng(15)
        x = rng.normal(size=(T, 1)).astype(np.float32)
        y = rng.normal(size=(T, 1)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=3),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.mi_estimate is not None
        # Standard path should NOT have rigorous keys
        assert 'gammas_used' not in result.details

    def test_transfer_rigorous_params_flag_set(self):
        """result.params should record rigorous=True for rigorous transfer runs."""
        import neural_mi as nmi
        T = 400
        rng = np.random.default_rng(16)
        x = rng.normal(size=(T, 1)).astype(np.float32)
        y = rng.normal(size=(T, 1)).astype(np.float32)
        result = nmi.run(
            x, y,
            mode='transfer',
            transfer=Transfer(history_window=4, rigorous=True,
                              gamma_range=range(1, 4), min_gamma_points=2),
            model=_MODEL, training=_TRAINING,
            verbose=False, show_progress=False,
        )
        assert result.params.get('rigorous'), (
            "result.params should have rigorous=True for rigorous transfer"
        )


# ---------------------------------------------------------------------------
# Unit conversion (nats -> bits) for rigorous conditional/transfer
# ---------------------------------------------------------------------------

_NATS_TO_BITS = 1 / math.log(2)

# A fixed stand-in for run_rigorous_scalar_analysis's return value -- mocking
# the training-dependent computation out entirely isolates the unit-conversion
# bug from unrelated run-to-run training variance.
_FIXED_RIG_RESULT = {
    'mi_corrected': 1.0, 'mi_error': 0.1, 'mi_error_pred': 0.2, 'slope': -0.05,
    'is_reliable': True, 'gammas_used': [1, 2, 3], 'r_squared': 0.9,
    'fit_quality_warning': False, 'leverage_warning': False,
    'max_abs_residual': 1.0, 'loo_intercept_shift': 0.01,
}


def _fixed_rig_result():
    return dict(_FIXED_RIG_RESULT,
               raw_results_df=pd.DataFrame({'gamma': [1, 2, 3], 'train_mi': [1.2, 1.1, 1.0]}))


class TestRigorousConditionalTransferUnitConversion:
    """result.mi_estimate / details['mi_error'] and dataframe['train_mi'] must be
    converted to bits exactly once, matching every other mode's convention.

    Two bugs found together in run.py's rigorous conditional/transfer branches:
    (1) _convert_mi_units's dict branch never converted 'mi_corrected'/'mi_error'/
        'mi_error_pred'/'slope' (only the DataFrame and list-of-dicts branches did),
        so these stayed in nats even with output_units='bits' (the default).
    (2) 'raw_results_df' was converted once inside _convert_mi_units(rig_details, ...)
        (dict branch recurses into it), then popped and converted a SECOND time,
        double-scaling 'train_mi' by NATS_TO_BITS**2 instead of NATS_TO_BITS.
    """

    def test_conditional_rigorous_scalar_fields_converted_exactly_once(self):
        import neural_mi as nmi
        from neural_mi import Model, Split, Training as _Training, Conditional, Output
        x = np.random.default_rng(0).normal(size=(50, 1)).astype(np.float32)
        y = np.random.default_rng(1).normal(size=(50, 1)).astype(np.float32)
        w = np.random.default_rng(2).normal(size=(50, 1)).astype(np.float32)
        common = dict(model=Model(embedding_dim=4, hidden_dim=8),
                     training=_Training(n_epochs=1, batch_size=32),
                     split=Split(mode='random'), show_progress=False)

        with patch('neural_mi.analysis.rigorous.run_rigorous_scalar_analysis',
                  return_value=_fixed_rig_result()):
            r_bits = nmi.run(x, y, mode='conditional', output=Output(units='bits'),
                             conditional=Conditional(w_data=w, rigorous=True, min_gamma_points=3),
                             n_workers=1, **common)

        assert r_bits.mi_estimate == pytest.approx(1.0 * _NATS_TO_BITS)
        assert r_bits.details['mi_error'] == pytest.approx(0.1 * _NATS_TO_BITS)
        assert r_bits.details['mi_error_pred'] == pytest.approx(0.2 * _NATS_TO_BITS)
        assert r_bits.details['slope'] == pytest.approx(-0.05 * _NATS_TO_BITS)
        assert r_bits.dataframe['train_mi'].tolist() == pytest.approx(
            [1.2 * _NATS_TO_BITS, 1.1 * _NATS_TO_BITS, 1.0 * _NATS_TO_BITS]
        )

    def test_transfer_rigorous_scalar_fields_converted_exactly_once(self):
        import neural_mi as nmi
        from neural_mi import Model, Split, Training as _Training, Transfer, Output
        rng = np.random.default_rng(3)
        x = rng.normal(size=(200, 1)).astype(np.float32)
        y = rng.normal(size=(200, 1)).astype(np.float32)
        common = dict(model=Model(embedding_dim=4, hidden_dim=8),
                     training=_Training(n_epochs=1, batch_size=32),
                     split=Split(mode='random'), show_progress=False)

        with patch('neural_mi.analysis.rigorous.run_rigorous_scalar_analysis',
                  return_value=_fixed_rig_result()):
            r_bits = nmi.run(x, y, mode='transfer', output=Output(units='bits'),
                             transfer=Transfer(history_window=4, rigorous=True, min_gamma_points=3),
                             n_workers=1, **common)

        assert r_bits.mi_estimate == pytest.approx(1.0 * _NATS_TO_BITS)
        assert r_bits.details['mi_error'] == pytest.approx(0.1 * _NATS_TO_BITS)
        assert r_bits.dataframe['train_mi'].tolist() == pytest.approx(
            [1.2 * _NATS_TO_BITS, 1.1 * _NATS_TO_BITS, 1.0 * _NATS_TO_BITS]
        )
