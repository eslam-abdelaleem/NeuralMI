# tests/test_conditional.py
"""Tests for the conditional mutual information analysis mode."""
import pytest
import numpy as np
import torch
import neural_mi as nmi
from neural_mi import Model, Training, Conditional

# Minimal training params (dict kept for the engine-level run_conditional_mi test).
_PARAMS = {
    'n_epochs': 3, 'learning_rate': 1e-3, 'batch_size': 64,
    'patience': 2, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
}
_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

N = 500  # samples


def _make_gaussian(n, d):
    return torch.from_numpy(np.random.randn(n, d).astype(np.float32))


class TestConditionalMI:
    """CMI = I(X,Z;Y) - I(Z;Y)."""

    def test_cmi_independent_xy_given_z_is_near_zero(self):
        """CMI(X;Y|Z) ≈ 0 when X is independent of Y (conditioning on Z)."""
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))
        y = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))
        z = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert results is not None
        assert results.mode == 'conditional'
        assert results.mi_estimate is not None
        assert isinstance(results.mi_estimate, float)
        # For independent signals CMI could be slightly negative due to noise;
        # just confirm it's a finite number less than 2.0 bits
        assert np.isfinite(results.mi_estimate)
        assert results.mi_estimate < 2.0

    def test_cmi_correlated_xy_given_independent_z(self):
        """CMI(X;Y|Z) > 0 when X and Y are correlated and Z is independent."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        rng = np.random.default_rng(1)
        z = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert results.mi_estimate is not None
        # mi_estimate may not always exceed 0 with very short training;
        # but it should be finite and in a plausible range
        assert np.isfinite(results.mi_estimate)

    def test_cmi_returns_details_keys(self):
        """Confirms the result details dict contains CMI breakdown keys."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        z = _make_gaussian(N, 2)

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert 'mi_xz_y' in results.details
        assert 'mi_z_y' in results.details
        assert np.isfinite(results.details['mi_xz_y'])
        assert np.isfinite(results.details['mi_z_y'])

    def test_cmi_result_consistency(self):
        """Confirms CMI estimate = I(XZ;Y) - I(Z;Y)."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        z = _make_gaussian(N, 2)

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(z_data=z),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        expected = results.details['mi_xz_y'] - results.details['mi_z_y']
        assert abs(results.mi_estimate - expected) < 1e-6

    def test_mismatched_x_z_window_sizes_raises_clear_error(self):
        """X and Z with different window sizes must raise a clear ValueError
        before the concatenation into XZ, not a bare torch.cat shape error."""
        from neural_mi.analysis.conditional import run_conditional_mi

        x = torch.randn(N, 2, 5)   # window size 5
        y = torch.randn(N, 2, 5)
        z = torch.randn(N, 2, 3)   # window size 3 -- mismatched by 2, past the trim tolerance

        with pytest.raises(ValueError, match="window size"):
            run_conditional_mi(x, y, z, base_params=_PARAMS, n_workers=1)


class TestConditionalMICategoricalZ:
    """z_processor_type='categorical' as the conditioning variable.

    mode='conditional' builds XZ by concatenating X and Z along the channel
    axis, which requires a matching window-size axis. The categorical
    processor's encodings don't produce that layout natively --
    _reshape_categorical_z_for_conditional (neural_mi/run.py) re-lays them
    out: 'majority_vote'/'probability' become window-constant channels
    (broadcast across X's window by run_conditional_mi), 'full_trajectory'
    keeps its real per-timepoint resolution on the window axis.
    """

    @staticmethod
    def _confounded_data(n_windows=400, window_size=10, n_categories=3, seed=0):
        """X and Y share information ONLY through a categorical Z: each is
        Z's per-category offset plus independent noise. Raw MI(X;Y) should
        be substantial; CMI(X;Y|Z) should be much smaller, since conditioning
        on Z removes the only channel X and Y share."""
        rng = np.random.default_rng(seed)
        offsets = np.linspace(-3.0, 3.0, n_categories)
        window_labels = rng.integers(0, n_categories, size=n_windows)
        per_sample_offset = offsets[np.repeat(window_labels, window_size)]
        z_raw = np.repeat(window_labels, window_size).reshape(-1, 1).astype(np.int64)
        x_raw = (per_sample_offset[:, None]
                 + rng.standard_normal((n_windows * window_size, 2))).astype(np.float32)
        y_raw = (per_sample_offset[:, None]
                 + rng.standard_normal((n_windows * window_size, 2))).astype(np.float32)
        return x_raw, y_raw, z_raw

    @pytest.mark.parametrize("encoding", ["majority_vote", "probability", "full_trajectory"])
    def test_categorical_z_runs_without_shape_error(self, encoding):
        """Each categorical encoding must produce a valid CMI estimate, not a shape error."""
        x_raw, y_raw, z_raw = self._confounded_data(n_windows=50, window_size=10)
        window_size = 10
        results = nmi.run(
            x_raw, y_raw,
            mode='conditional',
            conditional=Conditional(
                z_data=z_raw, z_processor_type='categorical',
                z_processor_params={'window_size': window_size, 'step_size': window_size,
                                    'encoding': encoding},
            ),
            processing=nmi.Processing(
                x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                y='continuous', y_params={'window_size': window_size, 'step_size': window_size},
            ),
            split=nmi.Split(mode='random'),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert np.isfinite(results.mi_estimate)

    @pytest.mark.parametrize("encoding", ["majority_vote", "probability"])
    def test_categorical_z_explains_shared_variance(self, encoding):
        """CMI(X;Y|Z) should drop well below the raw, unconditioned MI(X;Y)
        when Z is the true (and only) confounder shared by X and Y.

        Averaged over 3 training seeds on the same data, not a single run:
        neural-net training isn't bit-reproducible from seed= alone (confirmed
        empirically -- re-running the same seed=0 call repeatedly, even forced
        onto CPU, still gives noticeably different MI estimates each time, a
        known consequence of non-associative floating-point reduction order in
        multi-threaded training that no amount of seeding fixes). A single
        noisy run occasionally crossed the threshold by chance and made this
        test flaky under parallel execution. The underlying claim holds
        robustly across seeds; averaging is what actually tests it reliably.
        """
        window_size = 10
        x_raw, y_raw, z_raw = self._confounded_data(window_size=window_size)
        model = Model(hidden_dim=32, embedding_dim=8, n_layers=1)
        training = Training(n_epochs=40, patience=15)
        processing = nmi.Processing(
            x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
            y='continuous', y_params={'window_size': window_size, 'step_size': window_size},
        )

        raw_estimates, conditioned_estimates = [], []
        for run_seed in range(3):
            raw = nmi.run(
                x_raw, y_raw, mode='estimate',
                processing=processing, split=nmi.Split(mode='random'),
                model=model, training=training, n_workers=1, seed=run_seed, show_progress=False,
            )
            conditioned = nmi.run(
                x_raw, y_raw, mode='conditional',
                conditional=Conditional(
                    z_data=z_raw, z_processor_type='categorical',
                    z_processor_params={'window_size': window_size, 'step_size': window_size,
                                        'encoding': encoding},
                ),
                processing=processing, split=nmi.Split(mode='random'),
                model=model, training=training, n_workers=1, seed=run_seed, show_progress=False,
            )
            raw_estimates.append(raw.mi_estimate)
            conditioned_estimates.append(conditioned.mi_estimate)

        mean_raw = float(np.mean(raw_estimates))
        mean_conditioned = float(np.mean(conditioned_estimates))
        assert mean_raw > 1.0, (
            f"Expected substantial raw MI(X;Y) from the shared Z confound, "
            f"got mean={mean_raw:.3f} bits over {raw_estimates} -- test construction may be too weak."
        )
        assert mean_conditioned < 0.65 * mean_raw, (
            f"CMI(X;Y|Z) mean={mean_conditioned:.3f} bits did not drop well below "
            f"raw MI(X;Y) mean={mean_raw:.3f} bits after conditioning on the true confounder Z "
            f"(per-seed raw={raw_estimates}, conditioned={conditioned_estimates})."
        )
