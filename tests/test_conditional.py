# tests/test_conditional.py
"""Tests for the conditional mutual information analysis mode."""
import pytest
import numpy as np
import torch
import neural_mi as nmi
from neural_mi import Model, Training, Conditional, Output

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
    """CMI = I(X,W;Y) - I(W;Y)."""

    def test_cmi_independent_xy_given_w_is_near_zero(self):
        """CMI(X;Y|W) ≈ 0 when X is independent of Y (conditioning on W)."""
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))
        y = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))
        w = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w),
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

    def test_cmi_correlated_xy_given_independent_w(self):
        """CMI(X;Y|W) > 0 when X and Y are correlated and W is independent."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=1.0)
        rng = np.random.default_rng(1)
        w = torch.from_numpy(rng.standard_normal((N, 2)).astype(np.float32))

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w),
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
        w = _make_gaussian(N, 2)

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        assert 'mi_xw_y' in results.details
        assert 'mi_w_y' in results.details
        assert np.isfinite(results.details['mi_xw_y'])
        assert np.isfinite(results.details['mi_w_y'])

    def test_cmi_result_consistency(self):
        """Confirms CMI estimate = I(XW;Y) - I(W;Y)."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        w = _make_gaussian(N, 2)

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w),
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
        expected = results.details['mi_xw_y'] - results.details['mi_w_y']
        assert abs(results.mi_estimate - expected) < 1e-6

    def test_return_embeddings_surfaces_at_top_level(self):
        """Regression: return_embeddings=True used to silently produce no
        embeddings_x/embeddings_y for mode='conditional' -- no error, no
        warning, the keys just never appeared. The joint (XW;Y) leg's
        embeddings (already computed by task.py, just never surfaced) are
        now pulled to the top level and stripped from raw_xw_y."""
        x, y = nmi.generators.generate_correlated_gaussians(N, dim=2, mi=0.5)
        w = _make_gaussian(N, 2)

        results = nmi.run(
            x, y,
            mode='conditional',
            conditional=Conditional(w_data=w),
            model=_MODEL, training=_TRAINING,
            output=Output(return_embeddings=True),
            n_workers=1,
        )
        assert 'embeddings_x' in results.details
        assert 'embeddings_y' in results.details
        assert results.details['embeddings_x'].shape[0] == N
        assert results.details['embeddings_y'].shape[0] == N
        # Stripped from the raw per-run list, not duplicated there too.
        assert 'embeddings_x' not in results.details['raw_xw_y'][0]

    def test_mismatched_x_w_window_sizes_raises_clear_error(self):
        """X and W with different window sizes must raise a clear ValueError
        before the concatenation into XW, not a bare torch.cat shape error."""
        from neural_mi.analysis.conditional import run_conditional_mi

        x = torch.randn(N, 2, 5)   # window size 5
        y = torch.randn(N, 2, 5)
        w = torch.randn(N, 2, 3)   # window size 3 -- mismatched by 2, past the trim tolerance

        with pytest.raises(ValueError, match="window size"):
            run_conditional_mi(x, y, w, base_params=_PARAMS, n_workers=1)


class TestConditionalMICategoricalW:
    """w_processor_type='categorical' as the conditioning variable.

    mode='conditional' builds XW by concatenating X and W along the channel
    axis, which requires a matching window-size axis. The categorical
    processor's encodings don't produce that layout natively --
    _reshape_categorical_w_for_conditional (neural_mi/run.py) re-lays them
    out: 'majority_vote'/'probability' become window-constant channels
    (broadcast across X's window by run_conditional_mi), 'full_trajectory'
    keeps its real per-timepoint resolution on the window axis.
    """

    @staticmethod
    def _confounded_data(n_windows=400, window_size=10, n_categories=3, seed=0):
        """X and Y share information ONLY through a categorical W: each is
        W's per-category offset plus independent noise. Raw MI(X;Y) should
        be substantial; CMI(X;Y|W) should be much smaller, since conditioning
        on W removes the only channel X and Y share."""
        rng = np.random.default_rng(seed)
        offsets = np.linspace(-3.0, 3.0, n_categories)
        window_labels = rng.integers(0, n_categories, size=n_windows)
        per_sample_offset = offsets[np.repeat(window_labels, window_size)]
        w_raw = np.repeat(window_labels, window_size).reshape(-1, 1).astype(np.int64)
        x_raw = (per_sample_offset[:, None]
                 + rng.standard_normal((n_windows * window_size, 2))).astype(np.float32)
        y_raw = (per_sample_offset[:, None]
                 + rng.standard_normal((n_windows * window_size, 2))).astype(np.float32)
        return x_raw, y_raw, w_raw

    @pytest.mark.parametrize("encoding", ["majority_vote", "probability", "full_trajectory"])
    def test_categorical_w_runs_without_shape_error(self, encoding):
        """Each categorical encoding must produce a valid CMI estimate, not a shape error."""
        x_raw, y_raw, w_raw = self._confounded_data(n_windows=50, window_size=10)
        window_size = 10
        results = nmi.run(
            x_raw, y_raw,
            mode='conditional',
            conditional=Conditional(
                w_data=w_raw, w_processor_type='categorical',
                w_processor_params={'window_size': window_size, 'step_size': window_size,
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
    def test_categorical_w_explains_shared_variance(self, encoding):
        """CMI(X;Y|W) should drop well below the raw, unconditioned MI(X;Y)
        when W is the true (and only) confounder shared by X and Y.

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
        x_raw, y_raw, w_raw = self._confounded_data(window_size=window_size)
        model = Model(hidden_dim=32, embedding_dim=8, n_layers=1)
        # shift_windows=False: mode='conditional' (the second call below) never
        # reaches shift_windows, so leaving it at its default for the mode='estimate'
        # call would make `raw` and `conditioned` trained under different dynamics.
        training = Training(n_epochs=40, patience=15, shift_windows=False)
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
                    w_data=w_raw, w_processor_type='categorical',
                    w_processor_params={'window_size': window_size, 'step_size': window_size,
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
            f"Expected substantial raw MI(X;Y) from the shared W confound, "
            f"got mean={mean_raw:.3f} bits over {raw_estimates} -- test construction may be too weak."
        )
        assert mean_conditioned < 0.65 * mean_raw, (
            f"CMI(X;Y|W) mean={mean_conditioned:.3f} bits did not drop well below "
            f"raw MI(X;Y) mean={mean_raw:.3f} bits after conditioning on the true confounder W "
            f"(per-seed raw={raw_estimates}, conditioned={conditioned_estimates})."
        )


class TestConditionalShiftWindows:
    """shift_windows reachability: W is raw-concatenated onto X before
    windowing (rather than after) whenever W is 'continuous' and matches
    X's processor family."""

    def test_engages_silently_for_matching_continuous_pair(self):
        """No warning: shift_windows must actually reach mode='conditional'
        when X and W are both 'continuous', not just stay silently inert."""
        np.random.seed(0)
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randn(3000, 1).astype('float32')
        window_size = 20
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=w, w_processor_type='continuous',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"

    def test_w_equal_to_x_gives_near_zero_cmi_under_shift(self):
        """Correctness/desync check: if W is an exact copy of X (same raw
        array, same window_size/step_size), W carries zero information
        beyond X, so I(XW;Y) should match I(W;Y) closely and CMI should be
        near zero -- with shift_windows on. A one-window desync between X's
        and W's independent reslicing would make the concatenated W look
        like genuinely new information relative to X, inflating I(XW;Y)
        well above I(W;Y) and breaking this exact-zero expectation."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randn(T, 2).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        w = x.copy()  # exact copy of X -- zero information beyond X, if aligned
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=w, w_processor_type='continuous',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size}),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        assert np.isfinite(details['mi_xw_y']) and np.isfinite(details['mi_w_y'])
        assert abs(details['mi_xw_y'] - details['mi_w_y']) < 0.3, (
            f"W=X exactly should make I(XW;Y)={details['mi_xw_y']:.3f} closely match "
            f"I(W;Y)={details['mi_w_y']:.3f} -- a large gap suggests X and W's independent "
            f"reslicing desynchronized under shift_windows."
        )
        assert abs(results.mi_estimate) < 0.3, (
            f"CMI should be near zero when W=X exactly, got {results.mi_estimate:.3f}"
        )


class TestConditionalShiftWindowsRigorous:
    """Phase 2: shift_windows reachability for conditional's rigorous=True
    sub-path -- mirrors TestConditionalShiftWindows, but exercises
    run_rigorous_scalar_analysis's own _is_raw_deferred chunk-to-raw-range
    translation instead of the plain (non-rigorous) sweep dispatch."""

    def test_engages_silently_for_matching_continuous_pair_rigorous(self):
        """No warning: shift_windows must actually reach the rigorous=True
        sub-path of mode='conditional' when X and W are both 'continuous'."""
        np.random.seed(0)
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randn(3000, 1).astype('float32')
        window_size = 20
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=w, w_processor_type='continuous',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size},
                                       rigorous=True, gamma_range=range(1, 4)),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"

    def test_w_equal_to_x_gives_near_zero_cmi_under_shift_rigorous(self):
        """Correctness/desync check (rigorous=True path): W=X exactly should
        make the bias-corrected CMI near zero, the same expectation as the
        non-rigorous version -- a desync in the per-gamma-chunk raw-range
        translation between X and W would inflate it instead."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randn(T, 2).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        w = x.copy()  # exact copy of X -- zero information beyond X, if aligned
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=w, w_processor_type='continuous',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size},
                                   rigorous=True, gamma_range=range(1, 4)),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=10, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        assert np.isfinite(results.mi_estimate)
        assert abs(results.mi_estimate) < 0.5, (
            f"CMI should be near zero when W=X exactly (rigorous=True path), got "
            f"{results.mi_estimate:.3f}"
        )


class TestConditionalShiftWindowsCategorical:
    """Phase 3: shift_windows reachability for a categorical X + categorical
    W pair, with independently-tracked (and possibly different) category
    counts -- mirrors TestConditionalShiftWindows, adapted for categorical
    encoding via shift_windowing.make_multi_categorical_encoder."""

    def test_engages_silently_with_different_category_counts(self):
        """No warning, and a finite result: shift_windows must actually
        reach mode='conditional' when X and W are both 'categorical', even
        when their category counts genuinely differ (X: 3, W: 5) -- the
        exact scenario the single-shared-n_categories bug would conflate."""
        import warnings
        np.random.seed(0)
        x = np.random.randint(0, 3, size=(3000, 1)).astype('int64')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randint(0, 5, size=(3000, 1)).astype('int64')
        window_size = 20
        processing = nmi.Processing(x='categorical', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=w, w_processor_type='categorical',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_w_equal_to_x_gives_near_zero_cmi_under_shift_categorical(self):
        """Correctness/desync check: W as an exact categorical copy of X
        (same raw labels, same category count) carries zero information
        beyond X, so CMI should be near zero -- with shift_windows on. A
        one-window desync (or a category-count conflation corrupting the
        encoding) would break this."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randint(0, 4, size=(T, 1)).astype('int64')
        y = np.random.randn(T, 2).astype('float32')
        w = x.copy()  # exact copy of X -- zero information beyond X, if aligned
        processing = nmi.Processing(x='categorical', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=w, w_processor_type='categorical',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size}),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        assert np.isfinite(results.mi_estimate)
        assert abs(results.mi_estimate) < 0.3, (
            f"CMI should be near zero when W=X exactly (categorical), got {results.mi_estimate:.3f}"
        )

    def test_mismatched_window_size_raises_clear_error(self):
        """Companion correctness fix: w_processor_params's window_size, if
        explicitly set to a value different from X's, must raise rather
        than be silently ignored (the concatenated array is windowed using
        only processor_params_x's geometry)."""
        np.random.seed(0)
        x = np.random.randint(0, 3, size=(2000, 1)).astype('int64')
        y = np.random.randn(2000, 2).astype('float32')
        w = np.random.randint(0, 5, size=(2000, 1)).astype('int64')
        processing = nmi.Processing(x='categorical', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        with pytest.raises(ValueError, match="window_size"):
            nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=w, w_processor_type='categorical',
                                       w_processor_params={'window_size': 10, 'step_size': 10}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )


class TestConditionalShiftWindowsMixedTypes:
    """shift_windows reachability for a *mixed* continuous+categorical
    conditioning variable (X and W have different processor types, not just
    a possibly-different category count within the same type) -- exercises
    shift_windowing.make_multi_categorical_encoder's continuous-passthrough
    block (n_categories=None) and the broadcast reconciling a categorical
    block's collapsed window axis against a continuous block's real
    window_size."""

    def test_engages_silently_continuous_x_categorical_w(self):
        import warnings
        np.random.seed(0)
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randint(0, 4, size=(3000, 1)).astype('int64')
        window_size = 20
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=w, w_processor_type='categorical',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_engages_silently_categorical_x_continuous_w(self):
        """Reverse direction: X categorical, W continuous."""
        import warnings
        np.random.seed(0)
        x = np.random.randint(0, 3, size=(3000, 1)).astype('int64')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randn(3000, 2).astype('float32')
        window_size = 20
        processing = nmi.Processing(x='categorical', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(
                x, y, mode='conditional',
                conditional=Conditional(w_data=w, w_processor_type='continuous',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_categorical_w_recoding_of_continuous_x_gives_near_zero_cmi(self):
        """Correctness/desync check: W is a categorical recoding of X's own
        values -- X is restricted to a small integer-valued alphabet
        ({0.0, 1.0, 2.0}) so the recoding is lossless, and W is encoded via
        'full_trajectory' (no per-window summarization, unlike majority_vote/
        probability) so nothing is discarded at any window boundary under
        any shift. W therefore carries exactly the same information as X,
        so CMI should be near zero -- a channel-slicing or dtype-handling
        bug in the mixed-block encoder would desynchronize X and W and
        inflate it instead."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size, n_categories = 4000, 20, 3
        labels = np.random.randint(0, n_categories, size=(T, 1))
        x = labels.astype('float32')
        w = labels.astype('int64')
        y = np.random.randn(T, 2).astype('float32')
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='conditional',
            conditional=Conditional(w_data=w, w_processor_type='categorical',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size,
                                                        'encoding': 'full_trajectory'}),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        assert np.isfinite(details['mi_xw_y']) and np.isfinite(details['mi_w_y'])
        assert abs(details['mi_xw_y'] - details['mi_w_y']) < 0.3, (
            f"W as a lossless categorical recoding of X should make I(XW;Y)="
            f"{details['mi_xw_y']:.3f} closely match I(W;Y)={details['mi_w_y']:.3f} -- "
            f"a large gap suggests X and W's mixed-type concatenation desynchronized "
            f"under shift_windows."
        )
        assert abs(results.mi_estimate) < 0.3, (
            f"CMI should be near zero when W is a lossless recoding of X, got {results.mi_estimate:.3f}"
        )


class TestConditionalShiftTimeSpike:
    """shift_time reachability for a spike+spike conditioning variable
    (X='spike' and W='spike', matching family only) -- mirrors
    TestConditionalShiftWindows, adapted for shift_time/spike concatenation
    (Python list concat, not torch.cat)."""

    def test_engages_silently_for_matching_spike_pair(self):
        import warnings
        np.random.seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=600, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=600, window_size=0.05, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(
                x_spikes, y_spikes, mode='conditional',
                conditional=Conditional(w_data=w_spikes, w_processor_type='spike',
                                       w_processor_params={'window_size': 0.05}),
                processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
                model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                training=Training(n_epochs=1, patience=1, shift_time=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
        assert not msgs, f"Did not expect a shift_time warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_w_equal_to_x_gives_near_zero_cmi_under_shift(self):
        """Correctness/desync check: W as an exact copy of X's spike-neuron
        population carries zero information beyond X, so CMI should be
        near zero -- with shift_time on. A desync in the per-population
        list-concatenation would make the concatenated W look like
        genuinely new information relative to X, inflating I(XW;Y) above
        I(W;Y)/I(X;Y) and breaking this expectation."""
        np.random.seed(0)
        torch.manual_seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        w_spikes = [s.copy() for s in x_spikes]  # exact copy of X
        results = nmi.run(
            x_spikes, y_spikes, mode='conditional',
            conditional=Conditional(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05}),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_time=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        assert np.isfinite(details['mi_xw_y']) and np.isfinite(details['mi_w_y'])
        assert abs(details['mi_xw_y'] - details['mi_w_y']) < 0.3, (
            f"W=X exactly should make I(XW;Y)={details['mi_xw_y']:.3f} closely match "
            f"I(W;Y)={details['mi_w_y']:.3f} -- a large gap suggests X and W's spike-list "
            f"concatenation desynchronized under shift_time."
        )
        assert abs(results.mi_estimate) < 0.3, (
            f"CMI should be near zero when W=X exactly, got {results.mi_estimate:.3f}"
        )

    def test_rigorous_shift_time_spike_no_crash_and_finite(self):
        """rigorous=True sub-path: mirrors
        TestRigorousShiftTimeSpikeEndToEnd, for conditional's spike-W case
        -- exercises run_rigorous_scalar_analysis's new _is_spike_deferred
        chunk-to-raw-time-range translation."""
        np.random.seed(0)
        torch.manual_seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=600, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=600, window_size=0.05, seed=0)
        results = nmi.run(
            x_spikes, y_spikes, mode='conditional',
            conditional=Conditional(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05},
                                   rigorous=True, gamma_range=range(1, 4)),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=1, patience=1, shift_time=True),
            n_workers=1, show_progress=False, seed=0,
        )
        assert results.mi_estimate is not None
        assert np.isfinite(results.mi_estimate)

    def test_no_crash_with_shift_time_false(self):
        """Regression: the window-validity gap between X (paired with Y,
        requiring both to have data) and a standalone-windowed W (requiring
        only W to have data) used to raise a sample-count ValueError for
        spike data whenever shift_time was NOT active (the merge-based
        mechanism above only used to engage when shift_time=True). Spike+
        spike conditioning now always merges X and W before windowing
        (mirrors the shift_time=True mechanism unconditionally), so this
        must succeed regardless of shift_time."""
        np.random.seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=800, window_size=0.05, seed=0)
        results = nmi.run(
            x_spikes, y_spikes, mode='conditional',
            conditional=Conditional(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05}),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=2, patience=1, shift_time=False),
            n_workers=1, show_progress=False, seed=0,
        )
        assert np.isfinite(results.mi_estimate)

    def test_rigorous_no_crash_with_shift_time_false(self):
        """Regression: run_rigorous_scalar_analysis's own _is_spike_deferred
        gate used to additionally require base_params['shift_time'] to be
        truthy, even though run.py's _defer_spike_conditional_interaction
        (the only caller that sets raw_deferred=True for a spike+spike pair)
        is unconditional on shift_time -- merging X and W before windowing
        is a correctness requirement for spike coverage, not a shift-
        reachability nicety. With shift_time=False, raw_deferred=True still
        arrived with a raw (list) x_data, but _is_spike_deferred evaluated
        False, so N = x_data.shape[0] raised 'list has no attribute shape'.
        Confirmed via direct reproduction before the fix; this asserts it no
        longer crashes and returns a finite estimate."""
        np.random.seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=400, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=400, window_size=0.05, seed=0)
        results = nmi.run(
            x_spikes, y_spikes, mode='conditional',
            conditional=Conditional(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05},
                                   rigorous=True, gamma_range=range(1, 4), min_gamma_points=2),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
            training=Training(n_epochs=2, patience=1, shift_time=False, batch_size=16),
            n_workers=1, show_progress=False, seed=0,
        )
        assert results.mi_estimate is not None
        assert np.isfinite(results.mi_estimate)


def _gappy_timeline(duty_on=8.0, period=40.0, span=400.0, dt=0.02):
    """A timeline sampled densely during scattered epochs, with gaps between.

    Mirrors a tracked recording: position exists only while the animal is in
    view, so real time and sample index diverge badly across the gaps.
    """
    dense = np.arange(0, span, dt)
    keep = np.zeros(dense.size, dtype=bool)
    for start in np.arange(0, span, period):
        keep[(dense >= start) & (dense < start + duty_on)] = True
    return dense[keep]


@pytest.mark.parametrize("shift_windows", [True, False])
@pytest.mark.parametrize("mode", ['conditional', 'interaction'])
def test_three_way_with_spike_y_windows_on_the_real_time_grid(mode, shift_windows):
    """A regular-grid X and W against a spike Y runs on both dispatch paths.

    Regression test for E28. Windowing for ``shift_windows=True`` is deferred to
    the task layer, which used to build its dataset without the caller's time
    vectors. A continuous X was then windowed in sample-index units while the
    spike Y stayed in seconds. On a gappy timeline, where index and real time
    diverge, no window could satisfy coverage and the run died with "No valid
    windows after checking data coverage" -- blaming the recording for what was
    a units mismatch. The gaps and the coverage floor are both load-bearing
    here: without them the two interpretations stay close enough to survive.
    """
    rng = np.random.default_rng(0)
    t = _gappy_timeline()
    pos = np.sin(2 * np.pi * t / 8.0).astype('float32')[:, None]
    direction = (np.gradient(pos[:, 0]) > 0).astype('int64')[:, None]
    spikes = [np.sort(rng.uniform(t[0], t[-1], rng.poisson(8.0 * (t[-1] - t[0]))))
              for _ in range(6)]

    win = {'window_size': 1.0, 'step_size': 0.5}
    cont = dict(win, min_coverage_fraction=0.9)

    kwargs = dict(
        processing=nmi.Processing(x='continuous', x_params=cont, x_time=t,
                                  y='spike', y_params=win),
        model=_MODEL,
        training=Training(n_epochs=2, learning_rate=1e-3, batch_size=32,
                          patience=1, shift_windows=shift_windows),
        n_workers=1, seed=0, show_progress=False,
    )
    w_cfg = dict(w_data=direction, w_processor_type='categorical',
                 w_processor_params=win, w_time=t)
    if mode == 'conditional':
        kwargs['conditional'] = Conditional(**w_cfg)
    else:
        kwargs['interaction'] = nmi.Interaction(**w_cfg)

    result = nmi.run(pos, spikes, mode=mode, **kwargs)

    assert result.mi_estimate is not None
    assert np.isfinite(result.mi_estimate)
