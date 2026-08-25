# tests/test_interaction.py
"""Tests for mode='interaction' (interaction information)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Interaction, Output

_MODEL = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=3, learning_rate=1e-3, batch_size=64, patience=2)

# Full base_params (dict kept for the engine-level run_interaction_information tests
# that actually train a model, unlike the error-path-only tests elsewhere in this
# file, which never reach build_critic and so can get away with a handful of keys).
from neural_mi.defaults import BASE_PARAMS_SCHEMA as _SCHEMA
_PARAMS = {k: v['default'] for k, v in _SCHEMA.items() if 'default' in v}
_PARAMS.update({
    'n_epochs': 3, 'learning_rate': 1e-3, 'batch_size': 64,
    'patience': 2, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
})

N = 300  # samples


class _StaticTripleOracle:
    """Z ~ N(0,1); X = a*Z + eps_x; Y = b*Z + eps_y; W = c*Z + eps_w, IID
    across samples. All three driven by one shared cause -> X and W are
    redundant proxies for Y's information, so exact II is negative (the
    standard "redundancy" signature)."""

    def __init__(self, a=1.0, b=1.0, c=1.0, sx=0.5, sy=0.5, sw=0.5):
        self._loadings = {'x': a, 'y': b, 'w': c}
        self._noises = {'x': sx, 'y': sy, 'w': sw}

    def _cov_entry(self, vi, vj):
        v = self._loadings[vi] * self._loadings[vj]
        if vi == vj:
            v += self._noises[vi] ** 2
        return v

    def _cov_matrix(self, vars_):
        n = len(vars_)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = self._cov_entry(vars_[i], vars_[j])
        return M

    def mi_bits(self, vars_a, vars_b):
        joint = self._cov_matrix(vars_a + vars_b)
        ca = self._cov_matrix(vars_a)
        cb = self._cov_matrix(vars_b)
        _, ld_j = np.linalg.slogdet(joint)
        _, ld_a = np.linalg.slogdet(ca)
        _, ld_b = np.linalg.slogdet(cb)
        return float((ld_a + ld_b - ld_j) / (2 * np.log(2)))

    def ii_exact(self):
        mi_xw_y = self.mi_bits(['x', 'w'], ['y'])
        mi_x_y = self.mi_bits(['x'], ['y'])
        mi_w_y = self.mi_bits(['w'], ['y'])
        return mi_xw_y - mi_x_y - mi_w_y, mi_xw_y, mi_x_y, mi_w_y

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=n)
        x = (self._loadings['x'] * z + self._noises['x'] * rng.normal(size=n)).astype(np.float32)
        y = (self._loadings['y'] * z + self._noises['y'] * rng.normal(size=n)).astype(np.float32)
        w = (self._loadings['w'] * z + self._noises['w'] * rng.normal(size=n)).astype(np.float32)
        return x.reshape(-1, 1), y.reshape(-1, 1), w.reshape(-1, 1)


class TestInteractionInformationPlumbing:
    """Fast shape/wiring checks, no accuracy claims."""

    def test_returns_results_object(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        assert isinstance(r, nmi.Results)
        assert r.mode == 'interaction'
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)

    def test_details_keys(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        for key in ('interaction_info', 'mi_xw_y', 'mi_x_y', 'mi_w_y'):
            assert key in r.details

    def test_ii_equals_combination(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        expected = r.details['mi_xw_y'] - r.details['mi_x_y'] - r.details['mi_w_y']
        assert abs(r.mi_estimate - expected) < 1e-6

    def test_return_embeddings_surfaces_at_top_level(self):
        """Regression: return_embeddings=True used to silently produce no
        embeddings_x/embeddings_y for mode='interaction'. The joint (X,W;Y)
        leg's embeddings are now pulled to the top level."""
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, output=Output(return_embeddings=True),
                    n_workers=1, show_progress=False)
        assert 'embeddings_x' in r.details
        assert 'embeddings_y' in r.details
        assert r.details['embeddings_x'].shape[0] == N
        assert 'embeddings_x' not in r.details['raw_xw_y'][0]

    def test_missing_w_data_raises(self):
        x, y = np.random.randn(N, 1), np.random.randn(N, 1)
        with pytest.raises((ValueError, TypeError)):
            nmi.run(x, y, mode='interaction', model=_MODEL, training=_TRAINING, n_workers=1)

    def test_mismatched_window_size_raises(self):
        x = np.random.randn(N, 1, 4)
        y = np.random.randn(N, 1, 4)
        w = np.random.randn(N, 1, 6)  # different window size than x
        with pytest.raises(ValueError):
            nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                   model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)

    def test_sample_count_trim_tolerance_matches_conditional(self):
        """Regression: interaction.py's non-raw_deferred (eager) path used
        to hard-raise on any x/w window-count mismatch, unlike
        conditional.py's identical W-paired-with-Y construction (see
        run.py), which tolerates a 1-window boundary difference between two
        separately-built create_dataset calls. Calls run_interaction_information
        directly (bypassing nmi.run's orchestration) to construct the exact
        boundary condition: W with one fewer window than X/Y."""
        from neural_mi.analysis.interaction import run_interaction_information
        x = torch.randn(50, 1, 4)
        y = torch.randn(50, 1, 4)
        w = torch.randn(49, 1, 4)  # exactly _SAMPLE_COUNT_TRIM_TOLERANCE short
        raw = run_interaction_information(x, y, w, _PARAMS,
                                          n_workers=1)
        assert np.isfinite(raw['interaction_info'])
        assert raw['raw_xw_y'][0]['train_mi'] is not None

    def test_window_size_broadcast_matches_conditional(self):
        """Regression: a W with a collapsed (size-1) window axis -- e.g. a
        per-window categorical summary -- used to hard-raise in
        interaction.py's eager path instead of broadcasting across X's
        window like conditional.py already does."""
        from neural_mi.analysis.interaction import run_interaction_information
        x = torch.randn(50, 1, 4)
        y = torch.randn(50, 1, 4)
        w = torch.randn(50, 1, 1)  # collapsed window axis
        raw = run_interaction_information(x, y, w, _PARAMS,
                                          n_workers=1)
        assert np.isfinite(raw['interaction_info'])

    def test_window_size_gap_beyond_tolerance_still_raises(self):
        """The trim tolerance must stay narrow: a gap bigger than
        _WINDOW_SIZE_TRIM_TOLERANCE (and not a size-1 broadcast case) is
        still a hard error, unchanged from before this fix."""
        from neural_mi.analysis.interaction import run_interaction_information
        x = torch.randn(50, 1, 4)
        y = torch.randn(50, 1, 4)
        w = torch.randn(50, 1, 6)
        with pytest.raises(ValueError):
            run_interaction_information(x, y, w, _PARAMS, n_workers=1)

    def test_summary_runs(self, capsys):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        r.summary()
        captured = capsys.readouterr()
        assert 'II' in captured.out

    def test_plot_runs(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        ax = r.plot(show=False)
        assert ax is not None
        plt.close('all')

    def test_rigorous_runs(self):
        x, y, w = np.random.randn(600, 1), np.random.randn(600, 1), np.random.randn(600, 1)
        r = nmi.run(x, y, mode='interaction',
                   interaction=Interaction(w_data=w, rigorous=True, gamma_range=range(1, 4)),
                   model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        assert r.mi_estimate is not None
        assert np.isfinite(r.mi_estimate)

    def test_permutation_test_runs(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False,
                    permutation_test=True, n_permutations=3)
        assert 'null_distribution' in r.details
        assert len(r.details['null_distribution']) == 3


class TestInteractionInformationAccuracy:
    """Validate against exact Gaussian ground truth (a redundancy case:
    negative II is the expected direction, checked both exactly and,
    loosely, at the estimated level)."""

    def test_redundancy_gives_negative_ii(self):
        oracle = _StaticTripleOracle()
        ii_exact, mi_xw_y_exact, mi_x_y_exact, mi_w_y_exact = oracle.ii_exact()
        assert ii_exact < 0  # exact-math sanity check on the oracle itself

        x, y, w = oracle.sample(4000, seed=1)
        training = Training(n_epochs=30, learning_rate=1e-3, batch_size=128, patience=8)
        r = nmi.run(
            torch.from_numpy(x), torch.from_numpy(y), mode='interaction',
            interaction=Interaction(w_data=torch.from_numpy(w)),
            model=_MODEL, training=training, n_workers=1, seed=0, show_progress=False,
        )
        assert r.mi_estimate < 0.3  # not a tight match (II is a 3-term combination,
        # the same "small residual" fragility discussed in THEORY.md applies), but the
        # redundancy signature (clearly not a large positive synergy value) should hold.


class TestInteractionShiftWindows:
    """shift_windows reachability: W is raw-concatenated onto X before
    windowing (rather than after) whenever W is 'continuous' and matches
    X's processor family."""

    def test_engages_silently_for_matching_continuous_pair(self):
        """No warning: shift_windows must actually reach mode='interaction'
        when X and W are both 'continuous', not just stay silently inert."""
        import warnings
        np.random.seed(0)
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randn(3000, 1).astype('float32')
        window_size = 20
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(
                x, y, mode='interaction',
                interaction=Interaction(w_data=w, w_processor_type='continuous',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w_msg.message) for w_msg in caught if 'shift_windows' in str(w_msg.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"

    def test_w_equal_to_x_gives_consistent_components_under_shift(self):
        """Correctness/desync check: if W is an exact copy of X (same raw
        array, same window_size/step_size), W adds zero information beyond
        X, so I(X,W;Y) and I(W;Y) should both closely match I(X;Y) -- with
        shift_windows on. A one-window desync between X's and W's
        independent reslicing would make the concatenated W look like
        genuinely new information relative to X, inflating I(X,W;Y) above
        I(X;Y)/I(W;Y) and breaking this expectation."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randn(T, 2).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        w = x.copy()  # exact copy of X -- zero information beyond X, if aligned
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='interaction',
            interaction=Interaction(w_data=w, w_processor_type='continuous',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size}),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        for a, b, name_a, name_b in [
            (details['mi_xw_y'], details['mi_x_y'], 'I(X,W;Y)', 'I(X;Y)'),
            (details['mi_w_y'], details['mi_x_y'], 'I(W;Y)', 'I(X;Y)'),
        ]:
            assert np.isfinite(a) and np.isfinite(b)
            assert abs(a - b) < 0.3, (
                f"W=X exactly should make {name_a}={a:.3f} closely match {name_b}={b:.3f} -- "
                f"a large gap suggests X and W's independent reslicing desynchronized "
                f"under shift_windows."
            )


class TestInteractionShiftWindowsRigorous:
    """Phase 2: shift_windows reachability for interaction's rigorous=True
    sub-path -- mirrors TestInteractionShiftWindows, but exercises
    run_rigorous_scalar_analysis's own _is_raw_deferred chunk-to-raw-range
    translation instead of the plain (non-rigorous) sweep dispatch."""

    def test_engages_silently_for_matching_continuous_pair_rigorous(self):
        """No warning: shift_windows must actually reach the rigorous=True
        sub-path of mode='interaction' when X and W are both 'continuous'."""
        import warnings
        np.random.seed(0)
        x = np.random.randn(3000, 2).astype('float32')
        y = np.random.randn(3000, 2).astype('float32')
        w = np.random.randn(3000, 1).astype('float32')
        window_size = 20
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(
                x, y, mode='interaction',
                interaction=Interaction(w_data=w, w_processor_type='continuous',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size},
                                       rigorous=True, gamma_range=range(1, 4)),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w_msg.message) for w_msg in caught if 'shift_windows' in str(w_msg.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"

    def test_w_equal_to_x_gives_consistent_components_under_shift_rigorous(self):
        """Correctness/desync check (rigorous=True path): W=X exactly should
        make I(X,W;Y) and I(W;Y) both closely match I(X;Y), the same
        expectation as the non-rigorous version."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randn(T, 2).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        w = x.copy()  # exact copy of X -- zero information beyond X, if aligned
        processing = nmi.Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='interaction',
            interaction=Interaction(w_data=w, w_processor_type='continuous',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size},
                                   rigorous=True, gamma_range=range(1, 4)),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=10, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        assert np.isfinite(results.mi_estimate)
        assert abs(results.mi_estimate) < 0.5, (
            f"II should be near zero when W=X exactly (rigorous=True path), got "
            f"{results.mi_estimate:.3f}"
        )


class TestInteractionShiftWindowsCategorical:
    """Phase 3: shift_windows reachability for a categorical X + categorical
    W pair, with independently-tracked (and possibly different) category
    counts -- mirrors TestInteractionShiftWindows, adapted for categorical
    encoding via shift_windowing.make_multi_categorical_encoder."""

    def test_engages_silently_with_different_category_counts(self):
        """No warning, and a finite result: shift_windows must actually
        reach mode='interaction' when X and W are both 'categorical', even
        when their category counts genuinely differ (X: 3, W: 5)."""
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
                x, y, mode='interaction',
                interaction=Interaction(w_data=w, w_processor_type='categorical',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w_msg.message) for w_msg in caught if 'shift_windows' in str(w_msg.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_w_equal_to_x_gives_consistent_components_under_shift_categorical(self):
        """Correctness/desync check: W as an exact categorical copy of X
        should make I(X,W;Y) and I(W;Y) both closely match I(X;Y) -- with
        shift_windows on."""
        np.random.seed(0)
        torch.manual_seed(0)
        T, window_size = 4000, 20
        x = np.random.randint(0, 4, size=(T, 1)).astype('int64')
        y = np.random.randn(T, 2).astype('float32')
        w = x.copy()  # exact copy of X -- zero information beyond X, if aligned
        processing = nmi.Processing(x='categorical', x_params={'window_size': window_size, 'step_size': window_size},
                                    y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(
            x, y, mode='interaction',
            interaction=Interaction(w_data=w, w_processor_type='categorical',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size}),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        for a, b, name_a, name_b in [
            (details['mi_xw_y'], details['mi_x_y'], 'I(X,W;Y)', 'I(X;Y)'),
            (details['mi_w_y'], details['mi_x_y'], 'I(W;Y)', 'I(X;Y)'),
        ]:
            assert np.isfinite(a) and np.isfinite(b)
            assert abs(a - b) < 0.3, (
                f"W=X exactly should make {name_a}={a:.3f} closely match {name_b}={b:.3f} "
                f"(categorical) -- a large gap suggests desync or category-count conflation."
            )

    def test_mismatched_window_size_raises_clear_error(self):
        """Companion correctness fix: w_processor_params's window_size, if
        explicitly set to a value different from X's, must raise rather
        than be silently ignored."""
        np.random.seed(0)
        x = np.random.randint(0, 3, size=(2000, 1)).astype('int64')
        y = np.random.randn(2000, 2).astype('float32')
        w = np.random.randint(0, 5, size=(2000, 1)).astype('int64')
        processing = nmi.Processing(x='categorical', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        with pytest.raises(ValueError, match="window_size"):
            nmi.run(
                x, y, mode='interaction',
                interaction=Interaction(w_data=w, w_processor_type='categorical',
                                       w_processor_params={'window_size': 10, 'step_size': 10}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )


class TestInteractionShiftWindowsMixedTypes:
    """shift_windows reachability for a *mixed* continuous+categorical W
    (X and W have different processor types) -- exercises
    shift_windowing.make_multi_categorical_encoder's continuous-passthrough
    block and the broadcast reconciling a categorical block's collapsed
    window axis against a continuous block's real window_size."""

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
                x, y, mode='interaction',
                interaction=Interaction(w_data=w, w_processor_type='categorical',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w_msg.message) for w_msg in caught if 'shift_windows' in str(w_msg.message)]
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
                x, y, mode='interaction',
                interaction=Interaction(w_data=w, w_processor_type='continuous',
                                       w_processor_params={'window_size': window_size, 'step_size': window_size}),
                processing=processing,
                training=Training(n_epochs=1, patience=1, shift_windows=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w_msg.message) for w_msg in caught if 'shift_windows' in str(w_msg.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_categorical_w_recoding_of_continuous_x_gives_consistent_components(self):
        """Correctness/desync check: W is a categorical recoding of X's own
        values -- X is restricted to a small integer-valued alphabet
        ({0.0, 1.0, 2.0}) so the recoding is lossless, and W is encoded via
        'full_trajectory' (no per-window summarization) so nothing is
        discarded at any window boundary under any shift. W therefore
        carries exactly the same information as X, so I(X,W;Y) and I(W;Y)
        should both closely match I(X;Y)."""
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
            x, y, mode='interaction',
            interaction=Interaction(w_data=w, w_processor_type='categorical',
                                   w_processor_params={'window_size': window_size, 'step_size': window_size,
                                                        'encoding': 'full_trajectory'}),
            processing=processing,
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        for a, b, name_a, name_b in [
            (details['mi_xw_y'], details['mi_x_y'], 'I(X,W;Y)', 'I(X;Y)'),
            (details['mi_w_y'], details['mi_x_y'], 'I(W;Y)', 'I(X;Y)'),
        ]:
            assert np.isfinite(a) and np.isfinite(b)
            assert abs(a - b) < 0.3, (
                f"W as a lossless categorical recoding of X should make {name_a}={a:.3f} "
                f"closely match {name_b}={b:.3f} -- a large gap suggests X and W's "
                f"mixed-type concatenation desynchronized under shift_windows."
            )


class TestInteractionShiftTimeSpike:
    """shift_time reachability for a spike+spike W (X='spike' and W='spike',
    matching family only) -- mirrors TestInteractionShiftWindows, adapted
    for shift_time/spike concatenation (Python list concat, not torch.cat)."""

    def test_engages_silently_for_matching_spike_pair(self):
        import warnings
        np.random.seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=30.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        w_spikes, _ = nmi.generators.generate_correlated_spike_trains(
            n_neurons=4, duration=30.0, firing_rate=8.0, delay=0.01, jitter=0.002
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(
                x_spikes, y_spikes, mode='interaction',
                interaction=Interaction(w_data=w_spikes, w_processor_type='spike',
                                       w_processor_params={'window_size': 0.05}),
                processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
                model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
                training=Training(n_epochs=1, patience=1, shift_time=True),
                n_workers=1, show_progress=False, seed=0,
            )
        msgs = [str(w_msg.message) for w_msg in caught if 'shift_time' in str(w_msg.message)]
        assert not msgs, f"Did not expect a shift_time warning; got: {msgs}"
        assert np.isfinite(r.mi_estimate)

    def test_w_equal_to_x_gives_consistent_components_under_shift(self):
        """Correctness/desync check: W as an exact copy of X's spike-neuron
        population should make I(X,W;Y) and I(W;Y) both closely match
        I(X;Y) -- with shift_time on."""
        np.random.seed(0)
        torch.manual_seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=40.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        w_spikes = [s.copy() for s in x_spikes]  # exact copy of X
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05}),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_time=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = results.details
        for a, b, name_a, name_b in [
            (details['mi_xw_y'], details['mi_x_y'], 'I(X,W;Y)', 'I(X;Y)'),
            (details['mi_w_y'], details['mi_x_y'], 'I(W;Y)', 'I(X;Y)'),
        ]:
            assert np.isfinite(a) and np.isfinite(b)
            assert abs(a - b) < 0.3, (
                f"W=X exactly should make {name_a}={a:.3f} closely match {name_b}={b:.3f} -- "
                f"a large gap suggests X and W's spike-list concatenation desynchronized "
                f"under shift_time."
            )

    def test_rigorous_shift_time_spike_no_crash_and_finite(self):
        """rigorous=True sub-path: exercises run_rigorous_scalar_analysis's
        new _is_spike_deferred chunk-to-raw-time-range translation."""
        np.random.seed(0)
        torch.manual_seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=30.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        w_spikes, _ = nmi.generators.generate_correlated_spike_trains(
            n_neurons=4, duration=30.0, firing_rate=8.0, delay=0.01, jitter=0.002
        )
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes, w_processor_type='spike',
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
        """Regression: mirrors test_conditional.py's identical fix -- spike+
        spike interaction's W used to be windowed standalone (paired only
        with itself), requiring only "W has data" instead of X's own
        "X has data AND Y has data", producing a sample-count mismatch large
        enough to raise whenever shift_time was NOT active. Now always
        merges X and W before windowing (matching family), regardless of
        shift_time."""
        np.random.seed(0)
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=40.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        w_spikes, _ = nmi.generators.generate_correlated_spike_trains(
            n_neurons=4, duration=40.0, firing_rate=8.0, delay=0.01, jitter=0.002
        )
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes, w_processor_type='spike',
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
        x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
            n_neurons=5, duration=20.0, firing_rate=10.0, delay=0.01, jitter=0.002
        )
        w_spikes, _ = nmi.generators.generate_correlated_spike_trains(
            n_neurons=4, duration=20.0, firing_rate=8.0, delay=0.01, jitter=0.002
        )
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes, w_processor_type='spike',
                                   w_processor_params={'window_size': 0.05},
                                   rigorous=True, gamma_range=range(1, 4), min_gamma_points=2),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}),
            model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
            training=Training(n_epochs=2, patience=1, shift_time=False, batch_size=16),
            n_workers=1, show_progress=False, seed=0,
        )
        assert results.mi_estimate is not None
        assert np.isfinite(results.mi_estimate)
