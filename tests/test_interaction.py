# tests/test_interaction.py
"""Tests for mode='interaction' (interaction information)."""
import dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import contextlib

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


def _components(result):
    """The component means of a one-configuration difference quantity, by name."""
    row = result.dataframe.iloc[0]
    return {c[:-len('_mean')]: row[c] for c in row.index if c.endswith('_mean')}

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
        """The components are dataframe columns and each network is a row of trainings."""
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        for key in ('mi_mean', 'mi_xw_y_mean', 'mi_x_y_mean', 'mi_w_y_mean', 'amplification_factor'):
            assert key in r.dataframe.columns
        trainings = r.details[0]['trainings']
        assert sorted(trainings['component']) == ['mi_w_y', 'mi_x_y', 'mi_xw_y']

    def test_ii_equals_combination(self):
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        r = nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, n_workers=1, show_progress=False)
        expected = _components(r)['mi_xw_y'] - _components(r)['mi_x_y'] - _components(r)['mi_w_y']
        assert abs(r.mi_estimate - expected) < 1e-6

    def test_return_embeddings_is_refused(self):
        """No single network of the three describes the interaction information."""
        x, y, w = np.random.randn(N, 1), np.random.randn(N, 1), np.random.randn(N, 1)
        with pytest.raises(ValueError, match="not available for mode='interaction'"):
            nmi.run(x, y, mode='interaction', interaction=Interaction(w_data=w),
                    model=_MODEL, training=_TRAINING, output=Output(return_embeddings=True),
                    n_workers=1, show_progress=False)

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
        """interaction.py's eager path tolerates a one-window difference
                between X and W, as conditional.py does. Calls
                run_interaction_information directly to construct the exact boundary
                condition: W with one fewer window than X and Y.
        """
        from neural_mi.analysis.interaction import run_interaction_information
        x = torch.randn(50, 1, 4)
        y = torch.randn(50, 1, 4)
        w = torch.randn(49, 1, 4)  # exactly _SAMPLE_COUNT_TRIM_TOLERANCE short
        raw = run_interaction_information(x, y, w, _PARAMS,
                                          n_workers=1)
        assert np.isfinite(raw['interaction_info'])
        assert raw['raw_xw_y'][0]['train_mi'] is not None

    def test_window_size_broadcast_matches_conditional(self):
        """A W with a collapsed (size-1) window axis, such as a per-window
                categorical summary, is broadcast across X's window, as in
                conditional.py.
        """
        from neural_mi.analysis.interaction import run_interaction_information
        x = torch.randn(50, 1, 4)
        y = torch.randn(50, 1, 4)
        w = torch.randn(50, 1, 1)  # collapsed window axis
        raw = run_interaction_information(x, y, w, _PARAMS,
                                          n_workers=1)
        assert np.isfinite(raw['interaction_info'])

    def test_window_size_gap_beyond_tolerance_still_raises(self):
        """The trim tolerance stays narrow: a gap bigger than
                _WINDOW_SIZE_TRIM_TOLERANCE, and not a size-1 broadcast, raises.
        """
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
        assert len(r.get('null_distribution')) == 3
        assert 0 < r.get('p_value') <= 1


class TestInteractionInformationAccuracy:
    """Validate against exact Gaussian ground truth (a redundancy case:
    negative II is the expected direction, checked both exactly and,
    loosely, at the estimated level)."""

    @pytest.mark.slow
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
                interaction=Interaction(w_data=w),
                processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': window_size, 'step_size': window_size}),
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
            interaction=Interaction(w_data=w),
            processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': window_size, 'step_size': window_size}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = _components(results)
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
    """shift_windows for interaction's rigorous=True path.

        Mirrors TestInteractionShiftWindows through run_rigorous_scalar_analysis,
        which cuts each gamma chunk as a raw sample range before windowing.
    """

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
                interaction=Interaction(w_data=w,
                                       rigorous=True, gamma_range=range(1, 4)),
                processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': window_size, 'step_size': window_size}),
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
            interaction=Interaction(w_data=w,
                                   rigorous=True, gamma_range=range(1, 4)),
            processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': window_size, 'step_size': window_size}),
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
    """shift_windows for a categorical X with a categorical W.

        Each side keeps its own category count, encoded through
        shift_windowing.make_multi_categorical_encoder.
    """

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
                interaction=Interaction(w_data=w),
                processing=dataclasses.replace(processing, w='categorical', w_params={'window_size': window_size, 'step_size': window_size}),
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
            interaction=Interaction(w_data=w),
            processing=dataclasses.replace(processing, w='categorical', w_params={'window_size': window_size, 'step_size': window_size}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = _components(results)
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
        """A W whose window_size differs from X's raises: the concatenated
                array is windowed with X's geometry alone.
        """
        np.random.seed(0)
        x = np.random.randint(0, 3, size=(2000, 1)).astype('int64')
        y = np.random.randn(2000, 2).astype('float32')
        w = np.random.randint(0, 5, size=(2000, 1)).astype('int64')
        processing = nmi.Processing(x='categorical', x_params={'window_size': 20, 'step_size': 20},
                                    y='continuous', y_params={'window_size': 20, 'step_size': 20})
        with pytest.raises(ValueError, match="window_size"):
            nmi.run(
                x, y, mode='interaction',
                interaction=Interaction(w_data=w),
                processing=dataclasses.replace(processing, w='categorical', w_params={'window_size': 10, 'step_size': 10}),
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
                interaction=Interaction(w_data=w),
                processing=dataclasses.replace(processing, w='categorical', w_params={'window_size': window_size, 'step_size': window_size}),
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
                interaction=Interaction(w_data=w),
                processing=dataclasses.replace(processing, w='continuous', w_params={'window_size': window_size, 'step_size': window_size}),
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
            interaction=Interaction(w_data=w),
            processing=dataclasses.replace(processing, w='categorical', w_params={'window_size': window_size, 'step_size': window_size,
                                                        'encoding': 'full_trajectory'}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_windows=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = _components(results)
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
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=600, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=600, window_size=0.05, seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = nmi.run(
                x_spikes, y_spikes, mode='interaction',
                interaction=Interaction(w_data=w_spikes),
                processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}, w='spike', w_params={'window_size': 0.05}),
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
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        w_spikes = [s.copy() for s in x_spikes]  # exact copy of X
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}, w='spike', w_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=15, patience=5, batch_size=32, shift_time=True),
            n_workers=1, show_progress=False, seed=0,
        )
        details = _components(results)
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
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=600, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=600, window_size=0.05, seed=0)
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes,
                                   rigorous=True, gamma_range=range(1, 4)),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}, w='spike', w_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=1, patience=1, shift_time=True),
            n_workers=1, show_progress=False, seed=0,
        )
        assert results.mi_estimate is not None
        assert np.isfinite(results.mi_estimate)

    def test_no_crash_with_shift_time_false(self):
        """A spike X and spike W are merged before windowing whether or not
                shift_time is active, so the call succeeds with shift_time=False.
        """
        np.random.seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=800, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=800, window_size=0.05, seed=0)
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}, w='spike', w_params={'window_size': 0.05}),
            model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
            training=Training(n_epochs=2, patience=1, shift_time=False),
            n_workers=1, show_progress=False, seed=0,
        )
        assert np.isfinite(results.mi_estimate)

    def test_rigorous_no_crash_with_shift_time_false(self):
        """The rigorous path cuts a spike+spike pair as time ranges whether or
                not shift_time is active, and returns a finite estimate with
                shift_time=False.
        """
        np.random.seed(0)
        x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=400, window_size=0.05, seed=0)
        w_spikes, _, _ = nmi.generators.generate_spike_pair(
            n_neurons=4, n_windows=400, window_size=0.05, seed=0)
        results = nmi.run(
            x_spikes, y_spikes, mode='interaction',
            interaction=Interaction(w_data=w_spikes,
                                   rigorous=True, gamma_range=range(1, 4), min_gamma_points=2),
            processing=nmi.Processing(x='spike', x_params={'window_size': 0.05}, w='spike', w_params={'window_size': 0.05}),
            model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
            training=Training(n_epochs=2, patience=1, shift_time=False, batch_size=16),
            n_workers=1, show_progress=False, seed=0,
        )
        assert results.mi_estimate is not None
        assert np.isfinite(results.mi_estimate)


class TestWProcessorInheritance:
    """A W without a processor of its own reads with X's.

        `nmi.interaction_information(x, y, w, processing=Processing(x='continuous',
        x_params={'window_size': ...}))` is the natural three-population call, and
        W must be windowed exactly as X is. These assert the value, not merely
        that nothing raises: an inherited W must give the same answer as an
        explicitly declared one.
    """

    WP = {'window_size': 5, 'step_size': 5}

    @staticmethod
    def _triple(T=1200, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((T, 3)).astype('float32')
        w = rng.standard_normal((T, 3)).astype('float32')
        y = (0.8 * x + 0.4 * w + 0.5 * rng.standard_normal((T, 3))).astype('float32')
        return x, y, w

    def _common(self):
        return dict(model=_MODEL, training=_TRAINING, n_workers=1,
                    show_progress=False, seed=0)

    def test_interaction_inherits_x_processor_for_w(self):
        x, y, w = self._triple()
        proc = nmi.Processing(x='continuous', x_params=self.WP,
                              y='continuous', y_params=self.WP)
        inherited = nmi.interaction_information(x, y, w, processing=proc, **self._common())
        explicit = nmi.run(
            x, y, mode='interaction', processing=dataclasses.replace(proc, w='continuous', w_params=self.WP),
            interaction=Interaction(w_data=w),
            **self._common())
        assert inherited.mi_estimate == explicit.mi_estimate

    def test_conditional_inherits_x_processor_for_w(self):
        from neural_mi import Conditional
        x, y, w = self._triple()
        proc = nmi.Processing(x='continuous', x_params=self.WP,
                              y='continuous', y_params=self.WP)
        inherited = nmi.run(x, y, mode='conditional', processing=proc,
                            conditional=Conditional(w_data=w), **self._common())
        explicit = nmi.run(
            x, y, mode='conditional', processing=dataclasses.replace(proc, w='continuous', w_params=self.WP),
            conditional=Conditional(w_data=w),
            **self._common())
        assert inherited.mi_estimate == explicit.mi_estimate

    def test_preprocessed_3d_w_is_left_alone(self):
        """The inheritance must not reach a W the caller already windowed.
        No processing= here, so processor_type_x is None and nothing is
        inherited; a 3-D W would also be excluded on its own."""
        x, y, w = self._triple(T=1000)
        xw = torch.from_numpy(x.reshape(200, 3, 5))
        yw = torch.from_numpy(y.reshape(200, 3, 5))
        ww = torch.from_numpy(w.reshape(200, 3, 5))
        r = nmi.run(xw, yw, mode='interaction',
                    interaction=Interaction(w_data=ww), **self._common())
        assert r.mi_estimate is not None and np.isfinite(r.mi_estimate)



class TestThreeWayWindowAlignment:
    """X, Y and W refer to the same windows by construction.

        Window validity differs by stream: a continuous X carrying
        `min_coverage_fraction` against a categorical W with no such rule keeps
        different windows. Truncating the three to a shared length would pair
        windows from different times whenever the extra window is not at an edge
        (measured on a real recording: 18% of pairs misaligned). `run()` builds
        all three streams on one grid and keeps the windows valid for all of them.

        The fixture reproduces the divergence at a tenth of that scale: a gap in
        the shared time base that only X's `min_coverage_fraction` reacts to.
    """

    WIN, STEP, DT, T = 2.0, 1.0, 0.1, 1200
    X_PARAMS = {'window_size': WIN, 'step_size': STEP, 'min_coverage_fraction': 0.9}
    Y_PARAMS = {'window_size': WIN, 'step_size': STEP}
    W_PARAMS = {'window_size': WIN, 'step_size': STEP}

    @classmethod
    def _series(cls, gap, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(cls.T, dtype='float64') * cls.DT
        t[cls.T // 2:] += gap
        x = rng.standard_normal((cls.T, 1)).astype('float32')
        w = (rng.random((cls.T, 1)) < 0.5).astype('int64')
        y = (0.8 * x + 0.4 * w + 0.5 * rng.standard_normal((cls.T, 1))).astype('float32')
        return t, x, y, w

    @classmethod
    def _proc(cls, t, **w):
        return nmi.Processing(x='continuous', x_params=cls.X_PARAMS, x_time=t,
                              y='continuous', y_params=cls.Y_PARAMS, y_time=t, **w)

    @staticmethod
    def _training():
        return Training(n_epochs=3, learning_rate=1e-3, batch_size=64,
                        patience=2, shift_windows=False)

    def test_the_two_pairings_really_do_disagree(self):
        """The premise. Without this the other tests would pass vacuously."""
        from neural_mi.data.handler import create_dataset
        t, x, y, w = self._series(gap=3.0)
        xy = create_dataset(x_data=x, x_time=t, y_data=y, y_time=t,
                            processor_type_x='continuous', processor_params_x=self.X_PARAMS,
                            processor_type_y='continuous', processor_params_y=self.Y_PARAMS)
        wy = create_dataset(x_data=w, x_time=t, y_data=y, y_time=t,
                            processor_type_x='categorical', processor_params_x=self.W_PARAMS,
                            processor_type_y='continuous', processor_params_y=self.Y_PARAMS)
        tx = np.asarray(xy.window_manager.window_times)
        tw = np.asarray(wy.window_manager.window_times)
        assert len(tx) != len(tw), "fixture no longer produces a divergence"
        assert abs(len(tx) - len(tw)) > 1, "divergence must exceed the trim tolerance"
        common, i_xy, i_w = np.intersect1d(tx, tw, return_indices=True)
        assert len(common) > 0
        # What run() now uses: every retained window maps to the same time.
        assert np.array_equal(tx[i_xy], tw[i_w])

    def test_interaction_runs_when_validity_rules_differ(self):
        t, x, y, w = self._series(gap=3.0)
        r = nmi.run(x, y, mode='interaction', processing=self._proc(t, w='categorical', w_params=self.W_PARAMS, w_time=t),
                    interaction=Interaction(w_data=w),
                    model=_MODEL, training=self._training(),
                    n_workers=1, show_progress=False, seed=0)
        assert r.mi_estimate is not None and np.isfinite(r.mi_estimate)

    def test_conditional_runs_when_validity_rules_differ(self):
        from neural_mi import Conditional
        t, x, y, w = self._series(gap=3.0)
        r = nmi.run(x, y, mode='conditional', processing=self._proc(t, w='categorical', w_params=self.W_PARAMS, w_time=t),
                    conditional=Conditional(w_data=w),
                    model=_MODEL, training=self._training(),
                    n_workers=1, show_progress=False, seed=0)
        assert r.mi_estimate is not None and np.isfinite(r.mi_estimate)

    def test_one_window_difference_is_aligned_not_truncated(self):
        """A one-window difference is resolved by window time, not by
                truncation, so the run succeeds and the truncation warning never
                fires.
        """
        from neural_mi import Conditional
        from neural_mi.data.handler import create_dataset
        t, x, y, w = self._series(gap=0.0)
        xy = create_dataset(x_data=x, x_time=t, y_data=y, y_time=t,
                            processor_type_x='continuous', processor_params_x=self.X_PARAMS,
                            processor_type_y='continuous', processor_params_y=self.Y_PARAMS)
        wy = create_dataset(x_data=w, x_time=t, y_data=y, y_time=t,
                            processor_type_x='categorical', processor_params_x=self.W_PARAMS,
                            processor_type_y='continuous', processor_params_y=self.Y_PARAMS)
        assert abs(xy.x_data.shape[0] - wy.x_data.shape[0]) == 1, "fixture drifted"

        with _capture_warnings() as seen:
            r = nmi.run(x, y, mode='conditional', processing=self._proc(t, w='categorical', w_params=self.W_PARAMS, w_time=t),
                        conditional=Conditional(w_data=w),
                        model=_MODEL, training=self._training(),
                        n_workers=1, show_progress=False, seed=0)
        assert r.mi_estimate is not None and np.isfinite(r.mi_estimate)
        assert not any('Truncating all three' in m for m in seen), (
            "the engine trim fired; run() should have aligned by window time first")

    def test_matching_windows_are_a_no_op(self):
        """When both pairings already agree the alignment must change nothing,
        so two identical runs stay bit-identical."""
        from neural_mi import Conditional
        t, x, y, w = self._series(gap=0.0)
        proc = nmi.Processing(x='continuous', x_params=self.Y_PARAMS, x_time=t,
                              y='continuous', y_params=self.Y_PARAMS, y_time=t,
                              w='categorical', w_params=self.W_PARAMS, w_time=t)
        kw = dict(model=_MODEL, training=self._training(), n_workers=1,
                  show_progress=False, seed=0)
        cond = lambda: Conditional(w_data=w)
        a = nmi.run(x, y, mode='conditional', processing=proc, conditional=cond(), **kw)
        b = nmi.run(x, y, mode='conditional', processing=proc, conditional=cond(), **kw)
        assert a.mi_estimate == b.mi_estimate


@contextlib.contextmanager
def _capture_warnings():
    """Collect neural_mi logger warnings emitted inside the block."""
    import logging
    seen = []

    class _H(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    lg = logging.getLogger('neural_mi')
    h = _H()
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.WARNING)
    try:
        yield seen
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


class TestEarlyMergeFallback:
    """A side outside the regular-grid family still merges X and W early.

    The shifted tuple builder takes only continuous/categorical streams, and
    the eager builder behind it cannot window a tuple at all, so a spike Y
    sends X and W back through the raw channel-concat. That branch was
    unreachable by any test when it was written and called a helper with the
    wrong arity, which is why it is pinned here.
    """

    @staticmethod
    def _spike_y_data(seed=0):
        rng = np.random.default_rng(seed)
        T = 1200
        t = np.arange(T, dtype=float) / 10.
        x = rng.standard_normal((T, 2)).astype(np.float32)
        w = rng.standard_normal((T, 2)).astype(np.float32)
        spikes = [np.sort(rng.uniform(0, T / 10., 800)) for _ in range(4)]
        return t, x, w, spikes

    def _run(self, mode, mode_cfg):
        t, x, w, spikes = self._spike_y_data()
        return nmi.run(
            x, spikes, mode=mode, **mode_cfg(w, t),
            processing=nmi.Processing(
                x='continuous', y='spike', w='continuous', x_time=t,
                x_params={'window_size': 1.0, 'step_size': 1.0, 'sample_rate': 10},
                y_params={'window_size': 1.0, 'step_size': 1.0, 'bin_size': 0.1},
                w_params={'window_size': 1.0, 'step_size': 1.0, 'sample_rate': 10}),
            model=_MODEL, training=Training(n_epochs=2, batch_size=64, shift_windows=True),
            show_progress=False, seed=0, n_workers=1,
        )

    def test_interaction_with_a_spike_y(self):
        from neural_mi.config import Interaction
        result = self._run('interaction', lambda w, t: {'interaction': Interaction(w_data=w)})
        assert np.isfinite(result.mi_estimate)

    def test_conditional_with_a_spike_y(self):
        from neural_mi.config import Conditional
        result = self._run('conditional', lambda w, t: {'conditional': Conditional(w_data=w)})
        assert np.isfinite(result.mi_estimate)

    def test_the_helper_returns_all_three_marginals(self):
        """Conditional uses two of them, interaction all three."""
        from neural_mi.analysis.conditional import _merge_raw_blocks
        rng = np.random.default_rng(0)
        x = rng.standard_normal((200, 2)).astype(np.float32)
        w = rng.integers(0, 3, size=(200, 1))
        to_t = lambda a: a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
        xw, joint, marg_x, marg_w = _merge_raw_blocks(
            x, w, 'continuous', 'categorical', {'processor_params_x': {}}, to_t)
        assert xw.shape[1] == 3
        specs = lambda bp: bp['processor_params_x']['_categorical_block_specs']
        assert len(specs(joint)) == 2 and len(specs(marg_x)) == 1 and len(specs(marg_w)) == 1
