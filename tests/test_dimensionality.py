# tests/test_dimensionality.py
"""Tests for run_dimensionality_analysis — index split, embedding history
helpers, and the cross-run stability/near-degeneracy/ceiling-proximity checks
that replaced the old PR-ceiling/noise-injection-ladder machinery."""
import warnings
import pytest
import numpy as np
import torch
from unittest.mock import patch

import pandas as pd

from neural_mi.analysis.dimensionality import (
    run_dimensionality_analysis,
    _extract_embedding_history,
    _strip_embeddings,
    _compute_stability_report,
    _group_adjacent,
    _classify_regime,
    _warn_if_near_ceiling,
)


# ---------------------------------------------------------------------------
# Helpers / minimal mocks
# ---------------------------------------------------------------------------

def _make_x(n=60, c=6, w=None):
    """Return a small float32 Tensor of shape (n, c) or (n, c, w)."""
    if w is None:
        return torch.randn(n, c)
    return torch.randn(n, c, w)


def _minimal_result(n_epochs=3, embed_dim=4, n_tracked=10):
    """Synthetic result dict as produced by a trainer run."""
    row = {
        'train_mi': 0.5,
        'test_mi': 0.5,
        'pr_eig': 2.0, 'pr_singular': 2.0,
        'split_id': 0,
    }
    row['embedding_history_x'] = [
        np.random.randn(n_tracked, embed_dim).astype(np.float32)
        for _ in range(n_epochs)
    ]
    row['embedding_history_y'] = [
        np.random.randn(n_tracked, embed_dim).astype(np.float32)
        for _ in range(n_epochs)
    ]
    return row


# ---------------------------------------------------------------------------
# _extract_embedding_history
# ---------------------------------------------------------------------------

class TestExtractEmbeddingHistory:
    def test_returns_empty_when_no_history(self):
        rows = [{'train_mi': 0.5}, {'train_mi': 0.6}]
        result = _extract_embedding_history(rows)
        assert result == {}

    def test_returns_history_from_last_result(self):
        row0 = {'embedding_history_x': ['a'], 'embedding_history_y': ['b']}
        row1 = {'embedding_history_x': ['c'], 'embedding_history_y': ['d']}
        rows = [row0, row1]
        result = _extract_embedding_history(rows)
        # Should pick row1 (last)
        assert result['embedding_history_x'] == ['c']
        assert result['embedding_history_y'] == ['d']

    def test_returns_first_match_from_reverse(self):
        row0 = {'train_mi': 0.5}
        row1 = {'embedding_history_x': ['x'], 'embedding_history_y': ['y']}
        rows = [row0, row1]
        result = _extract_embedding_history(rows)
        assert result['embedding_history_x'] == ['x']


# ---------------------------------------------------------------------------
# _strip_embeddings
# ---------------------------------------------------------------------------

class TestStripEmbeddings:
    def test_removes_all_embedding_keys(self):
        row = {
            'train_mi': 0.5,
            'embeddings_x': np.zeros((5, 4)),
            'embeddings_y': np.zeros((5, 4)),
            'embeddings_x_rotated': np.zeros((5, 4)),
            'embeddings_y_rotated': np.zeros((5, 4)),
            'embeddings_rotation_singular_values': np.zeros(4),
            'embeddings_rotation_x': np.zeros((4, 4)),
            'embeddings_rotation_y': np.zeros((4, 4)),
            'embedding_history_x': [],
            'embedding_history_y': [],
        }
        _strip_embeddings([row])
        for key in ('embeddings_x', 'embeddings_y', 'embeddings_x_rotated',
                    'embeddings_y_rotated', 'embeddings_rotation_singular_values',
                    'embeddings_rotation_x', 'embeddings_rotation_y',
                    'embedding_history_x', 'embedding_history_y'):
            assert key not in row
        assert 'train_mi' in row  # non-embedding key unaffected

    def test_no_error_when_keys_absent(self):
        row = {'train_mi': 0.5}
        _strip_embeddings([row])  # should not raise


# ---------------------------------------------------------------------------
# run_dimensionality_analysis — index split: input validation
# ---------------------------------------------------------------------------

class TestIndexSplitValidation:
    """Unit tests that exercise only the validation logic — no training required."""

    def test_missing_channel_indices_x_raises(self):
        x = _make_x()
        with pytest.raises(ValueError, match="channel_indices_x"):
            run_dimensionality_analysis(
                x, base_params={'n_epochs': 1}, split_method='index'
            )

    def test_out_of_range_index_raises(self):
        x = _make_x(c=6)
        with pytest.raises(ValueError, match="must be integers in"):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1},
                split_method='index',
                channel_indices_x=[0, 1, 99],  # 99 >= n_channels=6
            )

    def test_all_channels_to_x_raises(self):
        x = _make_x(c=4)
        with pytest.raises(ValueError, match="Y would be empty"):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1},
                split_method='index',
                channel_indices_x=[0, 1, 2, 3],  # all 4 channels
            )

    def test_empty_channel_indices_x_raises(self):
        x = _make_x(c=4)
        with pytest.raises(ValueError, match="X would be empty"):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1},
                split_method='index',
                channel_indices_x=[],
            )

    def test_unknown_split_method_raises(self):
        x = _make_x()
        with pytest.raises(ValueError, match="Unknown split_method"):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1},
                split_method='invalid_method',
            )


# ---------------------------------------------------------------------------
# run_dimensionality_analysis — index split: shared_encoder guard
# ---------------------------------------------------------------------------

class TestIndexSplitSharedEncoderGuard:
    """Verify that unequal channel counts disable shared_encoder with a warning.

    These tests call run_dimensionality_analysis directly (not through run()),
    so base_params is treated as fully user-set (see the user_set_keys
    fallback in run_dimensionality_analysis) -- an explicit
    shared_encoder=True in base_params is respected as user intent, exactly
    like calling run() with model=Model(shared_encoder=True) would be.
    """

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_unequal_channels_disables_shared_encoder(self, mock_dispatch, caplog):
        """shared_encoder=True should be overridden to False when |X| != |Y|."""
        import logging
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=6)
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1, 'shared_encoder': True},
                split_method='index',
                channel_indices_x=[0, 1],  # X=2, Y=4 → unequal
                n_splits=1,
            )
        # Warning is emitted via logger.warning(), not warnings.warn()
        assert any('shared_encoder' in r.message for r in caplog.records), (
            f"Expected a shared_encoder warning in log, got: {[r.message for r in caplog.records]}"
        )
        # The params forwarded to _dispatch_splits should have shared_encoder=False
        call_args = mock_dispatch.call_args[0]  # positional args
        split_tasks = call_args[0]
        _, _, forwarded_params, _, _ = split_tasks[0]
        assert forwarded_params.get('shared_encoder') is False

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_equal_channels_keeps_shared_encoder(self, mock_dispatch):
        """shared_encoder should remain True when both sides have equal channel counts."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1, 'shared_encoder': True},
                split_method='index',
                channel_indices_x=[0, 1, 2],  # X=3, Y=3 → equal
                n_splits=1,
            )
        messages = [str(w.message) for w in caught
                    if 'shared_encoder' in str(w.message)]
        assert not messages, f"Unexpected shared_encoder warning: {messages}"
        call_args = mock_dispatch.call_args[0]
        split_tasks = call_args[0]
        _, _, forwarded_params, _, _ = split_tasks[0]
        # shared_encoder should NOT have been silently overridden
        assert forwarded_params.get('shared_encoder') is True


# ---------------------------------------------------------------------------
# run_dimensionality_analysis — index split: channel slicing
# ---------------------------------------------------------------------------

class TestIndexSplitChannelSlicing:
    """Verify that x_a and x_b contain the right channels."""

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_2d_data_correct_channel_split(self, mock_dispatch):
        """For 2-D data (N, C), x_a[:,i] == x_data[:,channel_indices_x[i]]."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = torch.arange(60, dtype=torch.float32).reshape(10, 6)
        run_dimensionality_analysis(
            x,
            base_params={'n_epochs': 1},
            split_method='index',
            channel_indices_x=[0, 2, 4],
            n_splits=1,
        )
        call_args = mock_dispatch.call_args[0]
        split_tasks = call_args[0]
        x_a, x_b, _, _, _ = split_tasks[0]
        assert x_a.shape == (10, 3)
        assert x_b.shape == (10, 3)
        # Column 0 of x_a must equal column 0 of original x
        np.testing.assert_array_equal(x_a[:, 0].numpy(), x[:, 0].numpy())
        np.testing.assert_array_equal(x_a[:, 1].numpy(), x[:, 2].numpy())
        np.testing.assert_array_equal(x_a[:, 2].numpy(), x[:, 4].numpy())
        # x_b must contain the complement channels: [1, 3, 5]
        np.testing.assert_array_equal(x_b[:, 0].numpy(), x[:, 1].numpy())

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_3d_data_correct_channel_split(self, mock_dispatch):
        """For 3-D data (N, C, W), channel dim is 1."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = torch.randn(10, 6, 8)
        run_dimensionality_analysis(
            x,
            base_params={'n_epochs': 1},
            split_method='index',
            channel_indices_x=[0, 1],
            n_splits=1,
        )
        call_args = mock_dispatch.call_args[0]
        split_tasks = call_args[0]
        x_a, x_b, _, _, _ = split_tasks[0]
        assert x_a.shape == (10, 2, 8)
        assert x_b.shape == (10, 4, 8)

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_n_splits_creates_correct_number_of_tasks(self, mock_dispatch):
        """For index split, n_splits tasks should be dispatched."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': i}
            for i in range(3)
        ]
        x = _make_x(c=6)
        run_dimensionality_analysis(
            x,
            base_params={'n_epochs': 1},
            split_method='index',
            channel_indices_x=[0, 1, 2],
            n_splits=3,
        )
        call_args = mock_dispatch.call_args[0]
        split_tasks = call_args[0]
        assert len(split_tasks) == 3
        # Each task should have a distinct split_id
        split_ids = [t[4] for t in split_tasks]
        assert split_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# run_dimensionality_analysis — embedding_dim / shared_encoder / track_embeddings
# defaults, and the user_set_keys mechanism that makes them work correctly
# whether called through run() or directly.
# ---------------------------------------------------------------------------

class TestDefaults:
    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_embedding_dim_defaults_to_8_when_unset(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1}, split_method='random', n_splits=1)
        _, _, forwarded_params, _, _ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('embedding_dim') == 8

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_embedding_dim_explicit_value_respected(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(
            x, base_params={'n_epochs': 1, 'embedding_dim': 64},
            split_method='random', n_splits=1,
        )
        _, _, forwarded_params, _, _ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('embedding_dim') == 64

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_shared_encoder_true_for_intrinsic_by_default(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1}, split_method='random', n_splits=1)
        _, _, forwarded_params, _, _ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('shared_encoder') is True

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_shared_encoder_false_for_interaction_by_default(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x, y = _make_x(c=4), _make_x(c=4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1}, y_data=y, n_splits=1)
        _, _, forwarded_params, _, _ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('shared_encoder') is False

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_user_set_keys_from_run_distinguishes_explicit_from_schema_default(self, mock_dispatch):
        """The exact bug this mechanism fixes: base_params already contains
        embedding_dim=64 (the schema default, applied by run()'s
        ParameterValidator.apply_defaults() before dispatch) even though the
        user never set it -- only user_set_keys (the pre-defaults key set)
        can tell these apart."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(
            x, base_params={'n_epochs': 1, 'embedding_dim': 64},  # schema-defaulted value
            split_method='random', n_splits=1,
            user_set_keys=set(),  # ...but the user never actually set it
        )
        _, _, forwarded_params, _, _ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('embedding_dim') == 8, (
            "embedding_dim=64 in base_params without being in user_set_keys must be "
            "treated as the schema default, not a user override"
        )

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_track_embeddings_not_forced(self, mock_dispatch):
        """track_embeddings no longer gets a dimensionality-specific override
        (the rotated-embedding extraction this mode relies on doesn't need
        it) -- whatever the caller passed (or didn't) flows through unchanged."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1}, split_method='random', n_splits=1)
        _, _, forwarded_params, _, _ = mock_dispatch.call_args[0][0][0]
        assert 'track_embeddings' not in forwarded_params


# ---------------------------------------------------------------------------
# _compute_stability_report — the core mechanism this mode is built on
# ---------------------------------------------------------------------------

class TestComputeStabilityReport:
    def _make_split(self, seed, n=500, embedding_dim=3, shared=None, strengths=None):
        rng = np.random.default_rng(seed)
        if shared is None:
            shared = np.zeros((n, 0))
        strengths = strengths or [1.0] * embedding_dim
        z = np.zeros((n, embedding_dim))
        for i in range(embedding_dim):
            if i < shared.shape[1]:
                z[:, i] = strengths[i] * shared[:, i] + 0.05 * rng.standard_normal(n)
            else:
                z[:, i] = rng.standard_normal(n)
        return z

    def test_reproducible_direction_is_stable_and_above_floor(self):
        """A direction driven by a shared signal across all 'splits' should be
        flagged stable and individually trustworthy."""
        rng = np.random.default_rng(0)
        shared_signal = rng.standard_normal((500, 1))
        per_split = []
        for seed in (1, 2, 3):
            z = self._make_split(seed, embedding_dim=1, shared=shared_signal, strengths=[2.0])
            per_split.append({'zx_rotated_test': z, 'singular_values': np.array([2.0])})
        report = _compute_stability_report(per_split, stability_threshold=0.7,
                                           degeneracy_ratio_threshold=1.3, min_strength_fraction=0.05)
        assert report['stable_directions'] == [1]
        assert report['n_stable_total'] == 1

    def test_pure_noise_channel_excluded_even_with_spurious_correlation(self):
        """Regression test for the mechanism the whole mode hinges on: a pure
        noise direction with near-zero singular-value strength must NOT be
        reported as stable, even if it happens to show a high cross-run
        correlation by chance (confirmed directly in the validation battery --
        see SOURCE_OF_TRUTH.md's hard_entangled_troublezone result)."""
        rng = np.random.default_rng(42)
        n = 500
        # A shared real signal on rank 1, plus a rank-2 "noise" direction whose
        # values happen to correlate across the two draws purely by chance
        # (fixed seed search would be needed to guarantee this in general, so
        # instead we construct the spurious correlation directly: same tiny,
        # near-zero-strength vector reused with a small independent perturbation).
        shared = rng.standard_normal((n, 1))
        base_noise_direction = rng.standard_normal(n) * 1e-4  # near-zero strength
        per_split = []
        for seed in (1, 2):
            local_rng = np.random.default_rng(seed)
            z = np.zeros((n, 2))
            z[:, 0] = 2.0 * shared[:, 0] + 0.05 * local_rng.standard_normal(n)
            # Same base direction (spurious cross-run correlation) but negligible strength.
            z[:, 1] = base_noise_direction + 1e-6 * local_rng.standard_normal(n)
            per_split.append({'zx_rotated_test': z, 'singular_values': np.array([2.0, 1e-4])})
        report = _compute_stability_report(per_split, stability_threshold=0.7,
                                           degeneracy_ratio_threshold=1.3, min_strength_fraction=0.05)
        assert report['per_rank'][2]['below_noise_floor'] is True
        assert 2 not in report['stable_directions']
        assert report['n_stable_total'] == 1

    def test_unstable_direction_excluded_despite_real_strength(self):
        """A direction with real singular-value strength but that doesn't
        reproduce across splits must be excluded -- the noise-floor gate and
        the correlation gate are independent checks, not one catching for
        the other."""
        rng = np.random.default_rng(0)
        per_split = []
        for seed in (1, 2, 3):
            local_rng = np.random.default_rng(seed)
            z = local_rng.standard_normal((500, 1)) * 1.5  # real strength, but independent per split
            per_split.append({'zx_rotated_test': z, 'singular_values': np.array([1.5])})
        report = _compute_stability_report(per_split, stability_threshold=0.7,
                                           degeneracy_ratio_threshold=1.3, min_strength_fraction=0.05)
        assert report['per_rank'][1]['below_noise_floor'] is False
        assert report['per_rank'][1]['stable'] is False
        assert report['n_stable_total'] == 0

    def test_adjacent_close_strength_ranks_grouped_not_individually_ordered(self):
        """Two reproducible ranks of near-identical strength must be reported
        as a group, not individually ranked."""
        rng = np.random.default_rng(0)
        shared = rng.standard_normal((500, 2))
        per_split = []
        for seed in (1, 2, 3):
            z = self._make_split(seed, embedding_dim=2, shared=shared, strengths=[2.0, 1.9])
            per_split.append({'zx_rotated_test': z, 'singular_values': np.array([2.0, 1.9])})
        report = _compute_stability_report(per_split, stability_threshold=0.7,
                                           degeneracy_ratio_threshold=1.3, min_strength_fraction=0.05)
        assert report['stable_directions'] == []
        assert report['stable_but_degenerate_groups'] == [[1, 2]]
        assert report['n_stable_total'] == 2


class TestGroupAdjacent:
    def test_empty(self):
        assert _group_adjacent([]) == []

    def test_single(self):
        assert _group_adjacent([3]) == [[3]]

    def test_contiguous_and_separate_runs(self):
        assert _group_adjacent([1, 2, 5, 6, 7, 10]) == [[1, 2], [5, 6, 7], [10]]


# ---------------------------------------------------------------------------
# Ceiling-proximity diagnostic (lightweight, standalone -- not a remediation)
# ---------------------------------------------------------------------------

class TestCeilingProximity:
    def test_classify_regime_pinned_detached_collapsed(self):
        assert _classify_regime(9.5, 10.0, 1.0, 0.5) == ('pinned', False)
        assert _classify_regime(5.0, 10.0, 1.0, 0.5) == ('detached', True)
        assert _classify_regime(0.1, 10.0, 1.0, 0.5) == ('collapsed', False)
        assert _classify_regime(float('nan'), 10.0, 1.0, 0.5) == ('collapsed', False)

    def test_warn_if_near_ceiling_warns_when_pinned(self):
        eval_size = 100.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({'test_mi': [0.95 * ceiling] * 3, 'eval_size': [eval_size] * 3})
        with pytest.warns(UserWarning, match="near its evaluation ceiling"):
            _warn_if_near_ceiling(df, ceiling_mi_fraction=0.85)

    def test_warn_if_near_ceiling_silent_when_detached(self):
        eval_size = 1000.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({'test_mi': [0.3 * ceiling] * 3, 'eval_size': [eval_size] * 3})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_near_ceiling(df, ceiling_mi_fraction=0.85)
        assert not any('evaluation ceiling' in str(w.message) for w in caught)

    def test_ceiling_mi_fraction_is_configurable(self):
        eval_size = 100.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({'test_mi': [0.95 * ceiling] * 3, 'eval_size': [eval_size] * 3})
        with pytest.warns(UserWarning, match="near its evaluation ceiling"):
            _warn_if_near_ceiling(df, ceiling_mi_fraction=0.85)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_near_ceiling(df, ceiling_mi_fraction=0.99)
        assert not any('evaluation ceiling' in str(w.message) for w in caught)

    def test_missing_columns_no_error(self):
        _warn_if_near_ceiling(pd.DataFrame({'foo': [1, 2, 3]}))  # should not raise


class TestAnalysisWorkflowDoesNotMutateCallerDict:
    """Regression test: AnalysisWorkflow.__init__ used to assign
    self.base_params = base_params (same reference) then .update() it,
    mutating the caller's dict in place."""

    def test_base_params_not_mutated(self):
        from neural_mi.analysis.rigorous import AnalysisWorkflow
        original = {'n_epochs': 5}
        original_copy = dict(original)
        x = torch.randn(20, 3, 4)
        y = torch.randn(20, 3, 4)
        AnalysisWorkflow(x, y, original)
        assert original == original_copy, (
            f"caller's base_params was mutated: {original} != {original_copy}"
        )


# ---------------------------------------------------------------------------
# _n_samples_for_shared_split -- shift_windows reachability
# ---------------------------------------------------------------------------

class TestNSamplesForSharedSplit:
    """When windowing is deferred (raw 2-D x_data + shift_windows=True), the
    shared train/test split must be computed over the shift-invariant window
    count, not the raw sample count -- otherwise it produces indices that
    don't correspond to any real window in each split's actual dataset."""

    def test_unchanged_for_already_windowed_data(self):
        from neural_mi.analysis.dimensionality import _n_samples_for_shared_split
        x = torch.randn(50, 4, 8)  # already windowed (N, C, W)
        n = _n_samples_for_shared_split(x, None, {'shift_windows': True})
        assert n == 50

    def test_unchanged_for_raw_data_when_shift_windows_off(self):
        from neural_mi.analysis.dimensionality import _n_samples_for_shared_split
        x = torch.randn(3000, 4)  # raw (T, C), shift not requested
        n = _n_samples_for_shared_split(x, None, {'shift_windows': False})
        assert n == 3000

    def test_intrinsic_mode_matches_safe_n_windows(self):
        from neural_mi.analysis.dimensionality import _n_samples_for_shared_split
        from neural_mi.data.shift_windowing import safe_n_windows
        x = torch.randn(3000, 4)
        params = {
            'shift_windows': True,
            'processor_params_x': {'window_size': 20, 'step_size': 20},
        }
        n = _n_samples_for_shared_split(x, None, params)
        assert n == safe_n_windows(3000, 20, 20)
        assert n < 3000  # sanity: genuinely different from the raw count

    def test_interaction_mode_matches_min_of_paired_shifter(self):
        from neural_mi.analysis.dimensionality import _n_samples_for_shared_split
        from neural_mi.data.shift_windowing import PairedWindowShifter
        x = torch.randn(3000, 4)
        y = torch.randn(2000, 3)  # different length -> different n_windows
        params = {
            'shift_windows': True,
            'processor_type_x': 'continuous', 'processor_type_y': 'continuous',
            'processor_params_x': {'window_size': 20, 'step_size': 20},
        }
        n = _n_samples_for_shared_split(x, y, params)
        expected = PairedWindowShifter(x, y, 20, 20).n_windows
        assert n == expected
        assert n < 3000


class TestDimensionalityShiftWindowsEndToEnd:
    """A real (un-mocked) run confirming the shared split stays valid once
    each split independently, really windows its own raw data -- if
    _n_samples_for_shared_split computed the wrong (too-large) count, the
    shared test/train indices would be out of bounds for each split's
    actual (much smaller) windowed dataset and this would crash."""

    def test_intrinsic_random_split_shift_windows_no_crash(self):
        import neural_mi as nmi
        from neural_mi import Training, Dimensionality

        torch.manual_seed(0)
        np.random.seed(0)
        x = np.random.randn(3000, 4).astype('float32')
        proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20})
        results = nmi.run(
            x, mode='dimensionality', processing=proc,
            training=Training(n_epochs=2, batch_size=16, shift_windows=True),
            dimensionality=Dimensionality(split_method='random', n_splits=2),
            n_workers=1, show_progress=False, seed=0,
        )
        df = results.details['raw_results']
        assert len(df) == 2
        assert np.all(np.isfinite(df['train_mi'].values))
