# tests/test_dimensionality.py
"""Tests for run_dimensionality_analysis — index split and embedding history helpers."""
import warnings
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

import pandas as pd

from neural_mi.analysis.dimensionality import (
    run_dimensionality_analysis,
    _extract_embedding_history,
    _strip_embeddings,
    _report_dimensionality_reliability,
    _warn_if_ladder_not_plateaued,
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
            'embedding_history_x': [],
            'embedding_history_y': [],
        }
        _strip_embeddings([row])
        assert 'embeddings_x' not in row
        assert 'embeddings_y' not in row
        assert 'embedding_history_x' not in row
        assert 'embedding_history_y' not in row
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
    """Verify that unequal channel counts disable shared_encoder with a warning."""

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
# run_dimensionality_analysis — track_embeddings default in dimensionality mode
# ---------------------------------------------------------------------------

class TestTrackEmbeddingsDefault:
    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_track_embeddings_defaults_to_512(self, mock_dispatch):
        """track_embeddings should auto-default to 512 when not in base_params."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(
            x, base_params={'n_epochs': 1}, split_method='random', n_splits=1
        )
        call_args = mock_dispatch.call_args[0]
        split_tasks = call_args[0]
        _, _, forwarded_params, _, _ = split_tasks[0]
        assert forwarded_params.get('track_embeddings') == 512

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_track_embeddings_false_respected(self, mock_dispatch):
        """Explicit track_embeddings=False in base_params should not be overridden."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        x = _make_x(c=4)
        run_dimensionality_analysis(
            x,
            base_params={'n_epochs': 1, 'track_embeddings': False},
            split_method='random',
            n_splits=1,
        )
        call_args = mock_dispatch.call_args[0]
        split_tasks = call_args[0]
        _, _, forwarded_params, _, _ = split_tasks[0]
        assert forwarded_params.get('track_embeddings') is False


# ---------------------------------------------------------------------------
# P4: three separate dimensionality-reliability conditions
# ---------------------------------------------------------------------------

class TestDimensionalityReliabilityConditions:
    """Regression tests for the three distinct reliability conditions (P4):
    (1) ceiling corruption, (2) no spectral gap, (3) no plateau across the
    noise sweep. Deliberately kept as separate signals, not one is_reliable
    flag."""

    def test_condition1_ceiling_corruption_warns(self):
        """Scalar MI near log(eval_size) must warn that the PR readout is
        unreliable, regardless of the PR value itself."""
        eval_size = 100.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({
            'test_mi': [0.95 * ceiling] * 3,
            'eval_size': [eval_size] * 3,
            'pr_singular': [3.0, 3.2, 2.9],  # low PR -- would look "fine" without the MI check
        })
        with pytest.warns(UserWarning, match="near its evaluation ceiling"):
            _report_dimensionality_reliability(df, {'embedding_dim': 64})

    def test_condition2_no_spectral_gap_is_informational_not_a_failure(self, caplog):
        """High PR with MI safely below ceiling is a genuine finding -- must
        be an informational note, not a warning/failure."""
        import logging
        eval_size = 1000.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({
            'test_mi': [0.3 * ceiling] * 3,   # far from the ceiling
            'eval_size': [eval_size] * 3,
            'pr_singular': [40.0, 41.0, 39.0],  # 40/64 = 62.5%: high but not truncated (<80%)
        })
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _report_dimensionality_reliability(df, {'embedding_dim': 64})
        assert not any(issubclass(w.category, UserWarning) for w in caught), \
            "high-but-untruncated PR must not raise a UserWarning"
        assert any('distributed across many dimensions' in r.message for r in caplog.records)

    def test_embedding_ceiling_truncation_still_warns(self):
        """The pre-existing embedding-capacity truncation check must still fire
        (independent of the new MI-ceiling condition)."""
        eval_size = 1000.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({
            'test_mi': [0.3 * ceiling] * 3,
            'eval_size': [eval_size] * 3,
            'pr_singular': [55.0, 56.0, 54.0],  # 55/64 = 86%: past the 80% truncation threshold
        })
        with pytest.warns(UserWarning, match="embedding dimension ceiling"):
            _report_dimensionality_reliability(df, {'embedding_dim': 64})

    def test_no_warning_when_mi_low_and_pr_low(self):
        """The ordinary, unremarkable case: MI well below ceiling, PR well
        below embedding_dim -- no warnings, no reliability messaging at all."""
        eval_size = 1000.0
        ceiling = np.log(eval_size)
        df = pd.DataFrame({
            'test_mi': [0.3 * ceiling] * 3,
            'eval_size': [eval_size] * 3,
            'pr_singular': [3.0, 3.1, 2.9],
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _report_dimensionality_reliability(df, {'embedding_dim': 64})
        assert len(caught) == 0

    def test_condition3_no_plateau_warns(self):
        """PR that keeps drifting across detached sigma_add rungs must warn
        that the ladder hasn't stabilized."""
        ladder = pd.DataFrame({
            'sigma_add': [1.0, 2.0, 3.0],
            'regime': ['detached', 'detached', 'detached'],
            'pr_singular_mean': [3.0, 8.0, 15.0],  # clearly still climbing, not stable
        })
        with pytest.warns(UserWarning, match="has not.*plateaued"):
            _warn_if_ladder_not_plateaued(ladder)

    def test_condition3_plateaued_ladder_does_not_warn(self):
        """A stable PR across detached rungs must not trigger the plateau warning."""
        ladder = pd.DataFrame({
            'sigma_add': [1.0, 2.0, 3.0],
            'regime': ['detached', 'detached', 'detached'],
            'pr_singular_mean': [5.0, 5.1, 4.9],  # stable
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_ladder_not_plateaued(ladder)
        assert not any('plateaued' in str(w.message) for w in caught)

    def test_condition3_single_detached_rung_skipped(self):
        """Fewer than two detached rungs can't demonstrate a plateau either
        way -- must not warn."""
        ladder = pd.DataFrame({
            'sigma_add': [1.0, 2.0, 3.0],
            'regime': ['pinned', 'detached', 'collapsed'],
            'pr_singular_mean': [60.0, 5.0, 0.1],
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_ladder_not_plateaued(ladder)
        assert not any('plateaued' in str(w.message) for w in caught)


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
