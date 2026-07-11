# tests/test_dimensionality.py
"""Tests for run_dimensionality_analysis — index split and embedding history helpers."""
import warnings
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from neural_mi.analysis.dimensionality import (
    run_dimensionality_analysis,
    _extract_embedding_history,
    _strip_embeddings,
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
        'participation_ratio': 2.0,
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': 0}
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': 0}
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': 0}
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': 0}
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': i}
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': 0}
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
            {'train_mi': 0.5, 'test_mi': 0.5, 'participation_ratio': 2.0, 'split_id': 0}
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
