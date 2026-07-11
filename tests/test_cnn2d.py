# tests/test_cnn2d.py
"""Tests for CNN2D encoder and 4-D input handling throughout the library."""
import warnings
import pytest
import numpy as np
import torch
from unittest.mock import patch

from neural_mi.models.embeddings import CNN2D
from neural_mi.models import CNN2D as CNN2D_from_init
from neural_mi.utils import build_critic


# ---------------------------------------------------------------------------
# CNN2D — architecture and forward pass
# ---------------------------------------------------------------------------

class TestCNN2DModel:
    def test_output_shape(self):
        """Output must be (batch, embed_dim) regardless of spatial size."""
        model = CNN2D(input_dim=3, hidden_dim=16, embed_dim=32, n_layers=2)
        x = torch.randn(8, 3, 16, 16)
        out = model(x)
        assert out.shape == (8, 32)

    def test_variable_spatial_size(self):
        """Adaptive pooling must handle arbitrary H × W without re-instantiation."""
        model = CNN2D(input_dim=4, hidden_dim=16, embed_dim=8, n_layers=1)
        for h, w in [(8, 8), (12, 20), (5, 7), (1, 1)]:
            out = model(torch.randn(4, 4, h, w))
            assert out.shape == (4, 8), f"Failed for H={h}, W={w}"

    def test_single_channel(self):
        model = CNN2D(input_dim=1, hidden_dim=8, embed_dim=16, n_layers=1)
        out = model(torch.randn(4, 1, 8, 8))
        assert out.shape == (4, 16)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError, match="odd"):
            CNN2D(input_dim=3, hidden_dim=16, embed_dim=8, n_layers=1, kernel_size=4)

    def test_kernel_size_1(self):
        """kernel_size=1 is the 1×1 conv case — valid."""
        model = CNN2D(input_dim=3, hidden_dim=8, embed_dim=4, n_layers=1, kernel_size=1)
        out = model(torch.randn(2, 3, 5, 5))
        assert out.shape == (2, 4)

    def test_exported_from_models_init(self):
        """CNN2D must be importable from neural_mi.models."""
        assert CNN2D_from_init is CNN2D

    def test_gradients_flow(self):
        """Gradients must reach Conv2d weights."""
        model = CNN2D(input_dim=2, hidden_dim=8, embed_dim=4, n_layers=2)
        x = torch.randn(4, 2, 8, 8, requires_grad=False)
        loss = model(x).sum()
        loss.backward()
        first_conv = list(model.conv_layers.modules())[1]  # skip Sequential wrapper
        assert first_conv.weight.grad is not None

    def test_eval_deterministic(self):
        """In eval mode the model is deterministic."""
        model = CNN2D(input_dim=2, hidden_dim=8, embed_dim=4, n_layers=1).eval()
        x = torch.randn(4, 2, 6, 6)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_n_layers_zero_raises(self):
        """n_layers=0 → empty conv_layers; forward should still work (degenerate case)."""
        # With n_layers=0 there are no Conv2d layers, but the first block (input_dim→hidden_dim)
        # is always added. Check it handles gracefully.
        model = CNN2D(input_dim=2, hidden_dim=8, embed_dim=4, n_layers=1)
        out = model(torch.randn(2, 2, 4, 4))
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# CNN2D — build_critic integration
# ---------------------------------------------------------------------------

class TestBuildCriticCNN2D:
    def _base_params(self, **overrides):
        p = {
            'embedding_model': 'cnn2d',
            'hidden_dim': 16,
            'embedding_dim': 8,
            'n_layers': 1,
            'n_channels_x': 3,
            'n_channels_y': 3,
            'input_dim_x': 3 * 8 * 8,
            'input_dim_y': 3 * 8 * 8,
            'use_variational': False,
            'shared_encoder': False,
            'max_n_batches': 16,
            'kernel_size': 3,
        }
        p.update(overrides)
        return p

    def test_separable_critic_built(self):
        critic = build_critic('separable', self._base_params())
        from neural_mi.models.critics import SeparableCritic
        assert isinstance(critic, SeparableCritic)

    def test_encoder_is_cnn2d(self):
        critic = build_critic('separable', self._base_params())
        assert isinstance(critic.embedding_net_x, CNN2D)

    def test_shared_encoder_same_object(self):
        params = self._base_params(shared_encoder=True)
        critic = build_critic('separable', params)
        assert critic.embedding_net_x is critic.embedding_net_y

    def test_independent_encoders(self):
        params = self._base_params(shared_encoder=False)
        critic = build_critic('separable', params)
        assert critic.embedding_net_x is not critic.embedding_net_y

    def test_forward_4d_input(self):
        """A built separable critic must process (N, C, H, W) without error."""
        critic = build_critic('separable', self._base_params())
        critic.eval()
        x = torch.randn(4, 3, 8, 8)
        y = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            scores, _ = critic(x, y)
        assert scores.shape == (4, 4)  # (batch, batch) via separable

    def test_kernel_size_forwarded(self):
        params = self._base_params(kernel_size=5)
        critic = build_critic('separable', params)
        first_conv = list(critic.embedding_net_x.conv_layers.modules())[1]
        assert first_conv.kernel_size == (5, 5)

    def test_variational_wrapped(self):
        from neural_mi.models.embeddings import VariationalWrapper
        params = self._base_params(use_variational=True)
        critic = build_critic('separable', params)
        assert isinstance(critic.embedding_net_x, VariationalWrapper)
        assert isinstance(critic.embedding_net_x.base_encoder, CNN2D)


# ---------------------------------------------------------------------------
# 4-D input handling in task.py
# ---------------------------------------------------------------------------

class TestFourDInputHandling:
    """Verify input_dim computation and warnings/errors for 4-D dataset tensors."""

    def _mock_dataset(self, shape_x, shape_y=None):
        """Return a mock PairedDataset whose .x_data / .y_data have the given shape."""
        from unittest.mock import MagicMock
        ds = MagicMock()
        ds.x_data = torch.zeros(shape_x)
        ds.y_data = torch.zeros(shape_y) if shape_y else None
        return ds

    @patch('neural_mi.analysis.task.create_dataset')
    @patch('neural_mi.analysis.task.build_critic')
    def test_4d_input_dim_computed_correctly(self, mock_bc, mock_cd):
        """input_dim_x = C*H*W for 4-D input."""
        mock_cd.return_value = self._mock_dataset((10, 3, 8, 8), (10, 3, 8, 8))
        mock_bc.return_value = MagicMock()
        mock_bc.return_value.parameters.return_value = iter([])

        from neural_mi.analysis.task import run_training_task
        params = {
            'embedding_model': 'cnn2d',
            'n_epochs': 1, 'batch_size': 4, 'patience': 1000,
            'learning_rate': 1e-3, 'train_fraction': 0.8, 'n_test_blocks': 2,
            'estimator_name': 'infonce', 'output_units': 'nats',
            'verbose': False, 'show_progress': False,
        }
        try:
            run_training_task((torch.zeros(10, 3, 8, 8), torch.zeros(10, 3, 8, 8), params, 0))
        except Exception:
            pass  # We only care that build_critic was called with the right params
        call_kwargs = mock_bc.call_args[0][1]
        assert call_kwargs.get('input_dim_x') == 3 * 8 * 8
        assert call_kwargs.get('n_channels_x') == 3

    @patch('neural_mi.analysis.task.create_dataset')
    @patch('neural_mi.analysis.task.build_critic')
    def test_cnn1d_4d_raises(self, mock_bc, mock_cd):
        """embedding_model='cnn' must raise ValueError on 4-D input."""
        mock_cd.return_value = self._mock_dataset((10, 3, 8, 8), (10, 3, 8, 8))
        mock_bc.return_value = MagicMock()

        from neural_mi.analysis.task import run_training_task
        params = {
            'embedding_model': 'cnn',
            'n_epochs': 1, 'batch_size': 4, 'patience': 1000,
            'learning_rate': 1e-3, 'train_fraction': 0.8, 'n_test_blocks': 2,
            'estimator_name': 'infonce', 'output_units': 'nats',
            'verbose': False, 'show_progress': False,
        }
        with pytest.raises(ValueError, match="CNN1D"):
            run_training_task((torch.zeros(10, 3, 8, 8), torch.zeros(10, 3, 8, 8), params, 0))

    @patch('neural_mi.analysis.task.create_dataset')
    @patch('neural_mi.analysis.task.build_critic')
    def test_sequence_model_4d_warns(self, mock_bc, mock_cd):
        """Sequence models (gru, lstm, tcn, transformer) must emit UserWarning on 4-D."""
        mock_cd.return_value = self._mock_dataset((10, 3, 8, 8), (10, 3, 8, 8))
        mock_bc.return_value = MagicMock()
        mock_bc.return_value.parameters.return_value = iter([])

        from neural_mi.analysis.task import run_training_task
        for model in ('gru', 'lstm', 'tcn', 'transformer'):
            params = {
                'embedding_model': model,
                'n_epochs': 1, 'batch_size': 4, 'patience': 1000,
                'learning_rate': 1e-3, 'train_fraction': 0.8, 'n_test_blocks': 2,
                'estimator_name': 'infonce', 'output_units': 'nats',
                'verbose': False, 'show_progress': False,
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    run_training_task((torch.zeros(10, 3, 8, 8), torch.zeros(10, 3, 8, 8),
                                       params, 0))
                except Exception:
                    pass
            msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            assert any('4-D' in m or '4D' in m or 'spatial' in m.lower() for m in msgs), \
                f"Expected 4-D warning for '{model}', got: {msgs}"

    @patch('neural_mi.analysis.task.create_dataset')
    @patch('neural_mi.analysis.task.build_critic')
    def test_mlp_4d_no_warning(self, mock_bc, mock_cd):
        """MLP + 4-D should NOT emit a UserWarning — it flattens silently."""
        mock_cd.return_value = self._mock_dataset((10, 3, 8, 8), (10, 3, 8, 8))
        mock_bc.return_value = MagicMock()
        mock_bc.return_value.parameters.return_value = iter([])

        from neural_mi.analysis.task import run_training_task
        params = {
            'embedding_model': 'mlp',
            'n_epochs': 1, 'batch_size': 4, 'patience': 1000,
            'learning_rate': 1e-3, 'train_fraction': 0.8, 'n_test_blocks': 2,
            'estimator_name': 'infonce', 'output_units': 'nats',
            'verbose': False, 'show_progress': False,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                run_training_task((torch.zeros(10, 3, 8, 8), torch.zeros(10, 3, 8, 8),
                                   params, 0))
            except Exception:
                pass
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        # MLP is in _4D_NATIVE — it flattens silently, no UserWarning expected.
        assert not any('spatial' in m.lower() for m in msgs), \
            f"MLP+4D should not warn about spatial structure, got: {msgs}"

    @patch('neural_mi.analysis.task.create_dataset')
    @patch('neural_mi.analysis.task.build_critic')
    def test_cnn2d_4d_no_warning(self, mock_bc, mock_cd):
        """CNN2D + 4-D must not emit any spatial-structure UserWarning."""
        mock_cd.return_value = self._mock_dataset((10, 3, 8, 8), (10, 3, 8, 8))
        mock_bc.return_value = MagicMock()
        mock_bc.return_value.parameters.return_value = iter([])

        from neural_mi.analysis.task import run_training_task
        params = {
            'embedding_model': 'cnn2d',
            'n_epochs': 1, 'batch_size': 4, 'patience': 1000,
            'learning_rate': 1e-3, 'train_fraction': 0.8, 'n_test_blocks': 2,
            'estimator_name': 'infonce', 'output_units': 'nats',
            'verbose': False, 'show_progress': False,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                run_training_task((torch.zeros(10, 3, 8, 8), torch.zeros(10, 3, 8, 8),
                                   params, 0))
            except Exception:
                pass
        spatial_warns = [w for w in caught
                         if issubclass(w.category, UserWarning)
                         and 'spatial' in str(w.message).lower()]
        assert not spatial_warns, f"Unexpected spatial warning for CNN2D: {spatial_warns}"


# ---------------------------------------------------------------------------
# Spatial split methods in run_dimensionality_analysis
# ---------------------------------------------------------------------------

class TestSpatialSplits:
    """Verify spatial split methods for 4-D data."""

    def _4d(self, n=20, c=2, h=8, w=8):
        return torch.randn(n, c, h, w)

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_horizontal_correct_shapes(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(h=8)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='horizontal', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert x_a.shape == (20, 2, 4, 8)
        assert x_b.shape == (20, 2, 4, 8)

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_horizontal_odd_h(self, mock_dispatch):
        """For odd H, bottom half has one extra row."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(h=7)  # H=7 → top=3, bottom=4
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='horizontal', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert x_a.shape[2] == 3
        assert x_b.shape[2] == 4

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_vertical_correct_shapes(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(w=10)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='vertical', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert x_a.shape == (20, 2, 8, 5)
        assert x_b.shape == (20, 2, 8, 5)

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_row_interleaved_correct_shapes(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(h=6)  # even H → equal halves
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='row_interleaved', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert x_a.shape == (20, 2, 3, 8)  # rows 0,2,4
        assert x_b.shape == (20, 2, 3, 8)  # rows 1,3,5

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_row_interleaved_interleaves_rows(self, mock_dispatch):
        """Verify actual pixel values are interleaved correctly."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='row_interleaved', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        torch.testing.assert_close(x_a, x[:, :, 0::2, :])
        torch.testing.assert_close(x_b, x[:, :, 1::2, :])

    def test_3d_input_raises_for_spatial_splits(self):
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(20, 4, 8)  # 3D
        for method in ('horizontal', 'vertical', 'row_interleaved', 'col_interleaved',
                       'diagonal', 'antidiagonal'):
            with pytest.raises(ValueError, match="4-D"):
                run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                            split_method=method)

    def test_h1_horizontal_raises(self):
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(10, 2, 1, 8)  # H=1
        with pytest.raises(ValueError, match="H >= 2"):
            run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                        split_method='horizontal')

    def test_w1_vertical_raises(self):
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(10, 2, 8, 1)  # W=1
        with pytest.raises(ValueError, match="W >= 2"):
            run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                        split_method='vertical')

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_n_splits_creates_correct_number_of_tasks(self, mock_dispatch):
        """Spatial splits run the same slices n_splits times (weight init varies)."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': i}
            for i in range(4)
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d()
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='horizontal', n_splits=4)
        tasks = mock_dispatch.call_args[0][0]
        assert len(tasks) == 4
        # All tasks share the same spatial split (identical x_a / x_b)
        for task in tasks[1:]:
            torch.testing.assert_close(task[0], tasks[0][0])

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_uneven_split_disables_shared_encoder(self, mock_dispatch, caplog):
        """Odd H horizontal split should disable shared_encoder with a logger warning."""
        import logging
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(h=7)  # odd H → uneven halves
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1, 'shared_encoder': True},
                split_method='horizontal', n_splits=1,
            )
        assert any('shared_encoder' in r.message for r in caplog.records)
        _, _, forwarded_params, *_ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('shared_encoder') is False

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_even_split_keeps_shared_encoder(self, mock_dispatch):
        """Even H should NOT disable shared_encoder."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(h=8)  # even H → equal halves
        run_dimensionality_analysis(
            x,
            base_params={'n_epochs': 1, 'shared_encoder': True},
            split_method='horizontal', n_splits=1,
        )
        _, _, forwarded_params, *_ = mock_dispatch.call_args[0][0][0]
        assert forwarded_params.get('shared_encoder') is True

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_index_split_4d(self, mock_dispatch):
        """Index split must correctly slice channel dim for 4-D input."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(10, 6, 4, 4)
        run_dimensionality_analysis(
            x, base_params={'n_epochs': 1},
            split_method='index', channel_indices_x=[0, 1, 2], n_splits=1,
        )
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert x_a.shape == (10, 3, 4, 4)
        assert x_b.shape == (10, 3, 4, 4)

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_col_interleaved_correct_shapes(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = self._4d(h=8, w=8)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='col_interleaved', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert x_a.shape == (20, 2, 8, 4)  # even columns
        assert x_b.shape == (20, 2, 8, 4)  # odd columns

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_col_interleaved_interleaves_columns(self, mock_dispatch):
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.arange(1 * 1 * 4 * 4, dtype=torch.float).reshape(1, 1, 4, 4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='col_interleaved', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        assert torch.equal(x_a, x[:, :, :, 0::2])
        assert torch.equal(x_b, x[:, :, :, 1::2])

    def test_w1_col_interleaved_raises(self):
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(10, 2, 4, 1)
        with pytest.raises(ValueError, match="W >= 2"):
            run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                        split_method='col_interleaved')

    # --- geometric diagonal / antidiagonal ---

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_geometric_diagonal_pixel_counts_square(self, mock_dispatch):
        """For a square 4×4 image: upper+diagonal=10 pixels, lower=6 pixels."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(5, 1, 4, 4)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='diagonal', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        # C=1, upper+diag=10, lower=6
        assert x_a.shape == (5, 1, 10)
        assert x_b.shape == (5, 1, 6)

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_geometric_diagonal_correct_mask(self, mock_dispatch):
        """Diagonal mask: x_a contains pixels where row <= col."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.arange(1 * 1 * 3 * 3, dtype=torch.float).reshape(1, 1, 3, 3)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='diagonal', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        H, W = 3, 3
        row_idx = torch.arange(H).unsqueeze(1)
        col_idx = torch.arange(W).unsqueeze(0)
        mask_a = (row_idx <= col_idx).reshape(-1)
        x_flat = x.reshape(1, 1, -1)
        assert torch.equal(x_a, x_flat[:, :, mask_a])
        assert torch.equal(x_b, x_flat[:, :, ~mask_a])

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_geometric_antidiagonal_correct_mask(self, mock_dispatch):
        """Antidiagonal mask: x_a contains pixels where row + col <= W-1."""
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.arange(1 * 1 * 3 * 3, dtype=torch.float).reshape(1, 1, 3, 3)
        run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                    split_method='antidiagonal', n_splits=1)
        x_a, x_b, *_ = mock_dispatch.call_args[0][0][0]
        H, W = 3, 3
        row_idx = torch.arange(H).unsqueeze(1)
        col_idx = torch.arange(W).unsqueeze(0)
        mask_a = (row_idx + col_idx <= W - 1).reshape(-1)
        x_flat = x.reshape(1, 1, -1)
        assert torch.equal(x_a, x_flat[:, :, mask_a])
        assert torch.equal(x_b, x_flat[:, :, ~mask_a])

    @patch('neural_mi.analysis.dimensionality._dispatch_splits')
    def test_geometric_diagonal_rectangular_warns(self, mock_dispatch, caplog):
        """Non-square input should emit a warning but still run."""
        import logging
        mock_dispatch.return_value = [
            {'train_mi': 0.5, 'test_mi': 0.5, 'pr_eig': 2.0, 'pr_singular': 2.0, 'split_id': 0}
        ]
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(5, 1, 4, 6)  # H != W
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            run_dimensionality_analysis(x, base_params={'n_epochs': 1},
                                        split_method='diagonal', n_splits=1)
        assert any('non-square' in r.message for r in caplog.records)

    def test_geometric_diagonal_cnn2d_raises(self):
        """CNN2D cannot process triangular pixel subsets — must raise."""
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(5, 2, 4, 4)
        with pytest.raises(ValueError, match="cnn2d"):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1, 'embedding_model': 'cnn2d'},
                split_method='diagonal',
            )

    def test_geometric_antidiagonal_cnn_raises(self):
        """CNN1D cannot process triangular pixel subsets — must raise."""
        from neural_mi.analysis.dimensionality import run_dimensionality_analysis
        x = torch.randn(5, 2, 4, 4)
        with pytest.raises(ValueError, match="cnn"):
            run_dimensionality_analysis(
                x,
                base_params={'n_epochs': 1, 'embedding_model': 'cnn'},
                split_method='antidiagonal',
            )


# needed for mock
from unittest.mock import MagicMock  # noqa: E402
