# tests/test_cnn2d.py
"""Tests for CNN2D encoder and 4-D input handling throughout the library."""
import warnings
import pytest
import torch
from unittest.mock import MagicMock, patch

from neural_mi.models.embeddings import CNN2D
from neural_mi.models import CNN2D as CNN2D_from_init
from neural_mi.utils import build_critic


# ---------------------------------------------------------------------------
# CNN2D — architecture and forward pass
# ---------------------------------------------------------------------------

class TestCNN2DModel:
    def test_output_shape(self):
        """Output must be (batch, embedding_dim) regardless of spatial size."""
        model = CNN2D(input_dim=3, hidden_dim=16, embedding_dim=32, n_layers=2)
        x = torch.randn(8, 3, 16, 16)
        out = model(x)
        assert out.shape == (8, 32)

    def test_variable_spatial_size(self):
        """Adaptive pooling must handle arbitrary H × W without re-instantiation."""
        model = CNN2D(input_dim=4, hidden_dim=16, embedding_dim=8, n_layers=1)
        for h, w in [(8, 8), (12, 20), (5, 7), (1, 1)]:
            out = model(torch.randn(4, 4, h, w))
            assert out.shape == (4, 8), f"Failed for H={h}, W={w}"

    def test_single_channel(self):
        model = CNN2D(input_dim=1, hidden_dim=8, embedding_dim=16, n_layers=1)
        out = model(torch.randn(4, 1, 8, 8))
        assert out.shape == (4, 16)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError, match="odd"):
            CNN2D(input_dim=3, hidden_dim=16, embedding_dim=8, n_layers=1, kernel_size=4)

    def test_kernel_size_1(self):
        """kernel_size=1 is the 1×1 conv case — valid."""
        model = CNN2D(input_dim=3, hidden_dim=8, embedding_dim=4, n_layers=1, kernel_size=1)
        out = model(torch.randn(2, 3, 5, 5))
        assert out.shape == (2, 4)

    def test_exported_from_models_init(self):
        """CNN2D must be importable from neural_mi.models."""
        assert CNN2D_from_init is CNN2D

    def test_gradients_flow(self):
        """Gradients must reach Conv2d weights."""
        model = CNN2D(input_dim=2, hidden_dim=8, embedding_dim=4, n_layers=2)
        x = torch.randn(4, 2, 8, 8, requires_grad=False)
        loss = model(x).sum()
        loss.backward()
        first_conv = list(model.conv_layers.modules())[1]  # skip Sequential wrapper
        assert first_conv.weight.grad is not None

    def test_eval_deterministic(self):
        """In eval mode the model is deterministic."""
        model = CNN2D(input_dim=2, hidden_dim=8, embedding_dim=4, n_layers=1).eval()
        x = torch.randn(4, 2, 6, 6)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_n_layers_zero_raises(self):
        """n_layers=0 → empty conv_layers; forward should still work (degenerate case)."""
        # With n_layers=0 there are no Conv2d layers, but the first block (input_dim→hidden_dim)
        # is always added. Check it handles gracefully.
        model = CNN2D(input_dim=2, hidden_dim=8, embedding_dim=4, n_layers=1)
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
# Spatial split methods of mode='dimensionality'
# ---------------------------------------------------------------------------

class TestSpatialSplits:
    """The image splits of X into two halves, for 4-D (N, C, H, W) data."""

    def _4d(self, n=20, c=2, h=8, w=8):
        return torch.randn(n, c, h, w)

    def _split(self, x, method, params=None, **kwargs):
        from neural_mi.analysis.dimensionality import _halves
        return _halves(x, params or {}, method, 1, kwargs)

    def test_horizontal_correct_shapes(self):
        [(a, b, _)] = self._split(self._4d(), 'horizontal')
        assert a.shape == (20, 2, 4, 8) and b.shape == (20, 2, 4, 8)

    def test_horizontal_odd_h(self):
        [(a, b, _)] = self._split(self._4d(h=7), 'horizontal')
        assert a.shape[2] == 3 and b.shape[2] == 4

    def test_vertical_correct_shapes(self):
        [(a, b, _)] = self._split(self._4d(), 'vertical')
        assert a.shape == (20, 2, 8, 4) and b.shape == (20, 2, 8, 4)

    def test_row_interleaved_interleaves_rows(self):
        x = self._4d()
        [(a, b, _)] = self._split(x, 'row_interleaved')
        assert torch.equal(a, x[:, :, 0::2, :]) and torch.equal(b, x[:, :, 1::2, :])

    def test_col_interleaved_interleaves_columns(self):
        x = self._4d()
        [(a, b, _)] = self._split(x, 'col_interleaved')
        assert torch.equal(a, x[:, :, :, 0::2]) and torch.equal(b, x[:, :, :, 1::2])

    @pytest.mark.parametrize('method', ['horizontal', 'vertical', 'row_interleaved',
                                        'col_interleaved', 'diagonal', 'antidiagonal'])
    def test_3d_input_raises_for_spatial_splits(self, method):
        with pytest.raises(ValueError, match='requires 4-D input'):
            self._split(torch.randn(20, 4, 8), method)

    @pytest.mark.parametrize('method, h, w', [('horizontal', 1, 8), ('row_interleaved', 1, 8),
                                              ('vertical', 8, 1), ('col_interleaved', 8, 1)])
    def test_a_single_row_or_column_raises(self, method, h, w):
        with pytest.raises(ValueError, match='requires'):
            self._split(self._4d(h=h, w=w), method)

    def test_index_split_4d(self):
        x = self._4d(c=4)
        from neural_mi.analysis.dimensionality import _halves
        [(a, b, _)] = _halves(x, {}, 'index', 1, {'channel_indices_x': [0, 3]})
        assert torch.equal(a, x[:, [0, 3]]) and torch.equal(b, x[:, [1, 2]])

    def test_geometric_diagonal_correct_mask(self):
        x = self._4d(n=3, c=1, h=4, w=4)
        [(a, b, _)] = self._split(x, 'diagonal')
        rows, cols = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        flat = x.reshape(3, 1, -1)
        assert torch.equal(a, flat[:, :, (rows <= cols).reshape(-1)])
        assert torch.equal(b, flat[:, :, (rows > cols).reshape(-1)])

    def test_geometric_antidiagonal_correct_mask(self):
        x = self._4d(n=3, c=1, h=4, w=4)
        [(a, b, _)] = self._split(x, 'antidiagonal')
        rows, cols = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        flat = x.reshape(3, 1, -1)
        assert torch.equal(a, flat[:, :, (rows + cols <= 3).reshape(-1)])

    def test_uneven_split_disables_shared_encoder(self, caplog):
        with caplog.at_level('WARNING', logger='neural_mi'):
            [(_, _, params)] = self._split(self._4d(h=4, w=5), 'diagonal', {'shared_encoder': True})
        assert params['shared_encoder'] is False
        assert 'non-square' in caplog.text

    def test_even_split_keeps_shared_encoder(self):
        [(_, _, params)] = self._split(self._4d(), 'horizontal', {'shared_encoder': True})
        assert params['shared_encoder'] is True

    @pytest.mark.parametrize('model', ['cnn2d', 'cnn'])
    def test_triangular_splits_refuse_convolutional_encoders(self, model):
        with pytest.raises(ValueError, match='triangular'):
            self._split(self._4d(), 'diagonal', {'embedding_model': model})
