# tests/test_decoders.py
"""Tests for build_decoder's dispatch and the CNN1D/CNN2D decoder classes.

Covers that build_decoder correctly reaches CNN1DDecoder for
embedding_model='cnn' and CNN2DDecoder for embedding_model='cnn2d' (matching
the real embedding_model names, not 'cnn1d'), and that use_decoder=True works
end-to-end for both without a shape mismatch in the reconstruction loss.
"""
import logging

import numpy as np
import torch

import neural_mi as nmi
from neural_mi import Model, Training
from neural_mi.models.decoders import (
    build_decoder, MLPDecoder, CNN1DDecoder, CNN2DDecoder,
)


class TestBuildDecoderDispatch:
    """Unit-level: build_decoder's string dispatch, independent of nmi.run()."""

    def test_cnn_key_dispatches_to_cnn1d_decoder(self):
        """'cnn' is the real embedding_model name (see ALLOWED_VALUES); 'cnn1d' is not."""
        d = build_decoder('cnn', embed_dim=4, hidden_dim=8, n_channels=2, window_size=10, n_layers=2)
        assert isinstance(d, CNN1DDecoder)
        out = d(torch.randn(3, 4))
        assert out.shape == (3, 2, 10)

    def test_cnn2d_dispatches_to_cnn2d_decoder_with_explicit_shape(self):
        d = build_decoder('cnn2d', embed_dim=4, hidden_dim=8, n_channels=1,
                          window_size=64, n_layers=2, height=8, width=8)
        assert isinstance(d, CNN2DDecoder)
        out = d(torch.randn(3, 4))
        assert out.shape == (3, 1, 8, 8)

    def test_cnn2d_falls_back_to_square_shape_when_height_width_missing(self):
        d = build_decoder('cnn2d', embed_dim=4, hidden_dim=8, n_channels=1,
                          window_size=16, n_layers=2)
        out = d(torch.randn(3, 4))
        assert out.shape == (3, 1, 4, 4)

    def test_unknown_embedding_model_falls_back_to_mlp_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            d = build_decoder('some_unregistered_model', embed_dim=4, hidden_dim=8,
                              n_channels=2, window_size=10, n_layers=2)
        assert isinstance(d, MLPDecoder)
        assert "No dedicated decoder" in caplog.text
        assert "some_unregistered_model" in caplog.text

    def test_pretrained_backbone_falls_back_to_mlp_with_warning(self, caplog):
        """pretrained_backbone has no dedicated decoder; must warn, not silently swap."""
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            d = build_decoder('pretrained_backbone', embed_dim=4, hidden_dim=8,
                              n_channels=3, window_size=49, n_layers=2)
        assert isinstance(d, MLPDecoder)
        assert "No dedicated decoder" in caplog.text

    def test_dedicated_decoders_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            build_decoder('gru', embed_dim=4, hidden_dim=8, n_channels=2, window_size=10, n_layers=1)
        assert "No dedicated decoder" not in caplog.text


class TestUseDecoderEndToEnd:
    """End-to-end: use_decoder=True through nmi.run(), first-batch shapes."""

    def test_cnn_embedding_use_decoder_trains_without_crashing(self):
        rng = np.random.default_rng(0)
        # Pre-processed 3D data: (n_samples, n_channels, window_size).
        x = rng.standard_normal((80, 1, 10)).astype('float32')
        y = rng.standard_normal((80, 1, 10)).astype('float32')
        res = nmi.run(
            x, y, mode='estimate',
            model=Model(embedding_dim=4, hidden_dim=8, n_layers=1,
                       embedding_model='cnn', use_decoder=True),
            training=Training(n_epochs=1, batch_size=16),
            show_progress=False, seed=0,
        )
        assert np.isfinite(res.mi_estimate)

    def test_cnn2d_embedding_use_decoder_trains_without_crashing(self):
        """embedding_model='cnn2d' + use_decoder=True on genuine 4-D (N,C,H,W) data."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((80, 1, 8, 8)).astype('float32')
        y = rng.standard_normal((80, 1, 8, 8)).astype('float32')
        res = nmi.run(
            x, y, mode='estimate',
            model=Model(embedding_dim=4, hidden_dim=8, n_layers=1,
                       embedding_model='cnn2d', use_decoder=True),
            training=Training(n_epochs=1, batch_size=16),
            show_progress=False, seed=0,
        )
        assert np.isfinite(res.mi_estimate)

    def test_cnn2d_embedding_use_decoder_asymmetric_xy_shapes(self):
        """X and Y may have different channel counts / spatial sizes; each gets its own decoder."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((80, 1, 8, 8)).astype('float32')
        y = rng.standard_normal((80, 2, 6, 6)).astype('float32')
        res = nmi.run(
            x, y, mode='estimate',
            model=Model(embedding_dim=4, hidden_dim=8, n_layers=1,
                       embedding_model='cnn2d', use_decoder=True),
            training=Training(n_epochs=1, batch_size=16),
            show_progress=False, seed=0,
        )
        assert np.isfinite(res.mi_estimate)
