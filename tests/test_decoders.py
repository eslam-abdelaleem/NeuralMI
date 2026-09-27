# tests/test_decoders.py
"""Tests for build_decoder's dispatch and the CNN1D/CNN2D decoder classes.

Covers that build_decoder correctly reaches CNN1DDecoder for
embedding_model='cnn' and CNN2DDecoder for embedding_model='cnn2d' (matching
the real embedding_model names, not 'cnn1d'), and that use_decoder=True works
end-to-end for both without a shape mismatch in the reconstruction loss.
"""
import logging

import numpy as np
import pytest
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
        d = build_decoder('cnn', embedding_dim=4, hidden_dim=8, n_channels=2, window_size=10, n_layers=2)
        assert isinstance(d, CNN1DDecoder)
        out = d(torch.randn(3, 4))
        assert out.shape == (3, 2, 10)

    def test_cnn2d_dispatches_to_cnn2d_decoder_with_explicit_shape(self):
        d = build_decoder('cnn2d', embedding_dim=4, hidden_dim=8, n_channels=1,
                          window_size=64, n_layers=2, height=8, width=8)
        assert isinstance(d, CNN2DDecoder)
        out = d(torch.randn(3, 4))
        assert out.shape == (3, 1, 8, 8)

    def test_cnn2d_falls_back_to_square_shape_when_height_width_missing(self):
        d = build_decoder('cnn2d', embedding_dim=4, hidden_dim=8, n_channels=1,
                          window_size=16, n_layers=2)
        out = d(torch.randn(3, 4))
        assert out.shape == (3, 1, 4, 4)

    def test_unknown_embedding_model_falls_back_to_mlp_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            d = build_decoder('some_unregistered_model', embedding_dim=4, hidden_dim=8,
                              n_channels=2, window_size=10, n_layers=2)
        assert isinstance(d, MLPDecoder)
        assert "No dedicated decoder" in caplog.text
        assert "some_unregistered_model" in caplog.text

    def test_pretrained_backbone_falls_back_to_mlp_with_warning(self, caplog):
        """pretrained_backbone has no dedicated decoder; must warn, not silently swap."""
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            d = build_decoder('pretrained_backbone', embedding_dim=4, hidden_dim=8,
                              n_channels=3, window_size=49, n_layers=2)
        assert isinstance(d, MLPDecoder)
        assert "No dedicated decoder" in caplog.text

    def test_dedicated_decoders_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            build_decoder('gru', embedding_dim=4, hidden_dim=8, n_channels=2, window_size=10, n_layers=1)
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


class TestBottleneckWeighting:
    """The reconstruction term sits inside beta's bracket.

    The objective is ``KL - beta * [MI - lambda_x * rec_x - lambda_y * rec_y]``,
    so beta scales reconstruction and MI together and each lambda is read
    against the MI term. ``decoder_recon_loss`` reports the contribution with
    those coefficients attached, which makes the scaling checkable: freezing
    the weights with ``learning_rate=0.0`` leaves the reported value as exactly
    ``beta * lambda * MSE`` at initialisation.
    """

    @staticmethod
    def _recon(beta=1.0, lam=0.01, variational=True, **model_kw):
        import random
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(256, 2, 8)).astype(np.float32)
        y = rng.normal(size=(256, 2, 8)).astype(np.float32)
        result = nmi.run(
            x, y, mode='estimate',
            model=Model(embedding_model='mlp', hidden_dim=8, embedding_dim=4,
                        n_layers=1, use_decoder=True, decoder_lambda=lam,
                        use_variational=variational, beta=beta, **model_kw),
            training=Training(n_epochs=1, batch_size=128, patience=100,
                              learning_rate=0.0),
            verbose=False, show_progress=False,
        )
        return result.get('decoder_recon_loss')

    def test_beta_scales_the_reconstruction_term(self):
        base = self._recon(beta=1.0)
        assert base > 0
        assert self._recon(beta=4.0) == pytest.approx(4.0 * base, rel=1e-6)

    def test_lambda_scales_the_reconstruction_term(self):
        base = self._recon(lam=0.01)
        assert self._recon(lam=0.02) == pytest.approx(2.0 * base, rel=1e-6)

    def test_beta_is_inert_without_a_variational_encoder(self):
        # With no KL term there is nothing for beta to trade against, so the
        # lambdas carry their own weight and beta drops out entirely.
        base = self._recon(beta=1.0, variational=False)
        assert self._recon(beta=4.0, variational=False) == pytest.approx(base, rel=1e-6)

    def test_shared_lambda_reaches_both_axes(self):
        # decoder_lambda_x/y are present as None once defaults are applied, so
        # the shared decoder_lambda has to survive that None on the way through.
        both = self._recon(lam=0.01)
        x_only = self._recon(decoder_lambda_x=0.01, decoder_lambda_y=0.0)
        assert 0.0 < x_only < both
