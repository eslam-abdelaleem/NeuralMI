"""Tests for Library Fix 1 (physics params tracking) and Library Fix 2
(pretrained backbone spatial dimension mismatch).
"""

import warnings
import numpy as np
import pytest
import torch

import neural_mi as nmi


# ──────────────────────────────────────────────────────────────────────────────
# Fix 1: physics_params_history / physics_params_final
# ──────────────────────────────────────────────────────────────────────────────

def test_non_biased_model_no_physics_params():
    """Standard CNN should produce no physics_params_history."""
    np.random.seed(3)
    X = np.random.randn(80, 2, 32).astype(np.float32)
    Y = 0.5 * X + 0.5 * np.random.randn(80, 2, 32).astype(np.float32)

    result = nmi.run(
        x_data=X, y_data=Y,
        mode='estimate',
        split_mode='random',
        base_params={
            'n_epochs': 5,
            'patience': 5,
            'batch_size': 32,
            'hidden_dim': 16,
            'embedding_dim': 8,
            'n_layers': 1,
            'embedding_model': 'cnn',
        },
        random_seed=3,
        show_progress=False,
    )

    assert 'physics_params_history' not in result.details, (
        "Standard CNN should not produce physics_params_history"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fix 2: Pretrained backbone spatial dimension mismatch
# ──────────────────────────────────────────────────────────────────────────────

class TestPretrainedBackboneSpatialMismatch:
    """Tests for the spatial dimension mismatch handling in PretrainedBackboneEmbedding."""

    def _make_image_data(self, n: int, n_ch: int, h: int, w: int):
        return (
            np.random.randn(n, n_ch, h, w).astype(np.float32),
            np.random.randn(n, n_ch, h, w).astype(np.float32),
        )

    def test_28x28_emits_warning_and_runs(self):
        """Passing 28×28 images to a (pretrained=False) ResNet18 should
        emit a UserWarning and still produce valid MI output.
        Use 3 channels to match ResNet18's expected input channels.
        """
        try:
            import torchvision  # noqa: F401
        except ImportError:
            pytest.skip("torchvision not installed")

        X, Y = self._make_image_data(60, 3, 28, 28)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = nmi.run(
                x_data=X, y_data=Y,
                mode='estimate',
                split_mode='random',
                base_params={
                    'n_epochs': 3,
                    'patience': 3,
                    'batch_size': 16,
                    'hidden_dim': 16,
                    'embedding_dim': 8,
                    'n_layers': 1,
                    'embedding_model': 'pretrained_backbone',
                    'pytorch_predefined': 'resnet18',
                    'pretrained': False,  # avoid downloading weights
                },
                random_seed=0,
                show_progress=False,
            )

        spatial_warnings = [w for w in caught if 'spatial size' in str(w.message).lower()
                            or 'upsample' in str(w.message).lower()]
        assert len(spatial_warnings) > 0, (
            "Expected a UserWarning about spatial size mismatch for 28×28 input"
        )
        assert result.mi_estimate is not None
        assert np.isfinite(result.mi_estimate)

    def test_224x224_no_warning(self):
        """Passing 224×224 images should produce no spatial mismatch warning."""
        try:
            import torchvision  # noqa: F401
        except ImportError:
            pytest.skip("torchvision not installed")

        X, Y = self._make_image_data(20, 3, 224, 224)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(
                x_data=X, y_data=Y,
                mode='estimate',
                split_mode='random',
                base_params={
                    'n_epochs': 2,
                    'patience': 2,
                    'batch_size': 8,
                    'hidden_dim': 16,
                    'embedding_dim': 8,
                    'n_layers': 1,
                    'embedding_model': 'pretrained_backbone',
                    'pytorch_predefined': 'resnet18',
                    'pretrained': False,
                },
                random_seed=0,
                show_progress=False,
            )

        spatial_warnings = [w for w in caught if 'spatial size' in str(w.message).lower()
                            or 'upsample' in str(w.message).lower()]
        assert len(spatial_warnings) == 0, (
            f"No spatial warning expected for 224×224 input, got: {spatial_warnings}"
        )

    def test_backbone_weights_frozen_after_training(self):
        """Backbone parameters should not change after training (requires_grad=False)."""
        try:
            import torchvision  # noqa: F401
        except ImportError:
            pytest.skip("torchvision not installed")

        from neural_mi.models.embeddings import PretrainedBackboneEmbedding

        emb = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embed_dim=8, n_layers=1,
            pytorch_predefined='resnet18', pretrained=False,
        )

        # Record backbone parameters before a fake forward pass
        backbone_params_before = {
            n: p.data.clone() for n, p in emb.backbone.named_parameters()
        }

        # Simulate forward + backward to check frozen backbone
        dummy = torch.randn(4, 3, 224, 224)
        out = emb(dummy)  # forward
        loss = out.sum()
        loss.backward()

        for name, param_after in emb.backbone.named_parameters():
            assert param_after.grad is None or (param_after.grad == 0).all(), (
                f"Backbone parameter '{name}' received non-zero gradients — "
                "the backbone is not properly frozen."
            )
            # Data should be unchanged
            assert torch.allclose(backbone_params_before[name], param_after.data), (
                f"Backbone parameter '{name}' changed after backward pass."
            )

    def test_channel_adapter_gradient_backbone_frozen_bn_eval(self):
        """Regression test for the dead-channel-adapter / BN-train-mode bugs.

        With ``input_dim != backbone_in_ch`` a trainable 1x1 conv channel
        adapter is inserted before the backbone. Removing the stray
        ``torch.no_grad()`` around the backbone forward (which previously
        severed the adapter's gradient path) must let gradient reach the
        adapter, while the backbone itself stays frozen and its BatchNorm
        layers stay in eval mode even when the outer module is in train().
        """
        try:
            import torchvision  # noqa: F401
        except ImportError:
            pytest.skip("torchvision not installed")

        from neural_mi.models.embeddings import PretrainedBackboneEmbedding

        # input_dim=1 != resnet18's expected 3 channels -> forces the adapter.
        emb = PretrainedBackboneEmbedding(
            input_dim=1, hidden_dim=16, embed_dim=8, n_layers=1,
            pytorch_predefined='resnet18', pretrained=False,
        )
        assert emb._channel_adapt is not None, (
            "Expected a channel adapter to be created for input_dim != backbone_in_ch"
        )

        # Simulate the trainer calling .train() on the whole model.
        emb.train()
        bn_modules = [m for m in emb.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert len(bn_modules) > 0, "Expected resnet18 backbone to contain BatchNorm2d layers"
        assert all(not bn.training for bn in bn_modules), (
            "Backbone BatchNorm layers must stay in eval() mode even after emb.train()"
        )

        backbone_params_before = {
            n: p.data.clone() for n, p in emb.backbone.named_parameters()
        }

        dummy = torch.randn(4, 1, 224, 224)
        out = emb(dummy)

        # BN must still be in eval mode during/after the forward pass.
        assert all(not bn.training for bn in bn_modules), (
            "Backbone BatchNorm layers must remain in eval() mode during forward"
        )

        loss = out.sum()
        loss.backward()

        # (a) The channel adapter must receive a non-None, non-zero gradient.
        adapter_grad = emb._channel_adapt.weight.grad
        assert adapter_grad is not None, (
            "Channel adapter received no gradient — the no_grad() regression is back."
        )
        assert (adapter_grad != 0).any(), (
            "Channel adapter gradient is all-zero — gradient is not flowing through the backbone."
        )

        # (b) Backbone stays frozen: no grad, and weights unchanged.
        for name, param_after in emb.backbone.named_parameters():
            assert not param_after.requires_grad, (
                f"Backbone parameter '{name}' unexpectedly has requires_grad=True."
            )
            assert param_after.grad is None or (param_after.grad == 0).all(), (
                f"Backbone parameter '{name}' received non-zero gradients."
            )
            assert torch.allclose(backbone_params_before[name], param_after.data), (
                f"Backbone parameter '{name}' changed after backward pass."
            )
