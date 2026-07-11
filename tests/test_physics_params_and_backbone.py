"""Tests for Library Fix 1 (physics params tracking) and Library Fix 2
(pretrained backbone spatial dimension mismatch).
"""

import warnings
import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.generators import generate_windowed_oscillatory


# ──────────────────────────────────────────────────────────────────────────────
# Fix 1: physics_params_history / physics_params_final
# ──────────────────────────────────────────────────────────────────────────────

def test_sinc_physics_params_tracked():
    """physics_params_history is populated and contains non-trivial filter movement
    after training SincCNN on synthetic oscillatory data.
    """
    np.random.seed(0)
    torch.manual_seed(0)

    X, Y, _ = generate_windowed_oscillatory(
        n_windows=200,
        n_channels=2,
        window_size=128,
        f_carrier_hz=10.0,
        sample_rate=256.0,
        latent_mi=1.0,
        snr=3.0,
    )

    result = nmi.run(
        x_data=X, y_data=Y,
        mode='estimate',
        split_mode='random',
        processor_params_x={'sample_rate': 256.0},
        processor_params_y={'sample_rate': 256.0},
        base_params={
            'n_epochs': 30,
            'patience': 30,
            'batch_size': 64,
            'hidden_dim': 32,
            'embedding_dim': 16,
            'n_layers': 2,
            'embedding_model': 'sinc_cnn',
            'n_sinc_filters': 4,
        },
        random_seed=0,
        show_progress=False,
    )

    # (a) physics_params_history must be present
    assert 'physics_params_history' in result.details, (
        "physics_params_history not found in result.details for sinc_cnn"
    )

    hist = result.details['physics_params_history']
    assert 'x_f_low_hz' in hist, "Expected key 'x_f_low_hz' in physics_params_history"
    assert 'x_f_high_hz' in hist, "Expected key 'x_f_high_hz' in physics_params_history"

    # (b) must have one entry per training epoch
    n_epochs_run = len(result.details['test_mi_history'])
    assert len(hist['x_f_low_hz']) == n_epochs_run, (
        f"physics_params_history length ({len(hist['x_f_low_hz'])}) "
        f"!= n_epochs_run ({n_epochs_run})"
    )

    # (c) physics_params_final must be present
    assert 'physics_params_final' in result.details, (
        "physics_params_final not found in result.details"
    )
    final = result.details['physics_params_final']
    assert 'x_f_low_hz' in final
    assert 'x_f_high_hz' in final

    # (d) filters must have moved from initialisation
    # Initial f_low values come from classical EEG bands (see SincEmbedding.__init__)
    init_f_low = hist['x_f_low_hz'][0]   # list of floats at epoch 0
    fin_f_low  = hist['x_f_low_hz'][-1]

    if isinstance(init_f_low, list):
        init_arr = np.array(init_f_low)
        fin_arr  = np.array(fin_f_low)
    else:
        init_arr = np.array([init_f_low])
        fin_arr  = np.array([fin_f_low])

    max_shift = float(np.max(np.abs(fin_arr - init_arr)))
    assert max_shift > 0.01, (
        f"Sinc filter cutoffs did not move during training "
        f"(max shift = {max_shift:.4f} Hz, expected > 0.01 Hz). "
        "The physics_params tracking may be broken."
    )


def test_calcium_physics_params_not_tracked_when_fixed():
    """When learn_calcium_kernel=False, no physics_params_history should be stored."""
    np.random.seed(1)
    torch.manual_seed(1)

    X = np.random.randn(100, 2, 30).astype(np.float32)
    Y = np.random.randn(100, 2, 30).astype(np.float32)

    result = nmi.run(
        x_data=X, y_data=Y,
        mode='estimate',
        split_mode='random',
        processor_params_x={'sample_rate': 30.0},
        processor_params_y={'sample_rate': 30.0},
        base_params={
            'n_epochs': 5,
            'patience': 5,
            'batch_size': 32,
            'hidden_dim': 16,
            'embedding_dim': 8,
            'n_layers': 1,
            'embedding_model': 'calcium_cnn',
            'learn_calcium_kernel': False,  # fixed → no physics tracking
        },
        random_seed=1,
        show_progress=False,
    )

    # When kernel is fixed, get_physics_params() returns {} → nothing tracked
    assert 'physics_params_history' not in result.details, (
        "physics_params_history should be absent when learn_calcium_kernel=False"
    )


def test_calcium_physics_params_tracked_when_learnable():
    """When learn_calcium_kernel=True, physics_params_history must contain tau values."""
    np.random.seed(2)
    torch.manual_seed(2)

    X = np.random.randn(100, 2, 60).astype(np.float32)
    Y = 0.5 * X + 0.5 * np.random.randn(100, 2, 60).astype(np.float32)

    result = nmi.run(
        x_data=X, y_data=Y,
        mode='estimate',
        split_mode='random',
        processor_params_x={'sample_rate': 30.0},
        processor_params_y={'sample_rate': 30.0},
        base_params={
            'n_epochs': 10,
            'patience': 10,
            'batch_size': 32,
            'hidden_dim': 16,
            'embedding_dim': 8,
            'n_layers': 1,
            'embedding_model': 'calcium_cnn',
            'learn_calcium_kernel': True,
            'tau_rise': 0.05,
            'tau_decay': 0.4,
        },
        random_seed=2,
        show_progress=False,
    )

    assert 'physics_params_history' in result.details, (
        "physics_params_history must be present when learn_calcium_kernel=True"
    )
    hist = result.details['physics_params_history']
    assert 'x_tau_rise_s' in hist, "Expected 'x_tau_rise_s' in physics_params_history"
    assert 'x_tau_decay_s' in hist, "Expected 'x_tau_decay_s' in physics_params_history"
    assert 'physics_params_final' in result.details


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
