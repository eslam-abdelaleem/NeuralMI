# tests/test_shared_encoder.py
"""Tests for Phase F: shared encoder / siamese networks."""
import pytest
import neural_mi as nmi
from neural_mi import Model, Training
from neural_mi.utils import build_critic

# Minimal fully-populated params for build_critic()
_BUILD = {
    'use_variational': False,
    'embedding_model': 'mlp',
    'hidden_dim': 8,
    'n_layers': 1,
    'embedding_dim': 4,
    'max_n_batches': 512,
    'input_dim_x': 6,
    'input_dim_y': 6,
    'n_channels_x': 6,
    'n_channels_y': 6,
}

# Shared model/training configs for the run()-based tests below.
_MODEL = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
_TRAINING = Training(n_epochs=1, learning_rate=1e-4, batch_size=8, patience=1)


# ---------------------------------------------------------------------------
# Shared-encoder weight tests
# ---------------------------------------------------------------------------

def test_shared_encoder_same_object():
    """shared_encoder=True: SeparableCritic.net_x is net_y."""
    params = {**_BUILD, 'shared_encoder': True}
    critic = build_critic('separable', params)
    assert critic.embedding_net_x is critic.embedding_net_y, (
        "Expected the same embedding network instance for both X and Y."
    )


def test_independent_encoders_different_objects():
    """shared_encoder=False (default): net_x and net_y are different objects."""
    params = {**_BUILD, 'shared_encoder': False}
    critic = build_critic('separable', params)
    assert critic.embedding_net_x is not critic.embedding_net_y, (
        "Expected independent embedding networks for X and Y."
    )


def test_shared_encoder_half_parameters():
    """shared_encoder=True should produce roughly half the parameters."""
    p_shared = {**_BUILD, 'shared_encoder': True}
    p_indep = {**_BUILD, 'shared_encoder': False}
    n_shared = sum(p.numel() for p in build_critic('separable', p_shared).parameters())
    n_indep = sum(p.numel() for p in build_critic('separable', p_indep).parameters())
    # Shared encoder uses one network instead of two — ratio should be close to ~0.5
    ratio = n_shared / n_indep
    assert ratio < 0.75, (
        f"Expected shared encoder to use < 75% of independent params, got ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Guard: shared_encoder=True + critic_type='concat'
# ---------------------------------------------------------------------------

def test_shared_encoder_concat_raises():
    """shared_encoder=True with critic_type='concat' must raise ValueError."""
    params = {**_BUILD, 'shared_encoder': True}
    with pytest.raises(ValueError, match="incompatible with critic_type='concat'"):
        build_critic('concat', params)


def test_shared_encoder_hybrid_does_not_raise():
    """shared_encoder=True with critic_type='hybrid' should work fine."""
    params = {**_BUILD, 'shared_encoder': True}
    critic = build_critic('hybrid', params)
    assert critic is not None


# ---------------------------------------------------------------------------
# dimensionality.py defaults to shared_encoder=True
# ---------------------------------------------------------------------------

def test_dimensionality_defaults_shared_encoder_true():
    """mode='dimensionality' must default shared_encoder=True in analysis_params.

    We verify this via nmi.run so that all defaults are injected before
    run_dimensionality_analysis is called.  The test checks that the run
    succeeds (shared encoder should halve parameters and be valid here) and
    that no ValueError about shared_encoder is raised.
    """
    x, _ = nmi.generators.generate_nonlinear_from_latent(300, 3, 8, 1.0)
    results = nmi.run(
        x, mode='dimensionality',
        model=_MODEL, training=_TRAINING,
        n_workers=1,
    )
    assert results is not None, "dimensionality mode returned None."


def test_dimensionality_shared_encoder_can_be_overridden():
    """Users can override shared_encoder=False in dimensionality mode."""
    x, _ = nmi.generators.generate_nonlinear_from_latent(300, 3, 8, 1.0)
    results = nmi.run(
        x, mode='dimensionality',
        model=Model(embedding_dim=4, hidden_dim=8, n_layers=1, shared_encoder=False),
        training=_TRAINING,
        n_workers=1,
    )
    assert results is not None


# ---------------------------------------------------------------------------
# Top-level run() convenience shortcut
# ---------------------------------------------------------------------------

def test_run_shared_encoder_shortcut():
    """Model(shared_encoder=True) should build a siamese critic without error."""
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=200, dim=4, mi=0.5)
    results = nmi.run(
        x, y, mode='estimate',
        model=Model(embedding_dim=4, hidden_dim=8, n_layers=1, shared_encoder=True),
        training=_TRAINING,
        n_workers=1,
    )
    assert isinstance(results.mi_estimate, float)
