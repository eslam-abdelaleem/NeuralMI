# tests/test_safety.py
"""Regression tests for Phase A safety fixes (A1-A5)."""
import inspect
import warnings

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Transfer
from neural_mi.analysis.sweep import ParameterSweep
from neural_mi.exceptions import TrainingError
from neural_mi.training.trainer import Trainer

# _BASE dict is still used by the ParameterSweep/build_critic engine-level tests below.
_BASE = {
    'n_epochs': 2, 'learning_rate': 1e-4, 'batch_size': 8,
    'patience': 1, 'embedding_dim': 4, 'hidden_dim': 8, 'n_layers': 1,
}
# Config equivalents for the run()-based tests.
_MODEL = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
_TRAINING = Training(n_epochs=2, learning_rate=1e-4, batch_size=8, patience=1)


# ---------------------------------------------------------------------------
# A1 — ValueError on 3-D input to mode='transfer'
# ---------------------------------------------------------------------------

def test_a1_transfer_3d_x_raises():
    """A1: 3-D x_data passed to mode='transfer' must raise ValueError."""
    x = np.random.randn(20, 3, 5)  # 3-D (pre-windowed)
    y = np.random.randn(20, 3, 5)
    with pytest.raises(ValueError, match="mode='transfer' requires 2-D"):
        nmi.run(
            x, y,
            mode='transfer',
            model=_MODEL, training=_TRAINING,
            transfer=Transfer(history_window=2),
            n_workers=1,
        )


def test_a1_transfer_2d_does_not_raise():
    """A1: 2-D input to mode='transfer' should proceed (not raise shape error)."""
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=100, dim=3, mi=0.5)
    try:
        nmi.run(
            x, y,
            mode='transfer',
            model=_MODEL, training=_TRAINING,
            transfer=Transfer(history_window=2),
            n_workers=1,
        )
    except ValueError as e:
        if "requires 2-D" in str(e):
            pytest.fail(f"Unexpected 3-D shape error on 2-D input: {e}")


# ---------------------------------------------------------------------------
# A2 — beta default unified to 1024
# ---------------------------------------------------------------------------

def test_a2_trainer_beta_default_is_1024():
    """A2: Trainer.__init__ default for beta must be 1024 (not 512)."""
    sig = inspect.signature(Trainer.__init__)
    assert sig.parameters['beta'].default == 1024, (
        f"Expected beta default=1024, got {sig.parameters['beta'].default}"
    )


def test_a2_defaults_schema_beta_is_1024():
    """A2: BASE_PARAMS_SCHEMA['beta']['default'] must be 1024."""
    from neural_mi.defaults import BASE_PARAMS_SCHEMA
    assert BASE_PARAMS_SCHEMA['beta']['default'] == 1024.0


# ---------------------------------------------------------------------------
# A3 — Warning when train_subset_size is clamped
# ---------------------------------------------------------------------------

def test_a3_train_subset_size_clamp_emits_warning():
    """A3: train_subset_size larger than available samples must emit a warning."""
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=100, dim=2, mi=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            nmi.run(
                x, y,
                mode='estimate',
                model=_MODEL,
                training=Training(n_epochs=2, learning_rate=1e-4, batch_size=8,
                                  patience=1, train_subset_size=50_000),
                n_workers=1,
            )
        except Exception:
            pass  # Only care about the warning, not success/failure
    msgs = [str(w.message) for w in caught]
    assert any("train_subset_size" in m for m in msgs), (
        f"Expected a train_subset_size warning; got: {msgs}"
    )


# ---------------------------------------------------------------------------
# A4 — TrainingError after 3 consecutive NaN epochs
# ---------------------------------------------------------------------------

def test_a4_nan_streak_raises_training_error():
    """A4: Three consecutive NaN epochs must raise TrainingError.

    We mock Trainer._safe_eval_mi() to return NaN so that the real PyTorch
    training step can still run (and compute valid gradients) while the
    *evaluation* path always reports NaN MI — triggering the streak counter.
    """
    from unittest.mock import patch
    from neural_mi.data.handler import PairedDataset
    from neural_mi.data.static import StaticDataset
    from neural_mi.estimators import infonce_lower_bound
    from neural_mi.utils import build_critic

    params = {
        'use_variational': False, 'embedding_model': 'mlp',
        'hidden_dim': 8, 'n_layers': 1, 'embedding_dim': 4,
        'max_n_batches': 512,
        'input_dim_x': 2, 'input_dim_y': 2, 'n_channels_x': 2, 'n_channels_y': 2,
        'shared_encoder': False,
    }
    critic = build_critic('separable', params)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    device = torch.device('cpu')
    trainer = Trainer(critic.to(device), infonce_lower_bound, optimizer, device)

    x_t = torch.randn(200, 2, 1)
    y_t = torch.randn(200, 2, 1)
    ds = PairedDataset(StaticDataset(x_t), StaticDataset(y_t))

    with patch.object(Trainer, '_safe_eval_mi', return_value=float('nan')):
        with pytest.raises(TrainingError, match="consecutive NaN"):
            trainer.train(ds, n_epochs=10, batch_size=32, patience=100, show_progress=False)


def test_a4_single_nan_does_not_raise():
    """A4: A single NaN epoch (e.g. via very short training) must NOT raise immediately."""
    # Smoke-test: a real model trained for a very short time shouldn't trip the guard
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=200, dim=2, mi=0.5)
    try:
        nmi.run(
            x, y,
            mode='estimate',
            model=_MODEL, training=_TRAINING,
            n_workers=1,
        )
    except TrainingError as e:
        if "consecutive NaN" in str(e):
            pytest.fail("Unexpected consecutive-NaN TrainingError on valid data.")


# ---------------------------------------------------------------------------
# A5 — ValueError for ConcatCritic + embedding_dim in sweep
# ---------------------------------------------------------------------------

def test_a5_concat_embedding_dim_sweep_raises():
    """A5: Sweeping embedding_dim with concat critic must raise ValueError (not warn)."""
    x = torch.randn(80, 4, 1)
    y = torch.randn(80, 4, 1)
    bp = {
        **_BASE,
        'critic_type': 'concat',
        'input_dim_x': 4, 'input_dim_y': 4,
        'n_channels_x': 4, 'n_channels_y': 4,
    }
    sweep = ParameterSweep(x, y, bp)
    with pytest.raises(ValueError, match="embedding_dim"):
        sweep._prepare_tasks(
            {'embedding_dim': [4, 8]},
            is_proc_sweep=False,
            max_samples_per_task=None,
        )


def test_a5_separable_embedding_dim_sweep_does_not_raise():
    """A5: Sweeping embedding_dim with separable critic must not raise."""
    x = torch.randn(80, 4, 1)
    y = torch.randn(80, 4, 1)
    bp = {
        **_BASE,
        'critic_type': 'separable',
        'input_dim_x': 4, 'input_dim_y': 4,
        'n_channels_x': 4, 'n_channels_y': 4,
    }
    sweep = ParameterSweep(x, y, bp)
    tasks = sweep._prepare_tasks(
        {'embedding_dim': [4, 8]},
        is_proc_sweep=False,
        max_samples_per_task=None,
    )
    assert len(tasks) == 2
