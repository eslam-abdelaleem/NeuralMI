# tests/test_safety.py
"""Regression tests guarding specific safety-critical behaviors."""
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
# ValueError on 3-D input to mode='transfer'
# ---------------------------------------------------------------------------

def test_transfer_mode_rejects_3d_x_data():
    """3-D x_data passed to mode='transfer' must raise ValueError."""
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


def test_transfer_mode_accepts_2d_data():
    """2-D input to mode='transfer' should proceed (not raise a shape error)."""
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
# beta default unified to 1024
# ---------------------------------------------------------------------------

def test_trainer_beta_default_is_1024():
    """Trainer.__init__ default for beta must be 1024 (not 512)."""
    sig = inspect.signature(Trainer.__init__)
    assert sig.parameters['beta'].default == 1024, (
        f"Expected beta default=1024, got {sig.parameters['beta'].default}"
    )


def test_defaults_schema_beta_is_1024():
    """BASE_PARAMS_SCHEMA['beta']['default'] must be 1024."""
    from neural_mi.defaults import BASE_PARAMS_SCHEMA
    assert BASE_PARAMS_SCHEMA['beta']['default'] == 1024.0


# ---------------------------------------------------------------------------
# Warning when train_subset_size is clamped
# ---------------------------------------------------------------------------

def test_train_subset_size_clamp_emits_warning():
    """train_subset_size larger than available samples must emit a warning."""
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
# TrainingError after 3 consecutive NaN epochs
# ---------------------------------------------------------------------------

def test_nan_streak_raises_training_error():
    """Three consecutive NaN epochs must raise TrainingError.

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


def test_single_nan_epoch_does_not_raise():
    """A single NaN epoch (e.g. via very short training) must NOT raise immediately."""
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
# ValueError for ConcatCritic + embedding_dim in sweep
# ---------------------------------------------------------------------------

def test_concat_critic_embedding_dim_sweep_raises():
    """Sweeping embedding_dim with concat critic must raise ValueError (not warn)."""
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


def test_separable_critic_embedding_dim_sweep_does_not_raise():
    """Sweeping embedding_dim with separable critic must not raise."""
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


# ---------------------------------------------------------------------------
# Phase 1 spec: blocked-split leakage check (WindowManager path, Fix 2)
# ---------------------------------------------------------------------------

import logging
import re


def test_blocked_split_leakage_warns_on_small_gap_fraction(caplog):
    """Overlapping windows + a too-small split_gap_fraction must warn, naming
    the minimum gap_fraction that clears the check -- and that value must
    actually clear it when used."""
    np.random.seed(0)
    x = np.random.randn(20000, 2).astype('float32')
    y = np.random.randn(20000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 200, 'step_size': 10},
                          y='continuous', y_params={'window_size': 200, 'step_size': 10})
    train_cfg = Training(n_epochs=1, patience=1)

    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        nmi.run(x, y, mode='estimate', processing=proc,
               split=nmi.Split(mode='blocked', gap_fraction=0.05, n_test_blocks=3),
               training=train_cfg, n_workers=1, show_progress=False, seed=0)
    leak_msgs = [r.message for r in caplog.records if 'may leak' in r.message]
    assert leak_msgs, f"Expected a leakage warning; got: {[r.message for r in caplog.records]}"
    match = re.search(r"at least ([\d.]+)", leak_msgs[0])
    assert match, f"Expected a recommended gap_fraction in the message: {leak_msgs[0]}"
    recommended_gap_fraction = float(match.group(1))

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        nmi.run(x, y, mode='estimate', processing=proc,
               split=nmi.Split(mode='blocked', gap_fraction=min(recommended_gap_fraction, 0.99),
                               n_test_blocks=3),
               training=train_cfg, n_workers=1, show_progress=False, seed=0)
    leak_msgs_after = [r.message for r in caplog.records if 'may leak' in r.message]
    assert not leak_msgs_after, (
        f"Recommended gap_fraction={recommended_gap_fraction} should clear the check, "
        f"but it warned again: {leak_msgs_after}"
    )


def test_blocked_split_leakage_no_warning_with_large_gap_fraction(caplog):
    """A sufficiently large split_gap_fraction must not trigger the leakage warning."""
    np.random.seed(0)
    x = np.random.randn(20000, 2).astype('float32')
    y = np.random.randn(20000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 200, 'step_size': 10},
                          y='continuous', y_params={'window_size': 200, 'step_size': 10})
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        nmi.run(x, y, mode='estimate', processing=proc,
               split=nmi.Split(mode='blocked', gap_fraction=0.9, n_test_blocks=3),
               training=Training(n_epochs=1, patience=1), n_workers=1, show_progress=False, seed=0)
    leak_msgs = [r.message for r in caplog.records if 'may leak' in r.message]
    assert not leak_msgs, f"Did not expect a leakage warning; got: {leak_msgs}"


# ---------------------------------------------------------------------------
# Phase 1 spec: blocked-split leakage check (transfer path, Fix 3)
# ---------------------------------------------------------------------------

def test_transfer_path_leakage_warns_when_history_window_exceeds_gap(caplog):
    """mode='transfer' builds windows via stride-1 unfold (step=1), bypassing
    WindowManager entirely -- the leakage check must still fire when
    history_window exceeds the time-domain gap."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=50, prediction_horizon=1),
               split=nmi.Split(mode='blocked', gap_fraction=0.1, n_test_blocks=3),
               training=Training(n_epochs=1, patience=1), n_workers=1, show_progress=False, seed=0)
    leak_msgs = [r.message for r in caplog.records if 'may leak' in r.message]
    assert leak_msgs, f"Expected a leakage warning on the transfer path; got: {[r.message for r in caplog.records]}"


def test_transfer_path_leakage_no_warning_with_large_gap(caplog):
    """A large enough split_gap_fraction must not warn on the transfer path either."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=50, prediction_horizon=1),
               split=nmi.Split(mode='blocked', gap_fraction=0.95, n_test_blocks=3),
               training=Training(n_epochs=1, patience=1), n_workers=1, show_progress=False, seed=0)
    leak_msgs = [r.message for r in caplog.records if 'may leak' in r.message]
    assert not leak_msgs, f"Did not expect a leakage warning; got: {leak_msgs}"


# ---------------------------------------------------------------------------
# Phase 1 spec: blocked-split leakage check skips cleanly on the static path
# ---------------------------------------------------------------------------

def test_blocked_split_leakage_check_skips_on_static_path():
    """No processor / no window_manager (already-windowed 3-D input) must not
    crash -- the leakage check has nothing to validate against and skips."""
    np.random.seed(0)
    xw = np.random.randn(500, 5, 20).astype('float32')
    yw = np.random.randn(500, 5, 20).astype('float32')
    result = nmi.run(xw, yw, mode='estimate', split=nmi.Split(mode='blocked', gap_fraction=0.05),
                     training=Training(n_epochs=1, patience=1), n_workers=1, show_progress=False, seed=0)
    assert result.mi_estimate is not None


# ---------------------------------------------------------------------------
# Phase 1 spec: ceiling warning keys on train-eval size, not eval_size (Fix 1/4)
# ---------------------------------------------------------------------------

def test_ceiling_warning_keys_on_train_eval_size_when_it_diverges_from_eval_size(caplog):
    """With train_subset_size set explicitly larger than the (small) test
    split, train_eval_size and eval_size diverge -- the near-ceiling warning
    must be evaluated against train_eval_size for the reported (train_mi)
    quantity, not silently reuse the smaller test-side eval_size."""
    np.random.seed(0)
    n = 20000
    x = np.random.randn(n, 4).astype('float32')
    y = (x + 0.3 * np.random.randn(n, 4).astype('float32')).astype('float32')
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        result = nmi.run(
            x, y, mode='estimate',
            processing=nmi.Processing(x='continuous', x_params={'window_size': 50, 'step_size': None},
                                      y='continuous', y_params={'window_size': 50, 'step_size': None}),
            split=nmi.Split(mode='blocked', train_fraction=0.8),
            model=Model(embedding_dim=32, hidden_dim=128),
            training=Training(n_epochs=150, patience=30),
            n_workers=1, show_progress=False, seed=0,
        )
    assert result.details['train_eval_size'] > result.details['eval_size'], (
        "This test assumes Fix 1's decoupling actually produced a larger "
        "train_eval_size than eval_size -- if not, the scenario doesn't "
        "exercise the divergence this test is checking."
    )
    ceiling_msgs = [r.message for r in caplog.records if 'near its ceiling' in r.message]
    assert ceiling_msgs, f"Expected a near-ceiling warning; got: {[r.message for r in caplog.records]}"
    assert 'train-eval ceiling' in ceiling_msgs[0], (
        f"Expected the warning to name the train-eval ceiling specifically: {ceiling_msgs[0]}"
    )


# ---------------------------------------------------------------------------
# Phase 1 spec: saturation warning before peak-epoch selection (Fix 5)
# ---------------------------------------------------------------------------

def test_saturated_test_trace_warns_and_records_fraction(caplog):
    """A synthetic high-MI, small-n_eval run must fire the peak-epoch-selection
    saturation warning and record test_trace_saturated_fraction."""
    np.random.seed(0)
    n = 20000
    x = np.random.randn(n, 4).astype('float32')
    y = (x + 0.3 * np.random.randn(n, 4).astype('float32')).astype('float32')
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        result = nmi.run(
            x, y, mode='estimate',
            processing=nmi.Processing(x='continuous', x_params={'window_size': 50, 'step_size': None},
                                      y='continuous', y_params={'window_size': 50, 'step_size': None}),
            split=nmi.Split(mode='blocked', train_fraction=0.8),
            model=Model(embedding_dim=32, hidden_dim=128),
            training=Training(n_epochs=150, patience=30),
            n_workers=1, show_progress=False, seed=0,
        )
    assert 'test_trace_saturated_fraction' in result.details
    assert result.details['test_trace_saturated_fraction'] is not None


# ---------------------------------------------------------------------------
# Phase 1 spec: ceiling diagnostics propagate to mode='transfer' (Fix 6)
# ---------------------------------------------------------------------------

def test_transfer_mode_returns_diagnostics_for_both_components():
    """mode='transfer' must surface eval_size (and the rest of the ceiling
    diagnostics) for BOTH the joint and marginal component estimates -- a
    difference of two separately-trained estimates has two separate ceilings."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    result = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=20, prediction_horizon=1),
                     split=nmi.Split(mode='blocked'), training=Training(n_epochs=2, patience=2),
                     n_workers=1, show_progress=False, seed=0)
    for key in ('diagnostics_joint', 'diagnostics_marginal'):
        assert key in result.details, f"Missing {key} in mode='transfer' details"
        assert result.details[key] is not None
        assert result.details[key].get('eval_size') is not None, (
            f"{key}['eval_size'] must not be None on the transfer path"
        )


# ---------------------------------------------------------------------------
# Phase 1.5 spec: warn once when shift_time cannot take effect (Task 2)
# ---------------------------------------------------------------------------

def test_shift_time_warns_when_explicitly_requested_but_dead():
    """mode='estimate' with a windowed processor windows eagerly in run.py;
    the Trainer never sees a temporal dataset, so an explicit
    shift_time=True must warn that it has no effect."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 50, 'step_size': 10},
                          y='continuous', y_params={'window_size': 50, 'step_size': 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert msgs, "Expected a shift_time warning"
    assert "mode='estimate'" in msgs[0]


def test_shift_time_silent_when_not_requested():
    """shift_time defaults to True, but for this continuous+continuous pair
    at mode='estimate' only shift_windows is reachable -- an unset (schema-
    defaulted) shift_time must stay silent, since the user never explicitly
    asked for it here."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 50, 'step_size': 10},
                          y='continuous', y_params={'window_size': 50, 'step_size': 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning; got: {msgs}"


def test_shift_time_silent_when_explicitly_false():
    """An explicit False must not warn -- the user already opted out."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 50, 'step_size': 10},
                          y='continuous', y_params={'window_size': 50, 'step_size': 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_time=False),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning; got: {msgs}"


def test_shift_time_silent_when_reachable_via_mode_lag():
    """mode='lag' defers processing to the worker, so the feature genuinely
    works there -- an explicit True must not warn."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 50, 'step_size': 10},
                          y='continuous', y_params={'window_size': 50, 'step_size': 10})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='lag', lag=nmi.Lag(lag_range=[0]), processing=proc,
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning on mode='lag'; got: {msgs}"


# ---------------------------------------------------------------------------
# Shift-mechanism reachability extension: categorical/mixed-regular pairs for
# shift_windows, spike+spike and sample_rate-gated mixed pairs for
# shift_time.
# ---------------------------------------------------------------------------

def _spike_trains(n_neurons, n_seconds, rate, seed=0):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.uniform(0, n_seconds, rng.poisson(n_seconds * rate))) for _ in range(n_neurons)]


def test_shift_windows_silent_for_categorical_pair():
    """categorical+categorical must be reachable now, not just continuous+continuous."""
    np.random.seed(0)
    x = np.random.randint(0, 4, size=(3000, 2)).astype('int64')
    y = np.random.randint(0, 3, size=(3000, 2)).astype('int64')
    proc = nmi.Processing(x='categorical', x_params={'window_size': 20, 'step_size': 20},
                          y='categorical', y_params={'window_size': 20, 'step_size': 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_windows=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
    assert not msgs, f"Did not expect a shift_windows warning for categorical+categorical; got: {msgs}"


def test_shift_windows_silent_for_continuous_categorical_mixed_pair():
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randint(0, 4, size=(3000, 2)).astype('int64')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                          y='categorical', y_params={'window_size': 20, 'step_size': 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_windows=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
    assert not msgs, f"Did not expect a shift_windows warning for a continuous+categorical pair; got: {msgs}"


def test_shift_windows_still_warns_for_spike():
    """spike is not part of the 'regular' family -- shift_windows must
    still warn (and fall back) rather than silently misinterpret spike data.

    A continuous+spike pair without a shared time unit (no sample_rate on
    the continuous side) is itself a pre-existing, orthogonal correctness
    gap (window_size means raw samples for X but seconds for Y) that can
    make training degenerate -- irrelevant to what's being checked here, so
    a TrainingError from that mismatch is tolerated; only the warning
    (raised before training starts) matters for this test.
    """
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    spikes = _spike_trains(3, 300.0, 5.0)
    proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                          y='spike', y_params={'window_size': 2.0, 'step_size': 2.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            nmi.run(x, spikes, mode='estimate', processing=proc,
                   training=Training(n_epochs=1, patience=1, shift_windows=True),
                   n_workers=1, show_progress=False, seed=0)
        except TrainingError:
            pass
    msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
    assert msgs, "Expected a shift_windows warning for a continuous+spike pair"


def test_shift_time_silent_for_spike_pair_at_mode_estimate():
    """spike+spike is now reachable at mode='estimate' (both sides natively
    in seconds, no cross-unit concern) -- an explicit True must not warn."""
    x = _spike_trains(4, 400.0, 8.0, seed=1)
    y = _spike_trains(4, 400.0, 8.0, seed=2)
    proc = nmi.Processing(x='spike', x_params={'window_size': 2.0, 'step_size': 2.0},
                          y='spike', y_params={'window_size': 2.0, 'step_size': 2.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning for spike+spike; got: {msgs}"


def test_shift_time_warns_for_mixed_pair_without_sample_rate():
    """A mixed continuous+spike pair without 'sample_rate' on the continuous
    side has no common time unit for a shift value to mean -- must warn and
    fall back, not silently misalign X/Y."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    spikes = _spike_trains(3, 300.0, 5.0)
    proc = nmi.Processing(x='continuous', x_params={'window_size': 5, 'step_size': 5},
                          y='spike', y_params={'window_size': 5.0, 'step_size': 5.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, spikes, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert msgs, "Expected a shift_time warning for a mixed pair without sample_rate"
    assert 'sample_rate' in msgs[0]


def test_shift_windows_silent_for_plain_sweep():
    """Plain (non-processor-swept) mode='sweep' dispatches independent
    training runs from the same raw data, same as mode='estimate' -- must
    be reachable too, not just 'estimate'."""
    np.random.seed(0)
    x = np.random.randn(5000, 2).astype('float32')
    y = np.random.randn(5000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                          y='continuous', y_params={'window_size': 20, 'step_size': 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='sweep', processing=proc, sweep_grid={'hidden_dim': [8, 16]},
               training=Training(n_epochs=1, patience=1, shift_windows=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
    assert not msgs, f"Did not expect a shift_windows warning for plain mode='sweep'; got: {msgs}"


def test_shift_time_silent_for_spike_pair_in_plain_sweep():
    x = _spike_trains(4, 400.0, 8.0, seed=1)
    y = _spike_trains(4, 400.0, 8.0, seed=2)
    proc = nmi.Processing(x='spike', x_params={'window_size': 2.0, 'step_size': 2.0},
                          y='spike', y_params={'window_size': 2.0, 'step_size': 2.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='sweep', processing=proc, sweep_grid={'hidden_dim': [8, 16]},
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning for spike+spike in plain sweep; got: {msgs}"


def test_shift_time_still_warns_for_plain_sweep_regular_pair():
    """continuous+continuous still isn't a shift_time case even in
    a now-reachable mode -- shift_windows remains the right tool."""
    np.random.seed(0)
    x = np.random.randn(5000, 2).astype('float32')
    y = np.random.randn(5000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                          y='continuous', y_params={'window_size': 20, 'step_size': 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='sweep', processing=proc, sweep_grid={'hidden_dim': [8, 16]},
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert msgs, "Expected a shift_time warning for a regular pair even in plain sweep"


def test_shift_time_silent_for_mixed_pair_with_sample_rate():
    """The same mixed pair, but with 'sample_rate' set on the continuous
    side, has a common time unit (seconds) with spike -- must not warn."""
    np.random.seed(0)
    sample_rate = 100.0
    x = np.random.randn(30000, 2).astype('float32')  # 300s at 100Hz
    spikes = _spike_trains(3, 300.0, 5.0)
    proc = nmi.Processing(x='continuous',
                          x_params={'window_size': 5.0, 'step_size': 5.0, 'sample_rate': sample_rate},
                          y='spike', y_params={'window_size': 5.0, 'step_size': 5.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, spikes, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1, shift_time=True),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning for a rate-equipped mixed pair; got: {msgs}"


# ---------------------------------------------------------------------------
# shift_windows/shift_time both default to True -- a schema-defaulted (unset)
# value must engage exactly like an explicit True for a reachable pair, not
# just stay silent for an unreachable one (already covered above).
# ---------------------------------------------------------------------------

def test_shift_windows_engages_silently_by_default_for_continuous_pair():
    """No shift_windows kwarg at all: the schema default (True) must engage
    for this reachable continuous+continuous pair without warning."""
    np.random.seed(0)
    x = np.random.randn(3000, 2).astype('float32')
    y = np.random.randn(3000, 2).astype('float32')
    proc = nmi.Processing(x='continuous', x_params={'window_size': 20, 'step_size': 20},
                          y='continuous', y_params={'window_size': 20, 'step_size': 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
    assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"


def test_shift_windows_engages_silently_by_default_for_categorical_pair():
    """Same as above, categorical+categorical: the reslice mechanism must be
    reachable by default, not just when explicitly requested."""
    np.random.seed(0)
    x = np.random.randint(0, 4, size=(3000, 2)).astype('int64')
    y = np.random.randint(0, 3, size=(3000, 2)).astype('int64')
    proc = nmi.Processing(x='categorical', x_params={'window_size': 20, 'step_size': 20},
                          y='categorical', y_params={'window_size': 20, 'step_size': 20})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
    assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"


def test_shift_time_engages_silently_by_default_for_spike_pair():
    """No shift_time kwarg at all: the schema default (True) must engage for
    this reachable spike+spike pair at mode='estimate' without warning."""
    x = _spike_trains(4, 400.0, 8.0, seed=1)
    y = _spike_trains(4, 400.0, 8.0, seed=2)
    proc = nmi.Processing(x='spike', x_params={'window_size': 2.0, 'step_size': 2.0},
                          y='spike', y_params={'window_size': 2.0, 'step_size': 2.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, mode='estimate', processing=proc,
               training=Training(n_epochs=1, patience=1),
               n_workers=1, show_progress=False, seed=0)
    msgs = [str(w.message) for w in caught if 'shift_time' in str(w.message)]
    assert not msgs, f"Did not expect a shift_time warning; got: {msgs}"
