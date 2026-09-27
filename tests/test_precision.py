# tests/test_precision.py
import numpy as np
import pytest
import torch
import pandas as pd

import neural_mi.analysis.precision as precision_module
from neural_mi.analysis.precision import run_precision_analysis
from neural_mi.data.corruption import corrupt
from neural_mi.data.shift_windowing import WindowShifter, PairedWindowShifter
from neural_mi.training.trainer import Trainer


def test_rounding_moves_values_to_bin_centres():
    data = torch.tensor([0.2, 0.8, 1.2, 1.8, 2.1, -0.3])
    corrupted = corrupt(data, 1.0, 'rounding')
    assert torch.allclose(corrupted, torch.tensor([0.5, 0.5, 1.5, 1.5, 2.5, -0.5]))


def test_rounding_is_idempotent():
    data = torch.rand(500) * 3
    once = corrupt(data, 0.25, 'rounding')
    assert torch.allclose(corrupt(once, 0.25, 'rounding'), once)


def test_noise_is_bounded_by_half_tau():
    data = torch.full((1000,), 5.0)
    corrupted = corrupt(data, 2.0, 'noise')
    assert torch.all((corrupted >= 4.0) & (corrupted <= 6.0))
    assert torch.abs(torch.mean(corrupted) - 5.0) < 0.1


@pytest.mark.parametrize('method', ['rounding', 'noise'])
@pytest.mark.parametrize('empty', [0.0, -1.0])
def test_entries_without_a_measurement_are_never_corrupted(method, empty):
    """No method creates a spike in an unused slot or moves a spike onto one."""
    data = torch.full((200, 3, 8), empty)
    data[:, :, :3] = torch.rand(200, 3, 3) * 0.2      # spikes early in the window
    corrupted = corrupt(data, 0.3, method, empty)
    held = data != empty
    assert torch.all(corrupted[~held] == empty)
    assert torch.all(corrupted[held] != empty)


def test_tau_zero_leaves_the_data_unchanged():
    data = torch.rand(10)
    for method in ('rounding', 'noise'):
        assert torch.equal(corrupt(data, 0.0, method), data)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="rounding"):
        corrupt(torch.rand(3), 0.1, 'jitter')


def test_empty_value_follows_each_sides_processor():
    empty = precision_module._empty_value
    spikes = {'processor_type_x': 'spike', 'processor_params_x': {'window_size': 1.0}}
    assert empty(spikes, 'x') == 0.0
    assert empty({**spikes, 'processor_params_x': {'no_spike_value': -1.0}}, 'x') == -1.0
    # Binned spikes mark an empty bin with zero, whatever no_spike_value says.
    assert empty({**spikes, 'processor_params_x': {'bin_size': 0.1, 'no_spike_value': -1.0}}, 'x') == 0.0
    # Y reads with X's processor and parameters when it names none.
    assert empty({**spikes, 'processor_params_x': {'no_spike_value': -1.0}}, 'y') == -1.0
    assert empty({'processor_type_x': 'continuous'}, 'x') == 0.0


def test_precision_mode_never_corrupts_unused_spike_slots(monkeypatch):
    """Every tensor the frozen critic is scored on keeps the unused slots empty."""
    import neural_mi as nmi
    from neural_mi import generators
    x, y, _ = generators.generate_spike_pair(n_windows=300, window_size=1.0, coding='timing', seed=0)
    seen = []
    original = Trainer._safe_eval_mi

    def spy(self, x_eval, y_eval, max_eval):
        seen.append(x_eval.detach().cpu())
        return original(self, x_eval, y_eval, max_eval)

    monkeypatch.setattr(Trainer, '_safe_eval_mi', spy)
    nmi.run(x, y, mode='precision',
            processing=nmi.Processing(x='spike', x_params={'window_size': 1.0}, y='spike'),
            model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
            training=nmi.Training(n_epochs=1, batch_size=64, shift_time=False),
            precision=nmi.Precision(tau_grid=[0.05, 0.3], corruption_method='noise', n_noise_samples=2),
            show_progress=False, seed=0, device='cpu')
    # The sweep scores the training partition, clean data first; the other calls are
    # the held-out evaluations made during training.
    sweep = [t for t in seen if t.shape == seen[-1].shape]
    clean = sweep[0]
    empty = clean == 0.0
    assert empty.any() and len(sweep) >= 5   # tau=0, then 2 draws at each of 2 taus
    for corrupted in sweep[1:]:
        assert torch.all(corrupted[empty] == 0.0)
        assert torch.all(corrupted[~empty] != 0.0)


def test_threshold_ratio_accepts_a_list():
    import neural_mi as nmi
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2))
    result = nmi.run(x, x + 0.5 * rng.standard_normal((300, 2)), mode='precision',
                     model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                     training=nmi.Training(n_epochs=1, batch_size=64),
                     precision=nmi.Precision(tau_grid=[0.5, 1.0], threshold_ratio=[0.9, 0.5]),
                     show_progress=False, seed=0, device='cpu')
    assert set(result.get('precision_thresholds')) == {0.9, 0.5}


def test_precision_mode_uses_the_callers_split(monkeypatch):
    import neural_mi as nmi
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2))
    train, test = np.arange(0, 240), np.arange(240, 300)
    received = {}
    original = Trainer.train

    def spy(self, dataset, *args, **kwargs):
        received['train'], received['test'] = kwargs.get('train_indices'), kwargs.get('test_indices')
        return original(self, dataset, *args, **kwargs)

    monkeypatch.setattr(Trainer, 'train', spy)
    nmi.run(x, x + 0.5 * rng.standard_normal((300, 2)), mode='precision',
            model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
            training=nmi.Training(n_epochs=1, batch_size=64),
            split=nmi.Split(train_indices=train, test_indices=test),
            precision=nmi.Precision(tau_grid=[0.5]),
            show_progress=False, seed=0, device='cpu')
    assert np.array_equal(received['train'], train) and np.array_equal(received['test'], test)


def test_a_split_the_library_chooses_raises_no_custom_split_warning(caplog):
    import logging
    import neural_mi as nmi
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2))
    with caplog.at_level(logging.WARNING, logger='neural_mi'):
        nmi.run(x, x + 0.5 * rng.standard_normal((300, 2)), mode='precision',
                model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                training=nmi.Training(n_epochs=1, batch_size=64),
                precision=nmi.Precision(tau_grid=[0.5]),
                show_progress=False, seed=0, device='cpu')
    assert 'Custom train_indices' not in caplog.text


def test_run_precision_analysis_end_to_end():
    """Tests that the precision sweep trains a model and evaluates the tau grid."""
    x_data = torch.randn(100, 2)
    y_data = torch.randn(100, 2)
    
    base_params = {
        'critic_type': 'separable',
        'n_epochs': 1,       # Keep training lightning fast for the test
        'batch_size': 10,
        'learning_rate': 5e-4,
        'device': 'cpu',
        'input_dim_x': 2,
        'input_dim_y': 2,
        'hidden_dim': 8,
        'embedding_dim': 4,
        'n_layers': 1,
        'use_variational': False,
        'embedding_model': 'mlp',
        'max_n_batches': 512,
        'kernel_size': 3,
        'bidirectional': False,
        'nhead': 4
    }
    
    tau_grid = [0.1, 0.5, 1.0, 5.0]
    
    results = run_precision_analysis(
        x_data, y_data, base_params, 
        tau_grid=tau_grid, 
        corrupt_target='x', 
        corruption_method='rounding',
        threshold_ratio=0.9
    )
    
    # 1. Check Output Structure
    assert 'dataframe' in results
    assert 'details' in results
    
    df = results['dataframe']
    details = results['details']
    
    # 2. Check DataFrame
    assert isinstance(df, pd.DataFrame)
    assert 'tau' in df.columns
    assert 'train_mi' in df.columns
    assert len(df) == 5 # 4 tau values + the 0.0 baseline
    
    # 3. Check Details
    assert 'baseline_mi' in details
    assert 'precision_tau' in details
    assert details['corrupt_target'] == 'x'


def test_run_precision_analysis_corrupt_target_both():
    """corrupt_target='both' should run without error and tag results correctly."""
    x_data = torch.randn(100, 2)
    y_data = torch.randn(100, 2)
    base_params = {
        'critic_type': 'separable',
        'n_epochs': 1,
        'batch_size': 10,
        'learning_rate': 5e-4,
        'device': 'cpu',
        'input_dim_x': 2,
        'input_dim_y': 2,
        'hidden_dim': 8,
        'embedding_dim': 4,
        'n_layers': 1,
        'use_variational': False,
        'embedding_model': 'mlp',
        'max_n_batches': 512,
        'kernel_size': 3,
        'bidirectional': False,
        'nhead': 4,
    }
    results = run_precision_analysis(
        x_data, y_data, base_params,
        tau_grid=[0.5, 1.0],
        corrupt_target='both',
        corruption_method='rounding',
        threshold_ratio=0.9,
    )
    assert results['details']['corrupt_target'] == 'both'
    assert 'dataframe' in results


def test_run_precision_analysis_shift_windows_reaches_reachable_pair():
    """shift_windows=True with a real continuous processor + window_size
    must engage (build a shift-capable dataset) rather than being silently
    dropped -- no crash, finite results."""
    np.random.seed(0)
    torch.manual_seed(0)
    T, C, window_size = 3000, 2, 20
    x_data = np.random.randn(T, C).astype('float32')
    y_data = np.random.randn(T, C).astype('float32')
    base_params = {
        'critic_type': 'separable', 'n_epochs': 2, 'batch_size': 16,
        'learning_rate': 5e-4, 'device': 'cpu', 'hidden_dim': 8,
        'embedding_dim': 4, 'n_layers': 1, 'use_variational': False,
        'embedding_model': 'mlp', 'max_n_batches': 512, 'kernel_size': 3,
        'bidirectional': False, 'nhead': 4,
        'processor_type_x': 'continuous',
        'processor_params_x': {'window_size': window_size, 'step_size': window_size},
        'shift_windows': True,
    }
    results = run_precision_analysis(
        x_data, y_data, base_params, tau_grid=[0.1, 0.5], corrupt_target='x',
        corruption_method='rounding', threshold_ratio=0.9,
    )
    assert np.isfinite(results['details']['baseline_mi'])
    assert np.all(np.isfinite(results['dataframe']['train_mi'].values))


def test_run_precision_analysis_corruption_sweep_uses_frozen_snapshot_not_live_shift_state(monkeypatch):
    """The corruption sweep must corrupt the frozen, canonical (pre-shift)
    view -- not whatever shift state the live dataset happens to be in
    when training ends. Forces every drawn shift to a large, fixed,
    non-zero value so the dataset's live .data is guaranteed to differ
    from the canonical shift=0 view by the time training completes, then
    checks the tau=0.0 corruption call (a no-op, so it receives
    x_train_raw/y_train_raw unchanged) against an independently
    reconstructed canonical view."""
    np.random.seed(0)
    torch.manual_seed(0)
    T, C, window_size = 3000, 2, 20
    x_data = np.random.randn(T, C).astype('float32')
    y_data = np.random.randn(T, C).astype('float32')

    monkeypatch.setattr(WindowShifter, 'random_shift', lambda self, generator=None: window_size - 1)

    _captured_split = {}
    _real_create_blocked_split = Trainer._create_blocked_split

    def _capture_split(self, *args, **kwargs):
        result = _real_create_blocked_split(self, *args, **kwargs)
        _captured_split['train_idx'], _captured_split['test_idx'] = result
        return result
    monkeypatch.setattr(Trainer, '_create_blocked_split', _capture_split)

    _captured_tau0 = []
    _real_corrupt = precision_module.corrupt

    def _capture_corruption(data, tau, method, empty_value=0.0):
        if tau == 0.0:
            _captured_tau0.append(data.clone())
        return _real_corrupt(data, tau, method, empty_value)
    monkeypatch.setattr(precision_module, 'corrupt', _capture_corruption)

    base_params = {
        'critic_type': 'separable', 'n_epochs': 3, 'batch_size': 16,
        'learning_rate': 5e-4, 'device': 'cpu', 'hidden_dim': 8,
        'embedding_dim': 4, 'n_layers': 1, 'use_variational': False,
        'embedding_model': 'mlp', 'max_n_batches': 512, 'kernel_size': 3,
        'bidirectional': False, 'nhead': 4,
        'processor_type_x': 'continuous',
        'processor_params_x': {'window_size': window_size, 'step_size': window_size},
        'shift_windows': True,
    }
    run_precision_analysis(
        x_data, y_data, base_params, tau_grid=[0.5], corrupt_target='both',
        corruption_method='rounding', threshold_ratio=0.9,
    )

    assert len(_captured_tau0) == 2, "Expected one tau=0.0 corruption call each for x and y"
    train_idx = _captured_split['train_idx']

    # Independently reconstruct the canonical (shift=0) view from the same
    # raw arrays -- deterministic regardless of what shift the live dataset
    # ended training at.
    raw_x = torch.as_tensor(x_data)
    raw_y = torch.as_tensor(y_data)
    shifter = PairedWindowShifter(raw_x, raw_y, window_size, window_size)
    x0, y0 = shifter.windows_at(0)

    torch.testing.assert_close(_captured_tau0[0], x0[train_idx])
    torch.testing.assert_close(_captured_tau0[1], y0[train_idx])

def test_precision_windows_on_the_timestamps_like_every_other_mode(monkeypatch):
    """mode='precision' builds its own dataset, and it windows on the
        caller's timestamps as every other mode does: a 25 Hz recording with
        window_size=1.0 gives 80 windows 25 samples wide. The spy stops the run
        at the build, which is the only part under test.
    """

    class Built(Exception):
        pass

    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise Built

    monkeypatch.setattr(precision_module, 'create_dataset', spy)

    t = np.arange(0, 80, 1 / 25.0)
    x = np.sin(2 * np.pi * 0.3 * t)[:, None]
    y = np.cos(2 * np.pi * 0.3 * t)[:, None]
    base_params = {
        'x_time': t, 'y_time': t,
        'shift_windows': False,
        'processor_type_x': 'continuous', 'processor_type_y': 'continuous',
        'processor_params_x': {'window_size': 1.0, 'step_size': 1.0},
        'processor_params_y': {'window_size': 1.0, 'step_size': 1.0},
    }
    with pytest.raises(Built):
        precision_module.run_precision_analysis(x, y, base_params, tau_grid=[0.1])

    assert seen.get('x_time') is not None
    assert seen.get('y_time') is not None

    # And the grid that reaches training is the one the timestamps describe.
    from neural_mi.data.handler import create_dataset
    dataset = create_dataset(x, y, x_time=t, y_time=t,
                             processor_type_x='continuous',
                             processor_params_x={'window_size': 1.0, 'step_size': 1.0},
                             processor_type_y='continuous',
                             processor_params_y={'window_size': 1.0, 'step_size': 1.0})
    assert dataset.x_data.shape[2] == 25
    assert len(dataset) == 80
