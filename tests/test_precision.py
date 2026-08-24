# tests/test_precision.py
import numpy as np
import torch
import pandas as pd

import neural_mi.analysis.precision as precision_module
from neural_mi.analysis.precision import apply_corruption, run_precision_analysis
from neural_mi.data.shift_windowing import WindowShifter, PairedWindowShifter
from neural_mi.training.trainer import Trainer

def test_apply_corruption_rounding():
    """Tests the deterministic rounding mathematical logic."""
    # We use unambiguous floats to avoid round-to-even edge cases in the test
    data = torch.tensor([0.2, 0.8, 1.2, 1.8, 2.1])
    tau = 1.0
    
    corrupted = apply_corruption(data, tau, 'rounding')
    
    # Expected: [0.0, 1.0, 1.0, 2.0, 2.0]
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0])
    assert torch.allclose(corrupted, expected)

def test_apply_corruption_noise():
    """Tests that additive uniform noise is bounded correctly."""
    data = torch.zeros(1000) # Baseline of all zeros
    tau = 2.0
    
    corrupted = apply_corruption(data, tau, 'noise')
    
    # Noise should be uniformly distributed between [-tau/2, tau/2], which is [-1.0, 1.0]
    assert torch.all(corrupted >= -1.0)
    assert torch.all(corrupted <= 1.0)
    # Mean should be very close to 0
    assert torch.abs(torch.mean(corrupted)) < 0.1

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
    _real_apply_corruption = precision_module.apply_corruption

    def _capture_corruption(data, tau, method):
        if tau == 0.0:
            _captured_tau0.append(data.clone())
        return _real_apply_corruption(data, tau, method)
    monkeypatch.setattr(precision_module, 'apply_corruption', _capture_corruption)

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

    assert len(_captured_tau0) == 2, "Expected one tau=0.0 apply_corruption call each for x and y"
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