# tests/test_precision.py
import pytest
import torch
import pandas as pd
import numpy as np

from neural_mi.analysis.precision import apply_corruption, run_precision_analysis

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