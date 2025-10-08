# tests/test_integration.py
import pytest
import torch
import numpy as np
import pandas as pd
import neural_mi as nmi

BASE_PARAMS_TEST = {
    'n_epochs': 2, 'learning_rate': 1e-4, 'batch_size': 32,
    'patience': 1, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1
}

@pytest.fixture
def raw_data():
    x, y = nmi.datasets.generate_correlated_gaussians(n_samples=200, dim=5, mi=2.0)
    return x.T, y.T # Return as (channels, timepoints)

def test_run_estimate_mode(raw_data):
    x, y = raw_data
    results = nmi.run(
        x_data=x, y_data=y, mode='estimate',
        processor_type='continuous', processor_params={'window_size': 1},
        base_params=BASE_PARAMS_TEST, random_seed=42, n_workers=1
    )
    assert isinstance(results.mi_estimate, float)
    assert not np.isnan(results.mi_estimate)

def test_run_sweep_mode(raw_data):
    x, y = raw_data
    results = nmi.run(
        x_data=x, y_data=y, mode='sweep',
        processor_type='continuous', processor_params={'step_size': 1},
        base_params=BASE_PARAMS_TEST, sweep_grid={'window_size': [5, 10]},
        random_seed=42, n_workers=1
    )
    assert isinstance(results.dataframe, pd.DataFrame)
    assert len(results.dataframe) == 2

def test_run_dimensionality_mode():
    x, _ = nmi.datasets.generate_nonlinear_from_latent(200, 3, 20, 2.0)
    results = nmi.run(
        x_data=x.T, mode='dimensionality',
        processor_type='continuous', processor_params={'window_size': 1},
        base_params=BASE_PARAMS_TEST, sweep_grid={'embedding_dim': [2, 4]},
        n_splits=2, random_seed=42, n_workers=1
    )
    assert isinstance(results.dataframe, pd.DataFrame)
    assert 'mi_mean' in results.dataframe.columns