# tests/test_splitting.py
import pytest
import torch
import numpy as np
import neural_mi as nmi

@pytest.fixture
def iid_data():
    """Generate simple IID Gaussian data."""
    x_data, y_data = nmi.datasets.generate_correlated_gaussians(
        n_samples=100, dim=2, mi=1.0
    )
    # Reshape to be (n_samples, n_channels, n_features)
    return x_data.unsqueeze(1), y_data.unsqueeze(1)

def test_random_split_mode(iid_data):
    """
    Tests that the 'random' split mode runs without error and produces a result.
    """
    x_data, y_data = iid_data
    base_params = {
        'n_epochs': 2,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'embedding_dim': 4,
        'hidden_dim': 16,
        'n_layers': 2,
        'patience': 1,
    }
    
    results = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='estimate',
        base_params=base_params,
        split_mode='random', # Explicitly test the new random split mode
        verbose=False
    )
    
    assert results.mi_estimate is not None
    assert not np.isnan(results.mi_estimate)

def test_custom_indices_split(iid_data):
    """
    Tests that providing custom train/test indices works correctly.
    """
    x_data, y_data = iid_data
    n_samples = x_data.shape[0]
    
    # Create custom indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:80]
    test_indices = indices[80:]
    
    base_params = {
        'n_epochs': 2,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'embedding_dim': 4,
        'hidden_dim': 16,
        'n_layers': 2,
        'patience': 1,
    }

    results = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='estimate',
        base_params=base_params,
        train_indices=train_indices,
        test_indices=test_indices,
        verbose=False
    )

    assert results.mi_estimate is not None
    assert not np.isnan(results.mi_estimate)

def test_default_blocked_split(iid_data):
    """
    Ensures the default 'blocked' split mode still works as expected.
    """
    x_data, y_data = iid_data
    base_params = {
        'n_epochs': 2,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'embedding_dim': 4,
        'hidden_dim': 16,
        'n_layers': 2,
        'patience': 1,
    }

    results = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='estimate',
        base_params=base_params,
        split_mode='blocked', # Explicitly test the default
        verbose=False
    )

    assert results.mi_estimate is not None
    assert not np.isnan(results.mi_estimate)