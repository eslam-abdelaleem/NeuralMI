import pytest
import torch
import numpy as np
import pandas as pd
import neural_mi as nmi
from neural_mi.results import Results

# Base parameters that are used across multiple tests
BASE_PARAMS = {
    'n_epochs': 1,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'patience': 1,
    'embedding_dim': 8,
    'hidden_dim': 32,
    'n_layers': 1
}

@pytest.fixture
def gaussian_data():
    """Generate pre-processed 3D correlated Gaussian data."""
    x_data, y_data = nmi.datasets.generate_correlated_gaussians(
        n_samples=200, dim=5, mi=2.0
    )
    x_data_3d = x_data.reshape(200, 1, 5)
    y_data_3d = y_data.reshape(200, 1, 5)
    return x_data_3d, y_data_3d

@pytest.fixture
def raw_gaussian_data():
    """Generate raw 2D correlated Gaussian data."""
    x_data, y_data = nmi.datasets.generate_correlated_gaussians(
        n_samples=500, dim=5, mi=2.0
    )
    return x_data, y_data

def test_run_estimate_mode_returns_results_with_float(gaussian_data):
    """
    Verifies that mode='estimate' returns a Results object with a float mi_estimate.
    """
    x_data, y_data = gaussian_data
    result = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='estimate',
        base_params=BASE_PARAMS,
        output_units='nats',
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)
    assert result.dataframe is None

def test_run_sweep_mode_returns_results_with_dataframe(gaussian_data):
    """
    Verifies that mode='sweep' returns a Results object with a DataFrame.
    """
    x_data, y_data = gaussian_data
    sweep_grid = {'embedding_dim': [4, 8]}
    result = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='sweep',
        base_params=BASE_PARAMS,
        sweep_grid=sweep_grid,
        output_units='nats',
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert 'embedding_dim' in result.dataframe.columns
    assert len(result.dataframe) == 2
    assert result.mi_estimate is None

def test_run_rigorous_mode_returns_results_with_details(gaussian_data):
    """
    Verifies that mode='rigorous' returns a Results object with mi_estimate, dataframe, and details.
    """
    x_data, y_data = gaussian_data
    try:
        result = nmi.run(
            x_data=x_data,
            y_data=y_data,
            mode='rigorous',
            base_params=BASE_PARAMS,
            output_units='nats',
            gamma_range=range(1, 6),  # Increased range to avoid fit error
            n_workers=1,
            min_gamma_points=2, # Lower requirement for test speed
            delta_threshold=1000.0 # Make pruning very lenient for this noisy test
        )
    except ValueError as e:
        pytest.fail(f"Rigorous mode failed unexpectedly with ValueError: {e}")

    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert isinstance(result.details, dict)
    assert 'mi_error' in result.details

def test_run_dimensionality_mode_returns_results_with_dataframe(gaussian_data):
    """
    Verifies that mode='dimensionality' returns a Results object with a DataFrame.
    """
    x_data, _ = gaussian_data
    x_data_multi_channel = x_data.reshape(200, 5, 1)
    sweep_grid = {'embedding_dim': [4, 8]}
    result = nmi.run(
        x_data=x_data_multi_channel,
        mode='dimensionality',
        base_params=BASE_PARAMS,
        sweep_grid=sweep_grid,
        output_units='nats',
        n_splits=1,
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert 'mi_mean' in result.dataframe.columns
    assert 'embedding_dim' in result.dataframe.columns
    assert result.mi_estimate is None

def test_run_with_continuous_processor_returns_results(raw_gaussian_data):
    """
    Tests that the raw data pipeline returns a correct Results object.
    """
    x_raw, y_raw = raw_gaussian_data
    result = nmi.run(
        x_data=x_raw,
        y_data=y_raw,
        mode='estimate',
        processor_type='continuous',
        processor_params={'window_size': 10, 'step_size': 5, 'data_format': 'channels_last'},
        base_params=BASE_PARAMS,
        output_units='nats',
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)