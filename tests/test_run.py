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
    result = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='rigorous',
        base_params=BASE_PARAMS,
        output_units='nats',
        gamma_range=range(1, 2),
        n_workers=1
    )
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
        processor_type_x='continuous',
        processor_params_x={'window_size': 10, 'step_size': 5},
        base_params=BASE_PARAMS,
        output_units='nats',
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)

def test_run_rigorous_mode_returns_results_with_details(gaussian_data):
    """
    Verifies that mode='rigorous' returns a Results object with mi_estimate, dataframe, and details.
    """
    # Original gaussian_data fixture is too small (200 samples)
    # Generate larger data for this specific test
    x_data, y_data = nmi.datasets.generate_correlated_gaussians(
        n_samples=1000, dim=5, mi=2.0
    )
    x_data = x_data.reshape(1000, 1, 5)
    y_data = y_data.reshape(1000, 1, 5)

    result = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='rigorous',
        base_params=BASE_PARAMS,
        output_units='nats',
        gamma_range=range(1, 4), # Use a smaller gamma range for speed
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert isinstance(result.details, dict)
    assert 'mi_error' in result.details

# Define the custom critic class at the module level (outside the test function)
class MyCustomCritic(nmi.models.BaseCritic):
    def __init__(self):
        super().__init__()
        # This layer's parameters will be used to connect to the graph
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        batch_size = x.shape[0]
        # Create the base tensor
        scores = torch.ones(batch_size, batch_size, device=x.device)
        # Multiply by a parameter to ensure it's part of the computation graph
        # This doesn't change the value but fixes the gradient issue.
        scores = scores + self.dummy_param * 0
        return scores, torch.tensor(0.0, device=x.device)

def test_run_with_custom_critic(gaussian_data):
    """
    Tests that the `run` function can accept a pre-initialized custom critic.
    """
    x_data, y_data = gaussian_data
    custom_critic_instance = MyCustomCritic()

    result = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='estimate',
        base_params=BASE_PARAMS,
        custom_critic=custom_critic_instance, # Pass the instance here
        n_workers=1,
        output_units='nats' # Use nats for direct comparison with np.log
    )
    assert isinstance(result, nmi.results.Results)
    assert isinstance(result.mi_estimate, float)
    
    # For a score matrix of all 1s, the InfoNCE bound is log(N) + E[diag - logsumexp]
    # which calculates to log(N) + 1 - (log(exp(1)*N)) = log(N) + 1 - (1 + log(N)) = 0.
    assert np.isclose(result.mi_estimate, 0.0, atol=1e-6)