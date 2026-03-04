import pytest
import torch
import numpy as np
import pandas as pd
import neural_mi as nmi
from neural_mi.results import Results
from unittest.mock import patch

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
    x_data, y_data = nmi.generators.generate_correlated_gaussians(
        n_samples=200, dim=5, mi=2.0
    )
    # Shape: (n_samples, n_features, n_channels)
    x_data_3d = x_data.reshape(200, 1, 5)
    y_data_3d = y_data.reshape(200, 1, 5)
    return x_data_3d, y_data_3d

@pytest.fixture
def raw_gaussian_data():
    """Generate raw 2D correlated Gaussian data."""
    x_data, y_data = nmi.generators.generate_correlated_gaussians(
        n_samples=500, dim=5, mi=2.0
    )
    # run() expects (time, channels) for continuous data.
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
    # We can use the smaller, faster gaussian_data fixture now
    x_data, y_data = gaussian_data

    # Mock run_rigorous_analysis to prevent macOS multiprocessing serialization crashes during routing tests
    with patch('neural_mi.run.run_rigorous_analysis') as mock_rigorous:
        mock_rigorous.return_value = {
            'raw_results_df': pd.DataFrame([{'gamma': 1.0, 'test_mi': 2.0}]),
            'corrected_results': [{'mi_corrected': 2.5, 'mi_error': 0.1, 'slope': -0.05}]
        }

        result = nmi.run(
            x_data=x_data,
            y_data=y_data,
            mode='rigorous',
            base_params=BASE_PARAMS,
            output_units='nats',
            n_workers=1
        )
        
    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert isinstance(result.details, dict)
    assert 'mi_error' in result.details
    
def test_run_dimensionality_mode_returns_results_with_dataframe(raw_gaussian_data):
    """
    Verifies that mode='dimensionality' returns a Results object with the new spectral metrics.
    Uses raw 2D data (N, C) so that shape[1] gives the channel count correctly.
    """
    x_data, _ = raw_gaussian_data
    
    # We no longer need to sweep embedding_dim for dimensionality.
    # The new engine handles the large bottleneck automatically.
    result = nmi.run(
        x_data=x_data,
        mode='dimensionality',
        base_params=BASE_PARAMS,
        output_units='nats',
        split_method='random',
        n_splits=2,
        n_workers=1,
        device='cpu'
    )
    
    assert isinstance(result, Results)
    assert isinstance(result.dataframe, pd.DataFrame)
    # Check for our new spectral metrics instead of mi_mean
    assert 'participation_ratio_mean' in result.dataframe.columns
    assert 'mi_mean' in result.dataframe.columns
    # embedding_dim is not in sweep_grid, so it won't be in columns if it wasn't swept
    # assert 'embedding_dim' in result.dataframe.columns
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
        processor_params_x={'window_size': 10},
        processor_type_y='continuous',
        processor_params_y={'window_size': 10},
        base_params=BASE_PARAMS,
        output_units='nats',
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.mi_estimate, float)

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

@patch('neural_mi.run.run_precision_analysis')
def test_run_precision_mode_returns_results_with_dataframe_and_estimate(mock_precision, gaussian_data):
    """
    Verifies that mode='precision' routes correctly and formats the Results object.
    """
    x_data, y_data = gaussian_data
    
    # Mock the return value of the precision engine
    mock_precision.return_value = {
        'dataframe': pd.DataFrame([{'tau': 0.0, 'test_mi': 2.0}, {'tau': 1.0, 'test_mi': 0.5}]),
        'details': {
            'baseline_mi': 2.0,
            'precision_tau': 1.0,
            'threshold_ratio': 0.9,
            'threshold_value': 1.8,
            'corruption_method': 'rounding',
            'corrupt_target': 'x'
        }
    }
    
    result = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='precision',
        base_params=BASE_PARAMS,
        output_units='nats',
        tau_grid=[0.5, 1.0, 2.0],
        corrupt_target='x',
        n_workers=1,
        device='cpu'
    )
    
    # Verify the routing successfully called the engine
    mock_precision.assert_called_once()
    
    # Verify the final Results object formatting
    assert isinstance(result, Results)
    assert result.mode == 'precision'
    assert isinstance(result.dataframe, pd.DataFrame)
    
    # The mi_estimate should be mapped to the precision_tau
    assert result.mi_estimate == 1.0
    
    # Ensure the details dictionary has all the metadata
    assert 'baseline_mi' in result.details
    assert 'raw_results' in result.details
