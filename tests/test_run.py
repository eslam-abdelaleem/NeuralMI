import pytest
import torch
import numpy as np
import pandas as pd
import neural_mi as nmi
from neural_mi import (Model, Training, Output, Processing,
                       Precision, Rigorous, Dimensionality)
from neural_mi.results import Results
from unittest.mock import patch

# Base model/training config used across multiple tests (was the BASE_PARAMS dict).
MODEL = Model(embedding_dim=8, hidden_dim=32, n_layers=1)
TRAINING = Training(n_epochs=1, learning_rate=1e-4, batch_size=64, patience=1)
NATS = Output(units='nats')

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
        x_data, y_data,
        mode='estimate',
        model=MODEL, training=TRAINING,
        output=NATS,
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
        x_data, y_data,
        mode='sweep',
        model=MODEL, training=TRAINING,
        sweep_grid=sweep_grid,
        output=NATS,
        n_workers=1
    )
    assert isinstance(result, Results)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert 'embedding_dim' in result.dataframe.columns
    assert len(result.dataframe) == 2


def test_run_sweep_mode_tolerates_unhashable_swept_values(gaussian_data):
    """sweep_grid values that are themselves lists (e.g. per-layer hidden_dim)
    used to crash pandas groupby with 'unhashable type: list'."""
    x_data, y_data = gaussian_data
    sweep_grid = {'hidden_dim': [[8, 8], [16]]}
    result = nmi.run(
        x_data, y_data,
        mode='sweep',
        model=MODEL, training=TRAINING,
        sweep_grid=sweep_grid,
        output=NATS,
        n_workers=1
    )
    assert isinstance(result.dataframe, pd.DataFrame)
    assert len(result.dataframe) == 2
    assert set(result.dataframe['hidden_dim']) == {(8, 8), (16,)}
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
            'raw_results_df': pd.DataFrame([{'gamma': 1.0, 'train_mi': 2.0}]),
            'corrected_results': [{'mi_corrected': 2.5, 'mi_error': 0.1, 'slope': -0.05}]
        }

        result = nmi.run(
            x_data, y_data,
            mode='rigorous',
            model=MODEL, training=TRAINING,
            output=NATS,
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
        x_data,
        mode='dimensionality',
        model=MODEL, training=TRAINING,
        output=NATS,
        dimensionality=Dimensionality(split_method='random', n_splits=2),
        n_workers=1,
        device='cpu'
    )

    assert isinstance(result, Results)
    assert isinstance(result.dataframe, pd.DataFrame)
    # Check for our new spectral metrics instead of mi_mean
    assert 'pr_eig_mean' in result.dataframe.columns
    assert 'pr_singular_mean' in result.dataframe.columns
    assert 'mi_mean' in result.dataframe.columns
    assert result.mi_estimate is None

def test_run_with_continuous_processor_returns_results(raw_gaussian_data):
    """
    Tests that the raw data pipeline returns a correct Results object.
    """
    x_raw, y_raw = raw_gaussian_data
    result = nmi.run(
        x_raw, y_raw,
        mode='estimate',
        processing=Processing(x='continuous', x_params={'window_size': 10},
                              y='continuous', y_params={'window_size': 10}),
        model=MODEL, training=TRAINING,
        output=NATS,
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
        x_data, y_data,
        mode='estimate',
        model=Model(embedding_dim=8, hidden_dim=32, n_layers=1,
                    custom_critic=custom_critic_instance),
        training=TRAINING,
        n_workers=1,
        output=NATS  # Use nats for direct comparison with np.log
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
        'dataframe': pd.DataFrame([{'tau': 0.0, 'train_mi': 2.0}, {'tau': 1.0, 'train_mi': 0.5}]),
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
        x_data, y_data,
        mode='precision',
        model=MODEL, training=TRAINING,
        output=NATS,
        precision=Precision(tau_grid=[0.5, 1.0, 2.0], corrupt_target='x'),
        n_workers=1,
        device='cpu'
    )

    # Verify the routing successfully called the engine
    mock_precision.assert_called_once()

    # Verify the final Results object formatting
    assert isinstance(result, Results)
    assert result.mode == 'precision'
    assert isinstance(result.dataframe, pd.DataFrame)

    # mi_estimate holds baseline_mi (MI at zero corruption), not precision_tau.
    # precision_tau remains accessible via result.details['precision_tau'].
    assert result.mi_estimate == result.details['baseline_mi']
    assert result.details['precision_tau'] == 1.0  # tau is still in details

    # Ensure the details dictionary has all the metadata
    assert 'baseline_mi' in result.details
    assert 'raw_results' in result.details


# --- Processor-level sweep and spike integration ---

MODEL_INTEGRATION = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
TRAINING_INTEGRATION = Training(n_epochs=2, learning_rate=1e-4, batch_size=32, patience=1)


def test_run_sweep_mode_processor_param(raw_gaussian_data):
    """Tests a sweep over a processor parameter (window_size), distinct from model param sweeps."""
    x, y = raw_gaussian_data
    results = nmi.run(
        x, y, mode='sweep',
        processing=Processing(x='continuous', x_params={}),
        model=MODEL_INTEGRATION, training=TRAINING_INTEGRATION,
        sweep_grid={'window_size': [5, 10]},
        seed=42, n_workers=1
    )
    assert isinstance(results.dataframe, pd.DataFrame)
    assert len(results.dataframe) == 2


def test_rigorous_mode_with_spike_data():
    """Full end-to-end pipeline: rigorous analysis on synthetic spike data."""
    x_spikes, y_spikes = nmi.generators.generate_correlated_spike_trains(
        n_neurons=5, duration=10.0, firing_rate=10.0, delay=0.01, jitter=0.002
    )
    results = nmi.run(
        x_spikes, y_spikes, mode='rigorous',
        processing=Processing(x='spike', x_params={'window_size': 0.05}),
        model=Model(embedding_dim=8, hidden_dim=16, n_layers=1),
        training=Training(n_epochs=2, learning_rate=1e-4, batch_size=32, patience=1),
        rigorous=Rigorous(gamma_range=range(1, 3)),
        n_workers=1, seed=42
    )
    assert isinstance(results, nmi.results.Results)
    assert isinstance(results.mi_estimate, float)
    assert results.dataframe is not None and not results.dataframe.empty
    assert 'mi_corrected' in results.details
    assert 'mi_error' in results.details
