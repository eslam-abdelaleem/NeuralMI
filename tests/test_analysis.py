# tests/test_analysis.py
import pytest
import numpy as np
import pandas as pd
import neural_mi as nmi
import torch
from unittest.mock import patch
from neural_mi.analysis.dimensionality import run_dimensionality_analysis

BASE_PARAMS_TEST = {
    'n_epochs': 2, 'learning_rate': 1e-4, 'batch_size': 32,
    'patience': 1, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1,
    'random_time_shifting': False # Disable time shifting to avoid dynamic window sizing issues in tests
}

@pytest.mark.parametrize("processor_type", ["continuous", "categorical", "spike"])
def test_run_lag_mode(processor_type):
    """
    Tests that mode='lag' runs for all processor types and returns a valid
    DataFrame with the correct columns.
    """
    if processor_type == "continuous":
        x_data, y_data = nmi.generators.generate_temporally_convolved_data(n_samples=500, lag=5)
        lag_range = range(-10, 11, 5)
        processor_params = {'window_size': 10}
    elif processor_type == "categorical":
        x_data, y_data = nmi.generators.generate_correlated_categorical_series(n_samples=500, n_categories=3)
        lag_range = range(-10, 11, 5)
        processor_params = {'window_size': 10}
    else: # spike
        x_data, y_data = nmi.generators.generate_correlated_spike_trains(duration=10.0, delay=0.02)
        lag_range = np.arange(-0.05, 0.06, 0.01)
        processor_params = {'window_size': 0.1, 'max_spikes_per_window': 10} # Added max_spikes for robustness

    results = nmi.run(
        x_data=x_data,
        y_data=y_data,
        mode='lag',
        processor_type_x=processor_type,
        processor_params_x=processor_params,
        processor_type_y=processor_type,  # Added this line
        processor_params_y=processor_params,  # Added this line
        sweep_grid={'run_id': range(2)},
        base_params=BASE_PARAMS_TEST,
        lag_range=lag_range,
        n_workers=1,
        random_seed=42
    )

    assert isinstance(results, nmi.results.Results)
    assert isinstance(results.dataframe, pd.DataFrame)
    assert 'lag' in results.dataframe.columns
    assert 'mi_mean' in results.dataframe.columns
    assert len(results.dataframe) == len(lag_range)


@pytest.fixture
def mock_sweep():
    """Fixture to mock the ParameterSweep engine so we only test the orchestrator."""
    with patch('neural_mi.analysis.dimensionality.ParameterSweep') as MockSweep:
        # Setup the mock to return a dummy dataframe row
        instance = MockSweep.return_value
        instance.run.return_value = [{'test_mi': 1.0}]
        yield MockSweep

def test_dimensionality_forces_hybrid_and_metrics(mock_sweep):
    """Proves the orchestrator overrides user params to guarantee accurate dim estimation."""
    x_data = torch.randn(100, 4)
    # The user asks for a simple separable critic, but the orchestrator MUST override this
    base_params = {'critic_type': 'separable'} 
    
    df = run_dimensionality_analysis(x_data, base_params, split_method='spatial')
    
    # Extract the parameters that were actually passed to the Sweep Engine
    call_args = mock_sweep.call_args[1]
    analysis_params = call_args['base_params']
    
    # Assertions for overriding behavior
    assert analysis_params['critic_type'] == 'hybrid', "Failed to force Hybrid critic."
    assert analysis_params['track_spectral_metrics'] is True, "Failed to enable spectral metrics."
    assert analysis_params['embedding_dim'] == 64, "Failed to inject large default bottleneck."
    
    assert isinstance(df, pd.DataFrame)

def test_dimensionality_interaction_no_split(mock_sweep):
    """Proves Interaction Dimensionality passes X and Y directly without splitting."""
    x_data = torch.randn(100, 2)
    y_data = torch.randn(100, 2)
    
    # User provides a specific bottleneck, which should NOT be overridden
    base_params = {'embedding_dim': 16} 
    
    run_dimensionality_analysis(x_data, base_params, y_data=y_data)
    
    call_args = mock_sweep.call_args[1]
    analysis_params = call_args['base_params']
    
    # Verify exact X and Y were passed, not splits
    assert call_args['x_data'] is x_data
    assert call_args['y_data'] is y_data
    assert analysis_params['embedding_dim'] == 16

def test_dimensionality_intrinsic_splits(mock_sweep):
    """Proves Intrinsic Dimensionality correctly slices data based on split_method."""
    x_data = torch.randn(100, 4) # 100 timepoints, 4 channels
    base_params = {}
    
    # 1. Test Spatial Split
    run_dimensionality_analysis(x_data, base_params, split_method='spatial')
    call_args = mock_sweep.call_args[1]
    assert call_args['x_data'].shape == (100, 2), "Spatial split failed on X."
    assert call_args['y_data'].shape == (100, 2), "Spatial split failed on Y."
    
    # 2. Test Temporal Split (lag=2)
    run_dimensionality_analysis(x_data, base_params, split_method='temporal', lag=2)
    call_args = mock_sweep.call_args[1]
    assert call_args['x_data'].shape == (98, 4), "Temporal split failed on X."
    assert call_args['y_data'].shape == (98, 4), "Temporal split failed on Y."

    # 3. Test Random Split loops
    run_dimensionality_analysis(x_data, base_params, split_method='random', n_splits=3)
    # 1 call from spatial, 1 from temporal, 3 from random = 5 total calls to Sweep engine
    assert mock_sweep.return_value.run.call_count == 5, "Random split loop failed."

# --- Task Routing Tests ---

def test_task_parameter_routing():
    """Proves that run_training_task correctly passes new parameters to the Trainer."""
    from neural_mi.analysis.task import run_training_task
    
    # Dummy data
    x_data = torch.randn(10, 2)
    y_data = torch.randn(10, 2)
    
    # The new parameters we want to test
    params = {
        'processor_type_x': 'continuous',
        'processor_type_y': 'continuous',
        'processor_params_x': {'window_size': 2},  
        'processor_params_y': {'window_size': 2},
        'critic_type': 'separable',
        'learning_rate': 0.01,
        'estimator_name': 'infonce',
        'n_epochs': 1,
        'batch_size': 5,
        'patience': 1,
        'hidden_dim': 8,
        'n_layers': 1,
        'embedding_dim': 4,
        # Our newly wired parameters:
        'max_eval_samples': 42,
        'track_spectral_metrics': True,
        'spectral_output': 'all'
    }
    
    # We patch Trainer.train to intercept the call and check the kwargs
    with patch('neural_mi.analysis.task.Trainer.train') as mock_train:
        # Mock train to return dummy results
        mock_train.return_value = {'train_mi': 1.0, 'test_mi': 1.0, 'test_mi_history': [1.0]}
        
        # Run the task
        run_training_task((x_data, y_data, params, 'test_run'))
        
        # Verify trainer.train was called exactly once
        assert mock_train.call_count == 1
        
        # Extract the kwargs passed to trainer.train
        call_kwargs = mock_train.call_args[1]
        
        # Assert the new parameters made it through the pipeline
        assert call_kwargs['max_eval_samples'] == 42
        assert call_kwargs['track_spectral_metrics'] is True
        assert call_kwargs['spectral_output'] == 'all'