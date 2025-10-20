# tests/test_analysis.py
import pytest
import numpy as np
import pandas as pd
import neural_mi as nmi

BASE_PARAMS_TEST = {
    'n_epochs': 2, 'learning_rate': 1e-4, 'batch_size': 32,
    'patience': 1, 'embedding_dim': 4, 'hidden_dim': 16, 'n_layers': 1
}

@pytest.mark.parametrize("processor_type", ["continuous", "categorical", "spike"])
def test_run_lag_mode(processor_type):
    """
    Tests that mode='lag' runs for all processor types and returns a valid
    DataFrame with the correct columns.
    """
    if processor_type == "continuous":
        x_data, y_data = nmi.datasets.generate_temporally_convolved_data(n_samples=500, lag=5)
        lag_range = range(-10, 11, 5)
        processor_params = {'window_size': 10}
    elif processor_type == "categorical":
        x_data, y_data = nmi.datasets.generate_correlated_categorical_series(n_samples=500, n_categories=3)
        lag_range = range(-10, 11, 5)
        processor_params = {'window_size': 10}
    else: # spike
        x_data, y_data = nmi.datasets.generate_correlated_spike_trains(duration=10.0, delay=0.02)
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