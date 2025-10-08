# tests/test_end_to_end.py
import pytest
import neural_mi as nmi

def test_rigorous_mode_with_spike_data():
    """
    Tests the full end-to-end pipeline for rigorous analysis on spike data.
    """
    # 1. Generate synthetic spike data
    x_spikes, y_spikes = nmi.datasets.generate_correlated_spike_trains(
        n_neurons=5,
        duration=10.0,
        firing_rate=10.0,
        delay=0.01,
        jitter=0.002
    )

    # 2. Define a minimal set of parameters for a quick run
    base_params = {
        'n_epochs': 2,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'patience': 1,
        'embedding_dim': 8,
        'hidden_dim': 16,
        'n_layers': 1
    }

    # 3. Run the analysis in rigorous mode
    results = nmi.run(
        x_data=x_spikes,
        y_data=y_spikes,
        mode='rigorous',
        processor_type='spike',
        processor_params={'window_size': 0.05, 'step_size': 0.01},
        base_params=base_params,
        gamma_range=range(1, 3),  # Use a small gamma range for a quick test
        n_workers=1,  # Use a single worker for reproducibility
        random_seed=42
    )

    # 4. Assert that the results are in the expected format
    assert isinstance(results, nmi.results.Results)
    assert isinstance(results.mi_estimate, float)
    assert results.dataframe is not None
    assert not results.dataframe.empty
    assert 'mi_corrected' in results.details
    assert 'mi_error' in results.details