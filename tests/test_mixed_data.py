# tests/test_mixed_data.py
import pytest
import numpy as np
import neural_mi as nmi
from neural_mi.data.handler import DataHandler

def test_continuous_and_spike_alignment():
    """
    Tests alignment between a continuous stream and a spike stream.
    """
    # Continuous data: 1000 timepoints, window=10, step=5 -> (1000-10)/5 + 1 = 199 samples
    x_cont = np.random.randn(2, 1000)
    # Spike data: 10s duration, window=0.1, step=0.05 -> theoretical max of 199 samples
    y_spike, _ = nmi.datasets.generate_correlated_spike_trains(duration=10.0)

    handler = DataHandler(
        x_data=x_cont, y_data=y_spike,
        processor_type_x='continuous', processor_params_x={'window_size': 10, 'step_size': 5},
        processor_type_y='spike', processor_params_y={'window_size': 0.1, 'step_size': 0.05}
    )
    x_proc, y_proc = handler.process()

    # The key test: the number of samples must be identical after alignment.
    assert x_proc.shape[0] == y_proc.shape[0]
    
    # The number of samples should be close to the theoretical max.
    # We allow a small tolerance because the last spike might not be at exactly 10.0s.
    assert abs(x_proc.shape[0] - 199) <= 3 


def test_different_continuous_params_alignment():
    """
    Tests alignment between two continuous streams with different parameters.
    """
    x_data = np.random.randn(1, 100) # (100-10)/1 + 1 = 91 samples
    y_data = np.random.randn(1, 120) # (120-20)/2 + 1 = 51 samples (should be limiting)

    handler = DataHandler(
        x_data=x_data, y_data=y_data,
        processor_type_x='continuous', processor_params_x={'window_size': 10, 'step_size': 1},
        processor_type_y='continuous', processor_params_y={'window_size': 20, 'step_size': 2}
    )
    x_proc, y_proc = handler.process()
    
    assert x_proc.shape[0] == y_proc.shape[0]
    assert x_proc.shape[0] == 51