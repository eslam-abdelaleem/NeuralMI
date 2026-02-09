# tests/test_mixed_data.py
import pytest
import numpy as np
import neural_mi as nmi
from neural_mi.data.handler import create_dataset

def test_continuous_and_spike_alignment():
    """
    Tests alignment between a continuous stream and a spike stream.
    """
    # Continuous data: 1000 timepoints
    # Assume 100Hz -> 10s duration.
    time_cont = np.arange(1000) / 100.0
    x_cont = np.random.randn(1000, 2)

    # Spike data: 10s duration
    y_spike, _ = nmi.generators.generate_correlated_spike_trains(duration=10.0)

    dataset = create_dataset(
        x_data=x_cont, y_data=y_spike,
        x_time=time_cont,
        processor_type_x='continuous', processor_params_x={'window_size': 0.1},
        processor_type_y='spike', processor_params_y={'window_size': 0.1}
    )

    x_proc = dataset.x_data
    y_proc = dataset.y_data

    # The key test: the number of samples must be identical after alignment.
    assert x_proc.shape[0] == y_proc.shape[0]
    
    # Expected number of windows: 10s / 0.1s = 100 windows
    assert x_proc.shape[0] >= 90
    assert x_proc.shape[0] <= 110


def test_different_continuous_params_alignment():
    """
    Tests alignment between two continuous streams.
    """
    x_data = np.random.randn(100, 1) # 0..99
    y_data = np.random.randn(120, 1) # 0..119

    # create_dataset enforces shared window for PairedTemporalDataset.
    # But if we pass separate params, it will use params_x.

    dataset = create_dataset(
        x_data=x_data, y_data=y_data,
        processor_type_x='continuous', processor_params_x={'window_size': 10},
        processor_type_y='continuous', processor_params_y={'window_size': 10}
    )

    x_proc = dataset.x_data
    y_proc = dataset.y_data
    
    assert x_proc.shape[0] == y_proc.shape[0]
    # Expected number of windows: 100 / 10 = 10 windows
    assert x_proc.shape[0] >= 9