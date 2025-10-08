# tests/test_visualize.py
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from neural_mi.visualize.plot import plot_sweep_curve, plot_bias_correction_fit
from neural_mi.results import Results
from neural_mi.visualize.plot import set_publication_style

@pytest.fixture
def test_set_publication_style_runs_without_error():
    """Tests that set_publication_style can be called without raising an error."""
    try:
        set_publication_style()
    except Exception as e:
        pytest.fail(f"set_publication_style raised an exception: {e}")

@pytest.fixture
def sweep_results_df():
    """Provides a sample DataFrame from a parameter sweep."""
    data = {
        'embedding_dim': [4, 8, 16, 32],
        'mi_mean': [0.5, 0.9, 1.2, 1.25],
        'mi_std': [0.1, 0.12, 0.11, 0.15]
    }
    return pd.DataFrame(data)

@pytest.fixture
def rigorous_results_df():
    """Provides a sample DataFrame from a rigorous analysis."""
    gammas = np.repeat(np.arange(1, 6), 5)
    test_mi = 1.0 / gammas + 0.5 + np.random.randn(len(gammas)) * 0.1
    data = {'gamma': gammas, 'test_mi': test_mi}
    return pd.DataFrame(data)

@patch('matplotlib.pyplot.show')
def test_plot_sweep_curve_runs_without_error(mock_show, sweep_results_df):
    """Tests that plot_sweep_curve can be called without raising an error."""
    fig, ax = plt.subplots(1, 1)
    try:
        plot_sweep_curve(sweep_results_df, 'embedding_dim', ax=ax)
    except Exception as e:
        pytest.fail(f"plot_sweep_curve raised an exception: {e}")

@patch('matplotlib.pyplot.show')
def test_plot_bias_correction_fit_runs_without_error(mock_show, rigorous_results_df):
    """Tests that plot_bias_correction_fit can be called without raising an error."""
    fig, ax = plt.subplots(1, 1)
    corrected_result = {
        'slope': -0.5,
        'mi_corrected': 0.55,
        'mi_error': 0.05,
        'gammas_used': list(range(1, 6))
    }
    try:
        plot_bias_correction_fit(rigorous_results_df, corrected_result, ax=ax)
    except Exception as e:
        pytest.fail(f"plot_bias_correction_fit raised an exception: {e}")

@patch('neural_mi.visualize.plot.plot_sweep_curve')
def test_results_plot_dispatcher_for_sweep(mock_plot_sweep, sweep_results_df):
    """
    Tests that the Results.plot() method correctly calls the sweep plot function
    when mode is 'sweep'.
    """
    results = Results(
        mode='sweep',
        dataframe=sweep_results_df,
        params={'sweep_var': 'embedding_dim'}
    )
    results.plot(show=False)
    mock_plot_sweep.assert_called_once()

@patch('neural_mi.visualize.plot.plot_bias_correction_fit')
def test_results_plot_dispatcher_for_rigorous(mock_plot_bias, rigorous_results_df):
    """
    Tests that the Results.plot() method correctly calls the bias correction plot
    function when mode is 'rigorous'.
    """
    details = {'mi_corrected': 0.55, 'mi_error': 0.05}
    results = Results(
        mode='rigorous',
        dataframe=rigorous_results_df,
        details=details
    )
    results.plot(show=False)
    mock_plot_bias.assert_called_once()