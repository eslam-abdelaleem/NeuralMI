import warnings

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
    # One configuration, one repeat: one dataframe row, whose mean is the estimate.
    assert len(result.dataframe) == 1
    assert result.mi_estimate == result.dataframe['mi_mean'].iloc[0]
    assert len(result.runs) == 1

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
    """sweep_grid values that are themselves lists (per-layer hidden_dim)
        group correctly and come back as tuples.
    """
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
    with patch('neural_mi.analysis.rigorous.run_rigorous_analysis') as mock_rigorous:
        mock_rigorous.return_value = {
            'raw_results_df': pd.DataFrame([{'gamma': 1, 'chunk': 0, 'train_mi': 2.0}]),
            'corrected_results': [{'mi_corrected': 2.5, 'mi_error': 0.1, 'slope': -0.05,
                                   'is_reliable': True}]
        }

        result = nmi.run(
            x_data, y_data,
            mode='rigorous',
            model=MODEL, training=TRAINING,
            output=NATS,
            n_workers=1
        )

    assert isinstance(result, Results)
    assert result.mi_estimate == 2.5
    assert isinstance(result.dataframe, pd.DataFrame)
    # One repeat: the fit's own half-width is reported with the estimate.
    assert result.get('mi_error') == 0.1
    assert result.runs['slope'].iloc[0] == -0.05
    assert 'trainings' in result.details[0]

def test_run_dimensionality_mode_returns_results_with_dataframe(raw_gaussian_data):
    """mode='dimensionality' on X alone returns the curve over embedding dimensions,
    one dataframe row per split and embedding dimension, and the reading."""
    x_data, _ = raw_gaussian_data

    result = nmi.run(
        x_data,
        mode='dimensionality',
        model=MODEL, training=TRAINING,
        output=NATS,
        dimensionality=Dimensionality(split_method='random', n_splits=2, n_restarts=1,
                                      reference_dim=4, embedding_dims=[1, 2]),
        n_workers=1,
        device='cpu'
    )

    assert isinstance(result, Results)
    assert isinstance(result.dataframe, pd.DataFrame)
    df = result.dataframe
    assert sorted(set(df['split_id'])) == [0, 1]
    assert sorted(set(df['embedding_dim'])) == [1, 2, 4]
    assert {'mi_best', 'mi_curve', 'pr_eig_mean', 'pr_singular_mean'} <= set(df.columns)
    # Many rows, so no single estimate.
    assert result.mi_estimate is None
    assert len(result.runs) == 6
    details = result.details[0]
    assert set(details['dimension_at_most_per_split']) == {0, 1}
    assert details['reference_dim'] == 4
    # The halves of these X channels are independent, so there may be no reading.
    assert 'dimension_at_most' in details and 'plateau' in details

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
        model=Model(custom_critic=custom_critic_instance),
        training=TRAINING,
        n_workers=1,
        output=NATS  # Use nats for direct comparison with np.log
    )
    assert isinstance(result, nmi.results.Results)
    assert isinstance(result.mi_estimate, float)

    # For a score matrix of all 1s, the InfoNCE bound is log(N) + E[diag - logsumexp]
    # which calculates to log(N) + 1 - (log(exp(1)*N)) = log(N) + 1 - (1 + log(N)) = 0.
    assert np.isclose(result.mi_estimate, 0.0, atol=1e-6)

@patch('neural_mi.analysis.precision.run_precision_analysis')
def test_run_precision_mode_returns_results_with_dataframe_and_estimate(mock_precision, gaussian_data):
    """
    Verifies that mode='precision' routes correctly and formats the Results object.
    """
    x_data, y_data = gaussian_data

    # Mock the return value of the precision engine
    mock_precision.return_value = {
        'samples': [{'tau': 0.0, 'mi': 2.0}, {'tau': 1.0, 'mi': 0.5}],
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

    # One row per tau, so there is no single estimate; the baseline and the
    # threshold tau are read from details.
    assert result.mi_estimate is None
    assert list(result.dataframe['tau']) == [0.0, 1.0]
    assert result.get('baseline_mi') == 2.0
    assert result.get('precision_tau') == 1.0

    # The configuration's details hold the precision read; each evaluation is a row of runs.
    assert {'baseline_mi', 'precision_tau', 'corruption_method'} <= set(result.details[0])
    assert len(result.runs) == 2


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
    x_spikes, y_spikes, _ = nmi.generators.generate_spike_pair(
            n_neurons=5, n_windows=200, window_size=0.05, seed=0)
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
    assert results.get('mi_error') is not None
    assert 'trainings' in results.details[0]


# --- Where warnings point ---

def _run_from_a_helper(x, y):
    return nmi.run(x, y, mode='estimate', model=MODEL,
                   training=Training(n_epochs=1, batch_size=512), show_progress=False)


def test_warnings_point_at_the_callers_line(raw_gaussian_data):
    """A warning names the line in the caller's code, however deep it was raised."""
    import inspect
    import warnings
    x, y = raw_gaussian_data
    x, y = x[:100], y[:100]
    call_line = inspect.getsourcelines(_run_from_a_helper)[1] + 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _run_from_a_helper(x, y)
    ours = [w for w in caught if 'reach the estimator' in str(w.message)
            or 'exceeds the' in str(w.message)]
    # One raised in run() itself, one raised inside the trainer.
    assert len(ours) == 2
    for w in ours:
        assert (w.filename, w.lineno) == (__file__, call_line), str(w.message)


def test_a_warning_in_a_worker_names_the_task_entry_point():
    """A worker has no caller's code above the library, so its task's entry point is named."""
    import importlib
    import multiprocessing
    import os
    nmi_logger = importlib.import_module('neural_mi.logger')
    package_file = os.path.join(os.path.dirname(nmi_logger.__file__), 'worker_entry.py')
    runner_file = os.path.join(os.path.dirname(multiprocessing.__file__), 'pool.py')
    entry, runner = {}, {}
    exec(compile("def entry():\n    return user_stacklevel()\n", package_file, 'exec'),
         {'user_stacklevel': nmi_logger.user_stacklevel}, entry)
    exec(compile("def run(fn):\n    return fn()\n", runner_file, 'exec'), {}, runner)
    # Called from a worker's task runner, the warning stays on the library's entry point.
    assert runner['run'](entry['entry']) == 1
    # Called from the caller's own code, it moves out to that code.
    assert entry['entry']() == 2


def test_every_library_warning_uses_user_stacklevel():
    """A hardcoded stacklevel breaks whenever the call depth changes."""
    import ast
    import pathlib
    package = pathlib.Path(nmi.__file__).parent
    hardcoded = []
    for path in package.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'warn' and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ('warnings', '_warnings')):
                level = {k.arg: ast.unparse(k.value) for k in node.keywords}.get('stacklevel')
                if level != 'user_stacklevel()':
                    hardcoded.append(f"{path.relative_to(package)}:{node.lineno} stacklevel={level}")
    assert not hardcoded, hardcoded


@pytest.mark.parametrize('route', ['arrays', 'windows'])
def test_one_small_data_warning_for_both_routes(route):
    """Arrays passed in and windows the library builds get the same warning,
    below the same count, naming the scaling and the check."""
    import warnings
    rng = np.random.default_rng(0)
    n = 150 if route == 'arrays' else 150 * 10
    x = rng.standard_normal((n, 2)).astype(np.float32)
    y = (x + rng.standard_normal((n, 2))).astype(np.float32)
    processing = (None if route == 'arrays' else
                  Processing(x='continuous', x_params={'window_size': 10},
                             y='continuous', y_params={'window_size': 10}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, y, processing=processing, model=MODEL, training=TRAINING, show_progress=False)
    small = [str(w.message) for w in caught if 'reach the estimator' in str(w.message)]
    unit = 'samples' if route == 'arrays' else 'windows'
    # Shifted windows keep a one-window margin, so 1,500 samples give 149.
    assert len(small) == 1 and small[0].startswith(f"Only {150 if route == 'arrays' else 149} {unit} reach")
    assert "N ~ d^2/I" in small[0] and "mode='rigorous'" in small[0]


def test_no_small_data_warning_from_two_hundred_up():
    import warnings
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2)).astype(np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, x + rng.standard_normal((300, 2)).astype(np.float32),
                model=MODEL, training=TRAINING, show_progress=False)
    assert not any('reach the estimator' in str(w.message) or 'Small dataset' in str(w.message)
                   for w in caught)


def _small_data_messages(*args, **kwargs):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(*args, model=MODEL, training=TRAINING, show_progress=False, **kwargs)
    return [str(w.message) for w in caught if 'reach the estimator' in str(w.message)]


def test_small_data_warning_on_the_spike_route():
    """Spike windows are built inside each task. The count comes from the time span."""
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.uniform(0, 10.0, 80)) for _ in range(3)]      # 10 s of spikes
    messages = _small_data_messages(
        spikes, spikes, processing=Processing(x='spike', x_params={'window_size': 0.1},
                                              y='spike', y_params={'window_size': 0.1}))
    assert len(messages) == 1 and messages[0].startswith("Only 9")   # about 99 windows


@pytest.mark.parametrize('lags, warns', [(range(-5, 6), False), (range(-1200, 1201, 600), True)])
def test_small_data_warning_on_the_lag_route_counts_the_largest_lag(lags, warns):
    from neural_mi import Lag
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 2)).astype(np.float32)
    y = (x + rng.standard_normal((3000, 2))).astype(np.float32)
    messages = _small_data_messages(
        x, y, mode='lag', lag=Lag(lag_range=lags),
        processing=Processing(x='continuous', x_params={'window_size': 10},
                              y='continuous', y_params={'window_size': 10}))
    assert bool(messages) == warns


@pytest.mark.parametrize('critic, suggested', [('separable', True), ('hybrid', True)])
def test_small_data_warning_suggests_layer_norm_only_where_it_is_off(critic, suggested):
    """norm_layer='auto' is no normalisation outside mode='dimensionality'."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 3)).astype(np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, x + rng.standard_normal((100, 3)).astype(np.float32),
                model=Model(embedding_dim=8, hidden_dim=32, n_layers=1, critic_type=critic),
                training=TRAINING, show_progress=False)
    messages = [str(w.message) for w in caught if 'reach the estimator' in str(w.message)]
    assert len(messages) == 1
    assert ("norm_layer='layer'" in messages[0]) == suggested
