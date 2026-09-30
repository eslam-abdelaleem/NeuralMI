# tests/test_analysis.py
import pytest
import numpy as np
import pandas as pd
import neural_mi as nmi
from neural_mi import Model, Training, Processing, Lag, Split
import torch
from unittest.mock import patch

# shift_time disabled to avoid dynamic window sizing issues in tests.
MODEL_TEST = Model(embedding_dim=4, hidden_dim=16, n_layers=1)
TRAINING_TEST = Training(n_epochs=2, learning_rate=1e-4, batch_size=32,
                         patience=1, shift_time=False)

@pytest.mark.parametrize("processor_type", ["continuous", "categorical", "spike"])
def test_run_lag_mode(processor_type):
    """
    Tests that mode='lag' runs for all processor types and returns a valid
    DataFrame with the correct columns.
    """
    if processor_type == "continuous":
        x_data, y_data, _ = nmi.generators.generate_lagged_pair(n_samples=500, lag=5)
        lag_range = range(-10, 11, 5)
        processor_params = {'window_size': 10}
    elif processor_type == "categorical":
        x_data, y_data, _ = nmi.generators.generate_categorical_pair(
            n_samples=500, n_categories=3, use_torch=False, seed=0)
        lag_range = range(-10, 11, 5)
        processor_params = {'window_size': 10}
    else: # spike
        x_data, y_data, _ = nmi.generators.generate_spike_pair(
            n_neurons=10, n_windows=100, window_size=0.1, seed=0)
        lag_range = np.arange(-0.05, 0.06, 0.01)
        processor_params = {'window_size': 0.1, 'max_spikes_per_window': 10} # Added max_spikes for robustness

    results = nmi.run(
        x_data, y_data,
        mode='lag',
        processing=Processing(x=processor_type, x_params=processor_params,
                              y=processor_type, y_params=processor_params),
        sweep_grid={'run_id': range(2)},
        model=MODEL_TEST, training=TRAINING_TEST,
        lag=Lag(lag_range=lag_range),
        n_workers=1,
        seed=42
    )

    assert isinstance(results, nmi.results.Results)
    assert isinstance(results.dataframe, pd.DataFrame)
    assert 'lag' in results.dataframe.columns
    assert 'mi_mean' in results.dataframe.columns
    assert len(results.dataframe) == len(lag_range)


class TestLagRecoversKnownLag:
    """mode='lag' must find the lag it was given, both with and without an
    explicit processing= argument.

    With raw arrays and no processing=, the data must still be shifted by
    each lag: a flat profile would mean every lag saw the unshifted data. The
    recovered peak is checked in both cases.
    """

    TRUE_LAG = 20

    def _profile(self, **run_kwargs):
        x, y, exact = nmi.generators.generate_lagged_pair(
            n_samples=4000, lag=self.TRUE_LAG, dim=1, seed=0)
        results = nmi.run(
            np.asarray(x, dtype='float32'), np.asarray(y, dtype='float32'),
            mode='lag', lag=Lag(lag_range=range(0, 41, 10)),
            model=Model(embedding_dim=16, hidden_dim=64),
            training=Training(n_epochs=60, batch_size=128, patience=20,
                              learning_rate=1e-3),
            split=Split(mode='blocked'), n_workers=1, show_progress=False,
            seed=0, **run_kwargs)
        df = results.dataframe
        return {int(a): float(b) for a, b in zip(df['lag'], df['mi_mean'])}, exact

    @pytest.mark.slow
    def test_recovers_known_lag_without_processing(self):
        """Raw arrays, no processing=: each lag still shifts the data."""
        prof, _ = self._profile()
        peak = max(prof, key=prof.get)
        assert peak == self.TRUE_LAG, f"peak at {peak}, expected {self.TRUE_LAG}: {prof}"
        off_peak = [v for k, v in prof.items() if k != self.TRUE_LAG]
        assert prof[peak] > 2 * max(off_peak), (
            f"profile is nearly flat, which is what the unshifted-data bug looked "
            f"like: {prof}")

    @pytest.mark.slow
    def test_recovers_known_lag_with_processing(self):
        prof, _ = self._profile(
            processing=Processing(x='continuous', y='continuous',
                                  x_params={'window_size': 1, 'step_size': 1},
                                  y_params={'window_size': 1, 'step_size': 1}))
        peak = max(prof, key=prof.get)
        assert peak == self.TRUE_LAG, f"peak at {peak}, expected {self.TRUE_LAG}: {prof}"


class TestLagShiftWindows:
    """shift_windows already engages mechanically for mode='lag' with
    regular-grid data (try_build_shift_windows_dataset is mode-agnostic) --
    the only thing that was wrong is _warn_if_shift_windows_dead's
    reachable-modes list not including 'lag', producing a false "has no
    effect" warning when a user explicitly requests it."""

    def test_no_false_dead_warning_when_explicitly_requested(self):
        import warnings
        np.random.seed(0)
        x = np.random.randn(2000, 2).astype('float32')
        y = np.random.randn(2000, 2).astype('float32')
        proc = Processing(x='continuous', x_params={'window_size': 10, 'step_size': 10},
                          y='continuous', y_params={'window_size': 10, 'step_size': 10})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(x, y, mode='lag', lag=Lag(lag_range=range(-2, 3)), processing=proc,
                   model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                   training=Training(n_epochs=1, patience=1, shift_windows=True),
                   n_workers=1, show_progress=False, seed=0)
        msgs = [str(w.message) for w in caught if 'shift_windows' in str(w.message)]
        assert not msgs, f"Did not expect a shift_windows warning; got: {msgs}"

    def test_n_windows_reflects_true_window_count_not_raw_samples(self):
        """n_windows_built counts windows, not raw samples: with
        window_size > 1 it must be strictly smaller than the sample count."""
        np.random.seed(0)
        T, window_size = 2000, 10
        x = np.random.randn(T, 2).astype('float32')
        y = np.random.randn(T, 2).astype('float32')
        proc = Processing(x='continuous', x_params={'window_size': window_size, 'step_size': window_size},
                          y='continuous', y_params={'window_size': window_size, 'step_size': window_size})
        results = nmi.run(x, y, mode='lag', lag=Lag(lag_range=range(0, 1)), processing=proc,
                          model=Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                          training=Training(n_epochs=1, patience=1, shift_windows=True),
                          n_workers=1, show_progress=False, seed=0)
        # Canonical column name, shared with every other mode (task.py's).
        n_windows = results.dataframe['n_windows_built'].iloc[0]
        raw_sample_count = T  # lag=0 -> no truncation
        assert n_windows < raw_sample_count, (
            f"n_windows={n_windows} should be well below the raw sample count "
            f"({raw_sample_count}) once window_size={window_size} windowing is accounted for"
        )


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
        'use_variational': False,
        'embedding_model': 'mlp',
        'max_n_batches': 512,
        'kernel_size': 3,
        'bidirectional': False,
        'nhead': 4,
        # Our newly wired parameters:
        'max_eval_samples': 42,
        'track_spectral_history': True,
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
        assert call_kwargs['track_spectral_history'] is True