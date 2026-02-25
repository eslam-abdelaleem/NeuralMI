# tests/test_sweep_extended.py
import pytest
import torch
import numpy as np
from neural_mi.analysis.sweep import ParameterSweep

class TestSweepExtended:
    def test_sweep_init_with_tensor(self):
        # Testing logic that infers dims from tensor
        x = torch.randn(10, 5, 20) # (batch, channels, features)
        y = torch.randn(10, 5, 20)
        base_params = {}
        sweep = ParameterSweep(x, y, base_params)

        assert sweep.base_params['input_dim_x'] == 5 * 20
        assert sweep.base_params['n_channels_x'] == 5
        assert sweep.base_params['input_dim_y'] == 5 * 20
        assert sweep.base_params['n_channels_y'] == 5

    def test_prepare_tasks_concat_critic_warning(self, caplog):
        # Warning when embedding_dim is in sweep_grid for concat critic
        x = np.random.randn(10, 5)
        y = np.random.randn(10, 5)
        base_params = {'critic_type': 'concat'}
        sweep = ParameterSweep(x, y, base_params)

        sweep._prepare_tasks(sweep_grid={'embedding_dim': [4]}, is_proc_sweep=False, max_samples_per_task=None)
        assert "is not applicable for ConcatCritic" in caplog.text

    def test_prepare_tasks_max_samples(self):
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)
        base_params = {}
        sweep = ParameterSweep(x, y, base_params)

        tasks = sweep._prepare_tasks(sweep_grid={}, is_proc_sweep=False, max_samples_per_task=10)
        task_x, task_y, _, _ = tasks[0]
        assert task_x.shape[0] == 10
        assert task_y.shape[0] == 10

    def test_prepare_tasks_proc_sweep(self):
        # Checks that raw data is passed if is_proc_sweep=True
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)
        base_params = {}
        sweep = ParameterSweep(x, y, base_params)

        tasks = sweep._prepare_tasks(sweep_grid={'window_size': [10]}, is_proc_sweep=True, max_samples_per_task=None)
        task_x, task_y, params, _ = tasks[0]

        # Verify params got updated
        assert params['processor_params_x']['window_size'] == 10
        assert params['processor_params_y']['window_size'] == 10
        # Verify raw data passed (full size)
        assert task_x.shape[0] == 100

    def test_prepare_tasks_save_model_path(self):
        x = np.random.randn(10, 5)
        y = np.random.randn(10, 5)
        base_params = {'save_best_model_path': 'model.pth'}
        sweep = ParameterSweep(x, y, base_params)

        tasks = sweep._prepare_tasks(sweep_grid={'dim': [4]}, is_proc_sweep=False, max_samples_per_task=None)
        _, _, params, _ = tasks[0]
        assert params['save_best_model_path'] == 'model_dim_4.pth'
