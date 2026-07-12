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

    def test_prepare_tasks_concat_critic_warning(self):
        # A5: sweeping embedding_dim with concat critic must now raise ValueError
        # (previously emitted a log warning; changed to hard error to fail loudly).
        x = np.random.randn(10, 5)
        y = np.random.randn(10, 5)
        base_params = {'critic_type': 'concat'}
        sweep = ParameterSweep(x, y, base_params)

        with pytest.raises(ValueError, match="embedding_dim"):
            sweep._prepare_tasks(sweep_grid={'embedding_dim': [4]}, is_proc_sweep=False, max_samples_per_task=None)

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

    # ------------------------------------------------------------------
    # dataset_device tests
    # ------------------------------------------------------------------

    def test_dataset_device_cpu_default_in_tasks(self):
        """Tasks built without dataset_device default to 'cpu' in params."""
        x = np.random.randn(20, 4)
        y = np.random.randn(20, 4)
        sweep = ParameterSweep(x, y, {})
        tasks = sweep._prepare_tasks(sweep_grid={}, is_proc_sweep=False)
        _, _, params, _ = tasks[0]
        # dataset_device should be absent (resolved inside task.py) or 'cpu'
        assert params.get('dataset_device', 'cpu') in ('cpu', None)

    def test_dataset_cache_reuse_across_tasks(self):
        """Sequential tasks with identical data/params should reuse the cached dataset."""
        import neural_mi as nmi
        from neural_mi import Model, Training
        from neural_mi.analysis.task import _DATASET_CACHE
        _DATASET_CACHE.clear()

        x = np.random.randn(60, 4, 1)
        y = np.random.randn(60, 4, 1)
        # Run a small sweep — all tasks share the same data so the cache should be hit
        nmi.run(x, y, mode='sweep',
                model=Model(embedding_dim=4, hidden_dim=16, n_layers=1),
                training=Training(n_epochs=1, batch_size=32, patience=1),
                sweep_grid={'embedding_dim': [4, 8]}, n_workers=1)
        # Cache should have at least one entry for this static dataset
        assert len(_DATASET_CACHE) >= 1

    def test_memory_warning_for_on_device_sweep(self):
        """Large sweeps with dataset_device != 'cpu' should emit a UserWarning."""
        import warnings
        x = np.random.randn(50, 4, 1)
        y = np.random.randn(50, 4, 1)
        sweep = ParameterSweep(x, y, {'dataset_device': 'mps',
                                      'n_epochs': 1, 'batch_size': 16, 'patience': 1,
                                      'embedding_dim': 4, 'hidden_dim': 8, 'n_layers': 1})
        tasks = sweep._prepare_tasks(
            sweep_grid={'embedding_dim': list(range(4, 30))},  # >20 tasks
            is_proc_sweep=False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Trigger the warning path by calling _run_parallel directly — use an
            # empty task list so nothing actually trains; we're only testing the warning.
            # Monkeypatch tasks to have >20 entries but do no work.
            try:
                sweep._run_parallel(tasks[:21], n_workers=1)
            except Exception:
                pass  # training may fail on non-existent MPS; we only care about the warning
        warning_msgs = [str(wi.message) for wi in w if issubclass(wi.category, UserWarning)]
        assert any('dataset_device' in m for m in warning_msgs)
