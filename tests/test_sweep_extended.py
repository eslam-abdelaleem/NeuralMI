# tests/test_sweep_extended.py
import warnings
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
        # Sweeping embedding_dim with a concat critic must raise ValueError:
        # concat critics don't have a well-defined per-dimension score, so this
        # sweep configuration is a hard error rather than a silent warning.
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

    def test_dataset_cache_rejects_stale_data_ptr_collision(self):
        """A cache entry keyed by data_ptr()/id() alone would be served to any
        tensor whose allocation happens to land at the same address as a freed
        one. Simulate that collision by pre-seeding the cache with an entry
        whose weakref points to a *different* object than the incoming x_data;
        the cache must treat this as a miss and rebuild, not reuse the stale
        dataset."""
        from unittest.mock import MagicMock, patch
        from neural_mi.analysis.task import (
            _DATASET_CACHE, _DATASET_CACHE_LOCK, _dataset_cache_key, _safe_weakref,
        )
        _DATASET_CACHE.clear()

        x = np.random.randn(20, 4, 1).astype(np.float32)
        y = np.random.randn(20, 4, 1).astype(np.float32)
        import torch as _torch
        x_t, y_t = _torch.from_numpy(x), _torch.from_numpy(y)
        params = {'processor_type_x': None, 'processor_type_y': None,
                  'processor_params_x': {}, 'processor_params_y': {},
                  'dataset_device': 'cpu'}

        key = _dataset_cache_key(x_t, y_t, params)
        stale_dataset = MagicMock(name='stale_dataset_should_not_be_reused')
        decoy_tensor = _torch.randn(3, 3)  # weakly-referenceable, but NOT x_t
        with _DATASET_CACHE_LOCK:
            _DATASET_CACHE[key] = (stale_dataset, _safe_weakref(decoy_tensor), None)

        real_dataset = MagicMock()
        real_dataset.x_data = x_t
        real_dataset.y_data = y_t
        with patch('neural_mi.analysis.task.create_dataset', return_value=real_dataset) as mock_cd, \
             patch('neural_mi.analysis.task.build_critic') as mock_bc:
            mock_bc.return_value = MagicMock()
            mock_bc.return_value.parameters.return_value = iter([])
            full_params = {**params,
                           'embedding_model': 'mlp', 'n_epochs': 1, 'batch_size': 4,
                           'patience': 1000, 'learning_rate': 1e-3, 'train_fraction': 0.8,
                           'n_test_blocks': 2, 'estimator_name': 'infonce',
                           'output_units': 'nats', 'verbose': False, 'show_progress': False}
            from neural_mi.analysis.task import run_training_task
            try:
                run_training_task((x_t, y_t, full_params, 'collision_test'))
            except Exception:
                pass  # only verifying create_dataset was actually called, not full training
        mock_cd.assert_called_once()

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


class TestJointMarginalDifference:
    """Regression tests for the shared joint/marginal/difference helper
    (C-something: extracted from conditional.py and transfer.py x2, which had
    the identical pattern duplicated three times)."""

    def _patch_sweep(self, monkeypatch, joint_vals, marginal_vals):
        """Make ParameterSweep.run() return canned train_mi values in
        sequence: joint_vals on the first call, marginal_vals on the second."""
        from neural_mi.analysis import sweep as sweep_module
        calls = {'n': 0}

        def fake_run(self, sweep_grid=None, n_workers=1, is_proc_sweep=False, **kwargs):
            calls['n'] += 1
            vals = joint_vals if calls['n'] == 1 else marginal_vals
            return [{'train_mi': v} for v in vals]

        monkeypatch.setattr(sweep_module.ParameterSweep, 'run', fake_run)
        return calls

    def test_computes_correct_difference(self, monkeypatch):
        from neural_mi.analysis.sweep import _joint_marginal_difference
        self._patch_sweep(monkeypatch, joint_vals=[2.0, 2.2], marginal_vals=[0.5, 0.7])
        diff, mi_joint, mi_marginal, res_j, res_m = _joint_marginal_difference(
            None, None, None, None, {}, None, 1,
            quantity_name="Test Quantity", joint_label="J", marginal_label="M",
            joint_key="j_key", marginal_key="m_key",
        )
        assert mi_joint == pytest.approx(2.1)
        assert mi_marginal == pytest.approx(0.6)
        assert diff == pytest.approx(1.5)
        assert len(res_j) == 2 and len(res_m) == 2

    def test_raises_when_joint_runs_all_fail(self, monkeypatch):
        from neural_mi.analysis.sweep import _joint_marginal_difference
        from neural_mi.analysis import sweep as sweep_module

        def fake_run(self, sweep_grid=None, n_workers=1, is_proc_sweep=False, **kwargs):
            return [{'no_mi_here': True}]  # no 'train_mi' key -> filtered out

        monkeypatch.setattr(sweep_module.ParameterSweep, 'run', fake_run)
        with pytest.raises(RuntimeError, match="Test Quantity.*I\\(J\\)"):
            _joint_marginal_difference(
                None, None, None, None, {}, None, 1,
                quantity_name="Test Quantity", joint_label="J", marginal_label="M",
                joint_key="j_key", marginal_key="m_key",
            )

    def test_negative_difference_warns_with_correct_dict_keys(self, monkeypatch):
        """The warning must name *this call's* joint_key/marginal_key, not a
        hardcoded pair from one of the three original call sites."""
        from neural_mi.analysis.sweep import _joint_marginal_difference
        self._patch_sweep(monkeypatch, joint_vals=[0.1], marginal_vals=[0.9])
        with pytest.warns(UserWarning, match="my_joint_key.*my_marginal_key"):
            diff, *_ = _joint_marginal_difference(
                None, None, None, None, {}, None, 1,
                quantity_name="Test Quantity", joint_label="J", marginal_label="M",
                joint_key="my_joint_key", marginal_key="my_marginal_key",
            )
        assert diff < 0

    def test_positive_difference_does_not_warn(self, monkeypatch):
        from neural_mi.analysis.sweep import _joint_marginal_difference
        self._patch_sweep(monkeypatch, joint_vals=[0.9], marginal_vals=[0.1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _joint_marginal_difference(
                None, None, None, None, {}, None, 1,
                quantity_name="Test Quantity", joint_label="J", marginal_label="M",
                joint_key="j_key", marginal_key="m_key",
            )
        assert not any(issubclass(w.category, UserWarning) for w in caught)
