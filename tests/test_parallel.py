# tests/test_parallel.py
"""Tests for neural_mi/parallel.py: dispatch_tasks and the workers of worker_pool."""
import os

import torch

from neural_mi.parallel import dispatch_tasks, worker_pool


def _square(x: int) -> int:
    """Module-level (picklable) function for the n_workers>1 spawn path."""
    return x * x


def _threads(_) -> int:
    return torch.get_num_threads()


class TestDispatchTasks:
    """dispatch_tasks must behave identically whether run sequentially or
    across a worker pool -- only the execution path should differ."""

    def test_empty_tasks_returns_empty_list(self):
        assert dispatch_tasks([], _square, n_workers=1) == []
        assert dispatch_tasks([], _square, n_workers=4) == []

    def test_sequential_path_preserves_order(self):
        tasks = list(range(10))
        results = dispatch_tasks(tasks, _square, n_workers=1, show_progress=False)
        assert results == [x * x for x in tasks]

    def test_single_task_uses_sequential_path_even_with_n_workers_gt_1(self):
        # len(tasks) <= 1 should short-circuit to the sequential path
        # regardless of n_workers, matching the pairwise/dimensionality precedent.
        results = dispatch_tasks([7], _square, n_workers=4, show_progress=False)
        assert results == [49]

    def test_parallel_path_matches_sequential_path(self):
        tasks = list(range(20))
        sequential = dispatch_tasks(tasks, _square, n_workers=1, show_progress=False)
        parallel = dispatch_tasks(tasks, _square, n_workers=4, show_progress=False)
        assert parallel == sequential
        assert parallel == [x * x for x in tasks]


class TestWorkerThreads:
    """Workers that each started PyTorch's full thread pool would oversubscribe the cores."""

    def test_each_worker_gets_its_share_of_the_cores(self, monkeypatch):
        monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
        with worker_pool(2) as pool:
            assert set(pool.map(_threads, range(4))) == {max(1, os.cpu_count() // 2)}

    def test_threads_set_by_the_user_are_kept(self, monkeypatch):
        monkeypatch.setenv('OMP_NUM_THREADS', '3')
        monkeypatch.delenv('MKL_NUM_THREADS', raising=False)      # set to 1 in xdist workers (conftest.py)
        with worker_pool(2) as pool:
            assert set(pool.map(_threads, range(4))) == {3}
