# tests/test_parallel.py
"""Tests for neural_mi/parallel.py's dispatch_tasks helper."""
from neural_mi.parallel import dispatch_tasks


def _square(x: int) -> int:
    """Module-level (picklable) function for the n_workers>1 spawn path."""
    return x * x


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
