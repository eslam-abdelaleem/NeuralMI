# neural_mi/parallel.py
"""The worker pools the library runs work in, and a helper that dispatches tasks to one.

Every parallel path (sweeps, rigorous ladders, permutation tests, pairwise
channels, the terms and values of the named quantities) opens its pool with
:func:`worker_pool`, so every worker starts the same way: with the parent's
logging level, and with its share of the CPU threads.
"""
import os
from typing import Any, Callable, List

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from neural_mi.utils import _configure_multiprocessing
from neural_mi.logger import CapturedTask, _worker_log_init, logger, released


def threads_per_worker(n_processes: int) -> int:
    """CPU threads for each of ``n_processes`` workers.

    PyTorch gives every process a pool of threads sized to the machine. Workers
    that each start such a pool oversubscribe the cores, and small networks then
    train more slowly in parallel than they would with one thread each. A worker
    therefore gets an equal share of the cores, or the ``OMP_NUM_THREADS`` a user
    set. The count is set in each worker explicitly: how PyTorch reads
    ``OMP_NUM_THREADS`` on its own differs between platforms.
    """
    if 'OMP_NUM_THREADS' in os.environ:
        return int(os.environ['OMP_NUM_THREADS'])
    return max(1, (os.cpu_count() or 1) // n_processes)


def _init_worker(level: int, n_threads: int) -> None:
    _worker_log_init(level)
    torch.set_num_threads(n_threads)


def worker_pool(n_processes: int):
    """A ``spawn`` pool of ``n_processes`` workers, each with the parent's logging level and
    :func:`threads_per_worker` threads. Use it as a context manager."""
    _configure_multiprocessing()
    return mp.get_context('spawn').Pool(processes=n_processes, initializer=_init_worker,
                                        initargs=(logger.level, threads_per_worker(n_processes)))


def dispatch_tasks(
    tasks: List[Any],
    fn: Callable[[Any], Any],
    n_workers: int = 1,
    show_progress: bool = True,
    desc: str = "Dispatching tasks",
) -> List[Any]:
    """Run ``fn`` over every item in *tasks*, in parallel when it pays off.

    Sequential (optionally ``tqdm``-wrapped) when ``n_workers <= 1`` or there
    is only one task, since a pool has nothing to gain in either case.
    Otherwise dispatches to a :func:`worker_pool` of ``n_workers``, using
    ``imap`` to preserve task order.

    Parameters
    ----------
    tasks : list
        One entry per unit of work. Each entry is passed as the single
        argument to *fn*; callers with multiple pieces of data per task
        should bundle them into one tuple/dict per task themselves (matching
        the convention already used by ``_run_pair_task_for_pool`` etc.).
    fn : Callable[[Any], Any]
        Must be a module-level function (not a closure or lambda) to be
        picklable across the ``spawn`` boundary. Receives one task and
        returns its result.
    n_workers : int, default=1
        Number of worker processes. ``<= 1`` runs sequentially in-process.
    show_progress : bool, default=True
        Show a ``tqdm`` progress bar.
    desc : str
        Progress bar label.

    Returns
    -------
    list
        Results in the same order as *tasks*.
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return []

    if n_workers <= 1 or n_tasks <= 1:
        return [
            fn(task)
            for task in tqdm(tasks, desc=desc, disable=not show_progress or n_tasks == 1)
        ]

    logger.info(f"Dispatching {n_tasks} tasks across {n_workers} workers...")
    with worker_pool(n_workers) as pool:
        return list(tqdm(
            released(pool.imap(CapturedTask(fn), tasks)), total=n_tasks, desc=desc, disable=not show_progress
        ))
