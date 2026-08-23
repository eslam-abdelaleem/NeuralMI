# neural_mi/parallel.py
"""Shared task-dispatch helper for parallelising independent work across a pool.

This is the same ``mp.get_context('spawn').Pool(n_workers)`` idiom already
used independently in ``analysis/rigorous.py`` (x2), ``analysis/dimensionality.py``'s
``_dispatch_splits``, ``analysis/pairwise.py``'s ``_dispatch_pairs``,
``analysis/sweep.py``'s ``ParameterSweep._run_parallel``, and ``run.py``'s
permutation-test dispatch, factored into one place for ``neural_mi/quantities.py``
to use rather than adding a seventh independent copy.
"""
import multiprocessing as mp
from typing import Any, Callable, List

from tqdm.auto import tqdm

from neural_mi.utils import _configure_multiprocessing
from neural_mi.logger import logger


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
    Otherwise dispatches via a ``spawn``-context ``Pool(n_workers)``, using
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
    _configure_multiprocessing()
    with mp.get_context('spawn').Pool(processes=n_workers) as pool:
        return list(tqdm(
            pool.imap(fn, tasks), total=n_tasks, desc=desc, disable=not show_progress
        ))
