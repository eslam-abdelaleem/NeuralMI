# neural_mi/device.py
"""The device a call trains on when it names none.

The CPU is the default. A GPU is used only when it is clearly faster for the
call at hand, which depends on the network, the batch size and how many workers
would share the GPU, so it is measured instead of assumed. The networks typical
of neural data are small. On an Apple GPU they trained several times more
slowly than on the CPU, and parallel workers sharing one GPU wait for each
other, while each worker gets its own share of the CPU.

A call that names a device keeps it. Otherwise the process that is about to run
the call's trainings times a few training steps of its first network on the CPU,
with the threads each worker would get. If all the trainings together would
take less than ``GPU_WORTH_SECONDS`` on the CPU, a GPU could save too little to
be worth timing, and the CPU is used. Otherwise the same steps are timed on the
GPU, and the GPU is used if it is at least ``GPU_MARGIN`` times faster than the
CPU workers together. A worker process never times anything: it trains on the
CPU unless its call named a device, so that workers do not time a GPU they all
share or choose different devices.
"""
import multiprocessing as mp
from typing import List, Optional

import torch

from neural_mi.logger import logger
from neural_mi.parallel import threads_per_worker

GPU_WORTH_SECONDS = 120.0   # expected CPU training time below which a GPU is not timed
GPU_MARGIN = 1.5            # the GPU must beat the CPU workers together by this factor


def accelerator() -> Optional[str]:
    """The GPU this machine offers, ``'cuda'`` or ``'mps'``, or None."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return None


def with_device(tasks: List[tuple], n_workers: int) -> List[tuple]:
    """``(x, y, params, run_id)`` tasks with the device of :func:`choose_device` when they name none."""
    if not tasks or tasks[0][2].get('device') is not None:
        return tasks
    device = choose_device(tasks, n_workers)
    return [(x, y, {**params, 'device': device}, run_id) for x, y, params, run_id in tasks]


def choose_device(tasks: List[tuple], n_workers: int) -> str:
    """The device for these training tasks, run ``n_workers`` at a time."""
    gpu = accelerator()
    if gpu is None or mp.parent_process() is not None:
        return 'cpu'
    n_parallel = max(1, min(n_workers, len(tasks)))
    threads = threads_per_worker(n_parallel) if n_parallel > 1 else None
    cpu_step, steps_per_epoch = _time(tasks[0], 'cpu', threads)
    cpu_total = cpu_step * steps_per_epoch * tasks[0][2]['n_epochs'] * len(tasks) / n_parallel
    if cpu_total < GPU_WORTH_SECONDS:
        logger.debug(f"Training on the CPU: at most {cpu_total:.0f} s of training, too little for {gpu} to save.")
        return 'cpu'
    gpu_step, _ = _time(tasks[0], gpu, None)
    device = gpu if cpu_step >= GPU_MARGIN * n_parallel * gpu_step else 'cpu'
    cpu_note = f", with {threads} thread(s) in each of {n_parallel} workers" if n_parallel > 1 else ""
    gpu_note = ", which the workers would share" if n_parallel > 1 else ""
    logger.info(f"Training on {'the CPU' if device == 'cpu' else device}: a training step of the first network "
                f"took {cpu_step * 1e3:.1f} ms on the CPU{cpu_note}, and {gpu_step * 1e3:.1f} ms on {gpu}{gpu_note}. "
                f"Name a device to choose it yourself.")
    return device


def _time(task: tuple, device: str, threads: Optional[int]):
    """Seconds per training step of the task's network on ``device`` and its steps per epoch."""
    from neural_mi.analysis.task import run_training_task
    x, y, params, run_id = task
    previous = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        timed = run_training_task((x, y, {**params, 'device': device, '_time_steps': True}, run_id))
    finally:
        torch.set_num_threads(previous)
    return timed['seconds_per_step'], timed['steps_per_epoch']
