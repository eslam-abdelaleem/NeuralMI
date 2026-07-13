"""Shared pytest configuration.

When the suite runs under pytest-xdist (the default; see ``pytest.ini``), each
worker is a separate process. If every worker let torch/BLAS spin up as many
threads as there are cores, the workers would oversubscribe the CPU and run
*slower* than serial. Pin each worker to a single thread so the parallelism
comes from the workers, not from nested threading. This has no effect on what
the tests check -- only on how the numerical libraries schedule work.

A plain (non-xdist) run, e.g. ``pytest -n0``, is left multi-threaded.
"""
import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
