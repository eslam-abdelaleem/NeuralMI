# neural_mi/analysis/pairwise.py
"""Estimates a pairwise mutual information matrix across channel pairs.

**Self-pairwise** (x_data only): estimates MI between every unique pair of
channels ``(i, j)`` with ``i < j`` inside *x_data* and returns the upper
triangle of the symmetric MI matrix.

**Cross-pairwise** (x_data + y_data): estimates MI between every channel of
*x_data* and every channel of *y_data*, producing a full ``(n_ch_x × n_ch_y)``
matrix.

Results are returned as a :class:`pandas.DataFrame` with columns
``ch_x``, ``ch_y``, ``mi_mean``, ``mi_std``.

.. note::
   Each channel pair here trains a full neural MI estimator. For
   single-time-bin channels (no temporal structure within the window) this is
   more expensive and higher-variance than a classical (e.g. correlation- or
   histogram-based) estimator would be. Left as-is since neural estimation is
   what generalizes across modalities; consider a classical estimator instead
   if you know your channels are single-time-bin and need speed.
"""
import torch
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, List, Tuple

from neural_mi.analysis.sweep import ParameterSweep
from neural_mi.logger import logger
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu


def _run_pair_task(args: tuple) -> Dict[str, Any]:
    """Run one channel pair's MI sweep and return its summary row.

    ``n_workers`` here controls the *inner* sweep (e.g. averaging over a
    ``run_id`` sweep_grid for one pair) -- kept separate from how many pairs
    are dispatched concurrently, set by the caller (see ``_dispatch_pairs``).
    """
    i, j, xi, yj, base_params, sweep_grid, n_workers = args
    sweep = ParameterSweep(x_data=xi, y_data=yj, base_params=base_params.copy())
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False)
    vals = [r['train_mi'] for r in results if 'train_mi' in r]
    if not vals:
        logger.warning(f"  Pair (ch_x={i}, ch_y={j}): all runs failed, recording NaN.")
        mi_mean, mi_std = float('nan'), float('nan')
    else:
        mi_mean = float(np.mean(vals))
        mi_std = float(np.std(vals)) if len(vals) > 1 else 0.0
    return {'ch_x': i, 'ch_y': j, 'mi_mean': mi_mean, 'mi_std': mi_std}


def _run_pair_task_for_pool(args: tuple) -> Dict[str, Any]:
    """Top-level, picklable wrapper for ``Pool.imap`` -- forces the inner
    sweep to ``n_workers=1`` to avoid nested multiprocessing pools, since
    parallelism is spent across pairs instead (see ``_dispatch_pairs``).
    """
    i, j, xi, yj, base_params, sweep_grid = args
    return _run_pair_task((i, j, xi, yj, base_params, sweep_grid, 1))


def _dispatch_pairs(pair_tasks: List[tuple], n_workers: int, show_progress: bool) -> List[Dict[str, Any]]:
    """Execute per-pair MI sweeps, parallelising *across pairs* when ``n_workers > 1``.

    Mirrors ``analysis/dimensionality.py``'s ``_dispatch_splits``: a single
    pair forwards ``n_workers`` into its own inner sweep; multiple pairs with
    ``n_workers > 1`` are dispatched to a ``Pool(n_workers)`` with the inner
    sweep forced to ``n_workers=1`` to avoid nested pools.
    """
    n_pairs = len(pair_tasks)

    if n_workers <= 1 or n_pairs <= 1:
        inner_workers = n_workers if n_pairs == 1 else 1
        records = []
        for idx, (i, j, xi, yj, base_params, sweep_grid) in enumerate(
            tqdm(pair_tasks, desc="Pairwise MI", disable=not show_progress or n_pairs == 1)
        ):
            logger.info(f"  Pair {idx + 1}/{n_pairs}: ch_x={i}, ch_y={j}")
            records.append(_run_pair_task((i, j, xi, yj, base_params, sweep_grid, inner_workers)))
        return records

    logger.info(f"Parallelising {n_pairs} channel pairs across {n_workers} workers...")
    _configure_multiprocessing()
    with mp.get_context('spawn').Pool(processes=n_workers) as pool:
        records = list(tqdm(
            pool.imap(_run_pair_task_for_pool, pair_tasks), total=n_pairs,
            desc="Pairwise MI", disable=not show_progress
        ))
    return records


def run_pairwise_mi(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    y_data: Optional[torch.Tensor] = None,
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Estimates MI between channel pairs.

    **Self-pairwise** (y_data=None): MI between every unique pair ``(i, j)``
    with ``i < j`` from *x_data*.  Returns ``C(n_channels, 2)`` rows.

    **Cross-pairwise** (y_data provided): MI between every channel of *x_data*
    and every channel of *y_data*.  Returns ``n_ch_x × n_ch_y`` rows.

    Parameters
    ----------
    x_data : torch.Tensor
        Multi-channel data, shape ``(n_samples, n_channels_x, window_size)``.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator.
    y_data : torch.Tensor, optional
        Second multi-channel dataset for cross-pairwise mode,
        shape ``(n_samples, n_channels_y, window_size)``.  When *None* the
        function falls back to self-pairwise mode on *x_data*.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid (e.g. ``{'run_id': range(5)}``).
    n_workers : int, optional
        Number of parallel workers. Channel pairs are independent, so with
        more than one pair this parallelises *across pairs* (one pair per
        worker); a single pair instead forwards ``n_workers`` into its own
        inner sweep (e.g. a ``run_id`` sweep_grid). Defaults to 1.
    pairs : list of (int, int), optional
        Explicit list of ``(ch_x, ch_y)`` index pairs to estimate.  In
        self-pairwise mode the indices refer to channels of *x_data*.  In
        cross-pairwise mode ``ch_x`` indexes *x_data* and ``ch_y`` indexes
        *y_data*.  If *None*, all relevant pairs are generated automatically.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``'mi_matrix'`` : np.ndarray — MI matrix.
          Shape ``(n_ch_x, n_ch_x)`` for self-pairwise (symmetric, diagonal 0),
          or ``(n_ch_x, n_ch_y)`` for cross-pairwise.
        - ``'dataframe'`` : pd.DataFrame with columns ``ch_x``, ``ch_y``,
          ``mi_mean``, ``mi_std``.
        - ``'n_channels'`` : int or (int, int) — number of channels.
    """
    if x_data.ndim != 3:
        raise ValueError(
            "run_pairwise_mi expects x_data of shape (n_samples, n_channels, window_size). "
            f"Got shape {tuple(x_data.shape)}."
        )

    cross_mode = y_data is not None

    if cross_mode:
        if y_data.ndim != 3:
            raise ValueError(
                "run_pairwise_mi expects y_data of shape (n_samples, n_channels, window_size). "
                f"Got shape {tuple(y_data.shape)}."
            )
        n_ch_x = x_data.shape[1]
        n_ch_y = y_data.shape[1]
        if pairs is None:
            pairs = [(i, j) for i in range(n_ch_x) for j in range(n_ch_y)]

        logger.info(
            f"Pairwise MI (cross): estimating {len(pairs)} pairs "
            f"({n_ch_x} × {n_ch_y} channels)..."
        )

        mi_matrix = np.zeros((n_ch_x, n_ch_y))
        show_progress = base_params.get('show_progress', True)
        pair_tasks = [
            (i, j, _ensure_cpu(x_data[:, i: i + 1, :]), _ensure_cpu(y_data[:, j: j + 1, :]),
             base_params, sweep_grid)
            for (i, j) in pairs
        ]
        records = _dispatch_pairs(pair_tasks, n_workers, show_progress)
        for rec in records:
            mi_matrix[rec['ch_x'], rec['ch_y']] = rec['mi_mean']

        df = pd.DataFrame(records)
        logger.info("Pairwise MI (cross) estimation complete.")
        return {
            'mi_matrix': mi_matrix,
            'dataframe': df,
            'n_channels': (n_ch_x, n_ch_y),
        }

    else:
        # ---- Self-pairwise mode ------------------------------------------------
        n_channels = x_data.shape[1]
        if n_channels < 2:
            raise ValueError(
                f"Pairwise MI requires at least 2 channels, got n_channels={n_channels}."
            )
        if pairs is None:
            pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]

        logger.info(
            f"Pairwise MI (self): estimating {len(pairs)} channel pairs across "
            f"{n_channels} channels..."
        )

        mi_matrix = np.zeros((n_channels, n_channels))
        show_progress = base_params.get('show_progress', True)
        pair_tasks = [
            (i, j, _ensure_cpu(x_data[:, i: i + 1, :]), _ensure_cpu(x_data[:, j: j + 1, :]),
             base_params, sweep_grid)
            for (i, j) in pairs
        ]
        records = _dispatch_pairs(pair_tasks, n_workers, show_progress)
        for rec in records:
            mi_matrix[rec['ch_x'], rec['ch_y']] = rec['mi_mean']
            mi_matrix[rec['ch_y'], rec['ch_x']] = rec['mi_mean']  # symmetric

        df = pd.DataFrame(records)
        logger.info("Pairwise MI (self) estimation complete.")
        return {
            'mi_matrix': mi_matrix,
            'dataframe': df,
            'n_channels': n_channels,
        }
