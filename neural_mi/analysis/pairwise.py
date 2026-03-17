# neural_mi/analysis/pairwise.py
"""Estimates a pairwise mutual information matrix across channel pairs.

**Self-pairwise** (x_data only): estimates MI between every unique pair of
channels ``(i, j)`` with ``i < j`` inside *x_data* and returns the upper
triangle of the symmetric MI matrix.

**Cross-pairwise** (x_data + y_data): estimates MI between every channel of
*x_data* and every channel of *y_data*, producing a full ``(n_ch_x × n_ch_y)``
matrix.

Results are returned as a :class:`pandas.DataFrame` with columns
``ch_x``, ``ch_y``, ``mi_estimate``.
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

from neural_mi.analysis.sweep import ParameterSweep
from neural_mi.logger import logger


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
        Number of parallel workers for each pair's sweep. Defaults to 1.
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
          ``mi_estimate``.
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
        records = []

        for idx, (i, j) in enumerate(pairs):
            logger.info(f"  Pair {idx + 1}/{len(pairs)}: x_ch={i}, y_ch={j}")
            xi = x_data[:, i: i + 1, :]
            yj = y_data[:, j: j + 1, :]
            sweep = ParameterSweep(x_data=xi, y_data=yj, base_params=base_params.copy())
            results = sweep.run(
                sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False
            )
            vals = [r['train_mi'] for r in results if 'train_mi' in r]
            if not vals:
                logger.warning(f"  Pair x_ch={i}, y_ch={j}: all runs failed, recording NaN.")
                mi_ij = float('nan')
            else:
                mi_ij = float(np.mean(vals))
            mi_matrix[i, j] = mi_ij
            records.append({'ch_x': i, 'ch_y': j, 'mi_estimate': mi_ij})

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
        records = []

        for idx, (i, j) in enumerate(pairs):
            logger.info(f"  Pair {idx + 1}/{len(pairs)}: channels ({i}, {j})")
            xi = x_data[:, i: i + 1, :]
            xj = x_data[:, j: j + 1, :]
            sweep = ParameterSweep(x_data=xi, y_data=xj, base_params=base_params.copy())
            results = sweep.run(
                sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False
            )
            vals = [r['train_mi'] for r in results if 'train_mi' in r]
            if not vals:
                logger.warning(f"  Pair ({i}, {j}): all runs failed, recording NaN.")
                mi_ij = float('nan')
            else:
                mi_ij = float(np.mean(vals))
            mi_matrix[i, j] = mi_ij
            mi_matrix[j, i] = mi_ij  # symmetric
            records.append({'ch_x': i, 'ch_y': j, 'mi_estimate': mi_ij})

        df = pd.DataFrame(records)
        logger.info("Pairwise MI (self) estimation complete.")
        return {
            'mi_matrix': mi_matrix,
            'dataframe': df,
            'n_channels': n_channels,
        }
