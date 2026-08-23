# neural_mi/analysis/interaction.py
"""Implements interaction information (a three-population quantity).

Interaction information measures how much shared information between X and Y
changes once a third population W is also observed:

    II = I(X, W; Y) - I(X; Y) - I(W; Y)

Unlike every other quantity in this taxonomy, this is not a single
conditional-MI call -- it's three separate MI estimates combined by a
formula. The three-way orchestration is new; the underlying estimation
machinery (``ParameterSweep``, the joint/marginal-difference pattern) is
reused verbatim from ``analysis/sweep.py``/``analysis/conditional.py``.
"""
import torch
from typing import Dict, Any, Optional

from neural_mi.analysis.sweep import ParameterSweep, _joint_marginal_difference
from neural_mi.logger import logger


def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(-1) if t.ndim == 2 else t


def _single_mi_mean(
    x: torch.Tensor, y: torch.Tensor, base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]], n_workers: int,
    *, quantity_name: str, label: str,
) -> tuple:
    """Run one ``ParameterSweep``, return ``(mean(train_mi), raw_results)``.

    The single-term counterpart to ``_joint_marginal_difference``'s
    joint/marginal pair -- needed here for interaction information's
    standalone I(W;Y) term, which isn't itself a difference of two sweeps.
    """
    logger.info(f"{quantity_name}: estimating I({label})...")
    sweep = ParameterSweep(x_data=x, y_data=y, base_params=base_params.copy())
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=False)
    vals = [r['train_mi'] for r in results if 'train_mi' in r]
    if not vals:
        raise RuntimeError(f"{quantity_name}: all I({label}) runs failed, no valid train_mi values.")
    import numpy as np
    return float(np.mean(vals)), results


def run_interaction_information(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    w_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Estimates interaction information II = I(X,W;Y) - I(X;Y) - I(W;Y).

    Reuses ``_joint_marginal_difference`` once for the (joint=XW, marginal=X)
    pair, which yields both I(X,W;Y) and I(X;Y) for free, plus one standalone
    ``_single_mi_mean`` call for I(W;Y) -- three sweeps total, not four,
    avoiding a redundant recomputation of I(X;Y).

    Parameters
    ----------
    x_data : torch.Tensor
        Data for population X, shape ``(n_samples, n_channels_x, window_size)``.
    y_data : torch.Tensor
        Data for population Y, shape ``(n_samples, n_channels_y, window_size)``.
    w_data : torch.Tensor
        Data for the third population W, shape ``(n_samples, n_channels_w, window_size)``.
        Must share X's window size exactly (concatenated with X along the
        channel axis to build the joint I(X,W;Y) term).
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. Passed to all three sweeps.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid, e.g. ``{'run_id': range(5)}``.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - ``'interaction_info'`` : float — point estimate of II.
        - ``'mi_xw_y'`` : float — mean I(X,W;Y).
        - ``'mi_x_y'`` : float — mean I(X;Y).
        - ``'mi_w_y'`` : float — mean I(W;Y).
        - ``'raw_xw_y'``, ``'raw_x_y'``, ``'raw_w_y'`` : list of result dicts.
    """
    x_data = _ensure_3d(x_data)
    y_data = _ensure_3d(y_data)
    w_data = _ensure_3d(w_data)
    device = x_data.device
    y_data = y_data.to(device)
    w_data = w_data.to(device)

    if x_data.shape[0] != y_data.shape[0] or x_data.shape[0] != w_data.shape[0]:
        raise ValueError(
            "x_data, y_data, and w_data must have the same number of samples. "
            f"Got shapes {tuple(x_data.shape)}, {tuple(y_data.shape)}, {tuple(w_data.shape)}."
        )
    if x_data.shape[2] != w_data.shape[2]:
        raise ValueError(
            "x_data and w_data must have the same window size to be concatenated "
            f"into XW. Got window sizes {x_data.shape[2]} and {w_data.shape[2]} "
            f"(full shapes {tuple(x_data.shape)}, {tuple(w_data.shape)})."
        )

    xw_data = torch.cat([x_data, w_data], dim=1)

    _diff, mi_xw_y, mi_x_y, raw_xw_y, raw_x_y = _joint_marginal_difference(
        xw_data, y_data, x_data, y_data,
        base_params, sweep_grid, n_workers,
        quantity_name="Interaction information",
        joint_label="X,W;Y", marginal_label="X;Y",
        joint_key="mi_xw_y", marginal_key="mi_x_y",
    )
    mi_w_y, raw_w_y = _single_mi_mean(
        w_data, y_data, base_params, sweep_grid, n_workers,
        quantity_name="Interaction information", label="W;Y",
    )

    ii = mi_xw_y - mi_x_y - mi_w_y
    logger.info(
        f"Interaction information: I(X,W;Y)={mi_xw_y:.4f}, I(X;Y)={mi_x_y:.4f}, "
        f"I(W;Y)={mi_w_y:.4f}, II={ii:.4f} nats (converted to requested output_units by the caller)."
    )

    return {
        'interaction_info': ii,
        'mi_xw_y': mi_xw_y,
        'mi_x_y': mi_x_y,
        'mi_w_y': mi_w_y,
        'raw_xw_y': raw_xw_y,
        'raw_x_y': raw_x_y,
        'raw_w_y': raw_w_y,
    }


def _ii_rigorous_scalar(x_s, y_s, bp, w_data=None, sweep_grid=None) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of
    interaction information.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1`` -- must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching
    ``_cmi_rigorous_scalar``'s convention. ``w_data`` arrives here already
    sliced to this gamma-chunk's samples via ``extra_data``.
    """
    raw = run_interaction_information(x_s, y_s, w_data, bp, sweep_grid=sweep_grid, n_workers=1)
    return raw['interaction_info']
