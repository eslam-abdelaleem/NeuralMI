# neural_mi/analysis/conditional.py
"""Implements conditional mutual information (CMI) estimation.

CMI between X and Y given Z is computed by the chain-rule difference:
    I(X; Y | Z) = I(X, Z; Y) - I(Z; Y)

Both terms are estimated independently using ``ParameterSweep``, so the
existing MI machinery (estimators, critics, augmentation) is reused verbatim.
The conditioning variable Z is concatenated with X at the data level before
any windowing or embedding.
"""
import torch
from typing import Dict, Any, Optional

from neural_mi.analysis.sweep import _joint_marginal_difference
from neural_mi.logger import logger

# Continuous windowing adds a deliberate "+1" sample as an interpolation
# safety buffer at window edges (ContinuousWindowDataset._compute_max_samples_
# per_window); categorical windowing does not, since it never interpolates.
# X and a full-resolution categorical Z windowed with the same nominal
# window_size therefore differ by exactly this buffer, not by a real content
# mismatch -- trim to the shorter length rather than raising. Kept at exactly
# 1 (not a larger tolerance) so a genuinely different window_size between X
# and Z -- a real configuration error -- still raises.
_WINDOW_SIZE_TRIM_TOLERANCE = 1


def run_conditional_mi(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    z_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Estimates conditional mutual information I(X; Y | Z).

    Uses the chain-rule identity:
        I(X; Y | Z) = I(XZ; Y) - I(Z; Y)

    Both component MI values are estimated via ``ParameterSweep`` with the
    same ``base_params``, so all hyperparameters (estimator, critic, embedding,
    training schedule) are shared.

    Parameters
    ----------
    x_data : torch.Tensor
        Data for variable X, shape ``(n_samples, n_channels_x, window_size)``.
    y_data : torch.Tensor
        Data for variable Y, shape ``(n_samples, n_channels_y, window_size)``.
    z_data : torch.Tensor
        Conditioning variable Z, shape ``(n_samples, n_channels_z, window_size)``.
        Must share the same sample dimension as x_data and y_data. A window
        axis of size 1 is broadcast across x_data's window_size before
        concatenation, for conditioning variables with no temporal extent
        within a window (e.g. a categorical Z encoded with 'majority_vote' or
        'probability' — see ``run._reshape_categorical_z_for_conditional``).
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. Passed to both sweep runs.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid, e.g. ``{'run_id': range(5)}``.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - ``'cmi_estimate'`` : float — point estimate of I(X;Y|Z).
        - ``'mi_xz_y'`` : float — mean test MI for I(XZ; Y).
        - ``'mi_z_y'`` : float — mean test MI for I(Z; Y).
        - ``'raw_xz_y'`` : list of result dicts from the XZ→Y sweep.
        - ``'raw_z_y'`` : list of result dicts from the Z→Y sweep.
    """
    # Normalise all inputs to the same ndim before shape comparison and cat.
    # StaticDataset delivers (N, C, 1) tensors, but z_data is often passed as
    # raw 2-D (N, C) when no z_processor_type is given.  Unsqueeze the missing
    # trailing window dimension so that torch.cat works on a consistent axis.
    def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1) if t.ndim == 2 else t

    x_data = _ensure_3d(x_data)
    y_data = _ensure_3d(y_data)
    z_data = _ensure_3d(z_data)

    # Ensure all inputs are on the same device (x_data is the reference).
    # z_data in particular may arrive as a raw CPU tensor when no
    # z_processor_type is given, while x_data/y_data may be on MPS/CUDA.
    device = x_data.device
    y_data = y_data.to(device)
    z_data = z_data.to(device)

    if x_data.shape[0] != y_data.shape[0] or x_data.shape[0] != z_data.shape[0]:
        raise ValueError(
            "x_data, y_data, and z_data must have the same number of samples. "
            f"Got shapes {tuple(x_data.shape)}, {tuple(y_data.shape)}, {tuple(z_data.shape)}."
        )
    if x_data.shape[2] != z_data.shape[2]:
        if z_data.shape[2] == 1:
            # Z has no temporal extent within the window (e.g. a per-window
            # categorical summary already folded into channels by
            # _reshape_categorical_z_for_conditional, or any other
            # window-constant conditioning variable) -- broadcast it across
            # X's window so the two can be concatenated along the channel axis.
            z_data = z_data.expand(-1, -1, x_data.shape[2])
        elif abs(x_data.shape[2] - z_data.shape[2]) <= _WINDOW_SIZE_TRIM_TOLERANCE:
            min_w = min(x_data.shape[2], z_data.shape[2])
            logger.warning(
                f"x_data window size ({x_data.shape[2]}) and z_data window size "
                f"({z_data.shape[2]}) differ by {abs(x_data.shape[2] - z_data.shape[2])} "
                f"sample(s) -- likely the continuous processor's interpolation-edge "
                f"buffer (see _compute_max_samples_per_window). Trimming both to the "
                f"shared start, length {min_w}, rather than raising."
            )
            x_data = x_data[:, :, :min_w]
            z_data = z_data[:, :, :min_w]
        else:
            raise ValueError(
                "x_data and z_data must have the same window size to be concatenated "
                f"into XZ. Got window sizes {x_data.shape[2]} and {z_data.shape[2]} "
                f"(full shapes {tuple(x_data.shape)}, {tuple(z_data.shape)})."
            )

    # Build XZ by concatenating along the channel dimension (dim=1)
    xz_data = torch.cat([x_data, z_data], dim=1)

    cmi, mi_xz_y, mi_z_y, results_xz_y, results_z_y = _joint_marginal_difference(
        xz_data, y_data, z_data, y_data,
        base_params, sweep_grid, n_workers,
        quantity_name="Conditional MI",
        joint_label="XZ;Y", marginal_label="Z;Y",
        joint_key="mi_xz_y", marginal_key="mi_z_y",
    )

    return {
        'cmi_estimate': cmi,
        'mi_xz_y': mi_xz_y,
        'mi_z_y': mi_z_y,
        'raw_xz_y': results_xz_y,
        'raw_z_y': results_z_y,
    }
