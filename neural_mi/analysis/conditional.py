# neural_mi/analysis/conditional.py
"""Implements conditional mutual information (CMI) estimation.

CMI between X and Y given W is computed by the chain-rule difference:
    I(X; Y | W) = I(X, W; Y) - I(W; Y)

Both terms are estimated independently using ``ParameterSweep``, so the
existing MI machinery (estimators, critics, augmentation) is reused verbatim.
They read one plan of X, W and Y (:mod:`neural_mi.data.plan`), the joint term
X and W and the conditioning term W, so both terms train and are evaluated on
the same rows.
"""
from typing import Any, Dict, Optional

from neural_mi.analysis.sweep import (combined_spread, rung, _joint_marginal_difference,
                                      amplification_factor)
from neural_mi.data.plan import Roles, RowPlan, TermRows


def run_conditional_mi(
    plan: RowPlan,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    align: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimates conditional mutual information I(X; Y | W).

    Uses the chain-rule identity:
        I(X; Y | W) = I(XW; Y) - I(W; Y)

    Both component MI values are estimated via ``ParameterSweep`` with the
    same ``base_params``, so all hyperparameters (estimator, critic, embedding,
    training schedule) are shared, and on the same rows: each term selects its
    streams from ``plan``.

    Parameters
    ----------
    plan : RowPlan
        The plan of the streams ``'x'``, ``'w'`` and ``'y'``.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. Passed to both sweep runs.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid, e.g. ``{'run_id': range(5)}``.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.
    align : {None, 'dual_branch'}, optional
        ``None`` (the default) joins X and W along the channel axis into the
        joint term's X. ``'dual_branch'`` passes them as a pair, for the named
        quantities whose two parts span different stretches of time.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - ``'cmi_estimate'`` (float): point estimate of I(X;Y|W).
        - ``'mi_xw_y'`` (float): mean test MI for I(XW; Y).
        - ``'mi_w_y'`` (float): mean test MI for I(W; Y).
        - ``'amplification_factor'`` (float): error-amplification factor
          ``(|I(XW;Y)| + |I(W;Y)|) / |CMI|``.  CMI is a difference of two
          separately-trained estimates, so a relative error of ``eps`` on each
          component becomes roughly ``amplification_factor * eps`` on the
          result when their errors are independent.  Values near 1 are safe; values >= 10 mean the point
          estimate should not be read without its components.  See
          :func:`neural_mi.analysis.sweep.amplification_factor`.
        - ``'raw_xw_y'`` : list of result dicts from the XW→Y sweep.
        - ``'raw_w_y'`` : list of result dicts from the W→Y sweep.
    """
    dual = align == 'dual_branch'
    joint = TermRows(plan, Roles(x=('x', 'w'), join='pair' if dual else 'concat', channels=True))
    marginal = TermRows(plan, Roles(x=('w',), channels=True))
    cmi, mi_xw_y, mi_w_y, results_xw_y, results_w_y, _per_run = _joint_marginal_difference(
        joint, None, marginal, None,
        base_params, sweep_grid, n_workers,
        quantity_name="Conditional MI (dual-branch)" if dual else "Conditional MI",
        joint_label="XC;Y" if dual else "XW;Y", marginal_label="C;Y" if dual else "W;Y",
        joint_key="mi_xw_y", marginal_key="mi_w_y", condition_at=None if dual else 'last',
    )
    return {
        'cmi_estimate': cmi,
        'mi_xw_y': mi_xw_y,
        'mi_w_y': mi_w_y,
        'amplification_factor': amplification_factor([mi_xw_y, mi_w_y], cmi),
        'mi_estimate_std': combined_spread(_per_run, (1, -1)),
        'raw_xw_y': results_xw_y,
        'raw_w_y': results_w_y,
    }


def _cmi_rigorous_scalar(plan, _y, bp, sweep_grid=None, align=None) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of CMI.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1``, must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching the
    outer-loop-gets-workers / inner-loop-sequential convention used for
    dimensionality-mode splits. ``plan`` is the chunk's plan
    (:meth:`neural_mi.data.plan.RowPlan.cut`), whose rows both terms read.
    """
    raw = run_conditional_mi(plan, bp, sweep_grid=sweep_grid, n_workers=1, align=align)
    return rung(raw, (('mi_xw_y', 1), ('mi_w_y', -1)))
