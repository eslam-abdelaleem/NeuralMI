# neural_mi/analysis/interaction.py
"""Implements interaction information (a three-population quantity).

Interaction information measures how much shared information between X and Y
changes once a third population W is also observed:

    II = I(X, W; Y) - I(X; Y) - I(W; Y)

Unlike every other quantity in this taxonomy, this is not a single
conditional-MI call. It's three separate MI estimates combined by a
formula. The three-way orchestration is new; the underlying estimation
machinery (``ParameterSweep``, the joint/marginal-difference pattern) is
reused verbatim from ``analysis/sweep.py``/``analysis/conditional.py``.
"""
import math
from typing import Dict, Any, Optional

from neural_mi.analysis.sweep import (combined_spread, combine_terms, rung, ParameterSweep,
                                      _joint_marginal_difference, warn_combination,
                                      amplification_factor)
from neural_mi.data.plan import Roles, RowPlan, TermRows
from neural_mi.logger import logger
from neural_mi.embeddings_io import with_model_labels
from neural_mi.utils import mi_report_units


def _train_single_term(
    x, y, base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]], n_workers: int,
    *, quantity_name: str, label: str,
) -> list:
    """Train one term on its own with ``ParameterSweep`` and return its task results.

    The single-term counterpart to ``_joint_marginal_difference``'s
    joint/marginal pair, needed here for interaction information's
    standalone I(W;Y) term, which isn't itself a difference of two sweeps.
    The caller combines the terms (:func:`neural_mi.analysis.sweep.combine_terms`).
    """
    logger.info(f"{quantity_name}: estimating I({label})...")
    sweep = ParameterSweep(x_data=x, y_data=y, base_params=base_params.copy())
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers)
    if not any('train_mi' in r for r in results):
        raise RuntimeError(f"{quantity_name}: all I({label}) runs failed, no valid train_mi values.")
    return results


def run_interaction_information(
    plan: RowPlan,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """Estimates interaction information II = I(X,W;Y) - I(X;Y) - I(W;Y).

    Reuses ``_joint_marginal_difference`` once for the (joint=XW, marginal=X)
    pair, which yields both I(X,W;Y) and I(X;Y) for free, plus one standalone
    ``_train_single_term`` call for I(W;Y), three sweeps total, not four,
    avoiding a redundant recomputation of I(X;Y). The three are combined repeat
    by repeat by :func:`neural_mi.analysis.sweep.combine_terms`. Every term
    selects its streams from ``plan``, so all three read the same rows.

    Parameters
    ----------
    plan : RowPlan
        The plan of the streams ``'x'``, ``'w'`` and ``'y'``. The joint term
        reads the pair (W, X): the rest of X beside the conditioning set of
        the difference I(X,W;Y) - I(X;Y), which is X.
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
        - ``'interaction_info'`` (float): point estimate of II.
        - ``'mi_xw_y'`` (float): mean I(X,W;Y).
        - ``'mi_x_y'`` (float): mean I(X;Y).
        - ``'mi_w_y'`` (float): mean I(W;Y).
        - ``'amplification_factor'`` (float): error-amplification factor
          ``(|I(XW;Y)| + |I(X;Y)| + |I(W;Y)|) / |II|``.  Interaction information
          combines *three* separately-trained estimates, so it amplifies
          component error more readily than a two-term difference; a relative
          error of ``eps`` on each component becomes roughly
          ``amplification_factor * eps`` on II when their errors are independent.  Values >= 10 mean the sign of
          II may not be determined by the data.  See
          :func:`neural_mi.analysis.sweep.amplification_factor`.
        - ``'raw_xw_y'``, ``'raw_x_y'``, ``'raw_w_y'`` : list of result dicts.
    """
    _, _, _, raw_xw_y, raw_x_y, _ = _joint_marginal_difference(
        TermRows(plan, Roles(x=('w', 'x'))), None, TermRows(plan, Roles(x=('x',))), None,
        base_params, sweep_grid, n_workers,
        quantity_name="Interaction information",
        joint_label="X,W;Y", marginal_label="X;Y",
        joint_key="mi_xw_y", marginal_key="mi_x_y",
        reported=False,
    )
    raw_w_y = _train_single_term(
        TermRows(plan, Roles(x=('w',))), None,
        {**with_model_labels(base_params, component='mi_w_y'), '_difference_term': True},
        sweep_grid, n_workers,
        quantity_name="Interaction information", label="W;Y",
    )

    # The three terms are combined repeat by repeat, by the rule every combined
    # quantity follows: a repeat in which any of them failed has no value.
    combined = combine_terms([("X,W;Y", 'mi_xw_y', raw_xw_y, 1), ("X;Y", 'mi_x_y', raw_x_y, -1),
                              ("W;Y", 'mi_w_y', raw_w_y, -1)],
                             quantity_name="Interaction information", warn=True)
    ii = combined['value']
    mi_xw_y, mi_x_y, mi_w_y = (combined['means'][k] for k in ('mi_xw_y', 'mi_x_y', 'mi_w_y'))
    _scale, _units = mi_report_units(base_params)
    logger.info(
        f"Interaction information: I(X,W;Y)={mi_xw_y * _scale:.4f}, "
        f"I(X;Y)={mi_x_y * _scale:.4f}, I(W;Y)={mi_w_y * _scale:.4f}, "
        f"II={ii * _scale:.4f} {_units}."
    )
    if not math.isnan(ii):
        warn_combination("Interaction information", ii, ("X,W;Y", mi_xw_y, 'mi_xw_y'),
                         [("X;Y", mi_x_y, 'mi_x_y'), ("W;Y", mi_w_y, 'mi_w_y')],
                         base_params, signed=True)

    return {
        'interaction_info': ii,
        'mi_xw_y': mi_xw_y,
        'mi_x_y': mi_x_y,
        'mi_w_y': mi_w_y,
        'amplification_factor': amplification_factor([mi_xw_y, mi_x_y, mi_w_y], ii),
        'mi_estimate_std': combined_spread(combined['per_run'], (1, -1, -1)),
        'raw_xw_y': raw_xw_y,
        'raw_x_y': raw_x_y,
        'raw_w_y': raw_w_y,
    }


def _ii_rigorous_scalar(plan, _y, bp, sweep_grid=None) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of
    interaction information.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1``, must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching
    ``_cmi_rigorous_scalar``'s convention. ``plan`` is the chunk's plan.
    """
    raw = run_interaction_information(plan, bp, sweep_grid=sweep_grid, n_workers=1)
    return rung(raw, (('mi_xw_y', 1), ('mi_x_y', -1), ('mi_w_y', -1)))
