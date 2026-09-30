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
import torch
from typing import Dict, Any, Optional

from neural_mi.data.shift_windowing import (SHIFT_PACK_CONCAT, SHIFT_PACK_FIRST,
                                            SHIFT_PACK_SECOND,
                                            _REGULAR_GRID_PROCESSOR_TYPES)
from neural_mi.analysis.conditional import _merge_raw_blocks
from neural_mi.analysis.sweep import (combined_spread, ParameterSweep, _joint_marginal_difference,
                                      warn_combination, amplification_factor)
from neural_mi.logger import logger
from neural_mi.embeddings_io import saved_paths, with_model_labels
from neural_mi.utils import mi_report_units

# Mirrors conditional.py's identical constants exactly -- both modes window
# W paired with Y via the same create_dataset(x_data=w_data, y_data=y_data,
# ...) call (see run.py), so both are subject to the same small boundary
# effects: a window-count difference of up to _SAMPLE_COUNT_TRIM_TOLERANCE
# windows between X-paired-with-Y and W-paired-with-Y (a coverage-validation
# difference between two separate create_dataset calls, not a real duration
# mismatch), and a window-size difference of up to
# _WINDOW_SIZE_TRIM_TOLERANCE samples. Every processor now emits
# window_size slots for a window of window_size, so a one-sample difference
# has no known cause; the tolerance stays as a guard against rounding at a
# sample-rate boundary, and says so instead of naming a cause.
_WINDOW_SIZE_TRIM_TOLERANCE = 1
_SAMPLE_COUNT_TRIM_TOLERANCE = 1


def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(-1) if t.ndim == 2 else t


def _single_mi_mean(
    x: torch.Tensor, y: torch.Tensor, base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]], n_workers: int,
    *, quantity_name: str, label: str, is_proc_sweep: bool = False,
) -> tuple:
    """Run one ``ParameterSweep``, return ``(mean(train_mi), raw_results, values)``.

    The single-term counterpart to ``_joint_marginal_difference``'s
    joint/marginal pair, needed here for interaction information's
    standalone I(W;Y) term, which isn't itself a difference of two sweeps.
    """
    logger.info(f"{quantity_name}: estimating I({label})...")
    sweep = ParameterSweep(x_data=x, y_data=y, base_params=base_params.copy())
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers, is_proc_sweep=is_proc_sweep)
    vals = [r['train_mi'] for r in results if 'train_mi' in r]
    if not vals:
        raise RuntimeError(f"{quantity_name}: all I({label}) runs failed, no valid train_mi values.")
    import numpy as np
    return float(np.mean(vals)), results, vals


def run_interaction_information(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    w_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    raw_deferred: bool = False,
    w_processor_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimates interaction information II = I(X,W;Y) - I(X;Y) - I(W;Y).

    Reuses ``_joint_marginal_difference`` once for the (joint=XW, marginal=X)
    pair, which yields both I(X,W;Y) and I(X;Y) for free, plus one standalone
    ``_single_mi_mean`` call for I(W;Y), three sweeps total, not four,
    avoiding a redundant recomputation of I(X;Y).

    Parameters
    ----------
    x_data : torch.Tensor
        Data for population X, shape ``(n_samples, n_channels_x, window_size)``.
    y_data : torch.Tensor
        Data for population Y, shape ``(n_samples, n_channels_y, window_size)``.
    w_data : torch.Tensor
        Data for the third population W, shape ``(n_samples, n_channels_w, window_size)``.
        Concatenated with X along the channel axis to build the joint
        I(X,W;Y) term. Window count and window size must match X's, up to a
        1-sample edge-case boundary difference (see
        ``_SAMPLE_COUNT_TRIM_TOLERANCE``/``_WINDOW_SIZE_TRIM_TOLERANCE``)
        that's trimmed instead of raised.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. Passed to all three sweeps.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid, e.g. ``{'run_id': range(5)}``.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.
    raw_deferred : bool, optional
        ``True`` when ``x_data``/``y_data``/``w_data`` are raw, unwindowed
        arrays and windowing should be deferred to each sweep's own
        dispatch (shift_windows/shift_time reachability), the caller
        (``run.py``) has already verified W's processor family matches X's,
        so the raw channel-concat below produces the same paired,
        shift-aware windowing every other reachable mode already gets.
        Skips the windowed-shape validation below entirely, since raw 2-D
        arrays don't share a meaningful "window size" to compare yet.
    w_processor_type : str, optional
        W's own processor type when ``raw_deferred``, needed because W's
        type may now genuinely differ from X's (a mixed continuous +
        categorical pair). ``None`` (default) inherits X's own type.

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
          ``amplification_factor * eps`` on II.  Values >= 10 mean the sign of
          II may not be determined by the data.  See
          :func:`neural_mi.analysis.sweep.amplification_factor`.
        - ``'raw_xw_y'``, ``'raw_x_y'``, ``'raw_w_y'`` : list of result dicts.
    """
    # joint_bp/marginal_x_bp/marginal_w_bp: separate base_params for each of
    # the three sweeps below.  Identical to base_params (the marginal_x/w
    # overrides unused) except for a categorical raw_deferred X/W pair,
    # where each sweep's raw "X-role" data is a differently-shaped
    # concatenation (XW: two channel blocks; X alone / W alone: one block
    # each) and so needs its own processor_params_x['_categorical_block_specs']
    # -- try_build_shift_windows_dataset builds one encoder per call from
    # whatever's in processor_params_x, so a single shared dict can't serve
    # all three.
    joint_bp, marginal_x_bp, marginal_w_bp = base_params, None, base_params
    if raw_deferred:
        # x_data/y_data/w_data arrive here exactly as the caller passed
        # them in (not yet tensors, possibly numpy arrays) -- torch.cat
        # below needs actual tensors of a matching dtype.
        _to_t = lambda a: a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
        _x_kind = base_params.get('processor_type_x')
        _w_kind = w_processor_type if w_processor_type is not None else _x_kind
        # Y's own type may differ from X's/W's -- "None means inherit X" is
        # only the fallback when it isn't set explicitly.
        _y_kind = base_params.get('processor_type_y') or _x_kind
        y_data = list(y_data) if _y_kind == 'spike' else _to_t(y_data)

        # The shifted tuple builder takes only a regular-grid triple, and the
        # eager fallback behind it cannot window a tuple, so anything outside
        # that family still merges X and W before windowing.
        _all_regular = all(kind in _REGULAR_GRID_PROCESSOR_TYPES
                           for kind in (_x_kind, _w_kind, _y_kind))
        if _x_kind == 'spike' or not _all_regular:
            if _x_kind == 'spike':
                # A "list of per-neuron spike-time arrays" is never a tensor at
                # this stage, so this concatenation is a Python list concat.
                xw_data = list(x_data) + list(w_data)
                joint_bp = marginal_x_bp = marginal_w_bp = base_params
            else:
                xw_data, joint_bp, marginal_x_bp, marginal_w_bp = _merge_raw_blocks(
                    x_data, w_data, _x_kind, _w_kind, base_params, _to_t)
                # _merge_raw_blocks returns None for a marginal that needs no
                # block specs; interaction's third leg is estimated on its own
                # and always needs a real params dict.
                marginal_x_bp = marginal_x_bp or base_params
                marginal_w_bp = marginal_w_bp or base_params
            x_role, w_role = x_data, w_data
        else:
            # X and W stay separate raw streams, each windowed by its own
            # processor and merged only afterwards. All three legs read the same
            # pair; the pack picks out what each one needs, so no leg has to be
            # assembled early to survive a shift.
            if _x_kind != 'categorical':
                x_data = _to_t(x_data)
            if _w_kind != 'categorical':
                w_data = _to_t(w_data)
            _second = {
                '_second_processor_type': _w_kind,
                # run.py's gate requires W's window_size to match X's on this
                # path, and interaction is never handed W's params separately,
                # so X's are the right geometry for both streams.
                '_second_processor_params': base_params.get('processor_params_x'),
            }
            xw_data = x_role = w_role = (x_data, w_data)
            joint_bp = {**base_params, **_second, '_shift_pack': SHIFT_PACK_CONCAT}
            marginal_x_bp = {**base_params, **_second, '_shift_pack': SHIFT_PACK_FIRST}
            marginal_w_bp = {**base_params, **_second, '_shift_pack': SHIFT_PACK_SECOND}

    else:
        x_data = _ensure_3d(x_data)
        y_data = _ensure_3d(y_data)
        w_data = _ensure_3d(w_data)
        device = x_data.device
        y_data = y_data.to(device)
        w_data = w_data.to(device)

        if x_data.shape[0] != y_data.shape[0]:
            # X and Y are windowed together via a single shared WindowManager
            # (create_dataset), so they should always match exactly. A
            # mismatch here means something else is wrong -- always a hard
            # error (mirrors conditional.py's identical reasoning).
            raise ValueError(
                "x_data, y_data and w_data must have the same number of samples. Their "
                f"shapes are {tuple(x_data.shape)}, {tuple(y_data.shape)} and {tuple(w_data.shape)}."
            )
        if x_data.shape[0] != w_data.shape[0]:
            if abs(x_data.shape[0] - w_data.shape[0]) <= _SAMPLE_COUNT_TRIM_TOLERANCE:
                min_n = min(x_data.shape[0], w_data.shape[0])
                logger.warning(
                    f"mode='interaction': x_data and y_data have {x_data.shape[0]} windows and w_data "
                    f"has {w_data.shape[0]}. All three are truncated to the first {min_n}. This is "
                    f"correct only if the extra window is at an edge. A window missing in the "
                    f"middle pairs every later window with its neighbour. Once, at index 2730 of "
                    f"3332, that misaligned 18% of the windows. nmi.run() aligns the streams by "
                    f"window time first and never reaches this. Reaching it means raw tensors were "
                    f"passed to the engine directly without window times to align on. Pass arrays "
                    f"that already agree in length if the ordering matters."
                )
                x_data = x_data[:min_n]
                y_data = y_data[:min_n]
                w_data = w_data[:min_n]
            else:
                raise ValueError(
                    "x_data, y_data and w_data must have the same number of samples. Their "
                    f"shapes are {tuple(x_data.shape)}, {tuple(y_data.shape)} and {tuple(w_data.shape)}."
                )
        if x_data.shape[2] != w_data.shape[2]:
            if w_data.shape[2] == 1:
                # W has no temporal extent within the window (e.g. a
                # per-window categorical summary already folded into
                # channels) -- broadcast it across X's window so the two can
                # be concatenated along the channel axis.
                w_data = w_data.expand(-1, -1, x_data.shape[2])
            elif abs(x_data.shape[2] - w_data.shape[2]) <= _WINDOW_SIZE_TRIM_TOLERANCE:
                min_w = min(x_data.shape[2], w_data.shape[2])
                logger.warning(
                    f"mode='interaction': x_data window size ({x_data.shape[2]}) and w_data window size "
                    f"({w_data.shape[2]}) differ by {abs(x_data.shape[2] - w_data.shape[2])} "
                    f"sample(s). Every processor emits window_size slots for a window of "
                    f"window_size. Check that Processing(x_params=...) and Processing(w_params=...) "
                    f"agree on window_size and sample_rate. Both are trimmed to their first "
                    f"{min_w} samples."
                )
                x_data = x_data[:, :, :min_w]
                w_data = w_data[:, :, :min_w]
            else:
                raise ValueError(
                    "x_data and w_data must have the same window size to be concatenated "
                    f"into XW. Got window sizes {x_data.shape[2]} and {w_data.shape[2]} "
                    f"(full shapes {tuple(x_data.shape)}, {tuple(w_data.shape)})."
                )
        xw_data = torch.cat([x_data, w_data], dim=1)
        x_role, w_role = x_data, w_data

    _diff, mi_xw_y, mi_x_y, raw_xw_y, raw_x_y, _per_run = _joint_marginal_difference(
        xw_data, y_data, x_role, y_data,
        joint_bp, sweep_grid, n_workers,
        quantity_name="Interaction information",
        joint_label="X,W;Y", marginal_label="X;Y",
        joint_key="mi_xw_y", marginal_key="mi_x_y",
        is_proc_sweep=raw_deferred,
        marginal_base_params=marginal_x_bp,
        reported=False,
    )
    mi_w_y, raw_w_y, _w_vals = _single_mi_mean(
        w_role, y_data, with_model_labels(marginal_w_bp, component='mi_w_y'), sweep_grid, n_workers,
        quantity_name="Interaction information", label="W;Y",
        is_proc_sweep=raw_deferred,
    )

    ii = mi_xw_y - mi_x_y - mi_w_y
    _scale, _units = mi_report_units(base_params)
    logger.info(
        f"Interaction information: I(X,W;Y)={mi_xw_y * _scale:.4f}, "
        f"I(X;Y)={mi_x_y * _scale:.4f}, I(W;Y)={mi_w_y * _scale:.4f}, "
        f"II={ii * _scale:.4f} {_units}."
    )
    warn_combination("Interaction information", ii, ("X,W;Y", mi_xw_y, 'mi_xw_y'),
                     [("X;Y", mi_x_y, 'mi_x_y'), ("W;Y", mi_w_y, 'mi_w_y')],
                     base_params, signed=True)

    return {
        'interaction_info': ii,
        'mi_xw_y': mi_xw_y,
        'mi_x_y': mi_x_y,
        'mi_w_y': mi_w_y,
        'amplification_factor': amplification_factor([mi_xw_y, mi_x_y, mi_w_y], ii),
        'mi_estimate_std': combined_spread((*_per_run, _w_vals), (1, -1, -1)),
        'raw_xw_y': raw_xw_y,
        'raw_x_y': raw_x_y,
        'raw_w_y': raw_w_y,
    }


def _ii_rigorous_scalar(x_s, y_s, bp, w_data=None, sweep_grid=None, raw_deferred=False,
                        w_processor_type=None) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of
    interaction information.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1``, must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching
    ``_cmi_rigorous_scalar``'s convention. ``w_data`` arrives here already
    sliced to this gamma-chunk's samples via ``extra_data``.

    ``raw_deferred`` : forwarded straight through to
    ``run_interaction_information``, ``x_s``/``y_s``/``w_data`` are raw,
    unwindowed 2-D chunks (already translated to a raw sample range by
    ``run_rigorous_scalar_analysis``'s own ``_is_raw_deferred`` handling)
    when set, letting this gamma-chunk's own sweep dispatch reach
    ``shift_windows``.
    """
    raw = run_interaction_information(x_s, y_s, w_data, bp, sweep_grid=sweep_grid, n_workers=1,
                                      raw_deferred=raw_deferred, w_processor_type=w_processor_type)
    paths = saved_paths(raw)
    return (raw['interaction_info'], paths) if paths else raw['interaction_info']
