# neural_mi/analysis/conditional.py
"""Implements conditional mutual information (CMI) estimation.

CMI between X and Y given W is computed by the chain-rule difference:
    I(X; Y | W) = I(X, W; Y) - I(W; Y)

Both terms are estimated independently using ``ParameterSweep``, so the
existing MI machinery (estimators, critics, augmentation) is reused verbatim.
The conditioning variable W is concatenated with X at the data level before
any windowing or embedding.
"""
import torch
from typing import Dict, Any, Optional

from neural_mi.data.shift_windowing import (SHIFT_PACK_CONCAT, SHIFT_PACK_SECOND,
                                            _REGULAR_GRID_PROCESSOR_TYPES)
from neural_mi.analysis.sweep import (combined_spread, _joint_marginal_difference,
                                      amplification_factor)
from neural_mi.logger import logger
from neural_mi.embeddings_io import saved_paths

# Continuous windowing adds a deliberate "+1" sample as an interpolation
# safety buffer at window edges (ContinuousWindowDataset._compute_max_samples_
# per_window); categorical windowing does not, since it never interpolates.
# X and a full-resolution categorical W windowed with the same nominal
# window_size therefore differ by exactly this buffer, not by a real content
# mismatch -- trim to the shorter length instead of raising. "Full-resolution"
# is load-bearing: only encoding='full_trajectory' gives W a per-timestep
# width at all. The default 'majority_vote' and 'probability' encodings give
# one slot per category regardless of window_size, and those reach the
# broadcast path below instead of this tolerance. Kept at exactly
# 1 (not a larger tolerance) so a genuinely different window_size between X
# and W -- a real configuration error -- still raises.
_WINDOW_SIZE_TRIM_TOLERANCE = 1

# ContinuousWindowDataset and CategoricalWindowDataset validate window
# coverage with different implementations (searchsorted-based vs a
# two-pointer scan), which can disagree on a boundary window's validity by
# one sample and so produce window *counts* that differ by 1 even with
# identical window_size/step_size. This is the same class of processor-level
# edge-case discrepancy as the window-size buffer above -- not a real content
# mismatch -- and is reconciled the same way: create_dataset already
# truncates X/Y to the shorter of two window counts when aligning streams of
# different duration, so truncating X/Y/W to their shared minimum here (from
# the start, since all three begin at the same t_start) follows the same,
# already-established precedent. A larger difference still raises.
_SAMPLE_COUNT_TRIM_TOLERANCE = 1


def _merge_raw_blocks(x_data, w_data, x_kind, w_kind, base_params, to_tensor):
    """Concatenate X and W as raw channels.

    Used where the separated route cannot go. The shifted tuple builder takes
    only a regular-grid triple, so a spike side anywhere sends X and W to
    merging before windowing. One encoder then covers both blocks, so each
    categorical block has to be relabeled to its own ``0..n-1`` range first and
    declared in ``_categorical_block_specs``. Relabeling the combined array
    would infer one shared category count from the combined maximum and
    silently conflate the two.

    Returns ``(xw_data, joint_bp, marginal_x_bp, marginal_w_bp)``. Conditional
    MI uses the joint and the W-alone legs; interaction information uses all
    three. ``None`` for a marginal means the unmodified base params, the right
    choice whenever no block specs are involved.
    """
    import torch
    from neural_mi.data.temporal import relabel_categorical_data

    if x_kind == 'categorical' or w_kind == 'categorical':
        def _spec(data, kind):
            if kind == 'categorical':
                relabeled = relabel_categorical_data(data)
                n_cat = int(relabeled.max()) + 1 if relabeled.size else 1
                return torch.as_tensor(relabeled, dtype=torch.float32), (relabeled.shape[1], n_cat)
            t = to_tensor(data)
            return t, (t.shape[1], None)
        x_tensor, x_spec = _spec(x_data, x_kind)
        w_tensor, w_spec = _spec(w_data, w_kind)
        wp_x = base_params.get('processor_params_x') or {}
        joint_bp = {**base_params, 'processor_params_x': {
            **wp_x, '_categorical_block_specs': [x_spec, w_spec],
        }}
        marginal_x_bp = {**base_params, 'processor_params_x': {
            **wp_x, '_categorical_block_specs': [x_spec],
        }}
        marginal_w_bp = {**base_params, 'processor_params_x': {
            **wp_x, '_categorical_block_specs': [w_spec],
        }}
        return (torch.cat([x_tensor, w_tensor], dim=1),
                joint_bp, marginal_x_bp, marginal_w_bp)

    x_tensor, w_tensor = to_tensor(x_data), to_tensor(w_data)
    return torch.cat([x_tensor, w_tensor], dim=1), base_params, None, None


def run_conditional_mi(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    w_data: torch.Tensor,
    base_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]] = None,
    n_workers: int = 1,
    align: Optional[str] = None,
    c_data: Optional[torch.Tensor] = None,
    raw_deferred: bool = False,
    w_processor_type: Optional[str] = None,
    c_processor_type: Optional[str] = None,
    c_processor_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimates conditional mutual information I(X; Y | W).

    Uses the chain-rule identity:
        I(X; Y | W) = I(XW; Y) - I(W; Y)

    Both component MI values are estimated via ``ParameterSweep`` with the
    same ``base_params``, so all hyperparameters (estimator, critic, embedding,
    training schedule) are shared.

    Parameters
    ----------
    x_data : torch.Tensor
        Data for variable X, shape ``(n_samples, n_channels_x, window_size)``.
    y_data : torch.Tensor
        Data for variable Y, shape ``(n_samples, n_channels_y, window_size)``.
    w_data : torch.Tensor
        Conditioning variable W, shape ``(n_samples, n_channels_w, window_size)``.
        Must share the same sample dimension as x_data and y_data. A window
        axis of size 1 is broadcast across x_data's window_size before
        concatenation, for conditioning variables with no temporal extent
        within a window (e.g. a categorical W encoded with 'majority_vote' or
        'probability', see ``run._reshape_categorical_w_for_conditional``).
        Ignored (may be ``None``) when ``align='dual_branch'`` and ``c_data``
        is given instead.
    base_params : Dict[str, Any]
        Fixed parameters for the MI estimator. Passed to both sweep runs.
    sweep_grid : Dict[str, List], optional
        Optional hyperparameter grid, e.g. ``{'run_id': range(5)}``.
    n_workers : int, optional
        Number of parallel workers. Defaults to 1.
    align : {None, 'dual_branch'}, optional
        ``None`` (the default): concatenate
        X and W along the channel axis, tolerating only a 1-sample edge-case
        mismatch (see ``_WINDOW_SIZE_TRIM_TOLERANCE``) before raising.
        ``'dual_branch'``: for MI rate, instantaneous exchange, and directed
        information rate, where A and C genuinely differ in window length.
        Builds ``(x_data, c_data)`` as a tuple instead of concatenating, for
        a ``DualBranchEmbedding``-based ``custom_embedding_cls`` (set
        separately via ``Model(...)``, this function doesn't inject it) to
        process each at its own length. See ``THEORY.md``.
    c_data : torch.Tensor, optional
        The conditioning variable for the ``align='dual_branch'`` path,
        shape ``(n_samples, n_channels_c, window_size_c)``, ``window_size_c``
        may differ from ``x_data``'s window size, that's the entire point.
        Required (and ``w_data`` is ignored) when ``align='dual_branch'``.
    raw_deferred : bool, optional
        ``True`` when ``x_data``/``y_data``/``w_data`` are raw, unwindowed
        arrays and windowing should be deferred to each sweep's own
        dispatch (shift_windows/shift_time reachability), the caller
        (``run.py``) has already verified W's processor family matches X's,
        so the raw channel-concat below produces the same paired,
        shift-aware windowing every other reachable mode already gets.
        Mutually exclusive with ``align='dual_branch'``. Skips the
        windowed-shape validation/trim-tolerance logic entirely, since raw
        2-D arrays don't share a meaningful "window size" to compare yet.
    w_processor_type : str, optional
        W's own processor type when ``raw_deferred``, needed because W's
        type may now genuinely differ from X's (a mixed continuous +
        categorical conditioning pair). ``None`` (default) inherits X's own
        type, matching every other "None means inherit X" convention
        elsewhere in this library.
    c_processor_type, c_processor_params : optional
        C's own processor type/params when ``align='dual_branch'`` and
        ``raw_deferred``, C keeps its own window geometry (that's
        dual_branch's entire premise), read here instead of sharing X's.
        ``None`` (default) inherits X's own type/params.

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
          result.  Values near 1 are safe; values >= 10 mean the point
          estimate should not be read without its components.  See
          :func:`neural_mi.analysis.sweep.amplification_factor`.
        - ``'raw_xw_y'`` : list of result dicts from the XW→Y sweep.
        - ``'raw_w_y'`` : list of result dicts from the W→Y sweep.
    """
    if align == 'dual_branch':
        if c_data is None:
            raise ValueError("c_data must be provided when align='dual_branch'.")
        if raw_deferred:
            # X and C are raw, unwindowed arrays -- kept as a tuple, never
            # concatenated (dual_branch's entire premise: C keeps its own,
            # generally different, window geometry), and windowed
            # independently by
            # shift_windowing.try_build_shift_windows_dataset_dual_branch's
            # own per-side DualBranchWindowShifter (X, C, Y all shift in
            # sync, each in its own window units). y_data may itself still
            # be raw (its own processor_type_y governs that), so it isn't
            # blanket-converted the way the marginal-only float32 cast
            # below assumes.
            _to_t = lambda a: a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
            y_data = _to_t(y_data)
            xc_data = (x_data, c_data)
            joint_bp = {
                **base_params,
                '_dual_branch_c_processor_type': c_processor_type,
                '_dual_branch_c_processor_params': c_processor_params,
            }
            # Marginal leg (C alone vs Y) is a plain single-array sweep,
            # not a tuple -- routes through the ordinary (non-dual_branch)
            # try_build_shift_windows_dataset, so it needs processor_type_x/
            # processor_params_x overridden to C's own config (X's config
            # would be wrong -- C generally has a different window size).
            marginal_bp = {
                **base_params,
                'processor_type_x': c_processor_type if c_processor_type is not None
                                    else base_params.get('processor_type_x'),
                'processor_params_x': c_processor_params if c_processor_params is not None
                                      else base_params.get('processor_params_x'),
            }
            cmi, mi_xc_y, mi_c_y, results_xc_y, results_c_y, _per_run = _joint_marginal_difference(
                xc_data, y_data, c_data, y_data,
                joint_bp, sweep_grid, n_workers,
                quantity_name="Conditional MI (dual-branch)",
                joint_label="XC;Y", marginal_label="C;Y",
                joint_key="mi_xw_y", marginal_key="mi_w_y",
                is_proc_sweep=True,
                marginal_base_params=marginal_bp,
            )
            return {
                'cmi_estimate': cmi,
                'mi_xw_y': mi_xc_y,
                'mi_w_y': mi_c_y,
                'amplification_factor': amplification_factor([mi_xc_y, mi_c_y], cmi),
                'mi_estimate_std': combined_spread(_per_run, (1, -1)),
                'raw_xw_y': results_xc_y,
                'raw_w_y': results_c_y,
            }
        x_data = x_data.unsqueeze(-1) if x_data.ndim == 2 else x_data
        y_data = y_data.unsqueeze(-1) if y_data.ndim == 2 else y_data
        c_data = c_data.unsqueeze(-1) if c_data.ndim == 2 else c_data
        device = x_data.device
        y_data = y_data.to(device)
        c_data = c_data.to(device)
        if x_data.shape[0] != y_data.shape[0] or x_data.shape[0] != c_data.shape[0]:
            raise ValueError(
                "x_data, y_data and c_data must have the same number of samples. Their "
                f"shapes are {tuple(x_data.shape)}, {tuple(y_data.shape)} and {tuple(c_data.shape)}."
            )
        # No window-size trim/tolerance here -- a mismatch is expected and
        # required for this path, that's the entire reason it exists.
        xc_data = (x_data, c_data)
        cmi, mi_xc_y, mi_c_y, results_xc_y, results_c_y, _per_run = _joint_marginal_difference(
            xc_data, y_data, c_data, y_data,
            base_params, sweep_grid, n_workers,
            quantity_name="Conditional MI (dual-branch)",
            joint_label="XC;Y", marginal_label="C;Y",
            joint_key="mi_xw_y", marginal_key="mi_w_y",
        )
        return {
            'cmi_estimate': cmi,
            'mi_xw_y': mi_xc_y,
            'mi_w_y': mi_c_y,
            'amplification_factor': amplification_factor([mi_xc_y, mi_c_y], cmi),
            'mi_estimate_std': combined_spread(_per_run, (1, -1)),
            'raw_xw_y': results_xc_y,
            'raw_w_y': results_c_y,
        }

    if raw_deferred:
        # Raw channel-concat, same torch.cat(dim=1) as the windowed path
        # below, just on 2-D (T, C) arrays instead of 3-D (N, C, W) ones --
        # produces one combined raw "X-role" array that flows through the
        # ordinary paired shift-windows/shift-time mechanism, keeping X and
        # W synchronized under any shift since they're one array now.
        # x_data/y_data/w_data arrive here exactly as the caller passed them
        # in (not yet tensors, possibly numpy arrays) -- torch.cat needs
        # actual tensors of a matching dtype.
        _to_t = lambda a: a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
        _x_kind = base_params.get('processor_type_x')
        _w_kind = w_processor_type if w_processor_type is not None else _x_kind
        # Y's own type may differ from X's/W's (mode='conditional' never
        # required all three to match) -- inherit-from-X is only the
        # "None means inherit X" convention used when it isn't set
        # explicitly, matching handler.py's own convention elsewhere.
        _y_kind = base_params.get('processor_type_y') or _x_kind
        y_data = list(y_data) if _y_kind == 'spike' else _to_t(y_data)

        # The shifted tuple builder only takes a regular-grid triple, and the
        # eager fallback behind it cannot window a tuple at all. Where any side
        # falls outside that family, X and W still merge early: correct, just
        # not the separated form.
        _all_regular = all(kind in _REGULAR_GRID_PROCESSOR_TYPES
                           for kind in (_x_kind, _w_kind, _y_kind))
        if _x_kind == 'spike' or not _all_regular:
            # Spike+spike (the caller's gate only reaches here for a
            # matching family): concatenation is Python list concat, no
            # tensor op -- a "list of per-neuron spike-time arrays" is
            # never a tensor at this stage. No block-specs/type-override
            # needed: X-role and the marginal conditioning-variable-alone
            # role are both still legitimately 'spike'.
            if _x_kind == 'spike':
                xw_data = list(x_data) + list(w_data)
                joint_bp, marginal_bp = base_params, None
            else:
                xw_data, joint_bp, _, marginal_bp = _merge_raw_blocks(
                    x_data, w_data, _x_kind, _w_kind, base_params, _to_t)
            w_role = w_data
        else:
            # Regular grid. X and W stay separate raw streams, each windowed
            # and encoded by its own processor, and are merged afterwards. The
            # shifter carries both as named streams and keeps them in step, and
            # two categorical sides keep their own category counts.
            if _x_kind != 'categorical':
                x_data = _to_t(x_data)
            if _w_kind != 'categorical':
                w_data = _to_t(w_data)
            _second = {
                '_second_processor_type': _w_kind,
                # run.py hands W's own processor_params down as
                # c_processor_params; the conditioning variable is the same
                # object under both names on this path.
                '_second_processor_params': (c_processor_params
                                             or base_params.get('processor_params_x')),
            }
            xw_data = (x_data, w_data)
            w_role = xw_data           # same pair; the pack picks W out of it
            joint_bp = {**base_params, **_second, '_shift_pack': SHIFT_PACK_CONCAT}
            marginal_bp = {**base_params, **_second, '_shift_pack': SHIFT_PACK_SECOND}
        cmi, mi_xw_y, mi_w_y, results_xw_y, results_w_y, _per_run = _joint_marginal_difference(
            xw_data, y_data, w_role, y_data,
            joint_bp, sweep_grid, n_workers,
            quantity_name="Conditional MI",
            joint_label="XW;Y", marginal_label="W;Y",
            joint_key="mi_xw_y", marginal_key="mi_w_y",
            is_proc_sweep=True,
            marginal_base_params=marginal_bp,
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

    # Normalise all inputs to the same ndim before shape comparison and cat.
    # StaticDataset delivers (N, C, 1) tensors, but w_data is often passed as
    # raw 2-D (N, C) when no w_processor_type is given.  Unsqueeze the missing
    # trailing window dimension so that torch.cat works on a consistent axis.
    def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1) if t.ndim == 2 else t

    x_data = _ensure_3d(x_data)
    y_data = _ensure_3d(y_data)
    w_data = _ensure_3d(w_data)

    # Ensure all inputs are on the same device (x_data is the reference).
    # w_data in particular may arrive as a raw CPU tensor when no
    # w_processor_type is given, while x_data/y_data may be on MPS/CUDA.
    device = x_data.device
    y_data = y_data.to(device)
    w_data = w_data.to(device)

    if x_data.shape[0] != y_data.shape[0]:
        # X and Y are windowed together via a single shared WindowManager
        # (create_dataset), so they should always match exactly. A mismatch
        # here means something else is wrong -- always a hard error.
        raise ValueError(
            "x_data, y_data and w_data must have the same number of samples. Their "
            f"shapes are {tuple(x_data.shape)}, {tuple(y_data.shape)} and {tuple(w_data.shape)}."
        )
    if x_data.shape[0] != w_data.shape[0]:
        if abs(x_data.shape[0] - w_data.shape[0]) <= _SAMPLE_COUNT_TRIM_TOLERANCE:
            min_n = min(x_data.shape[0], w_data.shape[0])
            logger.warning(
                f"mode='conditional': x_data and y_data have {x_data.shape[0]} windows and w_data "
                f"has {w_data.shape[0]}. All three are truncated to the first {min_n}. This is "
                f"correct only if the extra window is at an edge. A window missing in the "
                f"middle pairs every later window with its neighbour. nmi.run() aligns the "
                f"streams by "
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
            # W has no temporal extent within the window (e.g. a per-window
            # categorical summary already folded into channels by
            # _reshape_categorical_w_for_conditional, or any other
            # window-constant conditioning variable) -- broadcast it across
            # X's window so the two can be concatenated along the channel axis.
            w_data = w_data.expand(-1, -1, x_data.shape[2])
        elif abs(x_data.shape[2] - w_data.shape[2]) <= _WINDOW_SIZE_TRIM_TOLERANCE:
            min_w = min(x_data.shape[2], w_data.shape[2])
            logger.warning(
                f"mode='conditional': x_data window size ({x_data.shape[2]}) and w_data window size "
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
                f"(full shapes {tuple(x_data.shape)}, {tuple(w_data.shape)}). "
                f"Pass align='dual_branch' (with c_data=w_data) if this mismatch is "
                f"expected -- see mi_rate/instantaneous_exchange/directed_information_rate "
                f"in neural_mi/quantities.py."
            )

    # Build XW by concatenating along the channel dimension (dim=1)
    xw_data = torch.cat([x_data, w_data], dim=1)

    cmi, mi_xw_y, mi_w_y, results_xw_y, results_w_y, _per_run = _joint_marginal_difference(
        xw_data, y_data, w_data, y_data,
        base_params, sweep_grid, n_workers,
        quantity_name="Conditional MI",
        joint_label="XW;Y", marginal_label="W;Y",
        joint_key="mi_xw_y", marginal_key="mi_w_y",
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


def _cmi_rigorous_scalar(x_s, y_s, bp, w_data=None, sweep_grid=None,
                         align=None, c_data=None, raw_deferred=False,
                         w_processor_type=None) -> float:
    """Top-level, picklable ``scalar_fn`` for rigorous bias correction of CMI.

    ``run_rigorous_scalar_analysis`` dispatches many of these (one per
    gamma-chunk) to a multiprocessing pool when ``n_workers > 1``, must be
    a module-level function (not a closure) to be picklable, and always runs
    with ``n_workers=1`` internally to avoid nested pools, matching the
    outer-loop-gets-workers / inner-loop-sequential convention used for
    dimensionality-mode splits.

    For ``align='dual_branch'``, ``c_data`` arrives here already sliced to
    this gamma-chunk's samples (via ``extra_data``, the same mechanism
    ``w_data`` already uses), ``x_s`` stays a plain tensor throughout
    ``run_rigorous_scalar_analysis``'s own chunking, the tuple is only
    assembled here, at this boundary, by ``run_conditional_mi`` itself.

    ``raw_deferred`` : forwarded straight through to ``run_conditional_mi``
   , ``x_s``/``y_s``/``w_data`` are raw, unwindowed 2-D chunks (already
    translated to a raw sample range by ``run_rigorous_scalar_analysis``'s
    own ``_is_raw_deferred`` handling) when set, letting this gamma-chunk's
    own sweep dispatch reach ``shift_windows``. ``w_processor_type`` is
    likewise forwarded straight through, for a mixed continuous+categorical
    or spike+spike X/W pair.
    """
    raw = run_conditional_mi(x_s, y_s, w_data, bp, sweep_grid=sweep_grid,
                             n_workers=1, align=align, c_data=c_data,
                             raw_deferred=raw_deferred, w_processor_type=w_processor_type)
    paths = saved_paths(raw)
    return (raw['cmi_estimate'], paths) if paths else raw['cmi_estimate']
