# neural_mi/analysis/dimensionality.py
"""Estimates the latent dimensionality of a dataset using spectral metrics.

This module forces the use of a Hybrid critic with a large bottleneck and
analyzes the cross-covariance spectrum of the resulting embeddings to
determine Intrinsic or Interaction Dimensionality.
"""
import hashlib
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from .sweep import ParameterSweep
from neural_mi.logger import logger
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu, anscombe_transform


# ---------------------------------------------------------------------------
# Noise-injection feature.
#
# Ceiling-escape via observation-space noise: `dimensionality` mode gains an
# optional `sigma_add` control that adds fixed, independent, per-channel
# Gaussian noise (in measured-per-channel-std units) to the observations once,
# so a saturated InfoNCE estimate can be de-saturated without moving the
# saturation dimension.
# ---------------------------------------------------------------------------

def _infer_modality(processor_type: Optional[str], processor_params: Optional[Dict[str, Any]]) -> str:
    """Classify data modality for the noise-injection / stabilization guards.

    Returns one of ``'continuous'``, ``'binned_spike'``, ``'raw_spike'``,
    ``'categorical'``. Unrecognised or ``None`` processor types are treated
    as ``'continuous'`` (the default, always-supported case).
    """
    if processor_type is None or processor_type == 'continuous':
        return 'continuous'
    if processor_type == 'spike':
        bin_size = (processor_params or {}).get('bin_size', None)
        return 'binned_spike' if bin_size is not None else 'raw_spike'
    if processor_type == 'categorical':
        return 'categorical'
    return 'continuous'


def _check_noise_modality(modality: str, label: str) -> None:
    """Reject modalities where sigma_add noise injection is undefined.

    Only called when ``sigma_add is not None`` — plain (no-noise) dimensionality
    analysis on these modalities is unaffected by this feature.
    """
    if modality == 'raw_spike':
        raise ValueError(
            f"sigma_add noise injection is not supported for raw spike-timestamp data "
            f"({label}): additive observation noise there perturbs timing precision "
            f"(a different axis, the substrate of 'precision' mode), not a canonical "
            f"correlation. Bin the spikes first (processor_type='{label.lower()}' "
            f"with a 'bin_size' in processor_params) to use the supported binned-count case."
        )
    if modality == 'categorical':
        raise ValueError(
            f"sigma_add noise injection is undefined for categorical data ({label}): "
            f"a label has no metric, so 'adding noise' can only mean label flipping, "
            f"which alters shared cross-covariance mass instead of merely scaling it "
            f"and breaks the rank-preservation guarantee this feature relies on."
        )


def _deterministic_seed(*parts: Any) -> int:
    """Derive a deterministic 31-bit seed from arbitrary hashable parts."""
    material = "_".join(str(p) for p in parts)
    return int(hashlib.md5(material.encode()).hexdigest(), 16) % (2 ** 31)


def _draw_base_noise(shape: Tuple[int, ...], global_seed: Any, split_id: int, view_tag: str) -> np.ndarray:
    """Deterministically reconstruct the base standard-normal tensor E for (split, view).

    ``sigma_add`` must NOT enter this seed — the same E is reused, scaled by
    every level on the ladder, so adjacent rungs differ only by scale.
    Depends only on primitive arguments (no live/shared RNG state), so every
    worker process reconstructs the identical E under parallelism.
    """
    rng = np.random.default_rng(_deterministic_seed(global_seed, split_id, view_tag, 'E'))
    return rng.standard_normal(size=shape).astype(np.float32)


def _deterministic_channel_permutation(n_channels: int, global_seed: Any, split_id: int) -> np.ndarray:
    """Deterministic per-split channel permutation, used only when sigma_add is engaged.

    (When sigma_add is unset, split_method='random' keeps its existing live-RNG
    permutation — unchanged behavior for users not using this feature.)
    """
    rng = np.random.default_rng(_deterministic_seed(global_seed, split_id, 'channel_perm'))
    return rng.permutation(n_channels)


def _per_channel_std(x: torch.Tensor) -> np.ndarray:
    """Per-channel standard deviation, pooling over samples (and window, if 3-D).

    ``x`` : shape ``(N, C)`` or ``(N, C, W)``. Returns a ``(C,)`` numpy array.
    """
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if arr.ndim == 2:
        return arr.std(axis=0)
    return arr.std(axis=(0, 2))


def _broadcast_channel_scale(scale_c: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a per-channel ``(C,)`` scale array to broadcast against ``(N, C)`` or ``(N, C, W)``."""
    if ndim == 2:
        return scale_c.reshape(1, -1)
    return scale_c.reshape(1, -1, 1)


def _resolve_absolute_scale(level: float, units: str, per_channel_std: np.ndarray) -> np.ndarray:
    """Resolve a sigma_add level to an absolute per-channel noise scale.

    ``'relative'`` (default): level is a multiple of measured per-channel std.
    ``'absolute'``: level is the actual noise std in the data's native units,
    applied uniformly to every channel.
    """
    if units == 'absolute':
        return np.full_like(per_channel_std, float(level))
    if units == 'relative':
        return level * per_channel_std
    raise ValueError(f"sigma_add_units must be 'relative' or 'absolute', got {units!r}.")


def _classify_regime(mi_value: float, ceiling: float, margin: float, floor: float) -> Tuple[str, bool]:
    """Classify an MI estimate relative to the log(eval_size) ceiling.

    Returns ``(regime, detached)`` where regime is one of
    ``'pinned'``, ``'collapsed'``, ``'detached'``.
    """
    if not np.isfinite(mi_value):
        return 'collapsed', False
    if mi_value >= ceiling - margin:
        return 'pinned', False
    if mi_value <= floor:
        return 'collapsed', False
    return 'detached', True


def _resolve_sigma_add_levels(sigma_add: Any) -> Tuple[List[float], bool]:
    """Resolve the ``sigma_add`` argument to a concrete list of levels.

    Returns ``(levels, is_auto)``.
    """
    if isinstance(sigma_add, str) and sigma_add == 'auto':
        return list(np.geomspace(0.25, 5.0, 7)), True
    if isinstance(sigma_add, (int, float)):
        return [float(sigma_add)], False
    return [float(v) for v in sigma_add], False


def _make_noise_ladder_tasks(
    x_data: torch.Tensor,
    y_data: Optional[torch.Tensor],
    analysis_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]],
    split_method: str,
    n_splits: int,
    levels: List[float],
    sigma_add_units: str,
    global_seed: Any,
) -> list:
    """Build one dispatchable split-task per (split_id, sigma_add level).

    Noise metadata (``sigma_add``, resolved absolute scales, ``sigma_add_units``)
    is baked into each task's own copy of ``analysis_params`` so it is echoed
    back as ordinary columns in the result rows — the same mechanism every
    other base_params value already uses to reach the output dataframe. No
    change to ``_run_single_split`` / ``_dispatch_splits`` is required.
    """
    tasks = []
    for split_id in range(n_splits):
        if y_data is not None:
            x_view, y_view = x_data, y_data
        else:
            n_channels = x_data.shape[1]
            if split_method == 'random':
                perm = _deterministic_channel_permutation(n_channels, global_seed, split_id)
            else:  # 'spatial' — fixed, deterministic split; no permutation needed
                perm = np.arange(n_channels)
            half = n_channels // 2
            idx_a, idx_b = perm[:half], perm[half:]
            if x_data.ndim == 2:
                x_view, y_view = x_data[:, idx_a], x_data[:, idx_b]
            else:
                x_view, y_view = x_data[:, idx_a, :], x_data[:, idx_b, :]

        std_x = _per_channel_std(x_view)
        std_y = _per_channel_std(y_view)
        E_x = _draw_base_noise(tuple(x_view.shape), global_seed, split_id, 'x')
        E_y = _draw_base_noise(tuple(y_view.shape), global_seed, split_id, 'y')

        for level in levels:
            abs_x = _resolve_absolute_scale(level, sigma_add_units, std_x)
            abs_y = _resolve_absolute_scale(level, sigma_add_units, std_y)
            x_noised = x_view + torch.as_tensor(
                _broadcast_channel_scale(abs_x, x_view.ndim) * E_x, dtype=x_view.dtype)
            y_noised = y_view + torch.as_tensor(
                _broadcast_channel_scale(abs_y, y_view.ndim) * E_y, dtype=y_view.dtype)

            task_params = analysis_params.copy()
            task_params['sigma_add'] = level
            task_params['log_sigma_add'] = float(np.log(level)) if level > 0 else float('-inf')
            task_params['sigma_add_units'] = sigma_add_units
            task_params['sigma_add_absolute_x'] = float(np.mean(abs_x))
            task_params['sigma_add_absolute_y'] = float(np.mean(abs_y))
            tasks.append((_ensure_cpu(x_noised), _ensure_cpu(y_noised), task_params, sweep_grid, split_id))
    return tasks


def _summarize_noise_ladder(all_results: list, ceiling_margin: float = 0.75,
                             ceiling_floor_frac: float = 0.05) -> pd.DataFrame:
    """Aggregate per-(split, level) rows into a per-rung ladder summary with regime labels.

    The ceiling is ``log(eval_size)`` (the InfoNCE evaluation denominator),
    read from each row's ``eval_size`` (set by ``Trainer.train`` — see
    ``training/trainer.py``), never ``log(batch_size)``.

    This aggregates ``test_mi``, not ``train_mi`` — deliberately, unlike every
    other analysis mode in the library. In dimensionality mode the MI *value*
    at each rung is not the target; the *PR read-off point* (where the
    estimate has detached from the ceiling) is. The ``log(eval_size)`` ceiling
    this regime classification is calibrated against applies to test-set
    evaluation, so comparing against ``test_mi`` is the consistent choice for
    this purpose. Consequently, ``sigma_add_ladder.mi_mean`` here and
    ``result.dataframe.mi_mean`` (which is ``train_mi``-based, per the
    library-wide convention) measure different quantities at the same rung —
    this is by design, not a bug.
    """
    df = pd.DataFrame(all_results)
    rows = []
    for level, grp in df.groupby('sigma_add'):
        mi_mean = float(grp['test_mi'].mean())
        mi_std = float(grp['test_mi'].std()) if len(grp) > 1 else 0.0
        row = {
            'sigma_add': level,
            'log_sigma_add': float(np.log(level)) if level > 0 else float('-inf'),
            'sigma_add_absolute_x_mean': float(grp['sigma_add_absolute_x'].mean()),
            'sigma_add_absolute_y_mean': float(grp['sigma_add_absolute_y'].mean()),
            'mi_mean': mi_mean,
            'mi_std': mi_std,
        }
        for metric in ('pr_eig', 'pr_singular'):
            if metric in grp.columns:
                row[f'{metric}_mean'] = float(grp[metric].mean())
                row[f'{metric}_std'] = float(grp[metric].std()) if len(grp) > 1 else 0.0
        if 'eval_size' in grp.columns and grp['eval_size'].notna().any():
            eval_size = float(grp['eval_size'].mean())
            ceiling = float(np.log(eval_size))
            floor = ceiling_floor_frac * ceiling
            regime, detached = _classify_regime(mi_mean, ceiling, ceiling_margin, floor)
        else:
            ceiling, regime, detached = None, 'unknown', False
        row['ceiling_nats'] = ceiling
        row['regime'] = regime
        row['detached'] = detached
        rows.append(row)
    return pd.DataFrame(rows).sort_values('sigma_add').reset_index(drop=True)


def _run_noise_ladder(
    x_data: torch.Tensor,
    y_data: Optional[torch.Tensor],
    analysis_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]],
    split_method: str,
    n_splits: int,
    n_workers: int,
    show_progress: bool,
    sigma_add: Any,
    sigma_add_units: str,
) -> Tuple[list, pd.DataFrame, Optional[Dict[str, Any]]]:
    """Run the sigma_add ladder (scalar, list, or 'auto') and build the output contract.

    Returns ``(all_results, ladder_summary, suggestion)`` — ``all_results`` is
    the flat list of per-(split, level) result dicts (fed through the same
    embedding-extraction / stripping pipeline as the no-noise path);
    ``ladder_summary`` is the per-rung summary table (one row per ``sigma_add``
    level, with ``mi_mean``/``mi_std``, PR mean/std, ``ceiling_nats``, and a
    ``regime`` label); ``suggestion`` is the top-level suggested operating
    level (only when ``sigma_add == 'auto'``), or ``None``.
    """
    estimator_name = str(analysis_params.get('estimator_name', 'infonce')).lower()
    if estimator_name != 'infonce':
        warnings.warn(
            f"sigma_add ceiling calibration (log(eval_size)) is derived for InfoNCE. "
            f"estimator_name='{estimator_name}' has a different, non-constant bias, so "
            f"the 'auto' bracketing and detached flags may not be meaningful here. "
            f"Proceeding anyway.",
            UserWarning, stacklevel=3,
        )

    global_seed = analysis_params.get('random_seed')
    levels, is_auto = _resolve_sigma_add_levels(sigma_add)

    def _dispatch(level_list):
        tasks = _make_noise_ladder_tasks(
            x_data, y_data, analysis_params, sweep_grid, split_method, n_splits,
            level_list, sigma_add_units, global_seed,
        )
        return _dispatch_splits(tasks, n_workers, show_progress)

    all_results = _dispatch(levels)
    ladder_summary = _summarize_noise_ladder(all_results)

    suggestion = None
    if is_auto:
        detached = ladder_summary[ladder_summary['regime'] == 'detached']
        if detached.empty:
            all_pinned = (ladder_summary['regime'] == 'pinned').all()
            all_collapsed = (ladder_summary['regime'] == 'collapsed').all()
            if all_pinned:
                widened = list(np.geomspace(1.0, 20.0, 7))
            elif all_collapsed:
                widened = list(np.geomspace(0.05, 1.0, 7))
            else:
                widened = list(np.geomspace(0.1, 10.0, 9))
            logger.info(
                f"'auto' sigma_add titration did not bracket a detached band on the "
                f"initial grid; widening once to {widened[0]:.3g}-{widened[-1]:.3g}."
            )
            all_results = all_results + _dispatch(widened)
            ladder_summary = _summarize_noise_ladder(all_results)
            detached = ladder_summary[ladder_summary['regime'] == 'detached']
            if detached.empty:
                warnings.warn(
                    "'auto' sigma_add titration did not bracket a detached regime even "
                    "after widening the search grid. Returning all rungs found; consider "
                    "specifying sigma_add as an explicit list instead of 'auto'.",
                    UserWarning, stacklevel=3,
                )
        if not detached.empty:
            lv = detached['sigma_add'].to_numpy()
            suggested_level = float(np.sqrt(lv.min() * lv.max()))  # geometric midpoint
            suggestion = {'sigma_add': suggested_level, 'regime': 'detached'}

    _warn_if_ladder_not_plateaued(ladder_summary)

    return all_results, ladder_summary, suggestion


def _warn_if_ladder_not_plateaued(ladder_summary: pd.DataFrame) -> None:
    """Checks the third dimensionality-reliability condition: no plateau
    across the noise sweep. Kept separate from the two checked in
    ``_report_dimensionality_reliability``, which don't apply here since they
    need a single-run MI/PR pair, not a multi-rung sweep. Even once MI has
    detached from the ceiling at multiple rungs, the PR readout needs to be
    *stable* across them to be trustworthy -- if it's still drifting with
    added noise, picking any single rung's dimensionality would be arbitrary.
    """
    detached_final = ladder_summary[ladder_summary['regime'] == 'detached']
    if len(detached_final) < 2 or 'pr_singular_mean' not in detached_final.columns:
        return
    pr_vals = detached_final['pr_singular_mean'].dropna()
    if len(pr_vals) < 2 or pr_vals.mean() <= 0:
        return
    cv = float(pr_vals.std() / pr_vals.mean())
    if cv > 0.2:
        warnings.warn(
            f"Dimensionality reliability: the participation ratio across "
            f"the {len(pr_vals)} detached sigma_add rung(s) has not "
            f"plateaued (coefficient of variation={cv:.1%} across "
            f"pr_singular_mean values {[round(v, 2) for v in pr_vals.tolist()]}). "
            f"The estimate is still changing with added noise rather than "
            f"settling on a stable value -- treat any single rung's "
            f"dimensionality as unreliable; consider widening or refining "
            f"the sigma_add grid.",
            UserWarning, stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Module-level picklable wrapper — must be defined at module scope so that
# multiprocessing can serialise it via its qualified name.
# ---------------------------------------------------------------------------

def _run_single_split_task(args):
    """Top-level wrapper for Pool.map — must be module-level for pickling.

    Each split is executed with ``n_workers=1`` internally to avoid nested
    multiprocessing pools.
    """
    x_a, x_b, analysis_params, sweep_grid, split_id = args
    return _run_single_split(x_a, x_b, analysis_params, sweep_grid,
                             n_workers=1, split_id=split_id)


def _dispatch_splits(split_tasks, n_workers, show_progress):
    """Execute split tasks, parallelising *across splits* when ``n_workers > 1``.

    Strategy
    --------
    * **Single split** (``len(split_tasks) == 1``): run sequentially and
      forward ``n_workers`` into the inner ``ParameterSweep`` so that any
      sweep-grid parallelism still uses the available workers.
    * **Multiple splits, ``n_workers > 1``**: dispatch splits to a
      ``Pool(n_workers)`` — each split's inner ``ParameterSweep`` gets
      ``n_workers=1`` to prevent nested pools.
    * **``n_workers <= 1``**: fully sequential.
    """
    n_tasks = len(split_tasks)

    if n_workers <= 1 or n_tasks <= 1:
        # Sequential path.
        # When there is only one split, pass n_workers through so the inner
        # ParameterSweep can use them for sweep-grid parallelism.
        inner_workers = n_workers if n_tasks == 1 else 1
        all_results = []
        for args in tqdm(split_tasks, desc="Dimensionality Splits",
                         disable=not show_progress or n_tasks == 1):
            x_a, x_b, analysis_params, sweep_grid, split_id = args
            rows = _run_single_split(x_a, x_b, analysis_params, sweep_grid,
                                     n_workers=inner_workers, split_id=split_id)
            all_results.extend(rows)
        return all_results

    # Parallel path — splits dispatched to a Pool, inner sweeps sequential.
    logger.info(f"Parallelising {n_tasks} dimensionality splits across {n_workers} workers...")
    _configure_multiprocessing()
    with mp.get_context('spawn').Pool(processes=n_workers) as pool:
        results_per_split = list(tqdm(
            pool.imap(_run_single_split_task, split_tasks),
            total=n_tasks,
            desc="Dimensionality Splits",
            disable=not show_progress,
        ))

    all_results = []
    for rows in results_per_split:
        all_results.extend(rows)
    return all_results


def run_dimensionality_analysis(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    y_data: Optional[torch.Tensor] = None,
    sweep_grid: Optional[Dict[str, Any]] = None,
    split_method: str = 'random',
    n_splits: int = 5,
    spectral_mode: str = 'summary',
    n_workers: int = 1,
    processor_type_x: Optional[str] = None,
    processor_type_y: Optional[str] = None,
    sigma_add: Any = None,
    sigma_add_units: str = 'relative',
    stabilize_counts: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Estimates dimensionality via embedding cross-covariance.

    Parameters
    ----------
    x_data : torch.Tensor
        Input data for variable X.
    base_params : Dict[str, Any]
        Dictionary of fixed parameters for the MI estimator's trainer.
        If train_indices and test_indices are present in base_params, they
        are passed through to the trainer for each split. These are indices into
        the temporal/sample dimension of x_data (not the channel dimension), so
        they remain valid after any spatial or random channel split.
    y_data : torch.Tensor, optional
        If provided, computes Interaction Dimensionality between X and Y directly.
        If None, computes Intrinsic Dimensionality by splitting x_data channels.
    split_method : {'random', 'spatial', 'temporal', 'index', 'horizontal', 'vertical', 'row_interleaved', 'col_interleaved', 'diagonal', 'antidiagonal'}, optional
        How to split x_data when y_data is None.

        - ``'random'``: randomly shuffles channels into two halves, repeated
          ``n_splits`` times so the result averages over different channel
          assignments.
        - ``'spatial'``: splits channels at the midpoint (first vs second half).
          Use when channels have a meaningful spatial ordering (e.g. electrode
          array).
        - ``'temporal'``: correlates x_data with a lag-shifted copy of itself.
          Pass ``lag=<int>`` (in samples) as a kwarg. Measures autocorrelation
          structure rather than cross-channel shared information.
        - ``'index'``: user-specified channel assignment. Pass
          ``channel_indices_x=[0, 1, 4, 5, 7]`` as a kwarg; Y is automatically
          the complement. Works for 3-D ``(N, C, W)`` and 4-D ``(N, C, H, W)``
          data. If X and Y have different channel counts, ``shared_encoder``
          is disabled with a warning. Multiple ``n_splits`` runs are still
          performed (same channel assignment, independent weight initialisations).
        - ``'horizontal'``: *4-D only.* Splits along the height axis — top half
          ``x[:, :, :H//2, :]`` → X, bottom half → Y. ``n_splits`` independent
          weight initialisations are performed with the same spatial assignment.
        - ``'vertical'``: *4-D only.* Splits along the width axis — left half
          ``x[:, :, :, :W//2]`` → X, right half → Y.
        - ``'row_interleaved'``: *4-D only.* Even-indexed rows → X, odd-indexed
          rows → Y.  Avoids contiguous spatial bias along height.
        - ``'col_interleaved'``: *4-D only.* Even-indexed columns → X,
          odd-indexed columns → Y.  Column-wise counterpart to ``'row_interleaved'``.
        - ``'diagonal'``: *4-D only; MLP/sequence models only.* True geometric
          split — upper-left triangle + main diagonal → X, lower-right triangle
          → Y (pixel mask ``row ≤ col``).  Rectangular input (H ≠ W) is allowed
          with a warning; ``shared_encoder`` is auto-disabled when halves differ.
          Raises ``ValueError`` for ``embedding_model='cnn2d'`` or ``'cnn'``.
        - ``'antidiagonal'``: *4-D only; MLP/sequence models only.* True geometric
          split — upper-right triangle + anti-diagonal → X, lower-left triangle
          → Y (pixel mask ``row + col ≤ W − 1``).  Same constraints as
          ``'diagonal'``.

        Defaults to ``'random'``.
    n_splits : int, optional
        Number of independent runs.  For intrinsic dimensionality with
        ``split_method='random'`` this controls how many distinct random
        channel-split assignments are evaluated.  For interaction
        dimensionality (``y_data`` provided) there is no channel split, so
        ``n_splits`` instead controls how many independent model fits are
        performed — each starting from a different random weight
        initialisation — giving a proper mean and standard deviation in the
        output.  Defaults to 5.
    spectral_mode : {'summary', 'full'}, optional
        Controls which spectral metrics are returned.

        - ``'summary'`` *(default)* — compute both participation-ratio
          variants, ``pr_eig`` and ``pr_singular``.
        - ``'full'`` — additionally compute ``effective_rank`` and
          ``spectral_entropy``, and include the raw singular values array in
          each result row.
    n_workers : int, optional
        Number of parallel workers.  When ``n_splits > 1`` the workers are
        distributed *across splits* (each split's inner sweep runs
        sequentially to avoid nested pools).  When ``n_splits == 1`` the
        workers are forwarded into the inner ``ParameterSweep`` to
        parallelise any sweep-grid combinations.  Defaults to 1.
    processor_type_x, processor_type_y : str, optional
        The processor type(s) originally used to build ``x_data``/``y_data``
        (e.g. ``'continuous'``, ``'spike'``, ``'categorical'``), used only to
        classify data modality for the ``sigma_add`` / ``stabilize_counts``
        guards below. Not needed for plain (no-noise) dimensionality runs.
    sigma_add : float, list of float, 'auto', or None, optional
        Ceiling-escape noise injection. When unset (default), no noise is
        added. When set, fixed, independent, per-channel
        Gaussian noise (in units of measured per-channel std) is added once to
        the observations, before the embedding, identically for train and eval:

        - A scalar runs that single noise level.
        - A list/range runs the full ladder, one result row per level.
        - ``'auto'`` searches a geometric ladder (~0.25x-5x per-channel std) to
          locate the regime where the InfoNCE estimate has detached from the
          ``log(eval_size)`` ceiling (never ``log(batch_size)``).

        Only supported for intrinsic ``split_method in ('random', 'spatial')``
        or interaction mode (``y_data`` provided); raw-spike (timestamp) and
        categorical data raise a clear error since observation-space noise is
        undefined for them. Permutation-test p-values are not computed per
        rung for the ladder -- they are omitted from the output rather than
        fabricated.
    sigma_add_units : {'relative', 'absolute'}, optional
        ``'relative'`` (default): ``sigma_add`` is a multiple of measured
        per-channel std. ``'absolute'``: ``sigma_add`` is the noise std in
        the data's native units.
    stabilize_counts : bool, optional
        For binned-spike data only: apply the canonical Anscombe
        variance-stabilizing transform before measuring per-channel std /
        injecting noise. Defaults to ``True`` and fires on every binned-spike
        dimensionality run regardless of ``sigma_add`` (documented default-on
        toggle; set ``False`` for plain, un-stabilized counts). Has no effect
        on non-binned-spike modalities.

    Returns
    -------
    pd.DataFrame
        One row per split (and per sweep combination). Columns include split_id,
        train_mi, pr_eig, pr_singular, and any additional spectral metrics.
    embeddings : dict or None
        If ``base_params`` contains ``return_embeddings=True``, a dict with
        keys ``'embeddings_x'`` and ``'embeddings_y'`` (numpy arrays, shape
        ``(n_samples, embedding_dim)``), taken from the **last** split's model.
        With ``n_splits > 1`` each split trains an independent model from a
        different random initialisation (or different channel assignment for
        ``split_method='random'``); only the last split's embeddings are
        returned, and a log message states which split was used.  If
        ``return_embeddings`` is not set, returns ``None``.

    """

    # 1. Force correct configuration for dimensionality
    analysis_params = base_params.copy()
    analysis_params['critic_type'] = 'hybrid'
    logger.info(
        "Dimensionality mode: using critic_type='hybrid' (required for spectral analysis "
        "via cross-covariance SVD)."
    )
    analysis_params['track_spectral_metrics'] = True
    if spectral_mode == 'full':
        analysis_params['spectral_output'] = 'all'
        analysis_params['return_spectrum'] = True
    else:  # 'summary' or any unrecognised value defaults to summary
        analysis_params['spectral_output'] = 'default'
        analysis_params['return_spectrum'] = False

    # Default shared_encoder=True: X and Y are always split halves of the same
    # distribution in dimensionality mode, so tying their embedding weights is
    # both theoretically justified and reduces the parameter count by half.
    # Users who want independent encoders can override via base_params.
    if 'shared_encoder' not in analysis_params:
        analysis_params['shared_encoder'] = True
        logger.info(
            "Dimensionality mode: using shared_encoder=True by default, as X and Y are "
            "treated as split views of the same data source. Set shared_encoder=False in "
            "base_params if the two data sources have structurally different representations."
        )

    if 'embedding_dim' not in analysis_params and 'embedding_dim' not in (sweep_grid or {}):
        logger.info("No embedding_dim specified. Defaulting to 64 for robust dimensionality capacity.")
        analysis_params['embedding_dim'] = 64

    # Default track_embeddings to 512 in dimensionality mode (where embeddings are
    # the primary output). Users can pass track_embeddings=False to disable, or any
    # other value to override.
    if 'track_embeddings' not in analysis_params:
        analysis_params['track_embeddings'] = 512

    # n_workers=None would crash the pool; default to 1
    if n_workers is None:
        n_workers = 1

    show_progress = analysis_params.get('show_progress', True)

    # 1b. Noise-injection modality guards + binned-spike stabilization.
    # Stabilization fires for binned-spike data regardless of sigma_add: it's a
    # default-on toggle (stabilize_counts), not conditional on noise injection
    # being engaged. Modality rejection (raw-spike / categorical) only fires
    # when sigma_add is actually engaged — plain dimensionality analysis on
    # those modalities is untouched by this feature.
    modality_x = _infer_modality(processor_type_x, analysis_params.get('processor_params_x'))
    modality_y = (_infer_modality(processor_type_y, analysis_params.get('processor_params_y'))
                  if y_data is not None else modality_x)

    if sigma_add is not None:
        _check_noise_modality(modality_x, 'X')
        if y_data is not None:
            _check_noise_modality(modality_y, 'Y')
        if y_data is None and split_method not in ('random', 'spatial'):
            raise ValueError(
                f"sigma_add is only supported for split_method in ('random', 'spatial') in "
                f"intrinsic mode, or interaction mode (y_data provided). Got "
                f"split_method='{split_method}' with y_data=None."
            )
        if sigma_add_units not in ('relative', 'absolute'):
            raise ValueError(f"sigma_add_units must be 'relative' or 'absolute', got {sigma_add_units!r}.")

    is_binned_x = modality_x == 'binned_spike'
    is_binned_y = (y_data is not None) and (modality_y == 'binned_spike')
    if is_binned_x or is_binned_y:
        if stabilize_counts:
            if is_binned_x:
                x_data = anscombe_transform(x_data)
            if is_binned_y:
                y_data = anscombe_transform(y_data)
            logger.info(
                "Dimensionality mode: binned-spike counts stabilized via the Anscombe "
                "transform (2*sqrt(x + 3/8)); this applies to this dimensionality run "
                "regardless of sigma_add. sigma_add, if used, is in units of stabilized "
                "per-channel standard deviation."
            )
        else:
            if sigma_add is not None:
                warnings.warn(
                    "stabilize_counts=False while injecting noise on binned-spike counts: "
                    "additive noise on raw counts is heteroscedastic and the per-channel-std "
                    "unit is not portable across channels, so the noise ladder may be uneven.",
                    UserWarning, stacklevel=2,
                )
            logger.info(
                "Dimensionality mode: binned-spike counts NOT stabilized "
                "(stabilize_counts=False); using plain counts."
            )

    # 1c. Noise-injection ladder dispatch (bypasses the plain split_method chain below).
    if sigma_add is not None:
        x_for_noise = _ensure_cpu(x_data)
        y_for_noise = _ensure_cpu(y_data) if y_data is not None else None
        all_results, ladder_summary, suggestion = _run_noise_ladder(
            x_for_noise, y_for_noise, analysis_params, sweep_grid, split_method, n_splits,
            n_workers, show_progress, sigma_add, sigma_add_units,
        )
        embeddings = _extract_last_split_embeddings(
            all_results, n_splits, analysis_params,
            split_method=('interaction' if y_data is not None else split_method),
        )
        embed_history = _extract_embedding_history(all_results)
        _strip_embeddings(all_results)
        df = pd.DataFrame(all_results)
        _report_dimensionality_reliability(df, analysis_params)
        logger.info("--- Dimensionality Analysis Complete (noise ladder) ---")

        embeddings = embeddings or {}
        embeddings['sigma_add_ladder'] = ladder_summary
        if suggestion is not None:
            embeddings['sigma_add_suggestion'] = suggestion
        if embed_history:
            embeddings['embedding_history_x'] = embed_history['embedding_history_x']
            embeddings['embedding_history_y'] = embed_history['embedding_history_y']
        return df, embeddings

    # 2. Interaction Dimensionality (X and Y both provided)
    if y_data is not None:
        logger.info(
            f"y_data provided. Computing Interaction Dimensionality "
            f"({n_splits} independent run{'s' if n_splits != 1 else ''})."
        )
        x_cpu, y_cpu = _ensure_cpu(x_data), _ensure_cpu(y_data)
        split_tasks = [
            (x_cpu, y_cpu, analysis_params, sweep_grid, i)
            for i in range(n_splits)
        ]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)
        embeddings = _extract_last_split_embeddings(all_results, n_splits,
                                                    analysis_params, split_method='interaction')
        embed_history = _extract_embedding_history(all_results)
        _strip_embeddings(all_results)
        df_out = pd.DataFrame(all_results)
        _report_dimensionality_reliability(df_out, analysis_params)
        if embed_history:
            embeddings = (embeddings or {})
            embeddings['embedding_history_x'] = embed_history['embedding_history_x']
            embeddings['embedding_history_y'] = embed_history['embedding_history_y']
        return df_out, embeddings

    # 3. Intrinsic Dimensionality (only X provided — channel split)
    logger.info(f"Computing Intrinsic Dimensionality using '{split_method}' splits.")

    # shape is (n_windows, n_channels, window_size) — channels are at dim 1, not dim -1
    n_channels = x_data.shape[1]

    if split_method == 'temporal':
        lag = kwargs.get('lag', None)
        if lag is None:
            raise ValueError(
                "split_method='temporal' requires a 'lag' kwarg (in samples). "
                "Example: run(..., lag=1)"
            )
        if not isinstance(lag, int) or lag < 1:
            raise ValueError(f"'lag' must be a positive integer, got {lag!r}.")
        x_a = x_data[:-lag, ...]
        x_b = x_data[lag:, ...]
        logger.info(
            f"Temporal split at lag={lag} samples: {x_a.shape[0]} aligned sample pairs."
        )
        # Only one temporal split — forward n_workers into the inner ParameterSweep
        split_tasks = [(_ensure_cpu(x_a), _ensure_cpu(x_b), analysis_params, sweep_grid, 0)]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)

    elif split_method in ('random', 'spatial'):
        if n_channels < 2:
            raise ValueError(
                f"Cannot perform '{split_method}' channel split with fewer than 2 channels. "
                f"x_data has shape {tuple(x_data.shape)}."
            )
        loops = n_splits if split_method == 'random' else 1

        # Pre-compute all (x_a, x_b) pairs before dispatching.
        split_tasks = []
        for i in range(loops):
            if split_method == 'random':
                indices = np.random.permutation(n_channels)
                half = n_channels // 2
                if x_data.ndim == 2:
                    x_a = x_data[:, indices[:half]]
                    x_b = x_data[:, indices[half:]]
                else:  # 3D (N, C, W)
                    x_a = x_data[:, indices[:half], :]
                    x_b = x_data[:, indices[half:], :]
            else:  # spatial
                half = n_channels // 2
                if x_data.ndim == 2:
                    x_a = x_data[:, :half]
                    x_b = x_data[:, half:]
                else:  # 3D (N, C, W)
                    x_a = x_data[:, :half, :]
                    x_b = x_data[:, half:, :]
            split_tasks.append((_ensure_cpu(x_a), _ensure_cpu(x_b), analysis_params, sweep_grid, i))

        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)

    elif split_method == 'index':
        channel_indices_x = kwargs.get('channel_indices_x')
        if channel_indices_x is None:
            raise ValueError(
                "split_method='index' requires a 'channel_indices_x' kwarg specifying "
                "which channel indices to assign to X. Y is the complement. "
                "Example: run(..., channel_indices_x=[0, 1, 4, 5, 7])"
            )
        channel_indices_x = list(channel_indices_x)
        if not all(isinstance(i, int) and 0 <= i < n_channels for i in channel_indices_x):
            raise ValueError(
                f"All channel_indices_x must be integers in [0, {n_channels - 1}]. "
                f"Got: {channel_indices_x}"
            )
        channel_indices_y = sorted(set(range(n_channels)) - set(channel_indices_x))
        if not channel_indices_y:
            raise ValueError(
                "channel_indices_x covers all channels; Y would be empty. "
                "Assign at least one channel to Y."
            )
        if not channel_indices_x:
            raise ValueError(
                "channel_indices_x is empty; X would be empty. "
                "Assign at least one channel to X."
            )

        # shared_encoder=True requires identical input_dim for X and Y.
        # Auto-disable with a warning when the two halves differ in size.
        _params_for_split = analysis_params
        if len(channel_indices_x) != len(channel_indices_y):
            if analysis_params.get('shared_encoder', True):
                logger.warning(
                    f"split_method='index' with unequal channel counts "
                    f"(X: {len(channel_indices_x)}, Y: {len(channel_indices_y)}) is "
                    f"incompatible with shared_encoder=True. "
                    f"Disabling shared_encoder for this run."
                )
                _params_for_split = {**analysis_params, 'shared_encoder': False}

        if x_data.ndim == 2:
            x_a = x_data[:, channel_indices_x]
            x_b = x_data[:, channel_indices_y]
        elif x_data.ndim == 3:  # (N, C, W)
            x_a = x_data[:, channel_indices_x, :]
            x_b = x_data[:, channel_indices_y, :]
        else:  # 4D (N, C, H, W)
            x_a = x_data[:, channel_indices_x, :, :]
            x_b = x_data[:, channel_indices_y, :, :]

        logger.info(
            f"Index split: X channels {channel_indices_x} ({len(channel_indices_x)} total), "
            f"Y channels {channel_indices_y} ({len(channel_indices_y)} total)."
        )
        # Index split is deterministic; a single split is sufficient.
        # n_splits independent runs (different weight initialisations) still apply.
        split_tasks = [
            (_ensure_cpu(x_a), _ensure_cpu(x_b), _params_for_split, sweep_grid, i)
            for i in range(n_splits)
        ]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)

    elif split_method in ('horizontal', 'vertical', 'row_interleaved', 'col_interleaved',
                          'diagonal', 'antidiagonal'):
        if x_data.ndim != 4:
            raise ValueError(
                f"split_method='{split_method}' requires 4-D input (N, C, H, W). "
                f"Got shape {tuple(x_data.shape)} ({x_data.ndim}-D). "
                "For 3-D or 2-D data, use split_method='random' or 'spatial' to "
                "split along the channel axis instead."
            )
        H, W = x_data.shape[2], x_data.shape[3]

        if split_method == 'horizontal':
            if H < 2:
                raise ValueError(
                    f"split_method='horizontal' requires H >= 2, got H={H}."
                )
            mid = H // 2
            x_a = x_data[:, :, :mid, :]
            x_b = x_data[:, :, mid:, :]
            logger.info(
                f"Horizontal split: top {mid} rows → X, bottom {H - mid} rows → Y "
                f"(input H={H})."
            )

        elif split_method == 'vertical':
            if W < 2:
                raise ValueError(
                    f"split_method='vertical' requires W >= 2, got W={W}."
                )
            mid = W // 2
            x_a = x_data[:, :, :, :mid]
            x_b = x_data[:, :, :, mid:]
            logger.info(
                f"Vertical split: left {mid} columns → X, right {W - mid} columns → Y "
                f"(input W={W})."
            )

        elif split_method == 'row_interleaved':
            if H < 2:
                raise ValueError(
                    f"split_method='row_interleaved' requires H >= 2, got H={H}."
                )
            x_a = x_data[:, :, 0::2, :]   # even-indexed rows
            x_b = x_data[:, :, 1::2, :]   # odd-indexed rows
            logger.info(
                f"Row-interleaved split: even rows → X ({x_a.shape[2]} rows), "
                f"odd rows → Y ({x_b.shape[2]} rows) (input H={H})."
            )

        elif split_method == 'col_interleaved':
            if W < 2:
                raise ValueError(
                    f"split_method='col_interleaved' requires W >= 2, got W={W}."
                )
            x_a = x_data[:, :, :, 0::2]   # even-indexed columns
            x_b = x_data[:, :, :, 1::2]   # odd-indexed columns
            logger.info(
                f"Col-interleaved split: even columns → X ({x_a.shape[3]} cols), "
                f"odd columns → Y ({x_b.shape[3]} cols) (input W={W})."
            )

        else:  # diagonal or antidiagonal — true geometric triangular splits
            _emb = analysis_params.get('embedding_model', 'mlp')
            if _emb in ('cnn2d', 'cnn'):
                raise ValueError(
                    f"split_method='{split_method}' produces irregularly-shaped "
                    f"triangular pixel subsets that cannot be represented as rectangular "
                    f"(N, C, H, W) tensors. embedding_model='{_emb}' requires rectangular "
                    "2-D spatial input. Use embedding_model='mlp' for geometric diagonal splits."
                )
            if H != W:
                logger.warning(
                    f"split_method='{split_method}' on non-square input (H={H}, W={W}): "
                    "the two triangular halves will have unequal pixel counts. "
                    "shared_encoder will be disabled automatically if flat dims differ."
                )
            row_idx = torch.arange(H, device=x_data.device).unsqueeze(1)  # (H, 1)
            col_idx = torch.arange(W, device=x_data.device).unsqueeze(0)  # (1, W)
            if split_method == 'diagonal':
                # upper-left triangle + main diagonal → X; lower-right triangle → Y
                mask_a = (row_idx <= col_idx).reshape(-1)
                mask_b = (row_idx > col_idx).reshape(-1)
            else:  # antidiagonal
                # upper-right triangle + anti-diagonal → X; lower-left triangle → Y
                mask_a = (row_idx + col_idx <= W - 1).reshape(-1)
                mask_b = (row_idx + col_idx > W - 1).reshape(-1)
            x_flat = x_data.reshape(x_data.shape[0], x_data.shape[1], -1)  # (N, C, H*W)
            x_a = x_flat[:, :, mask_a]   # (N, C, n_upper)
            x_b = x_flat[:, :, mask_b]   # (N, C, n_lower)
            logger.info(
                f"{'Diagonal' if split_method == 'diagonal' else 'Anti-diagonal'} split: "
                f"X gets {mask_a.sum().item()} pixels, Y gets {mask_b.sum().item()} pixels "
                f"(input H={H}, W={W})."
            )

        # shared_encoder guard: a shared encoder is built with input_dim_x, so if
        # the two halves have different flat sizes the shared network cannot process Y.
        # Auto-disable and warn. CNN2D is unaffected (AdaptiveAvgPool normalises size).
        _a_flat = int(np.prod(x_a.shape[1:]))
        _b_flat = int(np.prod(x_b.shape[1:]))
        _params_for_split = analysis_params
        if _a_flat != _b_flat and analysis_params.get('shared_encoder', True):
            logger.warning(
                f"split_method='{split_method}' produced unequal halves "
                f"(X flat dim: {_a_flat}, Y flat dim: {_b_flat}). "
                "Disabling shared_encoder for this run. "
                "Note: embedding_model='cnn2d' is unaffected (adaptive pooling "
                "normalises spatial size); this only matters for embedding_model='mlp'."
            )
            _params_for_split = {**analysis_params, 'shared_encoder': False}

        split_tasks = [
            (_ensure_cpu(x_a), _ensure_cpu(x_b), _params_for_split, sweep_grid, i)
            for i in range(n_splits)
        ]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)

    else:
        raise ValueError(
            f"Unknown split_method: '{split_method}'. "
            "Expected one of: 'random', 'spatial', 'temporal', 'index', "
            "'horizontal', 'vertical', 'row_interleaved', 'col_interleaved', "
            "'diagonal', 'antidiagonal'."
        )

    embeddings = _extract_last_split_embeddings(all_results, n_splits,
                                                analysis_params, split_method=split_method)
    embed_history = _extract_embedding_history(all_results)
    _strip_embeddings(all_results)
    df = pd.DataFrame(all_results)
    _report_dimensionality_reliability(df, analysis_params)
    logger.info("--- Dimensionality Analysis Complete ---")

    # Merge embedding history into the embeddings dict so run.py receives a single
    # optional dict (or None) — preserving the (df, embeddings) return signature.
    if embed_history:
        embeddings = (embeddings or {})
        embeddings['embedding_history_x'] = embed_history['embedding_history_x']
        embeddings['embedding_history_y'] = embed_history['embedding_history_y']

    return df, embeddings


def _strip_embeddings(results: list) -> None:
    """Remove embedding arrays from result dicts in-place.

    Embedding arrays must not end up as DataFrame columns — they are 2-D numpy
    arrays and would be stored as object-dtype cells, making the DataFrame
    unusable for aggregation.  Stripping them here is always safe; callers that
    need the embeddings collect them via ``_extract_last_split_embeddings`` and
    ``_extract_embedding_history`` before calling this.
    """
    for row in results:
        row.pop('embeddings_x', None)
        row.pop('embeddings_y', None)
        row.pop('embedding_history_x', None)
        row.pop('embedding_history_y', None)


def _extract_embedding_history(all_results: list) -> Optional[Dict[str, Any]]:
    """Return per-epoch embedding history from the last result that has it, or None.

    Called before ``_strip_embeddings`` so the lists are still present.
    Returns a dict with keys ``'embedding_history_x'`` and
    ``'embedding_history_y'`` (each a list of numpy arrays, one per epoch),
    or an empty dict when no embedding history was tracked.
    """
    for row in reversed(all_results):
        if 'embedding_history_x' in row:
            return {
                'embedding_history_x': row['embedding_history_x'],
                'embedding_history_y': row['embedding_history_y'],
            }
    return {}


def _extract_last_split_embeddings(
    all_results: list,
    n_splits: int,
    analysis_params: Dict[str, Any],
    split_method: str,
) -> Optional[Dict[str, Any]]:
    """Return embeddings from the last split's final result, or None.

    Only called when ``analysis_params.get('return_embeddings')`` is True.
    With ``n_splits > 1`` the last entry in ``all_results`` corresponds to the
    last split (highest ``split_id``); a log message informs the caller.
    """
    if not analysis_params.get('return_embeddings', False):
        return None

    # Scan from the end for the first result that has embeddings
    for row in reversed(all_results):
        if 'embeddings_x' in row and 'embeddings_y' in row:
            split_id = row.get('split_id', n_splits - 1)
            if n_splits > 1:
                logger.info(
                    f"return_embeddings=True with n_splits={n_splits}: "
                    f"returning embeddings from split {split_id} (last split). "
                    f"Each split trains an independent model; embeddings from a "
                    f"single split are sufficient for downstream alignment."
                )
            else:
                logger.debug(
                    f"return_embeddings=True ({split_method} dimensionality): "
                    f"embeddings extracted for all {row['embeddings_x'].shape[0]} samples."
                )
            return {
                'embeddings_x': row['embeddings_x'],
                'embeddings_y': row['embeddings_y'],
            }

    # return_embeddings=True was set but no embeddings found (e.g., y_data was None)
    logger.warning(
        "return_embeddings=True but no embeddings were found in the split results. "
        "Check that y_data is provided and the split produced valid results."
    )
    return None


def _report_dimensionality_reliability(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> None:
    """Report which dimensionality-reliability condition applies to this result.

    Three conditions are kept deliberately separate rather than collapsed into
    a single ``is_reliable`` flag, because each calls for a different response
    (a third, noise-sweep-specific condition is checked separately in
    ``_run_noise_ladder``, since it needs multiple ``sigma_add`` rungs):

    1. **Ceiling corruption.** The underlying scalar MI is near
       ``log(eval_size)`` (the InfoNCE evaluation ceiling). The participation
       ratio (PR) built on a saturated estimate is genuinely unreliable —
       fixable via more eval samples or ``sigma_add`` noise injection.
    2. **No spectral gap.** MI is safely below the ceiling (the measurement
       itself is trustworthy) but PR is a large fraction of the embedding's
       capacity without being ceiling-truncated. This is a real finding —
       shared structure distributed across many dimensions rather than
       concentrated in a few — and must not be reported as a failure.
    """
    if 'pr_singular' not in df.columns:
        return
    valid_prs = df['pr_singular'].dropna()
    if valid_prs.empty:
        return
    mean_pr = float(valid_prs.mean())
    embed_dim = analysis_params.get('embedding_dim', 64)

    # Condition 1: is the underlying MI estimate itself near its evaluation
    # ceiling? Checked first since it takes priority over condition 2's
    # framing below -- a saturated estimate makes the PR value untrustworthy
    # regardless of what that value is.
    near_mi_ceiling = False
    if 'test_mi' in df.columns and 'eval_size' in df.columns:
        valid = df[['test_mi', 'eval_size']].dropna()
        if not valid.empty:
            mean_test_mi = float(valid['test_mi'].mean())
            mean_eval_size = float(valid['eval_size'].mean())
            if mean_eval_size > 1:
                mi_ceiling = float(np.log(mean_eval_size))
                if mean_test_mi > 0.85 * mi_ceiling:
                    near_mi_ceiling = True
                    warnings.warn(
                        f"Dimensionality reliability: the underlying MI estimate "
                        f"({mean_test_mi:.3f} nats) is near its evaluation ceiling "
                        f"(log(eval_size)={mi_ceiling:.3f} nats). The participation "
                        f"ratio readout (pr_singular={mean_pr:.1f}) built on a "
                        f"saturated estimate is unreliable -- this reflects the "
                        f"measurement ceiling, not necessarily the true "
                        f"dimensionality. Consider raising max_eval_samples, or "
                        f"using sigma_add noise injection to de-saturate the "
                        f"estimate.",
                        UserWarning, stacklevel=3,
                    )

    # Ceiling-truncation check: PR is bounded above by embedding_dim. If the
    # estimated PR exceeds 80% of that ceiling the measurement is likely
    # truncated by the embedding's own capacity, independent of condition 1.
    threshold = 0.8 * embed_dim
    if mean_pr >= threshold:
        warnings.warn(
            f"The estimated participation ratio ({mean_pr:.1f}) is close to the "
            f"embedding dimension ceiling (embedding_dim={embed_dim}).  The true "
            f"dimensionality may be higher.  Consider increasing embedding_dim "
            f"(e.g. embedding_dim={embed_dim * 2}) in base_params to obtain an "
            f"untruncated estimate.",
            UserWarning,
            stacklevel=3,
        )
    # Condition 2: MI is trustworthy (no condition-1 warning above) and PR is
    # a substantial fraction of embedding_dim without being ceiling-truncated
    # -- report as a genuine high-dimensional finding, not a failure.
    elif not near_mi_ceiling and mean_pr >= 0.5 * embed_dim:
        logger.info(
            f"Dimensionality result: participation ratio ({mean_pr:.1f}) is a "
            f"substantial fraction of embedding_dim={embed_dim} without being "
            f"ceiling-truncated (MI is not near its evaluation ceiling). This "
            f"indicates shared structure distributed across many dimensions "
            f"rather than concentrated in a few -- a real answer, not a sign "
            f"that the measurement failed."
        )


def _run_single_split(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    analysis_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]],
    n_workers: int,
    split_id: int,
) -> list:
    """Run one channel-split and return result dicts with split_id attached."""
    sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=analysis_params)
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers,
                        is_proc_sweep=False)
    for res in results:
        res['split_id'] = split_id
    return results
