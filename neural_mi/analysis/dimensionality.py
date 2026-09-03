# neural_mi/analysis/dimensionality.py
"""Finds cross-seed-stable directions of shared structure between two views
(interaction dimensionality) or between split-halves of one dataset (intrinsic
dimensionality), plus a cheap separable-vs-entangled regime read.

This module deliberately does not return an exact dimensionality count. A
nonlinear encoder given more capacity than the true number of shared latent
factors doesn't just find those factors -- it can also construct combinations
of them (products, higher-order terms) that are indistinguishable from genuine
factors by any spectral measure of the trained embedding. Instead, this module
trains a modest-sized embedding a handful of times from independent random
initializations (or, for intrinsic dimensionality, independent channel splits)
and reports only the directions of shared structure that show up reliably
across every retraining, flagging any two directions too close in strength to
individually tell apart. See ``THEORY.md`` for the full argument.
"""
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from .sweep import ParameterSweep
from neural_mi.logger import logger, worker_init_args
from neural_mi.utils import mi_report_units
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu, compute_regime_diagnostic


def _classify_regime(mi_value: float, ceiling: float, margin: float, floor: float) -> Tuple[str, bool]:
    """Classify an MI estimate relative to the log(eval_size) ceiling.

    Returns ``(regime, detached)`` where regime is one of
    ``'pinned'``, ``'collapsed'``, ``'detached'``. Standalone, lightweight
    diagnostic -- not part of any remediation. An investigation into whether
    ceiling proximity corrupts this mode's stable-direction readout found it
    degrades gracefully (fewer directions found, correctly hedged with
    near-degeneracy flags) rather than misleadingly, so no noise-injection
    remedy is applied automatically; this only warns.
    """
    if not np.isfinite(mi_value):
        return 'collapsed', False
    if mi_value >= ceiling - margin:
        return 'pinned', False
    if mi_value <= floor:
        return 'collapsed', False
    return 'detached', True


def _warn_if_near_ceiling(df: pd.DataFrame, ceiling_mi_fraction: float = 0.85,
                          base_params: Optional[Dict[str, Any]] = None) -> None:
    """Lightweight, standalone warning: is the underlying MI estimate close
    enough to its evaluation ceiling (log(eval_size)) that any reading built
    on it deserves extra caution? Not a remediation -- see _classify_regime.
    """
    if 'test_mi' not in df.columns or 'eval_size' not in df.columns:
        return
    valid = df[['test_mi', 'eval_size']].dropna()
    if valid.empty:
        return
    mean_test_mi = float(valid['test_mi'].mean())
    mean_eval_size = float(valid['eval_size'].mean())
    if mean_eval_size <= 1:
        return
    ceiling = float(np.log(mean_eval_size))
    regime, _ = _classify_regime(mean_test_mi, ceiling, (1 - ceiling_mi_fraction) * ceiling, 0.0)
    if regime == 'pinned':
        _scale, _units = mi_report_units(base_params)
        warnings.warn(
            f"Dimensionality: the underlying MI estimate ({mean_test_mi * _scale:.3f} {_units}) is "
            f"near its evaluation ceiling (log(eval_size)={ceiling * _scale:.3f} {_units}). Stable "
            f"directions found under this condition are still trustworthy (this was "
            f"tested directly -- ceiling proximity was found to degrade the "
            f"stable-direction count conservatively, not misleadingly), but the count "
            f"may be an undercount of what a larger evaluation batch (max_eval_samples) "
            f"could resolve. Consider raising max_eval_samples if you need a fuller read.",
            UserWarning, stacklevel=3,
        )


def _safe_regime_diagnostic(x: torch.Tensor) -> Optional[Dict[str, Any]]:
    """compute_regime_diagnostic, skipped gracefully (returns None) when not
    applicable -- 4-D spatial (N, C, H, W) input with few channels (e.g.
    single-channel image/video data for a CNN2D encoder) has no meaningful
    within-view channel correlation to compute; this is a real data shape
    this mode supports (via split_method='horizontal'/'diagonal'/etc.), not
    an error case. Also skipped -- via the same ValueError catch, since
    numpy.linalg.LinAlgError subclasses it -- when the correlation matrix is
    singular/NaN-poisoned, e.g. a real recording with a silent (zero-variance)
    channel over the analysis window; confirmed directly on real hippocampal
    data (one silent unit in a 10k-timepoint slice).
    """
    if x.ndim > 3 or x.shape[1] < 2:
        return None
    try:
        return compute_regime_diagnostic(x)
    except ValueError:
        return None


def _n_samples_for_shared_split(x_data, y_data, analysis_params: Dict[str, Any]) -> int:
    """Number of units the shared train/test split should be computed over.

    For already-windowed data (the common case), this is just
    ``x_data.shape[0]``. When windowing has been deferred (``x_data`` is
    raw, 2-D, and ``shift_windows`` was requested for a regular-grid pair),
    ``x_data.shape[0]`` is a raw sample count, not a window count -- reusing
    it directly would compute indices in the wrong space entirely. Delegates
    to :func:`~neural_mi.data.shift_windowing.n_windows_if_deferred` for the
    shift-invariant window count in that case (intrinsic mode passes
    ``y_data=None`` since every split pairs two channel-groups of the same
    ``x_data``, sharing one ``window_size``/``step_size``; interaction mode
    passes the real ``y_data``, which may use different processor params).
    """
    from neural_mi.data.shift_windowing import n_windows_if_deferred
    return n_windows_if_deferred(x_data, y_data, analysis_params)


def _get_or_create_shared_split(analysis_params: Dict[str, Any], n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) shared across every split/rerun in
    this dimensionality run, so the cross-run stability check always compares
    directions on the exact same held-out samples.

    If the caller already supplied explicit train_indices/test_indices, those
    are used unchanged. Otherwise computes ONE split, ONCE, reusing Trainer's
    own splitting logic directly (never reimplemented here, so this can't
    drift from how the rest of the library splits data). Without this, each
    split/rerun would get a genuinely DIFFERENT random train/test partition --
    task-level seeding varies deliberately by run_id so that independent
    reruns really are independent -- which would make "held-out" mean
    different samples every time and break the cross-run comparison entirely.
    """
    existing_train = analysis_params.get('train_indices')
    if existing_train is not None:
        return existing_train, analysis_params.get('test_indices')

    from neural_mi.training.trainer import Trainer
    split_mode = analysis_params.get('split_mode', 'blocked')
    train_fraction = analysis_params.get('train_fraction', 0.9)
    if split_mode == 'random':
        train_idx, test_idx = Trainer._create_random_split(None, n_samples, train_fraction)
    else:
        n_test_blocks = analysis_params.get('n_test_blocks', 5)
        gap_fraction = analysis_params.get('split_gap_fraction', 0.5)
        train_idx, test_idx = Trainer._create_blocked_split(
            None, n_samples, train_fraction, n_test_blocks, gap_fraction)
    return train_idx, test_idx


def _compute_stability_report(
    per_split_rotated: List[Dict[str, Any]],
    stability_threshold: float,
    degeneracy_ratio_threshold: float,
    min_strength_fraction: float,
) -> Dict[str, Any]:
    """Cross-run stability + near-degeneracy report, computed on genuinely
    held-out rotated embeddings from every split/rerun.

    A rank is "stable" if the minimum pairwise correlation of its direction
    across every pair of splits clears ``stability_threshold``. A rank is
    "below noise floor" if its mean singular-value strength across splits is
    under ``min_strength_fraction`` of the top rank's strength -- this check
    is independent of and catches cases the correlation check alone misses: a
    pure noise direction can show a spuriously high cross-run correlation by
    chance despite carrying no real signal (confirmed directly in testing --
    see SOURCE_OF_TRUTH.md's hard_entangled_troublezone battery result).
    Adjacent ranks within ``degeneracy_ratio_threshold`` of each other in
    strength are reported as a group, existence confirmed but individual
    order/identity not claimed.

    Parameters
    ----------
    per_split_rotated : list of dict
        One entry per split, each with keys ``'zx_rotated_test'`` (ndarray,
        shape (n_test, embedding_dim)) and ``'singular_values'`` (ndarray or
        None).
    """
    n_splits = len(per_split_rotated)
    embedding_dim = per_split_rotated[0]['zx_rotated_test'].shape[1]

    rank_corrs = {r: [] for r in range(embedding_dim)}
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            zi = per_split_rotated[i]['zx_rotated_test']
            zj = per_split_rotated[j]['zx_rotated_test']
            for r in range(embedding_dim):
                c = np.corrcoef(zi[:, r], zj[:, r])[0, 1]
                rank_corrs[r].append(float(c) if np.isfinite(c) else 0.0)

    sv_lists = [row['singular_values'] for row in per_split_rotated if row['singular_values'] is not None]
    mean_strength = np.mean(np.array(sv_lists), axis=0) if sv_lists else np.full(embedding_dim, np.nan)
    strength_floor = min_strength_fraction * mean_strength[0] if len(mean_strength) > 0 else 0.0

    per_rank = {}
    for r in range(embedding_dim):
        abs_corrs = [abs(c) for c in rank_corrs[r]]
        min_abs = min(abs_corrs) if abs_corrs else float('nan')
        strength = float(mean_strength[r]) if r < len(mean_strength) else None
        below_floor = (strength is not None) and (strength < strength_floor)
        per_rank[r + 1] = {
            'pairwise_abs_corr': abs_corrs, 'min_abs_corr': min_abs,
            'mean_strength': strength, 'below_noise_floor': bool(below_floor),
            'stable': bool(min_abs >= stability_threshold) if abs_corrs else False,
            'near_degenerate_with_next': False, 'near_degenerate_with_prev': False,
        }
    for r in range(embedding_dim - 1):
        s1, s2 = mean_strength[r], mean_strength[r + 1]
        if s2 > 1e-8 and (s1 / s2) < degeneracy_ratio_threshold:
            per_rank[r + 1]['near_degenerate_with_next'] = True
            per_rank[r + 2]['near_degenerate_with_prev'] = True

    def _trustworthy(info):
        return info['stable'] and not info['below_noise_floor']

    trustworthy_individual_ranks = [
        r for r, info in per_rank.items()
        if _trustworthy(info) and not info['near_degenerate_with_next'] and not info['near_degenerate_with_prev']
    ]
    stable_but_degenerate_groups = _group_adjacent(sorted(
        r for r, info in per_rank.items()
        if _trustworthy(info) and (info['near_degenerate_with_next'] or info['near_degenerate_with_prev'])
    ))

    return {
        'per_rank': per_rank,
        'stable_directions': sorted(trustworthy_individual_ranks),
        'stable_but_degenerate_groups': stable_but_degenerate_groups,
        'n_stable_total': len(trustworthy_individual_ranks) + sum(len(g) for g in stable_but_degenerate_groups),
        'n_splits': n_splits,
    }


def _group_adjacent(ranks: List[int]) -> List[List[int]]:
    """Group a sorted list of rank indices into contiguous runs, e.g.
    [1, 2, 5] -> [[1, 2], [5]]."""
    if not ranks:
        return []
    groups, current = [], [ranks[0]]
    for r in ranks[1:]:
        if r == current[-1] + 1:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    return groups


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
    _log_init, _log_args = worker_init_args()
    with mp.get_context('spawn').Pool(processes=n_workers,
                                      initializer=_log_init, initargs=_log_args) as pool:
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
    n_splits: int = 3,
    n_workers: int = 1,
    processor_type_x: Optional[str] = None,
    processor_type_y: Optional[str] = None,
    user_set_keys: Optional[set] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Finds cross-seed-stable directions of shared structure, plus a regime read.

    Parameters
    ----------
    x_data : torch.Tensor
        Input data for variable X.
    base_params : Dict[str, Any]
        Dictionary of fixed parameters for the MI estimator's trainer.
        If train_indices and test_indices are present in base_params, they
        are used for every split/rerun (and honored as-is); otherwise a
        single train/test partition is computed once and shared across every
        split/rerun, so the cross-run stability check always compares
        directions on the same held-out samples.
    user_set_keys : set, optional
        Keys the caller explicitly set in base_params *before*
        ParameterValidator.apply_defaults() filled in the rest. By the time
        base_params reaches this function every BASE_PARAMS_SCHEMA key is
        already present (either the caller's value or the library-wide
        default), so this is the only way to tell "the caller wants
        embedding_dim=64" apart from "embedding_dim defaulted to 64" -- needed
        because this mode's own defaults (a modest embedding_dim; a
        sub-mode-conditional shared_encoder) differ from the library-wide
        ones. If not provided (e.g. calling this function directly rather than
        through ``run()``), defaults to every key already in ``base_params`` --
        i.e. treats a direct ``base_params={'embedding_dim': 64}`` as
        explicitly set, matching direct-call semantics.
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
          structure rather than cross-channel shared information. Only one
          split is performed (no cross-run stability check is meaningful here).
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
        initialisation. Both interpretations feed the same cross-run
        stability check; intrinsic mode's version additionally varies the
        channel split itself, a related but distinct notion of "stability"
        from interaction mode's fixed-data/varying-seed version. Defaults to 3
        (the minimum for a robust cross-run comparison; 2 gives a single pair).
    n_workers : int, optional
        Number of parallel workers.  When ``n_splits > 1`` the workers are
        distributed *across splits* (each split's inner sweep runs
        sequentially to avoid nested pools).  When ``n_splits == 1`` the
        workers are forwarded into the inner ``ParameterSweep`` to
        parallelise any sweep-grid combinations.  Defaults to 1.
    processor_type_x, processor_type_y : str, optional
        The processor type(s) originally used to build ``x_data``/``y_data``
        (e.g. ``'continuous'``, ``'spike'``, ``'categorical'``). Currently
        unused by this mode; accepted for API symmetry with other modes.

    Returns
    -------
    pd.DataFrame
        One row per split (and per sweep combination). Columns include
        split_id, train_mi, test_mi, best_epoch, pr_eig, pr_singular (kept as
        a labeled secondary diagnostic, not the mode's answer), and any
        additional spectral metrics.
    embeddings : dict or None
        Always includes ``'regime_x'`` (and ``'regime_y'`` if ``y_data`` is
        provided), ``'stable_directions'``, ``'stable_but_degenerate_groups'``,
        ``'n_stable_total'``, ``'converged'``. If ``base_params`` contains
        ``return_embeddings=True``, also includes ``'embeddings_x'`` and
        ``'embeddings_y'`` (numpy arrays, shape ``(n_samples, embedding_dim)``)
        from the **last** split's model, matching the pre-existing behavior for
        callers that want the full per-sample embeddings, not just which
        directions are trustworthy.
    """

    # 1. Force correct configuration for dimensionality
    analysis_params = base_params.copy()
    # When called directly (not via run(), which passes the real pre-defaults key
    # set) there's no separate defaulting step to distinguish from -- treat
    # everything already in base_params as "user set", matching direct-call
    # semantics before this parameter existed.
    user_set_keys = user_set_keys if user_set_keys is not None else set(base_params.keys())
    analysis_params['critic_type'] = 'hybrid'
    logger.info(
        "Dimensionality mode: using critic_type='hybrid' (required for spectral analysis "
        "via cross-covariance SVD)."
    )

    # shared_encoder default is conditional on sub-mode: True only makes sense
    # when X and Y are split halves of the same data source (intrinsic mode,
    # y_data is None). For interaction mode (two different populations/views)
    # tying their embedding weights isn't justified by anything about the
    # setup, so it defaults to the library-wide default (False) there.
    if 'shared_encoder' not in user_set_keys:
        if y_data is None:
            analysis_params['shared_encoder'] = True
            logger.info(
                "Dimensionality mode (intrinsic): using shared_encoder=True by default, as "
                "X and Y are split views of the same data source. Set shared_encoder=False "
                "in base_params if the two halves have structurally different representations."
            )
        else:
            analysis_params['shared_encoder'] = False
            logger.info(
                "Dimensionality mode (interaction): shared_encoder defaults to False, as X "
                "and Y are two different views/populations with no reason to assume identical "
                "structure. Set shared_encoder=True in base_params to tie their weights."
            )

    if 'embedding_dim' not in user_set_keys and 'embedding_dim' not in (sweep_grid or {}):
        logger.info(
            "No embedding_dim specified. Defaulting to 8: this mode does not want an "
            "oversized embedding -- over-provisioning is exactly what lets artifact "
            "directions (products/combinations of true factors, indistinguishable from "
            "genuine ones by any spectral measure) masquerade as real ones. Raise this if "
            "you have a specific reason to expect higher true dimensionality."
        )
        analysis_params['embedding_dim'] = 8

    # Always compute rotated embeddings internally -- required for the
    # cross-run stability check. This does NOT expose full per-sample
    # embeddings to the caller by itself; that remains opt-in via
    # return_embeddings (tracked below before being overridden).
    user_wants_full_embeddings = bool(analysis_params.get('return_embeddings', False))
    analysis_params['return_embeddings'] = True
    analysis_params['return_rotated_embeddings'] = True
    analysis_params.setdefault('rotated_embeddings_whitening', 'std')

    # n_workers=None would crash the pool; default to 1
    if n_workers is None:
        n_workers = 1

    show_progress = analysis_params.get('show_progress', True)

    # Regime diagnostic: cheap, no training, run once before anything else.
    # Informational context only -- not wired into any other computation here.
    # Not applicable to 4-D spatial (N, C, H, W) data with few channels (e.g.
    # single-channel image/video input for a CNN2D encoder) -- there isn't a
    # meaningful "within-view channel correlation" for that data shape, so
    # this is skipped gracefully rather than forced to fit or made to error.
    regimes = {}
    regime_x = _safe_regime_diagnostic(x_data)
    if regime_x is not None:
        regimes['regime_x'] = regime_x
    if y_data is not None:
        regime_y = _safe_regime_diagnostic(y_data)
        if regime_y is not None:
            regimes['regime_y'] = regime_y

    # Fix a single shared train/test partition (unless the caller already
    # supplied one) so every split/rerun evaluates stability on the same
    # held-out samples. Skipped for split_method='temporal', which trims the
    # sample dimension by `lag` and only ever performs a single split anyway
    # (no cross-run comparison is meaningful there).
    if y_data is not None or split_method != 'temporal':
        n_samples = _n_samples_for_shared_split(x_data, y_data, analysis_params)
        train_idx, test_idx = _get_or_create_shared_split(analysis_params, n_samples)
        analysis_params['train_indices'] = train_idx
        analysis_params['test_indices'] = test_idx

    # These flow through run_dimensionality_analysis's own **kwargs (like `lag`
    # and `channel_indices_x` below), NOT through base_params/analysis_params --
    # Dimensionality.to_analysis_kwargs() lowers them into analysis_kwargs, which
    # run.py passes as **kwargs here, not merged into base_params.
    stability_threshold = kwargs.get('stability_threshold', 0.7)
    degeneracy_ratio_threshold = kwargs.get('degeneracy_ratio_threshold', 1.3)
    min_strength_fraction = kwargs.get('min_strength_fraction', 0.05)
    ceiling_mi_fraction = kwargs.get('ceiling_mi_fraction', 0.85)

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
        test_idx_per_split = [analysis_params['test_indices']] * n_splits

    # 3. Intrinsic Dimensionality (only X provided — channel split)
    else:
        logger.info(f"Computing Intrinsic Dimensionality using '{split_method}' splits.")
        all_results, test_idx_per_split = _dispatch_intrinsic_splits(
            x_data, analysis_params, sweep_grid, split_method, n_splits, n_workers,
            show_progress, kwargs,
        )

    embeddings = _extract_last_split_embeddings(all_results, n_splits, analysis_params,
                                                split_method=('interaction' if y_data is not None else split_method)) \
        if user_wants_full_embeddings else None

    stability_input = _extract_per_split_rotated(all_results, test_idx_per_split)
    stability = None
    if len(stability_input) >= 2:
        stability = _compute_stability_report(
            stability_input, stability_threshold, degeneracy_ratio_threshold, min_strength_fraction)
    else:
        logger.warning(
            "Dimensionality: fewer than 2 splits produced usable rotated embeddings -- "
            "cross-run stability cannot be computed (need at least 2 to compare). No "
            "stable directions can be reported."
        )

    converged_flags = [row.get('best_epoch') is not None and row.get('n_epochs') is not None
                       and row['best_epoch'] < row['n_epochs'] - 1 for row in all_results]
    all_converged = bool(converged_flags) and all(converged_flags)
    if converged_flags and not all_converged:
        n_unconverged = sum(1 for c in converged_flags if not c)
        warnings.warn(
            f"Dimensionality: {n_unconverged} of {len(converged_flags)} split(s) did not "
            f"converge (early stopping never triggered within the epoch budget). The "
            f"stable-direction count below may be an undercount -- increase n_epochs or "
            f"lower patience to let training finish before trusting this reading fully.",
            UserWarning, stacklevel=2,
        )

    embed_history = _extract_embedding_history(all_results)
    _strip_embeddings(all_results)
    df = pd.DataFrame(all_results)
    _warn_if_near_ceiling(df, ceiling_mi_fraction, base_params)
    logger.info("--- Dimensionality Analysis Complete ---")

    embeddings = embeddings or {}
    if embed_history:
        embeddings['embedding_history_x'] = embed_history['embedding_history_x']
        embeddings['embedding_history_y'] = embed_history['embedding_history_y']
    embeddings.update(regimes)
    embeddings['converged'] = all_converged
    if stability is not None:
        embeddings['stable_directions'] = stability['stable_directions']
        embeddings['stable_but_degenerate_groups'] = stability['stable_but_degenerate_groups']
        embeddings['n_stable_total'] = stability['n_stable_total']
        embeddings['stability_per_rank'] = stability['per_rank']
    else:
        embeddings['stable_directions'] = []
        embeddings['stable_but_degenerate_groups'] = []
        embeddings['n_stable_total'] = 0

    return df, embeddings


def _dispatch_intrinsic_splits(
    x_data: torch.Tensor,
    analysis_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]],
    split_method: str,
    n_splits: int,
    n_workers: int,
    show_progress: bool,
    kwargs: Dict[str, Any],
) -> Tuple[list, List[np.ndarray]]:
    """Builds and dispatches split tasks for intrinsic (channel-split)
    dimensionality. Returns (all_results, test_idx_per_split) where the
    latter has one entry per split (all identical unless split_method is
    'temporal', which is not tracked for stability -- see caller).
    """
    n_channels = x_data.shape[1]
    test_idx = analysis_params.get('test_indices')

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
        logger.info(f"Temporal split at lag={lag} samples: {x_a.shape[0]} aligned sample pairs.")
        split_tasks = [(_ensure_cpu(x_a), _ensure_cpu(x_b), analysis_params, sweep_grid, 0)]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)
        return all_results, []  # no shared held-out set; not used for stability

    elif split_method in ('random', 'spatial'):
        if n_channels < 2:
            raise ValueError(
                f"Cannot perform '{split_method}' channel split with fewer than 2 channels. "
                f"x_data has shape {tuple(x_data.shape)}."
            )
        loops = n_splits if split_method == 'random' else 1
        half = n_channels // 2
        # An odd n_channels gives unequal halves (half vs n_channels - half),
        # incompatible with shared_encoder=True (a single encoder sized for
        # one half's input_dim can't process the other). Random/spatial never
        # needed this guard before shared_encoder correctly defaulted to True
        # for intrinsic mode (see the conditional default above) -- 'index'
        # and the geometric splits already have the equivalent guard.
        _params_for_split = analysis_params
        if half != n_channels - half and analysis_params.get('shared_encoder', False):
            logger.warning(
                f"split_method='{split_method}' on an odd channel count ({n_channels}) "
                f"produces unequal halves (X: {half}, Y: {n_channels - half}), incompatible "
                f"with shared_encoder=True. Disabling shared_encoder for this run."
            )
            _params_for_split = {**analysis_params, 'shared_encoder': False}
        split_tasks = []
        for i in range(loops):
            if split_method == 'random':
                indices = np.random.permutation(n_channels)
                if x_data.ndim == 2:
                    x_a, x_b = x_data[:, indices[:half]], x_data[:, indices[half:]]
                else:  # 3D (N, C, W)
                    x_a, x_b = x_data[:, indices[:half], :], x_data[:, indices[half:], :]
            else:  # spatial
                if x_data.ndim == 2:
                    x_a, x_b = x_data[:, :half], x_data[:, half:]
                else:
                    x_a, x_b = x_data[:, :half, :], x_data[:, half:, :]
            split_tasks.append((_ensure_cpu(x_a), _ensure_cpu(x_b), _params_for_split, sweep_grid, i))
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)
        return all_results, [test_idx] * loops

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
            raise ValueError("channel_indices_x covers all channels; Y would be empty.")
        if not channel_indices_x:
            raise ValueError("channel_indices_x is empty; X would be empty.")

        _params_for_split = analysis_params
        if len(channel_indices_x) != len(channel_indices_y):
            if analysis_params.get('shared_encoder', False):
                logger.warning(
                    f"split_method='index' with unequal channel counts "
                    f"(X: {len(channel_indices_x)}, Y: {len(channel_indices_y)}) is "
                    f"incompatible with shared_encoder=True. Disabling shared_encoder "
                    f"for this run."
                )
                _params_for_split = {**analysis_params, 'shared_encoder': False}

        if x_data.ndim == 2:
            x_a, x_b = x_data[:, channel_indices_x], x_data[:, channel_indices_y]
        elif x_data.ndim == 3:
            x_a, x_b = x_data[:, channel_indices_x, :], x_data[:, channel_indices_y, :]
        else:
            x_a, x_b = x_data[:, channel_indices_x, :, :], x_data[:, channel_indices_y, :, :]

        logger.info(
            f"Index split: X channels {channel_indices_x} ({len(channel_indices_x)} total), "
            f"Y channels {channel_indices_y} ({len(channel_indices_y)} total)."
        )
        split_tasks = [
            (_ensure_cpu(x_a), _ensure_cpu(x_b), _params_for_split, sweep_grid, i)
            for i in range(n_splits)
        ]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)
        return all_results, [test_idx] * n_splits

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
                raise ValueError(f"split_method='horizontal' requires H >= 2, got H={H}.")
            mid = H // 2
            x_a, x_b = x_data[:, :, :mid, :], x_data[:, :, mid:, :]
            logger.info(f"Horizontal split: top {mid} rows -> X, bottom {H - mid} rows -> Y (H={H}).")

        elif split_method == 'vertical':
            if W < 2:
                raise ValueError(f"split_method='vertical' requires W >= 2, got W={W}.")
            mid = W // 2
            x_a, x_b = x_data[:, :, :, :mid], x_data[:, :, :, mid:]
            logger.info(f"Vertical split: left {mid} cols -> X, right {W - mid} cols -> Y (W={W}).")

        elif split_method == 'row_interleaved':
            if H < 2:
                raise ValueError(f"split_method='row_interleaved' requires H >= 2, got H={H}.")
            x_a, x_b = x_data[:, :, 0::2, :], x_data[:, :, 1::2, :]
            logger.info(f"Row-interleaved split: even rows -> X ({x_a.shape[2]}), "
                       f"odd rows -> Y ({x_b.shape[2]}) (H={H}).")

        elif split_method == 'col_interleaved':
            if W < 2:
                raise ValueError(f"split_method='col_interleaved' requires W >= 2, got W={W}.")
            x_a, x_b = x_data[:, :, :, 0::2], x_data[:, :, :, 1::2]
            logger.info(f"Col-interleaved split: even cols -> X ({x_a.shape[3]}), "
                       f"odd cols -> Y ({x_b.shape[3]}) (W={W}).")

        else:  # diagonal or antidiagonal — true geometric triangular splits
            _emb = analysis_params.get('embedding_model', 'mlp')
            if _emb in ('cnn2d', 'cnn'):
                raise ValueError(
                    f"split_method='{split_method}' produces irregularly-shaped triangular "
                    f"pixel subsets that cannot be represented as rectangular (N, C, H, W) "
                    f"tensors. embedding_model='{_emb}' requires rectangular 2-D spatial "
                    "input. Use embedding_model='mlp' for geometric diagonal splits."
                )
            if H != W:
                logger.warning(
                    f"split_method='{split_method}' on non-square input (H={H}, W={W}): "
                    "the two triangular halves will have unequal pixel counts. "
                    "shared_encoder will be disabled automatically if flat dims differ."
                )
            row_idx = torch.arange(H, device=x_data.device).unsqueeze(1)
            col_idx = torch.arange(W, device=x_data.device).unsqueeze(0)
            if split_method == 'diagonal':
                mask_a = (row_idx <= col_idx).reshape(-1)
                mask_b = (row_idx > col_idx).reshape(-1)
            else:  # antidiagonal
                mask_a = (row_idx + col_idx <= W - 1).reshape(-1)
                mask_b = (row_idx + col_idx > W - 1).reshape(-1)
            x_flat = x_data.reshape(x_data.shape[0], x_data.shape[1], -1)
            x_a, x_b = x_flat[:, :, mask_a], x_flat[:, :, mask_b]
            logger.info(
                f"{'Diagonal' if split_method == 'diagonal' else 'Anti-diagonal'} split: "
                f"X gets {mask_a.sum().item()} pixels, Y gets {mask_b.sum().item()} pixels (H={H}, W={W})."
            )

        _a_flat = int(np.prod(x_a.shape[1:]))
        _b_flat = int(np.prod(x_b.shape[1:]))
        _params_for_split = analysis_params
        if _a_flat != _b_flat and analysis_params.get('shared_encoder', False):
            logger.warning(
                f"split_method='{split_method}' produced unequal halves "
                f"(X flat dim: {_a_flat}, Y flat dim: {_b_flat}). Disabling shared_encoder "
                f"for this run. Note: embedding_model='cnn2d' is unaffected (adaptive "
                f"pooling normalises spatial size); this only matters for embedding_model='mlp'."
            )
            _params_for_split = {**analysis_params, 'shared_encoder': False}

        split_tasks = [
            (_ensure_cpu(x_a), _ensure_cpu(x_b), _params_for_split, sweep_grid, i)
            for i in range(n_splits)
        ]
        all_results = _dispatch_splits(split_tasks, n_workers, show_progress)
        return all_results, [test_idx] * n_splits

    else:
        raise ValueError(
            f"Unknown split_method: '{split_method}'. "
            "Expected one of: 'random', 'spatial', 'temporal', 'index', "
            "'horizontal', 'vertical', 'row_interleaved', 'col_interleaved', "
            "'diagonal', 'antidiagonal'."
        )


def _extract_per_split_rotated(all_results: list, test_idx_per_split: List[Optional[np.ndarray]]) -> list:
    """Pull each split's rotated embeddings + singular values, sliced to that
    split's held-out (test) indices only -- never train-exposed data.
    Returns a list of dicts with keys 'zx_rotated_test', 'singular_values'.
    """
    out = []
    for row, test_idx in zip(all_results, test_idx_per_split):
        if test_idx is None or 'embeddings_x_rotated' not in row:
            continue
        zx_full = row['embeddings_x_rotated']
        sv = row.get('embeddings_rotation_singular_values')
        test_idx = np.asarray(test_idx)
        if len(test_idx) == 0 or zx_full is None:
            continue
        out.append({
            'zx_rotated_test': np.asarray(zx_full)[test_idx],
            'singular_values': np.asarray(sv) if sv is not None else None,
        })
    return out


def _extract_embedding_history(all_results: list) -> Optional[Dict[str, Any]]:
    """Return per-epoch embedding history from the last result that has it, or
    an empty dict. Only populated when the caller explicitly set
    track_embeddings (this mode no longer forces it on by default -- the
    rotated-embedding extraction the stability check relies on doesn't need
    per-epoch history at all). Called before ``_strip_embeddings`` so the
    lists are still present.
    """
    for row in reversed(all_results):
        if 'embedding_history_x' in row:
            return {
                'embedding_history_x': row['embedding_history_x'],
                'embedding_history_y': row['embedding_history_y'],
            }
    return {}


def _strip_embeddings(results: list) -> None:
    """Remove embedding arrays from result dicts in-place.

    Embedding arrays must not end up as DataFrame columns — they are 2-D numpy
    arrays and would be stored as object-dtype cells, making the DataFrame
    unusable for aggregation.  Stripping them here is always safe; callers that
    need the embeddings collect them via ``_extract_last_split_embeddings`` and
    ``_extract_per_split_rotated`` before calling this.
    """
    for row in results:
        row.pop('embeddings_x', None)
        row.pop('embeddings_y', None)
        row.pop('embeddings_x_rotated', None)
        row.pop('embeddings_y_rotated', None)
        row.pop('embeddings_rotation_singular_values', None)
        row.pop('embeddings_rotation_x', None)
        row.pop('embeddings_rotation_y', None)
        row.pop('embedding_history_x', None)
        row.pop('embedding_history_y', None)


def _extract_last_split_embeddings(
    all_results: list,
    n_splits: int,
    analysis_params: Dict[str, Any],
    split_method: str,
) -> Optional[Dict[str, Any]]:
    """Return full per-sample embeddings from the last split's final result, or
    None. Only called when the caller originally set return_embeddings=True
    (this mode always computes embeddings internally for the stability check,
    but only exposes the full per-sample arrays when explicitly requested).
    Also forwards the rotated variants and rotation matrices/singular values
    when present -- this mode always requests return_rotated_embeddings=True
    internally, but only surfaces the resulting keys to the caller under the
    same conditions as any other mode (return_embeddings=True for the rotated
    embeddings themselves, return_rotation_matrices=True for the matrices).

    With ``n_splits > 1`` the last entry in ``all_results`` corresponds to the
    last split (highest ``split_id``); a log message informs the caller.
    """
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
            out = {
                'embeddings_x': row['embeddings_x'],
                'embeddings_y': row['embeddings_y'],
            }
            for key in ('embeddings_x_rotated', 'embeddings_y_rotated',
                        'embeddings_rotation_singular_values',
                        'embeddings_rotation_x', 'embeddings_rotation_y'):
                if key in row:
                    out[key] = row[key]
            return out

    logger.warning(
        "return_embeddings=True but no embeddings were found in the split results. "
        "Check that y_data is provided and the split produced valid results."
    )
    return None


def _run_single_split(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    analysis_params: Dict[str, Any],
    sweep_grid: Optional[Dict[str, Any]],
    n_workers: int,
    split_id: int,
) -> list:
    """Run one channel-split and return result dicts with split_id attached.

    Varies random_seed deterministically by split_id. Without this, every
    split/rerun gets the IDENTICAL effective training seed regardless of
    split_id: ParameterSweep derives its per-task seed from the sweep
    combination index (sweep.py's `_seed_key = f"c{i_combo}"`), which is
    always "c0" for a single-combination sweep like each dimensionality split
    -- independent of split_id. Confirmed directly: without this fix, n_splits
    reruns produced bit-identical results (same weights, same everything),
    defeating both the cross-run stability check this mode is built on and
    the pre-existing "independent weight initialisations" claim for
    interaction-mode n_splits repeats.
    """
    split_params = analysis_params
    base_seed = analysis_params.get('random_seed')
    if base_seed is not None:
        split_params = analysis_params.copy()
        split_params['random_seed'] = (int(base_seed) + (split_id + 1) * 7919) % (2 ** 31)

    sweep = ParameterSweep(x_data=x_a, y_data=x_b, base_params=split_params)
    results = sweep.run(sweep_grid=sweep_grid or {}, n_workers=n_workers,
                        is_proc_sweep=False)
    for res in results:
        res['split_id'] = split_id
    return results
