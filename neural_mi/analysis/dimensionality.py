# neural_mi/analysis/dimensionality.py
"""Estimates the latent dimensionality of a dataset using spectral metrics.

This module forces the use of a Hybrid critic with a large bottleneck and
analyzes the cross-covariance spectrum of the resulting embeddings to
determine Intrinsic or Interaction Dimensionality.
"""
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

from .sweep import ParameterSweep
from neural_mi.logger import logger
from neural_mi.utils import _configure_multiprocessing, _ensure_cpu


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
    **kwargs
) -> pd.DataFrame:
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

        - ``'summary'`` *(default)* — compute the participation ratio only.
        - ``'full'`` — compute all spectral metrics and include the raw
          singular values array in each result row.
    n_workers : int, optional
        Number of parallel workers.  When ``n_splits > 1`` the workers are
        distributed *across splits* (each split's inner sweep runs
        sequentially to avoid nested pools).  When ``n_splits == 1`` the
        workers are forwarded into the inner ``ParameterSweep`` to
        parallelise any sweep-grid combinations.  Defaults to 1.

    Returns
    -------
    pd.DataFrame
        One row per split (and per sweep combination). Columns include split_id,
        train_mi, participation_ratio, and any additional spectral metrics.
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
        _warn_if_near_embed_ceiling(df_out, analysis_params)
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
    _warn_if_near_embed_ceiling(df, analysis_params)
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


def _warn_if_near_embed_ceiling(df: pd.DataFrame, analysis_params: Dict[str, Any]) -> None:
    """Issue a UserWarning when the participation ratio is close to the embedding ceiling.

    The participation ratio (PR) is bounded above by ``embedding_dim``.  If the
    estimated PR exceeds 80 % of that ceiling the measurement is likely truncated:
    the true dimensionality may be higher.
    """
    if 'participation_ratio' not in df.columns:
        return
    valid_prs = df['participation_ratio'].dropna()
    if valid_prs.empty:
        return
    mean_pr = float(valid_prs.mean())
    embed_dim = analysis_params.get('embedding_dim', 64)
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
