# neural_mi/analysis/dimensionality.py
"""How many embedding dimensions the estimator needs to carry what two views share.

The MI is estimated at a series of embedding dimensions ``k`` with the hybrid
critic. A restart that settled short of a latent direction can only read low.
At each ``k`` the best of several restarts counts. Because a larger embedding can
represent any smaller solution, the curve is the running maximum over ``k``. The
reading is the smallest ``k`` at which the curve reaches a set fraction of its
plateau. Directions an overparametrized encoder builds from the true factors add
no information and cannot raise the curve. The reading therefore bounds from
above the dimensions needed to carry that fraction of the shared information.

A large reference fit runs first. The participation ratio of its cross-covariance
spectrum chooses the grid of ``k``. Its MI is part of the plateau the grid is
judged against. See ``THEORY.md``.
"""
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .sweep import ParameterSweep
from neural_mi.embeddings_io import with_model_labels
from neural_mi.logger import logger, user_stacklevel
from neural_mi.utils import _ensure_cpu, mi_report_units

# The first grid reaches twice the reference fit's participation ratio. The ratio
# fell below the true dimension at 8 and 16 latent dimensions in the tests. The
# factor of two leaves room for that.
_GRID_SMALL, _GRID_MEDIUM, _LOG_POINTS = 10, 20, 10
# When twice the reference fit's participation ratio reaches this fraction of its
# size, the reference may be too small to hold every direction. It is refitted at
# four times the ratio.
_REFERENCE_FILL = 0.75
# Consecutive values of k at the threshold that end the default grid early.
_CONFIRM = 3
# Warn when restarts below the reading disagree by more than this fraction of the
# plateau, and when the held-out plateau sits this far below the training side.
_RESTART_SPREAD, _HELD_OUT_GAP = 0.10, 0.10

SPLIT_METHODS = ('random', 'spatial', 'temporal', 'index', 'horizontal', 'vertical',
                 'row_interleaved', 'col_interleaved', 'diagonal', 'antidiagonal')


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------

def _warn_if_near_ceiling(fits: List[Dict[str, Any]], ceiling_mi_fraction: float,
                          base_params: Optional[Dict[str, Any]]) -> None:
    """Warn when the reference fits' held-out MI is close to its evaluation ceiling."""
    pairs = [(f['test_mi'], f['eval_size']) for f in fits
             if f.get('test_mi') is not None and f.get('eval_size')]
    if not pairs:
        return
    test_mi = float(np.mean([p[0] for p in pairs]))
    eval_size = float(np.mean([p[1] for p in pairs]))
    if eval_size <= 1:
        return
    ceiling = math.log(eval_size)
    if test_mi >= ceiling_mi_fraction * ceiling:
        scale, units = mi_report_units(base_params)
        warnings.warn(
            f"Dimensionality: the reference fits' MI ({test_mi * scale:.3f} {units}) is near its "
            f"evaluation ceiling (log(eval_size)={ceiling * scale:.3f} {units}). The ceiling may "
            f"have set the plateau. Raise max_eval_samples.",
            UserWarning, stacklevel=user_stacklevel(),
        )


# ---------------------------------------------------------------------------
# Views: the pair of inputs each fit compares
# ---------------------------------------------------------------------------

def _n_samples_for_shared_split(x_data, y_data, analysis_params: Dict[str, Any]) -> int:
    """Number of rows the shared train/test split is computed over.

    For windowed data it is ``x_data.shape[0]``. When windowing is deferred
    (raw 2-D data with ``shift_windows`` on a regular grid) it is the window
    count, from :func:`~neural_mi.data.shift_windowing.n_windows_if_deferred`.
    """
    from neural_mi.data.shift_windowing import n_windows_if_deferred
    return n_windows_if_deferred(x_data, y_data, analysis_params)


def _get_or_create_shared_split(analysis_params: Dict[str, Any], n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """One train/test split shared by every fit of the call.

    Every value of ``k`` and every restart is then scored on the same held-out
    rows. Explicit ``train_indices``/``test_indices`` are used as given.
    Otherwise the split comes from the trainer's own splitting code.
    """
    existing_train = analysis_params.get('train_indices')
    if existing_train is not None:
        return existing_train, analysis_params.get('test_indices')

    from neural_mi.training.trainer import Trainer
    split_mode = analysis_params.get('split_mode', 'blocked')
    train_fraction = analysis_params.get('train_fraction', 0.9)
    if split_mode == 'random':
        return Trainer._create_random_split(None, n_samples, train_fraction)
    n_test_blocks = analysis_params.get('n_test_blocks', 5)
    gap_fraction = analysis_params.get('split_gap_fraction', 0.5)
    return Trainer._create_blocked_split(None, n_samples, train_fraction, n_test_blocks, gap_fraction)


def _halves(x_data: torch.Tensor, params: Dict[str, Any], split_method: str,
            n_splits: int, kwargs: Dict[str, Any]) -> List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """The two halves of X for each split, with the parameters that fit them.

    Only ``'random'`` draws a new channel assignment per split. Every other
    method has one fixed assignment and so gives one split.
    """
    n_channels = x_data.shape[1]

    def unequal(a, b):
        # Halves of different sizes cannot share one encoder.
        if params.get('shared_encoder', False) and int(np.prod(a.shape[1:])) != int(np.prod(b.shape[1:])):
            logger.warning(
                f"split_method='{split_method}' produced unequal halves "
                f"(X: {tuple(a.shape[1:])}, Y: {tuple(b.shape[1:])}). shared_encoder=True needs "
                f"equal halves and is turned off for this run."
            )
            return {**params, 'shared_encoder': False}
        return params

    if split_method == 'temporal':
        lag = kwargs.get('lag', 1)
        if not isinstance(lag, int) or lag < 1:
            raise ValueError(f"'lag' must be a positive integer, got {lag!r}.")
        x_a, x_b = x_data[:-lag, ...], x_data[lag:, ...]
        logger.info(f"Temporal split at lag={lag}: {x_a.shape[0]} aligned sample pairs.")
        return [(x_a, x_b, params)]

    if split_method in ('random', 'spatial'):
        if n_channels < 2:
            raise ValueError(
                f"Cannot perform '{split_method}' channel split with fewer than 2 channels. "
                f"x_data has shape {tuple(x_data.shape)}."
            )
        half = n_channels // 2
        out = []
        for _ in range(n_splits if split_method == 'random' else 1):
            order = np.random.permutation(n_channels) if split_method == 'random' else np.arange(n_channels)
            x_a, x_b = x_data[:, order[:half], ...], x_data[:, order[half:], ...]
            if half != n_channels - half and params.get('shared_encoder', False):
                logger.warning(
                    f"split_method='{split_method}' on an odd channel count ({n_channels}) "
                    f"produces unequal halves (X: {half}, Y: {n_channels - half}), incompatible "
                    f"with shared_encoder=True. Disabling shared_encoder for this run."
                )
                out.append((x_a, x_b, {**params, 'shared_encoder': False}))
            else:
                out.append((x_a, x_b, params))
        return out

    if split_method == 'index':
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
            raise ValueError("channel_indices_x covers all channels and leaves Y empty.")
        if not channel_indices_x:
            raise ValueError("channel_indices_x is empty and leaves X empty.")
        x_a, x_b = x_data[:, channel_indices_x, ...], x_data[:, channel_indices_y, ...]
        if len(channel_indices_x) != len(channel_indices_y) and params.get('shared_encoder', False):
            logger.warning(
                f"split_method='index' with unequal channel counts "
                f"(X: {len(channel_indices_x)}, Y: {len(channel_indices_y)}) is "
                f"incompatible with shared_encoder=True. Disabling shared_encoder "
                f"for this run."
            )
            return [(x_a, x_b, {**params, 'shared_encoder': False})]
        return [(x_a, x_b, params)]

    # The image splits.
    if x_data.ndim != 4:
        raise ValueError(
            f"split_method='{split_method}' requires 4-D input (N, C, H, W). "
            f"Got shape {tuple(x_data.shape)} ({x_data.ndim}-D). "
            "For 3-D or 2-D data, use split_method='random' or 'spatial' to "
            "split along the channel axis instead."
        )
    H, W = x_data.shape[2], x_data.shape[3]
    if split_method in ('horizontal', 'row_interleaved') and H < 2:
        raise ValueError(f"split_method='{split_method}' requires H >= 2, got H={H}.")
    if split_method in ('vertical', 'col_interleaved') and W < 2:
        raise ValueError(f"split_method='{split_method}' requires W >= 2, got W={W}.")
    if split_method == 'horizontal':
        x_a, x_b = x_data[:, :, :H // 2, :], x_data[:, :, H // 2:, :]
    elif split_method == 'vertical':
        x_a, x_b = x_data[:, :, :, :W // 2], x_data[:, :, :, W // 2:]
    elif split_method == 'row_interleaved':
        x_a, x_b = x_data[:, :, 0::2, :], x_data[:, :, 1::2, :]
    elif split_method == 'col_interleaved':
        x_a, x_b = x_data[:, :, :, 0::2], x_data[:, :, :, 1::2]
    else:  # diagonal or antidiagonal: triangular pixel masks
        embedding_model = params.get('embedding_model', 'mlp')
        if embedding_model in ('cnn2d', 'cnn'):
            raise ValueError(
                f"split_method='{split_method}' produces irregularly-shaped triangular "
                f"pixel subsets that cannot be represented as rectangular (N, C, H, W) "
                f"tensors. embedding_model='{embedding_model}' requires rectangular 2-D spatial "
                "input. Use embedding_model='mlp' for geometric diagonal splits."
            )
        if H != W:
            logger.warning(
                f"split_method='{split_method}' on non-square input (H={H}, W={W}): "
                "the two triangular halves will have unequal pixel counts. "
                "shared_encoder will be disabled automatically if flat dims differ."
            )
        rows = torch.arange(H, device=x_data.device).unsqueeze(1)
        cols = torch.arange(W, device=x_data.device).unsqueeze(0)
        if split_method == 'diagonal':
            mask_a, mask_b = (rows <= cols).reshape(-1), (rows > cols).reshape(-1)
        else:
            mask_a, mask_b = (rows + cols <= W - 1).reshape(-1), (rows + cols > W - 1).reshape(-1)
        flat = x_data.reshape(x_data.shape[0], x_data.shape[1], -1)
        x_a, x_b = flat[:, :, mask_a], flat[:, :, mask_b]
    return [(x_a, x_b, unequal(x_a, x_b))]


# ---------------------------------------------------------------------------
# The grid of k
# ---------------------------------------------------------------------------

def _log_grid(low: int, high: int, n: int = _LOG_POINTS) -> List[int]:
    """About ``n`` integers from ``low`` to ``high``, evenly spaced on a log scale."""
    if high <= low:
        return [int(low)]
    return sorted({int(round(v)) for v in np.geomspace(low, high, n)})


def _default_grid(pr: float) -> List[int]:
    """The first grid of k, with a message when it goes past 10."""
    reach = 2 * pr
    if reach <= _GRID_SMALL:
        return list(range(1, _GRID_SMALL + 1))
    if reach <= _GRID_MEDIUM:
        logger.warning(
            f"Dimensionality: the grid runs over embedding_dim 1 to {_GRID_MEDIUM} because the "
            f"reference fit has a participation ratio of {pr:.1f}. Pass embedding_dims to choose "
            f"the grid.")
        return list(range(1, _GRID_MEDIUM + 1))
    top = int(math.ceil(reach))
    grid = _log_grid(1, top)
    logger.warning(
        f"Dimensionality: the grid runs over {len(grid)} values of embedding_dim from 1 to {top} "
        f"on a log scale ({grid}) because the reference fit has a participation ratio of "
        f"{pr:.1f}. Pass embedding_dims as a list or a range for a finer grid.")
    return grid


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit(views: List[Tuple[int, Any, Any, Dict[str, Any]]], ks: Sequence[int], n_restarts: int,
         n_workers: int) -> List[Dict[str, Any]]:
    """Train ``n_restarts`` networks at every k of every view, all in one pool."""
    tasks, runner = [], None
    for split_id, x_a, x_b, params in views:
        sweep = ParameterSweep(x_data=x_a, y_data=x_b,
                               base_params=with_model_labels(params, split_id=split_id))
        runner = runner or sweep
        grid = {'embedding_dim': list(ks), 'run_id': list(range(n_restarts))}
        for task_x, task_y, task_params, task_id in sweep._prepare_tasks(grid, is_proc_sweep=False):
            task_params = dict(task_params, split_id=split_id)
            # A seed of its own for every split, k and restart.
            task_params['_seed_key'] = (f"s{split_id}k{task_params['embedding_dim']}"
                                        f"r{task_params['run_id']}")
            tasks.append((task_x, task_y, task_params, task_id))
    return runner._run_parallel(tasks, n_workers) if tasks else []


def _best(fits: List[Dict[str, Any]], split_id: int) -> Dict[int, float]:
    """The best restart's training-side MI at each k of one split, in nats."""
    best: Dict[int, float] = {}
    for f in fits:
        if f['split_id'] == split_id and f.get('train_mi') is not None:
            k = int(f['embedding_dim'])
            best[k] = max(best.get(k, -math.inf), float(f['train_mi']))
    return best


def _curve(best: Dict[int, float]) -> Dict[int, float]:
    """The running maximum of the best values over increasing k."""
    curve, top = {}, -math.inf
    for k in sorted(best):
        top = max(top, best[k])
        curve[k] = top
    return curve


def _reading(best: Dict[int, float], ratio: float) -> Tuple[Optional[int], float]:
    """The smallest k whose curve reaches ``ratio`` of the plateau, and the plateau."""
    curve = _curve(best)
    if not curve:
        return None, math.nan
    plateau = max(curve.values())
    if not plateau > 0:
        return None, plateau
    return next(k for k in sorted(curve) if curve[k] >= ratio * plateau), plateau


def _confirmed(best: Dict[int, float], order: Sequence[int], threshold: float) -> Optional[int]:
    """The first k of ``_CONFIRM`` consecutive grid values at the threshold, if any."""
    run = 0
    for i, k in enumerate(order):
        run = run + 1 if best.get(k, -math.inf) >= threshold else 0
        if run == _CONFIRM:
            return order[i - _CONFIRM + 1]
    return None


def _run_grid(views, grid: List[int], n_restarts: int, n_workers: int, ratio: float,
              fits: List[Dict[str, Any]], early_stop: bool) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Fit the grid from small k upward, stopping once every split has confirmed its reading."""
    if not early_stop:
        return _fit(views, grid, n_restarts, n_workers), list(grid)
    new, done = [], []
    for start in range(0, len(grid), _CONFIRM):
        wave = grid[start:start + _CONFIRM]
        new += _fit(views, wave, n_restarts, n_workers)
        done += wave
        seen = fits + new
        confirmed = []
        for split_id, *_ in views:
            best = _best(seen, split_id)
            plateau = max(best.values())
            confirmed.append(_confirmed(best, done, ratio * plateau))
        if all(c is not None for c in confirmed) and len(done) < len(grid):
            logger.warning(
                f"Dimensionality: the grid stopped at embedding_dim={done[-1]} once {_CONFIRM} "
                f"consecutive values reached {ratio:.0%} of the plateau. The rest of the grid "
                f"({grid[len(done)]} to {grid[-1]}) was not fitted. Pass embedding_dims to fit "
                f"every value."
            )
            break
    return new, done


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def run_dimensionality_analysis(
    x_data: torch.Tensor,
    base_params: Dict[str, Any],
    y_data: Optional[torch.Tensor] = None,
    *,
    embedding_dims: Optional[Sequence[int]] = None,
    n_restarts: int = 4,
    saturation_ratio: float = 0.95,
    reference_dim: Optional[int] = None,
    split_method: str = 'random',
    n_splits: Optional[int] = None,
    ceiling_mi_fraction: float = 0.85,
    n_workers: int = 1,
    user_set_keys: Optional[set] = None,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The MI against embedding dimension and the smallest dimension that carries it.

    Parameters
    ----------
    x_data : torch.Tensor
        X. Without ``y_data`` it is split into two halves by ``split_method``.
    base_params : dict
        The trainer's settings. ``embedding_dim`` is set by the analysis.
    y_data : torch.Tensor, optional
        Y. When given, the two views are X and Y.
    embedding_dims : sequence of int, optional
        The values of ``embedding_dim`` to fit. By default the grid is chosen
        from the reference fit and stops early once the reading is confirmed.
    n_restarts : int, default=4
        Networks trained at every ``embedding_dim`` of every split.
    saturation_ratio : float, default=0.95
        The fraction of the plateau the curve must reach.
    reference_dim : int, optional
        The embedding dimension of the large reference fit. By default it is 64,
        refitted larger when its participation ratio comes close to that.
    split_method : str, default='random'
        How X is split when ``y_data`` is not given.
    n_splits : int, optional
        Random channel splits of X (5 by default). Only ``'random'`` draws more
        than one.
    ceiling_mi_fraction : float, default=0.85
        Warn when the reference MI is this close to its evaluation ceiling.
    n_workers : int, default=1
        Worker processes for the fits.
    user_set_keys : set, optional
        The keys of ``base_params`` the caller set before defaults were filled,
        so that this mode's own defaults apply only where the caller chose nothing.
    **kwargs
        ``lag`` for ``split_method='temporal'`` and ``channel_indices_x`` for
        ``'index'``.

    Returns
    -------
    fits : list of dict
        One training result per network, with ``split_id``, ``embedding_dim``
        and ``run_id``.
    info : dict
        The reading and how it was reached: ``dimension_at_most`` (the median
        over splits), ``dimension_at_most_std``, ``dimension_at_most_per_split``,
        ``plateau`` (per split, nats), ``embedding_dims`` (every value fitted),
        ``reference_dim``, ``pr_singular``, ``pr_eig``, ``saturation_ratio``,
        ``n_restarts`` and ``stopped_early``.
    """
    params = dict(base_params)
    user_set = set(user_set_keys) if user_set_keys is not None else set(base_params)

    # The hybrid critic is this mode's default. A separable critic works with a
    # warning; a concat critic has no embedding of either side.
    critic_type = params.get('critic_type') if 'critic_type' in user_set else None
    if critic_type == 'concat':
        raise ValueError(
            "mode='dimensionality' cannot use critic_type='concat'. A concat critic embeds X "
            "and Y jointly and has no embedding of either side for the curve to vary. Use the "
            "default 'hybrid' or 'separable'."
        )
    if critic_type == 'separable':
        warnings.warn(
            "mode='dimensionality' with critic_type='separable' overstates the dimension. A "
            "dot-product critic needs more embedding dimensions than the shared structure has. "
            "This mode defaults to 'hybrid'.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    else:
        params['critic_type'] = 'hybrid'
    if params['critic_type'] == 'hybrid' and params.get('norm_layer', 'auto') == 'auto':
        # Layer norm stops the fits at k = d from stalling. It also divides out
        # each sample's scale. The reading is taken against a plateau measured
        # the same way.
        params['norm_layer'] = 'layer'
    if 'n_epochs' not in user_set:
        params['n_epochs'] = 500
    if 'patience' not in user_set:
        params['patience'] = 50
    if 'shared_encoder' not in user_set:
        # Two halves of one recording can share an encoder; X and Y need not.
        params['shared_encoder'] = y_data is None
    params.setdefault('whitening', 'std')

    if y_data is not None and n_splits is not None:
        raise ValueError(
            "Dimensionality(n_splits=...) counts random channel splits of X and applies without "
            "y_data. With y_data every fit compares X and Y. n_restarts sets the repeats."
        )
    if y_data is None and split_method not in SPLIT_METHODS:
        raise ValueError(f"Unknown split_method: '{split_method}'. Expected one of: {list(SPLIT_METHODS)}.")
    if y_data is None and split_method != 'random' and n_splits not in (None, 1):
        raise ValueError(
            f"split_method='{split_method}' splits X the same way every time. n_splits={n_splits} "
            f"would repeat one split. Use n_restarts for repeats or split_method='random' for "
            f"several channel assignments."
        )

    # The two views of every split.
    if y_data is not None:
        pairs = [(x_data, y_data, params)]
    else:
        pairs = _halves(x_data, params, split_method, n_splits or 5, kwargs)
    # One held-out set for every fit of the call.
    first_a, first_b, _ = pairs[0]
    n_rows = _n_samples_for_shared_split(first_a, first_b if y_data is not None else None, params)
    train_idx, test_idx = _get_or_create_shared_split(params, n_rows)
    views = [(i, _ensure_cpu(a), _ensure_cpu(b),
              {**p, 'train_indices': train_idx, 'test_indices': test_idx})
             for i, (a, b, p) in enumerate(pairs)]
    split_ids = [v[0] for v in views]
    logger.info(f"Dimensionality: {len(views)} split(s), {n_restarts} restart(s) per embedding_dim.")

    # 1. The reference fit: its participation ratio chooses the grid.
    reference_given = reference_dim is not None
    reference_dim = int(reference_dim) if reference_given else 64
    if embedding_dims is not None and max(int(k) for k in embedding_dims) >= reference_dim:
        raise ValueError(
            f"Dimensionality(embedding_dims=...) reaches {max(int(k) for k in embedding_dims)}, "
            f"at or above the reference embedding_dim={reference_dim}. Every value on the grid "
            f"must be smaller than the reference. Pass a larger reference_dim."
        )
    fits = _fit(views, [reference_dim], n_restarts, n_workers)
    pr = _median([f.get('pr_singular') for f in fits])
    if not reference_given and np.isfinite(pr) and 2 * pr >= _REFERENCE_FILL * reference_dim:
        larger = int(math.ceil(4 * pr))
        logger.warning(
            f"Dimensionality: the participation ratio of the reference fit ({pr:.1f}) is close to "
            f"its embedding_dim={reference_dim}. The reference is refitted at embedding_dim={larger}."
        )
        reference_dim = larger
        fits = _fit(views, [reference_dim], n_restarts, n_workers)
        pr = _median([f.get('pr_singular') for f in fits])
    reference_fits = list(fits)

    # 2. The grid, from small k upward.
    if embedding_dims is not None:
        grid = sorted({int(k) for k in embedding_dims})
        early_stop = False
    else:
        grid = [k for k in _default_grid(pr if np.isfinite(pr) else _GRID_SMALL / 2)
                if k < reference_dim]
        early_stop = True
    new, fitted = _run_grid(views, grid, n_restarts, n_workers, saturation_ratio, fits, early_stop)
    fits += new
    stopped_early = early_stop and len(fitted) < len(grid)

    # 3. A default grid whose curve never reached the threshold is extended once.
    unsaturated = [s for s in split_ids
                   if _reading(_best(fits, s), saturation_ratio)[0] == reference_dim]
    if unsaturated and embedding_dims is None and grid and grid[-1] + 1 < reference_dim:
        extra = [k for k in _log_grid(grid[-1] + 1, reference_dim - 1) if k not in grid]
        logger.warning(
            f"Dimensionality: the curve of split(s) {unsaturated} did not reach "
            f"{saturation_ratio:.0%} of its plateau by embedding_dim={grid[-1]}. The grid is "
            f"extended over {extra}."
        )
        new, more = _run_grid(views, extra, n_restarts, n_workers, saturation_ratio, fits, True)
        fits += new
        fitted += more

    # 4. The readings.
    readings, plateaus = {}, {}
    for s in split_ids:
        readings[s], plateaus[s] = _reading(_best(fits, s), saturation_ratio)
    reached = [k for k in readings.values() if k is not None]
    empty = [s for s, k in readings.items() if k is None]
    if empty:
        warnings.warn(
            f"Dimensionality: the views of split(s) {empty} share no information the estimator "
            f"can find. Their curves stay at 0 and give no reading.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    below_reference = [s for s, k in readings.items() if k is not None and k >= reference_dim]
    if below_reference:
        warnings.warn(
            f"Dimensionality: the curve of split(s) {below_reference} reached {saturation_ratio:.0%} "
            f"of its plateau only at the reference embedding_dim={reference_dim}. The reading equals "
            f"the reference and bounds nothing below it. Pass embedding_dims that reach closer to "
            f"the reference or a larger reference_dim.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    _warn_about_fits(fits, reference_fits, readings, plateaus, params)
    _warn_if_near_ceiling(reference_fits, ceiling_mi_fraction, base_params)

    info = {
        'dimension_at_most': _median_reading(reached),
        'dimension_at_most_std': float(np.std(reached, ddof=1)) if len(reached) > 1 else None,
        'dimension_at_most_per_split': readings,
        'plateau': plateaus,
        'embedding_dims': sorted(set(fitted) | {reference_dim}),
        'reference_dim': reference_dim,
        'pr_singular': pr,
        'pr_eig': _median([f.get('pr_eig') for f in reference_fits]),
        'saturation_ratio': saturation_ratio,
        'n_restarts': n_restarts,
        'stopped_early': bool(stopped_early),
    }
    logger.info(f"--- Dimensionality: at most {info['dimension_at_most']} embedding dimensions ---")
    return fits, info


def _median(values) -> float:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else math.nan


def _median_reading(readings: List[int]):
    if not readings:
        return None
    median = float(np.median(readings))
    return int(median) if median.is_integer() else median


def _warn_about_fits(fits, reference_fits, readings, plateaus, params) -> None:
    """The warnings about the fits behind a reading."""
    scale, units = mi_report_units(params)
    # Restarts that disagree below the reading.
    spread = []
    for s, reading in readings.items():
        if reading is None:
            continue
        for k in sorted({int(f['embedding_dim']) for f in fits if f['split_id'] == s and f['embedding_dim'] < reading}):
            values = [f['train_mi'] for f in fits
                      if f['split_id'] == s and f['embedding_dim'] == k and f.get('train_mi') is not None]
            if len(values) > 1 and max(values) - min(values) > _RESTART_SPREAD * plateaus[s]:
                spread.append(k)
    if spread:
        warnings.warn(
            f"Dimensionality: the restarts at embedding_dim {sorted(set(spread))} below the reading "
            f"differ by more than {_RESTART_SPREAD:.0%} of the plateau. The curve uses the best "
            f"restart at each value. Raise n_restarts if the best one may also have settled short "
            f"of a direction.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    # A held-out plateau well below the training side: too few samples for the
    # information per dimension, and the reading may overstate the dimension.
    for s, plateau in plateaus.items():
        held_out = _median([f.get('test_mi') for f in reference_fits if f['split_id'] == s])
        if plateau > 0 and np.isfinite(held_out) and held_out < (1 - _HELD_OUT_GAP) * plateau:
            warnings.warn(
                f"Dimensionality: at the reference fit the held-out MI ({held_out * scale:.3f} {units}) "
                f"is more than {_HELD_OUT_GAP:.0%} below the plateau ({plateau * scale:.3f} {units}). "
                f"The data hold few samples for the information each dimension carries. The reading "
                f"may then overstate the dimension while still bounding it from above. More data "
                f"tightens it.",
                UserWarning, stacklevel=user_stacklevel(),
            )
            break
