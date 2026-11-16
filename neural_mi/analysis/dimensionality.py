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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .sweep import ParameterSweep
from neural_mi.defaults import setting
from neural_mi.seeds import generator as seed_generator, network_seed
from neural_mi.embeddings_io import with_model_labels
from neural_mi.logger import logger, user_stacklevel
from neural_mi.utils import _ensure_cpu, mi_report_units

# How the grid of embedding dimensions is chosen and when its fits warn is set by
# Dimensionality(...): grid_reach, refit_reach, reference_fill, grid_small,
# grid_medium, grid_points, confirm_steps, restart_spread and held_out_gap, with
# their defaults in the mode's settings (analysis/modes.py, MODES).

SPLIT_METHODS = ('random', 'spatial', 'temporal', 'index', 'horizontal', 'vertical',
                 'row_interleaved', 'col_interleaved', 'diagonal', 'antidiagonal')


# ---------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------

def _warn_if_near_ceiling(fits: List[Dict[str, Any]], ceiling_mi_fraction: float,
                          base_params: Optional[Dict[str, Any]]) -> None:
    """Warn when the reference fits' held-out MI is close to its evaluation ceiling.

    The ceiling is the trainer's: the log of the held-out rows in one block of the evaluation, which is
    smaller than the log of all the held-out rows when they are scored in several blocks.
    """
    pairs = []
    for f in fits:
        if f.get('test_mi') is None or not f.get('eval_size'):
            continue
        block = -(-int(f['eval_size']) // int(f.get('test_eval_blocks') or 1))
        if block >= 2:
            pairs.append((f['test_mi'], math.log(block)))
    if not pairs:
        return
    test_mi = float(np.mean([p[0] for p in pairs]))
    ceiling = float(np.mean([p[1] for p in pairs]))
    if test_mi >= ceiling_mi_fraction * ceiling:
        scale, units = mi_report_units(base_params)
        warnings.warn(
            f"Dimensionality: the reference fits' MI ({test_mi * scale:.3f} {units}) is near its "
            f"evaluation ceiling ({ceiling * scale:.3f} {units}, the log of the held-out rows that one "
            f"evaluation scores). The ceiling may have set the plateau. Raise max_eval_samples.",
            UserWarning, stacklevel=user_stacklevel(),
        )


# ---------------------------------------------------------------------------
# Views: the pair of inputs each fit compares
# ---------------------------------------------------------------------------

def _n_samples_for_shared_split(x_data, y_data, analysis_params: Dict[str, Any]) -> int:
    """Number of rows the shared train/test split is computed over.

    For windowed data it is ``x_data.shape[0]``. When each fit windows raw
    streams itself it is the count of rows their plan holds
    (:func:`~neural_mi.data.plan.count_rows`).
    """
    from neural_mi.data.plan import count_rows
    return count_rows(x_data, y_data, analysis_params)


def _get_or_create_shared_split(analysis_params: Dict[str, Any], n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """One train/test split shared by every fit of the call.

    Every value of ``k`` and every restart then trains on the same training rows
    and is scored on the same held-out set. A fit scores a subset of up to
    ``max_eval_samples`` held-out rows of its own when the set is larger.
    Explicit ``train_indices``/``test_indices`` are used as given.
    Otherwise the split comes from the trainer's own splitting code, drawn from
    the call's seed (neural_mi/seeds.py).
    """
    existing_train = analysis_params.get('train_indices')
    if existing_train is not None:
        return existing_train, analysis_params.get('test_indices')

    from neural_mi.data.rows import row_geometry
    from neural_mi.training.trainer import Trainer
    from neural_mi.utils import warn_if_random_split_leaks
    split_mode = setting(analysis_params, 'split_mode')
    train_fraction = setting(analysis_params, 'train_fraction')
    rows = row_geometry(analysis_params)
    # The splitting code draws from its caller's generator (trainer._rng_of).
    drawer = SimpleNamespace(_rng=np.random.RandomState(
        network_seed(setting(analysis_params, 'random_seed'), 0, 'held-out rows')))
    if split_mode == 'random':
        if rows is not None and rows.overlapping:
            warn_if_random_split_leaks(rows)
        return Trainer._create_random_split(drawer, n_samples, train_fraction)
    n_test_blocks = setting(analysis_params, 'n_test_blocks')
    gap_fraction = setting(analysis_params, 'split_gap_fraction')
    geometry = {'span': rows.span, 'step': rows.step} if rows is not None else {}
    return Trainer._create_blocked_split(drawer, n_samples, train_fraction, n_test_blocks, gap_fraction,
                                         **geometry)


def _distinct_orders(n_channels: int, n_splits: int, seed, splits_chosen: bool) -> List[np.ndarray]:
    """Random channel orders whose first ``n_channels // 2`` channels form ``n_splits`` different pairs of
    halves, or every possible pair when the channels allow fewer. A repeated pair would count one split
    twice in the median and the spread over splits."""
    half = n_channels // 2
    possible = math.comb(n_channels, half) // (2 if 2 * half == n_channels else 1)
    if possible < n_splits:
        text = (f"Dimensionality: {n_channels} channels divide into only {possible} different pairs of "
                f"halves, so the mode runs {possible} splits instead of n_splits={n_splits}.")
        if splits_chosen:
            warnings.warn(text, UserWarning, stacklevel=user_stacklevel())
        else:
            logger.info(text)
    orders, seen, draw = [], set(), 0
    while len(orders) < min(n_splits, possible):
        order = seed_generator(seed, 0, f"channels{draw}").permutation(n_channels)
        draw += 1
        pair = frozenset((frozenset(order[:half].tolist()), frozenset(order[half:].tolist())))
        if pair not in seen:
            seen.add(pair)
            orders.append(order)
    return orders


def _halves(x_data: torch.Tensor, params: Dict[str, Any], split_method: str,
            n_splits: int, kwargs: Dict[str, Any], shared_chosen: bool = False,
            splits_chosen: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """The two halves of X for each split, with the parameters that fit them.

    Only ``'random'`` draws a new channel assignment per split, and no two of its
    splits divide the channels the same way. Every other method has one fixed
    assignment and so gives one split. ``shared_chosen`` and ``splits_chosen``
    are whether the user set ``shared_encoder`` and ``n_splits``, as opposed to
    the mode's defaults.
    """
    n_channels = x_data.shape[1]

    def unequal(a, b):
        # Halves of different sizes cannot share one encoder. Turning off a shared
        # encoder the user asked for warns; turning off the mode's default only logs.
        if setting(params, 'shared_encoder') and int(np.prod(a.shape[1:])) != int(np.prod(b.shape[1:])):
            text = (f"split_method='{split_method}' produced unequal halves "
                    f"(X: {tuple(a.shape[1:])}, Y: {tuple(b.shape[1:])}). shared_encoder=True needs "
                    f"equal halves and is turned off for this run.")
            if shared_chosen:
                warnings.warn(text, UserWarning, stacklevel=user_stacklevel())
            else:
                logger.info(text)
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
        orders = (_distinct_orders(n_channels, n_splits, setting(params, 'random_seed'), splits_chosen)
                  if split_method == 'random' else [np.arange(n_channels)])
        out = []
        for order in orders:
            x_a, x_b = x_data[:, order[:half], ...], x_data[:, order[half:], ...]
            out.append((x_a, x_b, unequal(x_a, x_b)))
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
        return [(x_a, x_b, unequal(x_a, x_b))]

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
        embedding_model = setting(params, 'embedding_model')
        if embedding_model in ('cnn2d', 'cnn'):
            raise ValueError(
                f"split_method='{split_method}' produces irregularly-shaped triangular "
                f"pixel subsets that cannot be represented as rectangular (N, C, H, W) "
                f"tensors. embedding_model='{embedding_model}' requires rectangular 2-D spatial "
                "input. Use embedding_model='mlp' for geometric diagonal splits."
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

def _log_grid(low: int, high: int, n: int) -> List[int]:
    """About ``n`` integers from ``low`` to ``high``, evenly spaced on a log scale."""
    if high <= low:
        return [int(low)]
    return sorted({int(round(v)) for v in np.geomspace(low, high, n)})


def _default_grid(pr: float, *, grid_reach: float, grid_small: int, grid_medium: int,
                  grid_points: int) -> List[int]:
    """The first grid of k, reaching ``grid_reach`` times the reference fit's participation ratio: every k
    up to ``grid_small`` or ``grid_medium``, or ``grid_points`` values on a log scale beyond that, with a
    message when it goes past ``grid_small``. The ratio can fall below the true dimension, and the reach
    leaves room for that."""
    reach = grid_reach * pr
    if reach <= grid_small:
        return list(range(1, grid_small + 1))
    if reach <= grid_medium:
        logger.info(
            f"Dimensionality: the grid runs over embedding_dim 1 to {grid_medium} because the "
            f"reference fit has a participation ratio of {pr:.1f}. Pass embedding_dims to choose "
            f"the grid.")
        return list(range(1, grid_medium + 1))
    top = int(math.ceil(reach))
    grid = _log_grid(1, top, grid_points)
    logger.info(
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
        for task_x, task_y, task_params, task_id in sweep._prepare_tasks(grid, windows_in_task=False):
            # A restart is a repeat (its run_id), and each split and k is a role
            # of its own, so every fit draws independently (neural_mi/seeds.py).
            task_params = dict(task_params, split_id=split_id,
                               _role=f"split{split_id}dim{task_params['embedding_dim']}")
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


def _confirmed(best: Dict[int, float], order: Sequence[int], threshold: float,
               confirm_steps: int) -> Optional[int]:
    """The first k of ``confirm_steps`` consecutive grid values at the threshold, if any."""
    run = 0
    for i, k in enumerate(order):
        run = run + 1 if best.get(k, -math.inf) >= threshold else 0
        if run == confirm_steps:
            return order[i - confirm_steps + 1]
    return None


def _run_grid(views, grid: List[int], n_restarts: int, n_workers: int, ratio: float,
              fits: List[Dict[str, Any]], early_stop: bool,
              confirm_steps: int) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]]]:
    """Fit the grid from small k upward, ``confirm_steps`` values at a time, and return the new fits with
    the values that each split fitted. With ``early_stop``, a split stops once ``confirm_steps``
    consecutive values reach the threshold, so its fits are the ones a call with that split alone makes."""
    if not early_stop:
        return _fit(views, grid, n_restarts, n_workers), {v[0]: list(grid) for v in views}
    new, done, stops, active = [], {v[0]: [] for v in views}, {}, list(views)
    for start in range(0, len(grid), confirm_steps):
        wave = grid[start:start + confirm_steps]
        new += _fit(active, wave, n_restarts, n_workers)
        seen, still = fits + new, []
        for view in active:
            split_id = view[0]
            done[split_id] += wave
            best = _best(seen, split_id)
            if _confirmed(best, done[split_id], ratio * max(best.values()), confirm_steps) is None:
                still.append(view)
            elif len(done[split_id]) < len(grid):
                stops[split_id] = done[split_id][-1]
        active = still
        if not active:
            break
    if stops:
        if len(set(stops.values())) == 1:
            k = next(iter(stops.values()))
            where, rest = f"embedding_dim={k}", f"{grid[grid.index(k) + 1]} to {grid[-1]}"
        else:
            where = "embedding_dim " + ", ".join(f"{k} (split {s})" for s, k in sorted(stops.items()))
            rest = f"up to {grid[-1]}"
        logger.info(
            f"Dimensionality: the grid stopped at {where} once {confirm_steps} consecutive values "
            f"reached {ratio:.0%} of the plateau. The rest of the grid ({rest}) was not fitted there. "
            f"Pass embedding_dims to fit every value."
        )
    return new, done


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def mode_defaults(params: Dict[str, Any], user_set: set, has_y: bool) -> Dict[str, Any]:
    """The settings mode='dimensionality' sets itself where the caller left them unset, declared once
    (analysis/modes.py, MODES) and applied wherever a configuration's settings are resolved: by run() for
    the call and here for each point of a grid, whose keys count as set."""
    out = {}
    if 'critic_type' not in user_set:
        out['critic_type'] = 'hybrid'
    if out.get('critic_type', params.get('critic_type')) == 'hybrid' and setting(params, 'norm_layer') == 'auto':
        # Layer norm stops the fits at k = d from stalling. It also divides out
        # each sample's scale. The reading is taken against a plateau measured
        # the same way.
        out['norm_layer'] = 'layer'
    if 'n_epochs' not in user_set:
        out['n_epochs'] = 500
    if 'patience' not in user_set:
        out['patience'] = 50
    if 'shared_encoder' not in user_set:
        # Two halves of one recording can share an encoder; X and Y need not.
        out['shared_encoder'] = not has_y
    return out


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
    grid_reach: float = 2.0,
    refit_reach: float = 4.0,
    reference_fill: float = 0.75,
    grid_small: int = 10,
    grid_medium: int = 20,
    grid_points: int = 10,
    confirm_steps: int = 3,
    restart_spread: float = 0.1,
    held_out_gap: float = 0.1,
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
        from the reference fit, and each split stops once consecutive values
        confirm its reported dimension.
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
        than one. No two of its splits divide the channels the same way.
    ceiling_mi_fraction : float, default=0.85
        Warn when the reference MI is this close to its evaluation ceiling.
    grid_reach, grid_small, grid_medium, grid_points : default=2.0, 10, 20, 10
        The default grid reaches ``grid_reach`` times the reference fit's
        participation ratio: every value up to ``grid_small`` or ``grid_medium``,
        or ``grid_points`` values on a log scale beyond that.
    reference_fill, refit_reach : float, default=0.75, 4.0
        When ``grid_reach`` times the reference fit's participation ratio reaches
        ``reference_fill`` of its embedding dimension, the reference is refitted
        at ``refit_reach`` times the ratio. The product of the two must exceed
        ``grid_reach``, so that the refitted reference is larger.
    confirm_steps : int, default=3
        Each split's default grid stops once this many consecutive values reach
        the threshold. The call warns when fewer values confirm a reported
        dimension.
    restart_spread, held_out_gap : float, default=0.1, 0.1
        Warn when restarts below the reported dimension differ by more than
        ``restart_spread`` of the plateau, and when the held-out MI of the
        reference networks falls short of their training MI by more than
        ``held_out_gap`` of it.
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
        ``n_restarts`` and ``stopped_early`` (whether the grid of any split
        stopped before its last value).
    """
    params = dict(base_params)
    user_set = set(user_set_keys) if user_set_keys is not None else set(base_params)

    # The hybrid critic is this mode's default. A separable critic works with a
    # warning. run() refuses a critic without an embedding of each side before
    # any work (MODES['dimensionality'].reads_embeddings).
    if 'critic_type' in user_set and params.get('critic_type') == 'separable':
        warnings.warn(
            "mode='dimensionality' with critic_type='separable' overstates the dimension. A "
            "dot-product critic needs more embedding dimensions than the shared structure has. "
            "This mode defaults to 'hybrid'.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    params.update(mode_defaults(params, user_set, y_data is not None))

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
    if reference_dim is None and refit_reach * reference_fill <= grid_reach:
        raise ValueError(
            f"Dimensionality(refit_reach={refit_reach:g}) times reference_fill={reference_fill:g} must "
            f"exceed grid_reach={grid_reach:g}. With a smaller product, a refitted reference can be "
            f"smaller than the one it replaces. Raise refit_reach or reference_fill, or set reference_dim."
        )

    # The two views of every split.
    if y_data is not None:
        pairs = [(x_data, y_data, params)]
    else:
        pairs = _halves(x_data, params, split_method, n_splits or 5, kwargs, 'shared_encoder' in user_set,
                        splits_chosen=n_splits is not None)
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
    if not reference_given and np.isfinite(pr) and grid_reach * pr >= reference_fill * reference_dim:
        larger = int(math.ceil(refit_reach * pr))
        logger.info(
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
        grid = [k for k in _default_grid(pr if np.isfinite(pr) else grid_small / grid_reach,
                                         grid_reach=grid_reach, grid_small=grid_small,
                                         grid_medium=grid_medium, grid_points=grid_points)
                if k < reference_dim]
        early_stop = True
    new, fitted = _run_grid(views, grid, n_restarts, n_workers, saturation_ratio, fits, early_stop,
                            confirm_steps)
    fits += new
    planned = {s: list(grid) for s in split_ids}

    # 3. A default grid whose curve never reached the threshold is extended once, only for the splits
    # whose curve did not reach it. The fits of every other split stay those of a call with that split alone.
    unsaturated = [s for s in split_ids
                   if _reading(_best(fits, s), saturation_ratio)[0] == reference_dim]
    if unsaturated and embedding_dims is None and grid and grid[-1] + 1 < reference_dim:
        extra = [k for k in _log_grid(grid[-1] + 1, reference_dim - 1, grid_points) if k not in grid]
        logger.info(
            f"Dimensionality: the curve of split(s) {unsaturated} did not reach "
            f"{saturation_ratio:.0%} of its plateau by embedding_dim={grid[-1]}. Their grid is "
            f"extended over {extra}."
        )
        new, more = _run_grid([v for v in views if v[0] in unsaturated], extra, n_restarts, n_workers,
                              saturation_ratio, fits, True, confirm_steps)
        fits += new
        for s in unsaturated:
            fitted[s] += more[s]
            planned[s] += extra
    stopped_early = any(len(fitted[s]) < len(planned[s]) for s in split_ids)

    # 4. The readings.
    readings, plateaus = {}, {}
    for s in split_ids:
        readings[s], plateaus[s] = _reading(_best(fits, s), saturation_ratio)
    reached = [k for k in readings.values() if k is not None]
    empty = [s for s, k in readings.items() if k is None]
    if empty:
        warnings.warn(
            f"Dimensionality: the views of split(s) {empty} share no information the estimator "
            f"can find. Their curves stay at 0, so they have no reported dimension.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    below_reference = [s for s, k in readings.items() if k is not None and k >= reference_dim]
    if below_reference:
        warnings.warn(
            f"Dimensionality: the curve of split(s) {below_reference} reached {saturation_ratio:.0%} "
            f"of its plateau only at the reference embedding_dim={reference_dim}. The reported dimension "
            f"then equals the reference and bounds nothing below it. Pass embedding_dims that reach closer to "
            f"the reference or a larger reference_dim.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    _warn_if_unconfirmed(fits, fitted, readings, plateaus, reference_dim, saturation_ratio, confirm_steps)
    _warn_about_fits(fits, reference_fits, readings, plateaus, params, restart_spread, held_out_gap)
    _warn_if_near_ceiling(reference_fits, ceiling_mi_fraction, base_params)

    info = {
        'dimension_at_most': _median_reading(reached),
        'dimension_at_most_std': float(np.std(reached, ddof=1)) if len(reached) > 1 else None,
        'dimension_at_most_per_split': readings,
        'plateau': plateaus,
        'embedding_dims': sorted(set().union(*fitted.values()) | {reference_dim}),
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


def _warn_if_unconfirmed(fits, fitted: Dict[int, List[int]], readings, plateaus, reference_dim: int,
                         ratio: float, confirm_steps: int) -> None:
    """Warn when the grid values from a reported dimension on do not confirm it: the best restart at one
    of the next values falls below the threshold again, or the grid ends before ``confirm_steps`` values."""
    falls, short = [], []
    for s, k in readings.items():
        if k is None or k >= reference_dim:
            continue
        best = _best(fits, s)
        after = [q for q in sorted(fitted[s]) if q >= k][:confirm_steps]
        if any(best.get(q, -math.inf) < ratio * plateaus[s] for q in after):
            falls.append(s)
        elif len(after) < confirm_steps:
            short.append(s)
    if falls:
        warnings.warn(
            f"Dimensionality: for split(s) {falls}, the best restart at the reported dimension reaches "
            f"{ratio:.0%} of the plateau, but at one of the next {confirm_steps - 1} values of "
            f"embedding_dim the best restart falls below that share again. One restart that came out "
            f"high may have set the reported dimension. Raise n_restarts, and check the curve with "
            f"result.plot().",
            UserWarning, stacklevel=user_stacklevel(),
        )
    if short:
        warnings.warn(
            f"Dimensionality: the reported dimension of split(s) {short} is among the last "
            f"{confirm_steps - 1} values of the grid, so fewer than "
            f"Dimensionality(confirm_steps={confirm_steps}) values in a row confirm it. Pass "
            f"embedding_dims that reach further beyond it.",
            UserWarning, stacklevel=user_stacklevel(),
        )


def _warn_about_fits(fits, reference_fits, readings, plateaus, params, restart_spread: float,
                     held_out_gap: float) -> None:
    """The warnings about the fits behind a reading."""
    scale, units = mi_report_units(params)
    # Restarts that disagree below the reported dimension.
    spread = []
    for s, reading in readings.items():
        if reading is None:
            continue
        for k in sorted({int(f['embedding_dim']) for f in fits if f['split_id'] == s and f['embedding_dim'] < reading}):
            values = [f['train_mi'] for f in fits
                      if f['split_id'] == s and f['embedding_dim'] == k and f.get('train_mi') is not None]
            if len(values) > 1 and max(values) - min(values) > restart_spread * plateaus[s]:
                spread.append(k)
    if spread:
        warnings.warn(
            f"Dimensionality: the restarts at embedding_dim {sorted(set(spread))} below the reported "
            f"dimension differ by more than Dimensionality(restart_spread={restart_spread:g}) of the plateau. The curve "
            f"uses the best restart at each value. Raise n_restarts if the best one may also have settled short "
            f"of a direction.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    # Reference networks whose held-out MI lies well below their training MI: too few samples for the
    # information per dimension, and the reported dimension may be too high. The two medians
    # describe the same networks.
    for s in plateaus:
        reference = [f for f in reference_fits if f['split_id'] == s]
        held_out = _median([f.get('test_mi') for f in reference])
        trained = _median([f.get('train_mi') for f in reference])
        if trained > 0 and np.isfinite(held_out) and held_out < (1 - held_out_gap) * trained:
            warnings.warn(
                f"Dimensionality: at the reference fit the held-out MI ({held_out * scale:.3f} {units}) "
                f"falls short of the training MI of the same networks ({trained * scale:.3f} {units}) "
                f"by more than Dimensionality(held_out_gap={held_out_gap:g}) of it. "
                f"The data hold few samples for the information each dimension carries. The reported "
                f"dimension may then be too high, although it still bounds the dimension from above. "
                f"More data make it tighter.",
                UserWarning, stacklevel=user_stacklevel(),
            )
            break
