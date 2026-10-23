# neural_mi/estimators/ksg.py
"""The Kraskov-Stögbauer-Grassberger (KSG) estimator of mutual information, and its conditional form.

Neither trains anything. Both read the information from the distances between
neighbouring rows: for each row, the distance to its k-th neighbour in the joint
space, and how many rows fall within that distance in each marginal space
(Kraskov, Stögbauer and Grassberger, 2004, their first algorithm). The
conditional form estimates ``I(X; Y | Z)`` from one neighbour search in the
space of all three (Frenzel and Pompe, 2007), so the biases of the terms cancel
inside the estimate.

Distances use the max norm. Before the search every column is scaled to unit
variance and given tie-breaking noise of ``NOISE`` standard deviations, drawn
from a seeded generator so a seeded call is reproducible. With
``gaussianize=True`` each column is then mapped through its ranks to a standard
normal (Holmes and Nemenman, 2019): the information is unchanged, and KSG's
bias falls on skewed and heavy-tailed data.

Rows of a time series are not independent. Rows a few steps apart resemble each
other, and the nearest neighbours of a row are then its own neighbours in time,
which reads as dependence. Rows within ``exclude`` steps of each other are
therefore never counted as neighbours (Theiler, 1986). ``exclude='auto'`` takes
the autocorrelation time of the rows: the first lag at which the correlation of
each column with itself that many rows later, averaged over the columns, falls
below 1/e.

The implementations checked against are ``bmi`` (Czyż et al., 2023), JIDT
(Lizier, 2014) and scikit-learn's ``mutual_info_regression``.
"""
from typing import Optional, Union

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import norm, rankdata

NOISE = 1e-8              # tie-breaking noise, in standard deviations of each column
KD_TREE_MAX_DIM = 20      # above this many dimensions comparing every pair is faster than a k-d tree
_CHUNK = 256              # rows compared with every row at once on the brute-force path


def prepare(a, gaussianize: bool, rng: np.random.Generator) -> np.ndarray:
    """Rows of ``a`` flattened to columns, each scaled to unit variance with tie-breaking noise, and with
    ``gaussianize`` mapped through its ranks to a standard normal."""
    a = np.asarray(a, dtype=float).reshape(len(a), -1)
    sd = a.std(axis=0)
    a = (a - a.mean(axis=0)) / np.where(sd > 0, sd, 1.0) + NOISE * rng.standard_normal(a.shape)
    if gaussianize:
        a = norm.ppf((rankdata(a, axis=0) - 0.5) / len(a))
    return a


def autocorrelation_time(*arrays) -> int:
    """Steps between rows below which they still resemble each other, 0 for independent rows.

    The first lag at which the correlation of each column with itself that many
    rows later, averaged over the columns, falls below 1/e; the value returned is
    one less, the largest separation still excluded.
    """
    a = np.hstack([np.asarray(x, dtype=float).reshape(len(x), -1) for x in arrays])
    a = a - a.mean(axis=0)
    var = (a ** 2).mean(axis=0)
    a = a[:, var > 0] / np.sqrt(var[var > 0])
    n = len(a)
    for lag in range(1, n // 2):
        if np.mean((a[:-lag] * a[lag:]).sum(axis=0) / (n - lag)) < np.exp(-1):
            return lag - 1
    return n // 2


def _resolve_exclude(exclude, arrays) -> int:
    return autocorrelation_time(*arrays) if exclude == 'auto' else int(exclude)


def _kth_distance(points: np.ndarray, k: int, exclude: int) -> np.ndarray:
    """Max-norm distance from each row to its k-th nearest row more than ``exclude`` rows away in time, by k-d tree."""
    n = len(points)
    rows = np.arange(n)
    tree = cKDTree(points)
    n_query = min(n, k + 1 + 2 * exclude)
    while True:
        dist, idx = tree.query(points, k=n_query, p=np.inf)
        kept = np.abs(idx - rows[:, None]) > exclude
        position = np.argmax(np.cumsum(kept, axis=1) >= k, axis=1)
        found = kept.sum(axis=1) >= k
        if found.all() or n_query == n:
            return dist[rows, position]
        n_query = min(n, 2 * n_query)


def _count_within(points: np.ndarray, radius: np.ndarray, exclude: int) -> np.ndarray:
    """For each row, the rows strictly closer than its ``radius`` in the max norm, leaving out every row
    within ``exclude`` rows of it in time, itself included; by k-d tree."""
    n = len(points)
    rows = np.arange(n)
    counts = cKDTree(points).query_ball_point(points, np.nextafter(radius, 0), p=np.inf, return_length=True)
    for shift in range(-exclude, exclude + 1):           # take back the neighbours in time the search counted
        i = rows[max(0, -shift):n - max(0, shift)]
        close = np.abs(points[i] - points[i + shift]).max(axis=1) < radius[i]
        np.subtract.at(counts, i[close], 1)
    return counts


def _neighbours(spaces, groups, k: int, exclude: int):
    """The distance from each row to its k-th neighbour in the joint space (``spaces`` side by side), and for each
    group of spaces the rows strictly closer than that distance in the group's own space. Rows within ``exclude``
    rows of each other in time are never neighbours. Returns the distances and one array of counts per group.

    With few dimensions in all, k-d trees answer each search. Otherwise every row is compared with every other, once:
    in the max norm a distance over several spaces is the largest of their distances, so one pass over the rows
    gives the k-th distance and every count."""
    joint = np.hstack(spaces)
    if joint.shape[1] <= KD_TREE_MAX_DIM:
        eps = _kth_distance(joint, k, exclude)
        return eps, [_count_within(np.hstack([spaces[i] for i in g]), eps, exclude) for g in groups]
    n = len(joint)
    rows = np.arange(n)
    data = [torch.from_numpy(a) for a in spaces]
    eps, counts = np.empty(n), [np.empty(n, dtype=int) for _ in groups]

    def largest(dists):
        out = dists[0].clone()
        for d in dists[1:]:
            torch.maximum(out, d, out=out)
        return out

    for start in range(0, n, _CHUNK):
        block = rows[start:start + _CHUNK]
        near = torch.from_numpy(np.abs(block[:, None] - rows[None, :]) <= exclude)
        dist = [torch.cdist(d[block], d, p=float('inf')).masked_fill_(near, float('inf')) for d in data]
        e = torch.kthvalue(largest(dist), k, dim=1).values
        eps[block] = e.numpy()
        for c, g in zip(counts, groups):
            c[block] = (largest([dist[i] for i in g]) < e[:, None]).sum(dim=1).numpy()
    return eps, counts


def _candidates(n: int, exclude: int) -> np.ndarray:
    """For each row, how many rows can be its neighbour: every row more than ``exclude`` rows away in time."""
    rows = np.arange(n)
    return n - 1 - (np.minimum(rows, exclude) + np.minimum(n - 1 - rows, exclude))


def ksg_mi(x, y, k: int = 4, gaussianize: bool = True, exclude: Union[int, str] = 'auto',
           seed: Optional[int] = 0) -> float:
    """KSG estimate of ``I(X; Y)`` in nats, one row per observation.

    ``I = psi(k) + <psi(M + 1)> - <psi(n_x + 1) + psi(n_y + 1)>``, where ``n_x``
    and ``n_y`` count the rows strictly within the distance to the k-th joint
    neighbour in each marginal space, and ``M`` is the number of rows a row can
    have as neighbours (``N - 1`` without an exclusion window).
    """
    rng = np.random.default_rng(seed)
    w = _resolve_exclude(exclude, (x, y))
    x, y = prepare(x, gaussianize, rng), prepare(y, gaussianize, rng)
    eps, (n_x, n_y) = _neighbours([x, y], [(0,), (1,)], k, w)
    return float(digamma(k) + digamma(_candidates(len(x), w) + 1).mean()
                 - (digamma(n_x + 1) + digamma(n_y + 1)).mean())


def ksg_cmi(x, y, z, k: int = 4, gaussianize: bool = True, exclude: Union[int, str] = 'auto',
            seed: Optional[int] = 0) -> float:
    """Frenzel-Pompe estimate of ``I(X; Y | Z)`` in nats, one row per observation.

    ``I = psi(k) - <psi(n_xz + 1) + psi(n_yz + 1) - psi(n_z + 1)>``, with the
    counts taken within the distance to the k-th neighbour in the space of all
    three.
    """
    rng = np.random.default_rng(seed)
    w = _resolve_exclude(exclude, (x, y, z))
    x, y, z = (prepare(a, gaussianize, rng) for a in (x, y, z))
    eps, (n_xz, n_yz, n_z) = _neighbours([x, y, z], [(0, 2), (1, 2), (2,)], k, w)
    return float(digamma(k) - (digamma(n_xz + 1) + digamma(n_yz + 1) - digamma(n_z + 1)).mean())
