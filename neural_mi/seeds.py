# neural_mi/seeds.py
"""Every seed of a call, from the call's seed, the repeat and the role.

A repeat is a value of ``run_id``. A call without one has the repeat 0. A role
names one network or one draw inside an estimate: a rung of the ladder by its
gamma and chunk, a fit of the dimensionality curve by its channel split and
width, a channel split itself, the held-out rows of a dimensionality call, the
noise mode='precision' adds, a permutation trial. A plain network and the terms
of a difference have no role. Nothing else enters a seed: not a
configuration's settings, not where a task sits in a grid, not the route that
runs it. Every configuration therefore runs a repeat from the same seeds,
whether it runs alone or in a grid, as a setting or as a grid value, in one
call or in one call per processor setting.
"""
import hashlib
from typing import Any, Optional

import numpy as np

_BOUND = 2 ** 31


def _mixed(seed: int, key: str) -> int:
    return (int(seed) + int(hashlib.md5(key.encode()).hexdigest(), 16)) % _BOUND


def repeat_seed(seed: Optional[int], run_id: Any = 0) -> Optional[int]:
    """The seed every network of one repeat shares: the shifts of its windows at
    each epoch and the order the ladder deals its rows into chunks. None without
    a seed."""
    return None if seed is None else _mixed(seed, f"repeat{run_id}")


def network_seed(seed: Optional[int], run_id: Any = 0, role: str = '') -> Optional[int]:
    """The seed of one network or one draw of a repeat: a network's
    initialisation, batch order and dropout, its split and its evaluation rows.
    The terms of a difference have no role, so they share it and read the same
    rows. None without a seed."""
    base = repeat_seed(seed, run_id)
    return None if base is None else _mixed(base, f"network{role}")


def generator(seed: Optional[int], run_id: Any = 0, role: str = '') -> np.random.Generator:
    """A NumPy generator for one draw made outside a training task."""
    return np.random.default_rng(network_seed(seed, run_id, role))
