# tests/test_seeds.py
"""Every seed comes from the call's seed, the repeat and the role (neural_mi/seeds.py).

A configuration then reads the same number whether it runs alone or in a grid,
in any order, as a setting or as a grid value, and in one call per processor
setting. Each repeat draws its own seeds, the shifts of its windows included.
The terms of a difference share theirs, so they read the same rows, and the
rungs of a ladder draw apart from one another.
"""
import numpy as np
import torch

import neural_mi as nmi
import neural_mi.analysis.task as task
from neural_mi import Conditional, Lag, Model, Rigorous, Split, Training
from neural_mi.analysis.dimensionality import _halves
from neural_mi.analysis.rigorous import _Chunker
from neural_mi.data.corruption import corrupt
from neural_mi.seeds import network_seed, repeat_seed

_KW = dict(model=Model(embedding_dim=8, hidden_dim=16), training=Training(n_epochs=3, batch_size=64),
           split=Split(mode='random'), seed=3, n_workers=1, show_progress=False)


def _pair(n=300, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, channels)).astype(np.float32)
    return x, (x + 0.3 * rng.normal(size=(n, channels))).astype(np.float32)


def _recorded(monkeypatch):
    """The settings of every training task and the two seeds it drew, in the order the tasks ran.
    A task that only times training steps for the choice of device is left out."""
    seen, real = [], task._seeds

    def recording(params):
        seeds = real(params)
        if not params.get('_time_steps'):
            seen.append({'params': params, 'network': seeds[0], 'repeat': seeds[1]})
        return seeds

    monkeypatch.setattr(task, '_seeds', recording)
    return seen


def test_a_seed_depends_on_the_call_seed_the_repeat_and_the_role_alone():
    assert network_seed(3, 0, '') == network_seed(3, 0, '')
    assert network_seed(3, 0, '') != network_seed(3, 1, '')
    assert network_seed(3, 0, '') != network_seed(4, 0, '')
    assert network_seed(3, 0, 'gamma2chunk0') != network_seed(3, 0, 'gamma2chunk1')
    # A network never draws what its repeat's shifts draw.
    assert network_seed(3, 0, '') != repeat_seed(3, 0)
    assert network_seed(None, 0, '') is None and repeat_seed(None, 0) is None


def test_a_configuration_reads_the_same_alone_and_in_a_grid_in_any_order():
    x, y = _pair()
    alone = nmi.run(x, y, mode='estimate', **_KW)
    grid = nmi.run(x, y, mode='sweep', sweep_grid={'embedding_dim': [4, 8]}, **_KW)
    flipped = nmi.run(x, y, mode='sweep', sweep_grid={'embedding_dim': [8, 4]}, **_KW)

    def value(result, k):
        return result.runs.loc[result.runs['embedding_dim'] == k, 'mi'].item()

    assert value(grid, 8) == alone.mi_estimate
    assert value(grid, 8) == value(flipped, 8)
    assert value(grid, 4) == value(flipped, 4)


def test_a_processor_setting_reads_the_same_in_a_grid_and_alone():
    x, y = _pair(n=800)

    def processing(step):
        params = {'window_size': 4, 'step_size': step}
        return nmi.Processing(x='continuous', x_params=params, y_params=params)

    grid = nmi.run(x, y, mode='sweep', sweep_grid={'step_size': [2, 4]}, processing=processing(2), **_KW)
    alone = nmi.run(x, y, mode='estimate', processing=processing(4), **_KW)
    assert grid.runs.loc[grid.runs['step_size'] == 4, 'mi'].item() == alone.mi_estimate


def test_each_repeat_draws_its_own_seeds_and_the_terms_of_a_repeat_share_them(monkeypatch):
    seen = _recorded(monkeypatch)
    x, y = _pair()
    w = np.random.default_rng(1).normal(size=(len(x), 1)).astype(np.float32)
    nmi.run(x, y, mode='conditional', conditional=Conditional(w_data=w),
            sweep_grid={'run_id': [0, 1]}, **_KW)
    seeds = {}
    for s in seen:
        seeds.setdefault(s['params']['run_id'], set()).add((s['network'], s['repeat']))
    # Two terms per repeat, one pair of seeds between them.
    assert len(seen) == 4 and all(len(pair) == 1 for pair in seeds.values())
    (net0, shift0), = seeds[0]
    (net1, shift1), = seeds[1]
    assert net0 != net1 and shift0 != shift1


def test_lag_seeds_its_networks_alike_with_and_without_run_id(monkeypatch):
    seen = _recorded(monkeypatch)
    x, y = _pair()
    nmi.run(x, y, mode='lag', lag=Lag(lag_range=[0, 2]), **_KW)
    without = [s['network'] for s in seen]
    seen.clear()
    nmi.run(x, y, mode='lag', lag=Lag(lag_range=[0, 2]), sweep_grid={'run_id': [0, 1]}, **_KW)
    with_run_id = [s['network'] for s in seen if s['params'].get('run_id') == 0]
    # Every lag of a repeat starts from the same seeds, whichever way the repeat is named.
    assert len(set(without)) == 1
    assert with_run_id == without


def test_the_rungs_of_a_ladder_draw_apart_and_configurations_share_them(monkeypatch):
    seen = _recorded(monkeypatch)
    x, y = _pair(n=400)
    nmi.run(x, y, mode='rigorous', sweep_grid={'embedding_dim': [4, 8]},
            rigorous=Rigorous(gamma_range=range(1, 3), min_gamma_points=2), **_KW)
    rungs = {}
    for s in seen:
        rungs.setdefault(s['params']['embedding_dim'], []).append((s['params']['_role'], s['network']))
    # gamma 1 and the two chunks at gamma 2, in the same order for each configuration.
    assert [role for role, _ in rungs[4]] == ['gamma1chunk0', 'gamma2chunk0', 'gamma2chunk1']
    assert len({seed for _, seed in rungs[4]}) == 3
    assert rungs[4] == rungs[8]


def test_the_rungs_of_a_difference_ladder_draw_apart_and_their_terms_share(monkeypatch):
    seen = _recorded(monkeypatch)
    x, y = _pair(n=400)
    w = np.random.default_rng(1).normal(size=(len(x), 1)).astype(np.float32)
    nmi.run(x, y, mode='conditional',
            conditional=Conditional(w_data=w, rigorous=True, gamma_range=range(1, 3), min_gamma_points=2),
            **_KW)
    rungs = {}
    for s in seen:
        rungs.setdefault(s['params']['_role'], set()).add(s['network'])
    assert sorted(rungs) == ['gamma1chunk0', 'gamma2chunk0', 'gamma2chunk1']
    # Both terms of a rung train from one seed, and no two rungs share one.
    assert all(len(seeds) == 1 for seeds in rungs.values())
    assert len(set().union(*rungs.values())) == 3


def test_ksg_draws_its_tie_breaking_from_each_repeat(monkeypatch):
    seen = _recorded(monkeypatch)
    x, y = _pair()
    nmi.run(x, y, mode='sweep', estimator='ksg', sweep_grid={'run_id': [0, 1]}, seed=3,
            n_workers=1, show_progress=False)
    assert len({s['network'] for s in seen}) == 2


def test_the_ladder_deals_its_rows_by_repeat_whatever_else_drew_first():
    rows = np.zeros((60, 1), dtype=np.float32)
    chunker = _Chunker(rows, rows, {'random_seed': 3}, temporal_chunking=False)
    first = chunker.order.copy()
    np.random.seed(0)
    np.random.permutation(10)
    chunker.reorder(0)
    assert np.array_equal(chunker.order, first)
    chunker.reorder(1)
    assert not np.array_equal(chunker.order, first)


def test_dimensionality_splits_its_channels_by_split_whatever_else_drew_first():
    x = torch.randn(50, 8)

    def orders():
        halves = _halves(x, {'random_seed': 3}, 'random', 3, {})
        return [a[0].tolist() for a, _, _ in halves]

    np.random.seed(0)
    first = orders()
    np.random.seed(1)
    assert orders() == first
    assert len({tuple(row) for row in first}) == 3


def test_corruption_noise_comes_from_its_generator_alone():
    data = torch.ones(20, 3)
    a = corrupt(data, 0.5, 'noise', generator=torch.Generator().manual_seed(5))
    torch.manual_seed(0)
    torch.rand(7)
    b = corrupt(data, 0.5, 'noise', generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
