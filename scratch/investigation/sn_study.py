"""Spectral norm on vs off across scenarios with exact MI (bits). Unit-scale inputs, library defaults
otherwise. Usage: python sn_study.py <scratchpad> <group> <outfile>"""
import json, sys, time, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
import neural_mi.utils as U
import neural_mi.models.embeddings as E
from neural_mi.generators.oracle import generate_spike_pair, generate_xor_pair, generate_windowed_multichannel

warnings.simplefilter('ignore')
group, outfile = sys.argv[2], sys.argv[3]
LN2 = np.log(2)


# --- every MLP the library builds follows the flag (heads are untethered today; scratch-only patch) ---
class _MLP(E.MLP):
    FORCE = None

    def __init__(self, *a, **k):
        if _MLP.FORCE is not None:
            k['use_spectral_norm'] = _MLP.FORCE
        super().__init__(*a, **k)


U.MLP = _MLP
E.MLP = _MLP


def zs(a):
    a = np.asarray(a, dtype=np.float64)
    return ((a - a.mean(0)) / a.std(0)).astype(np.float32)


def gauss(n, d, mi_bits, seed):
    rng = np.random.default_rng(seed)
    rho = np.sqrt(1 - np.exp(-2 * mi_bits * LN2 / d))
    x = rng.standard_normal((n, d))
    y = rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, d))
    return x, y


def scen(name, seed):
    """(x, y, exact_bits, run_kwargs) for one scenario; data seeded by `seed` too."""
    if name == 'gauss_low':
        x, y = gauss(8000, 5, 3.0, seed); return zs(x), zs(y), 3.0, {}
    if name == 'gauss_high':
        x, y = gauss(8000, 5, 6.5, seed); return zs(x), zs(y), 6.5, {}
    if name == 'highdim_lowrank':
        rng = np.random.default_rng(100 + seed)
        xs, ys = gauss(2000, 3, 3.0, seed)
        x = np.hstack([xs, rng.standard_normal((2000, 47))]); y = np.hstack([ys, rng.standard_normal((2000, 47))])
        qx, _ = np.linalg.qr(rng.standard_normal((50, 50))); qy, _ = np.linalg.qr(rng.standard_normal((50, 50)))
        return zs(x @ qx), zs(y @ qy), 3.0, {}
    if name == 'nonlinear_heavytail':
        x, y = gauss(8000, 5, 3.0, seed); return zs(np.sinh(1.5 * x)), zs(y ** 3), 3.0, {}
    if name == 'xor':
        x, y, exact = generate_xor_pair(8000, noise=0.1, use_torch=False, seed=seed)
        return zs(x), zs(y), float(exact), {}
    if name in ('spikes_count', 'spikes_timing'):
        coding = name.split('_')[1]
        x, y, exact = generate_spike_pair(n_windows=4000, window_size=0.1, n_neurons=4, coding=coding, seed=seed)
        p = {'window_size': 0.1, 'bin_size': 0.005, 'normalize_bins': False}
        return x, y, float(exact), dict(processing=nmi.Processing(x='spike', y='spike', x_params=p, y_params=p),
                                        training_extra=dict(shift_time=False, shift_windows=False))
    if name == 'windowed_mc':
        x, y, exact = generate_windowed_multichannel(n_windows=4000, n_channels=8, window_size=50, seed=seed)
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        return x / x.std(), y / y.std(), float(exact), {}
    raise ValueError(name)


def one(name, sn, lr, seed, critic='separable'):
    x, y, exact, kw = scen(name, seed)
    extra = kw.pop('training_extra', {})
    _MLP.FORCE = sn
    t = time.time()
    r = nmi.run(x, y, mode='estimate', seed=seed, show_progress=False,
                model=nmi.Model(use_spectral_norm=sn, critic_type=critic),
                training=nmi.Training(learning_rate=lr, **extra), **kw)
    run = r.runs.iloc[0]
    return {'scenario': name + ('' if critic == 'separable' else f'[{critic}]'), 'sn': sn, 'lr': lr, 'seed': seed,
            'reported': float(r.mi_estimate), 'held_out': float(run['test_mi']), 'exact': exact,
            'diag': run['curve_diagnosis'], 'epoch': int(run['best_epoch']), 'seconds': time.time() - t}


GROUPS = {
    'A': ['gauss_low', 'gauss_high'],
    'B': ['highdim_lowrank', 'nonlinear_heavytail', 'xor'],
    'C': ['spikes_count', 'spikes_timing', 'windowed_mc'],
}
rows = []


def save():
    json.dump(rows, open(outfile, 'w'), indent=1)


if group in GROUPS:
    for name in GROUPS[group]:
        for lr in (5e-4, 2e-3):
            for sn in (True, False):
                for seed in (0, 1, 2):
                    rows.append(one(name, sn, lr, seed)); print(rows[-1], flush=True); save()
elif group == 'D':
    # hybrid critic, every MLP (encoders and head) following the flag
    for name in ('gauss_low', 'highdim_lowrank'):
        for sn in (True, False):
            for seed in (0, 1):
                rows.append(one(name, sn, 5e-4, seed, critic='hybrid')); print(rows[-1], flush=True); save()
    # a difference quantity: mi_rate on pre-whitened realistic data (target 0.160 nats)
    from reallike import RealLike
    from whiten_helpers import ar_fit, whiten
    from harness import run_rate
    real = RealLike()
    xr, yr = real.sample(20000, seed=0)
    ax, ay = ar_fit(xr[:, 0].astype(float)), ar_fit(yr[:, 0].astype(float))
    xw = whiten(xr[:, 0].astype(float), ax); yw = whiten(yr[:, 0].astype(float), ay)
    xw, yw = zs(xw)[:, None], zs(yw)[:, None]
    for sn in (True, False):
        for seed in (0, 1):
            _MLP.FORCE = sn
            recs, _ = run_rate(xw, yw, h=5, half_width=20, seed=seed, model=nmi.Model(use_spectral_norm=sn),
                               output=nmi.Output(units='bits'))
            rec = recs[0]
            rows.append({'scenario': 'mi_rate_whitened', 'sn': sn, 'lr': 5e-4, 'seed': seed,
                         'reported': rec['mi_raw'] if rec['mi_raw'] is not None else rec['mi'],
                         'held_out': None, 'exact': 0.160 / LN2,
                         'diag': f"J {rec.get('J_diag')} M {rec.get('M_diag')}", 'J': rec['J'], 'M': rec['M'],
                         'J_test': rec.get('J_test'), 'M_test': rec.get('M_test')})
            print(rows[-1], flush=True); save()
print('DONE', flush=True)
