"""Usage: python routes_and_sn.py <scratchpad> <experiment> <outfile>

block_tiled    : block_mi at w = 5, 10, 20 on the realistic data (z-scored), default tiling windows
block_step1    : the same with step_size = 1 (overlapping windows)
sn_chain       : mi_rate (h=5, half_width=20) on z-scored data with use_spectral_norm=False
sn_plain       : plain MI, high MI and off-unit scales, spectral norm on vs off
"""
import json, sys, time, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
from reallike import RealLike
from harness import run_rate

warnings.simplefilter('ignore')
exp, outfile = sys.argv[2], sys.argv[3]
out = {'experiment': exp, 'rows': []}


def save():
    json.dump(out, open(outfile, 'w'), indent=1, default=float)


real = RealLike()
x, y = real.sample(20000, seed=0)
x, y = x / x.std(), y / y.std()
exact_block = {5: 0.574, 10: 1.418, 20: 3.102}   # nats, from block_vs_chain.py

if exp in ('block_tiled', 'block_step1'):
    x_params = {} if exp == 'block_tiled' else {'step_size': 1}
    for seed in (0, 1):
        t = time.time()
        r = nmi.block_mi(x, y, window_size=[5, 10, 20], seed=seed, show_progress=False,
                         processing=nmi.Processing(x='continuous', x_params=x_params),
                         output=nmi.Output(units='nats'))
        df = r.dataframe
        vals = {int(w): float(m) for w, m in zip(df['window_size'], df['mi_mean'])}
        tests = {int(w): float(m) for w, m in zip(df['window_size'], df['test_mi_mean'])}
        row = {'seed': seed, 'I': vals, 'test_I': tests,
               'slope_5_10': (vals[10] - vals[5]) / 5, 'slope_10_20': (vals[20] - vals[10]) / 10,
               'n_rows': {int(w): int(n) for w, n in zip(df['window_size'], df.get('n_windows_built', [0] * len(df)))},
               'seconds': time.time() - t}
        out['rows'].append(row)
        print(exp, row, flush=True)
        save()

elif exp == 'sn_chain':
    for seed in (0, 1):
        recs, msgs = run_rate(x, y, h=5, half_width=20, seed=seed,
                              model=nmi.Model(use_spectral_norm=False))
        rec = recs[0]
        J, M, d = real.terms(5, 20)
        rec.update(seed=seed, J_true=J, M_true=M, rate_true=d, warnings=msgs)
        out['rows'].append(rec)
        print(exp, {k: v for k, v in rec.items() if k != 'warnings'}, flush=True)
        save()

elif exp == 'sn_plain':
    rng = np.random.default_rng(0)
    d, n = 5, 8000
    for true_mi, scale in ((4.5, 1.0), (2.0, 1e-3), (2.0, 1e3)):
        rho = np.sqrt(1 - np.exp(-2 * true_mi / d))
        xx = rng.standard_normal((n, d))
        yy = rho * xx + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, d))
        for sn in (True, False):
            r = nmi.run((xx * scale).astype(np.float32), (yy * scale).astype(np.float32), mode='estimate',
                        split=nmi.Split(mode='random'), output=nmi.Output(units='nats'),
                        model=nmi.Model(use_spectral_norm=sn), seed=0, show_progress=False)
            run = r.runs.iloc[0]
            row = {'true_mi': true_mi, 'scale': scale, 'spectral_norm': sn, 'reported': r.mi_estimate,
                   'test_mi': float(run['test_mi']), 'diag': run['curve_diagnosis']}
            out['rows'].append(row)
            print(exp, row, flush=True)
            save()
print('DONE', flush=True)
