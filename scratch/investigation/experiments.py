"""Experiments on nmi.mi_rate. Usage: python experiments.py <scratchpad> <experiment> <outfile>"""
import sys
import torch

torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])

import numpy as np
import neural_mi as nmi
from reallike import RealLike
from exact_terms import Oracle
from harness import run_rate, dump

exp, outfile = sys.argv[2], sys.argv[3]
results = {'experiment': exp, 'runs': []}


def record(label, x, y, h, half_width, seeds, truth_fn, **kw):
    for s in seeds:
        recs, msgs = run_rate(x, y, h=h, half_width=half_width, seed=s, **kw)
        for rec in recs:
            hh, ww = int(rec['h']), int(rec['half_width'])
            J, M, d = truth_fn(hh, ww)
            rec.update(label=label, seed=s, J_true=J, M_true=M, rate_true=d, warnings=msgs)
            results['runs'].append(rec)
            print(label, s, {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in rec.items() if k not in ('warnings',)}, flush=True)
        dump(outfile, results)


real = RealLike()
x_raw, y_raw = real.sample(20000, seed=0)
sx, sy = x_raw.std(), y_raw.std()
x_std, y_std = x_raw / sx, y_raw / sy        # MI is invariant to this rescaling
truth_real = real.terms

if exp == 'defaults_seeds':
    # What a user gets: every default, a sweep of h, three seeds.
    record('raw', x_raw, y_raw, [0, 2, 5, 20], 20, [0, 1, 2], truth_real)

elif exp == 'scale':
    # Same data in three unit systems. The true values are identical.
    for label, (xx, yy) in {'zscored': (x_std, y_std), 'raw_x162': (x_raw, y_raw),
                            'volts_x1e-3': (x_std * 1e-3, y_std * 1e-3)}.items():
        record(label, xx, yy, 5, 20, [0, 1, 2], truth_real)

elif exp == 'budget':
    # Standardised data; more epochs, and a larger batch (training ceiling log(512)).
    record('zscored_ep200', x_std, y_std, 5, 20, [0, 1],
           truth_real, training=nmi.Training(n_epochs=200))
    record('zscored_ep200_bs512', x_std, y_std, 5, 20, [0, 1],
           truth_real, training=nmi.Training(n_epochs=200, batch_size=512))

elif exp == 'easy_and_stride':
    # The library's own accuracy-test regime (AR(1) + white noise), where it should work.
    easy = Oracle(0.85, sx=0.5, sy=0.5)
    rng = np.random.default_rng(10)
    T, burn = 20000, 500
    z = np.zeros(T + burn)
    for t in range(1, T + burn):
        z[t] = 0.85 * z[t - 1] + rng.standard_normal()
    z = z[burn:]
    xe = (z + 0.5 * rng.standard_normal(T)).astype(np.float32).reshape(-1, 1)
    ye = (z + 0.5 * rng.standard_normal(T)).astype(np.float32).reshape(-1, 1)
    record('easy_L5', xe, ye, 3, 5, [0, 1, 2], easy.terms)
    record('easy_L20', xe, ye, 3, 20, [0, 1, 2], easy.terms)
    # Realistic data, standardised, thinned rows.
    record('zscored_stride10', x_std, y_std, 5, 20, [0, 1], truth_real, stride=10)

print('DONE', flush=True)
