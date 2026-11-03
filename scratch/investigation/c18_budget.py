"""C18: the epoch budget under window shifting, measured on oracles with exact answers.

Every run trains for N_EPOCHS with early stopping off and records each network's held-out curve and its
train-side curve on the subset the reported value is read from (eval_train=True). Training is the same
epoch for epoch whatever the budget, so what a shorter budget, or a patience rule, would have reported is
replayed from the curves with the library's own smoothing and epoch rule.

    python c18_budget.py <repo> <out.json> [smoke]
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

REPO = sys.argv[1]
sys.path.insert(0, REPO)
N_EPOCHS = 300
SEEDS = (0, 1, 2)


def _oracles():
    from neural_mi.generators.oracle import LatentGaussian
    out = {}
    for name, dims, d, phi, coupling, w in (('block_2d', {'x': 2, 'y': 2}, 1, 0.9, 1.0, 10),
                                            ('block_8d', {'x': 8, 'y': 8}, 2, 0.9, 0.2, 10),
                                            ('block_slow', {'x': 4, 'y': 4}, 2, 0.97, 0.15, 20)):
        model = LatentGaussian.shared(dims=dims, d=d, phi=phi, coupling=coupling, seed=0)
        out[name] = dict(kind='block', model=model, window=w, n_rows=3000, exact=model.block_mi(w))
    model = LatentGaussian.shared(dims={'x': 4, 'y': 4}, d=2, phi=0.5, coupling=1.0, seed=0)
    h = half = 5
    exact = model.exact(A=[('x', s) for s in range(-half, half + 1)], B=[('y', 0)],
                        C=[('y', s) for s in range(-h, 0)])
    out['mi_rate_stride4'] = dict(kind='mi_rate', model=model, h=h, half=half, stride=4, n_rows=3000,
                                  exact=exact)
    out['spike_count'] = dict(kind='spike', n_rows=3000, exact=None)
    return out


def _run(task):
    name, shift, seed, n_epochs = task
    import torch
    torch.set_num_threads(2)
    import neural_mi as nmi
    spec = _oracles()[name]
    training = nmi.Training(n_epochs=n_epochs, eval_train=True, shift_windows=shift)
    common = dict(training=training, seed=seed, device='cpu', show_progress=False, verbose=False)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if spec['kind'] == 'block':
            w = spec['window']
            data = spec['model'].sample(spec['n_rows'] * w, seed=100 + seed)
            p = {'window_size': w, 'step_size': w}
            result = nmi.run(data['x'], data['y'], mode='estimate',
                             processing=nmi.Processing(x='continuous', x_params=p, y='continuous', y_params=p),
                             **common)
            exact = spec['exact']
        elif spec['kind'] == 'mi_rate':
            data = spec['model'].sample(spec['n_rows'] * spec['stride'] + 2 * spec['half'] + 1, seed=100 + seed)
            result = nmi.mi_rate(data['x'], data['y'], h=spec['h'], half_width=spec['half'],
                                 stride=spec['stride'], **common)
            exact = spec['exact']
        else:
            x, y, exact = nmi.generators.generate_spike_pair(n_windows=spec['n_rows'], window_size=1.0,
                                                              coding='count', seed=100 + seed)
            p = {'window_size': 1.0}
            result = nmi.run(x, y, mode='estimate',
                             processing=nmi.Processing(x='spike', x_params=p, y='spike', y_params=p),
                             **common)
    seconds = time.time() - t0
    if spec['kind'] == 'mi_rate':
        nets = result.details[0]['trainings']
    else:
        nets = result.runs
    records = []
    for _, row in nets.iterrows():
        records.append({k: (list(map(float, row[k])) if k.endswith('_history') else
                            (float(row[k]) if isinstance(row[k], (int, float, np.floating, np.integer)) else
                             (None if row[k] is None else str(row[k]))))
                        for k in row.index if k in ('test_mi_history', 'train_mi_history', 'best_epoch',
                                                     'peak_epoch', 'curve_diagnosis', 'train_mi', 'test_mi',
                                                     'term', 'label', 'component', 'sign', 'eval_size')})
    return dict(name=name, shift=shift, seed=seed, exact=float(exact), mi_estimate=float(result.mi_estimate),
                seconds=seconds, networks=records, columns=list(nets.columns),
                warnings=sorted({str(w.message)[:120] for w in caught}))


if __name__ == '__main__':
    out_path = sys.argv[2]
    smoke = len(sys.argv) > 3 and sys.argv[3] == 'smoke'
    names = list(_oracles())
    if smoke:
        tasks = [(n, True, 0, 6) for n in names]
    else:
        tasks = [(n, s, seed, N_EPOCHS) for n in names for s in (True, False) for seed in SEEDS]
    with Pool(6) as pool:
        results = []
        for r in pool.imap_unordered(_run, tasks):
            results.append(r)
            print(f"{r['name']:16s} shift={r['shift']!s:5s} seed={r['seed']} {r['seconds']:6.1f}s "
                  f"estimate={r['mi_estimate']:.3f} exact={r['exact']:.3f}", flush=True)
            with open(out_path, 'w') as f:
                json.dump(results, f)
