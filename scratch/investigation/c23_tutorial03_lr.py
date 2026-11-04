"""Tutorial 03's networks top within the first few epochs: the learning rate, behind C23.

Runs tutorial 03 section 4's quantities (the nine its quick pass keeps) at the
default learning rate and lower ones, two repeats each, and reports for each
quantity its error against the exact value, its spread, and how often the
library warned that the held-out curve reached its top too early or approached
it slowly.

Usage: python scratch/investigation/c23_tutorial03_lr.py <repo> [out.txt]
"""
import re
import sys
import time
import warnings

EARLY = 'reached its top at epoch'
SLOW = 'approached its top slowly'
COUNT = re.compile(r"^'(.*)' was raised (\d+) times")


def counts(caught):
    """How often each of the two curve warnings was raised: its count line, or 1 when it was shown alone."""
    out = {EARLY: 0, SLOW: 0}
    for kind in out:
        lines = [str(w.message) for w in caught if kind in str(w.message)]
        repeats = [int(COUNT.match(m).group(2)) for m in lines if COUNT.match(m)]
        out[kind] = sum(repeats) if repeats else len(lines)
    return out


def main(repo, out_path=None):
    sys.path.insert(0, repo)
    import neural_mi as nmi
    from neural_mi import Conditional
    from neural_mi.generators import LatentGaussian
    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    K, W_BLOCK = 8, 5
    oracle = LatentGaussian.shared(dims={'x': 4, 'y': 4, 'w': 4}, d=2, phi=0.9, coupling=0.4, noise=1.0, seed=0)
    sample = oracle.sample(T=10_000, seed=0)
    x, y, w = sample['x'], sample['y'], sample['w']

    def offs(name, lo, hi):
        return [(name, s) for s in range(lo, hi + 1)]

    x_past, y_past, w_past = offs('x', -K, -1), offs('y', -K, -1), offs('w', -K, -1)
    x_now, y_now, w_now = [('x', 0)], [('y', 0)], [('w', 0)]
    ii_exact = (oracle.exact(x_now + w_now, y_now) - oracle.exact(x_now, y_now) - oracle.exact(w_now, y_now))

    for lr in (None, 2e-4, 1e-4):
        common = dict(split=nmi.Split(mode='blocked'), estimator=nmi.Estimator('infonce'), show_progress=False,
                      sweep_grid={'run_id': [0, 1]}, n_workers=8, seed=0)
        if lr is not None:
            common['training'] = nmi.Training(learning_rate=lr)
        quantities = [
            ('instantaneous MI', oracle.exact(x_now, y_now), lambda: nmi.instantaneous_mi(x, y, **common)),
            (f'block MI (w={W_BLOCK})', oracle.block_mi(W_BLOCK),
             lambda: nmi.block_mi(x, y, window_size=W_BLOCK, **common)),
            ('active information storage', oracle.exact(x_past, x_now),
             lambda: nmi.active_information_storage(x, k=K, **common)),
            ('predictive information', oracle.exact(x_past, offs('x', 0, K - 1)),
             lambda: nmi.predictive_information(x, k=K, **common)),
            ('cross-predictive information', oracle.exact(x_past, offs('y', 0, K - 1)),
             lambda: nmi.cross_predictive_information(x, y, k=K, **common)),
            ('conditional MI', oracle.exact(x_now, y_now, w_now),
             lambda: nmi.run(x, y, mode='conditional', conditional=Conditional(w_data=w), **common)),
            ('transfer entropy', oracle.exact(x_past, y_now, y_past),
             lambda: nmi.transfer_entropy(x, y, history_window=K, **common)),
            ('conditional TE', oracle.exact(x_past, y_now, y_past + w_past),
             lambda: nmi.conditional_transfer_entropy(x, y, w, history_window=K, **common)),
            ('interaction information', ii_exact, lambda: nmi.interaction_information(x, y, w, **common)),
        ]
        say(f"learning_rate {'default' if lr is None else lr}")
        t_all, early, slow = time.time(), 0, 0
        for name, exact, call in quantities:
            t0 = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                row = call().dataframe.iloc[0]
            c = counts(caught)
            early, slow = early + c[EARLY], slow + c[SLOW]
            m, s = float(row['mi_mean']), float(row['mi_std'])
            say(f"  {name:<30} exact {float(exact):+.4f}  est {m:+.4f}  err {100 * (m - float(exact)) / abs(float(exact)):+6.1f}%"
                f"  sd {s:.4f}  early tops {c[EARLY]}  slow approaches {c[SLOW]}  ({time.time() - t0:.0f}s)")
        say(f"  in all: early tops {early}, slow approaches {slow}  ({time.time() - t_all:.0f}s)")
    if out_path:
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
