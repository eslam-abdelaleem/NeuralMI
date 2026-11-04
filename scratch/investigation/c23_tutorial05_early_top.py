"""Which run of tutorial 05 section 4 warned that its curve topped too early, behind C23.

Eslam's full run of 4 Nov printed that warning once, in the cell that trains
the three encoders of section 4 at five sample sizes. The cell prints its table
only at the end, so its output does not say which run raised it. This runs the
notebook's own setup cells (2, 4, 19 and 20) and repeats the cell's seeded
calls one at a time, catching each call's warnings. Its means and spreads match
the notebook's table.

Usage: python scratch/investigation/c23_tutorial05_early_top.py <repo> [out.txt]
"""
import contextlib
import io
import json
import sys
import time
import warnings

EARLY = 'reached its top at epoch'


def main(repo, out_path=None):
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, repo)
    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    notebook = json.load(open(f'{repo}/tutorials/05_Estimator_And_Architecture.ipynb'))
    g = {}
    for i in (2, 4, 19, 20):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][i]['source']), g)
    nmi, np = g['nmi'], g['np']

    hits, means = [], {}
    t0 = time.time()
    for n_samples in g['N_GRID']:
        for name, model in g['ARMS'].items():
            vals = []
            for seed in range(g['N_SEEDS']):
                x_pop, y_pop = g['population'](n_samples, seed)
                with warnings.catch_warnings(record=True) as caught, \
                        contextlib.redirect_stderr(io.StringIO()):
                    warnings.simplefilter('always')
                    r = nmi.run(x_pop, y_pop, mode='estimate', model=model,
                                training=nmi.Training(batch_size=min(128, max(32, n_samples // 4))),
                                split=nmi.Split(mode='random'),
                                seed=seed, n_workers=1, show_progress=False)
                vals.append(float(r.mi_estimate))
                early = [str(w.message) for w in caught if EARLY in str(w.message)]
                say(f'{n_samples:>5} {name:<16} seed {seed}  mi {r.mi_estimate:.3f}  '
                    f'warnings {len(caught)}  early top {"yes" if early else "no"}')
                if early:
                    hits.append(f'{n_samples} samples, {name}, seed {seed}: {early[0][:110]}')
            means[(n_samples, name)] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    say(f'({time.time() - t0:.0f}s)')
    say()
    say("mean +- sd over seeds, as the notebook's table prints them")
    for (n, name), (m, s) in means.items():
        say(f'{n:>5} {name:<16} {m:.3f} +-{s:.3f}')
    say()
    say('runs that warned of an early top:')
    for h in hits:
        say(f'  {h}')
    if out_path:
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
