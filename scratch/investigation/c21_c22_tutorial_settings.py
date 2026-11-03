"""Behind C21 and C22: which settings bring back two tutorial results that the full run on the defaults lost.

A (C22): tutorial 04 section 7 at 90% overlap. Its stored outputs, from a library before the correctness
fixes, read a random split far above the exact block MI and a blocked one a little below it. Each
configuration here is read under both splits, two repeats each, as a ratio to the exact value.

B (C21): tutorial 02 at its starved size of 400 samples. The spread of five repeats (section 4) and the
rigorous ladder (section 5), whose fit `is_reliable` passes.

    python scratch/investigation/c21_c22_tutorial_settings.py <repo>
"""
import sys
import time
import warnings


def main(repo):
    sys.path.insert(0, repo)
    import numpy as np
    import neural_mi as nmi
    from neural_mi.generators.oracle import LatentGaussian
    warnings.simplefilter('ignore')

    old04 = dict(model=nmi.Model(embedding_dim=8, hidden_dim=128, n_layers=2),
                 training=nmi.Training(n_epochs=200, batch_size=256, patience=30, learning_rate=1e-3))
    a = {'defaults': {},
         'defaults, shift_windows=False': dict(training=nmi.Training(shift_windows=False)),
         'defaults, stopping off': dict(training=nmi.Training(n_epochs=200, patience=200)),
         'old model and training': old04,
         'old model, default training': dict(model=old04['model']),
         'default model, old training': dict(training=old04['training']),
         'old model and training, stopping off': dict(
             model=old04['model'],
             training=nmi.Training(n_epochs=200, batch_size=256, patience=200, learning_rate=1e-3))}
    oracle = LatentGaussian.shared(dims={'x': 8, 'y': 8}, d=2, phi=0.9, coupling=0.4, noise=1.0, seed=0)
    s = oracle.sample(T=10_000, seed=0)
    exact = float(oracle.block_mi(10))
    print(f"A. tutorial 04, 90% overlap, exact {exact:.3f} bits: ratio to exact, mean of 2 repeats", flush=True)
    for name, kw in a.items():
        t0 = time.time()
        r = nmi.run(s['x'], s['y'], mode='sweep',
                    sweep_grid={'split_mode': ['blocked', 'random'], 'run_id': [0, 1]},
                    processing=nmi.Processing(x='continuous', x_params={'window_size': 10, 'step_size': 1},
                                              y_params={'window_size': 10, 'step_size': 1}),
                    estimator=nmi.Estimator('infonce'), n_workers=4, seed=0, show_progress=False, **kw)
        df = r.dataframe.set_index('split_mode')
        runs = r.runs
        best = {m: sorted(int(e) for e in runs[runs['split_mode'] == m]['best_epoch'])
                for m in ('blocked', 'random')}
        print(f"  {name:<38} blocked {df.loc['blocked', 'mi_mean'] / exact:.2f}"
              f"  random {df.loc['random', 'mi_mean'] / exact:.2f}  best epochs {best}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    old02 = dict(model=nmi.Model(hidden_dim=256, embedding_dim=32),
                 training=nmi.Training(n_epochs=150, batch_size=128, patience=30, learning_rate=1e-3))
    b = {'defaults': {}, 'old model and training': old02,
         'defaults, learning_rate=1e-3': dict(training=nmi.Training(learning_rate=1e-3))}
    x, y = nmi.generators.generate_nonlinear_from_latent(n_samples=400, latent_dim=10, observed_dim=500,
                                                         mi=4.0, seed=0)
    print("B. tutorial 02 at 400 samples, exact 4.0 bits", flush=True)
    for name, kw in b.items():
        t0 = time.time()
        sp = nmi.run(x, y, mode='sweep', sweep_grid={'run_id': list(range(5))}, split=nmi.Split(mode='random'),
                     estimator=nmi.Estimator('infonce'), n_workers=5, seed=0, show_progress=False, **kw)
        repeats = np.array(sp.runs['mi'])
        rg = nmi.run(x, y, mode='rigorous', split=nmi.Split(mode='random'), estimator=nmi.Estimator('infonce'),
                     n_workers=8, seed=0, show_progress=False, **kw)
        print(f"  {name:<30} repeats mean {repeats.mean():.2f} sd {repeats.std(ddof=1):.2f}"
              f" | rigorous {rg.mi_estimate:.2f} +/- {rg.get('mi_error'):.2f}  r2 {rg.get('r_squared'):.3f}"
              f"  is_reliable {rg.get('is_reliable')}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
