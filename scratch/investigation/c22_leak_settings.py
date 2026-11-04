"""Where a random split of overlapping windows still reads high on the current library, behind C22.

Tutorial 04 section 7's random-split inflation came from window shifting, which
moved windows one sample apart by up to a whole window each epoch until
e3c6d3b, so that training windows landed on the held-out ones
(c22_split_bisect.py). This asks where the overlap alone still leaks: a random
split puts each held-out window among training windows that share most of its
samples, which a network can exploit when those neighbours are near-copies as
inputs. Each setting reads a blocked and a random split of windows of 10
samples, one sample apart (90% overlap) and ten apart (none), on the defaults
unless noted, two repeats each, against the exact block MI of the
LatentGaussian pair.

Usage: python scratch/investigation/c22_leak_settings.py <repo> [out.txt] [name,name]

The optional third argument runs only the settings whose names start with one
of its comma-separated names.
"""
import sys
import time
import warnings


def main(repo, out_path=None):
    sys.path.insert(0, repo)
    import neural_mi as nmi
    from neural_mi.generators import LatentGaussian
    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    tutorial = dict(dims={'x': 8, 'y': 8}, d=2, phi=0.9, coupling=0.4, noise=1.0, seed=0)
    settings = [
        ('tutorial 04 data', tutorial, 10_000, {}),
        ('less data (T=2000)', tutorial, 2_000, {}),
        # Neighbouring windows are near-copies as vectors when the signal moves little in one
        # sample and the noise is small. A weak coupling keeps the exact value under the ceiling.
        ('slow latent, little noise', {**tutorial, 'phi': 0.99, 'noise': 0.1, 'coupling': 0.01}, 10_000, {}),
        ('little noise', {**tutorial, 'noise': 0.1, 'coupling': 0.03}, 10_000, {}),
        ('CNN encoder, mean pooling', tutorial, 10_000,
         {'model': nmi.Model(embedding_model='cnn', cnn_pooling='mean')}),
        ('GRU encoder', tutorial, 10_000, {'model': nmi.Model(embedding_model='gru')}),
    ]
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    for name, cfg, T, extra in settings:
        if only and not any(name.startswith(o) for o in only):
            continue
        oracle = LatentGaussian.shared(**cfg)
        sample = oracle.sample(T=T, seed=0)
        exact = float(oracle.block_mi(10))
        t0 = time.time()
        row = []
        for step in (1, 10):
            p = {'window_size': 10, 'step_size': step}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                swept = nmi.run(sample['x'], sample['y'], mode='sweep',
                                sweep_grid={'split_mode': ['blocked', 'random'], 'run_id': [0, 1]},
                                processing=nmi.Processing(x='continuous', x_params=p, y='continuous', y_params=p),
                                estimator=nmi.Estimator('infonce'), n_workers=4, seed=0, show_progress=False,
                                **extra)
            ratios = {r.split_mode: r.mi_mean / exact for r in swept.dataframe.itertuples()}
            epochs = {}
            for r in swept.runs.itertuples():
                epochs.setdefault(r.split_mode, []).append(int(r.best_epoch))
            row.append(f"{100 - 10 * step:>3}% overlap: blocked {ratios['blocked']:.2f} random "
                       f"{ratios['random']:.2f} (epochs {epochs['blocked']} / {epochs['random']})")
        say(f"{name}, exact {exact:.2f} bits  ({time.time() - t0:.0f}s)")
        for line in row:
            say('  ' + line)
    if out_path:
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
