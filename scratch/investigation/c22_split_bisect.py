"""Tutorial 04 section 7's random-split inflation on the library of any commit, behind C22.

Runs the October tutorial's call exactly: windows of 10 samples one sample apart
(90% overlap) on its LatentGaussian pair, its model and training, a blocked and a
random split, two repeats each, and prints each split's mean against the exact
block MI. The library comes from the checkout this script is pointed at, through
PYTHONPATH so that worker processes import it too:

    PYTHONPATH=<checkout> python scratch/investigation/c22_split_bisect.py <checkout> [label]

In October the random split read far above the exact value here and the blocked
one a little below it. On the library after the correctness fixes both read a
little below it.
"""
import sys
import time
import warnings


def main(checkout, label=''):
    import neural_mi as nmi
    from neural_mi.generators import LatentGaussian
    assert nmi.__file__.startswith(checkout), f"neural_mi imported from {nmi.__file__}, not {checkout}"
    oracle = LatentGaussian.shared(dims={'x': 8, 'y': 8}, d=2, phi=0.9, coupling=0.4, noise=1.0, seed=0)
    sample = oracle.sample(T=10_000, seed=0)
    exact = float(oracle.block_mi(10))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        swept = nmi.run(sample['x'], sample['y'], mode='sweep',
                        sweep_grid={'split_mode': ['blocked', 'random'], 'run_id': [0, 1]},
                        processing=nmi.Processing(x='continuous',
                                                  x_params={'window_size': 10, 'step_size': 1},
                                                  y_params={'window_size': 10, 'step_size': 1}),
                        model=nmi.Model(embedding_dim=8, hidden_dim=128, n_layers=2),
                        training=nmi.Training(n_epochs=200, batch_size=256, patience=30, learning_rate=1e-3),
                        estimator=nmi.Estimator('infonce'), n_workers=4, seed=0, show_progress=False)
    ratios = {row.split_mode: row.mi_mean / exact for row in swept.dataframe.itertuples()}
    epochs = {}
    if 'best_epoch' in swept.runs:
        for row in swept.runs.itertuples():
            epochs.setdefault(row.split_mode, []).append(int(row.best_epoch))
    print(f"{label:<10} blocked {ratios['blocked']:.2f}  random {ratios['random']:.2f}  "
          f"best epochs {epochs}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
