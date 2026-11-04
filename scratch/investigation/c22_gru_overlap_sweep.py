"""Tutorial 04 section 7's GRU cell over every overlap, before the full run: what it reads and prints.

Runs the GRU cell and the table and figure cells as the notebook holds them, on the
notebook's settings and data. The MLP's values come from Eslam's full run of 4 Nov,
which trained the same way on the library before C23. For each GRU run it reports the
epoch read, and it lists the warnings the cell raised.

Usage: python scratch/investigation/c22_gru_overlap_sweep.py <repo> [out.txt] [figure.png]
"""
import contextlib
import io
import json
import re
import sys
import time
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Eslam's full run of tutorial 04 on 4 Nov, section 7's default-encoder sweep: (mean, spread) in bits.
LEAK_MLP = {('blocked', 1): (4.1541, 0.0566), ('random', 1): (4.1808, 0.0471),
            ('blocked', 2): (4.0683, 0.0666), ('random', 2): (4.0147, 0.1000),
            ('blocked', 5): (3.9835, 0.1152), ('random', 5): (4.1583, 0.0812),
            ('blocked', 10): (4.0625, 0.2494), ('random', 10): (4.1008, 0.1257)}


def cell(cells, marker):
    """The source of the one code cell that contains `marker`."""
    found = [''.join(c['source']) for c in cells if c['cell_type'] == 'code' and marker in ''.join(c['source'])]
    assert len(found) == 1, (marker, len(found))
    return found[0]


def main(repo, out_path=None, fig_path=None):
    sys.path.insert(0, repo)
    import neural_mi as nmi
    from neural_mi.generators import LatentGaussian
    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    cells = json.load(open(f'{repo}/tutorials/04_Preparing_Your_Data.ipynb'))['cells']
    ns = dict(nmi=nmi, time=time, plt=plt, T=10_000, N_SEEDS=3, OVERLAP_STEPS=[1, 2, 5, 10],
              SPLIT_MODES=['blocked', 'random'], W=10, SEED=0, N_WORKERS=4, leak=LEAK_MLP)
    oracle = LatentGaussian.shared(dims={'x': 8, 'y': 8}, d=2, phi=0.9, coupling=0.4, noise=1.0, seed=0)
    sample = oracle.sample(T=ns['T'], seed=0)
    ns.update(X=sample['x'], Y=sample['y'], exact_W=float(oracle.block_mi(ns['W'])))

    def run(marker):
        """Execute a notebook cell in the shared names and keep what it prints."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(cell(cells, marker), ns)
        for line in out.getvalue().splitlines():
            say(line)

    say(f"exact block MI at window 10: {ns['exact_W']:.4f} bits")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run("embedding_model='gru'")
    epochs = {}
    for r in ns['gru'].runs.itertuples():
        epochs.setdefault((r.split_mode, r.step_size), []).append(int(r.best_epoch))
    say('epochs read, per run:')
    for step in ns['OVERLAP_STEPS']:
        say(f"  overlap {100 * (1 - step / ns['W']):>3.0f}%  blocked {epochs[('blocked', step)]}  "
            f"random {epochs[('random', step)]}")
    kinds = Counter(re.split(r'(?<=[.!?])\s', str(w.message), maxsplit=1)[0][:110] for w in caught)
    say('warnings raised (first sentence, how many):')
    for text, n in kinds.most_common():
        say(f'  {n:>3}  {text}')

    plt.show = (lambda: plt.savefig(fig_path, dpi=150, bbox_inches='tight')) if fig_path else (lambda: None)
    run("Each scheme is read against the exact value")
    run("One panel per encoder")
    if out_path:
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:4]))
