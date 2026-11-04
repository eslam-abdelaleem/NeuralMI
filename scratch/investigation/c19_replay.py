"""The epoch rule on the curves c18_budget.py recorded, with the noise of the whole curve and near the top, behind C19.

Replays every network's held-out curve at fixed budgets with stopping off and at
the default patience, and reads each run's value the way the library does: once
with the curve's noise measured over the whole curve, as before C19, and once
with the library's own rule, which measures it up to NOISE_SPAN epochs past the
smoothed top.

Usage: python scratch/investigation/c19_replay.py <repo> [out.txt]
"""
import json
import sys
from collections import defaultdict

import numpy as np

LN2 = np.log(2.0)
BUDGETS = (50, 100, 150, 200, 300)
PATIENCE = 30
MIN_IMPROVEMENT = 0.001


def main(repo, out_path=None):
    sys.path.insert(0, repo)
    from neural_mi.training import epochs
    from neural_mi.training.trainer import Trainer
    reach = Trainer._smoothing_reach(1.0, 5)
    span = epochs.NOISE_SPAN
    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    def smooth(h):
        return Trainer._smooth(None, list(h), 1.0, 5)

    def read(net, budget, whole_curve):
        """The epoch a run of `budget` epochs reads and its train-side value in bits."""
        h = np.asarray(net['test_mi_history'][:budget], dtype=float) * LN2   # stored in bits; the rule reads nats
        epochs.NOISE_SPAN = 10 ** 9 if whole_curve else span
        best = epochs.choose_epoch(h, smooth(h), 'plateau_start', reach, ran_to_limit=True,
                                   n_rows=int(net['eval_size']))[0]
        epochs.NOISE_SPAN = span
        return best, net['train_mi_history'][best]

    def stop_epoch(net, patience, cap):
        h = net['test_mi_history']
        best_mi, no_improve = -np.inf, 0
        for e in range(min(cap, len(h))):
            s = smooth(h[:e + 1])[-1]
            improvement = (s - best_mi) / (abs(best_mi) + 1e-8) if np.isfinite(best_mi) else np.inf
            if not np.isnan(s) and (improvement > MIN_IMPROVEMENT or np.isinf(best_mi)):
                best_mi, no_improve = s, 0
            else:
                no_improve += 1
            if no_improve >= patience:
                return e + 1
        return min(cap, len(h))

    def quantity(run, reader):
        nets = run['networks']
        if len(nets) == 1:
            return reader(nets[0])
        by = {n['component']: n for n in nets}
        j, c = reader(by['mi_xw_y']), reader(by['mi_w_y'])
        return max(j[0], c[0]), j[1] - c[1]

    runs = json.load(open(f'{repo}/scratch/investigation/results/c18_budget.json'))
    groups = defaultdict(list)
    for r in runs:
        groups[(r['name'], r['shift'])].append(r)
    say(f"NOISE_SPAN = {span}; percentages are of the exact value, epochs per seed\n")
    same_at_default, drift = 0, {True: [], False: []}
    for (name, shift), rs in sorted(groups.items()):
        exact = rs[0]['exact']
        say(f"== {name}  shift={shift}  exact={exact:.3f} bits  ({len(rs)} seeds)")
        say(f"   {'stopping':<16} {'whole curve (before C19)':>30} {'near the top (now)':>30}")
        cases = [(f'off, {b} epochs', lambda n, w, b=b: read(n, b, w)) for b in BUDGETS]
        cases.append((f'patience {PATIENCE}', lambda n, w: read(n, stop_epoch(n, PATIENCE, 300), w)))
        values = {}
        for label, reader in cases:
            row = []
            for whole in (True, False):
                v = [quantity(r, lambda n: reader(n, whole)) for r in rs]
                values[(label, whole)] = np.mean([x[1] for x in v])
                row.append(f"{100 * values[(label, whole)] / exact:6.1f}%  {str([x[0] for x in v]):<14}")
            say(f"   {label:<16} {row[0]:>30} {row[1]:>30}")
        same_at_default += int(np.isclose(values[(f'patience {PATIENCE}', True)], values[(f'patience {PATIENCE}', False)]))
        for whole in (True, False):
            drift[whole].append(100 * (values[('off, 300 epochs', whole)] - values[('off, 50 epochs', whole)]) / exact)
        say()
    say(f"default patience reads the same under both: {same_at_default} of {len(groups)} groups")
    for whole, label in ((True, 'whole curve'), (False, 'near the top')):
        say(f"stopping off, 300 against 50 epochs, {label}: {min(drift[whole]):+.1f} to {max(drift[whole]):+.1f} "
            f"points of the exact value")
    if out_path:
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
