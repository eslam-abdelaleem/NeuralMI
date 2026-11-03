"""Replay budgets and patience rules on the curves c18_budget.py recorded.

    python c18_analyse.py <repo> <results.json>
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, sys.argv[1])
from neural_mi.training.epochs import choose_epoch          # noqa: E402
from neural_mi.training.trainer import Trainer              # noqa: E402

LN2 = np.log(2.0)
SIGMA, MED = 1.0, 5
REACH = Trainer._smoothing_reach(SIGMA, MED)
BUDGETS = (50, 100, 150, 200, 300)
PATIENCE = (20, 30, 50)
MIN_IMPROVEMENT = 0.001


def smooth(h):
    return Trainer._smooth(None, list(h), SIGMA, MED)


def read(net, budget):
    """The epoch a run of `budget` epochs reads, its train-side value in bits, and its diagnosis."""
    h = np.asarray(net['test_mi_history'][:budget], dtype=float) * LN2       # stored in bits; the rule reads nats
    best, peak, diagnosis, _ = choose_epoch(h, smooth(h), 'plateau_start', REACH, ran_to_limit=True,
                                            n_rows=int(net['eval_size']))
    return best, net['train_mi_history'][best], diagnosis


def stop_epoch(net, patience, cap):
    """The epoch count a run with this patience would have trained for (the trainer's own rule)."""
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
    """The run's value from each network's reading: one network, or the joint minus the conditioning term."""
    nets = run['networks']
    if len(nets) == 1:
        return reader(nets[0])
    by = {n['component']: n for n in nets}
    j, c = reader(by['mi_xw_y']), reader(by['mi_w_y'])
    return (max(j[0], c[0]), j[1] - c[1], f"{j[2]}|{c[2]}")


def main():
    runs = json.load(open(sys.argv[2]))
    groups = defaultdict(list)
    for r in runs:
        groups[(r['name'], r['shift'])].append(r)
    # The replay must give what the library reported for the full budget.
    worst = max(abs(quantity(r, lambda n: read(n, 300))[1] - r['mi_estimate'])
                for r in runs if r['mi_estimate'] > 0)
    print(f"replay check: largest |replayed - reported| at 300 epochs = {worst:.2e} bits\n")
    for (name, shift), rs in sorted(groups.items()):
        exact = rs[0]['exact']
        print(f"== {name}  shift={shift}  exact={exact:.3f} bits  ({len(rs)} seeds, "
              f"{np.mean([r['seconds'] for r in rs]):.0f} s per run)")
        for net_i, net in enumerate(rs[0]['networks']):
            label = net.get('component', 'net')
            epochs = [r['networks'][net_i]['best_epoch'] for r in rs]
            peaks = [r['networks'][net_i]['peak_epoch'] for r in rs]
            print(f"   {label:8s} epoch read at 300: {epochs}   peak: {peaks}")
        for b in BUDGETS:
            vals = [quantity(r, lambda n: read(n, b)) for r in rs]
            est = np.array([v[1] for v in vals])
            diag = [v[2] for v in vals]
            rising = sum('still rising' in d for d in diag)
            print(f"   budget {b:3d}: {est.mean():6.3f} ± {est.std():.3f}  ({100 * est.mean() / exact:5.1f}% of exact)"
                  f"  still rising {rising}/{len(rs)}")
        for p in PATIENCE:
            stops = [quantity(r, lambda n: (stop_epoch(n, p, 300), 0.0, ''))[0] for r in rs]
            vals = [quantity(r, lambda n: read(n, stop_epoch(n, p, 300))) for r in rs]
            est = np.array([v[1] for v in vals])
            print(f"   patience {p:2d}: stops at {stops}  {est.mean():6.3f} ± {est.std():.3f}"
                  f"  ({100 * est.mean() / exact:5.1f}% of exact)")
        print()


if __name__ == '__main__':
    main()
