"""Summarise the spectral-norm study: per scenario, SN on vs off, each at the learning rate its own
held-out MI prefers (the selection rule), plus both learning rates."""
import glob, json, sys
from collections import defaultdict
import numpy as np

rows = []
for f in sorted(glob.glob(sys.argv[1] + '/out/sn_[A-D].json')):
    rows += json.load(open(f))

cells = defaultdict(list)
for r in rows:
    cells[(r['scenario'], r['sn'], r['lr'])].append(r)


def fmt(v):
    return f"{np.mean(v):5.2f}±{np.std(v):4.2f}"


print(f"{'scenario':26s} {'exact':>5s} | {'SN':>3s} {'lr':>6s} | {'held-out':>10s} {'reported':>10s} | resolved")
best = {}
for (scen, sn, lr), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][2], not kv[0][1])):
    exact = rs[0]['exact']
    ho = [r['held_out'] for r in rs if r['held_out'] is not None]
    rep = [r['reported'] for r in rs]
    ok = sum(1 for r in rs if r['diag'] in ('ok', 'fuzzy start'))
    print(f"{scen:26s} {exact:5.2f} | {'on' if sn else 'off':>3s} {lr:6.0e} | "
          f"{fmt(ho) if ho else '      n/a':>10s} {fmt(rep):>10s} | {ok}/{len(rs)}")
    key = (scen, sn)
    score = np.mean(ho) if ho else np.mean(rep)
    if key not in best or score > best[key][0]:
        best[key] = (score, lr, np.mean(rep), np.std(rep), exact, ok, len(rs))

print("\nEach setting at its own best learning rate (chosen by mean held-out MI):")
print(f"{'scenario':26s} {'exact':>5s} | {'SN on: lr, reported':>24s} | {'SN off: lr, reported':>24s}")
for scen in sorted({k[0] for k in best}):
    on, off = best.get((scen, True)), best.get((scen, False))
    show = lambda b: (f"{b[1]:6.0e}, {b[2]:5.2f}±{b[3]:4.2f} ({b[5]}/{b[6]})" if b else '')
    print(f"{scen:26s} {(on or off)[4]:5.2f} | {show(on):>24s} | {show(off):>24s}")
