"""Chain route (mi_rate) h-sweep on pre-whitened data, library defaults otherwise. Usage: <sp> <seed> <out>"""
import json, sys, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
from reallike import RealLike
from whiten_helpers import ar_fit, whiten, gauss_terms
from harness import run_rate

warnings.simplefilter('ignore')
seed = int(sys.argv[2])
real = RealLike()
x, y = real.sample(20000, seed=0)
ax, ay = ar_fit(x[:, 0].astype(float)), ar_fit(y[:, 0].astype(float))
xw, yw = whiten(x[:, 0].astype(float), ax), whiten(y[:, 0].astype(float), ay)
xw, yw = (xw / xw.std()).astype(np.float32)[:, None], (yw / yw.std()).astype(np.float32)[:, None]
xr, yr = real.sample(2_000_000, seed=123)
xrw, yrw = whiten(xr[:, 0].astype(float), ax), whiten(yr[:, 0].astype(float), ay)
hs = [1, 2, 5, 10]
targets = {h: gauss_terms(xrw, yrw, h, 20) for h in hs}
recs, msgs = run_rate(xw, yw, h=hs, half_width=20, seed=seed)
rows = []
for r in recs:
    J, M, d = targets[int(r['h'])]
    r.update(J_true=J, M_true=M, rate_true=d, seed=seed)
    rows.append(r)
    print(f"seed {seed} h={int(r['h']):2d}: est={r['mi_raw'] if r['mi_raw'] is not None else r['mi']:+.4f} "
          f"target {d:.4f} | J {r['J']:.3f}/{J:.3f} M {r['M']:.3f}/{M:.3f} | {r.get('J_diag')} {r.get('M_diag')}",
          flush=True)
json.dump({'rows': rows, 'warnings': msgs}, open(sys.argv[3], 'w'), indent=1, default=float)
print("true rate (infinite windows) 0.1637")
