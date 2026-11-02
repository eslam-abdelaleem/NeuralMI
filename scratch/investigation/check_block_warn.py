"""Does block_mi warn when a large window leaves too few windows and the estimate collapses?"""
import sys, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
from reallike import RealLike
from whiten_helpers import ar_fit, whiten

real = RealLike()
x, y = real.sample(20000, seed=0)
ax, ay = ar_fit(x[:, 0].astype(float)), ar_fit(y[:, 0].astype(float))
xw, yw = whiten(x[:, 0].astype(float), ax), whiten(y[:, 0].astype(float), ay)
xw, yw = (xw / xw.std()).astype(np.float32)[:, None], (yw / yw.std()).astype(np.float32)[:, None]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    r = nmi.block_mi(xw, yw, window_size=20, seed=0, show_progress=False,
                     processing=nmi.Processing(x='continuous'), output=nmi.Output(units='nats'))
run = r.runs.iloc[0]
print(f"w=20: reported {r.mi_estimate:.3f} (exact 3.118) | test_mi {run['test_mi']:.3f} | eval_size {run.get('eval_size')} "
      f"train_eval_size {run.get('train_eval_size')} | test_ceiling {run.get('test_ceiling_mi'):.2f} | diag {run['curve_diagnosis']}")
print("warnings raised:")
for m in sorted({str(w.message)[:200] for w in caught}):
    print("  -", m)
if not caught:
    print("  (none)")
