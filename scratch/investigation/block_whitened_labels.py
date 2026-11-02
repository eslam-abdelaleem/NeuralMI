"""Block route on pre-whitened data: held-out value, reported value and curve diagnosis per window."""
import sys, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
from reallike import RealLike
from whiten_helpers import ar_fit, whiten

warnings.simplefilter('ignore')
real = RealLike()
x, y = real.sample(20000, seed=0)
ax, ay = ar_fit(x[:, 0].astype(float)), ar_fit(y[:, 0].astype(float))
xw, yw = whiten(x[:, 0].astype(float), ax), whiten(y[:, 0].astype(float), ay)
xw, yw = (xw / xw.std()).astype(np.float32)[:, None], (yw / yw.std()).astype(np.float32)[:, None]
exact = {5: 0.745, 10: 1.526, 20: 3.118}
r = nmi.block_mi(xw, yw, window_size=[5, 10, 20], seed=0, show_progress=False,
                 processing=nmi.Processing(x='continuous'), output=nmi.Output(units='nats'))
for _, run in r.runs.iterrows():
    w = int(run['window_size'])
    print(f"w={w:2d}: held-out {run['test_mi']:.3f} | reported {run['train_mi']:.3f} | exact {exact[w]:.3f} "
          f"({100 * (run['train_mi'] / exact[w] - 1):+.0f}%) | held-out rows {int(run['eval_size'])} | "
          f"diagnosis {run['curve_diagnosis']}")
