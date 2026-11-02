"""Block route on pre-whitened data (tiled windows), library defaults otherwise."""
import json, sys, warnings
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
exact = {5: 0.745, 10: 1.526, 20: 3.118}   # whitened processes, large-sample Gaussian (block_vs_chain.py)
rows = []
for seed in (0, 1):
    r = nmi.block_mi(xw, yw, window_size=[5, 10, 20], seed=seed, show_progress=False,
                     processing=nmi.Processing(x='continuous'), output=nmi.Output(units='nats'))
    df = r.dataframe
    I = {int(w): float(m) for w, m in zip(df['window_size'], df['mi_mean'])}
    row = {'seed': seed, 'I': I, 'exact': exact, 'slope_5_10': (I[10] - I[5]) / 5, 'slope_10_20': (I[20] - I[10]) / 10}
    rows.append(row)
    print(row, flush=True)
json.dump(rows, open(sys.argv[2], 'w'), indent=1)
print("rate 0.1637; exact whitened slopes 0.156 (5->10), 0.159 (10->20)")
