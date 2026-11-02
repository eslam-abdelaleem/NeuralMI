"""Cross-check: the library's own KSG / Frenzel-Pompe path estimates the conditional term directly."""
import sys, time, warnings
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
from reallike import RealLike

warnings.simplefilter('ignore')
m = RealLike()
x, y = m.sample(20000, seed=0)
for h, L in ((2, 5), (5, 5)):
    J, M, d = m.terms(h, L)
    t = time.time()
    r = nmi.mi_rate(x, y, h=h, half_width=L, estimator='ksg', seed=0, show_progress=False,
                    output=nmi.Output(units='nats'))
    row = r.dataframe.iloc[0]
    print(f"KSG h={h} L={L}: est={row.get('mi_raw_mean', row['mi_mean']):+.4f} true={d:.4f} | "
          f"J {row['mi_xw_y_mean']:.3f}/{J:.3f}  M {row['mi_w_y_mean']:.3f}/{M:.3f} | {time.time() - t:.0f}s",
          flush=True)
