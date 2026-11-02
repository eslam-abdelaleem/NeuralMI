"""Is the unit dependence specific to mi_rate? Plain MI (mode='estimate') on i.i.d. Gaussians of known MI."""
import warnings
import numpy as np
import torch
torch.set_num_threads(1)
import neural_mi as nmi

warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
d, n, true_mi = 5, 8000, 2.0                       # 5 independent channel pairs, 2 nats in total
rho = np.sqrt(1 - np.exp(-2 * true_mi / d))
x = rng.standard_normal((n, d))
y = rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, d))
for scale in (1e-3, 1.0, 1e3):
    vals = []
    for s in (0, 1):
        r = nmi.run((x * scale).astype(np.float32), (y * scale).astype(np.float32), mode='estimate',
                    split=nmi.Split(mode='random'), output=nmi.Output(units='nats'),
                    seed=s, show_progress=False)
        vals.append(r.mi_estimate)
    print(f"scale {scale:g}: MI estimates {', '.join(f'{v:.3f}' for v in vals)} nats (true {true_mi})", flush=True)
