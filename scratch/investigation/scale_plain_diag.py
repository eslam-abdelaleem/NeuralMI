"""What happens at each scale: per-network values and the warnings raised."""
import warnings
import numpy as np
import torch
torch.set_num_threads(1)
import neural_mi as nmi

rng = np.random.default_rng(0)
d, n, true_mi = 5, 8000, 2.0
rho = np.sqrt(1 - np.exp(-2 * true_mi / d))
x = rng.standard_normal((n, d))
y = rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal((n, d))
for scale in (1e-3, 1e-1, 10.0, 1e2, 1e3):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = nmi.run((x * scale).astype(np.float32), (y * scale).astype(np.float32), mode='estimate',
                    split=nmi.Split(mode='random'), output=nmi.Output(units='nats'),
                    seed=0, show_progress=False)
    run = r.runs.iloc[0]
    hist = run['test_mi_history']
    print(f"scale {scale:g}: reported {r.mi_estimate:.3f} | train_mi {run['train_mi']:.3f} raw_train_mi "
          f"{run['raw_train_mi']:.3f} test_mi {run['test_mi']:.3f} | test curve first/max/last "
          f"{hist[0]:.3f}/{np.nanmax(hist):.3f}/{hist[-1]:.3f} | diag {run['curve_diagnosis']}", flush=True)
    for m in sorted({str(w.message)[:150] for w in caught}):
        print("     warning:", m)
