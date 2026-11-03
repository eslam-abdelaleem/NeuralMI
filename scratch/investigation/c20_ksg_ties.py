"""C20: KSG on spike windows reads a conditional MI where the truth is 0. Usage: <repo>

W is X's own spike record from 40 s on, so on the rows of the plan (from 40 s) W is X and
I(X; Y | W) = 0. The script checks that X and W are the same rows, then estimates the
conditional MI with KSG and with the default trained estimator, and KSG on an exact copy of
continuous data, where the identity survives the tie-breaking noise.
"""
import sys
import warnings
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import neural_mi as nmi                                         # noqa: E402
from neural_mi.data.plan import RowPlan, resolve_stream         # noqa: E402
from neural_mi.estimators.ksg import ksg_cmi                    # noqa: E402

warnings.simplefilter('ignore')
p = {'window_size': 1.0}
for seed in (0, 1, 2):
    x, y, _ = nmi.generators.generate_spike_pair(n_windows=400, window_size=1.0, coding='count', seed=seed)
    w = [s[s >= 40.0] for s in x]
    plan = RowPlan(OrderedDict(x=resolve_stream(x, None, 'spike', p), w=resolve_stream(w, None, 'spike', p),
                               y=resolve_stream(y, None, 'spike', p)))
    rows = {k: s.data for k, s in plan.materialize().streams.items()}
    proc = nmi.Processing(x='spike', x_params=p, y='spike', y_params=p, w='spike', w_params=p)
    ksg = nmi.run(x, y, mode='conditional', conditional=nmi.Conditional(w_data=w), processing=proc,
                  estimator='ksg', seed=seed, show_progress=False).dataframe.iloc[0]
    net = nmi.run(x, y, mode='conditional', conditional=nmi.Conditional(w_data=w), processing=proc,
                  seed=seed, show_progress=False)
    print(f"seed {seed}: X and W the same {len(rows['x'])} rows: {torch.equal(rows['x'], rows['w'])}  "
          f"KSG {ksg['mi_xw_y_mean'] - ksg['mi_w_y_mean']:+.4f} bits  default {net.mi_estimate:+.4f} bits")
    # The same rows conditioned on an exact copy of X: with the rank map, without it, and with the ties of
    # X and its copy broken alike (the library's own noise off).
    from neural_mi.estimators import ksg as K
    xr, yr = (rows[k].numpy().reshape(len(rows[k]), -1) for k in ('x', 'y'))
    ranked, plain = (ksg_cmi(xr, yr, xr, seed=seed, gaussianize=g) / np.log(2) for g in (True, False))
    rng = np.random.default_rng(seed)
    jitter, noise = 1e-6 * rng.standard_normal(xr.shape), K.NOISE
    K.NOISE = 0.0
    alike = ksg_cmi(xr + jitter, yr + 1e-6 * rng.standard_normal(yr.shape), xr + jitter, seed=seed) / np.log(2)
    K.NOISE = noise
    print(f"         KSG I(X; Y | X) {ranked:+.4f} bits, without the rank map {plain:+.4f}, ties broken alike {alike:+.4f}")
z = np.random.default_rng(0).standard_normal((2000, 3))
print(f"continuous, an exact copy: KSG {ksg_cmi(z, z[:, :1] + np.random.default_rng(1).standard_normal((2000, 1)), z, exclude=0) / np.log(2):+.4f} bits")
