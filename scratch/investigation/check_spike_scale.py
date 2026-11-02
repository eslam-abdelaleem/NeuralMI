"""Binned spikes: does normalize_bins=True (counts / bin width, 1 ms bins -> 1000 per spike) hurt,
and does spectral norm interact with it? Exact MI from the library's own generator."""
import warnings
import numpy as np
import torch
torch.set_num_threads(2)
import neural_mi as nmi
from neural_mi.generators.oracle import generate_spike_pair

warnings.simplefilter('ignore')
x, y, exact = generate_spike_pair(n_windows=4000, window_size=0.1, n_neurons=4, coding='count', seed=0)
print(f"exact MI {exact:.3f} bits")
for normalize in (True, False):
    for sn in (True, False):
        params = {'window_size': 0.1, 'bin_size': 0.001, 'normalize_bins': normalize}
        r = nmi.run(x, y, mode='estimate',
                    processing=nmi.Processing(x='spike', y='spike', x_params=params, y_params=params),
                    training=nmi.Training(shift_time=False, shift_windows=False),
                    model=nmi.Model(use_spectral_norm=sn), seed=0, show_progress=False)
        run = r.runs.iloc[0]
        print(f"normalize_bins={normalize!s:5s} (spike = {1000 if normalize else 1:>4}) spectral_norm={sn!s:5s}: "
              f"reported {r.mi_estimate:.3f} | held-out {run['test_mi']:.3f} | {run['curve_diagnosis']} | "
              f"rows {int(run['n_windows_built']) if run.get('n_windows_built') else 'n/a'}", flush=True)
