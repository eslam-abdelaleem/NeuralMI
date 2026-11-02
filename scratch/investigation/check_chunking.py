"""Which chunking does rigorous=True use for the offset-built quantities? (tiny, fast run)"""
import sys, warnings
import torch
torch.set_num_threads(1)
import numpy as np
import neural_mi as nmi

warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
T = 1500
z = np.cumsum(rng.standard_normal(T)) * 0.1
x = (z + 0.3 * rng.standard_normal(T)).astype(np.float32).reshape(-1, 1)
y = (z + 0.3 * rng.standard_normal(T)).astype(np.float32).reshape(-1, 1)
cheap = dict(model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
             training=nmi.Training(n_epochs=2), show_progress=False, seed=0,
             rigorous=nmi.Rigorous(gamma_range=range(1, 6)))

for name, call in {
    'mi_rate': lambda: nmi.mi_rate(x, y, h=2, half_width=2, **cheap),
    'directed_information_rate': lambda: nmi.directed_information_rate(x, y, k=2, **cheap),
    'active_information_storage': lambda: nmi.active_information_storage(x, k=3, **cheap),
    'transfer_entropy': lambda: nmi.transfer_entropy(x, y, history_window=3, **cheap),
}.items():
    try:
        r = call()
        print(f"{name:28s} chunking_mode = {r.runs['chunking_mode'].iloc[0]!r}")
    except Exception as e:  # noqa
        print(f"{name:28s} ERROR {type(e).__name__}: {str(e)[:150]}")
