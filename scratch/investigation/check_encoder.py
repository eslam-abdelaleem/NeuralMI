"""Which encoder does a default mi_rate / block_mi call build? (tiny run + the critic it builds)"""
import warnings
import numpy as np
import torch
torch.set_num_threads(1)
import neural_mi as nmi
from neural_mi.utils import build_critic

warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
x = rng.standard_normal((600, 1)).astype(np.float32)
y = (0.5 * x + rng.standard_normal((600, 1))).astype(np.float32)
r = nmi.mi_rate(x, y, h=2, half_width=3, training=nmi.Training(n_epochs=1), seed=0, show_progress=False)
bp = r.params['base_params']
print("mi_rate defaults: embedding_model =", bp.get('embedding_model'), "| critic_type =", bp.get('critic_type'),
      "| use_spectral_norm =", bp.get('use_spectral_norm'))
params = {**bp, 'input_dim_x': (7, 2), 'n_channels_x': (1, 1), 'input_dim_y': 1, 'n_channels_y': 1,
          '_difference_term': True}
critic = build_critic(bp.get('critic_type'), params, None)
print("joint term X encoder:", type(critic.embedding_net_x).__name__, "with branches",
      [type(b).__name__ for b in critic.embedding_net_x.branches], "| Y encoder:", type(critic.embedding_net_y).__name__)
r2 = nmi.block_mi(x, y, window_size=3, processing=nmi.Processing(x='continuous'),
                  training=nmi.Training(n_epochs=1), seed=0, show_progress=False)
print("block_mi defaults: embedding_model =", r2.params['base_params'].get('embedding_model'))
