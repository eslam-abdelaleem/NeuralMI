"""Is a custom_critic instance shared (and trained further) across repeats and terms?"""
import warnings
import numpy as np
import torch
torch.set_num_threads(1)
import neural_mi as nmi
from neural_mi.models import SeparableCritic, MLP

warnings.simplefilter('ignore')
rng = np.random.default_rng(0)
n = 3000
x = rng.standard_normal((n, 4)).astype(np.float32)
y = (0.8 * x + 0.6 * rng.standard_normal((n, 4))).astype(np.float32)

critic = SeparableCritic(MLP(4, 32, 8, 1), embedding_net_y=MLP(4, 32, 8, 1), embedding_dim=8)
w0 = critic.embedding_net_x.output_layer.weight.detach().clone()
r = nmi.run(x, y, mode='sweep', sweep_grid={'run_id': [0, 1, 2]}, model=nmi.Model(custom_critic=critic),
            training=nmi.Training(n_epochs=5), split=nmi.Split(mode='random'), seed=0, show_progress=False,
            output=nmi.Output(units='nats'))
changed = not torch.equal(w0, critic.embedding_net_x.output_layer.weight.detach())
print("caller's critic object was trained in place:", changed)
for rid, hist in zip(r.runs['run_id'], r.runs['test_mi_history']):
    print(f"  repeat {rid}: held-out MI at epoch 1 = {hist[0]:.3f}, at epoch 5 = {hist[-1]:.3f}")

try:
    w = rng.standard_normal((n, 4)).astype(np.float32)
    nmi.run(x, y, mode='conditional', conditional=nmi.Conditional(w_data=w),
            model=nmi.Model(custom_critic=SeparableCritic(MLP(4, 32, 8, 1), embedding_net_y=MLP(4, 32, 8, 1),
                                                          embedding_dim=8)),
            training=nmi.Training(n_epochs=2), seed=0, show_progress=False)
    print("conditional with custom_critic: ran")
except Exception as e:  # noqa
    print(f"conditional with custom_critic: {type(e).__name__}: {str(e)[:160]}")
