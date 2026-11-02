"""Can picking configurations by held-out MI rescue the directions that failed at defaults?

block <lr>  : block_mi at w=5 (realistic data, z-scored, tiled windows), sweep hidden_dim x n_epochs at
              one learning rate; exact I(5) = 0.574 nats.
chain <lr>  : mi_rate h=5, half_width=20 (realistic, z-scored, NOT whitened), batch_size {128, 1024},
              n_epochs 150 at one learning rate. Each term is later picked by its own held-out MI.
Usage: python select_by_test.py <scratchpad> <block|chain> <lr> <outfile>
"""
import json, sys, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import neural_mi as nmi
from reallike import RealLike

warnings.simplefilter('ignore')
kind, lr, outfile = sys.argv[2], float(sys.argv[3]), sys.argv[4]
real = RealLike()
x, y = real.sample(20000, seed=0)
x, y = x / x.std(), y / y.std()
rows = []

if kind == 'block':
    r = nmi.block_mi(x, y, window_size=5, seed=0, show_progress=False,
                     processing=nmi.Processing(x='continuous'), output=nmi.Output(units='nats'),
                     training=nmi.Training(learning_rate=lr),
                     sweep_grid={'hidden_dim': [64, 256], 'n_epochs': [50, 200]})
    for _, run in r.runs.iterrows():
        rows.append({'lr': lr, 'hidden_dim': int(run['hidden_dim']), 'n_epochs': int(run['n_epochs']),
                     'train_mi': float(run['train_mi']), 'test_mi': float(run['test_mi']),
                     'diag': run['curve_diagnosis'], 'exact': 0.574})
else:
    r = nmi.mi_rate(x, y, h=5, half_width=20, seed=0, show_progress=False, output=nmi.Output(units='nats'),
                    training=nmi.Training(learning_rate=lr, n_epochs=150),
                    sweep_grid={'batch_size': [128, 1024]})
    J, M, d = real.terms(5, 20)
    for cid, entry in r.details.items():
        tr = entry['trainings']
        bs = int(r.dataframe.loc[r.dataframe['config_id'] == cid, 'batch_size'].iloc[0])
        for comp in ('mi_xw_y', 'mi_w_y'):
            sub = tr[tr['component'] == comp].iloc[0]
            rows.append({'lr': lr, 'batch_size': bs, 'term': 'J' if comp == 'mi_xw_y' else 'M',
                         'train_mi': float(sub['train_mi']), 'test_mi': float(sub['test_mi']),
                         'diag': sub['curve_diagnosis'], 'exact': J if comp == 'mi_xw_y' else M})
for row in rows:
    print(row, flush=True)
json.dump(rows, open(outfile, 'w'), indent=1, default=float)
print('DONE', flush=True)
