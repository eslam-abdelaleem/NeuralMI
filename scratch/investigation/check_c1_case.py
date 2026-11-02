"""Rerun the mi_rate repeat that reported its joint term alone (raw_x162, h=5, half_width=20, seed=2)."""
import sys
import warnings

import torch

torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])

import neural_mi as nmi
from reallike import RealLike

real = RealLike()
x_raw, y_raw = real.sample(20000, seed=0)
J, M, d = real.terms(5, 20)
print(f"exact: J={J:.3f} M={M:.3f} rate={d:.3f} nats", flush=True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    r = nmi.mi_rate(x_raw, y_raw, h=5, half_width=20, seed=2, show_progress=False, verbose=False,
                    output=nmi.Output(units='nats'))
row = r.dataframe.iloc[0]
print('dataframe:', {k: row[k] for k in ('mi_mean', 'mi_std', 'n_runs', 'n_zero', 'n_failed')
                     if k in row}, flush=True)
print('runs:', r.runs[['mi', 'mi_xw_y', 'mi_w_y', 'failed_term']].to_dict('records'))
tr = r.details[0]['trainings']
print(tr[['component', 'train_mi', 'raw_train_mi', 'test_mi', 'eval_size', 'all_mi_negative',
          'curve_diagnosis']].to_string())
for w in caught:
    if w.category.__name__ == 'CombinationWarning' or 'train MI at the reported' in str(w.message):
        print('-', w.category.__name__, str(w.message)[:400])
