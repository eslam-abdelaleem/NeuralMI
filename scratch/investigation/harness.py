"""Run nmi.mi_rate and pull out the joint / marginal terms and per-network diagnostics (nats)."""
import json
import math
import time
import warnings

import numpy as np

import neural_mi as nmi


def _f(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def run_rate(x, y, h, half_width=20, seed=0, **kw):
    """One nmi.mi_rate call; returns a list of per-config dicts (one per value of h / half_width)."""
    kw.setdefault('output', nmi.Output(units='nats'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.time()
        r = nmi.mi_rate(x, y, h=h, half_width=half_width, seed=seed,
                        show_progress=False, verbose=False, **kw)
        elapsed = time.time() - t0
    msgs = sorted({str(w.message).split('.')[0][:110] for w in caught})
    out = []
    df = r.dataframe
    for i, row in df.iterrows():
        cid = int(row['config_id'])
        rec = {
            'h': _f(row['h']) if 'h' in row else h,
            'half_width': _f(row['half_width']) if 'half_width' in row else half_width,
            'mi': _f(row['mi_mean']), 'mi_raw': _f(row.get('mi_raw_mean')),
            'mi_std': _f(row.get('mi_std')),
            'J': _f(row.get('mi_xw_y_mean')), 'M': _f(row.get('mi_w_y_mean')),
            'amp': _f(row.get('amplification_factor')),
        }
        tr = (r.details.get(cid) or {}).get('trainings')
        if tr is not None and len(tr):
            for comp, tag in (('mi_xw_y', 'J'), ('mi_w_y', 'M')):
                sub = tr[tr['component'] == comp]
                if len(sub):
                    rec[f'{tag}_train'] = _f(sub['train_mi'].mean())
                    rec[f'{tag}_test'] = _f(sub['test_mi'].mean())
                    rec[f'{tag}_epoch'] = _f(sub['best_epoch'].mean())
                    rec[f'{tag}_diag'] = list(sub['curve_diagnosis'].astype(str))
                    if 'train_ceiling_mi' in sub:
                        rec[f'{tag}_ceiling'] = _f(sub['train_ceiling_mi'].mean())
        else:  # h = 0: a single network (mode='estimate')
            runs = r.runs[r.runs['config_id'] == cid] if 'config_id' in r.runs else r.runs
            rec['J'] = rec['mi']
            rec['M'] = 0.0
            rec['J_train'] = _f(runs['train_mi'].mean())
            rec['J_test'] = _f(runs['test_mi'].mean())
            rec['J_epoch'] = _f(runs['best_epoch'].mean())
            rec['J_diag'] = list(runs['curve_diagnosis'].astype(str))
        rec['seconds'] = elapsed / len(df)
        out.append(rec)
    return out, msgs


def dump(path, obj):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1, default=str)
