import warnings, logging, numpy as np, torch, sys
sys.path.insert(0,'/home/user/NeuralMI')
from neural_mi.data import create_dataset
from neural_mi.data.plan import RowPlan
rng=np.random.default_rng(0)
s=np.sort(rng.uniform(0,20,400)); sh=rng.permutation(s)
pp={'window_size':1.0,'bin_size':0.1}
def build(st,pp):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ds=create_dataset([st],processor_type_x='spike',processor_params_x=pp)
    return ds, [str(i.message)[:120] for i in w]
a,wa=build(s,pp); b,wb=build(sh,pp)
xa=a.x_data.data if hasattr(a.x_data,'data') else None
def tens(ds): return torch.stack([ds[i][0] if isinstance(ds[i],(tuple,list)) else ds[i] for i in range(len(ds))])
ta,tb=tens(a),tens(b)
print('binned sorted  : windows',len(a),'shape',tuple(ta.shape),'total count',float(ta.sum()),'warn',wa)
print('binned shuffled: windows',len(b),'shape',tuple(tb.shape),'total count',float(tb.sum()),'warn',wb)
pp2={'window_size':1.0}
a,wa=build(s,pp2); b,wb=build(sh,pp2)
ta,tb=tens(a),tens(b)
print('spiketime sorted/shuffled equal:',ta.shape==tb.shape and bool(torch.equal(ta,tb)),len(a),len(b))
# RowPlan (training path)
for st,l in ((s,'sorted'),(sh,'shuffled')):
    p=RowPlan.of_pair([st],None,dict(processor_type_x='spike',processor_params_x=pp,shift_windows=False))
    r=p.rows(); w=r['windows']['x']; w=w[0] if isinstance(w,tuple) else w
    print('RowPlan',l,p.route,tuple(w.shape),float(w.sum()))
# public validator
from neural_mi.validation import DataValidator
import inspect
print(inspect.signature(DataValidator.__init__))
