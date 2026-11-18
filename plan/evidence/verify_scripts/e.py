import sys, copy, importlib, warnings, traceback
sys.path.insert(0,'/home/user/NeuralMI')
import numpy as np, torch
import neural_mi as nmi
from neural_mi import Training, Model
import neural_mi.run as m
print("E5 type(neural_mi.run via import-as):", type(m), "| nmi.logger:", type(nmi.logger), "| sys.modules:", type(sys.modules['neural_mi.run']))
import neural_mi.logger as lg; print("E5 logger import-as:", type(lg))
rm = importlib.import_module('neural_mi.run')
from neural_mi.defaults import BASE_PARAMS_SCHEMA
snap = copy.deepcopy({k:v.get('default') for k,v in BASE_PARAMS_SCHEMA.items() if isinstance(v.get('default'),(dict,list))})
print("mutable defaults:", snap)
rng=np.random.default_rng(0); x=rng.normal(size=(300,3)).astype('float32'); y=(x+rng.normal(size=(300,3))).astype('float32')
# E2 capture
orig = rm._warn_if_shift_windows_dead
cap={}
def spy(bp, mode, shifts, user_set_keys=None, offsets=False):
    cap['k']=set(user_set_keys); cap['bp']=bp; return orig(bp,mode,shifts,user_set_keys=user_set_keys,offsets=offsets)
rm._warn_if_shift_windows_dead = spy
print("cudnn before:", torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
tr = Training(n_epochs=2, batch_size=64)
r = nmi.run(x,y,mode='estimate',training=tr,seed=1,show_progress=False)
print("cudnn after:", torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
print("E2 user_set_keys:", sorted(cap['k']))
bp=cap['bp']
for k in snap:
    print("E4 identity", k, bp.get(k) is BASE_PARAMS_SCHEMA[k]['default'])
r = nmi.run(x,y,mode='estimate',training=tr,seed=1,show_progress=False)
now = {k:BASE_PARAMS_SCHEMA[k].get('default') for k in snap}
print("E4 schema unchanged after 2 runs:", now==snap)
for n in (0,-1):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = nmi.run(x,y,mode='estimate',training=tr,seed=1,show_progress=False,permutation_test=True,n_permutations=n)
        print("E1 n=",n,"OK; warnings:",[str(i.message)[:200] for i in w])
        df=r.dataframe; print(df.filter(regex='p_|null|perm').to_dict('records') if hasattr(df,'filter') else df)
        print([a for a in dir(r) if 'null' in a or 'perm' in a or 'p_val' in a])
    except Exception as e:
        print("E1 n=",n,"EXC",type(e).__name__, str(e)[:600])
try: nmi.run(x,y,seeds=1)
except Exception as e: print("E6", type(e).__name__, e)
