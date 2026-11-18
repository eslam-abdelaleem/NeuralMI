import sys, warnings
sys.path.insert(0,'/home/user/NeuralMI')
import numpy as np, pandas as pd, torch
import neural_mi.analysis.rigorous as R
rng=np.random.default_rng(0)
x=torch.randn(600,3); y=torch.randn(600,3)
def fn(xs,ys,params,**kw):
    return 1.0+0.1*int(params['_role'].split('gamma')[1].split('chunk')[0])+0.01*float(torch.randn(()))
def ladder(gr):
    return pd.DataFrame([{'gamma':g,'chunk':c,'train_mi':1.0+0.1*g+0.01*rng.normal()} for g in gr for c in range(g)])
def show(tag,f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            r=f()
            r=r[0] if isinstance(r,list) and r else r
            print(tag,'->', {k:r[k] for k in ('mi_corrected','is_reliable','enough_gamma_points','gammas_used')} if r else r)
        except Exception as e: print(tag,'RAISED',type(e).__name__,str(e)[:150])
    for m in w: print('    WARN:',str(m.message)[:140])
for gr in ([1,2,3],[1,2]):
    print('gamma_range',gr)
    show(' main  ',lambda: R._post_process_and_correct(ladder(gr),{},2.0,5,0.68))
    show(' scalar',lambda: R.run_rigorous_scalar_analysis(fn,x,y,{'show_progress':False},gamma_range=gr,min_gamma_points=5))
