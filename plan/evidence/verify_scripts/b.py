import matplotlib; matplotlib.use('Agg')
import warnings, numpy as np, torch, inspect
import neural_mi as nmi
from neural_mi import quantities as q
rng=np.random.default_rng(0)
T=600
x=np.cumsum(rng.standard_normal((T,2)),0).astype('float32')*0.1; y=(x+0.5*rng.standard_normal((T,2))).astype('float32')
tr=nmi.Training(n_epochs=4, batch_size=64)
sig=inspect.signature(nmi.run).parameters
print([p for p in sig if 'seed' in p])
def go(name,f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r=f()
    print('==',name,'mode',r.mode,'mi',getattr(r,'mi_estimate',None))
    for k in ('_rows','rows','quantity','stride'): 
        if k in r.params: print('  param',k,r.params[k])
    for ww in w: print('  WARN',str(ww.message)[:260])
    return r
print("=== F5")
kw=dict(training=tr,show_progress=False,seed=0)
r1=go('run',lambda: nmi.run(x,y,**kw))
r2=go('inst',lambda: q.instantaneous_mi(x,y,**kw))
print(r1.dataframe.to_string() if r1.dataframe is not None else None); print(r2.dataframe.to_string() if r2.dataframe is not None else None)
print({k:v for k,v in r1.params.items() if 'row' in k}, {k:v for k,v in r2.params.items() if 'row' in k})
sp=dict(split=nmi.Split(mode='random'))
r1=go('run rand',lambda: nmi.run(x,y,**sp,**kw)); r2=go('inst rand',lambda: q.instantaneous_mi(x,y,**sp,**kw))
pk=dict(permutation_test=True,n_permutations=3,min_permutations=3)
r1=go('run perm',lambda: nmi.run(x,y,**pk,**kw)); r2=go('inst perm',lambda: q.instantaneous_mi(x,y,**pk,**kw))
print(r1.dataframe.filter(regex='mi_mean|p_|null').to_string()); print(r2.dataframe.filter(regex='mi_mean|p_|null').to_string())
rg=nmi.Rigorous(gamma_range=[1,2,3],min_gamma_points=2)
tr2=nmi.Training(n_epochs=2,batch_size=64); kw2=dict(training=tr2,show_progress=False,seed=0)
r1=go('run rig',lambda: nmi.run(x,y,mode='rigorous',rigorous=rg,**kw2)); r2=go('inst rig',lambda: q.instantaneous_mi(x,y,rigorous=rg,**kw2))
print("=== F6")
r=go('mi_rate',lambda: q.mi_rate(x,y,h=2,half_width=1,**kw)); print(r.summary())
import matplotlib.pyplot as plt
try:
    ax=r.plot(show=False); f=plt.gcf(); print('PLOT texts',[a.get_title() for a in f.axes],[t.get_text() for a in f.axes for t in (a.get_legend().get_texts() if a.get_legend() else [])],[a.get_ylabel() for a in f.axes],[l.get_text() for a in f.axes for l in a.get_xticklabels()])
except Exception as e: print('plot err',e)
r=go('ais5',lambda: q.active_information_storage(x,k=5,**kw)); print(r.summary())
r=go('ais[5]',lambda: q.active_information_storage(x,k=[5],**kw)); print(r.summary())
