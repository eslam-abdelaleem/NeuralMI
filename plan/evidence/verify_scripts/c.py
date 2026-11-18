import warnings, numpy as np, inspect
warnings.simplefilter('ignore')
import neural_mi as nmi
from neural_mi import quantities as q
import neural_mi.analysis.rigorous as R
cls=[c for n,c in vars(R).items() if inspect.isclass(c) and hasattr(c,'chunks') and hasattr(c,'reorder')][0]
orig=cls.reorder
def pr(self,run_id=0):
    orig(self,run_id); print('   chunker',cls.__name__,'mode=',self.mode,'order[:5]=',self.order[:5])
cls.reorder=pr
import neural_mi.run as RUN
rng=np.random.default_rng(0); T=1500
x=np.cumsum(rng.standard_normal((T,2)),0).astype('float32')*0.1; y=(x+0.5*rng.standard_normal((T,2))).astype('float32')
rg=nmi.Rigorous(gamma_range=[1,2],min_gamma_points=2)
kw=dict(training=nmi.Training(n_epochs=3,batch_size=64),show_progress=False,seed=0)
r1=nmi.run(x,y,mode='rigorous',rigorous=rg,**kw); r2=q.instantaneous_mi(x,y,rigorous=rg,**kw)
for r in (r1,r2):
    print(r.mode, r.dataframe.filter(regex='mi|gamma').to_string())
    d=getattr(r,'details',None)
    try:
        raw=r.raw_dataframe if hasattr(r,'raw_dataframe') else None
        print(raw.filter(regex='gamma|mi').head(6).to_string() if raw is not None else '')
    except Exception as e: print(e)
# random split with small eval subset
kw3=dict(training=nmi.Training(n_epochs=3,batch_size=64,max_eval_samples=50,test_subset_size=50),split=nmi.Split(mode='random'),show_progress=False,seed=0)
a=nmi.run(x,y,**kw3); b=q.instantaneous_mi(x,y,**kw3)
print('random split + eval subset: run',a.mi_estimate,a.dataframe.test_mi_mean[0],'inst',b.mi_estimate,b.dataframe.test_mi_mean[0])
