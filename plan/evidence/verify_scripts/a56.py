import sys, warnings, json
sys.path.insert(0,'/home/user/NeuralMI')
warnings.simplefilter('ignore')
import numpy as np
import neural_mi as nmi
from neural_mi import Training, Rigorous, Lag
rng=np.random.default_rng(0)
x=rng.normal(size=(600,3)).astype('float32'); y=(x+rng.normal(size=(600,3))).astype('float32')
T=Training(n_epochs=2,batch_size=64)
# A5 per_gamma
r=nmi.run(x,y,mode='rigorous',training=T,rigorous=Rigorous(gamma_range=range(1,4),min_gamma_points=3),show_progress=False,verbose=False)
blob=json.dumps(r.to_dict(),default=str)
print('rigorous runs cols:',list(r.runs.columns))
for k in ('per_gamma_train_mi_spread','per_gamma_ceiling_mi','per_gamma_saturation','saturated_gammas'):
    print(k,'in result:',k in blob)
# A6 pairwise, independent noise
xi=rng.normal(size=(300,3)).astype('float32')
r=nmi.run(xi,mode='pairwise',training=Training(n_epochs=2,batch_size=64),sweep_grid={'run_id':[0,1,2]},show_progress=False,verbose=False)
d=r.runs
print(d[['ch_x','ch_y','run_id','mi','train_mi','all_mi_negative']+[c for c in d.columns if c=='raw_train_mi']].to_string())
print('rows all_mi_negative with mi!=0:',int(((d.all_mi_negative==True)&(d.mi!=0)).sum()), 'of', int((d.all_mi_negative==True).sum()))
# A2
def go(mode,**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try: nmi.run(x,y,mode=mode,training=Training(n_epochs=1,batch_size=64,dataset_device='auto'),show_progress=False,verbose=False,**kw)
        except Exception as e: print('ERR',type(e).__name__,str(e)[:300])
    print(mode,'memory warning fired:',any('sequential tasks' in str(m.message) for m in w))
go('sweep',sweep_grid={'run_id':list(range(22))})
go('lag',lag=Lag(lag_range=range(-11,11)))
