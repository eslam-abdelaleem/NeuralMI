import sys, warnings
sys.path.insert(0,'/home/user/NeuralMI')
import numpy as np
import neural_mi as nmi
from neural_mi import Training, Lag
rng=np.random.default_rng(0)
x=rng.normal(size=(300,3)).astype('float32'); y=(x+rng.normal(size=(300,3))).astype('float32')
def go(mode,**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try: nmi.run(x,y,mode=mode,training=Training(n_epochs=1,batch_size=64,dataset_device='auto'),show_progress=False,verbose=False,**kw)
        except Exception as e: print('ERR',type(e).__name__,str(e)[:300])
    print(mode,'memory warning fired:',any('sequential tasks' in str(m.message) for m in w))
go('estimate',sweep_grid={'run_id':list(range(22))})
go('lag',lag=Lag(lag_range=range(-11,11)))
