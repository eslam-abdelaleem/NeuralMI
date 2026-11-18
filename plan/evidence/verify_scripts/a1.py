import sys, warnings, logging
sys.path.insert(0,'/home/user/NeuralMI')
warnings.simplefilter('ignore')
import numpy as np, torch
import neural_mi as nmi
import neural_mi.device as dev
import neural_mi.analysis.task as task
import neural_mi.analysis.sweep as sweep
calls=[]; seen=[]
dev.accelerator=lambda:'cuda'
def fake_choose(tasks,n): calls.append(len(tasks)); return 'cpu'   # stands in for "GPU chosen"; marker
dev.choose_device=fake_choose
orig=sweep.run_training_task
def spy(t): seen.append(t[2].get('device','<absent>')); return orig(t)
sweep.run_training_task=spy
rng=np.random.default_rng(0)
x=rng.normal(size=(400,4)).astype('float32'); y=(x+rng.normal(size=(400,4))).astype('float32')
from neural_mi import Training
def go(mode,**kw):
    calls.clear(); seen.clear()
    try:
        nmi.run(x,y,mode=mode,training=Training(n_epochs=2,batch_size=64),show_progress=False,verbose=False,**kw)
    except Exception as e:
        print(mode,'ERR',type(e).__name__,str(e)[:300])
    print(mode,'choose_device calls:',len(calls),'| task device values:',sorted(set(map(str,seen))), 'n_tasks',len(seen))
go('estimate')
from neural_mi import Lag
go('lag',lag=Lag(lag_range=range(-1,2)))
go('dimensionality')
