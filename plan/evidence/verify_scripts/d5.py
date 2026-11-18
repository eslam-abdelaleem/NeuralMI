import warnings, numpy as np, sys, torch
sys.path.insert(0,'/home/user/NeuralMI')
warnings.simplefilter('ignore')
import neural_mi as nmi
from neural_mi.data.handler import create_dataset
from neural_mi.training import trainer as T
rng=np.random.default_rng(0)
x=rng.standard_normal((500,3)).astype('float32'); y=x[:,:2]+rng.standard_normal((500,2)).astype('float32')
pp={'window_size':0.5,'sample_rate':25.0}
orig=T.Trainer.train
def spy(self,dataset,*a,**k):
    print('  Trainer.train got', type(dataset).__name__, 'n=',len(dataset), 'x',tuple(dataset.x_dataset.data.shape),'y',tuple(dataset.y_dataset.data.shape), 'shift_windows=',k.get('shift_windows'))
    return orig(self,dataset,*a,**k)
T.Trainer.train=spy
for shift in (True,False):
    print('run shift_windows',shift)
    r=nmi.run(x,y,mode='estimate',processing=nmi.Processing(x='continuous',x_params=pp),training=nmi.Training(n_epochs=1,batch_size=8,shift_windows=shift),verbose=False)
ds=create_dataset(x,y,processor_type_x='continuous',processor_params_x=pp)
print('create_dataset', len(ds), tuple(ds.x_data.shape) if hasattr(ds.x_data,'shape') else tuple(ds.x_dataset.data.shape), 'x_window_width',ds.x_window_width)
# D3
from neural_mi.utils import _shift_data
xs=[np.sort(rng.uniform(0,20,300))]; yc=rng.standard_normal((2000,1)).astype('float32')
for sr_x in (None, 1000.0):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        a,b=_shift_data(xs,yc,0.5,'spike','continuous',sample_rate=sr_x if sr_x else 100.0)
    print('D3 _shift_data lag=0.5s y rows left', len(b))
import neural_mi.analysis.lag as L, inspect
