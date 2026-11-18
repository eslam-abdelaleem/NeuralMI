import warnings, numpy as np, torch
warnings.simplefilter('ignore')
import sys; sys.path.insert(0,'/home/user/NeuralMI')
from neural_mi.data import create_dataset
from neural_mi.data.plan import RowPlan
rng=np.random.default_rng(0)
def shp(w): return tuple((w[0] if isinstance(w,tuple) else w).shape)
for label,pp,n in [('25Hz w=0.5',{'window_size':0.5,'sample_rate':25.0},500),('10Hz w=0.25',{'window_size':0.25,'sample_rate':10.0},500)]:
    x=rng.standard_normal((n,3)).astype('float32'); y=rng.standard_normal((n,2)).astype('float32')
    for shift in (True,False):
        p=RowPlan.of_pair(x,y,dict(processor_type_x='continuous',processor_params_x=pp,shift_windows=shift))
        r=p.rows(); print(label,'RowPlan shift',shift,p.route,{k:shp(v) for k,v in r['windows'].items()},'spans',r['spans'],'step',np.diff(r['times'][:3]))
    ds=create_dataset(x,y,processor_type_x='continuous',processor_params_x=pp)
    xw,yw=ds[0][:2] if False else (ds.x_data, ds.y_data)
    print(label,'create_dataset',type(ds).__name__,len(ds),[tuple(getattr(a,'shape',())) for a in ds[0]] if isinstance(ds[0],(tuple,list)) else ds[0])
    # grid route via RowPlan: y starts one sample late (offset start) 
    sr=pp['sample_rate']; tx=np.arange(n)/sr; ty=np.arange(n)/sr+1.0/sr
    pp2={'window_size':pp['window_size']}
    for shift in (True,False):
        p=RowPlan.of_pair(x,y,dict(processor_type_x='continuous',processor_params_x=pp2,x_time=tx,y_time=ty,shift_windows=shift))
        r=p.rows(); print(label,'RowPlan offset-start shift',shift,p.route,{k:shp(v) for k,v in r['windows'].items()},'step',np.diff(r['times'][:3]))
    p=RowPlan.of_pair(x,y,dict(processor_type_x='continuous',processor_params_x=pp2,x_time=tx,y_time=tx,shift_windows=True))
    r=p.rows(); print(label,'RowPlan same time vec',p.route,{k:shp(v) for k,v in r['windows'].items()},'step',np.diff(r['times'][:3]))
