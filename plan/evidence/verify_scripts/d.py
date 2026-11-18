import math, numpy as np, inspect, torch, warnings
warnings.simplefilter('ignore')
print('const eq', 1/np.log(2)==1.0/math.log(2), np.log(2.0)==math.log(2), float(1/np.log(2)))
import neural_mi.generators as g
def desc(o):
    if isinstance(o,tuple): return '('+', '.join(desc(i) for i in o)+')'
    if isinstance(o,dict): return '{'+', '.join(f'{k}:{desc(v)}' for k,v in list(o.items())[:4])+'}'
    if isinstance(o,torch.Tensor): return f'torch{tuple(o.shape)}'
    if isinstance(o,np.ndarray): return f'np{o.shape}' + ('[obj]' if o.dtype==object else '')
    if isinstance(o,list): return f'list[{len(o)}]'
    return type(o).__name__
for n in g.__all__:
    f=getattr(g,n)
    if inspect.isclass(f): print(n,'CLASS', inspect.signature(f.sample)); continue
    sig=inspect.signature(f); ps=sig.parameters
    size=[p for p in ps if p in('n_samples','n_windows','T','n')]
    ut={k:ps[k].default for k in ps if 'torch' in k}
    kw={}
    for p,v in ps.items():
        if v.default is inspect._empty:
            kw[p]={'n_samples':200,'T':200,'n_windows':200,'dim':2,'mi':1.0,'n_levels':3,'rho':0.5,'p':np.full((2,2),.25),'latent_dim':2,'observed_dim':4,'z_dim':2,'x_dim':4,'y_dim':4}.get(p,2)
    try: out=desc(f(**kw))
    except Exception as e: out='ERR '+str(e)[:120]
    print(f'{n} | size={size} | required={list(kw)} | torch-arg={ut} | returns {out}')
