import matplotlib; matplotlib.use('Agg')
import warnings, numpy as np, torch, inspect
import neural_mi as nmi
from neural_mi import quantities as q
rng=np.random.default_rng(0)
T=400
x=rng.standard_normal((T,2)).astype('float32'); y=(x+0.5*rng.standard_normal((T,2))).astype('float32')
tr=nmi.Training(n_epochs=3, batch_size=64)
print("=== F1")
try:
    r=nmi.run(x,y,training=tr,output=nmi.Output(track_embeddings=True),random_seed=0,verbose=False) if 'random_seed' in inspect.signature(nmi.run).parameters else nmi.run(x,y,training=tr,output=nmi.Output(track_embeddings=True))
    d=r._repeat_view(None,None,what='x')
    hx,hy=d['embedding_history_x'],d['embedding_history_y']
    print(len(hx),hx[0].shape,len(hy),hy[0].shape, np.allclose(hx[0],hy[0]))
    n=hx[0].shape[0]
    labels=np.array(['a','b']*(n//2)+['a']*(n%2))
    import matplotlib.pyplot as plt
    anim=r.animate(embedding_labels={'cls':labels},show=False,reduction='none')
    fig=plt.gcf()
    for ax in fig.axes:
        print('ax title=',repr(ax.get_title()),'legend=',ax.get_legend() is not None,'label=',ax.get_label(), 'ncoll', len(ax.collections))
    emb=[ax for ax in fig.axes if 'Embeddings' in ax.get_title()][0]
    off=emb.collections[0].get_offsets()
    print('offsets==x0',np.allclose(off,hx[0][:,:2]),'==y0',np.allclose(off,hy[0][:,:2]))
except Exception as e:
    import traceback; traceback.print_exc()
print("=== F4")
for kw in [dict(transfer={'prediction_horizon':2}), dict(transfer=nmi.Transfer(prediction_horizon=2)) if hasattr(nmi,'Transfer') else {}, dict(prediction_horizon=2)]:
    try:
        r=q.transfer_entropy(x,y,history_window=2,training=tr,show_progress=False,**kw)
        print(list(kw), 'OK', r.params.get('prediction_horizon'), {k:v for k,v in r.params.items() if 'horizon' in k or 'history' in k})
    except Exception as e:
        print(list(kw), type(e).__name__, str(e)[:300])
