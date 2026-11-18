import sys, warnings; sys.path.insert(0,'/home/user/NeuralMI'); warnings.simplefilter('ignore')
import numpy as np, neural_mi as nmi
from neural_mi.defaults import BASE_PARAMS_SCHEMA as S
rng=np.random.default_rng(0); x=rng.normal(size=(300,3)).astype('float32'); y=x.copy()
r=nmi.run(x,y,training=nmi.Training(n_epochs=2,batch_size=64),seed=1,show_progress=False)
bp=r.params['base_params']
for k in ('estimator_params','optimizer_params','scheduler_params','augmentation_params'):
    print(k, k in bp, bp.get(k) is S[k]['default'])
bp['optimizer_params']['lr']=123
print("schema now:", S['optimizer_params']['default'])
