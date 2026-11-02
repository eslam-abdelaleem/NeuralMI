"""What rows does block_mi actually train on? Shapes, and a linear (Gaussian) MI on those rows."""
import sys, warnings
import numpy as np
sys.path.insert(0, sys.argv[1])
warnings.simplefilter('ignore')
from neural_mi.data.handler import create_dataset
from reallike import RealLike

real = RealLike()
x, y = real.sample(20000, seed=0)
x, y = x / x.std(), y / y.std()


def gauss_mi(a, b):
    a = a.reshape(len(a), -1).astype(float); b = b.reshape(len(b), -1).astype(float)
    C = np.cov(np.hstack([a, b]), rowvar=False)
    da = a.shape[1]
    ld = lambda M: np.linalg.slogdet(M)[1]
    return 0.5 * (ld(C[:da, :da]) + ld(C[da:, da:]) - ld(C))


for w in (5, 10, 20):
    ds = create_dataset(x_data=x, y_data=y, processor_type_x='continuous', processor_type_y='continuous',
                        processor_params_x={'window_size': w}, processor_params_y={'window_size': w})
    X = ds.x_data.numpy(); Y = ds.y_data.numpy()
    first = X[0, 0, :min(6, X.shape[2])]
    print(f"w={w}: x rows {X.shape}, y rows {Y.shape}; first row starts {np.round(first, 3)} vs raw {np.round(x[:6, 0], 3)}")
    print(f"      linear-Gaussian MI on these rows: {gauss_mi(X, Y):.3f} nats (exact {dict(((5, 0.574), (10, 1.418), (20, 3.102)))[w]})")
