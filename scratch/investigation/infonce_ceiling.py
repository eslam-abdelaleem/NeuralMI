"""InfoNCE with the *optimal* critic, for a scalar Gaussian pair of known MI, at several numbers of negatives.

f*(x, y) = log p(y | x) - log p(y). Even this perfect critic cannot report more than log K,
and it reports noticeably less well before MI reaches log K.
"""
import numpy as np


def infonce_optimal(mi, K, n_batches=40, seed=0):
    rng = np.random.default_rng(seed)
    rho = np.sqrt(1 - np.exp(-2 * mi))
    vals = []
    for _ in range(n_batches):
        x = rng.standard_normal(K)
        y = rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal(K)
        s2 = 1 - rho ** 2
        # log p(y_j | x_i) - log p(y_j), K x K
        f = (-(y[None, :] - rho * x[:, None]) ** 2 / (2 * s2) - 0.5 * np.log(s2)
             + y[None, :] ** 2 / 2)
        m = f.max(axis=1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(f - m).sum(axis=1))) - np.log(K)
        vals.append(np.mean(np.diag(f) - lse))
    return float(np.mean(vals))


print(f"{'true MI':>8} | " + " ".join(f"K={K:<5}" for K in (128, 512, 5000)))
for mi in (0.2, 1.0, 2.0, 3.0, 3.7, 4.0, 4.66, 4.83, 6.0):
    row = [infonce_optimal(mi, K, n_batches=40 if K < 5000 else 4) for K in (128, 512, 5000)]
    print(f"{mi:8.2f} | " + " ".join(f"{v:7.3f}" for v in row))
print("log K:    " + " ".join(f"{np.log(K):7.3f}" for K in (128, 512, 5000)))
print("\nJ - M at the true values 4.83 - 4.66 = 0.17 nats; read through the perfect critic:")
for K in (128, 512, 5000):
    j, m = infonce_optimal(4.83, K, 40 if K < 5000 else 4), infonce_optimal(4.66, K, 40 if K < 5000 else 4, seed=1)
    print(f"  K={K:<5} J={j:.3f}  M={m:.3f}  J-M={j - m:+.3f}")
