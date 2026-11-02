"""Does shrinking the conditioning term rescue mi_rate?  Pre-whiten each signal with its own AR(p) fit.

An invertible linear filter on each process leaves the (infinite-window) MI rate unchanged, since
coherence is filter-invariant, but it removes most of Y's own-past predictability, i.e. the large
marginal term M that the rate was a small residual of.

Reference values for the finite windows (h, L) come from a long simulation: the processes are
Gaussian, so the Gaussian log-det MI of the population covariance is exact, and with T_ref = 2e6
rows its sampling error is ~1e-4 nats.
"""
import sys, time, warnings
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
from scipy.signal import lfilter
import neural_mi as nmi
from reallike import RealLike
from harness import run_rate

warnings.simplefilter('ignore')
P = 30


def ar_fit(s, p=P):
    """Yule-Walker AR(p) coefficients of a 1-D series (minimum phase)."""
    s = s - s.mean()
    r = np.array([np.dot(s[:len(s) - k], s[k:]) / len(s) for k in range(p + 1)])
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    return np.linalg.solve(R, r[1:p + 1])


def whiten(s, a):
    return lfilter(np.r_[1.0, -a], [1.0], s - s.mean())[P:]


def gauss_terms(x, y, h, L):
    """Gaussian chain-rule terms J, M, J-M from rows of long series (x, y 1-D)."""
    T = len(x)
    lo, hi = max(L, h), T - L
    idx = np.arange(lo, hi)
    A = np.stack([x[idx + s] for s in range(-L, L + 1)], 1)
    C = np.stack([y[idx + s] for s in range(-h, 0)], 1) if h else np.empty((len(idx), 0))
    B = y[idx][:, None]

    def ld(*blocks):
        Z = np.hstack(blocks)
        return np.linalg.slogdet(np.cov(Z, rowvar=False).reshape(Z.shape[1], Z.shape[1]))[1]

    J = 0.5 * (ld(A, C) + ld(B) - ld(A, C, B))
    M = 0.5 * (ld(C) + ld(B) - ld(C, B)) if h else 0.0
    return J, M, J - M


m = RealLike()
# 1) validate the large-sample reference against the exact values on raw data
xr, yr = m.sample(2_000_000, seed=123)
J, M, d = gauss_terms(xr[:, 0].astype(float), yr[:, 0].astype(float), 5, 20)
Je, Me, de = m.terms(5, 20)
print(f"reference check (raw, h=5, L=20): large-sample J={J:.3f} M={M:.3f} J-M={d:.4f} | exact {Je:.3f} {Me:.3f} {de:.4f}", flush=True)

# 2) whitening filters fitted on the analysis recording itself (what a user can do)
x, y = m.sample(20000, seed=0)
ax, ay = ar_fit(x[:, 0].astype(float)), ar_fit(y[:, 0].astype(float))
xw = whiten(x[:, 0].astype(float), ax); yw = whiten(y[:, 0].astype(float), ay)
xw, yw = (xw / xw.std()).astype(np.float32)[:, None], (yw / yw.std()).astype(np.float32)[:, None]

# exact-enough finite-window targets for the whitened processes (same filters on a long run)
xrw, yrw = whiten(xr[:, 0].astype(float), ax), whiten(yr[:, 0].astype(float), ay)
print(f"true MI rate (infinite windows, unchanged by whitening): {m.rate():.4f}", flush=True)
for h, L in ((5, 20),):
    Jw, Mw, dw = gauss_terms(xrw, yrw, h, L)
    print(f"whitened targets h={h} L={L}: J={Jw:.3f} M={Mw:.3f} J-M={dw:.4f} amp={(Jw + Mw) / dw:.1f}", flush=True)
    for s in (0, 1, 2):
        recs, msgs = run_rate(xw, yw, h=h, half_width=L, seed=s)
        r = recs[0]
        est = r['mi_raw'] if r['mi_raw'] is not None else r['mi']
        print(f"  library defaults, seed {s}: est={est:+.4f} (target {dw:.4f}) | J {r['J']:.3f}/{Jw:.3f} "
              f"M {r['M']:.3f}/{Mw:.3f} | diag J={r.get('J_diag')} M={r.get('M_diag')}", flush=True)
