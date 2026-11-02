import numpy as np
from scipy.signal import lfilter
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


