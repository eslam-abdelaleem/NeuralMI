"""Two routes to the MI rate, compared on exact values.

Block route: I(w) = I(X_{1:w}; Y_{1:w}) = rate*w + b once w passes the dependence timescale.
  The rate is a slope, (I(w2) - I(w1)) / (w2 - w1): a difference of two estimates whose signal
  grows with w2 - w1.
Chain route (mi_rate): I(X_{-L..L}; Y_0 | Y_{-h..-1}) = J - M, a difference whose signal is one
  step of rate, against M = I(Y_past; Y_0).

Amplification of a two-term difference = (|t1| + |t2|) / |t1 - t2|.
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from reallike import RealLike
from exact_terms import Oracle
from whiten_helpers import ar_fit, whiten, gauss_terms


def block_exact(model, w):
    A = [('x', s) for s in range(w)]
    B = [('y', s) for s in range(w)]
    return model.mi(A, B)


def block_gauss(x, y, w):
    """Gaussian block MI from long series (large-sample reference)."""
    n = len(x) - w + 1
    X = np.lib.stride_tricks.sliding_window_view(x, w)[:n]
    Y = np.lib.stride_tricks.sliding_window_view(y, w)[:n]
    Z = np.hstack([X, Y])
    C = np.cov(Z, rowvar=False)
    ld = lambda M: np.linalg.slogdet(M)[1]
    return 0.5 * (ld(C[:w, :w]) + ld(C[w:, w:]) - ld(C))


def report(name, curve, rate, chain_terms):
    ws = sorted(curve)
    print(f"\n=== {name}: rate {rate:.4f} nats/step")
    print("   w   I(w)   I(w)/w   slope over previous w   (rate*w + b => b)")
    prev = None
    for w in ws:
        I = curve[w]
        slope = (I - curve[prev]) / (w - prev) if prev else float('nan')
        print(f"  {w:3d} {I:6.3f}  {I / w:6.3f}   {slope:7.4f}                {I - rate * w:+.3f}")
        prev = w
    print("  two-point slope estimates (w1 -> w2): slope, amplification of the difference")
    for w1, w2 in ((5, 10), (10, 20), (10, 30), (20, 40)):
        if w1 in curve and w2 in curve:
            d = curve[w2] - curve[w1]
            print(f"    {w1:2d} -> {w2:2d}: slope {d / (w2 - w1):.4f}  amp {(curve[w1] + curve[w2]) / d:5.1f}"
                  f"   (I(w2) = {curve[w2]:.2f} nats)")
    J, M = chain_terms
    print(f"  chain route at h=5, L=20: J={J:.3f} M={M:.3f} J-M={J - M:.4f} amp {(J + M) / (J - M):.1f}")


ws = [1, 2, 5, 10, 15, 20, 30, 40, 60]
easy = Oracle(0.85, sx=0.5, sy=0.5)
report("easy (library test regime)", {w: block_exact(easy, w) for w in ws}, easy.spectral_rate(),
       easy.terms(5, 20)[:2])

real = RealLike()
report("realistic (raw)", {w: block_exact(real, w) for w in ws}, real.rate(), real.terms(5, 20)[:2])

# whitened realistic: large-sample Gaussian reference (filters fitted on the 20k analysis sample)
x, y = real.sample(20000, seed=0)
ax, ay = ar_fit(x[:, 0].astype(float)), ar_fit(y[:, 0].astype(float))
xr, yr = real.sample(2_000_000, seed=123)
xw, yw = whiten(xr[:, 0].astype(float), ax), whiten(yr[:, 0].astype(float), ay)
Jw, Mw, _ = gauss_terms(xw, yw, 5, 20)
report("realistic, pre-whitened (large-sample Gaussian)", {w: block_gauss(xw, yw, w) for w in ws[:-1]},
       real.rate(), (Jw, Mw))
