"""A Gaussian model closer to real recordings than AR(1)+white noise, with exact answers.

X_t = S_t + Px_t + sigma * eps_x
Y_t = S_t + Py_t + sigma * eps_y

S, Px, Py are independent AR(2) oscillators (damping r, frequency f in cycles/sample),
i.e. smooth, band-limited, oversampled signals like LFP. S is shared; Px and Py are
private slow dynamics; sigma is a small white measurement noise.

Exact values come from the autocovariances (inverse FFT of each spectrum):
  * true MI rate (L, h -> inf): -1/2 <log(1 - |coh(w)|^2)>_w
  * finite-window chain-rule terms of mi_rate(h, L):
      J = I([X_{-L..L}, Y_{-h..-1}]; Y_0),   M = I(Y_{-h..-1}; Y_0),   J - M
All in nats.
"""
import numpy as np
from scipy.signal import lfilter

NGRID = 1 << 16


def ar2(r, f):
    """AR(2) coefficients (a1, a2) of a damped oscillator with poles r*exp(+-2 pi i f)."""
    return 2 * r * np.cos(2 * np.pi * f), -r * r


def spectrum(r, f, gain=1.0):
    w = np.linspace(-np.pi, np.pi, NGRID, endpoint=False)
    a1, a2 = ar2(r, f)
    return gain ** 2 / np.abs(1 - a1 * np.exp(-1j * w) - a2 * np.exp(-2j * w)) ** 2


def autocov(spec):
    """gamma(k) for k = 0..NGRID-1 from a spectrum sampled on [-pi, pi)."""
    return np.real(np.fft.ifft(np.fft.ifftshift(spec)))


class RealLike:
    def __init__(self, shared=(0.98, 0.02, 1.0), priv_x=(0.99, 0.005, 1.0),
                 priv_y=(0.99, 0.005, 1.0), sigma=0.05):
        self.shared, self.priv_x, self.priv_y, self.sigma = shared, priv_x, priv_y, sigma
        self.Ss, self.Sx_p, self.Sy_p = (spectrum(*shared), spectrum(*priv_x), spectrum(*priv_y))
        self.gs, self.gpx, self.gpy = autocov(self.Ss), autocov(self.Sx_p), autocov(self.Sy_p)

    # ----- exact values -----
    def rate(self):
        sx = self.Ss + self.Sx_p + self.sigma ** 2
        sy = self.Ss + self.Sy_p + self.sigma ** 2
        coh2 = self.Ss ** 2 / (sx * sy)
        return -0.5 * np.mean(np.log1p(-coh2))

    def _c(self, vi, oi, vj, oj):
        lag = abs(oj - oi)
        g = self.gs[lag]
        if vi == vj == 'x':
            return g + self.gpx[lag] + (self.sigma ** 2 if lag == 0 else 0.0)
        if vi == vj == 'y':
            return g + self.gpy[lag] + (self.sigma ** 2 if lag == 0 else 0.0)
        return g

    def _logdet(self, spec):
        if not spec:
            return 0.0
        M = np.array([[self._c(vi, oi, vj, oj) for (vj, oj) in spec] for (vi, oi) in spec])
        s, ld = np.linalg.slogdet(M)
        assert s > 0, "covariance not positive definite"
        return ld

    def mi(self, A, B, C=()):
        A, B, C = list(A), list(B), list(C)
        return 0.5 * (self._logdet(A + C) + self._logdet(B + C)
                      - self._logdet(A + B + C) - self._logdet(C))

    def terms(self, h, L):
        A = [('x', s) for s in range(-L, L + 1)]
        B = [('y', 0)]
        C = [('y', s) for s in range(-h, 0)]
        J = self.mi(A + C, B)
        M = self.mi(C, B) if C else 0.0
        return J, M, J - M

    # ----- samples -----
    def sample(self, T, seed=0, burn=5000):
        rng = np.random.default_rng(seed)

        def osc(params):
            r, f, gain = params
            a1, a2 = ar2(r, f)
            return lfilter([gain], [1, -a1, -a2], rng.standard_normal(T + burn))[burn:]

        s, px, py = osc(self.shared), osc(self.priv_x), osc(self.priv_y)
        x = s + px + self.sigma * rng.standard_normal(T)
        y = s + py + self.sigma * rng.standard_normal(T)
        return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


if __name__ == '__main__':
    m = RealLike()
    x, y = m.sample(20000, seed=0)
    print(f"true MI rate: {m.rate():.4f} nats/step   (std x={x.std():.2f}, y={y.std():.2f})")
    print(f"{'L':>3} {'h':>3} {'J':>8} {'M':>8} {'J-M':>8} {'amp':>6}")
    for L in (5, 20, 50):
        for h in (0, 1, 2, 3, 5, 10, 20):
            J, M, d = m.terms(h, L)
            print(f"{L:>3} {h:>3} {J:8.3f} {M:8.3f} {d:8.4f} {(J + M) / abs(d):6.1f}")
    print("InfoNCE ceilings: log(128)=%.2f, log(5000)=%.2f nats" % (np.log(128), np.log(5000)))
