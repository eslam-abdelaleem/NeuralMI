"""Exact Gaussian values of mi_rate's two chain-rule terms for a shared-latent AR(1).

Model (same as tests/test_branched_embedding.py::_SharedLatentOracle):
    Z_t = phi Z_{t-1} + eta_t,  eta ~ N(0, 1)
    X_t = a Z_t + sx * eps_x,   Y_t = b Z_t + sy * eps_y

mi_rate(h, L) = I(X_{-L..L}; Y_0 | Y_{-h..-1})
             = I([X_{-L..L}, Y_{-h..-1}]; Y_0)  -  I(Y_{-h..-1}; Y_0)
                       joint term J                    marginal term M
All values in nats.
"""
import numpy as np


class Oracle:
    def __init__(self, phi, a=1.0, b=1.0, sx=0.5, sy=0.5):
        self.phi, self.a, self.b, self.sx, self.sy = phi, a, b, sx, sy

    def cz(self, lag):
        return self.phi ** abs(lag) / (1 - self.phi ** 2)

    def cov(self, spec):
        n = len(spec)
        M = np.empty((n, n))
        for i, (vi, oi) in enumerate(spec):
            for j, (vj, oj) in enumerate(spec):
                lag = oj - oi
                c = self.cz(lag)
                if vi == vj == 'x':
                    M[i, j] = self.a ** 2 * c + (self.sx ** 2 if lag == 0 else 0.0)
                elif vi == vj == 'y':
                    M[i, j] = self.b ** 2 * c + (self.sy ** 2 if lag == 0 else 0.0)
                else:
                    M[i, j] = self.a * self.b * c
        return M

    def logdet(self, spec):
        if not spec:
            return 0.0
        sign, ld = np.linalg.slogdet(self.cov(spec))
        assert sign > 0
        return ld

    def mi(self, A, B, C=()):
        A, B, C = list(A), list(B), list(C)
        return 0.5 * (self.logdet(A + C) + self.logdet(B + C)
                      - self.logdet(A + B + C) - self.logdet(C))

    def terms(self, h, L):
        A = [('x', s) for s in range(-L, L + 1)]
        B = [('y', 0)]
        C = [('y', s) for s in range(-h, 0)]
        joint = self.mi(A + C, B)
        marginal = self.mi(C, B) if C else 0.0
        return joint, marginal, joint - marginal

    def spectral_rate(self, n=1 << 16):
        w = np.linspace(-np.pi, np.pi, n, endpoint=False)
        sz = 1.0 / np.abs(1 - self.phi * np.exp(-1j * w)) ** 2
        sx = self.a ** 2 * sz + self.sx ** 2
        sy = self.b ** 2 * sz + self.sy ** 2
        coh2 = (self.a * self.b * sz) ** 2 / (sx * sy)
        return -np.mean(np.log1p(-coh2)) / 2.0   # nats per step


def report(name, o, hs=(0, 1, 2, 3, 5, 10, 20), Ls=(5, 20)):
    print(f"\n=== {name}: phi={o.phi} (tau={-1/np.log(o.phi):.1f} steps), "
          f"a={o.a}, b={o.b}, sx={o.sx}, sy={o.sy}")
    print(f"    true MI rate (L,h->inf, spectral): {o.spectral_rate():.4f} nats/step")
    for L in Ls:
        print(f"  L(half_width)={L}")
        print(f"    {'h':>3} {'joint J':>9} {'marg M':>9} {'J-M':>8} {'amp':>7}")
        for h in hs:
            J, M, d = o.terms(h, L)
            amp = (abs(J) + abs(M)) / abs(d) if d else float('inf')
            print(f"    {h:>3} {J:9.4f} {M:9.4f} {d:8.4f} {amp:7.1f}")


if __name__ == '__main__':
    report("library's own accuracy-test regime", Oracle(0.85, sx=0.5, sy=0.5), Ls=(5, 20))
    report("smooth, oversampled (real-data-like)", Oracle(0.98, sx=0.3, sy=0.3), Ls=(5, 20))
    report("very smooth, low noise (e.g. LFP / calcium)", Oracle(0.995, sx=0.1, sy=0.1), Ls=(20,))
    print("\nInfoNCE ceilings: log(128)=%.2f nats (training batch), log(5000)=%.2f nats "
          "(evaluation block)" % (np.log(128), np.log(5000)))
