# neural_mi/generators/oracle.py
"""Exact ground truth for temporal information quantities.

Most generators in this package produce data whose mutual information is known
for one specific pairing. :class:`SharedLatentGaussian` is stronger: it exposes
the exact value of ``I(A; B | C)`` for *any* choice of processes and time
offsets, so every quantity in the temporal taxonomy (active information
storage, excess entropy, MI rate, transfer entropy, instantaneous exchange,
directed information rate, conditional TE, interaction information) has an
exact number to check an estimate against.

The model is a set of observed processes driven by one shared autoregressive
latent::

    Z_t = phi * Z_{t-1} + eta_t            eta ~ N(0, I),  Z_t in R^d
    V_t = M_V Z_t + eps_V                  eps_V ~ N(0, s_V^2 I)   for each V

Everything is jointly Gaussian and jointly stationary, so every quantity is a
log-determinant of a covariance block and is exact up to floating point. The
one exception is :meth:`mi_rate`, which is a spectral integral and therefore
exact up to quadrature.

A shared latent deliberately violates Massey's no-feedback condition: Y's past
informs Z's past, which informs Z_t, which informs X_t. The directed quantities
therefore converge to strictly smaller values than the symmetric MI rate, and
only a two-sided (acausal) window over X recovers the rate. That is a property
of the system rather than a defect, and it is what makes the system a useful
test of whether an estimator is measuring the estimand it claims to.
"""
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

LOG2 = np.log(2.0)

# numpy 2.x renamed trapz to trapezoid; support both.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

Spec = Sequence[Tuple[str, int]]


def _slogdet(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise np.linalg.LinAlgError(
            f"Covariance block is not positive definite (slogdet sign={sign}). "
            f"This usually means a process was given zero observation noise, or "
            f"the same (process, offset) pair appears twice in one spec."
        )
    return float(logdet)


class SharedLatentGaussian:
    """Jointly Gaussian processes sharing one AR(1) latent, with exact MI.

    Parameters
    ----------
    dims : dict of str to int, optional
        Observed dimensionality of each process, keyed by name. Defaults to
        ``{'x': 8, 'y': 8}``. Add a third entry (conventionally ``'w'``) for
        interaction information or conditional transfer entropy, which need a
        third process to be defined at all.
    d : int, optional
        Dimensionality of the shared latent, so the true number of shared
        factors. Defaults to 2.
    phi : float, optional
        AR(1) coefficient of the latent, in ``[0, 1)``. Sets the correlation
        timescale ``tau = -1 / log(phi)``, so ``phi=0.9`` gives roughly 9.5
        bins. Defaults to 0.9.
    noise : float or dict of str to float, optional
        Observation-noise standard deviation, either one value shared by all
        processes or one per process. Defaults to 1.0.
    coupling : float, optional
        Scales every projection matrix, which raises mutual information without
        changing the timescale. Defaults to 1.0.
    seed : int, optional
        Seeds the projection matrices. Defaults to 0.

    Attributes
    ----------
    tau : float
        Correlation time of the latent, in bins.
    names : tuple of str
        Process names, in the order given.

    Examples
    --------
    >>> from neural_mi.generators import SharedLatentGaussian
    >>> oracle = SharedLatentGaussian(dims={'x': 8, 'y': 8}, d=2, phi=0.9)
    >>> ais = oracle.exact(A=[('x', s) for s in range(-10, 0)], B=[('x', 0)])
    >>> samples = oracle.sample(T=20000, seed=0)
    >>> samples['x'].shape
    (20000, 8)
    """

    def __init__(self, dims: Optional[Dict[str, int]] = None, d: int = 2,
                 phi: float = 0.9, noise=1.0, coupling: float = 1.0,
                 seed: int = 0):
        if dims is None:
            dims = {'x': 8, 'y': 8}
        if not dims:
            raise ValueError("dims must name at least one process.")
        if not 0.0 <= phi < 1.0:
            raise ValueError(f"phi must lie in [0, 1) for a stationary latent, got {phi}.")

        rng = np.random.default_rng(seed)
        self.dims = OrderedDict(dims)
        self.names = tuple(self.dims)
        self.d = d
        self.phi = phi
        self.coupling = coupling

        if isinstance(noise, dict):
            missing = set(self.dims) - set(noise)
            if missing:
                raise ValueError(f"noise dict is missing entries for {sorted(missing)}.")
            self.noise = {v: float(noise[v]) for v in self.dims}
        else:
            self.noise = {v: float(noise) for v in self.dims}

        self.proj = {v: coupling * rng.normal(size=(n, d)) / np.sqrt(d)
                     for v, n in self.dims.items()}
        self.Sigma_Z = np.eye(d) / (1.0 - phi ** 2)
        self.tau = -1.0 / np.log(phi) if phi > 0 else 0.0

    # ------------------------------------------------------------------
    # covariance machinery
    # ------------------------------------------------------------------
    def dim(self, name: str) -> int:
        """Observed dimensionality of one process."""
        try:
            return self.dims[name]
        except KeyError:
            raise KeyError(
                f"Unknown process {name!r}. This oracle defines {list(self.dims)}. "
                f"Pass a larger dims dict to the constructor to add one."
            ) from None

    def _latent_cov(self, lag: int) -> np.ndarray:
        return (self.phi ** abs(lag)) * self.Sigma_Z

    def _cross_cov(self, v1: str, v2: str, lag: int) -> np.ndarray:
        """``E[v1_t v2_{t+lag}^T]``, including observation noise on the diagonal."""
        block = self.proj[v1] @ self._latent_cov(lag) @ self.proj[v2].T
        if v1 == v2 and lag == 0:
            block = block + (self.noise[v1] ** 2) * np.eye(self.dims[v1])
        return block

    def _cov_from_spec(self, spec: Spec) -> np.ndarray:
        # Validate up front so an unknown name reports which processes exist,
        # rather than surfacing a bare KeyError from the projection lookup.
        for (v, _) in spec:
            self.dim(v)
        rows = []
        for (v1, o1) in spec:
            rows.append(np.hstack([self._cross_cov(v1, v2, o2 - o1) for (v2, o2) in spec]))
        return np.vstack(rows)

    def _width(self, spec: Spec) -> int:
        return sum(self.dim(v) for (v, _) in spec)

    # ------------------------------------------------------------------
    # the primitive
    # ------------------------------------------------------------------
    def exact(self, A: Spec, B: Spec, C: Spec = ()) -> float:
        """Exact ``I(A; B | C)`` in bits.

        Each of ``A``, ``B`` and ``C`` is a sequence of ``(process, offset)``
        pairs, where offset is measured in time bins relative to a common
        reference. Negative offsets are past, zero is present, positive is
        future. An empty ``C`` gives the unconditional ``I(A; B)``.

        Every named quantity in the taxonomy is this function under a different
        offset pattern. Active information storage is
        ``exact(A=[('x', -k)..[('x', -1)], B=[('x', 0)])``; transfer entropy
        from x to y is the same A with ``B=[('y', 0)]`` and
        ``C=[('y', -k)..('y', -1)]``.

        Parameters
        ----------
        A, B : sequence of (str, int)
            The two groups whose shared information is measured.
        C : sequence of (str, int), optional
            The conditioning group. Empty by default.

        Returns
        -------
        float
            Mutual information in bits.
        """
        A, B, C = list(A), list(B), list(C)
        if not A or not B:
            raise ValueError("A and B must each contain at least one (process, offset) pair.")

        spec = A + B + C
        cov = self._cov_from_spec(spec)
        w_a, w_b = self._width(A), self._width(B)
        idx_a = np.arange(w_a)
        idx_b = np.arange(w_a, w_a + w_b)
        idx_c = np.arange(w_a + w_b, cov.shape[0])

        def logdet(idx):
            return 0.0 if len(idx) == 0 else _slogdet(cov[np.ix_(idx, idx)])

        return (logdet(np.r_[idx_a, idx_c]) + logdet(np.r_[idx_b, idx_c])
                - logdet(idx_c) - logdet(np.r_[idx_a, idx_b, idx_c])) / (2.0 * LOG2)

    # ------------------------------------------------------------------
    # named conveniences
    # ------------------------------------------------------------------
    def block_mi(self, w: int, a: Optional[str] = None, b: Optional[str] = None) -> float:
        """Exact ``I(A_1^w; B_1^w)`` in bits, the extensive block quantity.

        Grows without bound as ``w`` increases, approaching ``rate * w + b``.
        Use :meth:`affine_fit` to recover the slope and intercept.
        """
        a, b = self._default_pair(a, b)
        offsets = range(w)
        return self.exact(A=[(a, o) for o in offsets], B=[(b, o) for o in offsets])

    def mi_rate(self, a: Optional[str] = None, b: Optional[str] = None,
                n_omega: int = 4096) -> float:
        """Exact information rate in bits per bin, via the coherence integral.

        This is the intensive counterpart of :meth:`block_mi`, and equals the
        slope that :meth:`affine_fit` recovers. Exact up to quadrature, so
        ``n_omega`` controls the only approximation in this class.
        """
        a, b = self._default_pair(a, b)
        omega = np.linspace(-np.pi, np.pi, n_omega, endpoint=False)
        gain = (1 - self.phi ** 2) / (1 - 2 * self.phi * np.cos(omega) + self.phi ** 2)
        Ma, Mb = self.proj[a], self.proj[b]
        eye_a, eye_b = np.eye(self.dims[a]), np.eye(self.dims[b])
        MaS, MbS = Ma @ self.Sigma_Z, Mb @ self.Sigma_Z
        vals = np.empty(n_omega)
        for i, g in enumerate(gain):
            Saa = g * MaS @ Ma.T + self.noise[a] ** 2 * eye_a
            Sbb = g * MbS @ Mb.T + self.noise[b] ** 2 * eye_b
            Sab = g * MaS @ Mb.T
            joint = np.block([[Saa, Sab], [Sab.T, Sbb]])
            vals[i] = _slogdet(Saa) + _slogdet(Sbb) - _slogdet(joint)
        return float(_TRAPZ(vals, omega) / (4 * np.pi) / LOG2)

    def affine_fit(self, w_lo: int, w_hi: int, a: Optional[str] = None,
                   b: Optional[str] = None) -> Tuple[float, float]:
        """Fit ``block_mi(w) = rate * w + intercept`` over ``[w_lo, w_hi]``.

        The slope converges to :meth:`mi_rate` and the intercept to the
        subextensive predictive information. Fit well away from ``w=1``, where
        edge effects still dominate.

        Returns
        -------
        tuple of (float, float)
            Slope in bits per bin, and intercept in bits.
        """
        if w_hi <= w_lo:
            raise ValueError(f"Need w_hi > w_lo to fit a line, got {w_lo} and {w_hi}.")
        widths = np.arange(w_lo, w_hi + 1)
        values = np.array([self.block_mi(int(w), a, b) for w in widths])
        slope, intercept = np.polyfit(widths, values, 1)
        return float(slope), float(intercept)

    def _default_pair(self, a: Optional[str], b: Optional[str]) -> Tuple[str, str]:
        if a is None or b is None:
            if len(self.names) < 2:
                raise ValueError(
                    "This oracle defines a single process, so there is no pair to "
                    "measure. Name two processes in dims, or pass a and b explicitly."
                )
            a = a if a is not None else self.names[0]
            b = b if b is not None else self.names[1]
        self.dim(a), self.dim(b)
        return a, b

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def sample(self, T: int, seed: int = 0, burn: int = 500) -> Dict[str, np.ndarray]:
        """Draw ``T`` time steps from the model.

        Parameters
        ----------
        T : int
            Number of time steps to return.
        seed : int, optional
            Seeds the noise draws, independently of the constructor seed that
            fixed the projections. Defaults to 0.
        burn : int, optional
            Latent steps discarded so the returned series starts stationary.
            Defaults to 500.

        Returns
        -------
        dict of str to ndarray
            One array per process, each of shape ``(T, n_channels)``, which is
            the timepoints-first convention the library's processors expect.
        """
        rng = np.random.default_rng(seed)
        latent = np.empty((T + burn, self.d))
        z = rng.multivariate_normal(np.zeros(self.d), self.Sigma_Z)
        for t in range(T + burn):
            z = self.phi * z + rng.normal(size=self.d)
            latent[t] = z
        latent = latent[burn:]
        return {v: latent @ self.proj[v].T + self.noise[v] * rng.normal(size=(T, n))
                for v, n in self.dims.items()}


def generate_shared_latent_gaussian(T: int = 20000, dims: Optional[Dict[str, int]] = None,
                                    d: int = 2, phi: float = 0.9, noise=1.0,
                                    coupling: float = 1.0, seed: int = 0):
    """Sample from a :class:`SharedLatentGaussian` and return the oracle with it.

    A convenience for the common case of wanting data and its exact values
    together. Equivalent to constructing the oracle and calling
    :meth:`~SharedLatentGaussian.sample`.

    Returns
    -------
    tuple of (dict, SharedLatentGaussian)
        The sampled processes keyed by name, and the oracle that generated
        them, so any exact value can be queried afterwards.
    """
    oracle = SharedLatentGaussian(dims=dims, d=d, phi=phi, noise=noise,
                                  coupling=coupling, seed=seed)
    return oracle.sample(T, seed=seed), oracle
