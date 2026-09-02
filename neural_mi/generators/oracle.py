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
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from contextlib import contextmanager

import numpy as np
import torch

LOG2 = np.log(2.0)

# numpy 2.x renamed trapz to trapezoid; support both.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

Spec = Sequence[Tuple[str, int]]


@contextmanager
def _torch_seed(seed: int):
    """Seed torch's global RNG for the duration of the block, then restore it.

    ``torch.nn.Linear`` draws its initial weights from the global generator and
    takes no ``generator=`` argument, so a generator that builds a network has
    no way to seed it locally. Saving and restoring the state keeps that from
    leaking into the caller's stream.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


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


# ---------------------------------------------------------------------------
# Closed-form generators
#
# Each of these fixes the mutual information by construction, so a sample
# comes with the number an estimator is supposed to recover. The windowed
# pair report the *observed* MI, computed from the SNR, rather than the
# latent MI they were built from.
# ---------------------------------------------------------------------------

def mi_to_rho(dim: int, mi: float) -> float:
    """Calculates the correlation coefficient `rho` for a given MI and dimension.

    This function is used for generating correlated Gaussian variables with a
    pre-defined mutual information. The formula is derived from the analytical
    expression for MI between two multivariate Gaussian variables.

    Parameters
    ----------
    dim : int
        The dimension of the Gaussian variables.
    mi : float
        The desired mutual information in bits.

    Returns
    -------
    float
        The corresponding correlation coefficient `rho`.
    """
    # Convert MI from bits to nats for the formula
    mi_nats = mi * np.log(2)
    return np.sqrt(1 - np.exp(-2.0 / dim * mi_nats))


def generate_correlated_gaussians(
    n_samples: int, dim: int, mi: float, use_torch: bool = True, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two correlated multivariate Gaussian datasets.

    The ground truth mutual information between these two variables can be
    calculated analytically.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    dim : int
        The number of dimensions for each variable.
    mi : float
        The ground truth mutual information in bits.
    use_torch : bool, optional
        If True, returns torch.Tensors; otherwise, returns NumPy arrays.
        Defaults to True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - **x** (*np.ndarray* or *torch.Tensor*): The first dataset, of shape `(n_samples, dim)`.
        - **y** (*np.ndarray* or *torch.Tensor*): The second dataset, of shape `(n_samples, dim)`.
    """
    rho = mi_to_rho(dim, mi)
    mean = np.zeros(2 * dim)
    cov = np.eye(2 * dim)
    cov[dim:, :dim] = np.eye(dim) * rho
    cov[:dim, dim:] = np.eye(dim) * rho
    
    data = np.random.default_rng(seed).multivariate_normal(mean, cov, size=n_samples)
    x = data[:, :dim]
    y = data[:, dim:]
    
    if use_torch:
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y


def generate_windowed_oscillatory(
    n_windows: int,
    n_channels: int = 1,
    window_size: int = 256,
    f_carrier_hz: float = 10.0,
    sample_rate: float = 512.0,
    latent_mi: float = 1.0,
    snr: float = 3.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate IID windows of amplitude-modulated oscillations with known MI.

    Each window pair ``(X[i], Y[i])`` shares a scalar latent amplitude drawn
    from correlated Gaussians.  The observable is:

        X[i, ch, t] = z_x[i] * sin(2π f t / fs) + ε_t

    The MI between X and Y is analytically computable from the SNR:

        ρ_obs = ρ_latent * v² / (v² + σ²/1)
        I_obs = −½ log₂(1 − ρ_obs²) per channel

    where v is the carrier template norm and σ = amplitude_std / snr.

    Parameters
    ----------
    n_windows : int
        Number of independent windows.
    n_channels : int, optional
        Number of channels. Defaults to 1.
    window_size : int, optional
        Number of timepoints per window. Defaults to 256.
    f_carrier_hz : float, optional
        Carrier frequency in Hz. Defaults to 10.0.
    sample_rate : float, optional
        Sampling rate in Hz. Defaults to 512.0.
    latent_mi : float, optional
        Desired MI in bits between the scalar latents. Defaults to 1.0.
    snr : float, optional
        Signal amplitude relative to noise std. Defaults to 3.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        ``(X, Y, true_mi)`` where X and Y have shape
        ``(n_windows, n_channels, window_size)`` and ``true_mi`` is in bits.
    """
    rho = mi_to_rho(1, latent_mi)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    _rng = np.random.default_rng(seed)
    latents = _rng.multivariate_normal([0.0, 0.0], cov, size=(n_windows, n_channels))
    z_x = latents[:, :, 0]  # (n_windows, n_channels) — independent per channel
    z_y = latents[:, :, 1]  # (n_windows, n_channels)

    t = np.arange(window_size) / sample_rate
    carrier = np.sin(2.0 * np.pi * f_carrier_hz * t)  # (window_size,)
    v_sq = float(np.dot(carrier, carrier))             # ||v||²

    noise_std = 1.0 / snr
    X = z_x[:, :, None] * carrier[None, None, :]      # (n_windows, n_channels, window_size)
    Y = z_y[:, :, None] * carrier[None, None, :]
    X = X + noise_std * _rng.standard_normal(X.shape)
    Y = Y + noise_std * _rng.standard_normal(Y.shape)

    # Analytical observable MI per channel
    sigma_sq = noise_std ** 2
    rho_obs = rho * v_sq / (v_sq + sigma_sq)
    rho_obs = float(np.clip(rho_obs, -1 + 1e-8, 1 - 1e-8))
    mi_per_channel = -0.5 * np.log2(1.0 - rho_obs ** 2)
    true_mi = float(n_channels * mi_per_channel)

    return X.astype(np.float32), Y.astype(np.float32), true_mi


def generate_windowed_multichannel(
    n_windows: int,
    n_channels: int = 8,
    window_size: int = 200,
    f_min_hz: float = 4.0,
    f_max_hz: float = 40.0,
    sample_rate: float = 500.0,
    latent_mi: float = 0.5,
    snr: float = 3.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate IID multi-channel windows where each channel has a different carrier.

    Channel ``c`` uses carrier frequency
    ``f_c = f_min + c * (f_max - f_min) / (n_channels - 1)``.
    The per-channel latents are independent: ``(z_{x,c}, z_{y,c})`` are drawn
    independently for each channel from correlated Gaussians with MI ``latent_mi``.
    Total observable MI = sum of per-channel observable MIs.

    Each channel's MI lives at a different frequency, so this is useful for
    validating estimator behaviour on multi-channel data where naively mixing
    channels would create cross-channel interference.

    Parameters
    ----------
    n_windows : int
        Number of independent windows.
    n_channels : int, optional
        Number of channels. Defaults to 8.
    window_size : int, optional
        Number of timepoints per window. Defaults to 200.
    f_min_hz : float, optional
        Carrier frequency for channel 0 in Hz. Defaults to 4.0.
    f_max_hz : float, optional
        Carrier frequency for the last channel in Hz. Defaults to 40.0.
    sample_rate : float, optional
        Sampling rate in Hz. Defaults to 500.0.
    latent_mi : float, optional
        Desired MI per channel in bits. Defaults to 0.5.
    snr : float, optional
        Signal amplitude relative to noise std. Defaults to 3.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        ``(X, Y, true_mi)`` where X and Y have shape
        ``(n_windows, n_channels, window_size)`` and ``true_mi`` is total bits.
    """
    rho = mi_to_rho(1, latent_mi)
    t = np.arange(window_size) / sample_rate
    noise_std = 1.0 / snr

    n_ch = max(n_channels, 2)
    _rng = np.random.default_rng(seed)
    freqs = [f_min_hz + c * (f_max_hz - f_min_hz) / (n_ch - 1) for c in range(n_channels)]

    X = np.zeros((n_windows, n_channels, window_size), dtype=np.float32)
    Y = np.zeros((n_windows, n_channels, window_size), dtype=np.float32)
    total_mi = 0.0

    for c, fc in enumerate(freqs):
        carrier = np.sin(2.0 * np.pi * fc * t)
        v_sq = float(np.dot(carrier, carrier))
        cov = np.array([[1.0, rho], [rho, 1.0]])
        latents = _rng.multivariate_normal([0.0, 0.0], cov, size=n_windows)
        z_x, z_y = latents[:, 0], latents[:, 1]
        X[:, c, :] = (z_x[:, None] * carrier[None, :] +
                      noise_std * _rng.standard_normal((n_windows, window_size)))
        Y[:, c, :] = (z_y[:, None] * carrier[None, :] +
                      noise_std * _rng.standard_normal((n_windows, window_size)))
        rho_obs = float(np.clip(rho * v_sq / (v_sq + noise_std ** 2), -1 + 1e-8, 1 - 1e-8))
        total_mi += -0.5 * np.log2(1.0 - rho_obs ** 2)

    return X, Y, float(total_mi)


def generate_nonlinear_from_latent(
    n_samples: int, latent_dim: int, observed_dim: int, mi: float,
    hidden_dim: int = 64, use_torch: bool = True, return_latents: bool = False,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two nonlinearly related datasets from a shared latent variable.

    A low-dimensional latent variable `z` is first generated. Two observed
    variables, `x` and `y`, are then created as nonlinear projections of `z`
    with added noise.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    latent_dim : int
        The dimensionality of the shared latent variable `z`.
    observed_dim : int
        The dimensionality of the observed variables `x` and `y`.
    mi : float
        The ground truth MI between the latent variables Z_x and Z_y in bits.
    hidden_dim : int, optional
        The hidden dimension of the transforming MLPs. Defaults to 64.
    use_torch : bool, optional
        If True, returns torch.Tensors. Defaults to True.
    return_latents : bool, optional
        If True, also return the shared latents `(z_x, z_y)` used to
        construct `x` and `y`. Defaults to False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:

        - **x** (*np.ndarray* or *torch.Tensor*): The first dataset, of shape `(n_samples, observed_dim)`.
        - **y** (*np.ndarray* or *torch.Tensor*): The second dataset, of shape `(n_samples, observed_dim)`.
        - **z_x**, **z_y** (*np.ndarray* or *torch.Tensor*, optional): The shared latents,
          of shape `(n_samples, latent_dim)`, only if `return_latents=True`.
    """
    z_x, z_y = generate_correlated_gaussians(n_samples, latent_dim, mi,
                                             use_torch=True, seed=seed)

    # The two projections are part of the construction, so they are seeded too:
    # without this the "same" call returns different observables every time.
    with _torch_seed(seed):
        mlp_x = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim), torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, observed_dim)
        )
        mlp_y = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim), torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, observed_dim)
        )

    with torch.no_grad():
        x = mlp_x(z_x)
        y = mlp_y(z_y)

    if not use_torch:
        x, y = x.numpy(), y.numpy()
        z_x, z_y = z_x.numpy(), z_y.numpy()
    if return_latents:
        return x, y, z_x, z_y
    return x, y


# ---------------------------------------------------------------------------
# Discrete-latent generators
#
# A shared discrete latent is drawn from a known joint pmf, so the pair's mutual
# information is that pmf's, computed exactly. The observable is then built so
# that the latent is recoverable from it and nothing else is shared, which makes
# the pmf value the mutual information of the observables too.
# ---------------------------------------------------------------------------


def symmetric_joint_pmf(n_levels: int, rho: float) -> np.ndarray:
    """Joint pmf over ``n_levels`` x ``n_levels`` with ``rho`` on the diagonal.

    Both marginals are uniform, so ``rho`` alone sets how much the two agree:
    ``rho=1/n_levels`` makes them independent (zero MI) and ``rho -> 1`` makes
    them identical (``log2(n_levels)`` bits).
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be at least 2, got {n_levels}.")
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must lie in (0, 1), got {rho}.")
    p = np.full((n_levels, n_levels), (1.0 - rho) / (n_levels * n_levels - n_levels))
    np.fill_diagonal(p, rho / n_levels)
    return p / p.sum()


def pmf_mi_bits(p: np.ndarray) -> float:
    """Exact mutual information of a joint pmf, in bits."""
    p = np.asarray(p, dtype=float)
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / np.outer(px, py)[nz])).sum())


def generate_spike_pair(n_windows: int = 4000, window_size: float = 1.0,
                        n_neurons: int = 4, coding: str = 'count',
                        n_levels: int = 4, rho: float = 0.85,
                        lag_windows: int = 0,
                        counts: Optional[Sequence[int]] = None,
                        count_range: Tuple[int, int] = (2, 9),
                        burst_sd: float = 0.035,
                        seed: int = 0):
    """Two spike populations sharing a discrete latent, with exact MI.

    Each window carries a latent level drawn from
    :func:`symmetric_joint_pmf`, and the window's spikes are generated so that
    the level is the only thing the two populations share. The mutual
    information between a window of X and the matching window of Y is therefore
    the pmf's, exactly.

    Two codings, differing in what carries the information:

    - ``'count'`` — the level sets how many spikes the window holds, and the
      times within the window are uniform noise. Information is in the rate.
    - ``'timing'`` — the level sets *where* in the window a spike burst falls,
      while the number of spikes is drawn independently for each population and
      so carries nothing. Information is in the timing.

    The two make different demands of an estimator, and of the spike
    representation: whether a padded slot is distinguishable from a real spike
    matters for ``'timing'`` in a way it does not for ``'count'``.

    .. warning::
       The returned value is the MI **between windows that align with this
       generator's own windows**. The window origin is handled here (a single
       spike at ``t=0`` pins the grid, since spike windowing takes its origin
       from the earliest spike), but the caller still has to match the window
       size and not re-tile::

           x, y, exact = generate_spike_pair(coding='timing', window_size=1.0)
           nmi.run(x, y, mode='estimate',
                   processing=nmi.Processing(x='spike', y='spike',
                                             x_params={'window_size': 1.0},
                                             y_params={'window_size': 1.0}),
                   training=nmi.Training(shift_time=False, shift_windows=False))

       With ``shift_time`` or ``shift_windows`` left on (both default ``True``),
       or with a different ``window_size``, an analysis window spans two
       independent latent draws and can carry *more* than the returned value.
       An estimate above it therefore means the setup is wrong rather than the
       estimator being wrong. A quick check: the number of windows built should
       equal ``n_windows``.

    Parameters
    ----------
    n_windows : int, optional
        Number of windows to generate. Defaults to 4000.
    window_size : float, optional
        Window duration, in the same units as the returned spike times.
        Defaults to 1.0.
    n_neurons : int, optional
        Neurons per population. Every neuron in a population sees the same
        latent, so more neurons make it easier to read, not more informative.
        Defaults to 4.
    coding : str, optional
        ``'count'`` or ``'timing'``. Defaults to ``'count'``.
    n_levels : int, optional
        Size of the latent alphabet. Caps the MI at ``log2(n_levels)``.
        Defaults to 4.
    rho : float, optional
        Probability the two populations share the same level. Defaults to 0.85.
    lag_windows : int, optional
        Whole-window delay of Y relative to X. The exact MI is unchanged and
        now sits at this lag rather than at zero, which is what ``mode='lag'``
        should recover. Defaults to 0.
    counts : sequence of int, optional
        ``coding='count'`` only: the spike count for each latent level.
        Defaults to ``[2, 5, 8, 11]`` truncated or extended to ``n_levels``.
    count_range : tuple of int, optional
        ``coding='timing'`` only: ``(low, high)`` for the nuisance spike count,
        drawn independently per population so it carries no information.
        Defaults to ``(2, 9)``.
    burst_sd : float, optional
        ``coding='timing'`` only: burst width as a fraction of the window.
        Defaults to 0.035.
    seed : int, optional
        Defaults to 0.

    Returns
    -------
    x_spikes : list of np.ndarray
        One sorted array of spike times per neuron.
    y_spikes : list of np.ndarray
        The same for the second population.
    exact_mi : float
        Exact mutual information in bits, per aligned window pair.

    Examples
    --------
    >>> x, y, mi = generate_spike_pair(n_windows=500, coding='count')
    >>> len(x), round(mi, 3)
    (4, 1.152)
    """
    if coding not in ('count', 'timing'):
        raise ValueError(f"coding must be 'count' or 'timing', got {coding!r}.")
    if lag_windows < 0:
        raise ValueError(f"lag_windows must be non-negative, got {lag_windows}.")
    if lag_windows >= n_windows:
        raise ValueError(
            f"lag_windows ({lag_windows}) must be smaller than n_windows "
            f"({n_windows}), or no overlapping windows remain.")

    rng = np.random.default_rng(seed)
    pmf = symmetric_joint_pmf(n_levels, rho)
    exact_mi = pmf_mi_bits(pmf)

    flat = rng.choice(pmf.size, size=n_windows, p=pmf.ravel())
    level_x, level_y = np.unravel_index(flat, pmf.shape)

    if coding == 'count':
        if counts is None:
            base = [2, 5, 8, 11]
            counts = [base[i] if i < len(base) else base[-1] + 3 * (i - len(base) + 1)
                      for i in range(n_levels)]
        counts = np.asarray(counts, dtype=int)
        if len(counts) != n_levels:
            raise ValueError(
                f"counts has {len(counts)} entries but n_levels is {n_levels}.")
    else:
        centres = (np.arange(n_levels) + 0.5) / n_levels

    def build(levels, window_offset):
        population = []
        for _ in range(n_neurons):
            times = []
            for w, lvl in enumerate(levels):
                if coding == 'count':
                    within = np.sort(rng.random(counts[lvl])) * 0.98
                else:
                    n = rng.integers(*count_range)   # nuisance, independent
                    within = np.sort(np.clip(
                        rng.normal(centres[lvl], burst_sd, n), 0.01, 0.99))
                times.append((w + window_offset) * window_size + within * window_size)
            population.append(np.concatenate(times) if times else np.array([]))
        return population

    x_pop = build(level_x, 0)
    y_pop = build(level_y, lag_windows)

    # Pin the window grid to zero. Spike windowing takes its origin from the
    # earliest spike in the data (SpikeWindowDataset.get_temporal_extent), which
    # otherwise lands mid-window and makes every analysis window straddle two of
    # the windows built here, spanning two independent latent draws. One spike
    # at t=0 in the first train of each population fixes the origin; against
    # n_windows * n_neurons windows of content its own contribution is
    # negligible, and without it the returned MI is not the quantity being
    # estimated.
    x_pop[0] = np.concatenate([[0.0], x_pop[0]])
    y_pop[0] = np.concatenate([[0.0], y_pop[0]])

    # Y is emitted `lag_windows` later, so X's window w pairs with Y's w + lag.
    return x_pop, y_pop, exact_mi


def generate_xor_pair(n_samples: int, noise: float = 0.1, use_torch: bool = True,
                      seed: int = 0):
    """The XOR synergy problem, with exact mutual information.

    ``X = (x1, x2)`` are independent fair bits and ``Y = (x1 XOR x2) + N(0, noise)``.
    Neither bit alone says anything about ``Y``, so ``I(x1; Y) = I(x2; Y) = 0``
    exactly, while the pair determines it. That gap is the point: the
    information is purely synergistic and cannot be found one variable at a
    time.

    The returned MI is ``H(Y) - H(Y | X)``. The conditional is a plain Gaussian,
    and the marginal is an equal mixture of ``N(0, noise)`` and ``N(1, noise)``
    whose entropy is evaluated by quadrature, so the value is exact to
    quadrature rather than in closed form. As ``noise -> 0`` it approaches
    exactly 1 bit.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    noise : float, optional
        Standard deviation of the Gaussian added to Y. Defaults to 0.1. Must be
        positive: at exactly zero Y is discrete and the differential entropy
        used here does not apply, though the limit is 1 bit.
    use_torch : bool, optional
        Return ``torch.Tensor`` rather than ``np.ndarray``. Defaults to True.
    seed : int, optional
        Defaults to None (uses global numpy state, matching the other generators).

    Returns
    -------
    x : array of shape (n_samples, 2)
    y : array of shape (n_samples, 1)
    exact_mi : float
        Mutual information ``I(X; Y)`` in bits.
    """
    if noise <= 0:
        raise ValueError(
            f"noise must be positive, got {noise}. At exactly zero Y is discrete "
            f"and this differential-entropy calculation does not apply; the "
            f"limiting value is 1 bit.")
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, size=n_samples) if seed is not None else np.random.randint(0, 2, size=n_samples)
    x2 = rng.integers(0, 2, size=n_samples) if seed is not None else np.random.randint(0, 2, size=n_samples)
    bit = np.bitwise_xor(x1, x2).astype(float)
    y = bit + (rng.normal(size=n_samples) if seed is not None
               else np.random.randn(n_samples)) * noise

    # H(Y): equal mixture of N(0, noise) and N(1, noise), by quadrature.
    grid = np.linspace(-6 * noise, 1 + 6 * noise, 20001)
    comp = np.exp(-0.5 * ((grid[:, None] - np.array([0.0, 1.0])) / noise) ** 2)
    comp /= noise * np.sqrt(2 * np.pi)
    dens = comp.mean(axis=1)
    nz = dens > 0
    h_y = -float(_TRAPZ(dens[nz] * np.log2(dens[nz]), grid[nz]))
    h_y_given_x = 0.5 * np.log2(2 * np.pi * np.e * noise ** 2)
    exact_mi = float(h_y - h_y_given_x)

    x = np.vstack([x1, x2]).T
    y = y.reshape(-1, 1)
    if use_torch:
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), exact_mi
    return x, y, exact_mi


def generate_categorical_pair(n_samples: int, n_channels: int = 1,
                              n_categories: int = 3, agreement: float = 0.9,
                              stay_probability: float = 0.95,
                              use_torch: bool = True, seed: int = 0):
    """Two correlated categorical series, with exact per-channel MI.

    ``x`` is a Markov chain that holds its state with probability
    ``stay_probability`` and otherwise redraws uniformly, giving a uniform
    stationary distribution. ``y_t`` copies ``x_t`` with probability
    ``agreement`` and is otherwise uniform, so the channel is

        ``P(y=j | x=i) = agreement * [i == j] + (1 - agreement) / n_categories``

    and the joint pmf, hence the mutual information, is known in closed form.

    Parameters
    ----------
    n_samples : int
        Series length.
    n_channels : int, optional
        Independent channels, each carrying the same per-channel MI. Defaults to 1.
    n_categories : int, optional
        Alphabet size. Defaults to 3.
    agreement : float, optional
        Probability that ``y`` copies ``x``. Defaults to 0.9.
    stay_probability : float, optional
        Probability the chain holds its state, which sets temporal smoothness
        without affecting the per-sample MI. Defaults to 0.95.
    use_torch : bool, optional
        Defaults to True.
    seed : int, optional
        Defaults to None.

    Returns
    -------
    x, y : arrays of shape (n_samples, n_channels), dtype int
    exact_mi : float
        Per-channel ``I(x_t; y_t)`` in bits. Channels are independent, so the
        joint MI over all channels is ``n_channels`` times this.
    """
    if not 0.0 < agreement <= 1.0:
        raise ValueError(f"agreement must lie in (0, 1], got {agreement}.")
    rng = np.random.default_rng(seed)
    K = n_categories

    joint = np.full((K, K), (1.0 - agreement) / (K * K))
    np.fill_diagonal(joint, joint[0, 0] + agreement / K)
    exact_mi = pmf_mi_bits(joint)

    x = np.zeros((n_samples, n_channels), dtype=int)
    y = np.zeros((n_samples, n_channels), dtype=int)
    for ch in range(n_channels):
        x[0, ch] = rng.integers(K)
        for t in range(1, n_samples):
            x[t, ch] = x[t - 1, ch] if rng.random() < stay_probability else rng.integers(K)
        copy = rng.random(n_samples) < agreement
        x[:, ch] = x[:, ch]
        y[:, ch] = np.where(copy, x[:, ch], rng.integers(K, size=n_samples))

    if use_torch:
        return torch.from_numpy(x), torch.from_numpy(y), exact_mi
    return x, y, exact_mi


def generate_lagged_pair(n_samples: int = 5000, lag: int = 30, dim: int = 1,
                         phi: float = 0.95, noise: float = 0.5,
                         coupling: float = 3.0, seed: int = 0):
    """A pair whose dependence sits at a known lag, with exact MI at that lag.

    Built from :class:`SharedLatentGaussian` and then offset, so both the lag
    and the mutual information at it are known. ``mode='lag'`` should recover
    ``lag`` as the peak, and the MI it reports there should approach
    ``exact_mi``.

    Parameters
    ----------
    n_samples : int, optional
        Series length. Defaults to 5000.
    lag : int, optional
        Samples by which Y trails X, so the peak sits at ``+lag``. Defaults to 30.
    dim : int, optional
        Channels per signal. Defaults to 1.
    phi : float, optional
        Latent autocorrelation, which sets how broad the lag peak is. Defaults
        to 0.95.
    noise : float, optional
        Observation-noise standard deviation. Defaults to 0.5.
    coupling : float, optional
        Scales the projections, raising the MI at the peak without moving it.
        Defaults to 3.0. Together with ``noise`` this sets ``exact_mi``, which
        is returned rather than assumed.
    seed : int, optional
        Defaults to 0.

    Returns
    -------
    x, y : np.ndarray of shape (n_samples, dim), float32
    exact_mi : float
        ``I`` in bits at the peak, i.e. between ``x[t]`` and ``y[t + lag]``.
        Y trails X by ``lag`` samples, so the dependence sits at ``x[t - lag]``
        for a given ``y[t]``, and ``mode='lag'`` reports the peak at ``+lag``.

    Examples
    --------
    >>> x, y, mi = generate_lagged_pair(n_samples=4000, lag=20)
    >>> x.shape, round(mi, 3)
    ((4000, 1), 0.969)
    """
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}.")
    oracle = SharedLatentGaussian(dims={'x': dim, 'y': dim}, d=1, phi=phi,
                                  noise=noise, coupling=coupling, seed=seed)
    sample = oracle.sample(n_samples + lag, seed=seed)
    # Taking X from later in the series makes Y trail it, so the cross-lag peak
    # sits at +lag. The value there is the aligned (offset-0) MI of the source
    # processes, since the offset is exactly what the slicing undoes.
    x = sample['x'][lag:lag + n_samples]
    y = sample['y'][:n_samples]
    exact_mi = oracle.exact(A=[('x', 0)], B=[('y', 0)])
    return x.astype('float32'), y.astype('float32'), exact_mi
