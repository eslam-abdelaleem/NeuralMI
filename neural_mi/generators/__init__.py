# neural_mi/generators/__init__.py
"""Synthetic data with a known answer, for testing and validating estimators.

Every generator here reports the quantity an estimate should be checked
against. An estimator cannot be validated on data whose truth nobody knows, so
there is nothing in this package that fails to provide one.

**Exact mutual information**

- :class:`SharedLatentGaussian` is the most general: jointly Gaussian processes
  driven by a shared autoregressive latent, exposing the exact ``I(A; B | C)``
  for any choice of processes and time offsets. Every quantity in the temporal
  taxonomy has an exact number to check against, and ``block_mi(w)`` gives the
  exact windowed MI at window size ``w``.
- :func:`generate_correlated_gaussians` fixes the MI of an IID pair directly.
- :func:`generate_windowed_oscillatory` and
  :func:`generate_windowed_multichannel` return windowed data alongside the
  *observed* MI, computed from the SNR rather than inherited from the latent.
- :func:`generate_spike_pair` returns two spike populations sharing a discrete
  latent, with the MI exact from that latent's pmf. Available in a rate coding
  and a timing coding, which make different demands of the spike
  representation.
- :func:`generate_categorical_pair` and :func:`generate_xor_pair` cover
  discrete and synergistic cases. XOR's individual terms ``I(x1; Y)`` and
  ``I(x2; Y)`` are exactly zero while the pair determines ``Y``, which is the
  whole point of it.
- :func:`generate_nonlinear_from_latent` fixes the MI of a shared latent pair
  and observes it through a smooth nonlinear projection. The projection is
  smooth enough to be near information-preserving in practice, so the ``mi``
  argument is usable as the target. Strictly it is an upper bound, since a
  noisy projection can only discard information.

**Exact lag**

- :func:`generate_lagged_pair` places the dependence at a known lag *and*
  reports the MI at that peak, so ``mode='lag'`` can be checked on both.

A note on windowed generators: :func:`generate_spike_pair`'s value is the MI
between windows that align with its own. Analysis must use the same
``window_size`` and must not re-tile (``shift_time=False, shift_windows=False``),
or a window spans two independent latent draws and can carry more.
"""
from .oracle import (
    # Exact ground truth for any I(A;B|C) over time offsets, and windowed MI
    SharedLatentGaussian,
    generate_shared_latent_gaussian,
    # Closed-form MI by construction
    mi_to_rho,
    generate_correlated_gaussians,
    generate_windowed_oscillatory,
    generate_windowed_multichannel,
    generate_nonlinear_from_latent,
    # Discrete-latent constructions
    symmetric_joint_pmf,
    pmf_mi_bits,
    generate_spike_pair,
    generate_categorical_pair,
    generate_xor_pair,
    # Known lag, with the MI at that lag
    generate_lagged_pair,
)

__all__ = [
    'SharedLatentGaussian',
    'generate_shared_latent_gaussian',
    'mi_to_rho',
    'generate_correlated_gaussians',
    'generate_windowed_oscillatory',
    'generate_windowed_multichannel',
    'generate_nonlinear_from_latent',
    'symmetric_joint_pmf',
    'pmf_mi_bits',
    'generate_spike_pair',
    'generate_categorical_pair',
    'generate_xor_pair',
    'generate_lagged_pair',
]
