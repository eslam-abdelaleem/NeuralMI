# neural_mi/generators/__init__.py
"""Synthetic data for testing, validating and demonstrating the estimators.

The package splits on the one distinction that matters when choosing a
generator: **whether the mutual information is known**.

``oracle`` holds the generators that come with an answer. Use one of these
whenever an estimate needs checking, since an estimator cannot be validated
against data whose true value nobody knows.

- :class:`SharedLatentGaussian` is the most general: jointly Gaussian processes
  driven by a shared autoregressive latent, exposing the exact ``I(A; B | C)``
  for any choice of processes and time offsets. Every quantity in the temporal
  taxonomy has an exact number to check against, and ``block_mi(w)`` gives the
  exact windowed MI at window size ``w``.
- :func:`generate_correlated_gaussians` fixes the MI of an IID pair directly.
- :func:`generate_windowed_oscillatory` and
  :func:`generate_windowed_multichannel` return windowed data alongside the
  *observed* MI, computed from the SNR rather than inherited from the latent.

- :func:`generate_nonlinear_from_latent` fixes the MI of a shared latent pair
  and observes it through a smooth nonlinear projection. The projection is
  smooth enough to be near information-preserving in practice, so the ``mi``
  argument is usable as the target value. Strictly it is an upper bound, since
  a noisy projection can only discard information, so treat a shortfall against
  it as expected rather than as estimator failure.

``synthetic`` holds the rest: processes built to exhibit a structure (a lag, a
history dependence, synergy) whose mutual information is not computed. They are
useful for demonstrating that a method responds to a structure, and unsuitable
for asking whether an estimate is correct.
"""
from .oracle import (
    # Exact ground truth for any I(A;B|C) over time offsets, and for windowed MI
    SharedLatentGaussian,
    generate_shared_latent_gaussian,
    # Closed-form MI by construction
    mi_to_rho,
    generate_correlated_gaussians,
    generate_windowed_oscillatory,
    generate_windowed_multichannel,
    generate_nonlinear_from_latent,
)
from .synthetic import (
    generate_temporally_convolved_data,
    generate_xor_data,
    generate_correlated_spike_trains,
    generate_correlated_categorical_series,
    generate_event_related_data,
    generate_linear_data,
    generate_nonlinear_data,
    generate_history_data,
    generate_full_data,
)

__all__ = [
    # --- Known MI: use these to check an estimate ---
    'SharedLatentGaussian',
    'generate_shared_latent_gaussian',
    'mi_to_rho',
    'generate_correlated_gaussians',
    'generate_windowed_oscillatory',
    'generate_windowed_multichannel',
    'generate_nonlinear_from_latent',
    # --- Structure without a known MI: use these to demonstrate a response ---
    'generate_temporally_convolved_data',
    'generate_xor_data',
    'generate_correlated_spike_trains',
    'generate_correlated_categorical_series',
    'generate_event_related_data',
    'generate_linear_data',
    'generate_nonlinear_data',
    'generate_history_data',
    'generate_full_data',
]
