API Reference
=============

The Core `run()` Function
-------------------------

The ``run()`` function is the main entry point for all analyses in the library. It is a unified interface that orchestrates data processing, model training, and results aggregation based on the specified ``mode``.

.. autofunction:: neural_mi.run

Configuration Objects
---------------------

Every call to ``run()`` is configured with grouped, typed dataclasses. The
**shared** configs apply to every mode; the **per-mode** configs carry options
specific to one analysis mode. All are importable directly from ``neural_mi``
(e.g. ``from neural_mi import Model, Training``). The signature of each class
lists its fields; see the *Config Fields Reference* in ``NEURALMI_REFERENCE.md``
for a description of every field.

Shared configs:

.. autoclass:: neural_mi.Model
.. autoclass:: neural_mi.Training
.. autoclass:: neural_mi.Split
.. autoclass:: neural_mi.Processing
.. autoclass:: neural_mi.Estimator
.. autoclass:: neural_mi.Output

Per-mode configs:

.. autoclass:: neural_mi.Rigorous
.. autoclass:: neural_mi.Precision
.. autoclass:: neural_mi.Lag
.. autoclass:: neural_mi.Transfer
.. autoclass:: neural_mi.Dimensionality
.. autoclass:: neural_mi.Conditional
.. autoclass:: neural_mi.Interaction

Named Quantities (`quantities`)
-------------------------------

Thin wrappers over ``run()`` for the standard information-theoretic quantities.
Each one builds the arrays its offset pattern needs and dispatches to an
existing mode, so all of them take the same ``model=``/``training=``/``split=``/
``estimator=``/``output=``/``seed=`` arguments as ``run()`` and return the same
``Results`` object. None adds estimation logic of its own. Every one is
importable directly from ``neural_mi`` (e.g. ``from neural_mi import
transfer_entropy``).

Where a quantity takes a length parameter (``k``, ``history_window``,
``window_size``), passing an iterable instead of an int sweeps it in parallel
and returns a ``Results`` shaped like ``mode='sweep'``.

The conditional quantities are chain-rule differences of larger estimates and
report an ``amplification_factor`` in ``details``; see the *Reading a
conditional quantity* section of ``NEURALMI_REFERENCE.md`` before quoting one.

.. automodule:: neural_mi.quantities
   :members: active_information_storage, excess_entropy, instantaneous_mi,
             cross_predictive_information, block_mi, transfer_entropy,
             conditional_transfer_entropy, interaction_information,
             mi_rate, instantaneous_exchange, directed_information_rate
   :undoc-members:

The Results Object
------------------

All calls to ``run()`` return a ``Results`` object. This object acts as a container for all the outputs of an analysis, providing convenient access to the final MI estimate, the raw data, and a built-in plotting method.

.. autoclass:: neural_mi.results.Results
   :members: plot
   :undoc-members:
   :show-inheritance:

Data Generation (`generators`)
------------------------------

Synthetic data for testing estimators and validating models. Every generator
reports the quantity an estimate should be checked against, whether that is a
mutual information or a lag.

.. automodule:: neural_mi.generators
   :members: SharedLatentGaussian, generate_shared_latent_gaussian,
             generate_correlated_gaussians, generate_nonlinear_from_latent,
             generate_windowed_oscillatory, generate_windowed_multichannel,
             generate_spike_pair, generate_categorical_pair, generate_xor_pair,
             generate_lagged_pair, symmetric_joint_pmf, pmf_mi_bits, mi_to_rho
   :undoc-members:

Visualization (`visualize`)
---------------------------

This module contains helper functions for creating publication-quality plots of analysis results. These functions are typically called automatically by the ``Results.plot()`` method but can also be used directly.

.. automodule:: neural_mi.visualize
   :members: plot_sweep_curve, plot_bias_correction_fit, plot_cross_correlation, analyze_mi_heatmap
   :undoc-members:

Logging
-------

Use this function to control the library's logging output level.

.. autofunction:: neural_mi.logger.set_verbosity

Exceptions
----------

These are the custom exceptions raised by the library to signal specific errors.

.. automodule:: neural_mi.exceptions
   :members:
   :undoc-members: