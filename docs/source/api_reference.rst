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

The Results Object
------------------

All calls to ``run()`` return a ``Results`` object. This object acts as a container for all the outputs of an analysis, providing convenient access to the final MI estimate, the raw data, and a built-in plotting method.

.. autoclass:: neural_mi.results.Results
   :members: plot
   :undoc-members:
   :show-inheritance:

Data Generation (`generators`)
------------------------------

This module provides functions to generate synthetic datasets with known properties. These are useful for testing estimators, validating models, and following the tutorials.

.. automodule:: neural_mi.generators
   :members: generate_correlated_gaussians, generate_nonlinear_from_latent, generate_linear_data, generate_nonlinear_data, generate_temporally_convolved_data, generate_xor_data, generate_event_related_data, generate_linear_data, generate_nonlinear_data, generate_history_data, generate_full_data
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