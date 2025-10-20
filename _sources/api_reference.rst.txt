API Reference
=============

Welcome to the technical reference for `NeuralMI`. This page provides detailed documentation on the public functions, classes, and modules that make up the library's API.

For a high-level overview and practical examples, please see the :doc:`tutorials`. For implementation details, feel free to explore the source code on our `GitHub repository <https://github.com/eslam-abdelaleem/NeuralMI>`_.

The Core `run()` Function
-------------------------

The ``run()`` function is the main entry point for all analyses in the library. It is a unified interface that orchestrates data processing, model training, and results aggregation based on the specified ``mode``.

.. autofunction:: neural_mi.run

The Results Object
------------------

All calls to ``run()`` return a ``Results`` object. This object acts as a container for all the outputs of an analysis, providing convenient access to the final MI estimate, the raw data, and a built-in plotting method.

.. autoclass:: neural_mi.results.Results
   :members: plot
   :undoc-members:
   :show-inheritance:

Data Generation (`datasets`)
----------------------------

This module provides functions to generate synthetic datasets with known properties. These are useful for testing estimators, validating models, and following the tutorials.

.. automodule:: neural_mi.datasets
   :members: generate_correlated_gaussians, generate_nonlinear_from_latent, generate_linear_data, generate_nonlinear_data, generate_temporally_convolved_data, generate_xor_data, generate_event_related_data, generate_linear_data, generate_nonlinear_data, generate_history_data, generate_full_data
   :undoc-members:

Visualization (`visualize`)
---------------------------

This module contains helper functions for creating publication-quality plots of analysis results. These functions are typically called automatically by the ``Results.plot()`` method but can also be used directly.

.. automodule:: neural_mi.visualize
   :members: plot_sweep_curve, plot_bias_correction_fit, plot_cross_correlation, analyze_mi_heatmap
   :undoc-members:

Configuration
-------------

Use this function to control the library's logging output level.

.. autofunction:: neural_mi.logger.set_verbosity

Exceptions
----------

These are the custom exceptions raised by the library to signal specific errors.

.. automodule:: neural_mi.exceptions
   :members:
   :undoc-members: