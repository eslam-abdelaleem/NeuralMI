API reference
=============

The ``run()`` function
----------------------

``run()`` is the entry point for every analysis. It prepares the data, trains
the networks and assembles the ``Results`` for the chosen ``mode``.

.. autofunction:: neural_mi.run

Configuration objects
---------------------

Every call to ``run()`` is configured with grouped, typed dataclasses. The
shared configs apply to every mode, and each per-mode config carries the options
of one analysis mode. All are importable from ``neural_mi``, as in
``from neural_mi import Model, Training``, and a plain ``dict`` with the same keys
works wherever a config is accepted. :doc:`PARAMETERS` describes every field
with its default.

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
.. autoclass:: neural_mi.Pairwise
.. autoclass:: neural_mi.Sweep

Named quantities (``quantities``)
---------------------------------

Functions for the standard information-theoretic quantities. Each builds the
arrays its offset pattern needs and calls ``run()``, so every one takes the
keyword arguments of ``run()`` and returns the same ``Results``. None adds
estimation logic of its own. All are importable from ``neural_mi``, as in
``from neural_mi import transfer_entropy``.

A quantity's own parameter (``k``, ``history_window``, ``window_size``, ``h``,
``half_width``) takes a scalar or an iterable. An iterable runs every value, in
parallel across ``n_workers``, and returns one ``Results`` with a configuration
per value. ``rigorous=True`` extrapolates each repeat to infinite data.

The conditional quantities are chain-rule differences of larger estimates and
report an ``amplification_factor`` in ``result.dataframe``. :doc:`THEORY`
explains how to read it before quoting a small value.

.. automodule:: neural_mi.quantities
   :members:
   :undoc-members:

The ``Results`` object
----------------------

Every call to ``run()`` and every named quantity returns a ``Results``.
``runs`` holds one row per repeat, ``dataframe`` one row per configuration and
axis value, ``mi_estimate`` the headline when there is exactly one such row,
``details`` the structured diagnostics of each configuration, and ``params`` the
full configuration of the call. :doc:`USING` describes each field.

.. autoclass:: neural_mi.results.Results
   :members:
   :show-inheritance:

Saved models are reloaded with ``extract_embeddings``.

.. autofunction:: neural_mi.extract_embeddings

Data generation (``generators``)
--------------------------------

Synthetic data for testing estimators and validating models. Every generator
reports the quantity an estimate should be checked against, whether that is a
mutual information or a lag.

.. automodule:: neural_mi.generators
   :members:
   :undoc-members:

Visualisation (``visualize``)
-----------------------------

The functions behind ``Results.plot()`` and ``Results.animate()``, each also
callable on its own.

.. automodule:: neural_mi.visualize
   :members:
   :undoc-members:

Logging
-------

These functions set the library's logging level.

.. autofunction:: neural_mi.logger.set_verbose
.. autofunction:: neural_mi.logger.set_verbosity

Exceptions
----------

The library's own exceptions, all derived from ``NeuralMIError``.

.. automodule:: neural_mi.exceptions
   :members:
   :undoc-members: