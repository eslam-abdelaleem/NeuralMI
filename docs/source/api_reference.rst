API Reference
=============

This page provides a detailed look at the functions, classes, and exceptions within the ``neural_mi`` library.

Core Function
-------------

This is the main entry point for all analyses in the library.

.. autofunction:: neural_mi.run

.. _results-object:

The Results Object
------------------

All calls to ``run()`` return a special ``Results`` object. It holds all the information from the analysis and has a built-in plotting method.

.. autoclass:: neural_mi.results.Results
   :members: plot
   :undoc-members:
   :show-inheritance:

Configuration
-------------

Use this function to control the library's logging output.

.. autofunction:: neural_mi.set_verbosity

Exceptions
----------

These are the custom exceptions raised by the library.

.. autoclass:: neural_mi.exceptions.NeuralMIError
.. autoclass:: neural_mi.exceptions.DataShapeError
.. autoclass:: neural_mi.exceptions.InsufficientDataError
.. autoclass:: neural_mi.exceptions.TrainingError