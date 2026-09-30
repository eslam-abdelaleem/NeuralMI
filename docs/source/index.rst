NeuralMI
========

**NeuralMI brings information-theoretic analysis to the scale of modern
neuroscience using neural-network-based information estimators.**

Mutual information measures how much two recordings share, in bits, including
the dependencies a correlation reports as zero. It also answers questions a correlation has no form for: how much a population's past says about its
own future, or what a pair of signals carries that neither carries alone.

Classical estimators need samples in proportion to the dimensions they are handed. Past roughly ten they outrun any recording you are likely to have. A neural estimator first learns a map into a low-dimensional embedding. The samples you need then follow the structure in the data and not the channel count it
arrived on.

.. image:: _static/sample_sweep.png
   :alt: Estimate against sample count at 1000 channels per side
   :align: center
   :width: 78%

Both sides above are a thousand channels wide and share ten latent dimensions
carrying exactly four bits. NeuralMI finds the four bits by a thousand samples
where KSG on the same data is still under two bits at five thousand.

Every analysis goes through the one function ``nmi.run()`` with a ``mode`` that
selects the quantity.

Where to start
--------------

:doc:`getting_started` installs the library and walks one estimate end to end.
The :doc:`tutorials` are five notebooks that read in order. :doc:`api_reference`
documents the public functions and classes. The source is on `GitHub
<https://github.com/eslam-abdelaleem/NeuralMI>`_.

:doc:`USING` shows the call for each task and how to read what comes back. :doc:`PARAMETERS` lists every setting with its default. :doc:`THEORY` explains
what the reported numbers mean and why the estimators behave as they do.
:doc:`MESSAGES` is keyed by the text of every warning the library prints.
:doc:`ANATOMY` builds an estimator from scratch in a few dozen lines of PyTorch. :doc:`INTERNALS` maps the codebase and says how to extend it. :doc:`TESTING` covers what the suite protects.

.. toctree::
   :maxdepth: 2
   :caption: Using the library

   getting_started
   tutorials
   api_reference

.. toctree::
   :maxdepth: 1
   :caption: Reference

   USING
   PARAMETERS
   THEORY
   MESSAGES
   ANATOMY
   INTERNALS
   TESTING

.. toctree::
   :maxdepth: 1
   :caption: Project

   CONTRIBUTING
   LICENSE
