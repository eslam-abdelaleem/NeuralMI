Tutorials
=========

This series of tutorials provides a guided tour through the features of `NeuralMI`,
from basic usage to advanced, scientifically rigorous analyses. We recommend
following them in order to build a comprehensive understanding of the library.

Part 0: Understanding MI Estimation
-----------------------------------

A conceptual on-ramp: what mutual information captures, how a neural estimator computes it,
and which number the library reports. Optional, but it makes the later choices intuitive.

- **00_Why_and_How_MI_Estimation_Works**: A conceptual on-ramp — why mutual
  information (not correlation), how a neural estimator turns dependence into
  a number, and which value the library reports.

.. toctree::
   :maxdepth: 1

   tutorials/00_Why_and_How_MI_Estimation_Works

Part 1: The Fundamentals
------------------------

These tutorials cover the essential mechanics of the library.

- **01_A_First_Estimate**: Learn the basics of ``nmi.run()`` and the
  ``Results`` object on a simple dataset.
- **02_Neural_Data_Formats**: Understand how to use the ``Continuous``,
  ``Spike``, and ``Categorical`` data processors.
- **03_Temporal_Correlations_and_Splits**: Learn how to handle temporal data
  and avoid leakage with blocked splitting.

.. toctree::
   :maxdepth: 1

   tutorials/01_A_First_Estimate
   tutorials/02_Neural_Data_Formats
   tutorials/03_Temporal_Correlations_and_Splits

Part 2: Core Concepts for Scientific Rigor
------------------------------------------

Learn how to go beyond a single estimate to perform robust analyses.

- **04_Sweeps**: Use ``mode='sweep'`` to explore and optimize hyperparameters.
- **05_Rigorous_Estimation**: A deep dive into ``mode='rigorous'`` for
  debiased, accurate MI estimates with a confidence interval.

.. toctree::
   :maxdepth: 1

   tutorials/04_Sweeps
   tutorials/05_Rigorous_Estimation

Part 3: Advanced Analysis and Applications
------------------------------------------

Explore the library's most powerful features and learn how to extend it.

- **06_Temporal_Questions**: Directed, time-resolved analyses —
  ``mode='lag'``, ``mode='precision'``, and transfer entropy.
- **07_Population_Questions**: Population geometry and connectivity —
  ``mode='dimensionality'``, conditional MI, and the ``mode='pairwise'`` MI
  matrix. Uses real hippocampal and Allen Brain Observatory recordings.
- **08_Models_Estimators_and_Validation**: Understand the trade-offs between
  different critic architectures, estimators, and custom models.

.. toctree::
   :maxdepth: 1

   tutorials/06_Temporal_Questions
   tutorials/07_Population_Questions
   tutorials/08_Models_Estimators_and_Validation
