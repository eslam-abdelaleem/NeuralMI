Tutorials
=========

A guided path through `NeuralMI`, from what an estimate means to defending one
on a real recording. Each part answers a different kind of question, and they
build on each other, so reading in order is worth it.

Part 0: What an estimate is
---------------------------

- **00_Why_and_How_MI_Estimation_Works**: Why mutual information rather than
  correlation, how a neural estimator turns dependence into a number, and which
  value the library reports.

.. toctree::
   :maxdepth: 1

   tutorials/00_Why_and_How_MI_Estimation_Works

Part 1: Getting your data in
----------------------------

- **01_A_First_Estimate**: One ``nmi.run()`` call on data with a known answer.
  What ``mi_estimate`` is versus ``details['test_mi']``, how the answer moves
  with sample size, and a KSG comparison showing when a neural estimator is
  worth its cost.
- **02_Neural_Data_Formats**: Spike times, binned counts and categorical
  labels, what windowing does to each, and which quantity
  ``drop_empty_windows`` selects.
- **03_Temporal_Correlations_and_Splits**: Why ``Split(mode='blocked')`` is the
  default, and what random splitting costs on autocorrelated data.

.. toctree::
   :maxdepth: 1

   tutorials/01_A_First_Estimate
   tutorials/02_Neural_Data_Formats
   tutorials/03_Temporal_Correlations_and_Splits

Part 2: Choosing the quantity that matches your question
--------------------------------------------------------

The library computes many named quantities. They are one primitive under
different offset patterns, and picking the right one matters more than tuning
the estimator.

- **04_Which_Quantity**: The taxonomy as a single ``I(A;B|C)`` call under
  different offsets. Also why windowed MI is extensive, so no window size
  reveals a plateau.
- **05_Storage_and_Rate**: How much a process predicts about its own future,
  and the per-step rate that survives as the window grows.
- **06_Direction_and_Delay**: ``mode='lag'``, ``mode='precision'``, transfer
  entropy and Massey's conservation law. Includes a measured demonstration of
  why transfer entropy is fragile: 25 to 40 times error amplification, with its
  reported direction reversing when the history window changes.

.. toctree::
   :maxdepth: 1

   tutorials/04_Which_Quantity
   tutorials/05_Storage_and_Rate
   tutorials/06_Direction_and_Delay

Part 3: Defending a number
--------------------------

- **07_Three_Variables**: Conditional MI and interaction information against an
  oracle with exact values, redundancy versus synergy, and the amplification
  factor that says how far a difference of estimates can be trusted.
- **08_Making_It_Rigorous**: Seed spread, ``mode='sweep'``, and
  ``mode='rigorous'`` with its diagnostics read honestly, including what a flat
  bias slope does and does not mean.

.. toctree::
   :maxdepth: 1

   tutorials/07_Three_Variables
   tutorials/08_Making_It_Rigorous

Part 4: Real recordings, where there is no ground truth
-------------------------------------------------------

Everything before this uses synthetic data, because you cannot check an
estimator without an answer to check against. These two use real recordings,
where you cannot ask whether an estimate is correct, only whether a claim is
defensible. Each section starts from a hypothesis and the controls carry the
argument.

- **09_What_A_Population_Encodes**: Hippocampal place cells and position,
  across two sessions from the same animal on different mazes.
- **10_Comparing_Brain_Areas**: Allen Brain Observatory recordings from VISp,
  VISpm and CA1 under natural movies and spontaneous activity. Functional
  coupling, intrinsic timescale, and which comparisons the data supports.

.. toctree::
   :maxdepth: 1

   tutorials/09_What_A_Population_Encodes
   tutorials/10_Comparing_Brain_Areas

Part 5: The machinery underneath
--------------------------------

- **11_Models_and_Machinery**: The two estimators and the InfoNCE ceiling, the
  ten embedding models, permutation nulls, and how to supply your own
  architecture.

.. toctree::
   :maxdepth: 1

   tutorials/11_Models_and_Machinery

Benchmark: NeuralMI vs. Classical Estimators
---------------------------------------------

Not a tutorial — a separate notebook answering a different question: *why*
reach for a neural estimator instead of a classical one? It compares
``NeuralMI`` against the KSG estimator and geometric intrinsic-dimension
estimators (MLE, Two-NN) on problems chosen to be hard for them. Useful if
you're deciding whether a neural estimator is the right tool for your data,
rather than learning the library itself.

.. toctree::
   :maxdepth: 1

   tutorials/vs_classical_estimators
