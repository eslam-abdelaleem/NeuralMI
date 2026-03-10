---
title: 'NeuralMI: A Python Toolbox for Rigorous Mutual Information Estimation in Neuroscience'
tags:
  - Python
  - neuroscience
  - mutual information
  - information theory
  - neural data analysis
  - PyTorch
authors:
  - name: [AUTHOR 1 — TO FILL]
    orcid: [ORCID — TO FILL]
    affiliation: 1
  - name: [AUTHOR 2 — TO FILL]
    orcid: [ORCID — TO FILL]
    affiliation: 1
affiliations:
  - name: [INSTITUTION — TO FILL]
    index: 1
date: [DATE — TO FILL]
bibliography: paper.bib
---

# Summary

`NeuralMI` is a Python library for estimating mutual information (MI) between
pairs of neural signals. MI quantifies how much information one variable carries
about another, capturing both linear and nonlinear statistical dependencies in a
single scalar quantity. Unlike Pearson correlation, MI makes no assumptions about
the form of the relationship and is therefore well-suited to the complex,
high-dimensional signals typical of modern neuroscience.

The library exposes all functionality through a single entry point, `nmi.run()`,
which accepts raw continuous recordings, spike trains, or pre-processed matrices
and returns a `Results` object with estimates, visualisations, and metadata. Nine
analysis modes cover the most common neuroscientific questions: a single MI
estimate (`estimate`), hyperparameter exploration (`sweep`), temporal lag analysis
(`lag`), spike-timing precision thresholds (`precision`), bias-corrected estimation
with confidence intervals (`rigorous`), latent dimensionality (`dimensionality`),
conditional MI (`conditional`), transfer entropy (`transfer`), and all-to-all
pairwise connectivity matrices (`pairwise`). The `rigorous` mode implements the
subsampling-and-extrapolation bias correction procedure introduced in
@abdelaleem2025accurate. The `dimensionality` mode implements the cross-covariance
spectral method and Participation Ratio introduced in @abdelaleem2025dimensionality.

# Statement of Need

Estimating MI from high-dimensional continuous data is difficult. Classic
non-parametric estimators such as $k$-nearest-neighbour methods
[@kraskov2004estimating] scale poorly beyond a few dimensions and provide no
mechanism for exploiting temporal structure. Neural network–based estimators
[@oord2018representation; @song2020understanding] overcome the dimensionality
barrier but introduce a new problem: with finite data they systematically
overestimate the true MI, and this upward bias can be substantial for the sample
sizes available in typical neuroscience experiments. Existing implementations of
these estimators — scattered across individual paper repositories — provide
point estimates only, with no bias correction, no confidence intervals, no
neuroscience-native data handling, and no unified interface across analysis types.

`NeuralMI` addresses all of these gaps in a single package designed around the
workflow of practising neuroscientists. First, it provides automated finite-sample
bias correction via a subsampling extrapolation procedure that leverages the
theoretically predicted $O(1/N)$ scaling of the bias [@abdelaleem2025accurate],
yielding a bias-corrected estimate with a confidence interval rather than a bare
point estimate. Second, it ships native processors for the three data formats most
common in systems neuroscience: continuous time series (LFP, EEG, calcium imaging,
kinematics), raw spike trains, and categorical behavioural state sequences. Each
processor applies memory-efficient non-overlapping windowing with dynamic
epoch-level jittering, so that a 10-minute recording at 1000 Hz never requires
materialising a full sliding-window array in memory. Third, `NeuralMI` provides
a principled two-mode data-splitting strategy — random for IID data and blocked
for temporal recordings — that prevents train–test leakage from autocorrelation,
a common source of inflated estimates in practice. Fourth, the `dimensionality`
mode provides a computationally efficient alternative to sweeping bottleneck size:
a single over-parameterised Hybrid Critic training run followed by SVD of the
cross-covariance of the learned embeddings, yielding a continuous Participation
Ratio as the dimensionality estimate [@abdelaleem2025dimensionality].

# Functionality

**Data handling.** Three processor classes — `ContinuousProcessor`,
`SpikeProcessor`, and `CategoricalProcessor` — convert raw recordings into the
unified `(n_samples, n_channels, window_size)` tensor that all models consume.
Multi-modal alignment (e.g., LFP paired with spike trains of different durations)
is handled automatically by truncating to the shorter stream.

**Model architecture.** The library supports six embedding networks (MLP, CNN,
GRU, LSTM, TCN, Transformer) and three critic architectures (`separable`,
`concat`, `hybrid`). The Separable Critic is the recommended default for standard
estimation; the Hybrid Critic is used automatically by `dimensionality` mode.
Two MI estimators are provided: InfoNCE [@oord2018representation], which is
low-variance with a $\log(N)$ ceiling, and SMILE [@song2020understanding], which
has no ceiling at the cost of moderately higher variance.

**Training.** The `Trainer` class implements memory-safe $N^2$ chunked evaluation
to prevent out-of-memory errors during full-dataset InfoNCE computation, in-memory
model checkpointing via `copy.deepcopy` to eliminate disk I/O bottlenecks, and
smoothed early stopping (median filter followed by Gaussian filter on the
validation curve) for robust convergence detection.

**Bias correction.** `mode='rigorous'` trains models on subsets of size $N/\gamma$
for $\gamma = 1, \ldots, 10$ and fits a weighted linear regression in $1/\gamma$
space, exploiting the theoretical $I_{\text{estimated}} \approx I_{\text{true}} +
a/N$ relationship. The y-intercept is the bias-corrected estimate; the fit
variance yields the confidence interval. A linearity check rejects points where
the quadratic contribution exceeds a configurable threshold, and a warning is
issued when fewer than five gamma points survive.

**Dimensionality.** `mode='dimensionality'` estimates the intrinsic dimensionality
of a neural population or of the shared information between two populations. After
training a Hybrid Critic with a large bottleneck, the library extracts embeddings
and computes the Participation Ratio from the singular value spectrum of the
cross-covariance matrix: $\text{PR} = (\sum_i \sigma_i)^2 / \sum_i \sigma_i^2$.
This avoids the computational cost and geometric artefacts of sweeping bottleneck
dimension.

**Tutorials.** Eight Jupyter notebooks cover the complete workflow from basic
estimation on IID data (Tutorial 1) through neural data formats (2), temporal
splits (3), hyperparameter sweeps (4), rigorous estimation (5), temporal analyses
(6), population dimensionality on real hippocampal and Allen Brain Observatory
recordings (7), and model and estimator selection (8).

# Acknowledgements

[ACKNOWLEDGEMENTS — TO FILL]

# References
