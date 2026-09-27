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
  - name: Eslam Abdelaleem
    orcid: 0009-0006-9429-3589
    affiliation: 1
affiliations:
  - name: Georgia Institute of Technology
    index: 1
date: 10 March 2026
bibliography: paper.bib
---

# Summary

`NeuralMI` is a Python library for estimating mutual information (MI) from
neural recordings with neural-network-based estimators. MI measures how much
knowing one variable narrows down another, in bits, and it captures linear and
nonlinear dependencies alike. Because it assumes no form for the relationship,
it applies to the high-dimensional, nonlinear signals of modern neuroscience.

Every analysis goes through one function, `nmi.run()`. It accepts continuous
recordings, spike times and categorical labels, each on its own clock, and
returns a `Results` object with the estimates, their spread across repeats and
the diagnostics of every trained network. Ten analysis modes cover a single estimate
(`estimate`), a hyperparameter grid (`sweep`), bias-corrected estimation with a
confidence interval (`rigorous`), temporal lags (`lag`), spike-timing precision
(`precision`), conditional MI (`conditional`), transfer entropy (`transfer`),
interaction information (`interaction`), all-to-all channel matrices
(`pairwise`) and directions of shared structure (`dimensionality`). Eleven named
quantities, from active information storage to the directed information rate,
are each one choice of time offsets in a single conditional MI,
$I(A;B \mid C)$. The `rigorous` mode implements the subsampling-and-extrapolation
bias correction of @abdelaleem2025accurate, and the `dimensionality` mode builds
on the cross-covariance spectral method of @gulati2026mutual.

# Statement of need

Classical
non-parametric estimators such as the $k$-nearest-neighbour method of
@kraskov2004estimating need samples in proportion to the dimensions they are
handed, and past roughly ten dimensions no recording of realistic length is
enough. Neural-network-based estimators [@oord2018representation;
@song2020understanding] learn a low-dimensional embedding of each variable and
estimate the information in it, so the samples they need follow the latent
structure of the data and not the channel count it arrived on. They bring
problems of their own. They are lower bounds with a ceiling set by how many
samples are scored together, and with finite data they carry a systematic bias
of order $1/N$ that can be large at the sample sizes of typical experiments.
Existing implementations, scattered across individual paper repositories, give
point estimates only, with no bias correction, no confidence intervals, no
handling of neural data formats and no common interface across analyses.

`NeuralMI` addresses these gaps in one package built around the workflow of
experimental neuroscience. First, it corrects the finite-sample bias by training
on nested subsets of the data and extrapolating along the predicted $1/N$ trend
[@abdelaleem2025accurate], and it reports the corrected value with a confidence
interval and a verdict on whether the extrapolation can be trusted. Second, it
ships processors for the three data formats most common in systems
neuroscience: continuous time series (LFP, EEG, calcium imaging, kinematics),
spike times, and categorical behavioural states. Streams on different clocks
are aligned on real time, and the window grid is redrawn at a random offset in
every epoch, so the network sees new windows without a sliding-window array
ever being stored. Third, the default train/test split is blocked, with a gap
between the blocks. Overlapping windows are near copies of one another, and a
random split that places copies on both sides inflates the estimate. Fourth,
every estimate reports the ceiling of the partition it was evaluated on, every
difference quantity reports how much component error it inherits, and every
warning the library can emit is documented by its text.

# Functionality

**Data.** `Processing` selects a processor for each stream (`'continuous'`,
`'spike'` or `'categorical'`) and its window settings, and each stream becomes
a `(n_samples, n_channels, window_size)` tensor. `create_dataset` aligns a pair
or any number of named streams on real time, honouring each stream's sample
rate or time vector, and keeps the windows in which every stream has data.

**Estimators and models.** InfoNCE [@oord2018representation] has low variance
and is capped at $\log K$, where $K$ is the number of samples scored together.
SMILE [@song2020understanding] clips the density ratio, has no such cap and is
noisier. Eleven embedding architectures, from an MLP to recurrent,
convolutional, transformer and pretrained image encoders, and three critics are
built in, and custom encoders and critics are accepted.

**Training and the reported number.** The data are split into a training and a
held-out partition. The held-out MI is smoothed across epochs, and its peak
selects the epoch whose weights are kept. The reported value is the MI on the
training partition at that epoch.

**Bias correction.** `mode='rigorous'` trains networks on the data cut into
$\gamma = 1, \ldots, 10$ equal parts and fits the estimate against $\gamma$ by
weighted least squares, following
$I_{\text{est}} \approx I_{\text{true}} + a\gamma/N$. The fit keeps the range of
$\gamma$ over which a quadratic term is statistically indistinguishable from
zero, and its intercept at $\gamma = 0$ is the corrected estimate. Four checks,
on the number of values of $\gamma$ left, the linear region, the leverage of
$\gamma = 1$ and the estimator's ceiling, decide whether the result is reliable.

**Quantities.** Each named quantity is $I(A; B \mid C)$ for one pattern of time
offsets. A conditional quantity is computed through the chain rule,
$I(A; B \mid C) = I([A, C]; B) - I(C; B)$, and reports an amplification factor,
the summed size of the two components relative to the result. A small
difference of two large estimates inherits their errors many times over, and the
factor measures by how much.

**Dimensionality.** `mode='dimensionality'` trains a Hybrid Critic `n_splits`
times on the same data, rotates each fit's embeddings by a singular value
decomposition of their cross-covariance so that the directions are ordered by
shared variance, and reports the directions that reproduce across every pair of
fits. It also reports two Participation Ratios (PR) of the singular values
$\sigma_i$: $\text{PR}_{\text{eig}} = (\sum_i \sigma_i^2)^2 / \sum_i \sigma_i^4$
and $\text{PR}_{\text{singular}} = (\sum_i \sigma_i)^2 / \sum_i \sigma_i^2$.

**Other analyses.** `mode='lag'` sweeps a time offset between the streams.
`mode='precision'` freezes a trained network and degrades the timing of its
input at increasing resolutions $\tau$, and it reports the $\tau$ at which the
information falls below a set fraction of its baseline. `mode='pairwise'` builds
all-to-all channel matrices. Permutation tests build a null distribution by
shifting X circularly in time while Y and W stay in place.

**Documentation.** Five tutorial notebooks, read in order, cover what mutual
information detects that correlation misses and where classical estimators
fail, what a single estimate is and what governs its accuracy, the catalogue of
quantities and their units, what processing and splitting do to a recording,
with two real hippocampal sessions [@grosmark2016diversity], and how to choose
the estimator and the architecture. Every quantitative claim in them is checked
against an exact value from a generator that knows the answer or against a
control measured on the same recording. Seven reference documents cover usage,
every parameter, the theory, every warning, and the internals.

# Acknowledgements

*Acknowledgements and funding sources to be added prior to submission.*

# References
