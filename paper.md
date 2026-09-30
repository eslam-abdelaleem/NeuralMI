---
title: 'NeuralMI: Information-theoretic analysis of neural data at scale'
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
    affiliation: 1,2
  - name: Leo Wood
    orcid: 0000-0003-1829-7781
    affiliation: 3
  - name: Audrey Sederberg
    orcid: 0000-0003-4458-3773
    affiliation: 1,2
affiliations:
  - name: School of Physics, Georgia Institute of Technology
    index: 1
  - name: School of Psychological and Brain Sciences, Georgia Institute of Technology
    index: 2
  - name: Princeton Neuroscience Institute
    index: 3
date: 30 September 2026
bibliography: paper.bib
---

# Summary

Neural signals depend on one another and on the world in ways that are rarely
linear. Information theory measures that dependence in bits whatever its form.
It also asks questions that simple linear measures like variances and correlations have no form for: how much a population's
past predicts its own future, which of two areas leads the other and by how
long, how precisely spikes must be timed to carry a message, what one signal
adds about another beyond what a third already says, and what a pair of signals
carries that neither carries alone. Each of these questions is answered by an
information quantity estimated from recorded samples.

Those estimates have been hard to obtain from modern recordings. Classical
estimators need a number of samples that grows quickly with the number of
dimensions they are given. Recordings now reach thousands of channels. At that
size the classical estimators need more data than an experiment can collect.
Information-theoretic analysis has therefore stayed with a few channels at a
time.

`NeuralMI` brings these analyses to the scale of modern neuroscience. It
estimates mutual information (MI) with neural networks that learn a compact
embedding of each signal that information is estimable in. This allows for accurate estimation at large dimensional setups that were not possible before.
The same estimation process can answer a whole family of questions like what we asked above. Also, spike times, continuous recordings and behavioural labels each pass through an encoder suited to their modality and are compared in the same unit.

# Statement of need

Estimating MI from finite samples is hard in general and especially hard for high-dimensional systems. Simple approaches like binning need a number of
samples that grows exponentially with the dimension. Classic approaches like the $k$-nearest-neighbour
estimator of @kraskov2004estimating lose accuracy past about ten dimensions
[@holmes2019estimation]. Neural estimators [@belghazi2018mutual;
@oord2018representation; @poole2019variational; @song2020understanding] train a
network to map each variable to a low-dimensional embedding in which the
information can be estimated. The samples they need then follow the latent
structure of the data and not the number of channels it arrived on
[@abdelaleem2025accurate]. \autoref{fig:sweep} shows the difference on two
populations of a thousand channels each.

![Two populations of 1000 channels share four bits through ten latent
dimensions. NeuralMI recovers the four bits by a thousand samples. The
$k$-nearest-neighbour estimator is still below two bits at five
thousand.\label{fig:sweep}](docs/source/_static/sample_sweep.png)


Every quantity below can be built from the conditional MI $I(A;B \mid C)$
between chosen signals at chosen time offsets. Using 
the chain rule we can write each conditional quantity as a difference of two MI estimates that we know how to estimate properly:
$$
I(A;B \mid C) = I([A,C];B) - I(C;B).
$$

Using this decomposition, we can estimate quantities like active information storage which asks what a signal's
past says about its present, transfer entropy that asks what one signal's past adds
about another's next step once the second signal's own past is known, interaction
information that compares what a pair of signals carries together with what each
carries alone, etc. 

\autoref{fig:taxonomy} shows twelve quantities as patterns of time offsets over
this one primitive. An estimator that makes MI work at scale therefore makes the
whole family work at scale.

![Twelve information quantities as patterns of time offsets. Each row marks the
time steps of $X$, $Y$ and $W$ that play $A$, $B$ and the conditioned-out $C$ in
$I(A;B \mid C)$.\label{fig:taxonomy}](docs/source/_static/taxonomy.png)


# State of the field

Several toolboxes estimate information-theoretic quantities from neural data.
JIDT [@lizier2014jidt] and IDTxl [@wollstadt2019idtxl] compute transfer entropy,
active information storage and related measures and infer networks from them.
NIT [@maffulli2022nit] estimates information in small populations with
limited-sampling bias corrections. Frites [@combrisson2022group] builds
group-level statistics on the Gaussian-copula estimator of @ince2017statistical.
dit [@james2018dit] works with discrete distributions. Their estimators count, find neighbours or assume Gaussian dependence. Each of these degrades or becomes
an approximation as the number of channels grows.

Neural estimators of MI are mostly released as code accompanying a paper. They are often used as an objective to obtain better representations. Only recently was it shown how to use them as accurate estimators with proper regularisation, stopping rules and error bars [@abdelaleem2025accurate]. Such literature rarely handles the formats, windowing, splits, units or bias of neural recordings.

The existing toolboxes are organised around estimators computed in one pass. A
neural estimate comes out of a procedure: a train and test split, an
epoch-selection rule, repeated fits and an extrapolation over subsets of the
data. Adding it to one of them would place a training loop and its diagnostics
under an interface designed for one-pass estimators. `NeuralMI` is built around
that loop.

# Software design

**Encoders for each modality.** Each stream is cut into windows on its own
clock. The windows of all streams are aligned on one grid in real time
(\autoref{fig:alignment}). The grid starts at a new random offset in every epoch
so that the network sees new windows without a sliding-window array ever being
stored. Each side of an estimate has its own encoder chosen for its modality:
fully connected, convolutional in one or two dimensions, recurrent,
temporal-convolutional, transformer, linear recurrent, set-based or a pretrained
image network. Custom encoders and critics plug in through the same interface.
Whatever the encoders, the estimate is information in bits. Spike trains,
positions and trial labels are then compared on the same ruler.

![Three streams on their own clocks (spike times, a position trace with a gap
and trial labels) cut into windows aligned in time. Each stream then passes
through its own encoder.\label{fig:alignment}](docs/source/_static/alignment.png){ width=90% }

**Two ways in.** The quantities are functions, one per question, grouped by the
number of signals they involve. `active_information_storage` and
`predictive_information` take one signal. `instantaneous_mi`, `block_mi`,
`cross_predictive_information`, `mi_rate`, `instantaneous_exchange`,
`directed_information_rate` and `transfer_entropy` take two.
`conditional_transfer_entropy` and `interaction_information` take three. Each
builds its time offsets from the raw series and returns the estimate. The modes
of `nmi.run(x, y, mode=...)` cover the analyses that are not a single quantity.

- `sweep` repeats an estimate and runs it over a grid of settings.
- `rigorous` corrects the finite-sample bias.
- `lag` scans the MI over time shifts between the two signals.
- `precision` measures how the MI falls as spike times or values are coarsened.
- `pairwise` estimates the MI between every pair of channels.
- `dimensionality` finds the smallest embedding that carries the shared
  information [@gulati2026mutual].

**One result.** Every quantity and every mode returns the same `Results` object:
one row per trained network, one row per configuration with the mean and spread
over repeats, and the diagnostics of the analysis. Repeats and grids of settings
work the same way everywhere. Settings are validated before any training. A
setting that would have no effect is refused or named in a warning.

**A number that can be reported.** Neural estimators are lower bounds capped at
the logarithm of the number of samples scored together [@mcallester2020formal].
`NeuralMI` trains each network on one partition of the data and evaluates both
partitions after every epoch. The peak of the smoothed held-out curve selects
the epoch. The training-side estimate at that epoch is reported with its ceiling
[@abdelaleem2025accurate]. The default split holds out contiguous blocks with a
gap because overlapping windows are near copies of one another.
`mode='rigorous'` estimates the MI on the data cut into $\gamma = 1, \ldots, 10$
equal parts and extrapolates linearly in $\gamma$ to infinite data
[@strong1998entropy; @holmes2019estimation; @abdelaleem2025accurate]. A
quantity built as a difference of estimates reports an amplification factor for
the error it inherits from them. A quantity that cannot be negative is reported as 0 when it
comes out negative and is left out of averages.

**Reproducibility.** Each task seeds itself from the call's seed and its
position in the grid. A result is the same at any number of worker processes.
Warnings raised in workers are relayed to the caller's line and shown once per
call or notebook cell with a count. Every message the library can print is documented by its
text. A test checks the documentation against the code in both directions.

# Research impact statement

`NeuralMI` implements the estimation and bias-correction procedure of
@abdelaleem2025accurate, the dimensionality analysis of @gulati2026mutual and
the variational encoders and decoders of @abdelaleem2025deep. It opens the door to answering questions that were not possible at that scale before. Five tutorial
notebooks check every quantitative claim against an exact value from a generator
that knows the answer. More than 1,500 tests, a documentation site and seven
reference documents come with the package. `NeuralMI` is released under
the MIT licence.

# AI usage disclosure

Generative AI (Anthropic's Claude, Google's Jules) was used under the authors' direction and based on their previously written code to
write and refactor code, tests, documentation, and tutorials. The authors reviewed its output. Correctness is checked against quantities that can
be computed exactly. The library generates data whose MI is known in closed form
(correlated Gaussians, nonlinear maps of Gaussian latents, discrete joint
distributions and jointly Gaussian time series with shared latents). Its tests
and tutorials compare every estimator and every quantity against those exact
values.

# Acknowledgements

# References
