# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-21

First public release. NeuralMI estimates mutual information from neural
recordings through a single `nmi.run()` entry point.

### The API

One function, `nmi.run()`, takes typed config objects (`Model`, `Training`,
`Split`, `Processing`, and one per mode) and returns a `Results` object carrying
the estimate, its metadata, and `plot()`.

Ten analysis modes cover one number (`estimate`), a parallelised hyperparameter
grid (`sweep`), finite-sample bias correction by extrapolation (`rigorous`),
temporal offsets (`lag`), spike-timing precision (`precision`), conditional MI
(`conditional`), transfer entropy (`transfer`), interaction information
(`interaction`), all-to-all matrices (`pairwise`), and the smallest embedding
dimension that carries the shared information (`dimensionality`).

Eleven named quantities (`active_information_storage`, `predictive_information`,
`cross_predictive_information`, `instantaneous_mi`, `block_mi`, `mi_rate`,
`instantaneous_exchange`, `directed_information_rate`, `transfer_entropy`,
`conditional_transfer_entropy` and `interaction_information`) are requested by
name and each build the offset pattern they need.

### Data

Continuous, spike and categorical processors take LFP, EEG, calcium,
kinematics, raw spike times and behavioural or stimulus labels onto one
`(n_samples, n_channels, window_size)` grid. `create_dataset` accepts a pair or
a mapping of any number of named streams and aligns them on real time, honouring
per-stream sample rates and time vectors.

Splitting is `blocked` by default and leaves a configurable gap between the
blocks. Overlapping windows are near copies of one another. A random split that
places copies on both sides inflates the estimate. `random` is there for
independent samples.

### Estimators and models

The estimator is InfoNCE, SMILE or KSG (Kraskov, Stögbauer and Grassberger),
selectable per run. KSG trains no network and takes the difference quantities
through its conditional form. The library builds ten embedding architectures
(`mlp`, `cnn`, `cnn2d`, `gru`, `lstm`, `tcn`, `transformer`, `lru`, `deepsets`
and `pretrained_backbone`) and three critics (`separable`, `concat` and
`hybrid`). Custom critics and embedding classes are supported. The two sides of
a critic can carry different encoders through the `_y` settings. Every term of a
difference quantity encodes each part of X on its own branch of the chosen
encoder under one small head. The terms then learn the conditioning set the
same way.

### Reading a number honestly

Because the neural estimators are variational lower bounds, the library reports
what bounds its own answer. The reported value is read at the earliest epoch at
the top of the smoothed held-out curve. `curve_diagnosis` says when the curve
cannot support that choice. Every estimate carries the ceiling of the partition it
was evaluated on. Difference quantities carry an `amplification_factor` saying
how much component error the answer inherits. `mode='rigorous'` reports
`is_reliable` and `mi_error` alongside the corrected estimate. Optional
permutation testing gives a null distribution where the mode supports one. A
quantity that cannot be negative is never reported below zero and keeps its
measured value beside the reported one. Averages over the repeats of a
difference or a rigorous fit use every repeat's measured value with its sign. A
single network's repeat that learned nothing that generalises is left out and
counted. A setting that would
have no effect is named in a warning. The call refuses settings it cannot
honour: an estimator parameter the estimator does not take, or custom split
indices on a mode that builds its own rows.

A message that needs attention is a Python warning, shown once per kind in a
call or a notebook cell and counted after. Every message the library can emit
is documented by its text in `reference/MESSAGES.md`.

### Documentation

Five tutorial notebooks meant to be read in order, a generated API reference,
and seven reference documents covering usage, every parameter, theory,
messages, the anatomy of an estimator, the internals, and the test suite.
