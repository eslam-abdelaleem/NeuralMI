# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

First public release. `NeuralMI` provides an end-to-end workflow for rigorous
mutual information estimation from neural data through a single `nmi.run()`
entry point.

### Added

- **Unified API.** All analysis modes are accessed through one `nmi.run()`
  function returning a standardized `Results` object with estimates, metadata,
  and built-in plotting.
- **Nine analysis modes:** `estimate` (single MI estimate), `sweep`
  (parallelized hyperparameter sweeps), `rigorous` (finite-sample bias
  correction with confidence intervals), `dimensionality` (latent
  dimensionality via participation ratio), `lag` (temporal offset analysis),
  `precision` (spike-timing precision thresholds), `conditional` (conditional
  MI), `transfer` (transfer entropy), and `pairwise` (all-to-all connectivity
  matrices).
- **Neuroscience-native data processors:** `ContinuousProcessor` (LFP, EEG,
  calcium, kinematics), `SpikeProcessor` (raw spike trains), and
  `CategoricalProcessor` (behavioural/stimulus state), all producing a unified
  `(n_samples, n_channels, window_size)` tensor with memory-efficient
  non-overlapping windowing.
- **Bias correction.** `mode='rigorous'` trains on subsets of decreasing size
  and extrapolates the theoretically predicted `O(1/N)` bias to the infinite-data
  limit via weighted least squares, with linearity and leverage diagnostics.
- **MI estimators:** InfoNCE (low variance, `log N` ceiling) and SMILE
  (unbounded), selectable and configurable per run.
- **Model architectures:** embedding networks for time series and images —
  MLP, CNN, CNN2D, GRU, LSTM, TCN, Transformer, and a frozen pretrained image
  backbone — and three critic types (`separable`, `concat`, `hybrid`), plus
  support for user-provided custom critics and embedding classes.
- **Smart data splitting:** `blocked` splitting with a configurable gap for
  temporal data (prevents autocorrelation leakage) and `random` splitting for
  IID data.
- **Statistical tooling:** optional permutation testing for null-distribution
  significance across supported modes.
- **Visualization:** dimensionality curves, bias-correction fits, pairwise
  heatmaps, noise-injection ladders, and embedding projections (PCA / t-SNE /
  UMAP via the optional `viz` extra).
- **Documentation:** nine tutorial notebooks, a full API reference, theory and
  concepts guides, and a hosted documentation site.
