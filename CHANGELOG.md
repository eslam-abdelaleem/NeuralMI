# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Item 1 — Enhanced Rigorous Mode Diagnostics
- **Standardized-residual check (Check A):** After the WLS bias-correction fit,
  `rigorous` mode now computes externally studentized residuals.  If
  `max(|rᵢ|) > residual_threshold` (default 2.5) **or** R² < `r2_threshold`
  (default 0.90), `fit_quality_warning=True` is stored in `result.details` and
  `is_reliable` is set to `False`.
- **LOO γ=1 intercept-stability check (Check B):** Refits WLS excluding all γ=1
  rows and measures the relative intercept shift
  `δ = |I_full − I_loo| / (|I_full| + ε)`.  If `δ > leverage_threshold`
  (default 0.20), `leverage_warning=True` is stored in `result.details` and
  `is_reliable` is set to `False`.
- Both checks store their source in `result.details`:
  `fit_quality_warning`, `leverage_warning`, `r_squared`, `max_abs_residual`,
  `loo_intercept_shift`.
- New configurable thresholds in `base_params` / `analysis_kwargs`:
  `residual_threshold` (default 2.5), `r2_threshold` (default 0.90),
  `leverage_threshold` (default 0.20).
- `Results.summary()` now prints diagnostic reasons when `is_reliable=False`.

#### Item 2 — Optional Decoder (Deep Symmetric Information Bottleneck)
- New `use_decoder=True` flag in `base_params` enables decoder-augmented training
  for all analysis modes.
- New `base_params` keys:
  - `use_decoder` (bool, default `False`)
  - `decoder_weight` (float, default 1.0) — reconstruction weight applied to both X and Y.
  - `decoder_weight_x` / `decoder_weight_y` (float | None, default `None`) —
    per-channel overrides; when `None` the shared `decoder_weight` is used.
  - `decoder_output_activation_x` / `decoder_output_activation_y` (str,
    default `'linear'`) — `'linear'` for continuous, `'sigmoid'` for binary/spike,
    `'softmax'` for categorical.
- New module `neural_mi/models/decoders.py` with decoder variants for all six
  embedding architectures: `MLPDecoder`, `CNN1DDecoder`, `GRUDecoder`,
  `LSTMDecoder`, `TCNDecoder`, `TransformerDecoder`, and a `build_decoder()`
  factory function.
- Training objective:
  - Deterministic: `L = −MI(Z_X; Z_Y) + w_x·MSE(X, X̂) + w_y·MSE(Y, Ŷ)`
  - Variational: `L = KL_X + KL_Y − β·MI(Z_X; Z_Y) + w_x·MSE + w_y·MSE`
- Decoder and encoder parameters are optimised jointly by the same optimizer.
- `result.details['decoder_recon_loss']` reports the weighted reconstruction loss.
- `Results.summary()` prints decoder reconstruction loss when present.

#### Item 3 — Rigorous Bias Correction for Conditional and Transfer Modes
- `mode='conditional'` and `mode='transfer'` now accept `rigorous=True` in
  `analysis_kwargs` (or as a top-level `run()` keyword) to produce a
  bias-corrected, extrapolated estimate.
- Uses correlated subsampling (same master permutation index for all component
  estimates at each γ) so noise partially cancels in the difference.
- New parameters for conditional/transfer rigorous mode: `gamma_range`,
  `delta_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`,
  `r2_threshold`, `leverage_threshold`.
- Returns a full rigorous details dict: `mi_corrected`, `mi_error`, `slope`,
  `is_reliable`, `gammas_used`, `fit_quality_warning`, `leverage_warning`,
  `r_squared`, `max_abs_residual`, `loo_intercept_shift`, `raw_results_df`.
- Graceful fallback: when the linear-region finder prunes too aggressively (noisy
  data with no clear 1/γ trend), `run_rigorous_scalar_analysis` falls back to
  using all available γ values and sets `is_reliable=False`.
- New public function `run_rigorous_scalar_analysis()` in
  `neural_mi/analysis/rigorous.py` for use with any scalar MI-derived quantity.
- Pairwise mode: per-pair rigorous estimation will be addressed in a future release.

### Changed
- `neural_mi/models/critics.py`: Added `get_training_embeddings(x, y)` method to
  `BaseCritic` — returns embeddings with gradient flow (used by decoders during
  training).
- `neural_mi/training/trainer.py`: Added `decoder_x`, `decoder_y`,
  `decoder_weight_x`, `decoder_weight_y` constructor parameters; training loop
  now incorporates decoder reconstruction loss when decoders are present.
- `neural_mi/analysis/task.py`: Builds and passes decoders to `Trainer`; added
  decoder keys to `_BUILD_PARAMS_KEYS` for model serialisation.
- `neural_mi/run.py`: Pairwise mode unit conversion now correctly handles
  `mi_mean`/`mi_std` columns in addition to legacy `mi_estimate`.
- `neural_mi/defaults.py`: `MODE_KWARGS_SCHEMA` now includes a `'conditional'`
  entry (previously missing); `'transfer'` entry extended with rigorous params.
- `neural_mi/results.py`: `Results.summary()` extended to display rigorous
  diagnostic reasons and decoder reconstruction loss.

### Tests Added
- `tests/test_rigorous_diagnostics.py` — unit and integration tests for
  `_compute_fit_diagnostics`, `_post_process_and_correct`, decoder shapes,
  output activations, end-to-end decoder training, and
  `run_rigorous_scalar_analysis`.
- `tests/test_conditional_transfer_rigorous.py` — end-to-end tests for
  `rigorous=True` in conditional and transfer modes.

## [Unreleased]

### Added

- **`dataset_device` parameter**: controls where dataset tensors are stored in
  memory, independent of the compute device (`device` param).  Default is
  `'cpu'` for all modes, which keeps large arrays in pageable system RAM so the
  OS can reclaim memory freely between tasks.  Pass `'auto'` to co-locate data
  with the compute device (MPS / CUDA), which avoids repeated host→device
  transfers when the same dataset is evaluated many times — precision analysis
  uses `'auto'` by default for exactly this reason.  Any explicit device string
  is also accepted.  Added to `BASE_PARAMS_SCHEMA` in `defaults.py`.
- **Module-level dataset cache in `task.py`**: sequential sweep tasks that share
  identical data and dataset-construction parameters (processor type / params /
  `dataset_device`) now reuse a single pre-built `PairedDataset` object instead
  of re-running `create_dataset()` for every task.  The cache is keyed by data
  memory address and construction fingerprint; LRU eviction keeps at most four
  entries.  Temporal datasets (`PairedTemporalDataset`) are intentionally
  excluded from caching because they are mutated in-place by `time_shift()`
  during training.  The cache is process-local and also benefits
  `multiprocessing` workers that handle more than one task.
- **Memory-pressure warning in `ParameterSweep`**: when `dataset_device` is not
  `'cpu'` and a sequential sweep has more than 20 tasks, a `UserWarning` is
  emitted before training starts, estimating dataset size and advising the user
  to set `dataset_device='cpu'` if memory pressure is a concern.

### Fixed

- **Root-cause fix for MPS/CUDA memory exhaustion during long sweeps**: all
  dataset classes (`StaticDataset`, `ContinuousWindowDataset`,
  `SpikeWindowDataset`, `BinnedSpikeDataset`, `CategoricalWindowDataset`)
  previously stored `self.data` on the accelerator device by default (via
  `get_device()`).  On Apple Silicon (unified DRAM) this caused the full
  dataset to be allocated on MPS for every training task in a sequential sweep,
  and PyTorch's MPS allocator does not return freed tensors to the OS without
  an explicit `torch.mps.empty_cache()` call.  With 300 tasks this caused
  monotonic memory growth and system crashes.  The fix moves all dataset tensor
  storage to CPU by default; batch loops in the `Trainer` already call
  `.to(device)` per batch so no training logic changed.
- **`SubsetView`: device-agnostic indexing**: index tensors are now always
  created as CPU LongTensors, and `__getitem__` converts any 0-dim index
  tensor to a Python `int` before delegating to the dataset.  Python `int`
  indices work on any device tensor (CPU or accelerator), eliminating the
  previous `RuntimeError` when dataset data was on CPU but index tensors were
  on MPS, and making `SubsetView` safe for use with `dataset_device='auto'`.
- **`SpikeWindowDataset.apply_precision()` now reads from `data_master`**:
  previously the method rounded `self.data` in-place while reading from
  `self.data` as the source.  Calling it twice at different precision levels
  would compound the rounding error rather than starting from the original
  spike times.  The fix mirrors the implementation in
  `ContinuousWindowDataset` and `BinnedSpikeDataset`, which already read from
  `self.data_master`.
- **`PairedDataset._align_datasets()` now performs effective truncation**:
  when X and Y datasets have different sample counts, the method now slices
  `self.data` on both sides so that `__len__()` (which reads
  `self.data.shape[0]`) reports the correct length.  The previous
  implementation set a phantom `n_windows` attribute that `StaticDataset` does
  not use, leaving mismatched datasets that would crash during training.  Any
  lazily-allocated `data_master` is also invalidated so it is re-cloned from
  the truncated data on next use.
- **`CategoricalWindowDataset._move_full_trajectory()` no longer assigns
  `data_master` twice**: the method previously set `self.data_master` at the
  end of its body while `move_data_to_windows()` also set it unconditionally
  after every encoding method returned.  The redundant internal assignment is
  removed; all three encoding paths now consistently delegate `data_master`
  initialization to their single caller.
- **`DEVELOPERS_GUIDE.md` processor file reference corrected**: entries
  referring to a non-existent `processors.py` file have been updated to point
  to the correct files (`handler.py`, `temporal.py`, `static.py`) and the
  step-by-step guide for adding a new data processor now reflects the actual
  codebase structure.

## [2.1.0] - 2026-03-13

### Added

- **Optimizer flexibility**: `optimizer` parameter accepts a string name
  (`'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'adagrad'`) or any
  `torch.optim.Optimizer` subclass; `optimizer_params` forwards extra
  constructor kwargs (e.g. `weight_decay`).
- **Learning-rate schedulers**: new `scheduler` / `scheduler_params` parameters
  in `run()` and `base_params`. Supported names: `'cosine'` (CosineAnnealingLR),
  `'cosine_warmup'` (linear warm-up + cosine), `'step'` (StepLR),
  `'plateau'` (ReduceLROnPlateau, monitors test MI). Custom
  `torch.optim.lr_scheduler` subclasses also accepted. Scheduler steps at the
  end of each training epoch; `ReduceLROnPlateau` receives the current test MI.
- **MLP regularisation**: `dropout` (float, default 0.0) and `norm_layer`
  (`None`/`'layer'`/`'batch'`, default `None`) parameters for MLP and VarMLP
  embedding networks, applied in the order Linear → Norm → Activation → Dropout.
- **Per-epoch train MI tracking**: `eval_train` parameter (`False` / `True` /
  fraction / sample count) records train-set MI at every epoch, populating
  `result.details['train_mi_history']`. The training curve plot overlays the
  dashed orange train curve automatically when this key is present.
- **Raw train MI**: `result.details['raw_train_mi']` is always populated with
  the true computed train-set MI regardless of whether the model generalised
  (when the model fails, `train_mi` is set to 0 while `raw_train_mi` preserves
  the actual value for diagnostic purposes).
- **Small-sample warnings**: after dataset creation, `run()` emits a `UserWarning`
  when the processed dataset has fewer than 200 windows (strong) or fewer than
  500 windows (mild) with specific regularisation suggestions.
- **Tutorial 09**: new end-to-end pipeline tutorial covering sanity check,
  window sweep, architecture sweep, training diagnostics, rigorous estimation,
  lag analysis, conditional MI, and summary reporting on synthetic hippocampal
  data.

### Fixed

- **Sweep MPS/CUDA bug**: `ParameterSweep._prepare_tasks` incorrectly called
  `.is_mps` on numpy arrays when `torch.backends.mps.is_available()` was True,
  causing `AttributeError` during parallel sweeps. Fixed operator precedence
  and added `isinstance(tensor)` guard.
- **`is_proc_sweep` auto-detection**: `ParameterSweep.run()` and
  `_prepare_tasks()` now infer `is_proc_sweep` automatically from data type
  (3-D `torch.Tensor` → pre-processed; everything else → raw). The parameter
  still accepts an explicit `bool` for backward compatibility.
- **Unit conversion in `estimate` mode**: `train_mi`, `raw_train_mi`,
  `test_mi_history`, and `train_mi_history` were returned in nats even when
  `output_units='bits'`. All four keys are now correctly converted.
- **All-MI-non-positive threshold**: changed from `< 0` to `<= 0` so that
  exactly-zero MI values correctly trigger the model-failure warning.

### Changed

- Tutorial 08 updated with new sections on optimizer choice, MLP regularisation,
  training diagnostics (`eval_train`), and LR schedulers.
- `NEURALMI_REFERENCE.md` updated with new parameters in the `run()` reference
  block and Base Parameters table (Optimizer/Scheduler section; dropout,
  norm_layer, eval_train).

## [2.0.0] - 2026-03-10

### Added

- Unified `run()` entry point supporting nine analysis modes: `estimate`, `sweep`,
  `rigorous`, `lag`, `dimensionality`, `precision`, `conditional`, `transfer`,
  `pairwise`.
- `rigorous` mode: automated finite-sampling bias correction via subsampling and
  linear extrapolation to the infinite-data limit, with confidence intervals.
- `dimensionality` mode: latent dimensionality estimation using a Hybrid Critic and
  cross-covariance SVD (Participation Ratio).
- `precision` mode: spike-timing precision analysis via "Train Once, Evaluate Many"
  with deterministic rounding and additive noise corruption methods.
- `transfer` mode: transfer entropy estimation using the chain-rule decomposition.
- `conditional` mode: conditional mutual information I(X;Y|Z).
- `pairwise` mode: channel-to-channel MI matrix (self-pairwise and cross-pairwise).
- Data processors: `ContinuousProcessor`, `SpikeProcessor`, `CategoricalProcessor`
  for LFP/EEG, spike-train, and categorical state data respectively.
- Embedding models: MLP, CNN1D, GRU, LSTM, TCN, Transformer.
- Critic architectures: `SeparableCritic`, `ConcatCritic`, `HybridCritic`.
- MI estimators: InfoNCE, SMILE.
- Blocked and random train/test splitting strategies for temporal and IID data.
- Built-in synthetic data generators (`generate_correlated_gaussians`,
  `generate_nonlinear_from_latent`, `generate_temporally_convolved_data`,
  `generate_correlated_spike_trains`, `generate_xor_data`,
  `generate_correlated_categorical_series`, `generate_event_related_data`).
- Permutation test support for null-distribution estimation.
- `extract_embeddings` utility for extracting learned latent representations.
- `spectral_mode` parameter for controlling spectral metric computation.
- Comprehensive tutorial series (Tutorials 01–08) covering basic estimation,
  neural data formats, temporal analysis, sweeps, rigorous estimation,
  population-level questions, and model/estimator selection.
  Tutorial 09 (full end-to-end pipeline) added in v2.1.0.
- `NEURALMI_REFERENCE.md`: complete library reference document.
- `THEORY.md`: theoretical background for all core methods.
- `CONCEPTS.md`: code-based walkthrough of MI estimator internals.
- `DEVELOPERS_GUIDE.md`: contributor guide to the codebase architecture.
