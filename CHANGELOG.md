# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **`z_time` parameter in `run()`**: a time vector can now be passed for the
  conditioning variable Z in `mode='conditional'` when `z_processor_type` is a
  temporal processor (e.g. `'continuous'`). Forwarded to `create_dataset` as
  `x_time=z_time`.
- **`Results.save(path=None)`**: serialises a Results object to a pickle file.
  Auto-generates a timestamped filename (`neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl`)
  in the current directory when no path is given; never overwrites existing files
  (appends a numeric suffix). Returns the absolute path of the saved file.
- **`Results.load(path)`**: classmethod that deserialises a Results object
  previously saved with `save()`.
- **`Results.to_json(path=None)`**: exports a human-readable JSON snapshot of
  scalar fields (`mode`, `mi_estimate`, `params`) and the DataFrame. Large objects
  in `details` (numpy arrays, raw result lists) are summarised by type and shape.
  Auto-naming and no-overwrite logic follow the same convention as `save()`.
- **`sample_rate` parameter wired** into `ContinuousWindowDataset` and
  `CategoricalWindowDataset`. When provided it overrides the period inferred from
  the time vector; now propagated from `processor_params_x/y` via `handler.py`.
- **`max_spikes_per_window` and `n_seconds` parameters wired** into
  `SpikeWindowDataset`. `max_spikes_per_window` caps the allocated spike slot
  count; `n_seconds` sets an explicit recording duration for temporal extent
  inference. Both are now propagated from `processor_params_x/y` via `handler.py`.

### Changed
- **Precision mode `Results.mi_estimate`**: now holds the baseline MI (at zero
  corruption) rather than the precision threshold τ. The threshold τ remains in
  `Results.details['precision_tau']`. `Results.summary()` for precision mode
  now shows baseline MI, τ, and the threshold MI value explicitly.
- **Pairwise mode DataFrame columns**: the `mi_estimate` column is replaced by
  `mi_mean` and `mi_std` (consistent with sweep/lag modes). The MI matrix
  continues to hold per-pair means.
- **`Results.summary()`** for `conditional`, `transfer`, and `pairwise` modes
  now prints mode-relevant detail (component MI values, directionality index,
  matrix range) in addition to the generic DataFrame shape.
- **`bidirectional_te` is exclusively a top-level `run()` parameter**. It has
  been removed from `MODE_KWARGS_SCHEMA['transfer']` to eliminate the
  dual-pathway inconsistency where the same parameter could be set in two places.
- **Non-processor, non-lag calls skip an intermediate `PairedDataset`**
  allocation when both `processor_type_x` and `processor_type_y` are `None`.
  Tensor conversion and length alignment now happen inline in `run._run_inner`,
  removing a redundant object construction on every non-temporal call.

### Fixed
- **`_initialize_windows` min/max inversion** (`handler.py`): the `min` and `max`
  operators were swapped, causing the temporal window range to *expand* beyond the
  original recording after `time_shift()` instead of being clamped to it.
- **`_BUILD_PARAMS_KEYS`** (`task.py`): `dropout`, `norm_layer`, and
  `use_spectral_norm` were missing. Models saved with `norm_layer='batch'` or
  `norm_layer='layer'` can now be reloaded correctly via `extract_embeddings()`.
- **Precision mode `n_test_blocks`** (`precision.py`): was read from `**kwargs`
  (ignored when set in `base_params`). Now reads from `base_params`.
- **Noise mask in `apply_noise`**: `ContinuousWindowDataset` and
  `SpikeWindowDataset` derived the non-zero position mask from `self.data` (the
  working copy), causing repeated noise applications to compound. Both now derive
  the mask from `self.data_master`.
- **`window_size` in `processor_params_y`** was silently ignored when Y specified
  a different value. Now emits a clear warning and uses X's `window_size` (shared
  `WindowManager` constraint).
- **Pairwise permutation test** was silently discarded when `permutation_test=True`.
  Now emits a `UserWarning` about computation cost and populates
  `results.details['null_distribution']` with the null MI samples.
- **Conditional MI log message** incorrectly annotated MI values as being in
  `output_units`; values are always in nats at that point in the code.
- **Transfer entropy `bidirectional=False` log level**: demoted from `warning`
  to `info` (this is the normal, expected default — not a user-facing warning).
- **`trainer.py` variable name**: `is_first_valid_epoch` renamed to
  `has_valid_baseline` to reflect its actual semantics (True once a baseline MI
  has been established, not on the first valid epoch).

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
