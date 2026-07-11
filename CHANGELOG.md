# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Ceiling-escape noise injection for `dimensionality` mode (`sigma_add`)** (`neural_mi/analysis/dimensionality.py`, `neural_mi/training/trainer.py`, `neural_mi/visualize/plot.py`):
  When the true MI exceeds the InfoNCE ceiling, the spectral (participation-ratio) readout becomes unreliable. `sigma_add` adds fixed, independent, per-channel Gaussian noise (in measured-per-channel-std units) to the observations once — before the embedding, identical for train and eval of a fit — to de-saturate the estimate without moving the true dimensionality.
  - `sigma_add`: a scalar (single level), a list (full ladder, one row per level), or `'auto'` (searches a geometric ladder for the detached regime, widening the search once if it doesn't bracket, then warning rather than looping).
  - `sigma_add_units`: `'relative'` (default, multiple of per-channel std) or `'absolute'`.
  - `stabilize_counts` (bool, default `True`): for binned-spike data, applies the canonical Anscombe transform before measuring std / injecting noise. Fires on every binned-spike dimensionality run regardless of `sigma_add`.
  - Ceiling classification keys on `log(eval_size)` (the actual InfoNCE evaluation denominator), never `log(batch_size)`; `Trainer.train()` now exposes `eval_size` in its results for this purpose.
  - Reproducible under `n_workers > 1`: the base noise tensor is deterministically reconstructed per `(global_seed, split_id, view)` from a pure hash-seeded RNG, no live shared state, and reused unscaled across every rung of the ladder within a split.
  - Supported for intrinsic `split_method in ('random', 'spatial')` and interaction mode; raw spike-timestamp and categorical data raise a clear error only when `sigma_add` is actually engaged.
  - New output: `result.details['sigma_add_ladder']` (per-rung `pr_eig`/`pr_singular` mean+std, resolved absolute scale, regime label, detached flag) and `result.details['sigma_add_suggestion']` (auto mode only, never a silent override). `result.dataframe` gains a `sigma_add` grouping column via the existing sweep-aggregation path.
  - New `plot_noise_ladder()`; `result.plot()` dispatches to it automatically when a ladder is present.

### Changed

- **Participation-ratio metrics renamed: `pr_eig` / `pr_singular`** (repo-wide): the vague, inconsistently-named `participation_ratio` / `pr_covariance` / `participation_ratio_singular` are gone. `pr_eig` = `(Σσᵢ²)²/Σσᵢ⁴` (eigenvalue/covariance-spectrum variant), `pr_singular` = `(Σσᵢ)²/Σσᵢ²` (singular-spectrum variant). `dimensionality` mode's lean/default spectral output now reports **both** variants (previously only one, under the vague name) — a real behavior change, not just a rename. This is a breaking change with no deprecated aliases (library is pre-publication). Downstream notebooks/scripts outside this repo that reference the old names need updating.
- **`HybridCritic.forward` now row-chunks pair scoring** (`neural_mi/models/critics.py`), matching `ConcatCritic`'s existing pattern: the full `(N², 2d)` pair tensor is never materialized at once, bounding peak memory during large-N evaluation (e.g. `dimensionality` mode's noise-injection ladder, which multiplies eval cost by levels × splits).
- **`PretrainedBackboneEmbedding` gradient/BatchNorm fix** (`neural_mi/models/embeddings.py`): removed a stray `torch.no_grad()` around the frozen backbone's forward pass that was silently severing gradient to the trainable channel adapter whenever `input_dim != backbone_in_ch` (the adapter was frozen at random init and never trained). Backbone freezing is already handled via `requires_grad=False` and doesn't need `no_grad`. Also added a `train()` override so the frozen backbone's BatchNorm layers stay in eval mode regardless of the outer model's train/eval state.
- **`min_coverage_fraction` semantics documented, not changed** (`neural_mi/data/temporal.py`): coverage is a source-*timestamp* count, not a value-validity check (NaN-valued-but-present-timestamp windows are not dropped), and gap interpolation is not bounded by the coverage fraction. Docstring-only.
- **`AnalysisWorkflow.__init__` input_dim now uses the full flattened shape** (`neural_mi/analysis/rigorous.py`): `int(np.prod(shape[1:]))` instead of `shape[1]*shape[2]`, which silently dropped the width axis for 4-D (`cnn2d`) inputs. No-op for existing 3-D callers.
- **`estimators/bounds.py`, `logmeanexp_nodiag`**: `dim=0` was falsy and silently fell through `dim or (0,1)` to reduce over both axes instead of just dim 0. Fixed to `dim if dim is not None else (0,1)`. Never fired in practice (only `None` and `(0,1)` are passed anywhere in the codebase today) — pure future-proofing.
- **`analysis/transfer.py`, `_build_te_arrays`**: replaced a Python list-comprehension + `torch.stack` with `tensor.unfold()`, which produces the same layout as a view instead of materializing three large window-array copies. Verified bit-exact equivalent before applying.
- **`data/temporal.py`, `SpikeWindowDataset`**: the `max_spikes_per_window` truncation message (data is silently dropped) is now a `logger.warning`, not `logger.info`.
- **`SincEmbedding` readout changed from mean pooling to log-band-power pooling** (`neural_mi/models/embeddings.py`): `features.mean(dim=-1)` discarded most of the signal, since a bandpassed oscillation is ~zero-mean; replaced with `log(features.pow(2).mean(dim=-1) + eps)`. Found and fixed as part of gating `sinc_cnn` against a generic CNN on a new band-power-vs-broadband-interference regime (`results/gate/decision_log.md`); with the fix, `sinc_cnn` beats the best generic CNN at every N tested and learned filter cutoffs migrate toward the true signal band.

### Fixed

- **`analysis/conditional.py`**: X and Z with mismatched window sizes now raise a clear `ValueError` before the `torch.cat` into XZ, instead of a bare shape-mismatch error.
- **`data/temporal.py`, `CategoricalWindowDataset.__init__`**: integer-typed labels with negative values now raise a clear `ValueError` instead of silently reaching `np.bincount` via `n_categories = data.max() + 1`. Non-integer labels (floats/strings) are unaffected — still auto-relabeled to consecutive non-negative integers as before.
- **`run()` silently dropped `optimizer_params`/`estimator_params`/`scheduler_params` set only in `base_params`** (`neural_mi/run.py`): the top-level kwargs default to `None`, were converted to `{}` via `X or {}` before being passed to `_inject()`, and `_inject()` only skips overwriting on a literal `None` — so the `{}` unconditionally clobbered a caller-supplied `base_params['optimizer_params']` whenever the matching top-level kwarg wasn't also passed. This silently zeroed any `weight_decay` set via `base_params` alone, including via this library's own documented call pattern. Fixed by passing the raw value (no `or {}`) so `_inject`'s None-guard and `apply_defaults()`'s missing-key backstop behave the same as every other base_params key (e.g. `dropout`). Covered by a new regression test (`tests/test_validation.py`).
- **`embedding_model='gru'`/`'lstm'` unconditionally rejected pre-processed 3-D input** (`neural_mi/run.py`): a top-level validation check raised `ValueError` whenever `processor_type_x=None`, even when `x_data` was already a legitimate pre-windowed `(N, C, W)` tensor — a case the rest of the pipeline (`ParameterSweep`'s own `is_proc_sweep` auto-detection) already supports. Fixed to skip the check when the array already has a time dimension (`x_data.ndim == 3`).

### Removed

- **`CalciumEmbedding` / `embedding_model='calcium_cnn'`** (`neural_mi/models/embeddings.py`) and its generator `generate_windowed_calcium`: cut rather than fixed. `_deconv_kernel` built the time-reversed, unit-normalized indicator impulse response — a **matched filter**, which further low-passes the signal, not a deconvolution (which would sharpen/invert the blur). The docstring's "FIR deconvolution" claim did not match what the code did. Independently, the only generator for it carries its shared information in firing rate, for which mean fluorescence is already near-sufficient, so a correct deconvolution would not have bought anything there either. All registry entries, base-params schema keys (`tau_rise`, `tau_decay`, `learn_calcium_kernel`), tests, tutorial sections, and reference-doc rows removed with it.
- **`SpikePhysicsEmbedding` / `embedding_model='spike_physics'`** (`neural_mi/models/embeddings.py`): removed after an empirical gate against a regularized generic MLP on rate-code spike data (Regime C, `results/gate/decision_log.md`) came back NO_HEADROOM under the gate's strict criterion (10x converged-N ratio, but overlapping ±1 std bands at the discriminating N). All registry entries, the `'features'`/`'concat'` fusion code path, base-params schema keys, tests, tutorial sections (Tutorials 8, 10, 11), and reference-doc rows removed with it.
- **Depthwise-separable first layer / `use_depthwise` on `embedding_model='cnn'`** (`neural_mi/models/embeddings.py`): removed after an empirical gate on a favorable multichannel regime (per-channel distinct carriers) showed no advantage over a plain `CNN1D` — at N=10000 plain CNN (0.62 bits) actually exceeded depthwise (0.47 bits); see `results/gate/decision_log.md`. The `use_depthwise` flag, base-params schema key, tests, tutorial sections (Tutorials 10, 11), and reference-doc rows removed with it. `generate_windowed_multichannel` (its only consumer) is retained since it still feeds the gate's own evidence chain.

- **SVD-aligned rotated embeddings (`return_rotated_embeddings`)** (`neural_mi/utils.py`, `neural_mi/training/trainer.py`, `neural_mi/analysis/task.py`):
  A new `compute_cross_covariance_rotation()` utility and four new `base_params` keys enable returning embeddings re-projected so that dimension 0 captures the most shared variance between the two modalities, dimension 1 the next most, and so on — consistent with the Participation Ratio ordering. This makes the first *k* dimensions directly interpretable without separately inspecting the SVD.

  New parameters:
  - `return_rotated_embeddings` (bool, default `False`) — enable the feature. Works alongside `return_embeddings` and/or `track_embeddings`; has no effect for `concat` critics (emits a `UserWarning`).
  - `rotated_embeddings_whitening` (str or None, default `'std'`) — whitening applied to the cross-covariance before SVD to derive the rotation axes. Does **not** affect the scale of the returned embeddings (which are always `ZX_centered @ U`/`ZY_centered @ V` in original embedding space). Matches the default used by `compute_cross_covariance_spectrum` for consistency with PR estimates.
  - `rotated_embeddings_per_epoch` (bool, default `False`) — when `track_embeddings` is also enabled: `False` (default) derives one global rotation from the best epoch's embeddings and applies it to all tracked epochs (consistent coordinate system across epochs); `True` computes a fresh SVD per tracked epoch (shows how latent structure emerges).
  - `return_rotation_matrices` (bool, default `False`) — include U and V in `result.details` so new data can be projected into the same aligned basis.

  New `result.details` keys:
  - `embeddings_x_rotated`, `embeddings_y_rotated`, `embeddings_rotation_singular_values` (+ optional `embeddings_rotation_x/y`) — from `return_embeddings` path.
  - `embedding_history_x_rotated`, `embedding_history_y_rotated`, `embedding_rotation_singular_values` (+ optional rotation matrices) — from `track_embeddings` path.

- **Physics parameter tracking (`sinc_cnn`)** (`neural_mi/models/embeddings.py`, `neural_mi/training/trainer.py`):
  Added `get_physics_params()` method to `SincEmbedding`. The trainer now
  calls this method after every evaluation epoch and stores results in `result.details`:
  - `result.details['physics_params_history']` — dict of per-epoch parameter lists (keys prefixed by
    `x_` or `y_`, e.g. `x_f_low_hz`, `x_f_high_hz`).
  - `result.details['physics_params_final']` — same keys with values from the best epoch.
  Both keys are absent for non-learnable embeddings (e.g. standard CNN).

- **Pretrained backbone spatial dimension mismatch handling** (`neural_mi/models/embeddings.py`):
  `PretrainedBackboneEmbedding` now automatically inserts a bilinear `nn.Upsample` layer when input
  images are not 224×224 (standard ImageNet resolution). The upsample is created lazily on the first
  forward pass and emits a `UserWarning` with the input and expected sizes.

- **Two new windowed generators with analytically known MI** (`neural_mi/generators/synthetic.py`):
  - `generate_windowed_oscillatory(n_windows, n_channels, window_size, f_carrier_hz, sample_rate, latent_mi, snr)` —
    windowed oscillatory LFP with shared latent carrier; MI computed from the linear-Gaussian
    `ρ_obs = ρ_latent × SNR² / (SNR² + 1)` formula.
  - `generate_windowed_multichannel(n_windows, n_channels, window_size, f_min_hz, f_max_hz, sample_rate, latent_mi, snr)` —
    same model with per-channel carrier frequencies uniformly spaced in `[f_min_hz, f_max_hz]`; total
    MI = sum over channels.
  Both return `(X, Y, true_mi)` and are exported from `neural_mi.generators`.

- **Tutorial 11 — Inductive Biases: Quantitative Validation** (`tutorials/raw tutorials/11_Inductive_Biases_Quantitative.py`,
  `tutorials/11_Inductive_Biases_Quantitative.ipynb`):
  Sample-efficiency curves (MI vs. N windows) comparing biased models to standard CNN baselines,
  with analytically known ground-truth MI. Covers: SincCNN (10 Hz LFP with filter convergence
  diagnostic) and pretrained backbone (MNIST vs Gaussian blobs).

- **`generate_timing_code_spike_trains` generator** (`neural_mi/generators/synthetic.py`):
  new function for generating a precise-timing spike code embedded in high-rate
  independent background Poisson noise.  Each neuron pair shares signal spikes
  (`signal_rate` Hz) that Y fires with a fixed `delay` + Gaussian `jitter`;
  both populations are additionally driven by `background_rate` Hz background.
  With `background_rate >> signal_rate`, summary statistics of the spike counts
  are dominated by noise, so GRU's ability to process actual spike timestamps
  gives it a detectable advantage.  Exported from `neural_mi.generators`.

- **`torchvision` optional dependency** (`setup.py`, `pyproject.toml`):
  added `vision` extras group (`pip install neural_mi[vision]`) for
  `PretrainedBackboneEmbedding`.  Was previously an undeclared dependency.

### Removed

- **`generate_oscillatory_lfp`** (`neural_mi/generators/synthetic.py`): replaced by
  `generate_windowed_oscillatory`, which returns IID pre-windowed arrays and an analytically
  computed true MI value.

### Fixed

- **`ContinuousWindowDataset` / `CategoricalWindowDataset` time-vector units**
  (`neural_mi/data/temporal.py`): when `sample_rate` is given but no
  `time_vector`, both datasets now construct a seconds-based time vector
  (`np.arange(N) / sample_rate`) instead of an integer-index vector.  With
  integer indices, a `window_size` in seconds (e.g., 0.5 s) was less than one
  sample, producing zero valid windows.

- **GRU/LSTM validation false-positive in `ParameterSweep`**
  (`neural_mi/analysis/sweep.py`): the check that errors when
  `embedding_model='gru'` and `processor_type_x=None` no longer fires when
  data has already been pre-processed upstream (detected via
  `processor_params_x['preprocessed'] == True`).

- **`'cnn2d'` missing from `ALLOWED_VALUES`** (`neural_mi/validation.py`):
  `embedding_model='cnn2d'` raised a validation error; added to the allowed
  list.

### Changed

- **Tutorial 10 — SincEmbedding section** (`tutorials/raw tutorials/10.py`):
  increased LFP signal SNR from 2.0 to 3.0 and added `LFP_PARAMS` (250
  epochs, patience 50) for both CNN and `sinc_cnn` comparisons.  `FAST_PARAMS`
  (60 epochs) was insufficient for sinc filters to converge; updated commentary
  explains this.

- **Online data augmentations (Batch 3)**: per-batch augmentations applied
  during training only (never at eval time).  Three new `base_params` keys:
  - `augmentation_params` — shared augmentation spec for both X and Y.
  - `augmentation_params_x` — per-variable override for X (`None` = use shared,
    `{}` = explicitly disable).
  - `augmentation_params_y` — per-variable override for Y (same semantics).
  Available augmentation keys (`neural_mi/augmentations.py`):
  - *Spatial (4-D input only)*: `random_flip_h`, `random_flip_v`,
    `random_rotation_90`, `random_crop`, `random_erase`, `time_mask`,
    `freq_mask`, `gaussian_blur`.
  - *Non-spatial (any ndim)*: `gaussian_noise`, `intensity_scale`,
    `channel_dropout`.
  - *Custom*: `custom` — a single callable or list of callables, each
    accepting an `(N, ...)` tensor and returning a tensor of the same shape.
  Application order is always: spatial → non-spatial → custom.  Spatial
  augmentations requested on non-4-D input emit a `UserWarning` and are
  skipped gracefully.

- **Plotting — `estimate` mode: `conservative_epoch` marker**: when
  `peak_fraction < 1.0` is used, `result.details` contains
  `'conservative_epoch'` (the epoch whose train MI is reported as the final
  estimate).  `Results.plot()` now draws a green dotted vertical line and a
  diamond scatter marker at that epoch alongside the existing red best-epoch
  marker.  Without `peak_fraction`, the plot is unchanged.

- **Plotting — `dimensionality` mode: dedicated two-panel figure**: a new
  `plot_dimensionality_curve()` function (also exported from
  `neural_mi.visualize`) replaces the previous `plot_sweep_curve` call for
  dimensionality results.  When participation-ratio data is available (always
  the case from `run()`), the figure has two stacked panels: MI on top and
  Participation Ratio (effective dimensionality) on the bottom — both sharing
  the same sweep x-axis when a sweep grid was used.  Without a sweep, scalar
  results are shown as annotated error-bar points.

- **Plotting — `conditional` mode**: `Results.plot()` now renders a vertical
  bar chart showing the three CMI components: `I(XZ;Y)`, `I(Z;Y)`, and
  `CMI I(X;Y|Z)`.  Bars are labelled with numeric values.  Previously raised
  `NotImplementedError`.

- **Plotting — `transfer` mode**: `Results.plot()` now renders a bar chart
  showing `TE(X→Y)` and (when available) `TE(Y→X)`.  The plot title includes
  the directionality index and a plain-English direction label when present.
  Previously raised `NotImplementedError`.

- **Plotting — `Results.compare()` for `estimate` mode**: overlay of test-MI
  training curves across multiple runs.  Each curve is drawn in a distinct
  colour with best-epoch markers as faint dashed vertical lines.  Previously
  raised `NotImplementedError`.

- **`plot_bias_correction_fit` now returns `ax`**: the function previously
  returned `None`; it now returns the `matplotlib.axes.Axes` used for the
  plot, enabling composability.

- **`plot_cross_correlation` composability**: added `ax`, `show`, and `xlim`
  parameters; function now returns the axes.  The previously hard-coded
  `xlim=(-100, 100)` is gone — the full lag range is shown by default.

- **`analyze_mi_heatmap` composability**: added `ax` and `show` parameters;
  function now returns the axes.  All `print()` statements replaced with
  `logger.info()` / `logger.warning()` calls so the function is silent in
  library use.

- **`_RESULT_COLS` extended**: `pr_eig`, `pr_eig_mean`, `pr_eig_std`,
  `pr_singular`, `pr_singular_mean`, `pr_singular_std`, and `split_id`
  added to the frozenset.  These were previously missing, causing the
  dimensionality sweep-variable inference to consider them as candidate
  x-axis columns and fail silently.

- **Rigorous plot `is_reliable=False` annotation**: when `is_reliable` is
  `False` in `result.details`, `Results.plot()` adds a red text box to the
  bias-correction figure so unreliable extrapolations are immediately visible
  without checking `result.summary()`.

- **`use_amp` parameter** (`'auto'` / `True` / `False`): mixed-precision (AMP)
  training via `torch.cuda.amp.autocast` + `GradScaler`. Enabled automatically
  on CUDA devices (`'auto'`); a no-op on CPU and MPS so all existing workflows
  are unaffected. Added to `BASE_PARAMS_SCHEMA` in `defaults.py`; wired through
  `run()`, `task.py`, and `Trainer`. On CUDA the forward pass runs in float16,
  reducing memory by ~2× and improving throughput on Ampere+ GPUs.

- **`Results.to_dict()`**: returns a fully JSON-serialisable `dict` with keys
  `mode`, `mi_estimate`, `params`, `details`, and `dataframe`. All numpy arrays
  are converted to nested Python lists; DataFrames are exported in `records`
  orientation; non-serialisable objects fall back to a `"<TypeName>"` string.

- **`Results.to_json()` now uses `to_dict()`**: arrays are serialised as nested
  lists (previously as `"<array shape=... dtype=...>"` strings), making the JSON
  output both human-readable and round-trippable. Existing call signatures and
  the auto-naming / no-overwrite behaviour are unchanged.

- **Named variable support in `run()`**: four new optional top-level arguments —
  `x_name` (str), `y_name` (str), `channel_names_x` (list of str),
  `channel_names_y` (list of str). Stored in `result.params` for use in plot
  axis labels. In pairwise mode, `channel_names_x/y` are injected into
  `result.details['variable_names_x/y']`, which drives the MI-matrix heatmap
  tick labels. Fallback when omitted is the current integer-index behaviour.

- **`return_embeddings` — full dataset, original order**: `result.details['embeddings_x']`
  and `result.details['embeddings_y']` now contain embeddings for **all** windows in
  original sample order, with no subsampling or shuffling. Previously the extraction
  block reused `max_eval_samples` (default 5000) and drew a random subset via
  `np.random.choice`, making the returned arrays impossible to align with
  time-indexed behavioural signals. Inference is now performed in mini-batches of
  512 (internal constant `_EMBEDDING_BATCH`) to avoid OOM on large datasets.
  `max_eval_samples` continues to control only the epoch-level evaluation MI estimate
  and has no effect on embedding extraction. Applies to all modes: `estimate`,
  `sweep`, and `dimensionality`.

- **`extract_embeddings()` — full dataset, original order**: same fix applied to the
  standalone function in `embeddings_io.py`. The `max_samples` parameter has been
  removed entirely; inference uses mini-batched ordered iteration over the full
  input. Code that previously passed `max_samples=N` will receive a `TypeError` and
  should be updated (pass the desired subset of the data directly).

- **Dimensionality mode — embedding arrays no longer corrupt the results DataFrame**:
  `run_dimensionality_analysis()` now strips `embeddings_x`/`embeddings_y` from the
  per-split result dicts before constructing the `pd.DataFrame`. Previously, if
  `return_embeddings=True` was set, 2-D numpy arrays would end up as object-dtype
  columns in the aggregated DataFrame, breaking groupby aggregation. The embeddings
  are now returned as a second value `(df, embeddings_dict_or_None)` from
  `run_dimensionality_analysis()`; `run()` unpacks this and places the arrays in
  `result.details['embeddings_x/y']`. With `n_splits > 1`, embeddings come from the
  last split's model (logged explicitly).

- **`show` parameter for plot utilities**: `plot_sweep_curve`, `plot_dimensionality_curve`,
  and `plot_bias_correction_fit` now accept `show: bool = True`. When `False`,
  `plt.show()` is suppressed, enabling these functions to be embedded in multi-panel
  figures or called in Jupyter notebooks without blocking execution.

- **Dimensionality mode — `split_method='index'`**: new channel-split option for
  `run_dimensionality_analysis()` / `run(..., mode='dimensionality')`.  Pass
  `channel_indices_x=[0, 1, 4, 5, 7]` as a keyword argument; Y is automatically the
  complement set.  Works for both 2-D `(N, C)` and 3-D `(N, C, W)` data.  When X and
  Y have different channel counts, `shared_encoder=True` is automatically disabled with
  a logger warning; this can be suppressed by explicitly setting `shared_encoder=False`
  in `base_params`.  Multiple `n_splits` independent model initialisations are still
  performed (same fixed channel assignment, different weight initialisation) so the
  output DataFrame retains the same mean/std structure as other split methods.

- **`track_embeddings` parameter**: controls per-epoch embedding extraction during
  training.  Accepted in `base_params` for all analysis modes; in `dimensionality` mode
  the default is `512` (track the first 512 samples each epoch); in all other modes the
  default is `False` (disabled).  Accepted values mirror the existing `eval_train`
  syntax: `False` (off), `True` (first 512 samples), a positive `int` (exact sample
  count), a `float` in `(0, 1)` (fraction of dataset), or `'full'` (entire dataset,
  emits a `UserWarning` about memory cost).  Embeddings are always extracted from the
  first N samples in original order (deterministic, aligns with user-provided labels).
  Per-epoch arrays are stored in `result.details['embedding_history_x']` and
  `result.details['embedding_history_y']` (each a list of `(n_tracked, embed_dim)`
  numpy arrays, one per epoch).

- **`animate_training()` and `result.animate()`**: new animation utility in
  `neural_mi.visualize.animate` that creates frame-by-frame GIF / MP4 animations from
  training history stored in `result.details`.  Panels are auto-detected from available
  data or specified explicitly via `panels=['mi', 'spectral_metrics', 'spectrum', 'embeddings']`.
  The `'mi'` panel always shows test MI vs epoch; train MI is overlaid when present.
  The `'spectral_metrics'` panel plots participation ratio vs epoch.  The `'spectrum'`
  panel shows an animated bar chart of singular values (requires `spectral_mode='full'`).
  The `'embeddings'` panel renders a 2-D or 3-D scatter of learned embeddings at each
  epoch, with PCA or UMAP reduction fitted jointly on all frames for consistent
  coordinates.  Pass `embedding_labels` as a 1-D array or a `dict` of name → array for
  categorical (tab10 palette) or continuous (viridis) point colouring; each dict entry
  produces its own subplot column.  Output is saved as a GIF via `PillowWriter` or MP4
  via `FFMpegWriter` when `output_path` is supplied.  `result.animate(**kwargs)` is a
  thin wrapper around `animate_training(result, **kwargs)`.

- **`umap-learn >= 0.5.0` added as a hard dependency**: required for UMAP dimensionality
  reduction in `animate_training()`.  Added to both `setup.py` and `pyproject.toml`.

- **`CNN2D` encoder — `embedding_model='cnn2d'`**: new 2-D convolutional encoder for
  image-like input of shape `(N, C, H, W)`. Architecture: stacked `Conv2d` blocks (same
  padding) → `AdaptiveAvgPool2d(1)` → `Flatten` → two `Linear` layers. The adaptive
  pooling head collapses any spatial size to a fixed `(1, 1)` representation so no
  `input_shape` parameter is needed — only `n_channels` is used, exactly as for `CNN1D`.
  Reuses the existing `kernel_size` base parameter (must be odd, default 3).  Exported
  from `neural_mi.models` and selectable via `embedding_model='cnn2d'` in any analysis
  mode.

- **4-D input support in `task.py`**: `run_training_task()` now handles 4-D tensors
  `(N, C, H, W)` when computing `input_dim_x/y` (previously assumed `(N, C, W)`).
  Behaviour by model type:
  - `'cnn2d'` — handled natively; no warning.
  - `'mlp'` — flattened to `C×H×W` features silently; no warning.
  - `'cnn'` (CNN1D) — raises `ValueError` (spatial axes are ambiguous for a 1-D kernel).
  - all other sequence/graph models (`'gru'`, `'lstm'`, `'tcn'`, `'transformer'`) —
    emit a `UserWarning` noting that spatial dimensions are not preserved.

- **Dimensionality mode — spatial split methods for 4-D data**: six new/updated
  `split_method` values for `run_dimensionality_analysis()` / `run(..., mode='dimensionality')`:
  - `'horizontal'` — top half vs. bottom half (height axis).
  - `'vertical'` — left half vs. right half (width axis).
  - `'row_interleaved'` — even-indexed rows → X, odd-indexed rows → Y. Fine-grained
    horizontal stripes; avoids contiguous spatial bias.
  - `'col_interleaved'` — even-indexed columns → X, odd-indexed columns → Y.
    Column-wise counterpart to `'row_interleaved'`.
  - `'diagonal'` — true geometric split: upper-left triangle + main diagonal → X
    (`row ≤ col`), lower-right triangle → Y. Works with `'mlp'` and sequence models;
    raises `ValueError` for `'cnn2d'` / `'cnn'`. Rectangular input (H ≠ W) is allowed
    with a warning; `shared_encoder` is always disabled (diagonal pixels go to X).
  - `'antidiagonal'` — true geometric split: upper-right triangle + anti-diagonal → X
    (`row + col ≤ W−1`), lower-left triangle → Y. Same constraints as `'diagonal'`.
  All six require 4-D input `(N, C, H, W)`. Unequal flat sizes disable `shared_encoder`
  automatically.

  > **Note:** what was previously called `'diagonal'` (interleaved rows) has been renamed
  > `'row_interleaved'`, and `'antidiagonal'` (interleaved columns) has been renamed
  > `'col_interleaved'`.  The names `'diagonal'` and `'antidiagonal'` now refer to the
  > true geometric triangular splits.

- **Dimensionality mode — `split_method='index'` extended to 4-D**: the existing index
  split now handles 4-D tensors `(N, C, H, W)` in addition to 2-D `(N, C)` and 3-D
  `(N, C, W)` data.

#### Physics-Informed (Inductive Bias) Embedding Models

- **`SincEmbedding` — `embedding_model='sinc_cnn'`**: learnable FIR bandpass filters
  for EEG / LFP data.  Filter cutoffs are parameterised as `log(f_low)` / `log(f_high)`
  and initialised to the five canonical neural frequency bands (δ, θ, α, β, γ) plus
  high-γ bands.  A Hamming window is applied to each filter to reduce spectral leakage.
  `n_sinc_filters` (default 8) controls the number of filters per channel.  Requires
  `sample_rate` (injected automatically from `processor_params_x/y`).  Supports
  `feature_fusion='features'` (default) or `'concat'`.

- **`PretrainedBackboneEmbedding` — `embedding_model='pretrained_backbone'`**: frozen
  torchvision backbone (e.g. ResNet18, EfficientNet-B0) used as a fixed feature
  extractor, followed by a trainable MLP head mapping to `embedding_dim`.  Set
  `pytorch_predefined` to the torchvision model name and `pretrained=True` to load
  ImageNet weights.  Expects 4-D input `(N, C, H, W)`.

- **Modality metadata plumbing** (`task.py`): `run_training_task()` now injects
  `sample_rate_x`, `sample_rate_y`, and `no_spike_value` into
  the params dict from `processor_params_x/y` before calling `build_critic()`.  These
  values are consumed by the inductive-bias constructors and do not need to be set
  manually by the user.

- **New synthetic data generators** (`neural_mi/generators/synthetic.py`):
  `generate_oscillatory_lfp`, `generate_modulated_spike_trains`,
  `generate_noisy_image_pairs`.  All three are exported from `neural_mi.generators`.

- **Tutorial 10 — Inductive Biases**: new tutorial (`tutorials/raw tutorials/10.py`)
  covering the physics-informed models with worked examples and the
  `feature_fusion` parameter.

### Fixed

- **`test_critic_chunking_equivalency[Separable]` flaky failure**: the test used
  unseeded `x_data`/`y_data` fixtures; in the full suite their values depended on
  cumulative RNG state, occasionally producing bilinear critic scores of magnitude
  10⁵+ where float32 differences between chunked and non-chunked forward passes
  exceeded the `atol=1e-4` tolerance.  Fix: move `torch.manual_seed(42)` to the
  very first line of the test body and construct `x_data`/`y_data` locally,
  making the test fully deterministic regardless of execution order.

- **`test_paired_time_shift_positive` flaky failure**: after undoing a time shift
  (`+d` then `−d`) the spike-time float64 round-trip can shift the reconstructed
  window range by ε, yielding ±1 window vs. the original and an index-offset in
  the window tensor (so `after_undo[i] ≈ original[i−1]`).  Fix: replace the
  fragile `torch.allclose` data comparison with a check that (1) the continuous
  time vector is approximately restored (`np.allclose(..., atol=1e-6)`) and (2)
  the window count is within ±1 of the original.

- **`_BUILD_PARAMS_KEYS` consolidation** (`task.py`): the module-level constant was missing
  all six decoder keys (`use_decoder`, `decoder_weight`, `decoder_weight_x`,
  `decoder_weight_y`, `decoder_output_activation_x`, `decoder_output_activation_y`). A
  redundant local redefinition inside `run_training_task()` held the complete list. The
  local redefinition has been removed; the module-level constant is now the single
  authoritative source used by both `run_training_task()` and `extract_embeddings()`.

- **`spectral_output` docstring** (`trainer.py`): the docstring incorrectly stated `'full'`
  as the value that returns all spectral metrics. The actual code checks for `== 'all'`;
  the docstring has been corrected to match.

- **Rigorous mode `is_reliable` false-positive for large datasets (R² gate)**:
  R² of the WLS linear fit was previously a condition for `fit_quality_warning`
  (and therefore `is_reliable=False`). With large N, the finite-sampling bias
  across gamma values is inherently small, producing a near-flat MI vs. gamma
  curve where R² collapses toward zero even when the fit and extrapolation are
  sound (observed: R²=0.10 at N=10 000, R²=0.00 at N=1 000 on well-behaved
  data). R² is now computed and stored in `result.details['r_squared']` for
  transparency but no longer affects `fit_quality_warning`. The `r2_threshold`
  parameter is retained in all public APIs for backward compatibility but is a
  no-op.

- **Rigorous mode `is_reliable` false-positive for large datasets (residual
  gate)**: `fit_quality_warning` (max externally studentized residual >
  threshold) was also a condition for `is_reliable=False`. The heteroscedastic
  WLS structure of rigorous mode — where low-gamma rows dominate the MSE while
  high-gamma training runs have natural noise — routinely produces large
  studentized residuals even for perfectly valid fits (observed: residuals of
  4.63, 9.03, and 3.22 at N=1 000/5 000/10 000 with threshold 2.5).
  `fit_quality_warning` is now **informational only** and does not affect
  `is_reliable`. `is_reliable` is now governed solely by (1) sufficient gamma
  points and (2) `leverage_warning` (LOO γ=1 intercept-stability check), which
  is scale-invariant and directly tests whether the extrapolation anchor is
  stable.

### Changed

- **`NEURALMI_REFERENCE.md`** updated to document recently added parameters: `peak_fraction`
  added to Training table; `track_spectral_metrics` and `return_spectrum` added to
  Spectral/Whitening table; new Decoder section covering `use_decoder`, `decoder_weight`,
  `decoder_weight_x/y`, and `decoder_output_activation_x/y`; `use_amp` added to Memory &
  Device table; `plot_dimensionality_curve` and `plot_bias_correction_fit` signatures in
  the Low-Level Utilities section updated to reflect the new `show` parameter; new Online
  Data Augmentations section covering all 11 built-in keys, the three `base_params`
  augmentation keys, application order, and usage examples; §4 extended with 4-D input
  note for CNN2D and spatial augmentations; §5 `run()` signature annotated with
  augmentation params; §10 CNN2D input shape corrected to 4-D; Quick Reference Card
  updated; §12 Design Decisions augmented with augmentation note.

### Tests Added

- `tests/test_augmentations.py`: 32 tests covering all 11 built-in augmentation
  types (shape preservation for 3-D and 4-D inputs, spatial-on-3D `UserWarning`,
  Gaussian noise / intensity scale / channel dropout semantics, time mask / freq
  mask / random erase / Gaussian blur correctness, custom callable and list of
  callables, `custom` invalid-type error, application order, and `True` shortcut
  defaults).

- `tests/test_amp_and_names.py`: 14 tests covering `use_amp` (`'auto'`, `True`,
  `False` on CPU; schema presence; sweep-mode passthrough) and named variables
  (`x_name`/`y_name` stored in params; pairwise `channel_names_x/y` injected
  into details; clean params when names are omitted; signature reflection).
- `tests/test_results_extended.py`: 9 new tests in `TestToDict` covering
  `to_dict()` return type, required keys, 1-D and 2-D numpy arrays as nested
  lists, training-history inclusion, DataFrame as records, `None` DataFrame, and
  `to_json()` round-trip fidelity.
- `tests/test_rigorous_diagnostics.py`: added `test_low_r2_does_not_trigger_fit_quality_warning`
  — regression test confirming that a near-flat MI curve (large N, tiny bias)
  with low R² does not set `fit_quality_warning=True`. Also added
  `test_gamma1_outlier_sets_is_reliable_false` — confirms that `leverage_warning`
  (not `fit_quality_warning`) is the gate for `is_reliable`.
- `tests/test_dimensionality.py`: 17 new tests covering `_extract_embedding_history`,
  `_strip_embeddings`, `split_method='index'` input validation (missing/empty/out-of-range
  `channel_indices_x`, all-channels-to-X, unknown split_method), the `shared_encoder`
  auto-disable guard (unequal vs equal channel counts), correct 2-D and 3-D channel
  slicing, `n_splits` task-count, and the `track_embeddings=512` auto-default.
- `tests/test_animate.py`: 27 new tests covering `_auto_panels`, `_fit_reducer` (no-op when
  `embed_dim ≤ n_components`; `reduction='none'`; PCA; empty input; unknown reduction),
  `_resolve_scatter_color` (None, float, categorical int, categorical str), and
  `animate_training()` smoke tests for MI-only, auto-panels, spectral, spectrum, train MI
  overlay, missing `test_mi_history` error, empty panel list error, single-array and dict
  embedding labels, missing embedding history warning, 3-D embeddings, `reduction='none'`,
  and the `result.animate()` delegate.

---

## [Unreleased]

### Added

#### Generic Variational Wrapper (`use_variational=True` for all encoders)
- **Removed `VarMLP`** — the purpose-built variational MLP is gone entirely.
  All `use_variational=True` runs now use `VariationalWrapper` instead.
- **New `VariationalWrapper` class** in `neural_mi/models/embeddings.py`:
  wraps *any* base encoder (MLP, CNN1D, GRU, LSTM, TCN, Transformer, or a
  custom module) with μ and log σ² projection heads plus the reparameterization
  trick.  At training time returns `(z_sampled, kl_loss / batch_size)`; at eval
  time returns `(μ, 0.0)` — identical protocol to the former `VarMLP`.
- **`build_critic()` updated**: when `use_variational=True`, `build_critic`
  builds the selected base encoder normally and then wraps it with
  `VariationalWrapper(base_encoder, embed_dim)`.  This applies to all six
  `embedding_model` choices: `'mlp'`, `'cnn'`, `'gru'`, `'lstm'`, `'tcn'`,
  `'transformer'`.
- **`shared_encoder` remains fully compatible** with variational mode: the
  shared encoder is built once and wrapped once; both `net_x` and `net_y`
  point to the same `VariationalWrapper` instance.
- `neural_mi/models/__init__.py`: exports `VariationalWrapper`; no longer
  exports `VarMLP`.

### Changed

#### Generic Variational Wrapper
- `neural_mi/utils.py`: `build_critic()` no longer has a special `VarMLP`
  branch for variational mode.  The model-selection tree is now strictly
  by `embedding_model` name; variational wrapping is a post-construction step.
- `neural_mi/models/decoders.py`: removed `'var_mlp'` from the name aliases in
  `build_decoder()` — it was never needed in practice.
- `neural_mi/run.py`: updated `use_spectral_norm`, `dropout`, and `norm_layer`
  docstrings to reference "MLP" rather than "MLP/VarMLP".

### Tests Added

#### Generic Variational Wrapper
- `tests/test_models.py` fully updated:
  - `test_varmlp_embedding` → `test_variational_wrapper_embedding`
  - `test_varmlp_kl_loss` → `test_variational_wrapper_kl_loss`
  - New `test_variational_wrapper_eval_returns_mu` — checks determinism in eval mode.
  - New `test_variational_wrapper_gradients_flow` — verifies gradients reach
    both the mu/log_var heads and the base encoder.
  - New class `TestVariationalWrapperAllEncoders` — 6 parametrized tests
    (one per encoder type) each checking output shape, positive KL in training
    mode, and zero KL in eval mode.
  - Critic tests updated: `test_separable_critic_with_varmlp` →
    `test_separable_critic_with_variational_wrapper`,
    `test_concat_critic_with_varmlp` →
    `test_concat_critic_with_variational_wrapper`.
  - `test_critic_chunking_equivalency` now builds `VariationalWrapper(MLP(…), embed_dim)`.
  - `critic_and_data` fixture: `"SeparableVarMLP"` renamed to `"SeparableVariational"`.

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
  data with no clear γ trend), `run_rigorous_scalar_analysis` falls back to
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
  (`None`/`'layer'`/`'batch'`, default `None`) parameters for MLP
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
