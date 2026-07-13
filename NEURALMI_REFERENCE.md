# NeuralMI — Comprehensive Library Reference

> **Purpose of this document:** A self-contained technical reference for the NeuralMI library.
> It covers concepts, architecture, every public API, all parameters, return types, and
> worked examples.

---

## Table of Contents

1. [Library Overview & Philosophy](#1-library-overview--philosophy)
2. [Installation & Quick Start](#2-installation--quick-start)
3. [Key Concepts](#3-key-concepts)
   - 3.1 Mutual Information & Neural Estimators
   - 3.2 Estimators (InfoNCE, SMILE)
   - 3.3 Embedding Models
   - 3.4 Critic Architectures
   - 3.5 Bias in Finite-Sample Estimation
4. [Data Formats & Processors](#4-data-formats--processors)
5. [The `run()` Function — Complete Reference](#5-the-run-function--complete-reference)
6. [Analysis Modes](#6-analysis-modes)
   - 6.1 `estimate` — Single MI Estimate
   - 6.2 `sweep` — Hyperparameter Sweep
   - 6.3 `dimensionality` — Latent Dimensionality
   - 6.4 `rigorous` — Bias-Corrected Estimate
   - 6.5 `lag` — Temporal Lag Analysis
   - 6.6 `precision` — Spike-Timing Precision
   - 6.7 `conditional` — Conditional MI
   - 6.8 `transfer` — Transfer Entropy
   - 6.9 `pairwise` — Channel-to-Channel MI Matrix
7. [The `Results` Object](#7-the-results-object)
8. [Base Parameters Reference](#8-base-parameters-reference)
9. [Data Generators](#9-data-generators)
10. [Model Architecture Reference](#10-model-architecture-reference)
11. [Exceptions](#11-exceptions)
12. [Design Decisions & Internals](#12-design-decisions--internals)

**Appendices**
- [A — Enhanced Rigorous Mode Diagnostics](#enhanced-rigorous-mode-diagnostics)
- [B — Optional Decoder (Deep Symmetric IB)](#optional-decoder-deep-symmetric-ib)
- [C — Rigorous Bias Correction for Conditional and Transfer Modes](#rigorous-bias-correction-for-conditional-and-transfer-modes)

---

## 1. Library Overview & Philosophy

**NeuralMI** is a Python library for rigorous, fast **mutual information (MI) estimation** from neural and time-series data. It wraps neural network–based MI estimators (e.g. InfoNCE) into a unified, scientist-facing API with:

- **One entry point**: `neural_mi.run()` handles all analysis modes.
- **Automated bias correction**: the `rigorous` mode extrapolates MI to the infinite-data limit.
- **Multiple data modalities**: continuous time series (LFP, EEG), spike trains, categorical signals.
- **Hyperparameter exploration**: sweep any parameter combination in parallel.
- **Temporal analyses**: lag, transfer entropy, spike-timing precision.
- **Spatial analyses**: pairwise channel MI matrix, latent dimensionality.

**Core dependency stack:** PyTorch ≥ 2.0, NumPy ≥ 1.23, Pandas ≥ 1.4, scikit-learn ≥ 1.0, statsmodels ≥ 0.13, Matplotlib ≥ 3.5, Seaborn ≥ 0.12.

---

## 2. Installation & Quick Start

```python
import neural_mi as nmi
import numpy as np

# Simplest usage: estimate MI between two continuous signals
x = np.random.randn(1000, 4)   # 1000 time points, 4 channels
y = 0.7 * x + 0.3 * np.random.randn(1000, 4)  # correlated copy

result = nmi.run(x, y, mode='estimate')

print(result.mi_estimate)   # MI in bits
```

---

## 3. Key Concepts

### 3.1 Mutual Information & Neural Estimators

Mutual information between two random variables X and Y is:

```
I(X; Y) = E[ log( p(x,y) / (p(x) p(y)) ) ]
```

It measures how much information X carries about Y (and vice versa), in units of **bits** (log base 2) or **nats** (natural log). Unlike correlation, MI captures nonlinear dependencies.

Neural MI estimators train a **critic network** `f(x, y)` that approximates the log density ratio. Given a batch of `N` paired samples `(xᵢ, yᵢ)` and `N²` unpaired combinations, the critic learns to distinguish "real" pairs from "shuffled" ones. The MI is estimated from the resulting critic scores.

### 3.2 Estimators

NeuralMI supports different estimators. All take a score matrix `S ∈ ℝ^{N×N}` where `S[i,j] = f(xᵢ, yⱼ)`.

| Estimator | Key Idea | Ceiling | Variance | Best for |
|-----------|----------|---------|----------|----------|
| **InfoNCE** | Noise-contrastive estimation | log(N) nats | Low | Default; MI < ~7 bits |
| **SMILE** | JS + clipped DV correction | None | Medium | High MI signals |

**InfoNCE** (default):
```
I_InfoNCE = log(N) + mean_i[ S[i,i] − logsumexp_j(S[i,j]) ]
```
The ceiling of `log(N)` nats means with batch_size=128 you can estimate up to ~4.6 nats (~6.6 bits). To go higher, increase `batch_size` or switch to `smile`.

**SMILE** adds a clipping correction to reduce variance. The `clip` parameter (default 5.0) controls the tradeoff: lower values → lower variance but more bias.

**Practical guidance:**
- Start with `infonce` (default). If `mi_estimate` is near the ceiling (`log(batch_size)` nats), increase `batch_size` or switch to `smile`.

### 3.3 Embedding Models

Before computing critic scores, each input passes through an **embedding model** that maps the raw input (shape `[batch, channels, window]`) to a fixed-size embedding vector. Available architectures:

| Model | `embedding_model` value | Notes |
|-------|------------------------|-------|
| Multi-layer Perceptron | `'mlp'` (default) | Flattens input; good default; handles 4-D silently |
| 1D Convolutional | `'cnn'` | Uses `kernel_size` param; 3-D input `(N,C,W)` only |
| **2D Convolutional** | **`'cnn2d'`** | **For image-like `(N,C,H,W)` input; uses `AdaptiveAvgPool2d`** |
| Gated Recurrent Unit | `'gru'` | For sequences; `bidirectional` option |
| Long Short-Term Memory | `'lstm'` | For sequences; `bidirectional` option |
| Temporal Convolutional Net | `'tcn'` | Dilated 1D conv; good for long windows |
| Transformer | `'transformer'` | Self-attention; needs `nhead` param |
| Pretrained Backbone | `'pretrained_backbone'` | Frozen torchvision backbone + trainable MLP head; for image data (`(N,C,H,W)`) |

All embeddings output a vector of size `embedding_dim` (default 64).

(A depthwise-separable `'cnn'` variant, a `'spike_physics'` embedding for raw spike timestamps, and a `'sinc_cnn'` bandpass-filter embedding for EEG/LFP were evaluated empirically against generic encoders, did not outperform them, and have been removed.)

### 3.4 Critic Architectures

The critic `f(x, y)` combines the two embeddings into a score. Three architectures:

| Critic | `critic_type` value | Notes |
|--------|--------------------|-|
| **Separable** | `'separable'` (default) | `f(x,y) = gₓ(embed(x))ᵀ g_y(embed(y))` — bilinear product of separate head networks |
| **Concat** | `'concat'` | Concatenates raw inputs before any embedding; ignores `embedding_dim` |
| **Hybrid** | `'hybrid'` | Embeds X and Y independently, concatenates the embeddings, then passes them through a small MLP decision head; used automatically by `dimensionality` mode |

**Choosing:** `separable` is the best general choice. `concat` is the most flexible but doesn't allow for embedding dimensionality and is very costly to train. `hybrid` is required for dimensionality analysis (set automatically by that mode) and can also be used when you want the geometric flexibility of a learned scoring function on top of the embeddings.

The decision head of the hybrid critic can be sized independently of the embedding networks via `hidden_dim_head` and `n_layers_head` (see parameter table below).

### 3.5 Bias in Finite-Sample Estimation

Neural MI estimators are **biased upward** at small sample sizes — the critic can memorize rather than generalize. The bias scales roughly as `O(1/N)`. The `rigorous` mode exploits this:

1. Train models on subsets of size `N/γ` for γ = 1, 2, …, 10.
2. Plot estimated MI vs `γ`. (Since `N_chunk = N/γ`, the bias `a/N_chunk = (a/N)γ` is linear in γ.)
3. Fit a line to the linear portion: `MI(γ) = MI_true + slope × γ`.
4. Extrapolate to `γ → 0` (infinite data): `MI_true ≈ intercept`.

This gives a **bias-corrected estimate** with a confidence interval from the fit variance.

---

## 4. Data Formats & Processors

### Raw Input Shapes

NeuralMI accepts three raw data types via `processor_type_x/y`:

| Data type | `processor_type` | Expected shape | Notes |
|-----------|-----------------|---------------|-------|
| Continuous (LFP, EEG, Ca²⁺) | `'continuous'` | `(n_channels, n_timepoints)` | Sliding windows extracted |
| Spike trains | `'spike'` | `List[np.ndarray]` of 1D spike time arrays | One array per neuron |
| Categorical states | `'categorical'` | `(n_channels, n_timepoints)` integer | One-hot or ordinal encoded |
| Pre-processed | `None` (default) | `(n_samples, n_channels)` or `(n_samples, n_channels, window)` | Passed directly |

**Processor parameters** (`processor_params_x/y` dict):

For `'continuous'` and `'categorical'`:
```python
{
    'window_size': 0.05,          # seconds; sliding window length
    'sample_rate': 1000,          # Hz; required for temporal processors
    'min_coverage_fraction': 0.8, # minimum fraction of window that must be valid
}
```

For `'spike'`:
```python
{
    'window_size': 0.05,          # seconds; binning window
    'sample_rate': 1000,          # Hz
    'n_seconds': 100.0,           # total recording duration
    'bin_size': 0.001,            # seconds; spike bin width
    'normalize_bins': True,       # normalize spike counts
    'no_spike_value': -1.0,        # value for empty bins
    'max_spikes_per_window': None,
    'exclude_bursty_neurons': False,
    'burst_threshold_multiplier': 5.0,
}
```

**4-D pre-processed input `(N, C, H, W)`** is also supported without any processor (`processor_type=None`). Pass the tensor directly. Use this when your data is already image-like (spectrogram, 2-D spike raster, etc.) and you want to use `embedding_model='cnn2d'` or any of the spatial augmentations (`random_flip_h`, `random_crop`, `time_mask`, etc.). All spatial augmentations silently skip 3-D input with a `UserWarning` and are only applied to 4-D batches.

### Post-Processing Shape Convention

After any processor, all data tensors are 3D: `(n_samples, n_channels, window_size)`. For pre-processed 2D data, a trailing dim-1 is added automatically: `(n_samples, n_channels, 1)`. This is the **internal tensor format** throughout the library. Pre-processed 4-D data `(N, C, H, W)` is passed through unchanged.

---

## 5. The `run()` Function — Complete Reference

All parameters are grouped into typed config objects (see `neural_mi.config`).
Every config is optional; omitted configs and unset fields fall back to the
library defaults. Anywhere a config is accepted, a plain `dict` with the same
keys works too, so importing the classes is optional.

```python
import neural_mi as nmi
from neural_mi import (Model, Training, Split, Estimator, Output, Processing,
                       Rigorous, Precision, Lag, Transfer, Dimensionality, Conditional)

result = nmi.run(
    x_data, y_data=None,             # data; y_data required by most modes (see §4 for shapes)
    mode='estimate',                 # 'estimate'|'sweep'|'dimensionality'|'rigorous'|'lag'|'precision'|'conditional'|'transfer'|'pairwise'

    processing=None,                 # Processing(...) — raw-data processors
    model=None,                      # Model(...)      — architecture
    training=None,                   # Training(...)   — optimization loop
    split=None,                      # Split(...)      — train/test splitting
    estimator=None,                  # Estimator(...) or a name string ('infonce'|'smile')
    output=None,                     # Output(...)     — units, embeddings, labels
    sweep_grid=None,                 # dict[str, list] for 'sweep' mode

    # mode-specific config (only the one matching `mode` is used):
    rigorous=None, precision=None, lag=None,
    transfer=None, dimensionality=None, conditional=None,

    # runtime:
    n_workers=1,                     # parallel workers
    seed=None,                       # int; use with n_workers=1 for full reproducibility
    device=None,                     # 'cpu'|'cuda'|'mps'|None (auto)
    verbose=False, show_progress=True,
    permutation_test=False, n_permutations=1,
)
```

### Config objects

**`Processing`** — raw-data processors (omit for pre-processed input):
`x`, `x_params`, `y`, `y_params`, `x_time`, `y_time`.
Example: `Processing(x='continuous', x_params={'window_size': 0.05})`.

**`Model`** — architecture:
`embedding_model` (`'mlp'|'cnn'|'cnn2d'|'gru'|'lstm'|'tcn'|'transformer'|'pretrained_backbone'`),
`embedding_dim`, `hidden_dim`, `n_layers`, `critic_type` (`'separable'|'concat'|'hybrid'`),
`kernel_size`, `bidirectional`, `nhead`, `dropout`, `norm_layer` (`'layer'|'batch'`),
`use_spectral_norm`, `shared_encoder`, `custom_critic`, `custom_embedding_cls`,
`use_variational`, `beta`, `use_decoder`, `decoder_weight`, `pytorch_predefined`, `pretrained`.

**`Training`** — optimization loop:
`n_epochs`, `learning_rate`, `batch_size`, `patience`,
`optimizer` (`'adam'|'adamw'|'sgd'|'rmsprop'|'adagrad'` or a subclass), `optimizer_params`,
`scheduler` (`'cosine'|'cosine_warmup'|'step'|'plateau'` or a class), `scheduler_params`,
`gradient_clip_val`, `use_amp`, `eval_train`, `peak_fraction`, `max_eval_samples`,
`train_subset_size`, `save_best_model_path`,
`augmentation_params` (+ `augmentation_params_x`/`_y`), `dataset_device` (`'cpu'|'auto'`).

**`Split`** — train/test splitting:
`mode` (`'blocked'|'random'`), `train_fraction`, `n_test_blocks`, `gap_fraction`,
`train_indices`, `test_indices`.

**`Estimator`** — MI estimator: `name` (`'infonce'|'smile'`), `params` (e.g. `{'clip': 5.0}`).
`estimator='smile'` is shorthand for `Estimator(name='smile')`.

**`Output`** — result formatting: `units` (`'bits'|'nats'`), `spectral_mode`,
`return_embeddings`, `x_name`, `y_name`, `channel_names_x`, `channel_names_y`.

### Mode-specific configs

- **`Rigorous`** — `gamma_range`, `delta_threshold`, `min_gamma_points`, `confidence_level`.
- **`Precision`** — `tau_grid` (required), `corrupt_target` (`'x'|'y'|'both'`), `corruption_method` (`'rounding'|'noise'`), `n_noise_samples`, `threshold_ratio`.
- **`Lag`** — `lag_range` (required), `equalize_n`.
- **`Transfer`** — `history_window` (required), `prediction_horizon`, `bidirectional`; set `rigorous=True` for bias-corrected TE.
- **`Dimensionality`** — `split_method`, `n_splits`, `channel_indices_x`, `sigma_add`.
- **`Conditional`** — `z_data` (required), `z_processor_type`, `z_processor_params`; set `rigorous=True` for bias-corrected CMI.

### Minimal Examples by Mode

```python
from neural_mi import Lag, Transfer, Conditional, Dimensionality

# estimate
result = nmi.run(x, y, mode='estimate')

# sweep (scan over embedding_dim)
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'embedding_dim': [32, 64, 128, 256]})

# rigorous
result = nmi.run(x, y, mode='rigorous')

# lag
result = nmi.run(x, y, mode='lag', lag=Lag(lag_range=range(-20, 21)))

# conditional
result = nmi.run(x, y, mode='conditional', conditional=Conditional(z_data=z))

# transfer entropy (x -> y)
result = nmi.run(x, y, mode='transfer', transfer=Transfer(history_window=10))

# pairwise (all channel pairs in x)
result = nmi.run(x, mode='pairwise')

# pairwise cross (x channels vs y channels)
result = nmi.run(x, y, mode='pairwise')

# dimensionality
result = nmi.run(x, mode='dimensionality', dimensionality=Dimensionality(n_splits=10))
```

---

## 6. Analysis Modes

### 6.1 `estimate` — Single MI Estimate

**What it does:** Trains one MI estimator (or an average over multiple parallel runs via `n_workers`) and returns a single MI value.

**Key kwargs:** `n_workers=1`

**Returns:** `Results` with:
- `result.mi_estimate` — float, MI in `output_units`
- `result.details` — dict with `test_mi`, `train_mi`, `best_epoch`, `loss_history`, `raw_train_mi` (final training MI before smoothing), `train_mi_history` (per-epoch list, present when `eval_train` is set)
- `result.dataframe` — None

```python
result = nmi.run(x, y, mode='estimate',
                 training=Training(n_epochs=100, batch_size=256),
                 estimator='smile',
                 n_workers=4)          # Runs 4 independent fits, returns mean
print(result.mi_estimate)             # e.g. 1.34 bits
```

---

### 6.2 `sweep` — Hyperparameter Sweep

**What it does:** Trains a model for every combination in `sweep_grid` (Cartesian product) and returns MI as a function of those parameters. Essential for finding the right architecture before running `rigorous`.

**Key kwargs:** `n_workers=1`, `max_samples_per_task=None`

**Returns:** `Results` with:
- `result.dataframe` — DataFrame with columns: [sweep_var(s), `mi_mean`, `mi_std`, `run_id`]
- `result.mi_estimate` — None
- `result.details['raw_results']` — full per-run DataFrame

```python
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={
                     'embedding_dim': [32, 64, 128],
                     'n_epochs': [50, 100],
                 },
                 n_workers=4)

result.plot()       # Line plot of MI vs sweep variable (auto-detected)
df = result.dataframe
best = df.loc[df['mi_mean'].idxmax()]
```

**Note on `sweep_grid`:** Keys must be `Model` / `Training` field names (see §8). Processor parameters like `window_size` can also be swept.

---

### 6.3 `dimensionality` — Latent Dimensionality

**What it does:** Estimates the **intrinsic dimensionality** of neural representations by probing how MI scales with embedding size, or by splitting channels into halves and measuring their shared information.

Two sub-modes controlled by whether `y_data` is provided:

| Sub-mode | `y_data` | What's measured |
|----------|----------|----------------|
| **Intrinsic** | None | MI between two halves of x channels |
| **Interaction** | Provided | Direct MI between x and y |

**Split methods** (`split_method` kwarg — intrinsic mode only):
- `'random'` — Random channel splits, repeated `n_splits` times (default)
- `'spatial'` — Single split at channel midpoint
- `'temporal'` — Correlates x with lag-shifted copy of itself (pass `lag=<int>`)
- `'index'` — User-specified channel assignment; pass `channel_indices_x=[0, 1, 4]`.
  Y is automatically the complement set.  Supports 2-D `(N,C)`, 3-D `(N,C,W)`, and
  4-D `(N,C,H,W)` input.  When X and Y have different channel counts,
  `shared_encoder=True` is disabled with a warning.  Multiple `n_splits` independent
  weight initialisations are still run so the output retains a proper mean/std.
- `'horizontal'` — **(4-D only)** top half vs. bottom half (splits height axis).
- `'vertical'` — **(4-D only)** left half vs. right half (splits width axis).
- `'row_interleaved'` — **(4-D only)** even-indexed rows → X, odd-indexed rows → Y. Fine-grained horizontal stripes; avoids contiguous spatial bias.
- `'col_interleaved'` — **(4-D only)** even-indexed columns → X, odd-indexed columns → Y. Column-wise counterpart; together with `'row_interleaved'` they probe spatial isotropy.
- `'diagonal'` — **(4-D only; MLP/sequence models only)** True geometric triangular split: upper-left triangle + main diagonal → X (`row ≤ col`), lower-right triangle → Y. Rectangular input (H ≠ W) is allowed with a warning. Raises `ValueError` for `embedding_model='cnn2d'` or `'cnn'`.
- `'antidiagonal'` — **(4-D only; MLP/sequence models only)** Upper-right triangle + anti-diagonal → X (`row + col ≤ W−1`), lower-left triangle → Y. Same constraints as `'diagonal'`.

All 6 spatial split methods require `(N, C, H, W)` input and raise `ValueError` for lower-dimensional data.
When the two halves have unequal flat sizes, `shared_encoder=True` is disabled with a warning. Geometric diagonal/antidiagonal splits always produce unequal halves (the diagonal pixels go to X), so `shared_encoder` is always disabled for `embedding_model='mlp'`.

**`n_splits` kwarg (default 5):**
- *Intrinsic mode* (`split_method='random'`): number of distinct random channel-split assignments evaluated
- *Interaction mode* (y_data provided): number of independent model fits from different random weight initialisations — gives a proper mean and std in the aggregated output

**Key kwargs:** `n_workers=1`, `split_method='random'`, `n_splits=5`, `lag=<int>` (for temporal), `channel_indices_x=<list>` (for index split)

**Returns:** `Results` with:
- `result.dataframe` — columns: `mi_mean`, `mi_std`, `pr_eig_mean`, `pr_eig_std`, `pr_singular_mean`, `pr_singular_std` (aggregated over splits/runs); `result.details['raw_results']` contains per-run rows with `split_id`
- `pr_eig` — effective dimensionality from eigenvalue (covariance) spectrum: `(Σσᵢ²)² / Σσᵢ⁴` — stricter, weights large singular values more
- `pr_singular` — PR from singular-value spectrum: `(Σσᵢ)² / Σσᵢ²` — less strict variant
- `result.details['embeddings_x']`, `result.details['embeddings_y']` — present only when `return_embeddings=True`; numpy arrays from the **last split's model**, in original sample order, index-aligned with the input data.
- `result.details['embeddings_x_rotated']`, `result.details['embeddings_y_rotated']` — present when `return_embeddings=True` and `return_rotated_embeddings=True`; same shape as the raw embeddings but re-projected so that dimension 0 captures the most shared variance, dimension 1 the next most, etc. See `return_rotated_embeddings` in the parameter table.
- `result.details['embeddings_rotation_singular_values']` — singular values of the (whitened) cross-covariance used to compute the rotation; shape `(min(d_x, d_y),)`.
- `result.details['embeddings_rotation_x']`, `result.details['embeddings_rotation_y']` — rotation matrices U and V (present when `return_rotation_matrices=True`); apply as `new_data_zx @ U` to project new data into the same basis.
- `result.details['embedding_history_x']`, `result.details['embedding_history_y']` — present when `track_embeddings != False`; a list of `(n_tracked, embed_dim)` numpy arrays, one per training epoch, from the last split's model. Defaults to tracking the first 512 samples each epoch in dimensionality mode (set `track_embeddings=False` to disable).
- `result.details['embedding_history_x_rotated']`, `result.details['embedding_history_y_rotated']` — present when `track_embeddings != False` and `return_rotated_embeddings=True`; same list structure as `embedding_history_x/y` but in the SVD-aligned basis. In global mode (default) all epochs share the same rotation (derived from the best epoch); in per-epoch mode each epoch has its own rotation.
- `result.details['embedding_rotation_singular_values']` — singular values used for the rotation; a single array in global mode, a list of arrays in per-epoch mode.
- `result.details['embedding_rotation_x']`, `result.details['embedding_rotation_y']` — rotation matrices (global mode, when `return_rotation_matrices=True`).
- `result.details['embedding_rotation_history_x']`, `result.details['embedding_rotation_history_y']` — per-epoch rotation matrices (per-epoch mode, when `return_rotation_matrices=True`).

**Ceiling-escape noise injection (`sigma_add` kwarg):**

When the true MI exceeds the InfoNCE ceiling (`log(eval_size)`, where `eval_size = min(len(test_idx), max_eval_samples)`), the spectral readout becomes unreliable. `sigma_add` adds fixed, independent, per-channel Gaussian noise (in measured-per-channel-std units) to the observations once — before the embedding, identical for train and eval of a fit — to lower the MI below the ceiling while leaving the true dimensionality unchanged.

- `sigma_add=None` *(default)*: no noise; non-binned-spike modalities behave exactly as without this feature.
- `sigma_add=<float>`: inject that single noise level.
- `sigma_add=[<float>, ...]`: run the full ladder, one result row per level.
- `sigma_add='auto'`: search a geometric ladder (~0.25x–5x per-channel std) for the regime where MI has detached from the ceiling; widens the search once if the initial grid doesn't bracket it, then warns if it still doesn't.
- `sigma_add_units`: `'relative'` *(default)* — a multiple of measured per-channel std; `'absolute'` — the noise std in native units.
- `stabilize_counts` (bool, default `True`): for binned-spike data only, applies the Anscombe transform before measuring std / injecting noise. Fires on every binned-spike dimensionality run regardless of `sigma_add` (default-on toggle; set `False` for plain, un-stabilized counts — a warning is emitted if noise is also injected in that case).
- Supported for intrinsic `split_method in ('random', 'spatial')` or interaction mode only. Raw spike-timestamp and categorical data raise a clear `ValueError` when `sigma_add` is set (bin the spikes first for the former).

**Output** (in addition to the standard `result.dataframe` / `result.details['raw_results']`, both of which gain a `sigma_add` grouping column):
- `result.details['sigma_add_ladder']` — one row per rung: `sigma_add`, `log_sigma_add`, `sigma_add_absolute_x_mean`, `sigma_add_absolute_y_mean`, `mi_mean`/`mi_std`, `pr_eig_mean`/`_std`, `pr_singular_mean`/`_std`, `ceiling_nats` (= `log(eval_size)`), `regime` (`'pinned'`/`'detached'`/`'collapsed'`), `detached` (bool).
- `result.details['sigma_add_suggestion']` — `{'sigma_add': <float>, 'regime': 'detached'}`, only when `sigma_add='auto'` found a detached band; a suggestion, never a silent override of the reported estimates.
- `result.plot()` automatically dispatches to `plot_noise_ladder` (both PR variants vs. `log(sigma_add)`, detached band shaded) when a noise ladder is present.
- Permutation-test p-values are not yet computed per rung for the ladder (omitted, not fabricated).

```python
# Intrinsic: MI between two random halves of x channels, 10 splits
result = nmi.run(x, mode='dimensionality',
                 training=Training(n_epochs=100),
                 dimensionality=Dimensionality(n_splits=10, split_method='random'),
                 n_workers=4)
result.plot()

# Intrinsic: user-specified channel assignment (e.g. two electrode shanks)
result = nmi.run(x, mode='dimensionality',
                 training=Training(n_epochs=100),
                 dimensionality=Dimensionality(n_splits=5, split_method='index',
                                               channel_indices_x=[0, 1, 2, 8, 9, 10]))

# Interaction: MI between x and y, 5 independent fits for mean/std
result = nmi.run(x, y, mode='dimensionality',
                 training=Training(n_epochs=100),
                 dimensionality=Dimensionality(n_splits=5),
                 n_workers=4)
print(result.dataframe[['mi_mean', 'mi_std', 'pr_eig_mean', 'pr_singular_mean']])

# Animate the training history (embeddings tracked by default)
result.animate(output_path='training.gif', fps=10)
```

---

### 6.4 `rigorous` — Bias-Corrected Estimate

**What it does:** Implements the bias extrapolation procedure (§3.5). Trains models on `N/γ` data subsets for each γ in `gamma_range` (default 1–10), fits MI vs γ, and extrapolates to γ→0 (infinite data).

**Key parameters:**
- `gamma_range`: range or list of denominators (default `range(1, 11)`)
- `delta_threshold=0.1`: quadratic curvature threshold — gamma points whose estimated quadratic coefficient exceeds this value are excluded before the linear regression (see THEORY.md §5)
- `min_gamma_points=5`: minimum points that must survive the curvature filter for a reliable fit
- `confidence_level=0.68`: width of the confidence interval (0.68 ≈ 1σ, 0.95 ≈ 2σ)

**Key kwargs:** `n_workers=1`

**Returns:** `Results` with:
- `result.mi_estimate` — bias-corrected float
- `result.details`:
  - `mi_corrected` — same as `mi_estimate`
  - `mi_error` — half-width of CI
  - `slope` — linear fit slope (indicates bias severity)
  - `is_reliable` — bool; False if fewer than `min_gamma_points` in linear region
  - `gammas_used` — list of γ values included in the fit
- `result.dataframe` — all gamma × sweep combinations

```python
result = nmi.run(x, y, mode='rigorous',
                 rigorous=Rigorous(gamma_range=range(1, 15), confidence_level=0.95),
                 n_workers=4)
print(f"MI = {result.mi_estimate:.3f} ± {result.details['mi_error']:.3f} bits")
result.plot()   # MI vs gamma with fit line and extrapolation point
```

**Typical workflow:** Run `estimate` or `sweep` first to find good hyperparameters, then run `rigorous` with those parameters for the final publication-quality estimate.

---

### 6.5 `lag` — Temporal Lag Analysis

**What it does:** Computes MI at each temporal offset between x and y. Useful for finding the time delay of peak information transfer.

**Required parameter:** `lag_range` — range, list, or `np.ndarray` of lag values.
- For sample lags: integers (e.g., `range(-20, 21)`)
- For time lags: floats in seconds (e.g., `np.arange(-0.1, 0.11, 0.01)`); requires `sample_rate` in processor params

**Key kwargs:** `n_workers=1`, `equalize_n=False`

- `equalize_n=True` — truncate all lag windows to the minimum sample count (for fair comparison)

**Returns:** `Results` with:
- `result.dataframe` — columns: `lag`, `train_mi`, `test_mi`, `n_windows`, plus any sweep params
- `result.mi_estimate` — None

```python
result = nmi.run(x, y, mode='lag',
                 lag=Lag(lag_range=range(-30, 31)),    # ±30 sample lags
                 training=Training(n_epochs=50),
                 n_workers=8)
result.plot()   # MI vs lag; peak indicates best offset
peak_lag = result.dataframe.loc[result.dataframe['train_mi'].idxmax(), 'lag']
```

---

### 6.6 `precision` — Spike-Timing Precision

**What it does:** Measures how precisely spike timing encodes information by progressively corrupting spike times and tracking MI decay. The **precision timescale** τ* is the jitter level at which MI drops to `threshold_ratio × baseline_MI`.

**Required parameter:** `tau_grid` — list of corruption levels (seconds or arbitrary units)

**Corruption methods:**
- `'rounding'` — rounds spike times to nearest τ (default; clean and interpretable)
- `'noise'` — adds uniform noise drawn from U(−τ/2, τ/2)

**Key parameters:**
- `corrupt_target='x'` — which signal to corrupt: `'x'`, `'y'`, or `'both'`
- `threshold_ratio=0.9` — defines the precision cutoff. Can be a **single float** (default 0.9 = 90% of baseline) **or a list of floats** to compute multiple thresholds simultaneously (e.g. `[0.9, 0.75, 0.5]`)
- `n_noise_samples=50` — for `'noise'` method: repeated samples per τ

**Returns:** `Results` with:
- `result.dataframe` — columns: `tau`, `train_mi`, `train_mi_std`
- `result.details`:
  - `baseline_mi` — MI at τ=0 (uncorrupted)
  - `precision_tau` — τ* for the primary (first) threshold ratio
  - `threshold_value` — actual MI value at the primary threshold
  - `threshold_ratio` — the original input (scalar or list)
  - `precision_thresholds` — dict mapping each ratio to `{'precision_tau', 'threshold_value'}`
  - `corruption_method`, `corrupt_target`

```python
# Single threshold (default)
result = nmi.run(spike_x, y, mode='precision',
                 processing=Processing(x='spike',
                                       x_params={'window_size': 0.05, 'n_seconds': 100.0}),
                 precision=Precision(tau_grid=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                     threshold_ratio=0.9))
print(f"Precision timescale: {result.details['precision_tau']*1000:.1f} ms")

# Multiple thresholds simultaneously
result = nmi.run(spike_x, y, mode='precision',
                 precision=Precision(tau_grid=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                     threshold_ratio=[0.9, 0.75, 0.5]))
for ratio, v in result.details['precision_thresholds'].items():
    print(f"  {ratio*100:.0f}% threshold: tau* = {v['precision_tau']*1000:.1f} ms")
result.plot()
```

---

### 6.7 `conditional` — Conditional Mutual Information

**What it does:** Computes I(X; Y | Z), the information X and Y share beyond what Z explains. Uses the chain rule:

```
I(X; Y | Z) = I(XZ; Y) − I(Z; Y)
```

Both terms are estimated independently with their own model fits.

**Required:** `z_data` (the conditioning variable)

**Optional:** `z_processor_type`, `z_processor_params` — same options as for x/y

**Returns:** `Results` with:
- `result.mi_estimate` — float: I(X; Y | Z)
- `result.details`:
  - `cmi_estimate` — same as `mi_estimate`
  - `mi_xz_y` — I(XZ; Y)
  - `mi_z_y` — I(Z; Y)
  - `raw_xz_y`, `raw_z_y` — per-run results for each term

```python
result = nmi.run(x, y, mode='conditional',
                 conditional=Conditional(z_data=z),
                 training=Training(n_epochs=100),
                 n_workers=4)
print(f"I(X;Y|Z) = {result.mi_estimate:.3f} bits")
print(f"I(XZ;Y) = {result.details['mi_xz_y']:.3f}, I(Z;Y) = {result.details['mi_z_y']:.3f}")
```

---

### 6.8 `transfer` — Transfer Entropy

**What it does:** Computes transfer entropy from X to Y, TE(X→Y), using the chain rule:

```
TE(X→Y) = I(x_past, y_past ; y_future) − I(y_past ; y_future)
```

where `x_past`, `y_past` are the `history_window` most recent samples and `y_future` is `prediction_horizon` samples ahead.

**Required:** `history_window` (int, number of past samples)

**Key parameters:**
- `prediction_horizon=1` — samples ahead to predict
- `Transfer(bidirectional=...)` — default `False`; if `True`, also compute TE(Y→X) and return a directionality index. When `False`, a warning is logged recommending bidirectional evaluation to detect spurious causal claims.

**Returns:** `Results` with:
- `result.mi_estimate` — float: TE(X→Y) in `output_units`
- `result.details`:
  - `te_estimate` — same as `mi_estimate`, alias for `te_xy`
  - `te_xy` — TE(X→Y) point estimate
  - `i_xypast_yfuture` — I(x_past, y_past ; y_future)
  - `i_ypast_yfuture` — I(y_past ; y_future)
  - `raw_xypast_yfuture`, `raw_ypast_yfuture` — per-run lists
  - `n_samples` — number of valid sliding windows created
  - `bidirectional` — bool

  With `Transfer(bidirectional=True)`, additionally:
  - `te_yx` — TE(Y→X) point estimate
  - `i_yxpast_xfuture` — I(y_past, x_past ; x_future), the joint term for TE(Y→X)
  - `i_xpast_xfuture` — I(x_past ; x_future), the marginal term for TE(Y→X)
  - `raw_yxpast_xfuture`, `raw_xpast_xfuture` — per-run lists for the TE(Y→X) terms
  - `directionality_index` — `(TE_xy − TE_yx) / (|TE_xy| + |TE_yx|)`; +1 = pure X→Y, −1 = pure Y→X, 0 = symmetric

```python
# Unidirectional (default) — logs a warning to consider bidirectional
result = nmi.run(x, y, mode='transfer',
                 transfer=Transfer(history_window=20, prediction_horizon=1),
                 training=Training(n_epochs=100),
                 n_workers=4)
print(f"TE(X→Y) = {result.mi_estimate:.3f} bits")

# Bidirectional — recommended for causal inference
result = nmi.run(x, y, mode='transfer',
                 transfer=Transfer(history_window=20, bidirectional=True),
                 n_workers=4)
print(f"TE(X→Y) = {result.details['te_xy']:.3f} bits")
print(f"TE(Y→X) = {result.details['te_yx']:.3f} bits")
print(f"Directionality index = {result.details['directionality_index']:.3f}")
```

**Note:** `x_data` and `y_data` must be 2D here: `(T, n_channels)`, i.e., a raw temporal sequence. The library builds sliding windows internally.

---

### 6.9 `pairwise` — Channel-to-Channel MI Matrix

**What it does:** Computes MI between every pair of channels. Two modes:

| Mode | Condition | Pairs computed | Matrix shape |
|------|-----------|----------------|-------------|
| **Self-pairwise** | `y_data=None` | All (i,j) with i<j from x | `(n_ch_x, n_ch_x)` |
| **Cross-pairwise** | `y_data` provided | All (i,j) across x and y | `(n_ch_x, n_ch_y)` |

**Key kwargs:** `n_workers=1`, `pairs=None` (optional explicit list of `(i,j)` tuples)

**Returns:** `Results` with:
- `result.dataframe` — columns: `ch_x`, `ch_y`, `mi_mean`, `mi_std`
- `result.details['mi_matrix']` — 2D numpy array of per-pair means; upper triangle filled for self-pairwise
- `result.details['n_channels']` — int (self) or tuple (cross)

```python
# Self-pairwise: MI between all neuron pairs
result = nmi.run(x, mode='pairwise',
                 training=Training(n_epochs=50),
                 n_workers=8)
mi_matrix = result.details['mi_matrix']   # shape (n_channels, n_channels)

# Cross-pairwise: all (spike neuron) × (LFP channel) pairs
result = nmi.run(spike_x, lfp_y, mode='pairwise',
                 processing=Processing(x='spike', y='continuous'),
                 n_workers=8)
df = result.dataframe    # ch_x, ch_y, mi_mean, mi_std
```

---

## 7. The `Results` Object

```python
from neural_mi.results import Results  # also exported from neural_mi
```

### Fields

```python
@dataclass
class Results:
    mode: str                         # Which analysis mode produced this
    params: Dict[str, Any]            # All parameters used
    mi_estimate: Optional[float]      # Single-value modes only
    dataframe: Optional[pd.DataFrame] # Multi-row modes
    details: Dict[str, Any]           # Mode-specific metadata
```

### Methods

#### `result.plot(ax=None, **kwargs) → plt.Axes`

Generates a mode-appropriate figure.  Pass `show=False` to suppress
`plt.show()` (useful when embedding in multi-panel figures or Jupyter
notebooks).

| Mode | Plot type | Notes |
|------|-----------|-------|
| `estimate` | Test MI vs epoch | `best_epoch` marked in red; `conservative_epoch` (when `peak_fraction < 1`) marked in green with a diamond |
| `sweep` / `lag` | MI vs swept variable | Line + shaded ±1 std |
| `dimensionality` | Two-panel: MI (top) + Participation Ratio (bottom) | Creates its own multi-panel figure; pass `axes=(ax_mi, ax_pr)` to supply existing axes |
| `rigorous` | MI vs γ with WLS fit and extrapolation | Red warning box added when `is_reliable=False` |
| `conditional` | Bar chart: I(XZ;Y), I(Z;Y), CMI I(X;Y\|Z) | |
| `transfer` | Bar chart: TE(X→Y), TE(Y→X) | Title shows directionality index when present |
| `precision` | MI vs τ with baseline and threshold lines | |
| `pairwise` | MI matrix heatmap | |

```python
fig, ax = plt.subplots()
result.plot(ax=ax, title="My analysis", show=False)

# Dimensionality: supply both axes for a two-panel layout in your own figure
fig, (ax_mi, ax_pr) = plt.subplots(2, 1, figsize=(8, 8))
result.plot(ax=(ax_mi, ax_pr), show=False)
```

#### `result.summary() → None`

Prints a human-readable summary to stdout. Includes mode, MI estimate, confidence intervals where applicable, and mode-specific detail (component MI values for `conditional`/`transfer`, matrix range for `pairwise`, baseline MI and τ for `precision`).

#### `result.save(path=None) → str`

Serialises the Results object to a pickle file. When `path` is `None` or a directory, a timestamped filename (`neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl`) is generated automatically in the current working directory. Existing files are never overwritten — a numeric suffix is appended instead. Returns the absolute path of the saved file.

```python
filepath = result.save()            # auto-named in cwd
filepath = result.save('/data/')    # auto-named in /data/
filepath = result.save('/data/my_result.pkl')  # explicit path
```

#### `Results.load(path) → Results`

Classmethod. Loads a Results object previously saved with `save()`.

```python
result = Results.load('/data/my_result.pkl')
print(result.mi_estimate)
```

#### `result.to_json(path=None) → str`

Exports a human-readable JSON snapshot containing scalar fields (`mode`, `mi_estimate`, `params`) and the DataFrame. Large objects in `details` (numpy arrays, raw result lists) are summarised by type and shape rather than fully serialised. For complete round-trip fidelity, use `save()` / `load()`. Returns the absolute path of the created file.

```python
filepath = result.to_json()         # auto-named .json in cwd
```

#### `Results.compare(results_list, labels=None, ax=None, **kwargs) → plt.Axes`

Static method for overlaying multiple results on one shared axes.  All Results
objects must share the same `mode`.

| Mode | Overlay type |
|------|-------------|
| `estimate` | Test-MI training curves per run; best-epoch dashed vertical lines |
| `sweep` / `lag` / `dimensionality` | Sweep curves with distinct colours |
| `rigorous` | Bias-correction fits with distinct colours |

```python
# Compare two training runs
r1 = nmi.run(x, y, mode='estimate', training=Training(...))
r2 = nmi.run(x, y, mode='estimate', training=Training(..., learning_rate=5e-4))
Results.compare([r1, r2], labels=['LR=1e-4', 'LR=5e-4'])

# Compare two lag sweeps
r1 = nmi.run(x, y1, mode='lag', lag=Lag(lag_range=range(-20, 21)))
r2 = nmi.run(x, y2, mode='lag', lag=Lag(lag_range=range(-20, 21)))

Results.compare([r1, r2], labels=['Condition A', 'Condition B'])
```

### Low-level plotting utilities (`neural_mi.visualize`)

These functions are composable (accept an `ax` parameter, return the axes,
support `show=False`) and are also available as `nmi.visualize.<name>`.

#### `plot_dimensionality_curve(summary_df, sweep_var, units, axes, show, **kwargs) → plt.Axes`

Dedicated two-panel plot for dimensionality results.  Top panel: MI vs sweep
variable.  Bottom panel: Participation Ratio vs sweep variable.  When
`axes=None` (default) a new figure is created.  To embed in an existing
figure pass a 2-tuple `axes=(ax_mi, ax_pr)` or a single `ax` (PR panel
skipped).  `show=False` suppresses `plt.show()`.  Returns the MI axes.

```python
from neural_mi.visualize import plot_dimensionality_curve
ax_mi = plot_dimensionality_curve(result.dataframe, sweep_var='embedding_dim', show=False)
ax_pr = ax_mi.figure.axes[1]   # access the PR panel
```

#### `plot_bias_correction_fit(raw_df, corrected_result, ax, units, show) → plt.Axes`

Visualises the WLS extrapolation used in rigorous mode.  `show=False` suppresses
`plt.show()`.  Returns the axes so the caller can add annotations.

#### `plot_cross_correlation(x, y, true_lag, ax, show, xlim) → plt.Axes`

Cross-correlation vs lag between two signals.  `ax=None` creates a new figure;
`show=False` suppresses `plt.show()`; `xlim=(left, right)` clips the x-axis
(defaults to the full lag range).

```python
ax = nmi.visualize.plot_cross_correlation(x, y, true_lag=5, show=False, xlim=(-30, 30))
```

#### `analyze_mi_heatmap(results_df, ..., ax, show) → plt.Axes`

Topological analysis of a 2-D MI heatmap (lag × window_size).  Finds the
Causal Contour and Parsimonious Region.  All diagnostic output now goes to
`logger.info()` / `logger.warning()` — no `print()` side-effects.  Accepts
`ax` and `show` for composability.  Returns the axes (or `None` if no
significant contour is found).

#### `animate_training(result, panels, fps, output_path, show, n_components, reduction, embedding_labels, **kwargs) → FuncAnimation`

Creates a frame-by-frame animation of the training history stored in `result.details`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `panels` | auto-detected | List of `'mi'`, `'spectral_metrics'`, `'spectrum'`, `'embeddings'`. Auto-detected from `result.details` when `None`. |
| `fps` | 10 | Frames per second. |
| `output_path` | None | Path to save animation. `.gif` → PillowWriter; `.mp4` → FFMpegWriter. |
| `show` | True | Call `plt.show()` after building the animation. |
| `n_components` | 2 | Scatter dimensionality for embedding panels (2 or 3). |
| `reduction` | `'pca'` | Dimensionality reduction for embeddings: `'pca'`, `'umap'`, or `'none'`. |
| `embedding_labels` | None | 1-D array or dict of name → array for colouring scatter points. Float arrays use viridis; int/str arrays use a discrete tab10 palette. Each dict entry adds one subplot column. |

Requires `result.details['test_mi_history']` (always present after a training run).
The reducer is fitted once on all frames concatenated, giving consistent coordinates
across the animation.  `result.animate(**kwargs)` is a thin wrapper around this function.

```python
result = nmi.run(x, mode='dimensionality', training=Training(n_epochs=50))

# Basic GIF
result.animate(output_path='training.gif', fps=8)

# With embedding labels (per-trial stimulus category + continuous position)
result.animate(
    output_path='training.gif',
    embedding_labels={'stimulus': stim_labels, 'position': pos_values},
    reduction='umap',
)

# Jupyter notebook inline display
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
anim = result.animate(show=False)
HTML(anim.to_jshtml())
```

---

## 8. Config Fields Reference

The fields below are available on the config objects. Pass each via its matching
config — `Model(...)` for architecture, `Training(...)` for the optimization loop,
`Split(...)` for splitting, `Estimator(...)` / `Output(...)` for the estimator and
output. (A plain `dict` with the same keys is accepted anywhere a config is.)

### Training
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `n_epochs` | int | 50 | Also settable as top-level `run()` arg |
| `learning_rate` | float | 5e-4 | Adam optimizer LR |
| `batch_size` | int | 128 | Also settable as top-level `run()` arg |
| `patience` | int | 1000 | Early stopping patience (epochs without improvement) |
| `max_n_batches` | int | 512 | Max critic computation chunk (memory control) |
| `train_subset_size` | int or None | None | Use a random subset of training data |
| `eval_train` | bool/float/int | False | Per-epoch train MI tracking; `True`, fraction, or sample count |
| `peak_fraction` | float | 1.0 | Controls best-epoch selection. `1.0` uses the smoothed-MI peak epoch. `< 1.0` uses the first epoch where smoothed MI ≥ `peak_fraction × max_MI`, giving a more conservative estimate. When `< 1.0`, `result.details` also contains `'conservative_epoch'` and `'train_mi_at_peak'`. |

### Optimizer & Scheduler
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `optimizer` | str or class | `'adam'` | `'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'adagrad'`, or `torch.optim.Optimizer` subclass |
| `lr_head_multiplier` | float or None | `None` | Multiplier on `learning_rate` for the hybrid critic's decision head. `None` or `1.0` → same LR as the encoders. Values > 1 (e.g. `5.0`) make the head adapt faster relative to the encoders, which can reduce staircase convergence plateaus. Ignored for `separable` and `concat` critics. |
| `optimizer_params` | dict | `{}` | Extra kwargs for optimizer constructor (e.g. `{'weight_decay': 1e-4}`) |
| `scheduler` | str, class, or None | `None` | `'cosine'`, `'cosine_warmup'`, `'step'`, `'plateau'`, or `torch.optim.lr_scheduler` subclass |
| `scheduler_params` | dict | `{}` | Extra kwargs for scheduler constructor |

### Architecture
| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `embedding_dim` | int | 64 | Size of embedding vectors |
| `hidden_dim` | int or list of int | 64 | Hidden layer width. An integer gives uniform-width layers; a list (e.g. `[256, 1024, 256]`) sets per-layer widths explicitly — `n_layers` is ignored in this case. Supported for MLP, CNN1D, CNN2D, and TCN. |
| `n_layers` | int | 2 | Depth of embedding network. Ignored when `hidden_dim` is a list. |
| `embedding_model` | str | `'mlp'` | `'mlp'`, `'cnn'`, `'cnn2d'`, `'gru'`, `'lstm'`, `'tcn'`, `'transformer'`, `'pretrained_backbone'` |
| `critic_type` | str | `'separable'` | `'separable'`, `'concat'`, `'hybrid'` |
| `hidden_dim_head` | int, list of int, or None | `None` | Hidden width of the hybrid critic's decision head. Accepts the same int-or-list form as `hidden_dim`. `None` → `min(64, hidden_dim)` |
| `n_layers_head` | int or None | `None` | Depth of the hybrid critic's decision head. `None` → `max(1, n_layers - 1)` |
| `kernel_size` | int | 3 | For CNN, CNN2D, TCN |
| `bidirectional` | bool | False | For GRU, LSTM |
| `nhead` | int | 4 | For Transformer |
| `shared_encoder` | bool | False | Share embedding weights between x and y |
| `dropout` | float | 0.0 | Dropout after each hidden layer (MLP only) |
| `norm_layer` | str or None | `None` | `'layer'` (LayerNorm) or `'batch'` (BatchNorm1d); MLP only |

### Pretrained Backbone Parameters

These parameters apply only to `embedding_model='pretrained_backbone'`.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `pytorch_predefined` | str or None | `None` | torchvision model name, e.g. `'resnet18'`, `'efficientnet_b0'` |
| `pretrained` | bool | `False` | Load ImageNet pretrained weights |

**Physics parameter tracking (extensibility hook):**
If an embedding class implements a `get_physics_params()` method, the library records its return value after every evaluation epoch and stores it in `result.details`. No currently-shipped embedding implements this — it exists so a custom embedding (via `custom_embedding_cls`) can expose learnable physical parameters (e.g. filter cutoffs) for post-hoc inspection.

- `result.details['physics_params_history']` — dict of lists, one entry per parameter name, one value per training epoch. Keys are prefixed by variable (`x_` or `y_`). Absent when the embedding does not implement `get_physics_params()`.
- `result.details['physics_params_final']` — same keys as `physics_params_history` but a scalar value from the **best epoch** (the epoch used to compute `result.mi_estimate`). Present whenever `physics_params_history` is present.

**Spatial dimension mismatch (`pretrained_backbone`):**
`PretrainedBackboneEmbedding` probes the backbone at 224×224 during construction (matching standard ImageNet training resolution). If input images are smaller (e.g. 28×28 MNIST), the model automatically inserts a bilinear `nn.Upsample` layer on the first forward pass and emits a `UserWarning`:

```
UserWarning: PretrainedBackboneEmbedding: input spatial size (28×28) does not match
the expected size (224×224). Adding a bilinear upsample layer...
```

No user action is required — training proceeds normally. To suppress the warning, pre-resize images to 224×224 before passing them to `nmi.run()`.

### Splitting
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `split_mode` | str | `'blocked'` | `'blocked'` (temporal) or `'random'` (IID) |
| `train_fraction` | float | 0.9 | |
| `n_test_blocks` | int | 5 | Number of contiguous test windows |
| `split_gap_fraction` | float | 0.5 | Gap buffer around test blocks |

### Spectral / Whitening
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `spectral_whitening` | str or None | `'std'` | Whitening applied before spectral metrics: `'std'` (standardize per dimension) or `'zca'` (ZCA whitening), or `None` |
| `spectral_mode` | str | `'none'` | High-level spectral tracking switch: `'none'` (off), `'summary'` (both `pr_eig`/`pr_singular`), `'full'` (adds `effective_rank`/`spectral_entropy` + raw spectrum). Internally maps to `track_spectral_metrics`, `spectral_output`, and `return_spectrum`. |
| `track_spectral_metrics` | bool | False | Low-level switch: if `True`, computes spectral metrics at every epoch (can be expensive). Prefer `spectral_mode='summary'` for convenience. |
| `return_spectrum` | bool | False | If `True`, includes the raw cross-covariance singular-value spectrum in `result.details['spectral_metrics_history']`. Only meaningful when `track_spectral_metrics=True`. |

### Decoder (Reconstruction Regularisation)

Adding a decoder that reconstructs the input from the embedding adds a reconstruction loss term alongside the MI objective, acting as a regulariser that prevents embedding collapse. Useful when `embedding_dim` is large relative to the data dimensionality.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `use_decoder` | bool | False | Enable auxiliary decoder for X and Y. Builds a decoder matching the chosen `embedding_model` architecture. |
| `decoder_weight` | float | 1.0 | Shared weight for both decoder losses. Applied when `decoder_weight_x` / `decoder_weight_y` are not set. |
| `decoder_weight_x` | float or None | None | Per-channel weight for the X reconstruction loss. Overrides `decoder_weight` for X when set. |
| `decoder_weight_y` | float or None | None | Per-channel weight for the Y reconstruction loss. Overrides `decoder_weight` for Y when set. |
| `decoder_output_activation_x` | str | `'linear'` | Output activation of the X decoder: `'linear'` (MSE loss), `'sigmoid'` (MSE loss), `'softmax'` (cross-entropy loss). |
| `decoder_output_activation_y` | str | `'linear'` | Output activation of the Y decoder. Same options as above. |

When `use_decoder=True`, `result.details['decoder_recon_loss']` contains the weighted reconstruction loss evaluated at the best epoch.

### Online Data Augmentations

Augmentations are applied **per-batch during training only** — never at eval time.
Three `Training` fields control augmentation:

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `augmentation_params` | dict | `{}` | Shared augmentation spec applied to both X and Y. |
| `augmentation_params_x` | dict or None | `None` | Per-variable override for X. `None` = use `augmentation_params`; `{}` = explicitly disable augmentation for X. |
| `augmentation_params_y` | dict or None | `None` | Per-variable override for Y. Same semantics as `augmentation_params_x`. |

**Spatial augmentations** (require 4-D input `(N, C, H, W)`; skipped with a `UserWarning` for lower-dimensional input):

| Key | Config | Description |
|-----|--------|-------------|
| `random_flip_h` | `True` or `{'prob': float}` | Flip each sample along the height axis with probability `prob` (default 0.5). |
| `random_flip_v` | `True` or `{'prob': float}` | Flip along the width axis. |
| `random_rotation_90` | `True` | Rotate each sample independently by 0/90/180/270°. |
| `random_crop` | `{'padding': int}` | Pad by `padding` pixels (reflect mode) then random-crop back to original size. |
| `random_erase` | `{'prob': float, 'scale': (min, max)}` | Zero a random rectangle with probability `prob`; area fraction sampled from `scale`. |
| `time_mask` | `{'max_width': int}` | Zero a random contiguous column band of width up to `max_width`. |
| `freq_mask` | `{'max_height': int}` | Zero a random contiguous row band of height up to `max_height`. |
| `gaussian_blur` | `{'kernel_size': int, 'sigma': float}` | Depthwise 2-D Gaussian blur (even `kernel_size` auto-corrected to odd). |

**Non-spatial augmentations** (work on any input dimensionality):

| Key | Config | Description |
|-----|--------|-------------|
| `gaussian_noise` | `{'std': float}` | Add i.i.d. Gaussian noise with standard deviation `std` (default 0.1). |
| `intensity_scale` | `{'lo': float, 'hi': float}` | Multiply each sample by a random scalar drawn from `Uniform(lo, hi)` (defaults 0.8, 1.2). |
| `channel_dropout` | `{'p': float}` | Zero each channel independently with probability `p` (default 0.1). |

**Custom augmentations** (any input dimensionality):

| Key | Config | Description |
|-----|--------|-------------|
| `custom` | `callable` or `list[callable]` | Each callable receives an `(N, ...)` tensor and must return a tensor of the same shape. Applied after all built-in augmentations. |

Application order is always: **spatial → non-spatial → custom**.

**Example — shared Gaussian noise:**
```python
training = Training(
    augmentation_params={'gaussian_noise': {'std': 0.05}},
)
```

**Example — different augmentations per variable:**
```python
training = Training(
    augmentation_params_x={'gaussian_noise': {'std': 0.1}},
    augmentation_params_y={},  # no augmentation for Y
)
```

### Variational Training
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `use_variational` | bool | False | Enable variational reparameterization for *any* embedding model. When `True`, `build_critic` wraps the selected encoder with `VariationalWrapper`, adding μ and log σ² projection heads. Works with all `embedding_model` choices. |
| `beta` | float | 1024.0 | MI weight in variational loss `L = KL − β·MI`. Large β (≥ 1) makes MI maximization dominate; decrease for stronger KL regularization |

### Memory & Device Layout
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `device` | str or None | None | Compute device: `'cpu'`, `'cuda'`, `'mps'`, or `None` (auto-detect). See note below on MPS performance. |
| `dataset_device` | str or None | `'cpu'` | Where dataset tensors are stored. `'cpu'` (default) keeps data in pageable RAM so the OS can reclaim memory between sweep tasks. `'auto'` co-locates data with the compute device (precision mode default). Any explicit device string is also accepted. |
| `use_amp` | bool or `'auto'` | `'auto'` | Mixed-precision (AMP) training. `'auto'` enables AMP on CUDA and is a no-op on CPU/MPS. `True` enables explicitly (CUDA only; silently ignored on other devices). `False` disables entirely. AMP can significantly speed up training on modern NVIDIA GPUs at the cost of slightly reduced numerical precision. |

**Apple Silicon (MPS) device note.** `device=None` auto-selects `mps` on Apple Silicon Macs. MPS has a fixed kernel-dispatch cost (~0.05–0.2 ms per GPU operation) that dominates when the actual computation is small. For tiny models or small batches — as in quick synthetic sanity checks — CPU can be 4–7× *faster* than MPS. The crossover point where MPS starts winning is roughly: `hidden_dim ≥ 128`, or `batch_size ≥ 256`, or the total input feature count `n_channels × window_size ≥ 512`. For the windowed LFP/spike data NeuralMI is designed for, MPS is almost always the right choice. When benchmarking with small synthetic datasets, add `device='cpu'` explicitly to avoid misleading timing comparisons.

### Other
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `output_units` | str | `'bits'` | `'bits'` or `'nats'` |
| `random_seed` | int or None | None | RNG seed for reproducibility; combine with `n_workers=1` for fully deterministic runs |
| `verbose` | bool | False | |
| `show_progress` | bool | True | Show tqdm progress bar during training |
| `return_embeddings` | bool | False | If `True`, adds `embeddings_x` and `embeddings_y` (numpy arrays, shape `(N_windows, embedding_dim)`) to `result.details`. All windows are embedded in original sample order — no cap, no shuffling — so the arrays are index-aligned with the caller's windowed data and can be directly paired with behavioral labels or other time-indexed signals. |
| `track_embeddings` | bool / int / float / `'full'` | `False` (global); `512` in dimensionality mode | Per-epoch embedding tracking for `animate_training()`. `False` disables. `True` or `512` tracks the first 512 samples each epoch. A positive `int` specifies an exact count; a `float` in (0,1) is a fraction of the dataset; `'full'` tracks all samples (emits a `UserWarning`). Embeddings are always taken from the first N samples in original order. Stored in `result.details['embedding_history_x']` and `result.details['embedding_history_y']` (lists of `(n_tracked, embed_dim)` arrays, one per epoch). |
| `return_rotated_embeddings` | bool | `False` | If `True`, computes an SVD-based rotation of the embeddings so that dimension 0 captures the most shared variance between X and Y, dimension 1 the next most, and so on — consistent with the Participation Ratio ordering. Works alongside `return_embeddings` (produces `embeddings_x_rotated`, `embeddings_y_rotated`) and/or `track_embeddings` (produces `embedding_history_x_rotated`, `embedding_history_y_rotated`). Has no effect for `concat` critics. |
| `rotated_embeddings_whitening` | str or None | `'std'` | Whitening applied to the cross-covariance **before** computing the rotation axes. Does **not** affect the scale of the returned embeddings (which remain in the original embedding space, simply re-projected). `'std'` (default) matches the PR computation. `'zca'` applies full sphering (requires N >> d). `None` uses raw covariance. |
| `rotated_embeddings_per_epoch` | bool | `False` | Applies only when `track_embeddings` and `return_rotated_embeddings` are both enabled. `False` (default): compute one rotation from the best epoch's embeddings and apply it uniformly to all tracked epochs — gives a consistent coordinate system for cross-epoch comparison. `True`: compute a fresh SVD per epoch — shows how the latent structure emerges during training. |
| `return_rotation_matrices` | bool | `False` | If `True`, includes the rotation matrices U (`rotation_x`) and V (`rotation_y`) in `result.details`. These can be used to project new data into the same aligned basis: `new_zx @ U`. |
| `save_best_model_path` | str or None | None | |
| `max_eval_samples` | int | 5000 | Max samples used for **evaluation MI** during training (controls GPU memory for the test-set forward pass). Does **not** affect embedding extraction (`return_embeddings`). |
| `max_index_reduction` | float | 0.05 | Max allowed loss of MI index during eval |

---

## 9. Data Generators

`neural_mi.generators` (also accessible as `nmi.generators`) provides synthetic data for testing and tutorials:

```python
from neural_mi import generators

# Correlated Gaussians with known MI
x, y = generators.generate_correlated_gaussians(
    n_samples=2000, dim=4, mi=1.5,   # mi in bits
    use_torch=True
)

# Nonlinear data via shared latent variable
x, y = generators.generate_nonlinear_from_latent(
    n_samples=2000, latent_dim=2, observed_dim=8, mi=1.0
)

# Time-lagged correlation (for lag analysis)
x, y = generators.generate_temporally_convolved_data(
    n_samples=5000, lag=30, noise=0.1
)

# XOR: high MI, purely nonlinear
x, y = generators.generate_xor_data(n_samples=2000, noise=0.05)

# Correlated spike trains (for spike-timing analysis)
spike_x, spike_y = generators.generate_correlated_spike_trains(
    n_neurons=10, duration=100.0, firing_rate=5.0,
    delay=0.02,          # 20 ms delay from x to y
    jitter=0.005         # 5 ms jitter
)

# Correlated categorical states
x, y = generators.generate_correlated_categorical_series(...)

# Event-related data
x, y = generators.generate_event_related_data(...)
```

**Utility:**
```python
# Convert MI (bits) to Pearson correlation for Gaussians
rho = generators.mi_to_rho(dim=4, mi=1.5)
```

---

## 10. Model Architecture Reference

### Embeddings (`neural_mi.models`)

Most embedding models take tensors of shape `(batch, n_channels, window_size)` and output `(batch, embedding_dim)`. **Exception:** `CNN2D` and `PretrainedBackboneEmbedding` expect 4-D input `(batch, n_channels, H, W)`.

```python
from neural_mi.models import (
    MLP, CNN1D, CNN2D, GRU, LSTM, TCN, Transformer,
    PretrainedBackboneEmbedding,
)
```

| Class | Input shape | Key init params |
|-------|-------------|----------------|
| `MLP` | `(N, C, W)` — flattened to `C×W` | `input_dim, embedding_dim, hidden_dim, n_layers` |
| `CNN1D` | `(N, C, W)` | `input_dim, embedding_dim, hidden_dim, kernel_size` |
| `CNN2D` | `(N, C, H, W)` ← **4-D** | `input_dim (= n_channels), embedding_dim, hidden_dim, kernel_size` |
| `GRU` | `(N, C, W)` | `input_dim, embedding_dim, hidden_dim, n_layers, bidirectional` |
| `LSTM` | `(N, C, W)` | `input_dim, embedding_dim, hidden_dim, n_layers, bidirectional` |
| `TCN` | `(N, C, W)` | `input_dim, embedding_dim, hidden_dim, kernel_size` |
| `Transformer` | `(N, C, W)` | `input_dim, embedding_dim, nhead, n_layers` |
| `PretrainedBackboneEmbedding` | `(N, C, H, W)` ← **4-D** | `input_dim, embedding_dim, pytorch_predefined, pretrained` |

`CNN2D` uses `AdaptiveAvgPool2d(1)` after the convolutional stack so it accepts any spatial size. All embeddings output `(batch, embedding_dim)`.

(A depthwise-separable `CNN1D` variant, a `SpikePhysicsEmbedding` class, and a `SincEmbedding` class were evaluated empirically against generic encoders, did not outperform them, and have been removed.)

### Critics (`neural_mi.models`)

```python
from neural_mi.models import SeparableCritic, ConcatCritic, HybridCritic
```

All critics output a score matrix `(batch_size, batch_size)`.

| Class | Behavior |
|-------|---------|
| `SeparableCritic` | Separate embedding networks + bilinear product |
| `ConcatCritic` | Concatenated inputs → shared MLP |
| `HybridCritic` | Large bottleneck; auto-used by `dimensionality` mode |

### Custom Models

```python
import torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, **kwargs):
        super().__init__()
        self.net = nn.Sequential(...)

    def forward(self, x):   # x: (batch, channels, window)
        return self.net(x)  # → (batch, embedding_dim)

result = nmi.run(x, y, model=Model(custom_embedding_cls=MyEmbedding, embedding_dim=64))

# Or pass a fully-built critic:
critic = SeparableCritic(...)
result = nmi.run(x, y, model=Model(custom_critic=critic))
```

---

## 11. Exceptions

```python
from neural_mi.exceptions import (
    NeuralMIError,          # Base exception
    DataShapeError,         # Invalid shape for given processor_type
    InsufficientDataError,  # Not enough data for the requested operation
    TrainingError,          # Model training failed (NaN loss, etc.)
)
```

---

## 12. Design Decisions & Internals

### Single Entry Point
All modes go through `neural_mi.run()`. Internally it dispatches to analysis-module functions after validation, preprocessing, and parameter normalization. The analysis modules are importable directly for advanced use:

```python
from neural_mi.analysis import (
    run_conditional_mi,
    run_transfer_entropy,
    run_pairwise_mi,
    run_rigorous_analysis,
)
```

### 3D Tensor Convention
All data inside the library is 3D: `(n_samples, n_channels, window_size)`. Pre-processed 2D data `(n_samples, n_channels)` gets a trailing `1` appended automatically via `StaticDataset`. This means `window_size=1` is the common case for pre-processed data.

### Unit Conversion
All internal computations are in **nats** (natural log). Conversion to bits (`× 1/ln(2)`) happens at the `run()` output stage if `output_units='bits'` (default). All sub-keys in `result.details` (e.g., `i_xypast_yfuture`, `cmi_estimate`, `mi_xz_y`) are converted consistently.

### ParameterSweep Class
`sweep`, `conditional`, `transfer`, `rigorous`, and `lag` modes all internally use the `ParameterSweep` class from `neural_mi.analysis.sweep`. It:
1. Generates the Cartesian product of `sweep_grid`.
2. Validates parameter combinations (e.g., warns that `embedding_dim` has no effect with `concat` critic).
3. Runs tasks in parallel via `concurrent.futures.ProcessPoolExecutor` if `n_workers > 1`.

### Blocked vs. Random Splits
- **`'blocked'`** (default): Test set consists of `n_test_blocks` contiguous blocks distributed across the recording. A `split_gap_fraction` buffer is excluded from training on either side of each block. Appropriate for time series with temporal correlations.
- **`'random'`**: IID random split. Use only when temporal correlations are not a concern.

### Rigorous Mode — γ Space
The `rigorous` mode trains on subsets of size `N/γ`. The bias correction works in `γ` space because the bias `a/N_chunk = (a/N)γ` is linear in γ when `N` is fixed. The functions `_find_linear_region` and `_extrapolate_mi` (in `analysis/rigorous.py`) fit MI vs γ, extrapolate to γ=0 (infinite data), and use the per-run `train_mi` as the dependent variable, consistent with every other mode.

### Pairwise Mode — Channel Naming
The output DataFrame uses columns `ch_x`, `ch_y`, `mi_mean`, `mi_std` (integer channel indices). `mi_mean` holds the mean MI across sweep runs; `mi_std` holds the standard deviation (0 when only one run is performed). The MI matrix in `result.details['mi_matrix']` holds per-pair means:
- **Self-pairwise**: upper triangle (symmetric; diagonal = 0 by convention)
- **Cross-pairwise**: full `(n_ch_x, n_ch_y)` matrix

### Transfer Entropy vs. Conditional MI
| Feature | `transfer` mode | `conditional` mode |
|---------|-----------------|---------------------|
| Formula | TE(X→Y) = I(x_past,y_past;y_future) − I(y_past;y_future) | CMI = I(XZ;Y) − I(Z;Y) |
| History built by | Library (sliding windows) | User provides z_data |
| Input shape | 2D `(T, channels)` raw | 3D `(samples, channels, window)` pre-processed |
| Use case | Directed temporal coupling | Controlling for known confounds |

### Online Augmentations — Training Only
Augmentations are applied inside the `Trainer` batch loop and are deliberately skipped during evaluation. This means the test-set MI estimate is always computed on clean data, regardless of augmentation settings — preventing artificially inflated generalisation scores. The application order within a batch is always **spatial → non-spatial → custom**, matching the order shown in §8. Spatial augmentations on non-4-D input emit a `UserWarning` and are skipped gracefully rather than raising an error, so the same `augmentation_params` dict can be used across model types without defensive branching.

### Logging
```python
import neural_mi as nmi
nmi.set_verbose(True)               # Enable INFO-level logs
nmi.set_verbosity(logging.DEBUG)    # Fine-grained control
```

Or pass `verbose=True` to `run()` for per-call verbosity.

---

## Quick Reference Card

```
nmi.run(x, y, mode=..., **kwargs) → Results

Modes:
  estimate     → result.mi_estimate
  sweep        → result.dataframe [sweep_var, mi_mean, mi_std]
  dimensionality → result.dataframe [embedding_dim, train_mi, pr_eig, pr_singular]
  rigorous     → result.mi_estimate ± result.details['mi_error']
  lag          → result.dataframe [lag, train_mi]
  precision    → result.mi_estimate (baseline MI); result.details['precision_tau'], ['precision_thresholds']
  conditional  → result.mi_estimate  (I(X;Y|Z)); z_time= for temporal Z
  transfer     → result.mi_estimate  (TE(X→Y)); Transfer(bidirectional=True) adds te_yx, directionality_index
  pairwise     → result.dataframe [ch_x, ch_y, mi_mean, mi_std]

Estimators: 'infonce' (default, has ceiling), 'smile' (no ceiling)
Embeddings:  'mlp' (default), 'cnn', 'cnn2d', 'gru', 'lstm', 'tcn', 'transformer', 'pretrained_backbone'
Critics:     'separable' (default), 'concat', 'hybrid'
Units:       'bits' (default) or 'nats'

Processors:  'continuous' | 'spike' | 'categorical' | None (pre-processed)
  sample_rate= wired for 'continuous' and 'categorical' (overrides period from time vector)
  max_spikes_per_window= and n_seconds= wired for 'spike'
  4-D tensors (N,C,H,W) pass through unchanged; use with embedding_model='cnn2d'

Augmentations (training-only, via the Training config):
  augmentation_params={'gaussian_noise': {'std': 0.05}}  # shared X and Y
  augmentation_params_x={...}   augmentation_params_y={}   # per-variable
  Spatial (4-D only): random_flip_h, random_flip_v, random_rotation_90, random_crop,
                      random_erase, time_mask, freq_mask, gaussian_blur
  Non-spatial:        gaussian_noise, intensity_scale, channel_dropout
  Custom:             {'custom': callable_or_list}

Results methods:  .plot()  .summary()  .save()  .to_json()  Results.load(path)  Results.compare([r1, r2], labels=[...])
```

---

## Enhanced Rigorous Mode Diagnostics

### Rigorous config fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `residual_threshold` | float | 2.5 | Flag `fit_quality_warning=True` if any externally studentized residual exceeds this value. |
| `r2_threshold` | float | 0.90 | R² is computed and reported in `result.details['r_squared']` but does not affect `fit_quality_warning` or `is_reliable` (see note below). |
| `leverage_threshold` | float | 0.20 | Flag `leverage_warning=True` if LOO intercept shift `δ = |I_full − I_loo|/(|I_full|+ε)` exceeds this value. |

### `result.details` keys (rigorous mode)

| Key | Type | Description |
|-----|------|-------------|
| `fit_quality_warning` | bool | `True` if max externally studentized residual > `residual_threshold`. **Informational only** — does not affect `is_reliable`. |
| `leverage_warning` | bool | `True` if LOO γ=1 intercept shift > `leverage_threshold`. Sets `is_reliable=False`. |
| `r_squared` | float | R² of the WLS linear fit. Reported for transparency; does **not** affect `is_reliable`. `nan` if fewer than 3 points. |
| `max_abs_residual` | float | Maximum absolute externally studentized residual. |
| `loo_intercept_shift` | float | Relative intercept shift when γ=1 is excluded. `nan` if no γ=1 rows or too few LOO points. |

Only `leverage_warning` affects `is_reliable`: if it fires, `is_reliable` is set to
`False`. `fit_quality_warning` is present in the output for transparency but does
**not** affect `is_reliable`.

**Why neither R² nor the residual check governs `is_reliable`:** Both statistics
are scale-dependent and behave pathologically in the heteroscedastic WLS structure
of rigorous mode. (1) R² = 1 − SS_res/SS_tot collapses when the total variance is
small — exactly the case with large N, where finite-sampling bias across gamma is
tiny and all MI estimates cluster tightly. (2) Externally studentized residuals blow
up because low-gamma rows (N samples, low noise) dominate the MSE, while high-gamma
training runs (N/γ samples, high noise) have naturally larger raw deviations — the
ratio e_i/s is then large even for a perfectly valid fit. The only correct
reliability gate is the LOO γ=1 intercept-stability check (Check B): it asks
"does removing the infinite-data anchor destabilize the extrapolation?" — a
scale-invariant question that answers whether the γ=1 → γ=0 extrapolation is safe.

---

## Optional Decoder (Deep Symmetric IB)

When `Model(use_decoder=True)` is set, the Trainer attaches a decoder
to each encoder and adds a weighted MSE reconstruction loss to the training
objective:

- **Deterministic:** `L = −MI(Z_X; Z_Y) + w_x·MSE(X, X̂) + w_y·MSE(Y, Ŷ)`
- **Variational:** `L = KL_X + KL_Y − β·MI + w_x·MSE(X, X̂) + w_y·MSE(Y, Ŷ)`

### Decoder config fields (`Model`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_decoder` | bool | `False` | Enable decoder-augmented training. |
| `decoder_weight` | float | 1.0 | Shared reconstruction weight for both X and Y decoders. |
| `decoder_weight_x` | float\|None | `None` | Per-channel weight override for X decoder. Falls back to `decoder_weight` if `None`. |
| `decoder_weight_y` | float\|None | `None` | Per-channel weight override for Y decoder. Falls back to `decoder_weight` if `None`. |
| `decoder_output_activation_x` | str | `'linear'` | Output activation for X decoder: `'linear'`, `'sigmoid'`, or `'softmax'`. |
| `decoder_output_activation_y` | str | `'linear'` | Output activation for Y decoder. |

### Decoder architecture summary

| Encoder | Decoder |
|---------|---------|
| `mlp` | Mirror MLP (`MLPDecoder`) |
| `cnn1d` | Linear expansion + `nn.Upsample` + `Conv1d` blocks (`CNN1DDecoder`) |
| `gru` | Linear projection → repeated sequence → GRU → `Linear` (`GRUDecoder`) |
| `lstm` | Linear projection → repeated sequence → LSTM → `Linear` (`LSTMDecoder`) |
| `tcn` | Linear expansion + `nn.Upsample` + dilated `Conv1d` blocks (`TCNDecoder`) |
| `transformer` | Linear projection + learned position queries + `TransformerDecoder` (`TransformerDecoder`) |

### `result.details` keys (when `use_decoder=True`)

| Key | Type | Description |
|-----|------|-------------|
| `decoder_recon_loss` | float | Final weighted reconstruction loss `w_x·MSE_x + w_y·MSE_y` evaluated on the training evaluation split. |

---

## Rigorous Bias Correction for Conditional and Transfer Modes

Both `mode='conditional'` and `mode='transfer'` support bias-corrected estimation
by setting `rigorous=True` on their `Conditional` / `Transfer` config:

```python
result = nmi.run(
    x, y,
    mode='conditional',
    conditional=Conditional(z_data=z, rigorous=True,
                            gamma_range=range(1, 11),   # default
                            min_gamma_points=5,          # default
                            confidence_level=0.68),      # default
)
```

The estimator uses a **master permutation** to subsample data consistently at
each γ, so both component estimates (e.g. I(XZ;Y) and I(Z;Y) for CMI) see
the same samples and their noise partially cancels in the difference.

### Config fields for rigorous conditional/transfer

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rigorous` | bool | `False` | Enable rigorous bias correction. |
| `gamma_range` | range\|list\|None | `range(1,11)` | Subsample ratios to sweep. |
| `delta_threshold` | float | 0.1 | Max quadratic-to-linear curvature for linear-region detection. |
| `min_gamma_points` | int | 5 | Minimum γ values required for a reliable fit. |
| `confidence_level` | float | 0.68 | Coverage for the half-CI error bar. |
| `residual_threshold` | float | 2.5 | Same as rigorous mode residual threshold. |
| `r2_threshold` | float | 0.90 | Same as rigorous mode R² threshold. |
| `leverage_threshold` | float | 0.20 | Same as rigorous mode LOO threshold. |

### `result.details` keys (rigorous conditional/transfer)

Same as `mode='rigorous'`: `mi_corrected`, `mi_error`, `slope`, `is_reliable`,
`gammas_used`, `fit_quality_warning`, `leverage_warning`, `r_squared`,
`max_abs_residual`, `loo_intercept_shift`.

`result.params['rigorous']` is set to `True` to distinguish these results from
standard conditional/transfer results.
