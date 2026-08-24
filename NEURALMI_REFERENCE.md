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
   - 6.3 `dimensionality` — Cross-Run-Stable Directions of Shared Structure
   - 6.4 `rigorous` — Bias-Corrected Estimate
   - 6.5 `lag` — Temporal Lag Analysis
   - 6.6 `precision` — Spike-Timing Precision
   - 6.7 `conditional` — Conditional MI
   - 6.8 `transfer` — Transfer Entropy
   - 6.9 `pairwise` — Channel-to-Channel MI Matrix
   - 6.10 `interaction` — Interaction Information
7. [The `Results` Object](#7-the-results-object)
8. [Base Parameters Reference](#8-base-parameters-reference)
9. [Data Generators](#9-data-generators)
10. [Model Architecture Reference](#10-model-architecture-reference)
11. [Exceptions](#11-exceptions)
12. [Design Decisions & Internals](#12-design-decisions--internals)
13. [Information-Quantities Convenience Functions](#13-information-quantities-convenience-functions)

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
- **Spatial analyses**: pairwise channel MI matrix, cross-run-stable directions of shared structure.

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
| Categorical states | `'categorical'` | `(n_channels, n_timepoints)`, any numeric dtype | One-hot or ordinal encoded |
| Pre-processed | `None` (default) | `(n_samples, n_channels)` or `(n_samples, n_channels, window)` | Passed directly |

**Categorical labels don't need to be pre-cast to integers.** Integer-typed data is used directly
as category codes (and must be non-negative). Any other numeric dtype (e.g. `float64` category
labels) is automatically relabeled to consecutive integer codes `0..n_categories-1`, in ascending
sorted order of the distinct values found — with a warning logged so you know it happened. Only
non-numeric data (e.g. strings) is rejected outright.

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

**`Output`** — result formatting: `units` (`'bits'|'nats'`), `track_spectral_history`,
`return_embeddings`, `x_name`, `y_name`, `channel_names_x`, `channel_names_y`.

### Mode-specific configs

- **`Rigorous`** — `gamma_range`, `delta_threshold`, `min_gamma_points`, `confidence_level`.
- **`Precision`** — `tau_grid` (required), `corrupt_target` (`'x'|'y'|'both'`), `corruption_method` (`'rounding'|'noise'`), `n_noise_samples`, `threshold_ratio`.
- **`Lag`** — `lag_range` (required), `equalize_n`.
- **`Transfer`** — `history_window` (required), `prediction_horizon`, `bidirectional`; set `rigorous=True` for bias-corrected TE.
- **`Dimensionality`** — `split_method`, `n_splits`, `channel_indices_x`, `stability_threshold`, `degeneracy_ratio_threshold`, `min_strength_fraction`, `ceiling_mi_fraction`.
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

result.plot()       # 2 swept params -> heatmap (embedding_dim x n_epochs) by default
df = result.dataframe
best = df.loc[df['mi_mean'].idxmax()]
```

**Note on `sweep_grid`:** Keys must be `Model` / `Training` field names (see §8). Processor parameters like `window_size` can also be swept.

**Plotting a multi-parameter sweep:** `result.plot()` auto-selects the plot kind from how many
parameters were swept (excluding `run_id`): 1 -> line (MI vs. the parameter, as before), 2 ->
heatmap (one param per axis, MI as colour), 3+ -> bar chart (one bar per parameter combination,
labelled `"p1=v1, p2=v2, ..."`). Override explicitly with `result.plot(kind='line'|'heatmap'|'bar')`
— e.g. `kind='bar'` also works for a 2-param sweep if you prefer bars to a heatmap.
`kind='heatmap'` requires exactly 2 swept parameters and raises a clear error otherwise.
`mode='lag'` sweeps (which always include `lag` as one swept "parameter") follow the same rule.
`Results.compare()` only supports single-parameter sweep results (a shared 1-D x-axis) — call
`.plot()` on each multi-parameter result individually instead.

---

### 6.3 `dimensionality` — Cross-Run-Stable Directions of Shared Structure

**What it does:** Trains a modest-capacity Hybrid-Critic embedding multiple independent times and reports which directions of shared structure between two views reproduce reliably across every retraining, plus a free, no-training read of whether the views look cleanly separable or entangled. Does not report a scalar dimensionality count: a nonlinear encoder given spare capacity can construct combinations of true factors that are spectrally indistinguishable from genuine ones, so no measure of a single trained spectrum can be trusted as an exact count. See THEORY.md §6 for the full reasoning.

Two sub-modes controlled by whether `y_data` is provided:

| Sub-mode | `y_data` | What's compared |
|----------|----------|----------------|
| **Intrinsic** | None | Two halves of x channels, repeated over different channel-split assignments |
| **Interaction** | Provided | x and y directly, repeated over independent weight initialisations |

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

**`n_splits` kwarg (default 3, min 2):**
- *Intrinsic mode* (`split_method='random'`): number of distinct random channel-split assignments evaluated
- *Interaction mode* (y_data provided): number of independent model fits from different random weight initialisations
- At least 2 splits are required to compute cross-run stability at all (there's nothing to compare a single run against); the default of 3 is a slightly more robust minimum.

**Key kwargs:** `n_workers=1`, `split_method='random'`, `n_splits=3`, `lag=<int>` (for temporal), `channel_indices_x=<list>` (for index split), `stability_threshold=0.7`, `degeneracy_ratio_threshold=1.3`, `min_strength_fraction=0.05`, `ceiling_mi_fraction=0.85`

**Returns:** `Results` with:
- `result.mi_estimate` — `None`; this mode does not produce a single scalar answer.
- `result.dataframe` — one row (or one row per swept-parameter combination if `sweep_grid` is used) with `mi_mean`/`mi_std`, `pr_eig_mean`/`pr_eig_std`, `pr_singular_mean`/`pr_singular_std`, aggregated across splits/reruns. `pr_eig`/`pr_singular` are kept as a labeled secondary diagnostic (see below), not the mode's headline output.
- `result.details['raw_results']` — per-split DataFrame, one row per split: `split_id`, `train_mi`, `test_mi`, `best_epoch`, `n_epochs`, `pr_eig`, `pr_singular`.
  - `pr_eig` — effective spread of that split's own spectrum, eigenvalue-weighted: `(Σσᵢ²)² / Σσᵢ⁴` — stricter, weights large singular values more
  - `pr_singular` — same, singular-value-weighted: `(Σσᵢ)² / Σσᵢ²` — less strict variant
- `result.details['regime_x']` — `{'eigvals', 'ratios', 'peak_rank', 'peak_val', 'regime'}` from the no-training regime diagnostic (THEORY.md §6); `regime` is `'separable-like'` or `'entangled-like'`. `result.details['regime_y']` likewise, interaction mode only. Absent (not an error) for data shapes the diagnostic doesn't support, e.g. 4-D spatial input with fewer than 2 channels.
- `result.details['converged']` — `bool`; `True` only if every split/rerun converged (`best_epoch` short of the final training epoch). A `UserWarning` names how many splits didn't converge when this is `False`.
- `result.details['stability_per_rank']` — `dict` keyed by rank (1-indexed int); each entry: `{'pairwise_abs_corr': [...], 'min_abs_corr': float, 'mean_strength': float, 'below_noise_floor': bool, 'stable': bool, 'near_degenerate_with_next': bool, 'near_degenerate_with_prev': bool}`. Present only when at least 2 splits produced usable rotated embeddings; absent (with a `UserWarning`) otherwise.
- `result.details['stable_directions']` — list of individually-trustworthy rank indices.
- `result.details['stable_but_degenerate_groups']` — list of rank-index groups, each trustworthy as a set but not individually ordered within the group.
- `result.details['n_stable_total']` — `int`; count of all reported directions (individual + grouped). Always a lower bound on genuine shared structure, never a claimed exact count.
- `result.details['embeddings_x']`, `result.details['embeddings_y']` — present only when `return_embeddings=True`; numpy arrays from the **last split's model**, in original sample order, index-aligned with the input data.
- `result.details['embeddings_x_rotated']`, `result.details['embeddings_y_rotated']`, `result.details['embeddings_rotation_singular_values']` — present whenever `result.details['embeddings_x']` is (this mode always applies `return_rotated_embeddings=True` internally); same last-split embeddings, re-projected so dimension 0 captures the most shared variance, dimension 1 the next most, etc.
- `result.details['embeddings_rotation_x']`, `result.details['embeddings_rotation_y']` — rotation matrices U and V, present additionally when `return_rotation_matrices=True`; apply as `new_data_zx @ U` to project new data into the same basis.
- `result.details['embedding_history_x']`, `result.details['embedding_history_y']` — present only when the caller explicitly sets `track_embeddings` (this mode does not force it on — the stability check's rotated-embedding extraction doesn't need per-epoch history).

**Cross-run stability + near-degeneracy — the mode's actual output.** Every split/rerun's rotated cross-covariance directions are compared against every other split's, on held-out test data only:

- **Stable**: a rank's direction is reproducible across splits (minimum pairwise correlation over every pair of splits ≥ `stability_threshold`) *and* above the noise floor (mean singular-value strength ≥ `min_strength_fraction` of the top rank's strength). The noise-floor check is independent of, and catches what, the correlation check alone misses — a pure-noise channel can show spuriously high cross-run correlation by chance despite carrying no real signal.
- **Stable but degenerate**: reproducible and above the noise floor, but within `degeneracy_ratio_threshold` of an adjacent rank's strength — reported as a group, not individually ordered.
- **Not reported**: everything else — didn't reproduce, or below the noise floor.

**Ceiling proximity.** The critic this mode trains still has the usual InfoNCE-family evaluation ceiling, `log(eval_size)` (`eval_size = min(test_size, max_eval_samples)`). A `UserWarning` fires when the mean underlying MI estimate is within `ceiling_mi_fraction` (default `0.85`) of that ceiling. This is informational only, not a remediation: convergence gating already excludes the large majority of near-ceiling cases in practice, and a genuinely converged, near-ceiling run degrades the stable-direction count conservatively rather than misleadingly (THEORY.md §6).

```python
# Intrinsic: cross-run-stable directions between two random halves of x channels, 5 splits
result = nmi.run(x, mode='dimensionality',
                 training=Training(n_epochs=100),
                 dimensionality=Dimensionality(n_splits=5, split_method='random'),
                 n_workers=4)
result.plot()
print(result.details['stable_directions'], result.details['regime_x']['regime'])

# Intrinsic: user-specified channel assignment (e.g. two electrode shanks)
result = nmi.run(x, mode='dimensionality',
                 training=Training(n_epochs=100),
                 dimensionality=Dimensionality(n_splits=5, split_method='index',
                                               channel_indices_x=[0, 1, 2, 8, 9, 10]))

# Interaction: cross-run-stable directions between x and y, 5 independent fits
result = nmi.run(x, y, mode='dimensionality',
                 training=Training(n_epochs=100),
                 dimensionality=Dimensionality(n_splits=5),
                 n_workers=4)
print(result.details['n_stable_total'], result.details['converged'])
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
  - `precision_tau` — τ* for the primary (first) threshold ratio, or `None` if
    MI never dropped below the threshold across `tau_grid` (check with
    `is None`, not `np.isnan`)
  - `threshold_value` — actual MI value at the primary threshold
  - `threshold_ratio` — the original input (scalar or list)
  - `precision_thresholds` — dict mapping each ratio to `{'precision_tau', 'threshold_value'}`
    (`precision_tau` is `None` per-ratio under the same not-found condition)
  - `corruption_method`, `corrupt_target`

```python
# Single threshold (default)
result = nmi.run(spike_x, y, mode='precision',
                 processing=Processing(x='spike',
                                       x_params={'window_size': 0.05, 'n_seconds': 100.0}),
                 precision=Precision(tau_grid=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                     threshold_ratio=0.9))
tau = result.details['precision_tau']
if tau is not None:
    print(f"Precision timescale: {tau*1000:.1f} ms")
else:
    print("MI never dropped below threshold — extend tau_grid to find tau*.")

# Multiple thresholds simultaneously
result = nmi.run(spike_x, y, mode='precision',
                 precision=Precision(tau_grid=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                                     threshold_ratio=[0.9, 0.75, 0.5]))
for ratio, v in result.details['precision_thresholds'].items():
    tau_i = v['precision_tau']
    label = f"{tau_i*1000:.1f} ms" if tau_i is not None else "not found"
    print(f"  {ratio*100:.0f}% threshold: tau* = {label}")
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

**Optional:** `z_processor_type`, `z_processor_params` — same options as for x/y.
Internally, XZ is built by concatenating X and Z's windowed tensors along the
channel axis, so Z's per-window representation is automatically re-laid out to
match X's window: `z_processor_type='categorical'`'s `'majority_vote'` and
`'probability'` encodings (window-constant summaries) are broadcast across
X's window, and `'full_trajectory'` (genuine per-timepoint resolution) keeps
its own timepoint axis.

**`align='dual_branch'`:** for the rarer case where X and Z genuinely differ
in window length beyond the small trim tolerance the default path applies
(MI rate, instantaneous exchange, directed information rate, see
`THEORY.md` §11 and §13 below). Instead of concatenating, X and Z flow
through the pipeline as a tuple, each embedded by its own sub-network via
`custom_embedding_cls=DualBranchEmbedding` (set on `Model(...)`; required
when `align='dual_branch'` is set, or the run raises a clear error). Leave
`align` unset (`None`, the default) unless a mismatch this large is
expected. Not supported together with `permutation_test=True` or
`use_variational=True`; both raise a clear error.

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
- `Transfer(w_data=...)` — optional third conditioning signal, `(T, n_channels_w)`, same leading dimension as `x_data`/`y_data`. Computes *conditional* transfer entropy instead of plain TE:
  ```
  TE(X→Y|W) = I(x_past, y_past, w_past ; y_future) − I(y_past, w_past ; y_future)
  ```
  `w_past` is built the same way as `x_past`/`y_past` (same `history_window`), folded into both the joint and marginal conditioning arrays, and into both directions when `bidirectional=True`. `w_data=None` (the default) is byte-for-byte identical to plain TE. Optional `w_time`/`w_processor_type`/`w_processor_params` mirror `Conditional`'s `z_*` fields if `w_data` needs its own preprocessing before use; the raw, already-numeric case (the common one) needs none of them. Works with `rigorous=True` (see Appendix C); `w_data` is chunked identically to `x_data`/`y_data` for every gamma-chunk.

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

# Conditional TE — control for a third signal's influence
result = nmi.run(x, y, mode='transfer',
                 transfer=Transfer(history_window=20, w_data=w),
                 n_workers=4)
print(f"TE(X→Y|W) = {result.mi_estimate:.3f} bits")
```

**Note:** `x_data`, `y_data`, and (if given) `w_data` must be 2D here: `(T, n_channels)`, i.e., a raw temporal sequence. The library builds sliding windows internally.

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

### 6.10 `interaction` — Interaction Information

**What it does:** Computes interaction information, how much shared information between X and Y changes once a third population W is also observed:

```
II = I(X, W; Y) − I(X; Y) − I(W; Y)
```

The one quantity in the taxonomy (`THEORY.md` §12) that isn't a single conditional MI call: three separate MI estimates combined by a formula rather than a two-term chain-rule difference. `x_data` and `w_data` are concatenated along the channel axis to build the joint `I(X,W;Y)` term, so they must share the same window size; `y_data` can differ.

**Required:** `Interaction(w_data=...)`, `w_data` shape `(n_samples, n_channels_w, window_size)` (or `(n_samples, n_channels_w)`, treated as window size 1), same leading dimension as `x_data`/`y_data`.

**Key parameters:**
- `Interaction(w_time=..., w_processor_type=..., w_processor_params=...)` — optional, mirrors `Conditional`'s `z_*` fields if `w_data` needs its own preprocessing before use.
- `Interaction(rigorous=True, gamma_range=..., ...)` — bias-corrected extrapolation, same mechanics as `mode='conditional'`/`mode='transfer'`'s rigorous path (Appendix C).

**Returns:** `Results` with:
- `result.mi_estimate` — float: II in `output_units` (can be negative: redundancy gives II < 0, synergy gives II > 0, see `THEORY.md` §12)
- `result.details`:
  - `interaction_info` — same as `mi_estimate`
  - `mi_xw_y` — I(X,W;Y)
  - `mi_x_y` — I(X;Y)
  - `mi_w_y` — I(W;Y)
  - `raw_xw_y`, `raw_x_y`, `raw_w_y` — per-run lists for the three component sweeps

```python
result = nmi.run(x, y, mode='interaction',
                 interaction=Interaction(w_data=w),
                 training=Training(n_epochs=100),
                 n_workers=4)
print(f"II = {result.mi_estimate:.3f} bits")
print(f"I(X,W;Y)={result.details['mi_xw_y']:.3f}, "
      f"I(X;Y)={result.details['mi_x_y']:.3f}, "
      f"I(W;Y)={result.details['mi_w_y']:.3f}")
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
| `sweep` / `lag` | Auto-selected by parameter count: line (1 param), heatmap (2), bar (3+) | Line: MI vs swept variable, shaded ±1 std. Heatmap/bar cover the parameters a line plot can't show. Override with `kind='line'\|'heatmap'\|'bar'` |
| `dimensionality` | Per-rank bar chart: stable / stable-but-degenerate / not-stable | Requires ≥2 splits (`stability_per_rank` in `result.details`); creates its own figure unless `ax=` is supplied |
| `rigorous` | MI vs γ with WLS fit and extrapolation | Reliability box always shown: green "reliable" when `is_reliable=True`, red with a dynamic reason (`fit_quality_warning`/`leverage_warning`) when `False` |
| `conditional` | Bar chart: I(XZ;Y), I(Z;Y), CMI I(X;Y\|Z) | |
| `transfer` | Bar chart: TE(X→Y), TE(Y→X) | Title shows directionality index when present |
| `precision` | MI vs τ with baseline and threshold lines | |
| `pairwise` | MI matrix heatmap | |

```python
fig, ax = plt.subplots()
result.plot(ax=ax, title="My analysis", show=False)

# Dimensionality: embed the per-rank stability chart in your own figure
fig, ax = plt.subplots(figsize=(8, 5))
result.plot(ax=ax, show=False)
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
| `sweep` / `lag` | Sweep curves with distinct colours |
| `rigorous` | Bias-correction fits with distinct colours per `labels=`, plus a per-result reliability text box |

Not supported for `dimensionality`: its per-rank stable/degenerate/below-floor chart isn't a
curve, so overlaying several isn't well-defined — call `result.plot()` on each result
individually instead.

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

#### `plot_dimensionality_curve(details, ax, show, **kwargs) → plt.Axes`

Per-rank bar chart of which directions of shared structure are trustworthy, built from
`details['stability_per_rank']` (plus `'stable_directions'` and
`'stable_but_degenerate_groups'`) — pass `result.details`, not `result.dataframe`. One bar
per embedding rank, height = mean singular-value strength across splits (log scale), coloured
green (stable), amber/hatched with a bracket annotation (stable but degenerate group), or gray
(not stable / below noise floor). Raises `ValueError` if `stability_per_rank` is missing (fewer
than 2 splits). When `ax=None` (default) a new figure is created. `show=False` suppresses
`plt.show()`.

```python
from neural_mi.visualize import plot_dimensionality_curve
ax = plot_dimensionality_curve(result.details, show=False)
```

#### `plot_bias_correction_fit(raw_df, corrected_result, ax, units, show, label, color) → plt.Axes`

Visualises the WLS extrapolation used in rigorous mode.  `show=False` suppresses
`plt.show()`.  `label`/`color`, if given, apply to every element of this result
(raw points, mean line, fit line, corrected-MI marker) and collapse them to one
legend entry under `label` -- this is what `Results.compare()` uses to keep
multiple overlaid results visually distinct.  Without them, uses the original
single-result scheme (gray points, black mean line, red fit/marker, three
descriptive legend entries).  Returns the axes so the caller can add
annotations.

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
# track_embeddings=True: dimensionality mode does not track per-epoch
# embeddings by default (not needed for its cross-run stability check).
result = nmi.run(x, mode='dimensionality', output=Output(track_embeddings=True),
                 training=Training(n_epochs=50))

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
| `track_spectral_history` | bool | False | If `True`, additionally records `pr_eig`, `pr_singular`, and the raw singular-value spectrum at *every* epoch (can be expensive -- evaluates on the same eval subset each epoch; size is governed by `train_subset_size`/`max_eval_samples`, same as everything else). |

`result.details` always contains, from the best epoch, at no extra cost regardless
of `track_spectral_history` (they're a byproduct of the participation-ratio
calculation that runs unconditionally):
- `pr_eig`, `pr_singular` — the two participation-ratio variants.
- `spectrum` — the raw cross-covariance singular values they were computed from.

With `track_spectral_history=True`, `result.details['spectral_metrics_history']`
additionally holds a list of per-epoch dicts, each with the same `pr_eig` /
`pr_singular` / `spectrum` keys plus `spectral_whitening` (echoing the setting
used). `effective_rank`/`spectral_entropy` are not computed -- both are cheaply
derivable from `spectrum` if needed.

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
| `track_embeddings` | bool / int / float / `'full'` | `False` | Per-epoch embedding tracking for `animate_training()`. `False` disables (including in `dimensionality` mode, which does not need per-epoch history for its cross-run stability check and does not force this on). `True` or `512` tracks the first 512 samples each epoch. A positive `int` specifies an exact count; a `float` in (0,1) is a fraction of the dataset; `'full'` tracks all samples (emits a `UserWarning`). Embeddings are always taken from the first N samples in original order. Stored in `result.details['embedding_history_x']` and `result.details['embedding_history_y']` (lists of `(n_tracked, embed_dim)` arrays, one per epoch). |
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

**`DualBranchEmbedding`** is the one exception to the "single window length" rule above: used via `custom_embedding_cls=DualBranchEmbedding` (not a `embedding_model=` string) together with `Conditional(align='dual_branch')` (§6.7), for the case where X and Z genuinely differ in window length (`mi_rate`/`instantaneous_exchange`/`directed_information_rate`, §13). `input_dim` is a 2-tuple `(dim_a, dim_c)` for the compound side, or a plain `int` for the ordinary side; `build_critic` passes `custom_embedding_cls` through unchanged to both, so one class instance serves each role. Two independent `branch_cls` sub-networks (default `GRU`), each at its own length, fused by a small MLP. Not compatible with `use_variational=True` or `mode='sweep'`'s `max_samples_per_task`, both raise a clear error. A non-default `branch_cls` needs a thin subclass (see the class docstring), since `build_critic` only ever passes the fixed `(input_dim, hidden_dim, embed_dim, n_layers)` contract through to a custom class.

### Critics (`neural_mi.models`)

```python
from neural_mi.models import SeparableCritic, ConcatCritic, HybridCritic
```

All critics output a score matrix `(batch_size, batch_size)`.

| Class | Behavior |
|-------|---------|
| `SeparableCritic` | Separate embedding networks + bilinear product |
| `ConcatCritic` | Concatenated inputs → shared MLP |
| `HybridCritic` | MLP decision head on the concatenated embeddings; auto-used by `dimensionality` mode |

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
3. Runs tasks in parallel via a `torch.multiprocessing` (`'spawn'` context) `Pool` if `n_workers > 1`.

### Parallelization Across Modes
Every mode with more than one independent unit of work dispatches it to a `torch.multiprocessing` `Pool(n_workers)` when `n_workers > 1`: `sweep`/`estimate` and non-rigorous `conditional`/`transfer` (via `ParameterSweep`), `rigorous` and rigorous `conditional`/`transfer` (one worker per gamma-subset task), `dimensionality` (one worker per channel-split), and `pairwise` (one worker per channel pair). When one of these dispatches to an outer pool, any *inner* sweep for that same unit of work (e.g. a `run_id` sweep_grid within one gamma-chunk or one channel pair) runs with `n_workers=1` internally to avoid nested pools — so parallelism is always spent on the outer, more numerous loop. `precision` mode has no independent inner loop to parallelize (it trains one baseline model, then evaluates it — inference only — across the `tau_grid`), so `n_workers` has no effect there by design.

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

### Temporal Window Shifting — Two Mechanisms, Different Reach

`Training(shift_windows=True)` and `Training(shift_time=True)` each re-tile a windowed dataset with a different random start offset every epoch, so training sees a different set of window boundaries each pass rather than one fixed tiling — a temporal augmentation, on by default wherever it applies. Which mechanism a given pair uses depends on its data structure: regularly-sampled grid data (`continuous`, `categorical`) can be re-tiled with a plain re-slice (`torch.Tensor.unfold`), while irregular, event-based data (`spike`) needs its window contents recomputed from the raw events on every shift.

Which pair uses which mechanism (`processor_type_x`, effective `processor_type_y`):

| X / Y | mechanism | notes |
|---|---|---|
| `continuous` + `continuous`, `categorical` + `categorical`, `continuous` + `categorical` | `shift_windows` | the "regular grid" family — cheap reslice |
| `spike` + `spike` | `shift_time` | both sides natively in seconds, no cross-unit concern |
| `spike` + (`continuous`/`categorical`), either order | `shift_time`, **only if** the continuous/categorical side has `sample_rate` set | otherwise a shift value would mean seconds on one side and raw samples on the other |
| anything + `None` (pre-processed/static) | neither | no raw signal to reslice from |

Both are reachable at `mode='estimate'`, `mode='pairwise'`, and plain (non-processor-swept) `mode='sweep'` — each pair's training run is independent, so there's no cross-run comparison for independent per-pair shift randomness to disturb. `shift_time` additionally reaches `mode='lag'` and processor-swept `mode='sweep'`. Neither reaches the comparison modes (`dimensionality`/`rigorous`/`precision`/`conditional`/`transfer`/`interaction`), which compare or subtract sub-run results against each other and need every sub-run to see identical windows.

- **`shift_windows`** is the mechanism for the "regular grid" family (`continuous`, `categorical`, in any combination). Because the sampling grid is regular, shifting is a plain re-slice (`torch.Tensor.unfold`, no interpolation), and the window count is held fixed across every shift, so it never disturbs the train/test split and never constructs a `SubsetView`. For `categorical` data the reslice is followed by a vectorized re-encoding step (`majority_vote`/`probability`/`full_trajectory`, matching `CategoricalWindowDataset`'s own three modes). If X and Y use different `sample_rate`s, each side's `window_size`/`step_size`/shift is converted to its own native sample count independently, keeping both sides' tiling in sync by real time. The shift is drawn fresh each epoch, uniformly from `[0, window_size)`, independent of `step_size`.
- **`shift_time`** takes effect when windowing is deferred to the training worker (see reach above). It rebuilds the entire windowed array on every shift (interpolation for continuous, a vectorized `searchsorted`-based rebin for spike), so every epoch pays a rebuild cost. For a mixed `spike`+regular-grid pair without `sample_rate` on the regular-grid side, the library refuses to shift (warns and leaves the pair unshifted) rather than silently misalign X and Y. `shift_windows` is always preferred for the regular-grid family; `shift_time` is the only option for `spike` data and for `mode='lag'`/processor-swept `mode='sweep'`.
- **A cross-unit gap, independent of shifting**: pairing `spike` with `continuous`/`categorical` when the regular-grid side lacks `sample_rate` produces a warning from `create_dataset` regardless of whether any shift is requested — `PairedTemporalDataset` combines both sides' temporal extents via plain numeric `min`/`max`, which is only meaningful if both sides are already in the same time unit.
- **Evaluation is decoupled from training's shift state.** The reported `test_mi`/`train_mi` (and everything derived from it: early stopping, best-epoch selection, decoder reconstruction loss, spectral metrics, per-epoch embedding tracking) is always measured against a frozen snapshot of the data taken before any shift, using the original test/train-eval index arrays rather than `SubsetView`'s live-updating `.indices`. Training batches see the live, currently-shifted data.

Set `shift_windows=False`/`shift_time=False` to disable shifting for a specific pair. If `n_epochs`/`patience` need adjusting for a particular dataset, the "under-trained lower bound" warning already flags it regardless of cause.

### Logging
```python
import neural_mi as nmi
nmi.set_verbose(True)               # Enable INFO-level logs
nmi.set_verbosity(logging.DEBUG)    # Fine-grained control
```

Or pass `verbose=True` to `run()` for per-call verbosity.

---

## 13. Information-Quantities Convenience Functions

`neural_mi/quantities.py` provides named functions for the temporal
information quantities described in `THEORY.md` §11, thin wrappers that
build the right offset arrays and call `run(mode='estimate', ...)`
underneath, so they accept the same `model=`/`training=`/etc. keyword
arguments as `run()` and return the same `Results` object. Most of these
need no new `mode=` string at all, `mode=` tracks estimation *mechanism*
(how many sweeps run and how they're combined), and every quantity except
interaction information is a single sweep under an existing mechanism
(`mode='estimate'`, `mode='transfer'`). Interaction information is the one
exception, a genuine three-sweep combination, so it gets its own
`mode='interaction'` (§6.10) with a matching convenience function.

Each function's construction parameter (`k`, `past_k`, `window_size`) accepts
either a scalar (one call, one `Results`) or an iterable, which dispatches a
parallel sweep across values via `n_workers` and returns a `pandas.DataFrame`
instead:

```python
import neural_mi as nmi

# Single value -> Results, same as mode='estimate'
r = nmi.active_information_storage(x_data, k=10)

# Iterable -> DataFrame with columns 'k', 'mi_estimate', dispatched in parallel
df = nmi.active_information_storage(x_data, k=[5, 10, 20], n_workers=4)
```

| Function | Quantity | Construction |
|---|---|---|
| `active_information_storage(x_data, k, future_k=1, ...)` | $I(X_{past}; X_0)$ | `offsets.build_past_future(x_data, past_len=k, future_len=future_k)` |
| `excess_entropy(x_data, k, future_k, ...)` | $I(X_{past}; X_{fut})$ | same, with a multi-sample `future_k` |
| `instantaneous_mi(x_data, y_data, ...)` | $I(X_0; Y_0)$ | direct pass-through to `mode='estimate'`, no construction |
| `cross_predictive_information(x_data, y_data, past_k, future_k=1, ...)` | $I(X_{past}; Y_{fut})$ | `offsets.build_cross_offset(x_data, y_data, past_len=past_k, future_len=future_k)` |
| `block_mi(x_data, y_data, window_size, ...)` | $I(X_{1:w}; Y_{1:w})$ | `Processing(x='continuous', y='continuous', x_params={'window_size': ...}, ...)` |
| `conditional_transfer_entropy(x_data, y_data, w_data, history_window, ...)` | $I(Y_0; X_{past} \mid Y_{past}, W_{past})$ | `mode='transfer'` with `Transfer(w_data=...)`, see §6.8 |
| `interaction_information(x_data, y_data, w_data, ...)` | $I(X,W;Y) - I(X;Y) - I(W;Y)$ | `mode='interaction'`, see §6.10 |
| `mi_rate(x_data, y_data, h, W=20, ...)` | $I(X_{all}; Y_0 \mid Y_{past}(h))$ | `mode='conditional'` with `align='dual_branch'`, see below |
| `instantaneous_exchange(x_data, y_data, k, ...)` | $I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))$ | same |
| `directed_information_rate(x_data, y_data, k, ...)` | $I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))$ | same |

`x_data`/`y_data` for the first four are raw, unwindowed `(T, n_channels)`
arrays. The offset construction builds the windowed shape internally, using
the same `torch.Tensor.unfold`-based sliding window `mode='transfer'`
already uses for its own past/future arrays (`neural_mi/analysis/offsets.py`).
`block_mi` is the exception: it routes through the library's own `Processing`
windowing rather than a hand-built offset, since $X_{1:w}$ vs. $Y_{1:w}$ is
literally what a windowed processor already builds correctly (interpolation,
coverage validation, blocked-split geometry included).

The sweep path (an iterable construction parameter) dispatches via
`neural_mi.parallel.dispatch_tasks`, the same `spawn`-context worker-pool
idiom used throughout the library (`analysis/rigorous.py`,
`analysis/dimensionality.py`, `analysis/pairwise.py`,
`analysis/sweep.py`), factored into one shared helper rather than a
seventh independent copy. It intentionally returns a plain `DataFrame`
rather than a `Results` with full `plot()`/`summary()` support, a
reasonable future enhancement, not required to make the parameter
sweepable.

`mi_rate`, `instantaneous_exchange`, and `directed_information_rate` are the
three exceptions to "no new estimation machinery": $A$ and $C$ genuinely
differ in window length for all three, so they need
`model=Model(embedding_model='gru', custom_embedding_cls=DualBranchEmbedding, ...)`
and route through `Conditional(align='dual_branch')` (§6.7) automatically.
Each raises a clear `ValueError` upfront if `model=` isn't configured with a
`DualBranchEmbedding`-based `custom_embedding_cls`, at the `h`/`k > 0`
boundary where a mismatch actually occurs (`h=0`/`k=0` has no conditioning
at all and needs no special model). See `THEORY.md` §11 for the array
construction and why directed information rate is estimated directly rather
than via its exact TE + instantaneous-exchange decomposition, and §10.

```python
from neural_mi import Model, Training, mi_rate
from neural_mi.models import DualBranchEmbedding

model = Model(embedding_model='gru', custom_embedding_cls=DualBranchEmbedding,
              embedding_dim=16, hidden_dim=64, n_layers=2)
df = mi_rate(x_data, y_data, h=[0, 5, 10, 20], W=20, model=model,
             training=Training(n_epochs=100), n_workers=4)
```

---

## Quick Reference Card

```
nmi.run(x, y, mode=..., **kwargs) → Results

Modes:
  estimate     → result.mi_estimate
  sweep        → result.dataframe [sweep_var, mi_mean, mi_std]
  dimensionality → result.details['stable_directions', 'stable_but_degenerate_groups', 'regime_x']
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
reliability gate is the LOO γ=1 intercept-stability check: it asks
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

`rigorous=True` also works with `Conditional(align='dual_branch', ...)`
(`mi_rate`/`instantaneous_exchange`/`directed_information_rate`, §13):
bias correction is a property of the InfoNCE-family estimator itself, not
of which array-construction path built X and Z, so no separate rigorous
mechanism is needed. X stays a plain tensor throughout the γ-chunking
internals; Z rides along via the same `extra_data` mechanism already used
for the ordinary conditional-rigorous path, and the tuple is assembled only
at the point where `run_conditional_mi` is actually called.
