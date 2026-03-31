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
| Multi-layer Perceptron | `'mlp'` (default) | Flattens input; good default |
| 1D Convolutional | `'cnn'` | Uses `kernel_size` param |
| Gated Recurrent Unit | `'gru'` | For sequences; `bidirectional` option |
| Long Short-Term Memory | `'lstm'` | For sequences; `bidirectional` option |
| Temporal Convolutional Net | `'tcn'` | Dilated 1D conv; good for long windows |
| Transformer | `'transformer'` | Self-attention; needs `nhead` param |

All embeddings output a vector of size `embedding_dim` (default 64).

### 3.4 Critic Architectures

The critic `f(x, y)` combines the two embeddings into a score. Three architectures:

| Critic | `critic_type` value | Notes |
|--------|--------------------|-|
| **Separable** | `'separable'` (default) | `f(x,y) = gₓ(embed(x))ᵀ g_y(embed(y))` — bilinear product of separate head networks |
| **Concat** | `'concat'` | Concatenates raw inputs before any embedding; ignores `embedding_dim` |
| **Hybrid** | `'hybrid'` | Similar to Separable with a concat embeddings instead of dot product; used automatically by `dimensionality` mode |

**Choosing:** `separable` is the best general choice. `concat` is the most flexible but doesn't allow for embedding dimensionality and very costly to train. `hybrid` is reserved for dimensionality analysis (library sets it automatically).

### 3.5 Bias in Finite-Sample Estimation

Neural MI estimators are **biased upward** at small sample sizes — the critic can memorize rather than generalize. The bias scales roughly as `O(1/N)`. The `rigorous` mode exploits this:

1. Train models on subsets of size `N/γ` for γ = 1, 2, …, 10.
2. Plot estimated MI vs `1/γ`.
3. Fit a line to the linear portion: `MI(1/γ) = MI_true + slope × (1/γ)`.
4. Extrapolate to `1/γ → 0` (infinite data): `MI_true ≈ intercept`.

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

### Post-Processing Shape Convention

After any processor, all data tensors are 3D: `(n_samples, n_channels, window_size)`. For pre-processed 2D data, a trailing dim-1 is added automatically: `(n_samples, n_channels, 1)`. This is the **internal tensor format** throughout the library.

---

## 5. The `run()` Function — Complete Reference

```python
import neural_mi as nmi

result = nmi.run(
    # ── Data ────────────────────────────────────────────────────────────────
    x_data,                          # Required. See §4 for shapes.
    y_data=None,                     # Optional (required by most modes).
    x_time=None,                     # np.ndarray of timestamps for x
    y_time=None,                     # np.ndarray of timestamps for y

    # ── Processors ──────────────────────────────────────────────────────────
    processor_type_x=None,           # 'continuous' | 'spike' | 'categorical' | None
    processor_params_x=None,         # dict; see §4
    processor_type_y=None,
    processor_params_y=None,

    # ── Analysis Mode ───────────────────────────────────────────────────────
    mode='estimate',                 # 'estimate'|'sweep'|'dimensionality'|'rigorous'|'lag'|'precision'|'conditional'|'transfer'|'pairwise'

    # ── Model Configuration ────────────────────────────────────────────────
    base_params=None,                # dict; see §8 for full reference
    sweep_grid=None,                 # dict[str, list] for 'sweep' mode
    estimator='infonce',             # 'infonce' | 'smile'
    estimator_params=None,           # dict; e.g. {'clip': 5.0} for smile
    custom_critic=None,              # Pre-built nn.Module
    custom_embedding_cls=None,       # Custom embedding class (not instance)

    # ── Output ──────────────────────────────────────────────────────────────
    output_units='bits',             # 'bits' | 'nats'
    return_embeddings=False,         # Include learned embeddings in result
    save_best_model_path=None,       # str path to save best checkpoint

    # ── Training / Splitting ───────────────────────────────────────────────
    n_epochs=None,                   # Override base_params['n_epochs'] (default 50)
    batch_size=None,                 # Override base_params['batch_size'] (default 128)
    shared_encoder=None,             # Override base_params['shared_encoder']
    split_mode='blocked',            # 'blocked' (temporal) | 'random' (IID)
    train_fraction=0.9,
    n_test_blocks=5,                 # For 'blocked' split: # contiguous test segments
    split_gap_fraction=0.5,          # Buffer fraction around each test block
    train_indices=None,              # np.ndarray of explicit train indices
    test_indices=None,               # np.ndarray of explicit test indices

    # ── Reproducibility ─────────────────────────────────────────────────────
    random_seed=None,                # int; use with n_workers=1 for full repro
    device=None,                     # 'cpu' | 'cuda' | 'mps' | None (auto)

    # ── Memory / device layout ───────────────────────────────────────────────
    # Where dataset tensors live — independent of `device` (the compute device).
    # The Trainer always moves batches to `device` via .to(device), so this
    # setting only controls dataset-level allocation.
    #
    #   'cpu'  (default) — data in pageable system RAM; OS can reclaim freely
    #                      between tasks. Use for sweep / dimensionality / lag.
    #   'auto'           — data on the compute device; avoids host→device copies
    #                      when evaluating the same dataset many times.
    #                      Default for precision mode.
    #   '<device_str>'   — any explicit PyTorch device string ('mps', 'cuda:0').
    dataset_device='cpu',

    # ── Progress & Logging ─────────────────────────────────────────────────
    verbose=False,
    show_progress=True,

    # ── Mode-Specific Parameters ───────────────────────────────────────────
    # (These can also go in **analysis_kwargs)

    # rigorous mode:
    delta_threshold=0.1,             # Max curvature to accept as "linear"
    min_gamma_points=5,              # Min gamma values needed for reliable fit
    confidence_level=0.68,           # CI level (0.68 ≈ 1σ)

    # lag mode:
    lag_range=None,                  # range/list/np.ndarray; REQUIRED for lag mode

    # precision mode:
    tau_grid=None,                   # list of floats; corruption levels. REQUIRED.
    corrupt_target='x',              # 'x' | 'y' | 'both'
    corruption_method='rounding',    # 'rounding' | 'noise'
    n_noise_samples=50,
    threshold_ratio=0.9,             # MI fraction defining precision_tau

    # conditional mode:
    z_data=None,                     # Conditioning variable. REQUIRED for conditional.
    z_processor_type=None,
    z_processor_params=None,

    # transfer entropy mode:
    history_window=None,             # int samples in past. REQUIRED for transfer.
    prediction_horizon=1,            # int samples ahead to predict

    # evaluation:
    max_eval_samples=5000,           # Max samples used during test eval (memory)
    train_subset_size=None,          # Use subset of training data

    # optimizer / scheduler:
    optimizer='adam',                # 'adam'|'adamw'|'sgd'|'rmsprop'|'adagrad' or subclass
    optimizer_params={},             # dict; extra kwargs for optimizer (e.g. weight_decay)
    scheduler=None,                  # 'cosine'|'cosine_warmup'|'step'|'plateau' or class
    scheduler_params={},             # dict; extra kwargs for scheduler

    # regularisation (MLP only):
    dropout=0.0,                     # Dropout probability after each hidden layer
    norm_layer=None,                 # None | 'layer' | 'batch'

    # training diagnostics:
    eval_train=False,                # Per-epoch train MI tracking: False|True|float|int

    # permutation test (any mode):
    permutation_test=False,
    n_permutations=1,

    # **analysis_kwargs also accepted:
    # n_workers=1, n_splits=5, split_method='random', equalize_n=False, pairs=None
)
```

### Minimal Examples by Mode

```python
# estimate
result = nmi.run(x, y, mode='estimate')

# sweep (scan over embedding_dim)
result = nmi.run(x, y, mode='sweep',
                 sweep_grid={'embedding_dim': [32, 64, 128, 256]})

# rigorous
result = nmi.run(x, y, mode='rigorous')

# lag
result = nmi.run(x, y, mode='lag', lag_range=range(-20, 21))

# conditional
result = nmi.run(x, y, mode='conditional', z_data=z)

# transfer entropy (x → y)
result = nmi.run(x, y, mode='transfer', history_window=10)

# pairwise (all channel pairs in x)
result = nmi.run(x, mode='pairwise')

# pairwise cross (x channels vs y channels)
result = nmi.run(x, y, mode='pairwise')

# dimensionality
result = nmi.run(x, mode='dimensionality', analysis_kwargs={'n_splits': 10})
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
                 base_params={'n_epochs': 100, 'batch_size': 256},
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

**Note on `sweep_grid`:** Keys must match parameter names from `base_params` schema (see §8). Processor parameters like `window_size` can also be swept.

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

**`n_splits` kwarg (default 5):**
- *Intrinsic mode* (`split_method='random'`): number of distinct random channel-split assignments evaluated
- *Interaction mode* (y_data provided): number of independent model fits from different random weight initialisations — gives a proper mean and std in the aggregated output

**Key kwargs:** `n_workers=1`, `split_method='random'`, `n_splits=5`, `lag=<int>` (for temporal)

**Returns:** `Results` with:
- `result.dataframe` — columns: `mi_mean`, `mi_std`, `participation_ratio_mean`, `participation_ratio_std` (aggregated over splits/runs); `result.details['raw_results']` contains per-run rows with `split_id`
- `participation_ratio` — effective dimensionality from eigenvalue (covariance) spectrum: `(Σσᵢ²)² / Σσᵢ⁴` — stricter, weights large singular values more
- `participation_ratio_singular` — PR from singular-value spectrum: `(Σσᵢ)² / Σσᵢ²` — less strict variant

```python
# Intrinsic: MI between two random halves of x channels, 10 splits
result = nmi.run(x, mode='dimensionality',
                 base_params={'n_epochs': 100},
                 n_splits=10, split_method='random',
                 n_workers=4)
result.plot()

# Interaction: MI between x and y, 5 independent fits for mean/std
result = nmi.run(x, y, mode='dimensionality',
                 base_params={'n_epochs': 100},
                 n_splits=5,
                 n_workers=4)
print(result.dataframe[['mi_mean', 'mi_std', 'participation_ratio_mean']])
```

---

### 6.4 `rigorous` — Bias-Corrected Estimate

**What it does:** Implements the bias extrapolation procedure (§3.5). Trains models on `N/γ` data subsets for each γ in `gamma_range` (default 1–10), fits MI vs 1/γ, and extrapolates to γ→∞.

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
                 gamma_range=range(1, 15),
                 confidence_level=0.95,
                 n_workers=4)
print(f"MI = {result.mi_estimate:.3f} ± {result.details['mi_error']:.3f} bits")
result.plot()   # MI vs 1/gamma with fit line and extrapolation point
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
                 lag_range=range(-30, 31),    # ±30 sample lags
                 base_params={'n_epochs': 50},
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
                 processor_type_x='spike',
                 processor_params_x={'window_size': 0.05, 'n_seconds': 100.0},
                 tau_grid=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                 threshold_ratio=0.9)
print(f"Precision timescale: {result.details['precision_tau']*1000:.1f} ms")

# Multiple thresholds simultaneously
result = nmi.run(spike_x, y, mode='precision',
                 tau_grid=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                 threshold_ratio=[0.9, 0.75, 0.5])
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
                 z_data=z,
                 base_params={'n_epochs': 100},
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
- `bidirectional_te=False` — if `True`, also compute TE(Y→X) and return a directionality index. When `False`, a warning is logged recommending bidirectional evaluation to detect spurious causal claims.

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

  If `bidirectional_te=True`, additionally:
  - `te_yx` — TE(Y→X) point estimate
  - `i_yxpast_xfuture` — I(y_past, x_past ; x_future), the joint term for TE(Y→X)
  - `i_xpast_xfuture` — I(x_past ; x_future), the marginal term for TE(Y→X)
  - `raw_yxpast_xfuture`, `raw_xpast_xfuture` — per-run lists for the TE(Y→X) terms
  - `directionality_index` — `(TE_xy − TE_yx) / (|TE_xy| + |TE_yx|)`; +1 = pure X→Y, −1 = pure Y→X, 0 = symmetric

```python
# Unidirectional (default) — logs a warning to consider bidirectional
result = nmi.run(x, y, mode='transfer',
                 history_window=20,
                 prediction_horizon=1,
                 base_params={'n_epochs': 100},
                 n_workers=4)
print(f"TE(X→Y) = {result.mi_estimate:.3f} bits")

# Bidirectional — recommended for causal inference
result = nmi.run(x, y, mode='transfer',
                 history_window=20,
                 bidirectional_te=True,
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
- `result.dataframe` — columns: `ch_x`, `ch_y`, `mi_estimate`
- `result.details['mi_matrix']` — 2D numpy array; upper triangle filled for self-pairwise
- `result.details['n_channels']` — int (self) or tuple (cross)

```python
# Self-pairwise: MI between all neuron pairs
result = nmi.run(x, mode='pairwise',
                 base_params={'n_epochs': 50},
                 n_workers=8)
mi_matrix = result.details['mi_matrix']   # shape (n_channels, n_channels)

# Cross-pairwise: all (spike neuron) × (LFP channel) pairs
result = nmi.run(spike_x, lfp_y, mode='pairwise',
                 processor_type_x='spike', processor_type_y='continuous',
                 n_workers=8)
df = result.dataframe    # ch_x, ch_y, mi_estimate
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

Generates a mode-appropriate figure:

| Mode | Plot type |
|------|-----------|
| `estimate` | Test MI vs epoch; best epoch marked with vertical dashed line |
| `sweep` / `lag` | MI vs swept variable; multiple lines if multi-param |
| `dimensionality` | MI vs embedding_dim with participation ratio |
| `rigorous` | MI vs 1/γ with linear fit and extrapolated point |
| `precision` | MI vs τ with baseline and threshold lines |

```python
fig, ax = plt.subplots()
result.plot(ax=ax, title="My analysis", show=False)
```

#### `result.summary() → None`

Prints a human-readable summary to stdout. Includes mode, MI estimate, confidence intervals where applicable.

#### `Results.compare(results_list, labels=None, ax=None, **kwargs) → plt.Axes`

Static method for overlaying multiple results on one plot. Supported modes: `sweep`, `lag`, `dimensionality`, `rigorous`.

```python
r1 = nmi.run(x, y1, mode='lag', lag_range=range(-20, 21))
r2 = nmi.run(x, y2, mode='lag', lag_range=range(-20, 21))

Results.compare([r1, r2], labels=['Condition A', 'Condition B'])
```

---

## 8. Base Parameters Reference

Pass any of these in the `base_params` dict:

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

### Optimizer & Scheduler
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `optimizer` | str or class | `'adam'` | `'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'adagrad'`, or `torch.optim.Optimizer` subclass |
| `optimizer_params` | dict | `{}` | Extra kwargs for optimizer constructor (e.g. `{'weight_decay': 1e-4}`) |
| `scheduler` | str, class, or None | `None` | `'cosine'`, `'cosine_warmup'`, `'step'`, `'plateau'`, or `torch.optim.lr_scheduler` subclass |
| `scheduler_params` | dict | `{}` | Extra kwargs for scheduler constructor |

### Architecture
| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `embedding_dim` | int | 64 | Size of embedding vectors |
| `hidden_dim` | int | 64 | Hidden layer width |
| `n_layers` | int | 2 | Depth of embedding network |
| `embedding_model` | str | `'mlp'` | `'mlp'`, `'cnn'`, `'gru'`, `'lstm'`, `'tcn'`, `'transformer'` |
| `critic_type` | str | `'separable'` | `'separable'`, `'concat'`, `'hybrid'` |
| `kernel_size` | int | 3 | For CNN, TCN |
| `bidirectional` | bool | False | For GRU, LSTM |
| `nhead` | int | 4 | For Transformer |
| `shared_encoder` | bool | False | Share embedding weights between x and y |
| `dropout` | float | 0.0 | Dropout after each hidden layer (MLP only) |
| `norm_layer` | str or None | `None` | `'layer'` (LayerNorm) or `'batch'` (BatchNorm1d); MLP only |

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
| `spectral_whitening` | str or None | `'std'` | Standardize embedding dimensions |
| `spectral_mode` | str | `'none'` | `'none'`, `'summary'`, `'full'` |

### Variational Training
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `use_variational` | bool | False | Enable variational reparameterization for *any* embedding model. When `True`, `build_critic` wraps the selected encoder with `VariationalWrapper`, adding μ and log σ² heads. Works with all `embedding_model` choices: `'mlp'`, `'cnn'`, `'gru'`, `'lstm'`, `'tcn'`, `'transformer'`. |
| `beta` | float | 1024.0 | MI weight in variational loss `L = KL − β·MI`. Large β (≥ 1) makes MI maximization dominate; decrease for stronger KL regularization |

### Memory & Device Layout
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `device` | str or None | None | Compute device: `'cpu'`, `'cuda'`, `'mps'`, or `None` (auto-detect) |
| `dataset_device` | str or None | `'cpu'` | Where dataset tensors are stored. `'cpu'` (default) keeps data in pageable RAM so the OS can reclaim memory between sweep tasks. `'auto'` co-locates data with the compute device (precision mode default). Any explicit device string is also accepted. |

### Other
| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `output_units` | str | `'bits'` | `'bits'` or `'nats'` |
| `random_seed` | int or None | None | RNG seed for reproducibility; combine with `n_workers=1` for fully deterministic runs |
| `verbose` | bool | False | |
| `show_progress` | bool | True | Show tqdm progress bar during training |
| `return_embeddings` | bool | False | |
| `save_best_model_path` | str or None | None | |
| `max_eval_samples` | int | 5000 | Max samples used for eval (GPU memory) |
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

All embedding models take tensors of shape `(batch, n_channels, window_size)` and output `(batch, embedding_dim)`.

```python
from neural_mi.models import MLP, CNN1D, GRU, LSTM, TCN, Transformer
```

| Class | Key init params |
|-------|----------------|
| `MLP` | `input_dim, embedding_dim, hidden_dim, n_layers` |
| `CNN1D` | `input_dim, embedding_dim, hidden_dim, kernel_size` |
| `GRU` | `input_dim, embedding_dim, hidden_dim, n_layers, bidirectional` |
| `LSTM` | `input_dim, embedding_dim, hidden_dim, n_layers, bidirectional` |
| `TCN` | `input_dim, embedding_dim, hidden_dim, kernel_size` |
| `Transformer` | `input_dim, embedding_dim, nhead, n_layers` |

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

result = nmi.run(x, y, custom_embedding_cls=MyEmbedding,
                 base_params={'embedding_dim': 64})

# Or pass a fully-built critic:
critic = SeparableCritic(...)
result = nmi.run(x, y, custom_critic=critic)
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

### Rigorous Mode — 1/γ Space
The `rigorous` mode trains on subsets of size `N/γ`. The bias correction works in `1/γ` space (not `γ` space) because the bias is approximately linear in `1/N`, hence linear in `1/γ` when `N` is fixed. The functions `_find_linear_region` and `_extrapolate_mi` (in `analysis/rigorous.py`) operate in this space using the per-run `train_mi` as the dependent variable, consistent with every other mode.

### Pairwise Mode — Channel Naming
The output DataFrame uses columns `ch_x`, `ch_y`, `mi_estimate` (integer channel indices). The MI matrix is `result.details['mi_matrix']`:
- **Self-pairwise**: upper triangle (symmetric; diagonal = 0 by convention)
- **Cross-pairwise**: full `(n_ch_x, n_ch_y)` matrix

### Transfer Entropy vs. Conditional MI
| Feature | `transfer` mode | `conditional` mode |
|---------|-----------------|---------------------|
| Formula | TE(X→Y) = I(x_past,y_past;y_future) − I(y_past;y_future) | CMI = I(XZ;Y) − I(Z;Y) |
| History built by | Library (sliding windows) | User provides z_data |
| Input shape | 2D `(T, channels)` raw | 3D `(samples, channels, window)` pre-processed |
| Use case | Directed temporal coupling | Controlling for known confounds |

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
  dimensionality → result.dataframe [embedding_dim, train_mi, participation_ratio]
  rigorous     → result.mi_estimate ± result.details['mi_error']
  lag          → result.dataframe [lag, train_mi]
  precision    → result.dataframe [tau, train_mi]; result.details['precision_tau'], ['precision_thresholds']
  conditional  → result.mi_estimate  (I(X;Y|Z))
  transfer     → result.mi_estimate  (TE(X→Y)); bidirectional_te=True adds te_yx, directionality_index
  pairwise     → result.dataframe [ch_x, ch_y, mi_estimate]

Estimators: 'infonce' (default, has ceiling), 'smile' (no ceiling)
Embeddings:  'mlp' (default), 'cnn', 'gru', 'lstm', 'tcn', 'transformer'
Critics:     'separable' (default), 'concat', 'hybrid'
Units:       'bits' (default) or 'nats'

Processors:  'continuous' | 'spike' | 'categorical' | None (pre-processed)

Results methods:  .plot()  .summary()  Results.compare([r1, r2], labels=[...])
```
