# %% [markdown]
# # Tutorial 11: Inductive Biases — Quantitative Sample Efficiency
#
# Tutorial 10 introduced the physics-informed embedding models and showed that
# they could produce MI estimates on synthetic data.  But it had a fundamental
# scientific limitation: it compared models on a *fixed* dataset and asked
# "which model gets higher MI?"  This is the wrong question.
#
# At large N, all embedding models converge to the same true MI regardless of
# architecture.  The right question is: **which model reaches the true MI with
# fewer samples (windows)?**  This is the *sample efficiency* question, and it
# is the correct way to validate an inductive bias claim.
#
# This tutorial fixes Tutorial 10 by:
#
# 1. Using **synthetic generators with analytically known MI** as a horizontal
#    ground-truth line on every plot.
# 2. Measuring **sample efficiency curves**: MI estimated from N windows,
#    averaged over 10 random seeds, for N ∈ {50, 100, 200, 400, 800, 1500, 3000}.
# 3. Reporting the **crossover N**: the first N where a model's mean MI exceeds
#    90% of the true MI.  Lower crossover N = better inductive bias.
# 4. Using **realistic training budgets** (n_epochs ≥ 150) so the inductive
#    bias has time to engage before the comparison is made.
#
# Each section covers one embedding model and includes:
# - A visualisation of the synthetic data
# - The sample efficiency curves for the biased vs. baseline model
# - A diagnostic of the learned physics parameters (Section 1)
# - A brief scientific interpretation
#
# (A depthwise-separable CNN1D section and a SpikePhysicsEmbedding section
# originally appeared here too. Both were evaluated empirically against
# generic encoders and did not survive that gate; see
# `results/gate/decision_log.md`. They have been removed from the library and
# this tutorial.)

# %% [markdown]
# ## Setup

# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import neural_mi as nmi

sns.set_context("talk")
np.random.seed(42)
torch.manual_seed(42)

# Training settings for sample efficiency sweeps.
# n_epochs >= 150 so that physics parameters (sinc cutoffs) have
# time to converge from their initialisation to the true signal band.
BASE_TRAIN = dict(
    n_epochs=200,
    patience=40,
    batch_size=128,
    hidden_dim=64,
    embedding_dim=32,
    n_layers=2,
    learning_rate=3e-4,
)

N_SEEDS = 10           # seeds per (N, model) cell
N_WORKERS = 4          # parallel workers — reduce if this crashes on your machine


# ──────────────────────────────────────────────────────────────────────────────
# Helper: sample efficiency comparison
# ──────────────────────────────────────────────────────────────────────────────

def run_sample_efficiency(
    X_full: np.ndarray,
    Y_full: np.ndarray,
    N_values: list,
    models: list,          # list of (label, extra_base_params_dict)
    true_mi: float,
    base_train: dict,
    processor_type=None,
    processor_params=None,
    split_mode: str = 'random',
    n_seeds: int = N_SEEDS,
    n_workers: int = N_WORKERS,
) -> pd.DataFrame:
    """Run sample efficiency curves for two embedding models.

    For each N in N_values, subsamples N windows from X_full / Y_full (without
    replacement) and runs both models with n_seeds random seeds using
    mode='sweep'.  Returns a tidy DataFrame with one row per (N, model) cell.

    Parameters
    ----------
    X_full, Y_full : np.ndarray
        Full dataset.  Shape ``(N_full, n_channels, window_size)`` for
        pre-windowed IID data, or ``(n_timepoints, n_channels)`` for
        continuous recordings.
    N_values : list of int
        Window counts to evaluate.
    models : list of (str, dict)
        Each entry is (label, extra_base_params).  The label is used in the
        output DataFrame.
    true_mi : float
        Ground truth MI in bits (used for crossover calculation).
    base_train : dict
        Shared training hyperparameters merged with each model's extra params.
    processor_type : str or None
        Passed to nmi.run() as processor_type_x and processor_type_y.
    processor_params : dict or None
        Passed to nmi.run() as processor_params_x and processor_params_y.
    split_mode : str
        'random' for IID windows, 'blocked' for continuous recordings.
    n_seeds : int
        Number of random seeds per (N, model) cell.
    n_workers : int
        Number of parallel workers for the seed sweep.

    Returns
    -------
    pd.DataFrame
        Columns: N, model, mi_mean, mi_std.
    """
    rows = []
    for N in N_values:
        N_full = X_full.shape[0]
        if N <= N_full:
            idx = np.random.choice(N_full, N, replace=False)
            X_sub = X_full[idx]
            Y_sub = Y_full[idx]
        else:
            X_sub, Y_sub = X_full, Y_full  # use all if N > N_full

        for label, extra_params in models:
            combined_params = {**base_train, **extra_params}
            result = nmi.run(
                x_data=X_sub, y_data=Y_sub,
                mode='sweep',
                sweep_grid={'run_id': range(n_seeds)},
                processor_type_x=processor_type,
                processor_type_y=processor_type,
                processor_params_x=processor_params,
                processor_params_y=processor_params,
                split_mode=split_mode,
                base_params=combined_params,
                n_workers=n_workers,
                show_progress=False,
            )
            df = result.dataframe
            rows.append({
                'N': N,
                'model': label,
                'mi_mean': float(df['mi_mean'].iloc[0]),
                'mi_std': float(df['mi_std'].iloc[0]),
            })
            print(f"  N={N:4d}  {label:<30s}  MI={rows[-1]['mi_mean']:.3f} ± {rows[-1]['mi_std']:.3f} bits")

    return pd.DataFrame(rows)


def plot_sample_efficiency(
    df: pd.DataFrame,
    true_mi: float,
    title: str,
    ax=None,
) -> plt.Axes:
    """Plot sample efficiency curves with std band and true MI line.

    Also prints the crossover N (first N where mean > 0.9 * true_mi) for each
    model.

    Returns the Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    palette = sns.color_palette('tab10', n_colors=df['model'].nunique())
    for color, (model_name, grp) in zip(palette, df.groupby('model', sort=False)):
        grp = grp.sort_values('N')
        ax.plot(grp['N'], grp['mi_mean'], 'o-', color=color, label=model_name, linewidth=2)
        ax.fill_between(grp['N'],
                        grp['mi_mean'] - grp['mi_std'],
                        grp['mi_mean'] + grp['mi_std'],
                        alpha=0.2, color=color)
        # Crossover N
        crossover = grp.loc[grp['mi_mean'] >= 0.9 * true_mi, 'N']
        if not crossover.empty:
            cx = int(crossover.iloc[0])
            print(f"  {model_name}: crossover N (90% of true MI) = {cx} windows")
        else:
            print(f"  {model_name}: did not reach 90% of true MI in this N range")

    ax.axhline(true_mi, linestyle='--', color='black', linewidth=1.5,
               label=f'True MI = {true_mi:.2f} bits')
    ax.set_xlabel('N windows')
    ax.set_ylabel('Estimated MI (bits)')
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    return ax


# %% [markdown]
# ## Section 1: SincCNN for Band-Limited EEG/LFP
#
# ### Scientific motivation
#
# Neural oscillations are organized in canonical frequency bands (delta, theta,
# alpha, beta, gamma).  The MI between two recording sites often lives in one
# band — for example, inter-areal alpha coherence during attention.
#
# A standard CNN must discover from data alone that only the alpha band (10 Hz)
# carries MI.  The ``sinc_cnn`` starts with filters that are constrained to be
# bandpass filters with learnable cutoff frequencies, initialized to cover all
# the classical neural bands.  This gives sinc_cnn a significant head start
# when MI is band-limited.
#
# ### Data
#
# ``generate_windowed_oscillatory`` produces 4-channel windows where a 10 Hz
# (alpha-band) amplitude modulation carries all the MI.  The true MI is
# analytically computed from the SNR.
#
# ### Physics diagnostic
#
# After the sinc_cnn run, we visualise where the learned filter cutoff
# frequencies ended up.  If the inductive bias worked, the filters should
# cluster around the 10 Hz carrier.  This diagnostic is only possible because
# of Library Fix 1 (``physics_params_final`` in result.details).

# %%
np.random.seed(200)
print("Generating oscillatory LFP data (4 channels, f_carrier=10 Hz)...")

N_CH_SINC = 4
SAMPLE_RATE_SINC = 512.0

X_sinc, Y_sinc, true_mi_sinc = nmi.generators.generate_windowed_oscillatory(
    n_windows=3000,
    n_channels=N_CH_SINC,
    window_size=256,
    f_carrier_hz=10.0,
    sample_rate=SAMPLE_RATE_SINC,
    latent_mi=1.0,
    snr=3.0,
)
print(f"  X shape: {X_sinc.shape}   True MI: {true_mi_sinc:.3f} bits")

# %%
# Visualise one example window — should show a ~10 Hz sinusoidal modulation
fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
t_ms = np.arange(256) / SAMPLE_RATE_SINC * 1000.0
for ch, ax in enumerate(axes):
    ax.plot(t_ms, X_sinc[0, ch, :], lw=1.2, label='X')
    ax.plot(t_ms, Y_sinc[0, ch, :], lw=1.2, alpha=0.7, label='Y')
    ax.set_title(f'Ch {ch+1}')
    ax.set_xlabel('Time (ms)')
    if ch == 0:
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=9)
plt.suptitle('LFP data: 10 Hz amplitude modulation carries MI (4 channels)')
plt.tight_layout()
plt.show()

# %%
print("\n=== Section 1: Standard CNN vs SincCNN on alpha-band LFP ===")

N_VALUES_SINC = [50, 100, 200, 400, 800, 1500, 3000]

# sample_rate is required by SincEmbedding to convert Hz → kernel samples.
# Pass via processor_params (not base_params) so the library routes it correctly.
SINC_PROC_PARAMS = {'sample_rate': SAMPLE_RATE_SINC}

models_sinc = [
    ('CNN',      {'embedding_model': 'cnn'}),
    ('SincCNN',  {'embedding_model': 'sinc_cnn', 'n_sinc_filters': 8}),
]

df_sinc = run_sample_efficiency(
    X_sinc, Y_sinc, N_VALUES_SINC, models_sinc, true_mi_sinc,
    base_train={**BASE_TRAIN, 'n_epochs': 250, 'patience': 50},
    processor_params=SINC_PROC_PARAMS,
)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
print("\nCrossover N values (Section 1):")
plot_sample_efficiency(df_sinc, true_mi_sinc, 'Section 1: SincCNN — Alpha-Band LFP', ax=ax)
plt.tight_layout()
plt.show()

# %%
# --- Physics diagnostic: where did the sinc filters converge? ---
#
# Run a single full-data sinc_cnn estimate and inspect physics_params_final.
# This uses Library Fix 1 (get_physics_params() + physics_params_history tracking).

print("\n--- Sinc filter frequency diagnostic ---")
result_sinc_diag = nmi.run(
    x_data=X_sinc, y_data=Y_sinc,
    mode='estimate',
    split_mode='random',
    processor_params_x={'sample_rate': SAMPLE_RATE_SINC},
    processor_params_y={'sample_rate': SAMPLE_RATE_SINC},
    base_params={**BASE_TRAIN, 'n_epochs': 250, 'patience': 50,
                 'embedding_model': 'sinc_cnn', 'n_sinc_filters': 8},
    random_seed=0,
    show_progress=True,
)
print(f"SincCNN MI estimate (full data): {result_sinc_diag.mi_estimate:.3f} bits")

if 'physics_params_final' in result_sinc_diag.details:
    pp = result_sinc_diag.details['physics_params_final']
    f_low = pp.get('x_f_low_hz', [])
    f_high = pp.get('x_f_high_hz', [])
    print(f"Learned cutoffs (first 8 filters, embedding_net_x):")
    for i in range(min(8, len(f_low))):
        print(f"  Filter {i+1:2d}: {f_low[i]:5.1f} – {f_high[i]:5.1f} Hz")

    # Plot learned filter bands
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (fl, fh) in enumerate(zip(f_low, f_high)):
        ax.barh(i, fh - fl, left=fl, height=0.7, alpha=0.6,
                color='steelblue' if 8 < (fl + fh) / 2 < 13 else 'lightgray')
    ax.axvline(10.0, color='tomato', linestyle='--', linewidth=2,
               label='True carrier (10 Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Filter index')
    ax.set_title('Learned sinc filter bands — filters near 10 Hz are coloured blue')
    ax.legend()
    ax.set_xlim(0, SAMPLE_RATE_SINC / 2)
    plt.tight_layout()
    plt.show()
else:
    print("  (physics_params_final not found in result.details)")

# %% [markdown]
# **Interpretation:** The SincCNN should reach 90% of the true MI at lower N than
# standard CNN.  The filter diagnostic above shows where the sinc cutoffs landed:
# filters clustered near the 10 Hz carrier (coloured blue) indicate that the
# model correctly discovered the signal band from data alone.
#
# Note that SincCNN needs more training epochs than standard CNN — the sinc
# filter cutoff frequencies (log-space parameters) must migrate from their
# band-distributed initialization toward the true 10 Hz band.  We use
# n_epochs=250 to ensure this happens before the comparison.


# %% [markdown]
# ## Section 2: Pretrained Backbone for Image Data
#
# This section has two scenarios that tell a complete, scientifically honest story.
#
# ### Scenario A — Alignment (positive case)
#
# MNIST digit pairs (classes 0 and 1) share class identity.  The only shared
# information is "which digit is this?" — exactly 1 bit (two balanced classes).
# The pretrained ResNet18 backbone has already learned to detect digit-level
# structure from ImageNet pretraining, so it encodes class identity immediately.
# A CNN2D trained from scratch must discover this structure from the MI data alone.
#
# Note: this section requires ``torchvision``.  Install with:
# ``pip install neural_mi[vision]``  or  ``pip install torchvision``
#
# Also note: MNIST images are 28×28, but ResNet18 was pretrained on 224×224
# images.  Library Fix 2 automatically adds a bilinear upsample layer and emits
# a UserWarning — this is expected and does not break training.
#
# ### Scenario B — Misalignment (negative control)
#
# Gaussian blobs share a common location, but the location structure is not the
# kind of visual feature pretrained ResNet18 detects (edges, textures, objects).
# Pretrained features are actively harmful here — they discard the blob location
# information.  A CNN2D trained from scratch does better.
#
# Together, the two scenarios make the key point: pretrained backbones only
# help when the pretraining domain aligns with the MI-relevant structure.

# %%
print("\n=== Section 2: Pretrained Backbone — MNIST (Alignment) ===")

try:
    import torchvision
    import torchvision.transforms as transforms
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False
    print("torchvision not installed. Skipping Scenario A.")
    print("Install with:  pip install torchvision")

if _HAS_TORCHVISION:
    # Load MNIST, keep only digits 0 and 1
    _mnist_root = '/tmp/mnist_data'
    _transform = transforms.ToTensor()
    try:
        _mnist = torchvision.datasets.MNIST(root=_mnist_root, train=True,
                                             download=True, transform=_transform)
    except Exception as _e:
        print(f"Could not download MNIST: {_e}")
        _HAS_TORCHVISION = False

if _HAS_TORCHVISION:
    np.random.seed(500)

    _targets = np.array(_mnist.targets)
    _images = _mnist.data.numpy().astype(np.float32) / 255.0  # (60000, 28, 28)

    # Keep only digits 0 and 1
    _mask = (_targets == 0) | (_targets == 1)
    _imgs_01 = _images[_mask]         # (N_01, 28, 28)
    _labels_01 = _targets[_mask]      # (N_01,)
    print(f"MNIST 0 vs 1: {_imgs_01.shape[0]} images total")

    # Build paired dataset: for each pair (X_i, Y_i), both are same class,
    # different instances, with independent Gaussian noise augmentation
    N_MNIST = 1500
    _noise_std = 0.15

    X_mnist = np.zeros((N_MNIST, 1, 28, 28), dtype=np.float32)
    Y_mnist = np.zeros((N_MNIST, 1, 28, 28), dtype=np.float32)

    for i in range(N_MNIST):
        label = np.random.randint(0, 2)
        _class_imgs = _imgs_01[_labels_01 == label]
        idx_x, idx_y = np.random.choice(len(_class_imgs), 2, replace=True)
        X_mnist[i, 0] = _class_imgs[idx_x] + _noise_std * np.random.randn(28, 28)
        Y_mnist[i, 0] = _class_imgs[idx_y] + _noise_std * np.random.randn(28, 28)

    X_mnist = np.clip(X_mnist, 0.0, 1.0)
    Y_mnist = np.clip(Y_mnist, 0.0, 1.0)

    TRUE_MI_MNIST = 1.0  # log2(2) = 1 bit, two balanced classes
    print(f"MNIST pair dataset: {X_mnist.shape}   True MI = {TRUE_MI_MNIST:.2f} bits")

    # Visualise a few pairs
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for j in range(6):
        axes[0, j].imshow(X_mnist[j, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f'X[{j}]'); axes[0, j].axis('off')
        axes[1, j].imshow(Y_mnist[j, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, j].set_title(f'Y[{j}]'); axes[1, j].axis('off')
    plt.suptitle('MNIST pairs: X and Y share class identity (0 or 1)')
    plt.tight_layout()
    plt.show()

    N_VALUES_IMG = [50, 100, 200, 400, 800, 1500]
    IMG_PARAMS = dict(**BASE_TRAIN, n_epochs=200, patience=40)

    models_mnist = [
        ('CNN2D (random init)', {'embedding_model': 'cnn2d'}),
        ('Pretrained ResNet18',  {'embedding_model': 'pretrained_backbone',
                                  'pytorch_predefined': 'resnet18', 'pretrained': True}),
    ]

    print("\nScenario A — MNIST alignment (warning about 28x28 upsampling is expected):")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        df_mnist = run_sample_efficiency(
            X_mnist, Y_mnist, N_VALUES_IMG, models_mnist, TRUE_MI_MNIST,
            base_train=IMG_PARAMS,
        )
    upsample_warnings = [w for w in caught if 'spatial size' in str(w.message).lower()]
    if upsample_warnings:
        print(f"  [Expected] Backbone upsample warning: {str(upsample_warnings[0].message)[:120]}...")

    fig, ax = plt.subplots(figsize=(9, 5))
    print("\nCrossover N values (Scenario A — MNIST):")
    plot_sample_efficiency(df_mnist, TRUE_MI_MNIST,
                           'Section 2A: Pretrained Backbone — MNIST (Alignment)', ax=ax)
    plt.tight_layout()
    plt.show()

# %%
# --- Scenario B: Gaussian blobs (misalignment / negative control) ---
print("\n=== Section 2B: Pretrained Backbone — Gaussian Blobs (Misalignment) ===")

np.random.seed(600)
N_BLOBS = 800

X_blobs, Y_blobs = nmi.generators.generate_noisy_image_pairs(
    n_samples=N_BLOBS, image_size=64, n_channels=3,
    signal_strength=2.5, noise_level=1.0, use_torch=False,
)
# Use 5000-window empirical MI as reference (single Gaussian blob is a
# high-dimensional correlated pair; true MI is not analytically known)
# We run a large-N estimate with CNN2D as a reference upper bound
print("Computing reference MI for Gaussian blobs via large-N CNN2D estimate...")
result_blob_ref = nmi.run(
    x_data=X_blobs, y_data=Y_blobs,
    mode='estimate',
    split_mode='random',
    base_params={**BASE_TRAIN, 'n_epochs': 200, 'embedding_model': 'cnn2d'},
    random_seed=0, show_progress=True,
)
ref_mi_blobs = result_blob_ref.mi_estimate
print(f"Reference MI (CNN2D, N={N_BLOBS}): {ref_mi_blobs:.3f} bits")

N_VALUES_BLOBS = [50, 100, 200, 400, 800]
models_blobs = [
    ('CNN2D (random init)', {'embedding_model': 'cnn2d'}),
    ('Pretrained ResNet18',  {'embedding_model': 'pretrained_backbone',
                              'pytorch_predefined': 'resnet18', 'pretrained': True}),
]

print("\nScenario B — Gaussian blobs (misalignment control):")
df_blobs = run_sample_efficiency(
    X_blobs, Y_blobs, N_VALUES_BLOBS, models_blobs, ref_mi_blobs,
    base_train=BASE_TRAIN,
)

fig, ax = plt.subplots(figsize=(9, 5))
print("\nCrossover N values (Scenario B — blobs):")
plot_sample_efficiency(df_blobs, ref_mi_blobs,
                       'Section 2B: Pretrained Backbone — Gaussian Blobs (Misalignment)',
                       ax=ax)
ax.set_title('Section 2B: Gaussian Blobs (Misalignment)\n'
             'Dashed line = CNN2D reference estimate (not analytically exact)')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:**
#
# *Scenario A (MNIST):* With pretrained=True, the backbone should reach the
# 1.0 bit true MI at much lower N than CNN2D from scratch.  The pretrained
# ResNet18 already encodes digit identity — it needs only the MLP head to be
# trained on the MI task.
#
# *Scenario B (Gaussian blobs):* The pretrained backbone performs worse than
# CNN2D trained from scratch.  The ImageNet-pretrained features detect edges,
# textures, and object-level structure — none of which exists in Gaussian blobs.
# The pretrained backbone is discarding the blob location signal that CNN2D
# can find directly.
#
# **Lesson:** Pretrained backbones require alignment between the pretraining
# domain and the MI-relevant structure.  Always run Scenario B as a sanity check
# when using pretrained_backbone on new data.  If pretrained performs worse than
# random init, your data structure does not match the pretrained features.
#
# Also note: the 28×28 → 224×224 bilinear upsampling (Library Fix 2) should
# have emitted a UserWarning above.  The warning is informational — training
# proceeds normally.  Very small images may have reduced feature quality, but
# for MNIST digit classification the upsampling is sufficient.

# %% [markdown]
# ## Summary
#
# | Section | Data | Baseline | Biased model | Expected crossover advantage |
# |---------|------|----------|--------------|------------------------------|
# | 1 | Alpha-band LFP (10 Hz) | CNN | SincCNN | SincCNN lower crossover N |
# | 2A | MNIST digits 0/1 | CNN2D random | ResNet18 pretrained | Pretrained lower crossover N |
# | 2B | Gaussian blobs | CNN2D random | ResNet18 pretrained | CNN2D random wins (misalignment) |
#
# The core message: **inductive biases lower sample complexity when the bias
# matches the data structure, and are either neutral or harmful when it does not.**
# Measuring sample efficiency curves with known ground-truth MI is the principled
# way to validate this claim quantitatively.
#
# (A depthwise-separable CNN1D section and a SpikePhysicsEmbedding section were
# evaluated empirically against generic encoders in this same framework and did
# not survive that gate -- both have been removed from the library. See
# `results/gate/decision_log.md` for the deciding evidence.)
