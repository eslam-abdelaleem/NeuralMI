# %% [markdown]
# # Tutorial 10: Inductive Biases — Physics-Informed Embedding Models
#
# The eight previous tutorials treated the embedding model as a black box
# whose only role is to map windowed data to a fixed-size vector.  This
# tutorial opens that box.
#
# Every embedding model encodes assumptions about what structure in the input
# is likely to carry the mutual information.  A generic MLP assumes nothing —
# it treats every element of the flattened window equally.  A CNN assumes that
# local temporal patterns matter.  But neural data comes with *much stronger*
# prior knowledge: EEG activity is organized in frequency bands, raw spike
# timestamps encode firing rate in the count of non-padding values, and
# images are better described by learned hierarchical visual features than
# by random filters.
#
# **Inductive bias** is the term for this prior knowledge baked into the model
# architecture.  When the bias is correct, the model learns faster, generalises
# better, and can extract MI from smaller datasets.  When it is wrong, the bias
# does not help — but in most cases it causes no harm either, because the
# downstream MLP head can compensate.
#
# NeuralMI ships physics-informed embedding models for two data types:
#
# | ``embedding_model`` | Data type | Bias |
# |---------------------|-----------|------|
# | ``'sinc_cnn'`` | EEG / LFP (continuous) | Learnable sinc bandpass filters initialized to neural frequency bands |
# | ``'pretrained_backbone'`` | Images | Frozen torchvision backbone (e.g. ResNet18) + trainable MLP head |
#
# This tutorial walks through each model in turn, explains what it does, when
# to use it, and demonstrates it on matching synthetic data.
#
# (Two other candidates -- a depthwise-separable first layer for
# ``embedding_model='cnn'``, and a ``'spike_physics'`` embedding for raw spike
# timestamps -- were evaluated empirically against generic encoders and did
# not survive that gate; see `results/gate/decision_log.md`. They have been
# removed from the library.)

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import neural_mi as nmi

sns.set_context("talk")
np.random.seed(42)
torch.manual_seed(42)

# Shared training settings — kept light so the tutorial runs quickly.
# In a real analysis you would use n_epochs=150+ and patience=30.
FAST_PARAMS = dict(n_epochs=60, patience=20, batch_size=128,
                   hidden_dim=64, embedding_dim=32, n_layers=2)

# SincEmbedding needs more gradient steps than CNN to tune its filter
# frequencies from their initialization to the actual signal band.
LFP_PARAMS = dict(**FAST_PARAMS, n_epochs=250, patience=50)

# GRU on the timing-code scenario needs more steps to detect the
# precise spike-timing correlation buried in background noise.
TIMING_PARAMS = dict(**FAST_PARAMS, n_epochs=120, patience=30)

# %% [markdown]
# ## 1. SincEmbedding for EEG / LFP Data
#
# ### The inductive bias
#
# Neural oscillations are organised in canonical frequency bands:
# delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz),
# gamma (30–100 Hz).  Much of the information in EEG and LFP is carried in
# one of these bands — for example, working memory is linked to alpha/beta
# power, hippocampal-cortical communication to theta.
#
# ``embedding_model='sinc_cnn'`` builds this knowledge into the first layer.
# Instead of arbitrary convolution kernels, it uses ``n_sinc_filters``
# **learnable sinc bandpass filters** per channel, parameterized by learnable
# lower and upper cutoff frequencies ``f_low`` and ``f_high`` (in Hz).  They
# are initialized to cover the classical neural bands.
#
# The full architecture:
#
# 1. **Sinc filter layer (learnable)** — per-channel, initialized to EEG bands.
#    Produces ``n_channels × n_sinc_filters`` feature maps.
# 2. **Convolutional body** — standard CNN1D body operating on filtered features.
# 3. **Global average pool + MLP head** → ``embed_dim``.
#
# ### Required parameters
#
# ``sinc_cnn`` **requires** ``sample_rate`` to convert filter cutoffs in Hz to
# kernel coefficients in samples.  Pass it via ``processor_params``.

# %%
# Generate LFP-like data with MI in the alpha band (8–13 Hz).
np.random.seed(1)
SAMPLE_RATE_LFP = 500.0   # Hz
N_CH_LFP = 4

x_lfp, y_lfp = nmi.generators.generate_oscillatory_lfp(
    n_timepoints=20_000,
    n_channels=N_CH_LFP,
    sample_rate=SAMPLE_RATE_LFP,
    coupling_band_hz=(8.0, 13.0),  # alpha band carries the MI
    snr=3.0,
)

proc_lfp = dict(window_size=0.2, step_size=0.1, sample_rate=SAMPLE_RATE_LFP)

print("--- Standard CNN vs SincEmbedding on alpha-band LFP ---")

result_cnn_lfp = nmi.run(
    x_data=x_lfp, y_data=y_lfp,
    mode='estimate',
    processor_type_x='continuous', processor_params_x=proc_lfp,
    processor_type_y='continuous', processor_params_y=proc_lfp,
    base_params={**LFP_PARAMS, 'embedding_model': 'cnn'},
    random_seed=1, show_progress=False,
)

result_sinc = nmi.run(
    x_data=x_lfp, y_data=y_lfp,
    mode='estimate',
    processor_type_x='continuous', processor_params_x=proc_lfp,
    processor_type_y='continuous', processor_params_y=proc_lfp,
    base_params={**LFP_PARAMS, 'embedding_model': 'sinc_cnn',
                 'n_sinc_filters': 8},
    random_seed=1, show_progress=False,
)

print(f"  Standard CNN:   {result_cnn_lfp.mi_estimate:.3f} bits")
print(f"  SincEmbedding:  {result_sinc.mi_estimate:.3f} bits")

# %% [markdown]
# SincEmbedding should outperform standard CNN when MI is truly band-limited,
# because the first layer is already initialized to isolate the relevant
# frequencies.  However, the sinc filters are still trainable — their
# band-centers must shift from their initialization to the exact signal band.
# This requires more gradient steps than CNN, which is why we use ``LFP_PARAMS``
# (250 epochs) for this comparison rather than ``FAST_PARAMS`` (60 epochs).
# With 60 epochs, the sinc filters may not have converged yet and CNN can
# appear comparable or better despite the inductive bias disadvantage.
#
# **Key hyperparameters:**
#
# - ``n_sinc_filters`` (default 8): number of bandpass filters per channel.
#   More filters cover more sub-bands but increase the width of the next layer.
#   For standard EEG analysis, 4–8 filters are usually sufficient.
# - ``sample_rate``: must match the sampling rate in ``processor_params``.
#
# **When to use ``sinc_cnn``:**
# - EEG, LFP, or any voltage signal where you expect MI in a specific frequency band.
# - When you have domain knowledge about which bands carry the relevant signal.
# - As a first step: run with ``sinc_cnn`` and inspect whether the learned
#   ``f_low`` / ``f_high`` parameters converge to physiologically meaningful values.
#
# **When to skip:**
# - Broadband signals with no obvious band structure (e.g., raw spike timestamps).
# - Very short windows (< 50 ms) where fewer than one oscillation cycle fits.

# %% [markdown]
# ### Accessing the learned filter frequencies
#
# Because the filter cutoffs are trainable parameters, you can inspect them
# after training to check which frequency bands the model found informative.

# %%
# Access the embedding model from the trained critic
critic = result_sinc.details.get('critic')
if critic is not None:
    # The embedding net is the SincEmbedding object
    emb = critic.embedding_net_x
    f_low_hz  = torch.exp(emb.log_f_low).detach().numpy()
    f_high_hz = torch.maximum(emb.log_f_low.exp() + 0.5,
                              emb.log_f_high.exp()).detach().numpy()
    print("Learned filter bands (first 8 filters, first channel):")
    for i in range(min(8, len(f_low_hz))):
        print(f"  Filter {i+1:2d}: {f_low_hz[i]:5.1f} – {f_high_hz[i]:5.1f} Hz")
else:
    print("(critic not stored in details — set return_embeddings=True or inspect manually)")

# %% [markdown]
# ## 2. PretrainedBackboneEmbedding for Image Data
#
# ### The inductive bias
#
# When the input is an image (2D spatial data), a randomly initialized
# convolutional network must learn to detect edges, textures, and objects from
# scratch.  If your dataset has only a few thousand images, this is often
# impossible.
#
# ``embedding_model='pretrained_backbone'`` solves this by loading a
# **pretrained torchvision model** (ResNet, VGG, EfficientNet, …) and using
# it as a frozen feature extractor.  Only a small trainable MLP head maps
# the backbone's feature vector to ``embed_dim``.  The backbone has already
# learned to detect the structures that matter in natural images, so the MLP
# only needs to learn which of those features carry the MI.
#
# **Important:** the backbone is always frozen (``requires_grad=False``).
# The ``pretrained=True`` flag loads ImageNet weights; ``pretrained=False``
# uses random backbone weights (useful to ablate the effect of pretraining).
#
# Input data must be shape ``(n_samples, n_channels, height, width)`` — the
# same format used by ``embedding_model='cnn2d'``.  Pass it directly to
# ``nmi.run`` with ``processor_type=None`` (no windowing).
#
# **Requires** ``torchvision`` (``pip install torchvision``).

# %%
np.random.seed(5)

# Generate image pairs where X and Y share a Gaussian blob (same position,
# different noise realizations).  This is the simplest testable visual MI.
N_IMG = 500
x_img, y_img = nmi.generators.generate_noisy_image_pairs(
    n_samples=N_IMG, image_size=64, n_channels=3,
    signal_strength=2.5, noise_level=1.0,
)
# Shape: (N, 3, 64, 64) — ready for pretrained_backbone or cnn2d.
print(f"Image shapes: x={tuple(x_img.shape)}, y={tuple(y_img.shape)}")

# %%
print("--- CNN2D (random init) vs PretrainedBackbone (ResNet18, no pretraining) ---")

result_cnn2d = nmi.run(
    x_data=x_img, y_data=y_img,
    mode='estimate',
    split_mode='random',
    base_params={**FAST_PARAMS, 'embedding_model': 'cnn2d'},
    random_seed=5, show_progress=False,
)

# pretrained=False for a fair comparison (both start from random weights,
# but pretrained_backbone uses ResNet18's architecture + frozen weights).
# Switch to pretrained=True to see the full power of transfer learning.
result_backbone = nmi.run(
    x_data=x_img, y_data=y_img,
    mode='estimate',
    split_mode='random',
    base_params={**FAST_PARAMS, 'embedding_model': 'pretrained_backbone',
                 'pytorch_predefined': 'resnet18', 'pretrained': False},
    random_seed=5, show_progress=False,
)

print(f"  CNN2D (random init):             {result_cnn2d.mi_estimate:.3f} bits")
print(f"  PretrainedBackbone (ResNet18):   {result_backbone.mi_estimate:.3f} bits")

# %% [markdown]
# **Key parameters:**
#
# - ``pytorch_predefined`` (str): any ``torchvision.models`` model name.
#   Case-insensitive.  Examples: ``'resnet18'``, ``'resnet50'``, ``'vgg16'``,
#   ``'efficientnet_b0'``.  Defaults to ``'resnet18'``.
# - ``pretrained`` (bool): whether to load ImageNet weights.  Defaults to
#   ``False``.  Set to ``True`` when your images are natural images or contain
#   structures that overlap with ImageNet categories.
#
# **When to use ``pretrained_backbone``:**
# - Image data (2D spatial inputs).
# - Small datasets where training a CNN from scratch is infeasible.
# - When the relevant structure resembles natural image features (edges,
#   textures, spatial patterns).
#
# **When to skip:**
# - Non-image data (time series, spikes, …).
# - Highly domain-specific images with no overlap with natural images
#   (e.g., raw microscopy data with unusual statistics).  In that case,
#   consider starting from random weights (``pretrained=False``) and using
#   a larger dataset.

# %% [markdown]
# ## 3. Quick Reference: Choosing the Right Model
#
# | Data type | Recommended model | Notes |
# |-----------|------------------|-------|
# | Binned spike counts | ``'mlp'`` or ``'gru'`` | Short windows: MLP. Temporal order matters: GRU. |
# | LFP / EEG (single channel) | ``'mlp'`` or ``'cnn'`` | Baseline. |
# | LFP / EEG (band-limited MI) | ``'sinc_cnn'`` | Pass ``sample_rate``. Start with 8 filters. |
# | Images | ``'pretrained_backbone'`` | Requires torchvision. Use ``pretrained=True`` for natural images. |
# | General / unknown | ``'mlp'`` first, then sweep | Always start with MLP as the baseline. |
#
# The general rule: **start with the MLP**.  Switch to an inductive-bias model
# only when (a) you have domain knowledge about the data structure, and (b) the
# MLP is failing to capture it.  Inductive biases are most valuable on limited
# data; on large datasets, generic models often converge to the same result.

# %% [markdown]
# ## Summary
#
# NeuralMI supports two physics-informed embedding choices:
#
# 1. **``embedding_model='sinc_cnn'``**: learnable sinc bandpass filters
#    initialized to EEG/LFP frequency bands.  Requires ``sample_rate`` in
#    ``processor_params``.  Use when MI is band-limited.
#
# 2. **``embedding_model='pretrained_backbone'``**: frozen torchvision backbone
#    + trainable MLP head.  Use for image data, especially with small datasets.
#    Specify the backbone with ``pytorch_predefined`` (e.g., ``'resnet18'``).
#
# Both models are fully compatible with ``mode='sweep'``, ``mode='rigorous'``,
# variational wrappers, custom critics, and the full NeuralMI pipeline.
#
# (A depthwise-separable CNN1D variant and a ``'spike_physics'`` embedding for
# raw spike timestamps were also evaluated but did not survive an empirical
# gate against generic encoders and have been removed; see
# `results/gate/decision_log.md` for the deciding evidence.)
