#!/usr/bin/env python
# coding: utf-8

# # Example 3: Finding Hidden Signals (Latent Dimensionality)
# 
# So far, we've focused on the relationship *between* two variables, X and Y. But what if we want to understand the internal complexity of a *single* high-dimensional dataset? For example, how many independent signals are present in a recording from 100 neurons?
# 
# This is the problem of estimating **latent dimensionality**. This notebook demonstrates how to use `NeuralMI` to do just that.
# 
# **Goal:**
# 1.  Introduce the concept of "Internal Information" ($I(X_A; X_B)$).
# 2.  Use `run(mode='dimensionality')` to automate this analysis.
# 3.  Generate data with a known `latent_dim` and see if we can recover it.

# ## 1. Imports

# In[1]:


import os
import tempfile

# Create a custom temp directory
custom_temp = os.path.expanduser('~/tmp_neural_mi')
os.makedirs(custom_temp, exist_ok=True)
os.environ['TMPDIR'] = custom_temp
os.environ['TEMP'] = custom_temp
os.environ['TMP'] = custom_temp
tempfile.tempdir = custom_temp

import neural_mi as nmi
import torch
import numpy as np
import neural_mi as nmi
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ## 2. The Concept: Internal Information
# 
# To measure the internal complexity of a dataset `X` (e.g., shape `[n_samples, n_neurons, n_features]`), we can't just compute $I(X;X)$, as that would be infinite. Instead, we do the following:
# 
# 1.  Randomly split the channels (neurons) of `X` into two non-overlapping halves, $X_A$ and $X_B$.
# 2.  Calculate the mutual information between these halves: $I(X_A; X_B)$.
# 3.  Repeat this process for many different random splits and average the results for robustness.
# 
# This **Internal Information** tells us how much redundancy or shared information exists within the channels of `X`. If the neurons are all independent, this value will be zero. If they are highly coordinated (driven by a shared latent signal), this value will be high.

# ## 3. Generating Data with a Known Latent Structure
# 
# We'll use `generate_nonlinear_from_latent`. This function is perfect for our goal. It will:
# 1.  Create a simple, low-dimensional latent signal (e.g., 4-dimensional).
# 2.  Use a nonlinear neural network to "project" this signal up to a high-dimensional observation (e.g., 50 dimensions, representing 50 neurons).
# 
# Our goal is to see if the library can analyze the 50-dimensional data and correctly deduce that the underlying, hidden dimensionality is 4.

# In[2]:


# --- Dataset Parameters ---
n_samples = 10000
true_latent_dim = 4      # The ground truth we want to recover
observed_dim = 50        # The number of "neurons" we observe
mi_between_latents = 3.0 # Strength of correlation in the latent space

# --- Generate Raw 2D Data ---
# We only need a single variable 'x_data' for this analysis.
# Shape: [n_samples, observed_dim]
x_raw, _ = nmi.datasets.generate_nonlinear_from_latent(
    n_samples=n_samples, 
    latent_dim=true_latent_dim,
    observed_dim=observed_dim,
    mi=mi_between_latents
)

print(f"Generated raw X data shape: {x_raw.shape}")


# ## 4. Running the Dimensionality Analysis
# 
# The key idea is to see how the Internal Information changes as we vary the capacity of our MI estimation model. We do this by sweeping over the `embedding_dim` parameter.
# 
# We expect the MI to increase as `embedding_dim` increases, but it should **plateau or saturate** once `embedding_dim` is large enough to capture the true latent dimensionality of the data. The location of this "elbow" is our estimate.
# 
# `run(mode='dimensionality')` automates this entire process. We pass the raw data and tell the processor to treat each row as an independent sample (`window_size=1`).

# In[3]:


# Base parameters for the trainer
base_params = {
    'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 5, 'hidden_dim': 128, 'n_layers': 3
}

# The sweep_grid MUST contain 'embedding_dim' for this mode
sweep_grid = {
    'embedding_dim': [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
}

# Run the analysis directly on the raw, 2D data
dim_results = nmi.run(
    x_data=x_raw,
    # y_data is not needed for this mode
    mode='dimensionality',
    processor_type='continuous',
    processor_params={'window_size': 1},
    base_params=base_params,
    sweep_grid=sweep_grid,
    # n_splits controls how many random channel splits to average over
    n_splits=5,
    n_workers=4
)

display(dim_results.dataframe.head())


# ## 5. Visualizing and Interpreting the Saturation Curve
# 
# The output is a `Results` object containing a DataFrame. We can use the built-in `.plot()` method to see the curve, and the `find_saturation_point` helper to estimate the elbow point.

# ### Understanding the `strictness` Parameter
# 
# The `find_saturation_point` function has a `strictness` parameter that naively controls how flat the curve must be to be considered a plateau. It's basically checks if  `MI(k) - MI(k-1) < strictness * STD(k)`.
# - **Low `strictness`** (e.g., <= 1) is 'stricter'. It requires a very flat plateau, so it might estimate a higher dimensionality if the curve is slow to level off.
# - **High `strictness`** (e.g., >= 1) is 'looser'. It will declare saturation earlier, even if the curve is still rising slightly.
# 
# By default, the function now tests a range of strictness values to give you a more complete picture of the potential estimates.

# In[4]:


# Automatically find the saturation point for a range of strictness values
estimated_dims = nmi.utils.find_saturation_point(dim_results.dataframe, strictness=[0.1, 1, 15])
print(f"True Latent Dimension: {true_latent_dim}")
print(f"Estimated Latent Dimensions: {estimated_dims}")

# Plot the curve using the built-in plot method
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
dim_results.plot()

# Add annotations for the true and estimated values
plt.axvline(x=true_latent_dim, color='black', linestyle='--', label=f'True Dim ({true_latent_dim})')
for s, est_dim in estimated_dims.items():
    plt.axvline(x=est_dim, linestyle=':', label=f'Est. Dim (strict={s})')

plt.title('Internal Information vs. Embedding Dimension')
plt.legend()
plt.show()


# ## 6. Conclusion
# 
# The result is fantastic! The plot clearly shows the MI estimate rising steadily and then flattening out right around the true latent dimension of 4. The automated `find_saturation_point` function successfully identifies this elbow.
# 
# This demonstrates a powerful exploratory capability of the `NeuralMI` library. You can take a high-dimensional neural recording, run this analysis, and get a quantitative estimate of its intrinsic complexity or the number of independent signals it contains.
# 
# In the next tutorial, we will tackle the most advanced feature: performing a rigorous, bias-corrected analysis to get accurate MI estimate.
