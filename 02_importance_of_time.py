#!/usr/bin/env python
# coding: utf-8

# # Example 2: The Importance of Time
# 
# In the first example, we assumed each sample $(x_i, y_i)$ was independent. However, in most biological and neural data, relationships are spread out over time. A stimulus now might affect a neural response hundreds of milliseconds later.
# 
# This notebook demonstrates how to handle such temporal dependencies.
# 
# **Goal:**
# 1.  Demonstrate how to handle raw time-series data using the built-in processor.
# 2.  Showcase `run(mode='sweep')` to find the optimal `window_size`.
# 3.  Demonstrate why choosing the correct timescale is critical for MI estimation.

# ## 1. Imports
# 
# We'll need our standard imports, plus `matplotlib` for plotting.

# In[1]:


import torch
import numpy as np
import neural_mi as nmi
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ## 2. Generating Temporally Dependent Data
# 
# We'll use `generate_temporally_convolved_data`. This function creates a latent signal `Z`, and then generates `X` and `Y` by smearing `Z` with different time kernels. This means the relationship between a single point $x_t$ and $y_t$ is weak, but the relationship between a *window* of X and a *window* of Y is strong.
# 
# For this type of data, the ground truth MI isn't easily known, which is realistic. Our goal is not to match a known value, but to find the timescale that *maximizes* the MI.

# In[2]:


# --- Dataset Parameters ---
n_samples = 10000

# --- Generate Raw 2D Data ---
# The output shape is [n_timepoints, n_channels]
x_raw, y_raw = nmi.datasets.generate_temporally_convolved_data(n_samples=n_samples)

print(f"Generated raw X data shape: {x_raw.shape}")
print(f"Generated raw Y data shape: {y_raw.shape}")

# Let's visualize the raw data to see the temporal relationship
plt.figure(figsize=(12, 4))
plt.plot(x_raw[0, :200], label='X', alpha=0.8)
plt.plot(y_raw[0, :200], label='Y', alpha=0.8)
plt.xlabel("Timepoints")
plt.ylabel("Signal")
plt.title("Raw Temporal Data (first 200 points)")
plt.legend()
plt.show()


# ## 3. The Problem: A Naive Estimate Fails
# 
# If we treat each timepoint as an independent sample (a window size of 1), we fail to capture the smeared relationship. Let's prove this by running a quick estimate with `window_size=1`.

# In[3]:


base_params = {
    'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 5, 'embedding_dim': 16, 'hidden_dim': 64, 'n_layers': 2
}

# Run the estimate with a window size of 1
naive_results = nmi.run(
    x_data=x_raw,
    y_data=y_raw,
    mode='estimate',
    processor_type='continuous',
    processor_params={'window_size': 1},
    base_params=base_params
)

print(f"\nNaive MI estimate (window_size=1): {naive_results.mi_estimate:.3f} bits")


# As expected, the MI estimate is very low. We've missed the real relationship.
# 
# ## 4. The Solution: Sweeping Over Window Size
# 
# To find the correct timescale, we need to test many different window sizes. This is a hyperparameter search, which is exactly what `mode='sweep'` is for.
# 
# The process is simple:
# 1.  Define a `sweep_grid`. This dictionary tells the `run` function which parameters to vary. Our key will be `window_size`.
# 2.  Pass the **raw data** and the `sweep_grid` to `nmi.run`.
# 3.  The library automatically applies the processor with each `window_size` from the grid before training the MI estimator.
# 
# *Note: The parameter `window_size` is a data processing parameter, not a model parameter. The library is designed to handle this by automatically routing processing-related keys from the `sweep_grid` to the `DataHandler`.*

# In[4]:


# The sweep will create a new ContinuousProcessor for each value.
sweep_grid = {
    'window_size': [1, 5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 500, 1000]
}

# Notice we pass the RAW data to the run function
sweep_results = nmi.run(
    x_data=x_raw,
    y_data=y_raw,
    mode='sweep',
    processor_type='continuous',
    processor_params={},
    base_params=base_params,
    sweep_grid=sweep_grid,
    n_workers=4  # Speed up the sweep with parallel workers
)

display(sweep_results.dataframe.head())


# ## 5. Analyzing the Results
# 
# The output is a `Results` object containing a DataFrame. We can now either plot the MI curve manually, or use the built-in `.plot()` method for a quick visualization.

# In[ ]:


# Use the built-in plot function!
sweep_results.plot()
plt.title("MI vs. Window Size")
plt.xscale('log')
plt.grid(True, linestyle=':')
plt.show()

# Find the best run programmatically
best_run = sweep_results.dataframe.loc[sweep_results.dataframe['test_mi'].idxmax()]
print("--- Best Result ---")
print(f"Optimal Window Size: {best_run['window_size']}")
print(f"Maximum MI Estimated: {best_run['test_mi']:.3f} bits")


# ## 6. Conclusion
# 
# The result is clear! The estimated MI peaks at a window size around $100\sim 200$ timepoints. By using the right window, we recovered a strong MI that was completely invisible to the naive, point-by-point estimate.
# 
# This example demonstrates a core workflow for analyzing real experimental data:
# 1.  Start with raw time-series data.
# 2.  Use a `sweep` over `window_size` to find the timescale that maximizes information.
# 3.  This optimal window size is itself a valuable scientific finding.
# 
# In the next example, we'll explore the internal structure of a single, high-dimensional dataset to estimate its 'latent dimensionality'.
