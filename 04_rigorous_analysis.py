#!/usr/bin/env python
# coding: utf-8

# # Example 4: From Naive Estimates to Rigorous Analysis
# 
# Using any MI estimator naively on a neural system is rarely a good idea. While our `'estimate'` mode is sophisticated, it doesn't account for a critical statistical pitfall: **finite-sampling bias**. With limited data, estimators tend to find spurious correlations, leading to an overestimation of the true MI. This means that if you tried to calcauate MI when you have a certain number of samples, then tried again using a different subset of those samples, you'll get a different answer --unless you have infinite amount of data--.
# 
# This final notebook demonstrates how to move from a simple estimate to a rigorous one by correcting for this bias and producing principled error bars.
# 
# **Goal:**
# 1.  Showcase the bias problem on a complex, nonlinear dataset.
# 2.  Use `run(mode='rigorous')` to get a debiased estimate and a confidence interval.
# 3.  Use the built-in plotting function to visualize the bias correction.

# ## 1. Imports

# In[ ]:


import torch
import numpy as np
import pandas as pd
import neural_mi as nmi
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ## 2. Generating Complex, Biologically-Plausible Data
# 
# To highlight the need for rigorous analysis, we'll use our most complex dataset, `generate_nonlinear_from_latent`. This mimics a scenario where two high-dimensional neural populations (`X` and `Y`) are driven by a shared, low-dimensional latent signal. 
# 
# We'll use a small number of samples (`n_samples=1500`) to ensure that finite-sampling bias will be a significant problem.

# In[ ]:


n_samples = 1500
latent_dim = 4
observed_dim = 100 # Each variable X and Y has 100 dimensions (features)
latent_mi_bits = 3.0 # The MI between the *hidden* variables

# Generate raw 2D data of shape [n_samples, observed_dim]
x_raw, y_raw = nmi.datasets.generate_nonlinear_from_latent(
    n_samples=n_samples, 
    latent_dim=latent_dim,
    observed_dim=observed_dim,
    mi=latent_mi_bits
)

print(f"Raw X data shape: {x_raw.shape}")


# ## 3. The Problem: A Naive Estimate is Unreliable
# 
# Let's run a simple `'estimate'` on this data. While we don't know the exact ground truth MI in the observed space (due to the nonlinear transform), we will see that the rigorous estimate provides a very different, more reliable answer.
# 
# Since each sample is independent, we'll use a `window_size=1`.

# In[ ]:


base_params = {
    'n_epochs': 60, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 7, 'embedding_dim': 20, 'hidden_dim': 128, 'n_layers': 3
}

naive_results = nmi.run(
    x_data=x_raw, 
    y_data=y_raw, 
    mode='estimate', 
    processor_type='continuous', 
    processor_params={'window_size': 1},
    base_params=base_params
)

print("--- Naive Estimate ---")
print(f"Naive Estimated MI:   {naive_results.mi_estimate:.3f} bits")


# Let's run it again, notice the first problem: each run gives slightly different answer.

# In[ ]:


naive_results_2 = nmi.run(x_data=x_raw, y_data=y_raw, mode='estimate', processor_type='continuous', processor_params={'window_size': 1}, base_params=base_params)

print("--- Naive Estimate ---")
print(f"Naive Estimated MI #2:   {naive_results_2.mi_estimate:.3f} bits")


# Let's run it again but with half the data, notice the second problem: finite sizes make a difference.

# In[ ]:


naive_results_half = nmi.run(
    x_data=x_raw[:n_samples//2], 
    y_data=y_raw[:n_samples//2], 
    mode='estimate', 
    processor_type='continuous', 
    processor_params={'window_size': 1},
    base_params=base_params
)

print("--- Naive Estimate ---")
print(f"Naive Estimated MI For half the data:   {naive_results_half.mi_estimate:.3f} bits")


# ## 4. The Solution: Rigorous, Bias-Corrected Estimation
# 
# MI is a property of the distributions, and it shouldn't depend on the samples. However, since we don't have access to the distrubtions, and unless we have infinite amount of data, then we will have finite sample size effects, and we need to correct for them.
# 
# The `'rigorous'` mode performs subsampling and extrapolation to remove the finite-sampling bias. This provides a more accurate point estimate and, just as importantly, a principled confidence interval (error bar).

# In[ ]:


rigorous_results = nmi.run(
    x_data=x_raw,
    y_data=y_raw,
    mode='rigorous',
    processor_type='continuous',
    processor_params={'window_size': 1},
    base_params=base_params,
    sweep_grid={'embedding_dim': [16]},
    n_workers=4
)

print("\n--- Rigorous Estimate ---")
print(f"Naive Estimate:   {naive_results.mi_estimate:.3f} bits")
print(f"Corrected MI:     {rigorous_results.mi_estimate:.3f} ± {rigorous_results.details['mi_error']:.3f} bits")


# ## 5. Visualizing the Correction
# 
# The plot shows the raw MI estimates for each number of subsets ($\gamma$) and the extrapolation line that leads to the final, debiased estimate. Notice how different runs/fractions give different results, and how more trustworthy extrapolated result are.
# 
# We can use the built-in `.plot()` method on the `Results` object to generate this visualization instantly.

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
rigorous_results.plot()

ax.axhline(y=naive_results.mi_estimate, color='purple', linestyle='--', label=f'Naive Estimate ({naive_results.mi_estimate:.2f} bits)')
ax.axhline(y=latent_mi_bits, color='green', linestyle='-', label=f'True Latent MI ({latent_mi_bits:.2f} bits)')
ax.legend()
ax.set_ylim(-0.1, 3.2)
plt.show()

