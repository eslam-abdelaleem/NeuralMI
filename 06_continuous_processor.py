#!/usr/bin/env python
# coding: utf-8

# # Tutorial 6: A Deep Dive into the ContinuousProcessor
# 
# Previous tutorials showed how to use `processor_type='continuous'` inside the `nmi.run()` function for automated processing. This notebook provides a more detailed look at what the `ContinuousProcessor` is actually doing under the hood.
# 
# **Goal:**
# 1.  Understand the role of `window_size` and `step_size`.
# 2.  Manually use the `ContinuousProcessor` to see how it transforms data.
# 3.  Show how this process relates to the simplified `nmi.run()` workflow.

# ## 1. Imports

# In[ ]:


import torch
import numpy as np
import neural_mi as nmi
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ## 2. Generating Raw Time-Series Data
# 
# First, let's create a simple, raw time-series dataset. We'll make it shape `(n_timepoints, n_channels)` to simulate a typical recording.

# In[ ]:


n_timepoints = 100
n_channels = 3

x_raw = np.random.randn(n_timepoints, n_channels)
y_raw = np.random.randn(n_timepoints, n_channels)

print(f"Raw data shape: {x_raw.shape}")


# ## 3. Manual Processing
# 
# The `ContinuousProcessor` takes a 2D array of shape `(n_channels, n_timepoints)` and converts it into a 3D array of shape `(n_samples, n_channels, window_size)`. Let's see this in action.
# 
# - `window_size`: The number of timepoints to include in each sample.
# - `step_size`: The number of timepoints to slide the window forward for the next sample.

# In[ ]:


# Note: The processor expects shape (n_channels, n_timepoints), so we transpose our raw data.
# The DataHandler inside nmi.run does this automatically with a heuristic.
x_raw_transposed = x_raw.T
print(f"Transposed raw data shape: {x_raw_transposed.shape}\n")

# Initialize the processor
processor = nmi.data.ContinuousProcessor(window_size=10, step_size=1)

# Process the data
x_processed = processor.process(x_raw_transposed)

print(f"Processed data shape: {x_processed.shape}")


# ### Visualizing a Window
# 
# To make this more concrete, let's plot the raw data and highlight how a single processed sample (a window) is extracted from it.

# In[ ]:


# Let's visualize the 5th window (index 4) to see how it maps to the raw data
window_idx = 4
window_size = processor.window_size
step_size = processor.step_size

# The start of our window in the raw data is simply window_idx * step_size
start_timepoint = window_idx * step_size
end_timepoint = start_timepoint + window_size

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1]})

# Plot the full raw data for the first channel on the top subplot
ax1.plot(x_raw[:, 0], label='Raw Data (Channel 1)', color='gray')
ax1.set_title('Raw Time-Series Data')
ax1.set_ylabel('Value')
ax1.legend()

# Highlight the region corresponding to our chosen window
ax1.axvspan(start_timepoint, end_timepoint - 1, color='red', alpha=0.3, label=f'Window {window_idx}')
ax1.legend()

# In the bottom subplot, plot the content of that processed window
ax2.plot(x_processed[window_idx, 0, :].numpy(), 'o-', color='red', label=f'Processed Data from Window {window_idx}')
ax2.set_title(f'Contents of Processed Window {window_idx}')
ax2.set_xlabel('Timepoint within Window')
ax2.set_ylabel('Value')
ax2.legend()

plt.tight_layout()
plt.show()


# The output shape `(91, 3, 10)` makes sense:
# - **91 samples**: We started with 100 timepoints. The last possible window of size 10 starts at index 90. With a step size of 1, this gives us 91 total windows (from index 0 to 90).
# - **3 channels**: The number of channels is preserved.
# - **10 features**: Each sample (window) now contains 10 timepoints, which are treated as features by the MI estimator.

# ## 4. The Simplified Workflow
# 
# While you *can* process the data manually as shown above, you don't need to. The `nmi.run` function does this for you when you provide the `processor_type` and `processor_params`. It even handles the transposition for you.
# 
# The following code achieves the same result as our manual processing, but in a single, clean step.

# In[ ]:


base_params = {
    'n_epochs': 5, 'learning_rate': 1e-3, 'batch_size': 32,
    'patience': 2, 'embedding_dim': 8, 'hidden_dim': 32, 'n_layers': 1
}

results = nmi.run(
    x_data=x_raw, # Pass the raw [time, channel] data
    y_data=y_raw,
    mode='estimate',
    processor_type='continuous',
    processor_params={'window_size': 10, 'step_size': 1},
    base_params=base_params
)

print(f"\nMI Estimate from automated pipeline: {results.mi_estimate:.3f} bits")


# ### A Note on Negative MI\n\nYou may have noticed that the MI estimate was negative. This is not a bug! The estimators used in this library (like InfoNCE) provide a *lower bound* on the true mutual information. When the true MI is very low (or zero, as is the case for this random data), the estimator can sometimes produce small negative values due to statistical noise. \n\n**A negative MI estimate should always be interpreted as an MI of zero.**

# ## 5. Conclusion
# 
# This tutorial demystified the `ContinuousProcessor`. By understanding how it windows data, you can now more effectively choose the `window_size` and `step_size` parameters when performing sweeps to find the characteristic timescale of your data.
