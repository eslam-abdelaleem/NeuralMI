#!/usr/bin/env python
# coding: utf-8

# # Tutorial 7: A Deep Dive into the SpikeProcessor
# 
# This tutorial focuses on the `SpikeProcessor`, designed specifically for handling neural spike train data. Spike data is often represented as a list of timestamps, which requires a different processing approach than continuous signals.
# 
# **Goal:**
# 1.  Understand the expected input format for spike data.
# 2.  Manually use the `SpikeProcessor` to see how it converts spike times into a 3D tensor.
# 3.  Explain the key parameters: `window_size`, `step_size`, and `max_spikes_per_window`.

# ## 1. Imports

# In[ ]:


import torch
import numpy as np
import neural_mi as nmi
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ## 2. Generating Raw Spike Data
# 
# The library expects spike data to be a **list of numpy arrays**. Each array in the list represents a single neuron (a channel) and contains the timestamps of its spikes.
# 
# Let's generate some synthetic data in this format.

# In[ ]:


x_spike_data, y_spike_data = nmi.datasets.generate_correlated_spike_trains(
    n_neurons=5, 
    duration=100.0 # Generate 100 seconds of data initially
)

print(f"Data is a list of {len(x_spike_data)} neurons.")
print(f"Spike times for first neuron in X: {x_spike_data[0][:10]}...")


# ## 3. Manual Processing with `SpikeProcessor`
# 
# The `SpikeProcessor` converts this list of spike times into a 3D tensor of shape `(n_samples, n_channels, n_features)`. 
# 
# Key parameters:
# - `window_size`: The duration of the time window in seconds.
# - `step_size`: The time to slide the window forward for each new sample.
# - `n_seconds`: The total duration of the data to analyze. This is useful for analyzing only a subset of a long recording.
# - `max_spikes_per_window`: This is crucial. Since the number of spikes in any window can vary, we need to pad the results into a consistent tensor. This parameter sets the maximum number of spikes to keep from any window. If a window has more spikes, they are truncated; if it has fewer, the rest of the feature vector is padded with zeros.

# In[ ]:


# Initialize the processor to analyze the first 20 seconds of the data
processor = nmi.data.SpikeProcessor(
    window_size=0.1,  # 100 ms window
    step_size=0.01,   # 10 ms step
    n_seconds=20,     # Analyze the first 20 seconds
    max_spikes_per_window=10 # Keep at most 10 spikes per window
)

# Process the data
x_processed = processor.process(x_spike_data)

print(f"Processed data shape: {x_processed.shape}")


# The output shape `(1991, 5, 10)` makes sense:
# - **1991 samples**: We started with 20 seconds of data. The last window of size 0.1s starts at time 19.90s. With a step of 0.01s, this gives 1991 total windows.
# - **5 channels**: The number of neurons is preserved.
# - **10 features**: Each sample contains the times of up to 10 spikes relative to the start of the window.

# ### Visualizing a Spike Window
# 
# A spike raster plot can help visualize this process. The plot below shows the raw spike times for each neuron. The red shaded region represents a single time window that becomes one sample in our processed tensor. The values in the processed tensor are the spike times *relative to the start of that window*.

# In[ ]:


# Let's visualize the 100th window (index 99)
window_idx = 99
window_size = processor.window_size
step_size = processor.step_size

# The start of our window in time is window_idx * step_size
start_time = window_idx * step_size
end_time = start_time + window_size

fig, ax = plt.subplots(figsize=(12, 6))

# Create the raster plot
ax.eventplot(x_spike_data, color='black', linelengths=0.8)
ax.set_title('Spike Raster Plot')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Neuron ID')
ax.set_yticks(np.arange(len(x_spike_data)))

# Highlight the region corresponding to our chosen window
ax.axvspan(start_time, end_time, color='red', alpha=0.3, label=f'Window {window_idx}')
ax.legend()
ax.set_xlim(start_time - window_size*2, end_time + window_size*2) # Zoom in on the window

plt.show()

# You can also inspect the contents of the processed tensor for this window
print(f"Contents of processed tensor for window {window_idx} (channel 0):")
print(x_processed[window_idx, 0, :])


# ### Auto-detection of `max_spikes_per_window`
# 
# If you don't provide `max_spikes_per_window`, the processor will automatically calculate it from the data by finding the window with the most spikes. This is convenient but can sometimes use more memory if there is one outlier window with an unusually high spike count.

# In[ ]:


base_params = {
    'n_epochs': 5, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 2, 'embedding_dim': 8, 'hidden_dim': 32, 'n_layers': 1
}

results = nmi.run(
    x_data=x_spike_data,
    y_data=y_spike_data,
    mode='estimate',
    processor_type='spike',
    processor_params={'window_size': 0.1, 'step_size': 0.01},
    base_params=base_params
)

print(f"\nMI Estimate from automated pipeline: {results.mi_estimate:.3f} bits")


# ## 5. Conclusion
# 
# This tutorial explained how to format spike train data and how the `SpikeProcessor` turns it into a tensor suitable for MI estimation. Understanding these parameters is key to analyzing spike data effectively with this library.
