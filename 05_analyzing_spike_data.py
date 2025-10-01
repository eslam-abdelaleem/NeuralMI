#!/usr/bin/env python
# coding: utf-8

# # Example 5: Analyzing Spike Train Data
# 
# This tutorial demonstrates the end-to-end workflow for analyzing relationships between populations of spiking neurons, including a method for improving the stability of the results.
# 
# **Goal:**
# 1.  Demonstrate the use of the `processor_type='spike'`.
# 2.  Use a sweep to find the characteristic timescale of a relationship.
# 3.  Demonstrate how to perform multiple runs to get a more robust, averaged MI estimate.

# ## 1. Imports

# In[ ]:


import torch
import numpy as np
import pandas as pd
import neural_mi as nmi
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ## 2. Generating Synthetic Spike Data
# 
# We'll simulate two neural populations, X and Y, where Y is driven by X with a ~20ms delay. Our goal is to recover this timescale from the data. The data should be a **list of numpy arrays**, where each array contains the spike times for one neuron.

# In[ ]:


x_spike_data, y_spike_data = nmi.datasets.generate_correlated_spike_trains(duration=100, firing_rate=30)\n
# Visualize the first two neurons to see the relationship
plt.figure(figsize=(12, 4))
plt.eventplot(x_spike_data[0], color='C0', linelengths=0.8, lineoffsets=1, label=r'Neuron $X_1$')
plt.eventplot(y_spike_data[0], color='C1', linelengths=0.8, lineoffsets=-1, label=r'Neuron $Y_1$')
plt.yticks([]); 
plt.xlabel("Time (s)"); 
plt.title("Spike Raster for First Neuron Pair"); 
plt.xlim(0, 10); 
plt.legend()
plt.show()


# ## 3. Improving Robustness with Multiple Runs
# 
# Due to the randomness in neural network initialization and training, a single MI estimate can be noisy. To get a more stable result, we can run the estimation multiple times for each `window_size` and average the results.
# 
# We achieve this by adding a `run_id` parameter to our sweep grid. By providing `range(5)`, we tell the sweep engine to perform 5 independent runs for every other parameter combination.

# In[ ]:


base_params = {
    'n_epochs': 100, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 10, 'embedding_dim': 16, 'hidden_dim': 64, 'n_layers': 2
}

sweep_grid = {
    'window_size': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    'run_id': range(5) # Perform 5 runs for each window size
}

# Pass the raw spike data (list of arrays) directly to the function
sweep_results = nmi.run(
    x_data=x_spike_data,
    y_data=y_spike_data,
    mode='sweep',
    base_params=base_params,
    sweep_grid=sweep_grid,
    processor_type='spike',
    processor_params={'step_size': 0.001},
    n_workers=4
)

display(sweep_results.dataframe.head())


# ## 4. Analyzing the Averaged Results
# 
# The output `Results` object contains a DataFrame with multiple `test_mi` values for each `window_size`. We can use `pandas` to group by `window_size` and calculate the mean and standard deviation, giving us a much more reliable picture.

# In[ ]:


# Group by the swept parameter and aggregate
summary_df = sweep_results.dataframe.groupby('window_size')['test_mi'].agg(['mean', 'std']).reset_index()

# Find the optimal window size that maximizes the mean MI
best_window_size = summary_df.loc[summary_df['mean'].idxmax()]['window_size']
print(f"Optimal Window Size: {best_window_size*1000:.1f} ms")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Use the generic plotting function on our aggregated data
nmi.visualize.plot_sweep_curve(
    summary_df=summary_df,
    param_col='window_size',
    mean_col='mean',
    std_col='std',
    estimated_values={'Optimal': best_window_size},
    ax=ax
)

ax.axvline(x=0.02, color='green', linestyle='--', label='True Delay (20ms)')
ax.set_xlabel("Window Size (seconds)")
ax.set_title("MI vs. Window Size for Spike Data (Averaged)")
ax.legend()
plt.show()

