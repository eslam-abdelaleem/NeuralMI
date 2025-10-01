#!/usr/bin/env python
# coding: utf-8

# # Example 1: A Quick First Estimate
# 
# This notebook covers the most basic use case of the `NeuralMI` library: getting a single, quick estimate of mutual information between two variables.
# 
# **Goal:**
# 1.  Introduce the main `neural_mi.run` function.
# 2.  Use a simple dataset where the ground truth MI is known analytically.
# 3.  Compare our estimate to the ground truth to verify the library is working.

# ## 1. Imports
# 
# We'll need `torch` for data handling, our `run` function, and the data generator.

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


# ## 2. Generating the Data
# 
# We will use the `generate_correlated_gaussians` function from the `datasets` module. This function creates two multidimensional Gaussian variables, `X` and `Y`, where we can precisely specify the mutual information between them in **bits**.
# 
# The analytical formula for MI between two multivariate Gaussians is:
# 
# $$ I(X;Y) = -\frac{1}{2} \log_2 \det(\Sigma_{XY}) $$
# 
# Where $\Sigma_{XY}$ is the correlation matrix. Our generator function handles this for us. Let's create data with a known MI of **2.0 bits**.

# In[2]:


# --- Dataset Parameters ---
n_samples = 5000
dim = 5
ground_truth_mi_bits = 2.0

# --- Generate Raw 2D Data ---
# This creates data of shape [n_samples, dim].
x_raw, y_raw = nmi.datasets.generate_correlated_gaussians(
    n_samples=n_samples, 
    dim=dim, 
    mi=ground_truth_mi_bits
)

print(f"Generated raw X data shape: {x_raw.shape}")
print(f"Generated raw Y data shape: {y_raw.shape}")


# ## 3. Defining the Analysis Parameters
# 
# The `run` function requires a `base_params` dictionary. This tells the internal `Trainer` how to configure the neural network and the training process. 
# 
# Since we are passing raw data, we also need to tell the library how to process it. We'll specify `processor_type='continuous'` and provide the `processor_params`. In this simple case, each sample is independent, so we use a `window_size` of 1, which tells the processor to treat each row as a distinct sample.

# In[3]:


# The processor will treat each row as a sample.
processor_params = {'window_size': 1}

# Basic model and training parameters
base_params = {
    'n_epochs': 50,          # Number of training epochs
    'learning_rate': 1e-3,   # Learning rate for the optimizer
    'batch_size': 128,       # Batch size for training
    'patience': 5,           # Early stopping patience
    
    # --- Network Architecture ---
    'embedding_dim': 16,     # Dimensionality of the learned embeddings
    'hidden_dim': 64,        # Number of units in hidden layers
    'n_layers': 2            # Number of hidden layers in the MLP
}


# ## 4. Running the MI Estimation
# 
# Now we call the main `nmi.run` function. We provide our raw data and the processing parameters. The library handles the rest.
# 
# The function returns a standardized `Results` object, which contains the MI estimate and other useful information.

# In[4]:


results = nmi.run(
    x_data=x_raw,             # Pass raw 2D data
    y_data=y_raw,             # Pass raw 2D data
    mode='estimate',
    processor_type='continuous', # Specify the processor
    processor_params=processor_params,
    base_params=base_params,
    output_units='bits',
)

# Access the estimate from the Results object
estimated_mi_bits = results.mi_estimate

print(f"\n--- Results ---")
print(f"Ground Truth MI:      {ground_truth_mi_bits:.3f} bits")
print(f"Estimated MI:         {estimated_mi_bits:.3f} bits")
print(f"Estimation Error:     {abs(estimated_mi_bits - ground_truth_mi_bits):.3f} bits")


# ## 5. Conclusion
# 
# Success! The estimated MI is very close to the ground truth value we specified. We were able to get this estimate without manually reshaping or processing our data, because the `DataHandler` inside the `run` function took care of it for us.
# 
# In the next example, we'll tackle a more complex problem where the relationship between X and Y isn't instantaneous, demonstrating the power of the built-in windowing processor.
