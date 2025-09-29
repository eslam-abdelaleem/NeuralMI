# NeuralMI: A Toolbox for Rigorous Mutual Information Estimation in Neuroscience

**NeuralMI** is a Python library designed to provide a complete, end-to-end workflow for robustly estimating mutual information from neuroscience data. Moving beyond simple point estimates, this toolkit incorporates essential techniques for scientific rigor, including automated bias correction, hyperparameter exploration, and analysis of intrinsic dimensionality.

It is built for neuroscientists who need to analyze complex relationships in continuous time-series (like LFP or EEG) and discrete spike train data, providing the tools to move from raw data to publishable, statistically sound results.

## Key Features

- **Unified & Simple API:** Access all analysis modes through a single, powerful `run()` function.
- **Multiple Analysis Modes:**
    - **`estimate`**: Get a quick, single MI estimate for initial exploration.
    - **`sweep`**: Perform parallelized sweeps over any model or data processing hyperparameter (e.g., `window_size`).
    - **`dimensionality`**: Characterize the internal complexity of a neural population by finding its latent dimensionality.
    - **`rigorous`**: The flagship mode that performs automated finite-sampling bias correction via subsampling and extrapolation, providing a debiased MI estimate with a confidence interval.
- **Neuroscience-Ready Data Processors:**
    - `ContinuousProcessor`: Seamlessly handle windowing of continuous time-series data.
    - `SpikeProcessor`: Convert raw spike times into an analyzable format.
- **Built-in Visualizations:** Generate publication-quality plots for dimensionality curves and bias-correction fits with a single command.
- **Flexible & Extensible:** Choose from multiple MI estimators (`InfoNCE`, `TUBA`, `NWJ`) and provide your own custom PyTorch embedding models for advanced use cases.

## Installation

To install the library and its dependencies from this repository, run the following command in the root directory:

```bash
pip install .
```

For developers who want to edit the code, install it in "editable" mode:

```bash
pip install -e .
```

Quickstart
Here is a simple example of estimating the MI between two correlated Gaussian variables:

```bash
import torch
import neural_mi as nmi

# 1. Generate data with a known MI of 2.0 bits
x_data, y_data = nmi.datasets.generate_correlated_gaussians(
    n_samples=5000, 
    dim=5, 
    mi=2.0
)
x_data = x_data.reshape(5000, 1, 5)
y_data = y_data.reshape(5000, 1, 5)

# 2. Define basic model and training parameters
base_params = {
    'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 5, 'embedding_dim': 16, 'hidden_dim': 64, 'n_layers': 2
}

# 3. Run the estimation
estimated_mi = nmi.run(
    x_data=x_data,
    y_data=y_data,
    mode='estimate',
    base_params=base_params
)

print(f"Estimated MI: {estimated_mi:.3f} bits")
```

