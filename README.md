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

## Quickstart

Here is a simple example of estimating the MI between two correlated Gaussian variables. This demonstrates how to use the built-in continuous data processor to automatically handle windowing.

```python
import torch
import neural_mi as nmi

# 1. Generate raw 2D data (e.g., 5 channels over 5000 timepoints)
x_raw, y_raw = nmi.datasets.generate_correlated_gaussians(
    n_samples=5000, 
    dim=5, 
    mi=2.0
)

# 2. Define processing and model parameters
# The processor will window the raw data into 3D tensors automatically.
processor_params = {'window_size': 10, 'step_size': 1}

# Basic model and training parameters
base_params = {
    'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 5, 'embedding_dim': 16, 'hidden_dim': 64, 'n_layers': 2
}

# 3. Run the estimation directly on raw data
results = nmi.run(
    x_data=x_raw,
    y_data=y_raw,
    mode='estimate',
    processor_type='continuous',
    processor_params=processor_params,
    base_params=base_params
)

# 4. Access the result from the standardized Results object
print(f"Estimated MI: {results.mi_estimate:.3f} bits")
```

