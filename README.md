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

To install development dependencies for running tests:

```bash
pip install -r requirements-dev.txt
```

## Quickstart

Here is a simple example of estimating the MI between two correlated Gaussian variables.


```python
import torch
import neural_mi as nmi
import numpy as np

# 1. Generate raw 2D data (e.g., 5 channels over 5000 timepoints)
# For raw continuous data, the expected shape is (n_channels, n_timepoints)
x_raw, y_raw = nmi.datasets.generate_correlated_gaussians(
    n_samples=5000, 
    dim=5, 
    mi=2.0
)
x_raw = x_raw.T  # Transpose to (channels, timepoints)
y_raw = y_raw.T  # Transpose to (channels, timepoints)


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
    base_params=base_params,
    random_seed=42
)

# 4. Access the result from the standardized Results object
print(f"Estimated MI: {results.mi_estimate:.3f} bits")

# 5. Visualize results (if applicable)
# results.plot() # Plotting is available for 'sweep', 'dimensionality', and 'rigorous' modes
```

## Usage Modes

# 1. Sweep Mode: Hyperparameter Search
Find optimal processing parameters (e.g., window size for temporal data).

```python
# Generate data where a temporal window is important
x_temp, y_temp = nmi.datasets.generate_temporally_convolved_data(n_samples=5000)

sweep_grid = {'window_size': [1, 10, 50, 100, 200]}

results_sweep = nmi.run(
    x_data=x_temp, y_data=y_temp,
    mode='sweep',
    processor_type='continuous',
    processor_params={}, # window_size is specified in sweep_grid
    base_params=base_params,
    sweep_grid=sweep_grid,
    n_workers=4,
    random_seed=42
)

# Plot and find optimal window
ax = results_sweep.plot(show=False)
optimal_ws = results_sweep.dataframe.loc[results_sweep.dataframe['mi_mean'].idxmax()]['window_size']
ax.set_title(f"Optimal Window Size: {optimal_ws}")
# ax.get_figure().show() # Or plt.show()
```

# 2. Dimensionality Mode: Estimate Latent Dimensions
Estimate the intrinsic dimensionality of high-dimensional neural data.

```python
# Generate data: 100D observations from 4D latent
x_latent, _ = nmi.datasets.generate_nonlinear_from_latent(
    n_samples=2000, latent_dim=4, observed_dim=100, mi=3.0
)

sweep_grid_dim = {'embedding_dim': [1, 2, 4, 6, 8, 12, 16]}

results_dim = nmi.run(
    x_data=x_latent.T, # Transpose to (channels, samples)
    mode='dimensionality',
    processor_type='continuous',
    processor_params={'window_size': 1},
    base_params=base_params,
    sweep_grid=sweep_grid_dim,
    n_splits=5,
    random_seed=42
)

print(f"Estimated dimensionality: {results_dim.details['estimated_dims']}")
# results_dim.plot()
```

# 3. Rigorous Mode: Bias-Corrected Estimation
Get statistically rigorous MI estimates with confidence intervals.

```python
results_rigorous = nmi.run(
    x_data=x_raw, y_data=y_raw,
    mode='rigorous',
    processor_type='continuous',
    processor_params={'window_size': 1},
    base_params=base_params,
    sweep_grid={'embedding_dim': [16]}, # Must specify params for the run
    gamma_range=range(1, 11),
    n_workers=4,
    random_seed=42
)

print(f"Corrected MI: {results_rigorous.mi_estimate:.3f} ± {results_rigorous.details['mi_error']:.3f} bits")
# results_rigorous.plot() # Shows extrapolation to infinite data
```

## Troubleshooting
# Out of memory with large sweeps
Here is a simple example of estimating the MI between two correlated Gaussian variables.
Reduce the number of parallel workers or use the ```max_samples_per_task``` argument in ```run()``` to subsample your data for each sweep task.

```python
results = nmi.run(..., n_workers=2, max_samples_per_task=10000)
```

# MI estimate is NaN
- Check for ```NaN``` or ```Inf``` in your input data.
- Ensure your dataset is large enough for the analysis.
- The ```batch_size``` in base_params might be larger than your number of training samples. The library will warn you and reduce it, but it's good practice to set it appropriately.

## Contributing
Contributions are welcome! Please see ```CONTRIBUTING.md``` for details on how to set up a development environment, run tests, and submit pull requests.

## License
This project is licensed under the MIT License - see the ```LICENSE``` file for details.