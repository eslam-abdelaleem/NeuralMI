# NeuralMI: A Toolbox for Mutual Information Estimation in Neuroscience

NeuralMI is a Python library for robust, bias-corrected mutual information estimation, tailored for neuroscience data like continuous time-series and spike trains.

## Features

- **Multiple Analysis Modes:** From quick estimates to rigorous, bias-corrected results with error bars.
- **Data Processors:** Built-in, flexible processors for windowing continuous data and spike trains.
- **Dimensionality Estimation:** Discover the latent dimensionality of neural populations using internal information.
- **Flexible and Tested:** Comes with a full suite of tutorials and a `pytest` test suite.

## Installation

To install the library directly from this repository, run the following command:

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

