# NeuralMI: Neural Mutual Information Estimation for Neural Data

A PyTorch-based library for estimating mutual information (MI) in neural data using neural network-based estimators.

## Features

- **Multiple Estimators**: InfoNCE, NWJ, TUBA, SMILE lower bounds
- **Flexible Data Processing**: Handle continuous time-series and spike train data
- **Exploratory Analysis**: Parameter sweeps, dimensionality estimation
- **Rigorous Statistics**: Bias-corrected estimates with confidence intervals
- **Production Ready**: GPU acceleration, multiprocessing, early stopping

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- pandas >= 1.2.0
- statsmodels >= 0.12.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

### Install from source

```bash
git clone https://github.com/yourusername/neural_mi.git
cd neural_mi
pip install -e .
```

### Install for development

```bash
pip install -e ".[dev]"
pytest  # Run tests
```

## Quick Start

```python
import neural_mi as nmi

# Generate correlated Gaussian data with known MI
x, y = nmi.datasets.generate_correlated_gaussians(
    n_samples=1000, dim=5, mi=2.0  # MI = 2.0 bits
)

# Configure training parameters
base_params = {
    'n_epochs': 50,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'patience': 5,
    'embedding_dim': 16,
    'hidden_dim': 64,
    'n_layers': 2
}

# Run MI estimation
results = nmi.run(
    x_data=x,
    y_data=y,
    mode='estimate',
    processor_type='continuous',
    processor_params={'window_size': 1, 'data_format': 'channels_last'},
    base_params=base_params,
    output_units='bits'
)

print(f"MI estimate: {results.mi_estimate:.3f} bits")
```

## Usage Modes

### 1. Estimate Mode: Quick MI Estimate
Get a single MI estimate between two variables.

```python
results = nmi.run(
    x_data=x, y_data=y,
    mode='estimate',
    processor_type='continuous',
    processor_params={'window_size': 1, 'data_format': 'channels_last'},
    base_params=base_params
)
print(results.mi_estimate)
```

### 2. Sweep Mode: Hyperparameter Search
Find optimal processing parameters (e.g., window size for temporal data).

```python
# Sweep over window sizes to find temporal scale
sweep_grid = {'window_size': [1, 10, 50, 100, 200, 500]}

results = nmi.run(
    x_data=x, y_data=y,
    mode='sweep',
    processor_type='continuous',
    processor_params={'data_format': 'channels_last'},
    base_params=base_params,
    sweep_grid=sweep_grid,
    n_workers=4
)

# Plot and find optimal window
results.plot()
optimal = results.dataframe.loc[results.dataframe['mi_mean'].idxmax()]
print(f"Optimal window: {optimal['window_size']}")
```

### 3. Dimensionality Mode: Estimate Latent Dimensions
Estimate the intrinsic dimensionality of high-dimensional neural data.

```python
# Generate data: 100D observations from 4D latent
x, _ = nmi.datasets.generate_nonlinear_from_latent(
    n_samples=2000, latent_dim=4, observed_dim=100, mi=3.0
)

sweep_grid = {'embedding_dim': [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]}

results = nmi.run(
    x_data=x,
    mode='dimensionality',
    processor_type='continuous',
    processor_params={'window_size': 1, 'data_format': 'channels_last'},
    base_params=base_params,
    sweep_grid=sweep_grid,
    n_splits=5
)

print(f"Estimated dimensions: {results.details['estimated_dims']}")
results.plot()
```

### 4. Rigorous Mode: Bias-Corrected Estimation
Get statistically rigorous MI estimates with confidence intervals.

```python
results = nmi.run(
    x_data=x, y_data=y,
    mode='rigorous',
    processor_type='continuous',
    processor_params={'window_size': 1, 'data_format': 'channels_last'},
    base_params=base_params,
    sweep_grid={'embedding_dim': [16]},
    gamma_range=range(1, 11),
    n_workers=4
)

print(f"MI: {results.mi_estimate:.3f} ± {results.details['mi_error']:.3f} bits")
results.plot()
```

## API Reference

- `nmi.run()`: Unified entry point for all analyses.
- `nmi.datasets`: Module for generating synthetic test data.
- `nmi.results.Results`: Standardized results object.

## Troubleshooting

- **"torch_shm_manager" error on macOS**: This is handled automatically by the library.
- **Out of memory with large sweeps**: Use the `max_samples_per_task` argument in `nmi.run()`.
- **Results not reproducible**: Set `random_seed` and use `n_workers=1`.
- **Hangs or Deadlocks**: Ensure `multiprocessing.set_start_method('spawn')` is called inside a `if __name__ == '__main__':` block in your main script.

## License

MIT License - see LICENSE file for details.