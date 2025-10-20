# Getting Started

## A Toolbox for Rigorous Mutual Information Estimation in Neuroscience

**NeuralMI** is a Python library designed to provide neuroscientists with a complete, end-to-end workflow for robustly and reproducibly estimating mutual information from complex neural data.

In modern neuroscience, a naive MI estimate is not enough. Estimates can be plagued by finite-sampling bias and estimator variance, leading to results that aren't scientifically rigorous. `NeuralMI` solves this by moving beyond simple point estimates to incorporate essential techniques for scientific rigor, including automated bias correction, hyperparameter exploration, and novel analyses of intrinsic dimensionality. It is built for researchers who need to analyze complex relationships in continuous time-series (like LFP or EEG), discrete spike trains, and categorical state data.

## Key Features

* **Unified & Simple API:** Access all analysis modes through a single, powerful `run()` function.
* **Scientifically Rigorous by Default:** The flagship `rigorous` mode performs automated finite-sampling bias correction via subsampling and extrapolation, providing a debiased MI estimate with a confidence interval.
* **Multiple Analysis Modes:**
    * `estimate`: Get a quick, single MI estimate for initial exploration.
    * `sweep`: Perform parallelized sweeps over any model or data processing hyperparameter.
    * `lag`: Find the precise temporal offset between two time-series.
    * `dimensionality`: Characterize the internal complexity of a neural population by finding its latent dimensionality.
* **Neuroscience-Ready Data Processors:**
    * `ContinuousProcessor`: Seamlessly handle windowing of LFP, EEG, or calcium imaging data.
    * `SpikeProcessor`: Convert raw spike times into an analyzable format.
    * `CategoricalProcessor`: Process discrete behavioral or stimulus state data.
* **Built-in Visualizations:** Generate high-quality plots for dimensionality curves and bias-correction fits with a single command.
* **Flexible & Extensible:** Choose from multiple MI estimators (`InfoNCE`, `SMILE`, etc.) and provide your own pre-initialized PyTorch models for advanced use cases.

## Quickstart: An Accurate Estimate

Scientists should not settle for a naive estimate. Go from raw data to a bias-corrected MI estimate with a confidence interval in a single step using `mode='rigorous'`.

```python
import neural_mi as nmi
import numpy as np

# 1. Generate raw data (e.g., 100 channels with 10 latent dims over 2500 timepoints)
x_raw, y_raw = nmi.datasets.generate_nonlinear_from_latent(
    n_samples=2500, latent_dim=10, observed_dim=100, mi=3.0
)

# 2. Define model and training parameters
base_params = {
    'n_epochs': 50, 'learning_rate': 1e-3, 'batch_size': 128,
    'patience': 10, 'embedding_dim': 16, 'hidden_dim': 64, 'n_layers': 2
}

# 3. Run the rigorous, bias-corrected estimation
# This performs multiple runs on data subsets and extrapolates to an infinite-data estimate.
results = nmi.run(
    x_data=x_raw.T, y_data=y_raw.T,
    mode='rigorous',
    processor_type_x='continuous',
    processor_params_x={'window_size': 1},
    base_params=base_params,
    n_workers=4,  # Use multiple cores for speed
    random_seed=42
)

# 4. Access and print the final, scientifically robust result
mi_est = results.mi_estimate
mi_err = results.details.get('mi_error', 0.0)
print(f"\nCorrected MI: {mi_est:.3f} ± {mi_err:.3f} bits")

# 5. Visualize the bias-correction procedure
# This plot shows the extrapolation to an infinite dataset size (1/N -> 0).
results.plot()
```

## Learning Path

To get the most out of `NeuralMI`, we recommend following our tutorial series in order. Each tutorial builds on the last, taking you from the basics to advanced applications.

### Part 1: The Fundamentals

* **Tutorial 01:** A First Estimate - Learn the basics of `nmi.run()` and the `Results` object on a simple dataset.
* **Tutorial 02:** A Visual Guide to Neuroscience Data Processing - Understand how to use the `Continuous`, `Spike`, and `Categorical` processors.

### Part 2: Core Concepts for Scientific Rigor

* **Tutorial 03:** Finding Temporal Relationships - Use `mode='sweep'` to find the optimal `window_size` and `mode='lag'` to analyze temporal data.
* **Tutorial 04:** Choosing the Right Model and Estimator - Understand the trade-offs between different critic architectures and MI estimators.

### Part 3: Advanced Analysis and Customization

* **Tutorial 05:** A Better Standard - Bias Correction - A deep dive into `mode='rigorous'` for accurate results.
* **Tutorial 06:** Uncovering Latent Dimensionality - Use `mode='dimensionality'` and variational models to explore data complexity.
* **Tutorial 07:** Advanced Customization - Learn how to use your own custom PyTorch models with the library.

## Installation

```bash
# 1. Clone the repository from GitHub (if in Jupyter or Colab, remember to add "!" before running terminal commands like the following

git clone https://github.com/eslam-abdelaleem/NeuralMI.git

# 2. Navigate into the project directory
cd NeuralMI

# 3. Install the library
# For standard use:
pip install .

# 4. For developers
pip install -e .
pip install -r requirements-dev.txt
```

## Further Reading

* [Theoretical Foundations](THEORY.md): A concise theoretical background for the core methods used in the library.
* [Core Concepts](CONCEPTS.md): A practical, code-based walkthrough of how a neural MI estimator is built and trained from scratch.
* [Developer's Guide](DEVELOPERS_GUIDE.md): A map of the codebase for contributors.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING](CONTRIBUTING.md) for details on how to set up a development environment, run tests, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.