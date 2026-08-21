# Getting Started

## A Toolbox for Rigorous Mutual Information Estimation in Neuroscience

**NeuralMI** is a Python library designed to provide neuroscientists with a complete, end-to-end workflow for robustly and reproducibly estimating mutual information from complex neural data.

In modern neuroscience, a naive MI estimate is not enough. Estimates can be plagued by finite-sampling bias and estimator variance, leading to results that aren't scientifically rigorous. `NeuralMI` solves this by moving beyond simple point estimates to incorporate essential techniques for scientific rigor, including automated bias correction, hyperparameter exploration, and cross-run-validated analysis of shared latent structure. It is built for researchers who need to analyze complex relationships in continuous time-series (like LFP or EEG), discrete spike trains, and categorical state data.

## Key Features

* **Unified & Simple API:** Access all analysis modes through a single, powerful `run()` function.
* **Scientifically Rigorous by Default:** The flagship `rigorous` mode performs automated finite-sampling bias correction via subsampling and extrapolation, providing a debiased MI estimate with a confidence interval.
* **Multiple Analysis Modes:**
    * `estimate`: Get a quick, single MI estimate for initial exploration.
    * `sweep`: Perform parallelized sweeps over any model or data processing hyperparameter.
    * `lag`: Find the precise temporal offset between two time-series.
    * `dimensionality`: Find directions of shared structure within a neural population, or between two, that reproduce reliably across independent retrainings.
* **Neuroscience-Ready Data Processors:**
    * `ContinuousProcessor`: Seamlessly handle windowing of LFP, EEG, or calcium imaging data.
    * `SpikeProcessor`: Convert raw spike times into an analyzable format.
    * `CategoricalProcessor`: Process discrete behavioral or stimulus state data.
* **Built-in Visualizations:** Generate high-quality plots for stable-direction charts and bias-correction fits with a single command.
* **Flexible & Extensible:** Choose from multiple MI estimators (`InfoNCE`, `SMILE`, etc.) and provide your own pre-initialized PyTorch models for advanced use cases.

## Quickstart: An Accurate Estimate

Scientists should not settle for a naive estimate. Go from raw data to a bias-corrected MI estimate with a confidence interval in a single step using `mode='rigorous'`.

```python
import neural_mi as nmi
from neural_mi import Model, Training, Processing

# 1. Generate raw data (e.g., 100 channels with 10 latent dims over 2500 timepoints)
x_raw, y_raw = nmi.generators.generate_nonlinear_from_latent(
    n_samples=2500, latent_dim=10, observed_dim=100, mi=3.0
)

# 2. Define model architecture and training configs
model = Model(embedding_dim=16, hidden_dim=64, n_layers=2)
training = Training(n_epochs=50, learning_rate=1e-3, batch_size=128, patience=10)

# 3. Run the rigorous, bias-corrected estimation
# This performs multiple runs on data subsets and extrapolates to an infinite-data estimate.
results = nmi.run(
    x_raw.T, y_raw.T,
    mode='rigorous',
    processing=Processing(x='continuous', x_params={'window_size': 1}),
    model=model, training=training,
    n_workers=4,  # Use multiple cores for speed
    seed=42,
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

### Part 0: Understanding MI Estimation

* **Tutorial 00:** Why and How MI Estimation Works - A conceptual on-ramp: why mutual information (not correlation), how a neural estimator turns dependence into a number, and which value the library reports.

### Part 1: The Fundamentals

* **Tutorial 01:** A First Estimate - Learn the basics of `nmi.run()` and the `Results` object on a simple dataset.
* **Tutorial 02:** Neural Data Formats - Understand how to use the `Continuous`, `Spike`, and `Categorical` processors.
* **Tutorial 03:** Temporal Correlations and Splits - Learn how to handle temporal data and avoid leakage with blocked splitting.

### Part 2: Core Concepts for Scientific Rigor

* **Tutorial 04:** Sweeps - Use `mode='sweep'` to explore and optimize hyperparameters.
* **Tutorial 05:** Rigorous Estimation - A deep dive into `mode='rigorous'` for debiased, accurate MI estimates.

### Part 3: Advanced Analysis and Applications

* **Tutorial 06:** Temporal Questions - Directed, time-resolved analyses: `mode='lag'`, `mode='precision'`, and transfer entropy.
* **Tutorial 07:** Population Questions - Population geometry and connectivity: `mode='dimensionality'`, conditional MI, and the `mode='pairwise'` MI matrix, on real hippocampal and Allen Brain Observatory recordings.
* **Tutorial 08:** Models, Estimators, and Validation - Understand the trade-offs between different critic architectures, estimators, and custom models.

Separately, `benchmarks/vs_classical_estimators.ipynb` compares `NeuralMI` against classical alternatives (the KSG estimator and geometric intrinsic-dimension estimators) on problems chosen to be hard for them — useful if you're deciding whether a neural estimator is the right tool for your data, rather than learning the library itself.

## Installation

```bash
# 1. Clone the repository from GitHub (if in Jupyter or Colab, remember to add "!" before running terminal commands like the following

git clone https://github.com/eslam-abdelaleem/NeuralMI.git

# 2. Navigate into the project directory
cd NeuralMI

# 3. Install the library
# For standard use:
pip install .

# 4. For developers (editable install + tests, docs, and viz extras)
pip install -e ".[dev]"
```

## Further Reading

* [Theoretical Foundations](THEORY.md): A concise theoretical background for the core methods used in the library.
* [Core Concepts](CONCEPTS.md): A practical, code-based walkthrough of how a neural MI estimator is built and trained from scratch.
* [Developer's Guide](DEVELOPERS_GUIDE.md): A map of the codebase for contributors.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up a development environment, run tests, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.