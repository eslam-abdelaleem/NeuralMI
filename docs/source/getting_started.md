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

To get the most out of `NeuralMI`, we recommend following the tutorial series in order. Each part answers a different kind of question, and they build on each other.

### Part 0: What an estimate is

* **Tutorial 00:** Why and How MI Estimation Works - Why mutual information rather than correlation, how a neural estimator turns dependence into a number, and which value the library reports.

### Part 1: Getting your data in

* **Tutorial 01:** A First Estimate - One `nmi.run()` call on data with a known answer, what `mi_estimate` is versus `details['test_mi']`, and a KSG comparison showing when a neural estimator is worth its cost.
* **Tutorial 02:** Neural Data Formats - Spike times, binned counts and categorical labels, and what windowing does to each.
* **Tutorial 03:** Temporal Correlations and Splits - Why blocked splitting is the default, and what random splitting costs on autocorrelated data.

### Part 2: Choosing the quantity that matches your question

* **Tutorial 04:** Which Quantity - The taxonomy as one `I(A;B|C)` primitive under different offsets, and why windowed MI is extensive.
* **Tutorial 05:** Storage and Rate - Self-prediction, and the per-step rate that survives as the window grows.
* **Tutorial 06:** Direction and Delay - `mode='lag'`, `mode='precision'`, transfer entropy, and a measured demonstration of why transfer entropy is fragile.

### Part 3: Defending a number

* **Tutorial 07:** Three Variables - Conditional MI and interaction information against an oracle, and the amplification factor governing differences of estimates.
* **Tutorial 08:** Making It Rigorous - Seed spread, `mode='sweep'`, and `mode='rigorous'` with its diagnostics read honestly.

### Part 4: Real recordings, where there is no ground truth

* **Tutorial 09:** What A Population Encodes - Hippocampal place cells and position, two sessions from the same animal.
* **Tutorial 10:** Comparing Brain Areas - Allen Brain Observatory recordings under natural movies and spontaneous activity.

### Part 5: The machinery underneath

* **Tutorial 11:** Models and Machinery - The two estimators and the InfoNCE ceiling, the ten embedding models, permutation nulls, and supplying your own architecture.

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