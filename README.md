# NeuralMI: A Toolbox for Rigorous Mutual Information Estimation in Neuroscience

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen)](https://eslam-abdelaleem.github.io/NeuralMI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/eslam-abdelaleem/NeuralMI/actions/workflows/tests.yml/badge.svg)](https://github.com/eslam-abdelaleem/NeuralMI/actions/workflows/tests.yml)

**NeuralMI** is a Python library designed to provide neuroscientists with a complete, end-to-end workflow for robustly and quickly estimating mutual information from complex neural data.

In modern neuroscience, MI estimation is usually not possible and using black-box methods is rarely enough. Estimates can be plagued by finite-sampling bias and estimator variance, leading to results that aren't scientifically rigorous. `NeuralMI` solves this by moving beyond simple point estimates to incorporate essential techniques for scientific rigor, including automated bias correction, hyperparameter exploration, and cross-run-validated analysis of shared latent structure. It is built for researchers who need to analyze complex relationships in continuous time-series (like LFP or EEG), discrete spike trains, and categorical state data.

**A note on what every estimate means:** `NeuralMI`'s estimators are variational **lower bounds** on the true mutual information, not the exact value. This is what makes them tractable on high-dimensional neural data, but it has two direct consequences worth knowing before you interpret a number: a reported estimate can under-report the true MI but never over-report it, and each estimator has a ceiling set by its evaluation batch size (`log(batch_size)` for the default InfoNCE estimator). See [Tutorial 0](tutorials/00_Why_and_How_MI_Estimation_Works.ipynb) for the full mechanics behind this.

## Key Features

* **Unified & Simple API:** Access all analysis modes through a single `run()` function.
* **Bias Correction:** The `rigorous` mode performs automated finite-sampling bias correction via subsampling and extrapolation, providing a debiased MI estimate with a confidence interval.
* **Multiple Analysis Modes:**
    * **`estimate`**: Get a quick, single MI estimate for initial exploration.
    * **`sweep`**: Perform parallelized sweeps over any model or data processing hyperparameter.
    * **`lag`**: Find the precise temporal offset between two time-series through a specialized sweep.
    * **`dimensionality`**: Find directions of shared structure within a neural population, or between two, that reproduce reliably across independent retrainings.
    * **`precision`**: Find the precise threshold at which spike-timing resolution matters.
    * **`conditional`**: Compute Conditional Mutual Information (CMI) to isolate direct relationships.
    * **`transfer`**: Estimate Transfer Entropy to understand directed information flow over time, optionally controlling for a third signal (conditional transfer entropy).
    * **`interaction`**: Compute Interaction Information to see how a third population changes what two others share, redundancy or synergy.
    * **`pairwise`**: Rapidly build all-to-all functional connectivity matrices.
* **Neuroscience-Ready Data Processors:**
    * `ContinuousProcessor`: Seamlessly handle windowing of LFP, EEG, or calcium imaging data.
    * `SpikeProcessor`: Convert raw spike times into an analyzable format.
    * `CategoricalProcessor`: Process discrete behavioral or stimulus state data.
* **Smart Data Splitting**: Automatically handles train/test splits for both **temporal** data (default `'blocked'` split) and **IID** data (`split_mode='random'`) to ensure valid, reliable estimates.
* **Built-in Visualizations:** Generate plots for stable-direction charts and bias-correction fits with a single command.
* **Flexible & Extensible:** Choose from multiple MI estimators (`InfoNCE`, `SMILE`, etc.) and provide your own pre-initialized PyTorch models for advanced use cases.

## Quickstart: An Accurate Estimate
Here's how to perform a rigorous, bias-corrected MI estimation between two independent (IID) variables.

```python
import neural_mi as nmi
from neural_mi import Processing, Split

# 1. Generate raw data (e.g., 100 channels with 10 latent dims over 2500 timepoints)
x_raw, y_raw = nmi.generators.generate_nonlinear_from_latent(
    n_samples=2500, latent_dim=10, observed_dim=100, mi=3.0
)

# 2. Run the rigorous, bias-corrected estimation
# This performs multiple runs on data subsets and extrapolates to an infinite-data estimate.
results = nmi.run(
    x_raw, y_raw,
    mode='rigorous',
    processing=Processing(x='continuous', x_params={'window_size': 1}),
    split=Split(mode='random'),  # random splitting for IID data
    n_workers=4,                 # use multiple cores for speed
    seed=42,
)

# 3. Access and print the final, scientifically robust result
mi_est = results.mi_estimate
mi_err = results.details.get('mi_error', 0.0)
print(f"\nCorrected MI: {mi_est:.3f} ± {mi_err:.3f} bits")

# 4. Visualize the bias-correction procedure
# This plot shows the extrapolation to an infinite dataset size (1/N -> 0).
results.plot()
```


## Learning Path
To get the most out of `NeuralMI`, we recommend following the tutorial series in order. Each tutorial builds on the last, taking you from the basics to advanced applications.

- **Part 0: What an estimate is**
    - **[00_Why_and_How_MI_Estimation_Works](tutorials/00_Why_and_How_MI_Estimation_Works.ipynb)**: Why mutual information rather than correlation, how a neural estimator turns dependence into a number, and which value the library reports.
- **Part 1: Getting your data in**
    - **[01_A_First_Estimate](tutorials/01_A_First_Estimate.ipynb)**: One `nmi.run()` call on data with a known answer. What `mi_estimate` is versus `details['test_mi']`, how the answer moves with sample size, and a KSG comparison showing when a neural estimator is worth its cost.
    - **[02_Neural_Data_Formats](tutorials/02_Neural_Data_Formats.ipynb)**: Spike times, binned counts and categorical labels, what windowing does to each, and which quantity `drop_empty_windows` selects.
    - **[03_Temporal_Correlations_and_Splits](tutorials/03_Temporal_Correlations_and_Splits.ipynb)**: Why `Split(mode='blocked')` is the default, and what random splitting costs on autocorrelated data.
- **Part 2: Choosing the quantity that matches your question**
    - **[04_Which_Quantity](tutorials/04_Which_Quantity.ipynb)**: The whole taxonomy as one `I(A;B|C)` primitive under different offset patterns. Also why windowed MI is extensive, so no window size reveals a plateau.
    - **[05_Storage_and_Rate](tutorials/05_Storage_and_Rate.ipynb)**: How much a process predicts about its own future, and the per-step rate that survives as the window grows.
    - **[06_Direction_and_Delay](tutorials/06_Direction_and_Delay.ipynb)**: `mode='lag'`, `mode='precision'`, transfer entropy and Massey's conservation law. Includes a measured demonstration of why transfer entropy is fragile: 25 to 40 times error amplification, with its reported direction reversing when the history window changes.
- **Part 3: Defending a number**
    - **[07_Making_It_Rigorous](tutorials/07_Making_It_Rigorous.ipynb)**: Seed spread, `mode='sweep'`, and `mode='rigorous'` with its diagnostics read honestly, including what a flat bias slope does and does not mean.
    - **[08_Three_Variables](tutorials/08_Three_Variables.ipynb)**: Conditional MI and interaction information against an oracle with exact values, redundancy versus synergy, and the amplification factor that says how far a difference of estimates can be trusted.
- **Part 4: Real recordings, where there is no ground truth**
    - **[09_What_A_Population_Encodes](tutorials/09_What_A_Population_Encodes.ipynb)**: Hippocampal place cells and position, across two sessions from the same animal. Each section starts from a hypothesis, and the controls are what carry the claims.
    - **[10_Comparing_Brain_Areas](tutorials/10_Comparing_Brain_Areas.ipynb)**: Allen Brain Observatory recordings from VISp, VISpm and CA1 under natural movies and spontaneous activity. Functional coupling, intrinsic timescale, and which comparisons the data actually supports.
- **Part 5: The machinery underneath**
    - **[11_Models_and_Machinery](tutorials/11_Models_and_Machinery.ipynb)**: The two estimators and the InfoNCE ceiling, the ten embedding models, permutation nulls, and how to supply your own architecture.

Separately, [`benchmarks/vs_classical_estimators.ipynb`](benchmarks/vs_classical_estimators.ipynb) compares `NeuralMI` against classical alternatives (the KSG estimator and geometric intrinsic-dimension estimators) on problems chosen to be hard for them — useful if you're deciding whether a neural estimator is the right tool for your data, rather than learning the library itself.

## Installation

> **Jupyter / Colab users:** prefix each shell command below with `!` (e.g. `!git clone ...`).

```bash
# 1. Clone the repository from GitHub
git clone https://github.com/eslam-abdelaleem/NeuralMI.git

# 2. Navigate into the project directory
cd NeuralMI

# 3. Install the library
# For standard use:
pip install .

# Optional extras:
#   pip install ".[viz]"     # UMAP / t-SNE / PCA embedding plots
#   pip install ".[vision]"  # pretrained-backbone embeddings (torchvision)

# 4. For developers (editable install + tests, docs, and viz extras)
pip install -e ".[dev]"
```

## Further Reading
Check out the fully updated documentation site: [https://eslam-abdelaleem.github.io/NeuralMI/](https://eslam-abdelaleem.github.io/NeuralMI/)

If you prefer exploring the repository directly, we also include several detailed guides:
- `NEURALMI_REFERENCE.md`: The complete technical reference covering the entire public API and parameter configurations.
- `THEORY.md`: A concise theoretical background for the core methods used in the library.
- `CONCEPTS.md`: A practical, code-based walkthrough of how a neural MI estimator is built and trained from scratch.
- `DEVELOPERS_GUIDE.md`: A guide to the codebase for contributors and advanced users.

## Contributing
Contributions are welcome! Please see ```CONTRIBUTING.md``` for details on how to set up a development environment, run tests, and submit pull requests.

## License
This project is licensed under the MIT License - see the ```LICENSE``` file for details.
