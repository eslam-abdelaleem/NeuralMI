# Under the Hood: A Developer's Guide to NeuralMI

This document provides a map of the `NeuralMI` codebase. It's intended for developers who want to contribute to the library or understand its internal architecture.

---

## Core Philosophy

The library is built around a central `run()` function (`neural_mi/run.py`) that acts as a controller. This function validates parameters, prepares the data, and then delegates the specific analysis to a dedicated module (e.g., `sweep`, `lag`, `rigorous`). This keeps the main entry point clean and makes it easy to add new analysis modes.

---

## Codebase Structure

If you want to modify a specific part of the library, here's where to look.

### `neural_mi/run.py`

This is the main entry point. All user interactions start here. It handles parameter validation and dispatches tasks to the appropriate analysis modules.

---

### `neural_mi/analysis/`

This directory contains the logic for the different analysis `modes`.

- **`workflow.py`**: Implements the `mode='rigorous'` analysis, including subsampling and extrapolation logic.
- **`sweep.py`**: A general-purpose engine for running parallelized hyperparameter sweeps (`mode='sweep'`).
- **`lag.py`**: Contains the logic for `mode='lag'`, which is a specialized `sweep` over the `lag` parameter.
- **`dimensionality.py`**: Implements the `mode='dimensionality'` analysis.
- **`task.py`**: A helper module that defines a single, runnable "task" (one training run of the MI estimator), which is used by all analysis modes.

---

### `neural_mi/data/`

This directory handles all data preprocessing.

- **`handler.py`**: The `DataHandler` class is the main interface. It takes the raw user data and uses the correct processor.
- **`processors.py`**: Contains the `ContinuousProcessor`, `SpikeProcessor`, and `CategoricalProcessor` classes, which transform raw neural data into a format ready for the models.

---

### `neural_mi/models/`

This directory defines all the PyTorch neural network architectures.

- **`critics.py`**: Contains the main critic architectures (e.g., `SeparableCritic`, `ConcatCritic`). These are the networks that actually output the MI estimate.
- **`embeddings.py`**: Defines the embedding networks (e.g., `MLPEmbedding`, `LSTMEmbedding`) that process the input data before it goes to the critic.

---

### `neural_mi/estimators/`

This is where the mathematical formulas for the different MI lower bounds are implemented.

- **`bounds.py`**: Contains the Python functions for `infonce`, `smile`, etc.

---

### `neural_mi/training/`

- **`trainer.py`**: Contains the `Trainer` class, which handles the entire PyTorch training loop: optimization, validation, early stopping, and checkpointing.

---

## How to... (A Contributor's Guide)

Here are some common development tasks and the files you would need to edit:

### Add a new MI estimator (e.g., a new lower bound)

1. Add the function for your new bound in `neural_mi/estimators/bounds.py`.
2. Register the new estimator's name in `neural_mi/run.py` in the `ParameterValidator`.

### Add a new data processor (e.g., for a new data type)

1. Create your new processor class in `neural_mi/data/processors.py`.
2. Register the processor's name in `neural_mi/data/handler.py`.

### Change the default neural network architecture

1. Modify the desired class in `neural_mi/models/critics.py` or `neural_mi/models/embeddings.py`.

### Add a new analysis mode

1. Create a new file in `neural_mi/analysis/` to contain the logic for your mode.
2. Import your new function into `neural_mi/run.py` and add a new `elif mode == 'your_new_mode':` block to call it.

---

## Testing Guidelines

When contributing new features, please ensure:

1. **All tests pass**: Run `pytest` before submitting a PR.
2. **High coverage**: New code should have near 100% test coverage. Check with `pytest --cov=neural_mi`.
3. **Type hints**: Use Python type hints for all function signatures.
4. **Documentation**: Add docstrings following the NumPy docstring format.

---

## Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and modular

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).