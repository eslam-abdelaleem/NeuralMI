# Under the Hood: A Developer's Guide to NeuralMI

This document provides a map of the `NeuralMI` codebase. It's intended for developers who want to contribute to the library or understand its internal architecture.

---

## Core Philosophy

The library is built around a central `run()` function (`neural_mi/run.py`) that acts as a controller. This function validates parameters, prepares the data, and then delegates the specific analysis to a dedicated module (e.g., `sweep`, `lag`, `rigorous`, `precision`). This keeps the main entry point clean and makes it easy to add new analysis modes.

---

## Codebase Structure

If you want to modify a specific part of the library, here's where to look.

### `neural_mi/run.py`

This is the main entry point. All user interactions start here. It handles parameter validation, backwards compatibility (e.g., deprecating old `processor_type` arguments in favor of separate `x` and `y` types), and dispatches tasks to the appropriate analysis modules.

---

### `neural_mi/analysis/`

This directory contains the logic for the different analysis `modes`.

- **`rigorous.py`**: Implements `mode='rigorous'` — contains `run_rigorous_analysis`, `AnalysisWorkflow`, and the internal helpers `_find_linear_region`, `_extrapolate_mi`, and `_post_process_and_correct`. Use `run_rigorous_analysis` as the primary entry point.
- **`workflow.py`**: Re-exports from `rigorous.py` for convenience. All analysis logic lives in `rigorous.py`.
- **`sweep.py`**: A general-purpose engine for running parallelized hyperparameter sweeps (`mode='sweep'`). It includes **"Smart Model Saving"** logic to dynamically generate safe filenames and prevent race conditions between parallel workers.
- **`lag.py`**: Contains the logic for `mode='lag'`, which is a specialized `sweep` over the `lag` parameter.
- **`dimensionality.py`**: Implements the `mode='dimensionality'` analysis. *Note: This module no longer uses sweeps. It orchestrates a single Hybrid Critic training run and triggers the SVD spectral metrics engine to compute the Participation Ratio.*
- **`precision.py`**: Implements the `mode='precision'` analysis. This module executes a "Train Once, Evaluate Many" pipeline, freezing a baseline network and sweeping over precision levels ($\tau$) using deterministic rounding or additive noise.
- **`task.py`**: A helper module that defines a single, runnable "task" (one training run of the MI estimator). It unpacks all top-level arguments and funnels them into the `Trainer`. Key implementation notes:
  - **`dataset_device` resolution**: the `dataset_device` param (default `'cpu'`) controls where dataset tensors live. `'auto'` resolves to the compute device; any explicit device string is forwarded directly.
  - **Module-level dataset cache**: static (`PairedDataset`) instances whose data pointer and construction params match a previous task are reused without re-running `create_dataset()`. This eliminates redundant tensor copies in sequential sweeps. Temporal datasets are excluded because they are mutated by `time_shift()` during training. The cache holds at most four entries (LRU eviction).

---

### `neural_mi/data/`

This directory handles all data preprocessing.

- **`handler.py`**: The `create_dataset` / `create_single_dataset` factory functions are the main interface here. Both accept a `data_device` parameter (default `'cpu'`) that controls where `self.data` tensors are stored inside the dataset objects. The compute device (`device`) is kept separately.
- **`static.py`** / **`temporal.py`**: All dataset classes store `self.data` on `self.data_device` (not on the compute device). This is the standard PyTorch pattern: data lives on CPU, batch loops call `.to(device)`. Changing where data lives at construction time has no effect on how training works.
- **`views.py`**: `SubsetView` index tensors are always stored as CPU LongTensors. `__getitem__` converts any 0-dim index tensor to a Python `int` before delegating to the dataset, making indexing device-agnostic (works whether `data_device` is `'cpu'` or an accelerator).

---

### `neural_mi/models/`

This directory defines all the PyTorch neural network architectures.

- **`critics.py`**: Contains the main critic architectures (`SeparableCritic`, `ConcatCritic`, and `HybridCritic`). 
  - *Crucial API:* All critics now expose a unified `get_embeddings(x, y)` method, allowing developers to easily extract chunked latent representations from saved models.
- **`embeddings.py`**: Defines the embedding networks that process the input data before it goes to the critic.  Standard models: `MLP`, `CNN1D`, `CNN2D`, `GRU`, `LSTM`, `TCN`, `Transformer`.  Physics-informed models: `SincEmbedding` (`'sinc_cnn'`), `SpikePhysicsEmbedding` (`'spike_physics'`), `PretrainedBackboneEmbedding` (`'pretrained_backbone'`).  To add a new model, subclass `BaseEmbedding`, register it in `build_critic()` in `utils.py`, add it to `ALLOWED_VALUES['embedding_model']` in `validation.py`, and add the relevant schema keys to `BASE_PARAMS_SCHEMA` in `defaults.py`.

---

### `neural_mi/estimators/`

This is where the mathematical formulas for the different MI lower bounds are implemented.

- **`bounds.py`**: Contains the Python functions for `infonce`, `smile`, etc.

---

### `neural_mi/training/`

- **`trainer.py`**: Contains the `Trainer` class, which handles the entire PyTorch training loop. This is the heavy-lifting engine of the library. Key architectural features to note:
  - **Memory-Safe Evaluation:** Uses $N^2$ flat-mapped chunking (`max_eval_samples`) to prevent Out-Of-Memory (OOM) crashes during full-dataset InfoNCE evaluation.
  - **Subset Tracking:** Locks in a `train_subset` to fast-track training MI without slowing down epochs.
  - **In-Memory Checkpointing:** Uses `copy.deepcopy` for early stopping to eliminate disk I/O bottlenecks.
  - **Spectral Engine:** Contains the `_extract_spectral_metrics` hook that computes cross-covariance SVD math for dimensionality modes.

---

## How to... (A Contributor's Guide)

Here are some common development tasks and the files you would need to edit:

### Add a new MI estimator (e.g., a new lower bound)

1. Add the function for your new bound in `neural_mi/estimators/bounds.py`.
2. Register the new estimator's name in `neural_mi/run.py` in the `ParameterValidator`.

### Add a new data processor (e.g., for a new data type)

1. Create your new dataset class in `neural_mi/data/temporal.py` (for temporal data) or `neural_mi/data/static.py` (for pre-processed data), subclassing `TemporalWindowDataset` or `BaseStaticDataset` respectively.
2. Register the new `proc_type` string in `neural_mi/data/handler.py`'s `create_single_dataset()` factory function with a matching `elif proc_type == '...'` branch.
3. Add the new type's allowed `processor_params` keys to `PROCESSOR_PARAMS_SCHEMA` in `neural_mi/defaults.py`.

### Change the default neural network architecture

1. Modify the desired class in `neural_mi/models/critics.py` or `neural_mi/models/embeddings.py`.

### Add a new analysis mode

1. Create a new file in `neural_mi/analysis/` to contain the logic for your mode.
2. Import your new function into `neural_mi/run.py` and add a new `elif mode == 'your_new_mode':` block to call it.

### Understand the `dataset_device` / `device` split

NeuralMI separates two concerns that are often conflated:

| Concept | Parameter | Default | Where used |
|---|---|---|---|
| **Compute device** — where model & optimizer live | `device` | `None` → auto-detect | `Trainer`, `run_training_task` |
| **Data storage device** — where `self.data` tensors live | `dataset_device` | `'cpu'` | `StaticDataset`, temporal datasets, `create_dataset` |

The `Trainer` already calls `.to(device)` on every batch and every evaluation call, so changing `dataset_device` from `'cpu'` to the compute device has no effect on training correctness — only on memory layout.

**Rule of thumb**: keep `dataset_device='cpu'` (the default) unless you have a good reason to co-locate data with the compute device (e.g. precision analysis, which evaluates the same dataset many times).

**Adding a new mode that runs many evaluations on the same dataset**: inject `dataset_device='auto'` as the default in your analysis function, following the pattern in `precision.py`:
```python
_data_device_raw = base_params.get('dataset_device', 'auto')
_data_device = str(device) if _data_device_raw == 'auto' else (_data_device_raw or 'cpu')
dataset = create_dataset(..., data_device=_data_device)
```

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