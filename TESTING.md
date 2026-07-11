# NeuralMI — Testing Guide

This document describes the test suite structure, how to run tests, and where to look when something breaks.

---

## Running the Tests

```bash
# Install development dependencies first (if not already installed)
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt

# Run the full suite
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific file
pytest tests/test_estimators.py -v

# Run a specific test
pytest tests/test_run.py::test_run_estimate_mode_returns_results_with_float -v

# Run fast tests only (exclude slow integration tests)
pytest tests/ -m "not slow" -v
```

---

## Test File Map

The test suite is organized into **22 files** grouped by functional area:

### 1. Core Estimators & Bounds
| File | What it covers |
|------|---------------|
| `test_estimators.py` | InfoNCE and SMILE bound functions, estimator accuracy on known-MI data, SMILE `clip` parameter in full pipeline |

### 2. Data Processors & Handlers
| File | What it covers |
|------|---------------|
| `test_data_processors.py` | Window creation (continuous, spike, categorical), time shifting, noise application, precision/rounding, train/test splitting modes (random, blocked, custom indices), mixed-modality alignment (continuous + spike streams) |
| `test_views.py` | `SubsetView` by index, channel, and time region; temporal view index conversion |

### 3. Data Generators
| File | What it covers |
|------|---------------|
| `test_generators.py` | All synthetic generators in both numpy and torch output modes: correlated Gaussians, nonlinear latent, XOR, temporally convolved, spike trains, categorical series, event-related, linear, nonlinear, history, and full-data generators |

### 4. Models & Architecture
| File | What it covers |
|------|---------------|
| `test_models.py` | Embedding networks (MLP, CNN1D, GRU, LSTM, TCN, Transformer), physics-informed models (CNN1D depthwise, SincEmbedding, SpikePhysicsEmbedding, PretrainedBackboneEmbedding), `VariationalWrapper` for all encoder types, critic architectures (Separable, Concat, Hybrid), chunking equivalency, gradient computation, `get_embeddings()` |
| `test_shared_encoder.py` | Phase F: shared/siamese encoder — weight identity, parameter count, concat+shared incompatibility, dimensionality-mode default, `run()` shortcut |

### 5. Training
| File | What it covers |
|------|---------------|
| `test_trainer.py` | Core training loop, chunked evaluation (OOM prevention), spectral metric tracking, custom smoothing |

### 6. The `run()` API & Integration
| File | What it covers |
|------|---------------|
| `test_run.py` | All mode routing (estimate, sweep, rigorous, dimensionality, precision, custom critic, continuous processor pipeline, processor-param sweep), spike-data rigorous end-to-end |

### 7. Analysis Modes
| File | What it covers |
|------|---------------|
| `test_analysis.py` | Lag mode (continuous, categorical, spike), dimensionality (random/spatial/temporal splits), spectral metrics (participation ratio, effective rank), task parameter routing |
| `test_precision.py` | Precision sweep — rounding and noise corruption methods, full pipeline |
| `test_conditional.py` | CMI independence/correlation properties, `mi_xz_y`/`mi_z_y` details dict |
| `test_transfer.py` | Transfer entropy unidirectional and bidirectional; `te_xy`, `te_yx`, directionality index |
| `test_pairwise.py` | Self-pairwise and cross-pairwise MI matrices, DataFrame columns, finiteness |
| `test_permutation.py` | Permutation testing, `null_distribution` in details, default `n_permutations`, rigorous-mode incompatibility |
| `test_sweep_extended.py` | Sweep mechanics — dimension inference, concat+embedding_dim error, `max_samples` subsampling, processor-param sweep, save-path name collision |
| `test_workflow_internals.py` | Rigorous workflow helpers — linear region detection, MI extrapolation, bias correction |

### 8. Validation & Safety
| File | What it covers |
|------|---------------|
| `test_validation.py` | `ParameterValidator` and `DataValidator` (shape, NaN, type), integration-level checks via `nmi.run` (unknown params, type/value errors, invalid processor params, defaults logging) |
| `test_safety.py` | Phase A safety fixes — 3-D input error for transfer mode, beta default, train_subset_size clamping warning, NaN-streak `TrainingError`, concat+embedding_dim sweep error |

### 9. Utilities
| File | What it covers |
|------|---------------|
| `test_utils.py` | Device auto-selection, critic building (`build_critic`), cross-covariance spectrum, participation ratio, effective rank |

### 10. Results & Visualization
| File | What it covers |
|------|---------------|
| `test_results_extended.py` | `Results.__repr__`, `.plot()`, `.summary()` across estimate/sweep/rigorous modes |
| `test_visualize.py` | Publication style, sweep curve plotting, bias-correction fit plot, Results plot dispatcher |
| `test_embedding_extraction.py` | Phase G — `return_embeddings` in estimate mode, embedding shape, model saving (extended format), `extract_embeddings()`, `plot_embeddings()` with PCA/color options |

---

## Where to Look When Something Breaks

| Area | Relevant test file(s) |
|------|----------------------|
| MI estimate is wrong / estimator math | `test_estimators.py` |
| Data windowing / alignment | `test_data_processors.py` |
| Split mode issues | `test_data_processors.py` (splitting section) |
| Model output shape or gradient issues | `test_models.py` |
| Training loop errors | `test_trainer.py` |
| `nmi.run()` API errors | `test_run.py`, `test_validation.py` |
| Lag analysis | `test_analysis.py` |
| Precision analysis | `test_precision.py` |
| Conditional MI | `test_conditional.py` |
| Transfer entropy | `test_transfer.py` |
| Pairwise matrix | `test_pairwise.py` |
| Rigorous / bias correction | `test_run.py`, `test_workflow_internals.py` |
| Sweep mechanics | `test_sweep_extended.py` |
| Parameter validation errors | `test_validation.py`, `test_safety.py` |
| Visualization / plotting | `test_visualize.py`, `test_results_extended.py` |
| Embedding extraction | `test_embedding_extraction.py` |

---

## Key Testing Principles

1. **Speed**: Most tests use 2–10 training epochs and small batch sizes. Slow integration tests (spike data rigorous) are kept minimal.
2. **Reproducibility**: Tests that are sensitive to random initialization set both `numpy` and `torch` seeds explicitly.
3. **Mocking**: Tests for routing-only logic (rigorous, precision mode dispatch) mock the underlying engine to avoid slow training.
4. **Fixtures**: Shared data fixtures are defined at the module level or in `conftest.py`. The `gaussian_data` and `raw_gaussian_data` fixtures in `test_run.py` are reused by many tests.
