# Testing

This page describes how the test suite is laid out and where to look when something breaks.

## Contents

- [Running the tests](#running-the-tests)
- [Test map](#test-map)
- [Where to look when something breaks](#where-to-look-when-something-breaks)
- [Principles](#principles)

---

## Running the tests

```bash
pip install -e ".[test]"      # or ".[dev]" for every optional part

pytest                        # the whole suite
pytest -m "not slow"          # skip the long accuracy runs
pytest -n0 -v                 # serial, for a debugger
pytest tests/test_run.py -v   # one file
pytest tests/test_run.py::test_run_estimate_mode_returns_results_with_float -v
pytest --cov=neural_mi        # coverage report
```

With `addopts = -n auto` in `pytest.ini`, pytest-xdist spreads the tests across
cores by default. Because the suite is training-bound, running in parallel cuts
wall time several-fold. Pass `-n0` to turn it off for `pdb`.

The long accuracy runs that need real convergence to assert a result carry
`@pytest.mark.slow`. Under `-n auto` the longest single test sets the floor on
wall time. Deselecting the slow tests shortens a run substantially and still
exercises every module. `pytest --collect-only -m slow` lists those tests and
`--durations=20` ranks the slowest ones.

CI runs the suite with the latest release of every dependency on Python 3.11 and
3.12. A second job pins every dependency at the floor stated in `pyproject.toml`
and runs the suite on Python 3.11. A change that needs a newer release fails
there first.

---

## Test map

Each file covers one functional area. Run
`pytest --collect-only -q` for the current totals.

### The shared result and the reference pages

| file | covers |
|---|---|
| `test_contract.py` | every mode with no grid, with repeats, with a configuration grid, with both, with a processor parameter in the grid, and with `rigorous=True` where it exists: each call returns the shared `Results` shape or refuses with a reason. Also embeddings kept per repeat, and the `n_workers` warning |
| `test_docs_drift.py` | `reference/PARAMETERS.md` against every config field, default and processor parameter, and `reference/MESSAGES.md` against every message the package emits, in both directions |
| `test_settings_are_read.py` | every declared setting and every config field reaches code that reads it. A setting that is accepted and then ignored fails. |
| `results_factory.py` | not a test: builds `Results` objects of any mode through the same `build_results` every mode uses, for the tests of `Results` and its plots |

### Estimators and bounds

| file | covers |
|---|---|
| `test_estimators.py` | the InfoNCE and SMILE bound functions, accuracy on known-MI data, SMILE's `clip` through the full pipeline |

### Data

| file | covers |
|---|---|
| `test_data_processors.py` | window construction for continuous, spike and categorical data, time shifting, noise, rounding, every split mode, and mixed-modality alignment |
| `test_shift_windowing.py` | the `unfold` reslice route: shift families, step-size resolution, categorical re-encoding parity with `CategoricalWindowDataset`, differing sample rates, cross-unit warnings |
| `test_augmentations.py` | each augmentation's effect and shape contract, the spatial-on-3-D warning, custom callables, application order, and the boolean shortcuts |
| `test_empty_windows.py` | `drop_empty_windows`, its independence from the continuous coverage rule, retention reporting, and the mixed-unit warning |
| `test_views.py` | `SubsetView` by index, channel and time region, and temporal index conversion |

### Generators and oracles

| file | covers |
|---|---|
| `test_oracle.py` | `SharedLatentGaussian`: the exact identities, sampling, SNR, error paths, and agreement between the offset builder and the named quantities |
| `test_generators.py` | every synthetic generator in numpy and torch modes |

### Models

| file | covers |
|---|---|
| `test_models.py` | every encoder, `VariationalWrapper` for each, all three critics, chunking equivalence, gradients, `get_embeddings()` |
| `test_critic_scoring.py` | the block scoring of the hybrid and concat critics against the pair-by-pair route, scores and gradients |
| `test_cnn2d.py` | the 2-D encoder, its `build_critic` path, 4-D input handling, the image splits of `mode='dimensionality'` |
| `test_dual_branch_embedding.py` | `DualBranchEmbedding`, its integration through `run()`, the quantities that require it, and accuracy against the oracle |
| `test_decoders.py` | decoder dispatch and the reconstruction-loss path end to end |
| `test_shared_encoder.py` | weight identity, parameter count, the concat incompatibility, the dimensionality-mode default |
| `test_pretrained_backbone.py` | the spatial-mismatch warning, and that the backbone stays frozen with BatchNorm in eval through training |

### Training

| file | covers |
|---|---|
| `test_trainer.py` | the training loop, chunked evaluation, spectral tracking, custom smoothing |
| `test_amp_and_names.py` | mixed-precision training and named-variable propagation into results |

### The `run()` API

| file | covers |
|---|---|
| `test_safety.py` | safety-critical regressions: shape errors, defaults, clamping warnings, the NaN-streak `TrainingError`, sweep guards, and the mode/option combinations that are defined but easy to leave unexercised |
| `test_validation.py` | `ParameterValidator` and `DataValidator`, plus integration-level checks through `nmi.run` |
| `test_config.py` | the typed config objects and how they lower into `base_params` |
| `test_run.py` | mode routing, the continuous processor pipeline, spike-data rigorous end to end, and that every warning points at the caller's line |
| `test_repeated_messages.py` | repeated warnings and log lines shown once with a count, across worker processes and at the caller's line |
| `test_run_config_api.py` | the config-object call surface of `run()` |

### Named quantities

| file | covers |
|---|---|
| `test_quantities.py` | offset shapes, every convenience function, transfer entropy and its conditional form, sweeps, accuracy against the oracle |
| `test_quantities_sweep.py` | sweeping a quantity's construction parameter, including `mi_rate` over either window |

### Analysis modes

| file | covers |
|---|---|
| `test_dimensionality.py` | the grid chosen from the reference fit, the curve and its reading against curves of known dimension, the early stop and the extension, the median over splits, every warning, the settings and refusals, every split method, and a run through `run()` |
| `test_interaction.py` | interaction information's plumbing and accuracy, and its shift routes across categorical, mixed and spike pairings |
| `test_permutation.py` | the permutation null: X moved by a circular shift or reordered blocks for windowed rows, raw series and spike times, one null per row, `p_value`, the default trial count, and the modes that compute no null |
| `test_conditional.py` | CMI independence and correlation properties, the component details, categorical conditioning variables |
| `test_rigorous_diagnostics.py` | fit diagnostics, their presence in corrected results, scalar rigorous analysis, chunk-range translation |
| `test_sweep.py` | sweep mechanics: dimension inference, subsampling, processor-param sweeps, save-path collisions |
| `test_transfer.py` | transfer entropy in both directions and the directionality index |
| `test_conditional_transfer_rigorous.py` | conditional MI and transfer entropy through the rigorous path, including unit conversion |
| `test_amplification_factor.py` | the amplification arithmetic and the warnings that report it |
| `test_analysis.py` | lag mode across data types, spectral metrics, task routing |
| `test_pairwise.py` | self- and cross-pairwise matrices, columns, finiteness |
| `test_precision.py` | the precision sweep under rounding and noise |
| `test_workflow_internals.py` | the rigorous helpers: linear-region detection, extrapolation, bias correction |

### Parallelism and reproducibility

| file | covers |
|---|---|
| `test_reproducibility.py` | seeded reproducibility in estimate, sweep and rigorous modes, and under parallelism |
| `test_parallel.py` | `dispatch_tasks`: empty and single-task shortcuts, order preservation, and the parallel path matching the sequential one |

### Utilities

| file | covers |
|---|---|
| `test_utils.py` | device selection, `build_critic`, the cross-covariance spectrum, participation ratio, effective rank |

### Results and visualisation

| file | covers |
|---|---|
| `test_results.py` | `Results` across modes: `get`, `summary`, `plot`, `save`, `load`, `to_dict` and `to_json` |
| `test_animate.py` | panel selection, reducer fitting, scatter colouring, and `animate_training` across its panel and label options, including `Results.animate()` |
| `test_embedding_extraction.py` | `return_embeddings`, shapes, model saving, `extract_embeddings()`, `plot_embeddings()` |
| `test_visualize.py` | the publication style, sweep curves, the bias-correction fit, per-mode plot dispatch for estimate, dimensionality, conditional, transfer and rigorous, comparison and the reliability annotation |

---

## Where to look when something breaks

| area | files |
|---|---|
| a wrong estimate or the estimator maths | `test_estimators.py`, `test_oracle.py` |
| a column, row or field of `Results` missing or misplaced | `test_contract.py`, `test_results.py` |
| a reference page that no longer matches the code | `test_docs_drift.py` |
| a permutation null or p-value | `test_permutation.py` |
| windowing, alignment or a split | `test_data_processors.py`, `test_shift_windowing.py` |
| an encoder's output shape or gradients | `test_models.py`, `test_cnn2d.py`, `test_dual_branch_embedding.py` |
| the training loop | `test_trainer.py` |
| a `run()` API error | `test_run.py`, `test_validation.py`, `test_config.py` |
| a named quantity | `test_quantities.py`, `test_quantities_sweep.py` |
| conditional MI, transfer entropy, interaction information | `test_conditional.py`, `test_transfer.py`, `test_interaction.py` |
| rigorous mode or bias correction | `test_workflow_internals.py`, `test_rigorous_diagnostics.py`, `test_conditional_transfer_rigorous.py` |
| dimensionality | `test_dimensionality.py` |
| amplification or a negative difference | `test_amplification_factor.py` |
| parallel dispatch or a result that moved between runs | `test_parallel.py`, `test_reproducibility.py` |
| plotting | `test_visualize.py`, `test_results.py`, `test_animate.py` |

---

## Principles

Most tests train for 2 to 10 epochs on small batches. The accuracy runs that
need real convergence carry `@pytest.mark.slow`.

Any test sensitive to initialisation seeds both numpy and torch explicitly.
`test_reproducibility.py` checks that contract.

Routing-only tests mock the training engine and check mode dispatch without
paying for a fit.

Shared data fixtures sit at module level or in `conftest.py`. The
`gaussian_data` and `raw_gaussian_data` fixtures in `test_run.py` are reused
widely. A test of `Results` or of a plot builds its result with
`results_factory.py` and trains nothing.
