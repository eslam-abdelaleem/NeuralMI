# Internals

This page describes how the library is put together and where to change it. Using
NeuralMI needs none of it; [USING.md](USING.md) covers that.

## Contents

- [Codebase map](#codebase-map)
- [Design decisions](#design-decisions)
- [Extending the library](#extending-the-library)

---

## Codebase map

A call enters `neural_mi.run()`, which validates the settings, prepares the
streams, and hands the prepared data to the mode's producer. The producer trains
and combines the networks, the shared assembly layer turns its rows into a
`Results`, and a permutation test, when asked for, reruns the same producer on
moved data.

| file | holds |
|---|---|
| `neural_mi/run.py` | `run()`, which lowers the config objects into keyword arguments for `_run_flat()`; `_run_flat()`, which validates, applies defaults, windows or defers windowing, and calls the producer; the call-level checks (`_announce_call`, the processor-grid loop `_run_processor_grid`, the `_SHIFT_*_SAFE_MODES` tuples that say where window shifting reaches) |
| `neural_mi/config.py` | the config dataclasses, `Processing` through `Sweep`, and `as_config` |
| `neural_mi/defaults.py` | `BASE_PARAMS_SCHEMA`, `MODE_KWARGS_SCHEMA` and `PROCESSOR_PARAMS_SCHEMA`: every setting's type and default |
| `neural_mi/validation.py` | `ParameterValidator`, the allowed values, and the per-stream processor checks |
| `neural_mi/results.py` | the `Results` object: `get`, `summary`, `plot`, `compare`, `animate`, `save`, `load`, `to_dict`, `to_json` |
| `neural_mi/quantities.py` | the named quantities, on one engine: `_named` expands an iterable parameter into configurations, `_call` runs one configuration through `run()`, `_task` is the unit sent to a worker |
| `neural_mi/parallel.py` | `dispatch_tasks`, the one worker pool (`spawn` context) every parallel loop uses |
| `neural_mi/embeddings_io.py` | `extract_embeddings`, which reloads a saved critic |
| `neural_mi/utils.py` | `build_critic` and `build_optimizer_and_scheduler`, `cross_covariance_svd` (the one whitening and SVD behind the participation ratios and the rotated embeddings), `build_offset_arrays`, `compute_regime_diagnostic`, `get_device` |

**`neural_mi/analysis/`** holds the modes.

| file | holds |
|---|---|
| `modes.py` | one producer per mode and the `produce()` dispatcher. A producer takes the prepared data and the grid and returns the repeat rows, the per-configuration details and the columns to aggregate |
| `assemble.py` | `build_results` and `aggregate` (repeat rows into `runs` and `dataframe`), `merge_results` (the parts of a processor grid into one result), `convert_record` (the one conversion from nats), `NETWORK_KEYS` (what a trained network reports about itself) |
| `permutation.py` | `shift_x` (moving X for a null trial), `permutation_nulls` (running the trials), `attach_nulls` (the null columns and `p_value`) |
| `sweep.py` | `ParameterSweep`, the engine that trains one network per grid point and repeat; `amplification_factor` |
| `task.py` | `run_training_task`, one training run from parameters to a record, with a small cache of static datasets |
| `rigorous.py` | the gamma ladder, `_find_linear_region`, `_extrapolate_mi` and the fit diagnostics, for `mode='rigorous'` and for the rigorous difference quantities |
| `lag.py`, `precision.py`, `pairwise.py`, `dimensionality.py` | the mode-specific procedures the producers call |
| `conditional.py`, `interaction.py`, `transfer.py` | the component trainings of the difference quantities |
| `offsets.py` | the past and future builders of the named quantities, `grid_rows` (streams onto one grid, then rows) and `one_step_rows` (rows one time step wide, for transfer entropy) |

**`neural_mi/data/`** holds the preprocessing. `handler.py` has
`create_dataset` for any number of streams on one grid, `create_single_dataset`
for one stream, and the stream containers (`AlignedStreams`, `StreamBundle`,
`PairedTemporalDataset`, `PairedDataset`). `temporal.py` and `static.py` have
the per-processor datasets, `views.py` the `SubsetView` used for splits, and
`shift_windowing.py` both window-shifting mechanisms, with their reach and
guarantees described in its module docstring.

**`neural_mi/models/`** has the critics (`SeparableCritic`, `HybridCritic`,
`ConcatCritic`), the encoders and `VariationalWrapper` in `embeddings.py`, and
the reconstruction decoders in `decoders.py`. **`neural_mi/estimators/`** has
the bounds in `bounds.py`, registered by name in `ESTIMATORS`.
**`neural_mi/training/trainer.py`** has the training loop, the chunked
evaluation, the epoch selection and the spectral metrics.
**`neural_mi/visualize/`** has the plots and the training animation, and
**`neural_mi/generators/`** the synthetic data.

---

## Design decisions

### One entry point, one pipeline

Every mode goes through `run()` and the same pipeline. The call is
expanded into configurations, axis values, repeats and the networks of each
repeat. Each network is trained and yields one record. The records of a repeat
are combined: unchanged for a single network, as a signed sum for a difference
quantity, or by a weighted extrapolation for a rigorous fit. The repeats are then
aggregated into one `dataframe` row per configuration and axis value. A mode only
declares its networks and its combine step, in its producer, and the shape of
`Results` is the same for every mode.

### One unit conversion

Every network trains and reports in nats. `assemble.convert_record` converts the
values that are information, and only those, once, after training and before
anything is combined, so components, nulls, ceilings and fits all reach the
result in the caller's units.

### Processor parameters in a grid

A grid key that is a processor parameter, such as `window_size`, changes the data
itself. `sweep` and `lag` window inside each task and take such a key directly.
Every other mode prepares its data once per processor setting in
`_run_processor_grid`, runs the rest of the grid on each preparation, and merges
the parts with `merge_results`, so configurations keep the order of the full
grid. A processor key that no stream's processor reads is refused.

### The permutation null

A null trial reruns the mode's producer with X moved in time, through the same
`produce()` call as the observed value, so the null is built from exactly the
rows the observed value was built from. Moving the source leaves Y and W in
place, so Y's history in transfer entropy and the link between Y and W in the
conditional quantities survive. The trials go through `dispatch_tasks` like every
other task.

### Parallelism and seeds

Every loop of independent work, whether configurations, repeats, lags, pairs,
rigorous chunks, splits or permutation trials, goes through `dispatch_tasks`.
When an outer loop runs in parallel, any loop inside one of its units runs with
`n_workers=1`, so a pool never starts inside a pool. Each task seeds itself from
the call's `seed` and a key fixed by its position in the grid, so the result does
not depend on which worker ran it or when.

### Tensors are three-dimensional

Inside the library every stream is `(n_samples, n_channels, window)`. Unwindowed
2-D input gains a trailing axis of 1, and image-like 4-D input passes through
unchanged.

### Blocked splits

The default split holds out `n_test_blocks` contiguous stretches spread over the
recording and excludes a gap of `gap_fraction` of a block on either side of each
from training, so neighbouring, correlated windows cannot sit on both sides of
the split. A random split suits independent samples only.

### Rigorous fits in $\gamma$

The ladder trains on chunks of $N/\gamma$ samples. For fixed $N$ the bias
$a/N_\text{chunk} = (a/N)\gamma$ is linear in $\gamma$, so the fit and the
extrapolation to $\gamma = 0$ work in $\gamma$ ([THEORY.md](THEORY.md#correcting-it)).
The dependent variable is each chunk's training-side estimate, the same quantity
every other mode reports.

### Evaluation on clean, unshifted data

Augmentations apply inside the training batch loop only, and evaluation reads a
snapshot of the data taken before any window shift. The reported values
therefore never depend on an augmentation or on the last shift drawn. Within a
batch the image augmentations run first, then the others, then the custom ones.

---

## Extending the library

To add an estimator, put the bound in `neural_mi/estimators/bounds.py` and
register its name in `ESTIMATORS` in `neural_mi/estimators/__init__.py`.

To add a processor, subclass `TemporalWindowDataset` in `neural_mi/data/temporal.py`,
or `BaseStaticDataset` in `neural_mi/data/static.py` for unwindowed data. Add a
branch for its name in `create_single_dataset()` in `neural_mi/data/handler.py`
and its parameters to `PROCESSOR_PARAMS_SCHEMA` in `neural_mi/defaults.py`, then
document the parameters in [PARAMETERS.md](PARAMETERS.md), where
`tests/test_docs_drift.py` checks them against the schema.

To add an encoder, subclass `BaseEmbedding`, register it in `build_critic()` in
`neural_mi/utils.py`, add its name to the allowed values of `embedding_model` in
`neural_mi/validation.py`, and add any new settings to `BASE_PARAMS_SCHEMA`.

To add an analysis mode:

1. Write the procedure, in a module under `neural_mi/analysis/` when it is more
   than a few lines.
2. Add a producer to `neural_mi/analysis/modes.py` that returns its repeat rows
   through `_produced(...)`, naming its axis keys and the columns to aggregate,
   and route the mode to it in `produce()`.
3. Add its config dataclass to `neural_mi/config.py` and export it from
   `neural_mi/__init__.py`, and add its settings and defaults to
   `MODE_KWARGS_SCHEMA` in `neural_mi/defaults.py`.
4. Register it in `neural_mi/run.py`: in `_MODES` and `_MODE_CONFIG_CLASSES`, in
   `_PERMUTABLE_MODES` if a permutation null means something for it, in
   `_WINDOWS_IN_TASK` if it windows inside its tasks, and in the
   `_SHIFT_*_SAFE_MODES` tuples that apply.
5. Decide what it does with repeats, a configuration grid and `rigorous`, add the
   decision to the capability table in [USING.md](USING.md#what-each-mode-accepts),
   and add the mode to `tests/test_contract.py`, which checks that every mode
   returns the shared `Results` shape or refuses clearly.
6. Document its config in [PARAMETERS.md](PARAMETERS.md) and the messages it
   prints in [MESSAGES.md](MESSAGES.md). `tests/test_docs_drift.py` checks both.
