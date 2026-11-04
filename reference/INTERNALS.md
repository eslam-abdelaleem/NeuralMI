# Internals

This page describes how the library is put together and where to change it. Using
NeuralMI needs none of it; [USING.md](USING.md) covers that.

## Contents

- [Codebase map](#codebase-map)
- [Design decisions](#design-decisions)
- [Extending the library](#extending-the-library)

---

## Codebase map

`neural_mi.run()` validates the settings of a call, prepares the streams and
hands the prepared data to the mode's producer. The producer trains and combines
the networks and returns rows that the shared assembly layer turns into a
`Results`. A permutation test reruns the same producer on moved data.

| file | holds |
|---|---|
| `neural_mi/run.py` | `run()` (lowers the config objects into keyword arguments for `_run_flat()`); `_run_flat()` (validates, applies defaults, builds the call's row plan and cuts its rows once or leaves them to the tasks, and calls the producer); the call-level checks (`_announce_call`, the processor-grid loop `_run_processor_grid`, `_warn_if_shift_windows_dead`) |
| `neural_mi/config.py` | the config dataclasses, `Processing` through `Pairwise`, and `as_config` |
| `neural_mi/defaults.py` | `BASE_PARAMS_SCHEMA`, `MODE_KWARGS_SCHEMA` and `PROCESSOR_DEFAULTS` (with `PROCESSOR_PARAMS_SCHEMA`, the names it declares): every setting's type and default |
| `neural_mi/validation.py` | `ParameterValidator`, the allowed values, and the per-stream processor checks |
| `neural_mi/results.py` | the `Results` object: `get`, `summary`, `plot`, `compare`, `animate`, `save`, `load`, `to_dict`, `to_json` |
| `neural_mi/quantities.py` | the named quantities, on one engine: `_named` expands an iterable parameter into configurations, `_call` runs one configuration through `run()`, `_task` is the unit sent to a worker |
| `neural_mi/device.py` | `choose_device` and `with_device`: the device a call trains on when it names none, chosen by timing the first network on the CPU and, when the call is long enough, on the GPU |
| `neural_mi/parallel.py` | `worker_pool`, the one worker pool every parallel loop opens (`spawn` context, the parent's logging level, a share of the CPU threads per worker), and `dispatch_tasks` |
| `neural_mi/embeddings_io.py` | `extract_embeddings` (reloads a saved critic) |
| `neural_mi/utils.py` | `build_critic` and `build_optimizer_and_scheduler`, `cross_covariance_svd` (the one whitening and SVD behind the participation ratios and the rotated embeddings), `build_offset_arrays`, `get_device` |

**`neural_mi/analysis/`** holds the modes.

| file | holds |
|---|---|
| `modes.py` | one producer per mode and the `produce()` dispatcher. A producer takes the prepared data and the grid and returns the repeat rows, the per-configuration details and the columns to aggregate |
| `assemble.py` | `build_results` and `aggregate` (repeat rows into `runs` and `dataframe`), `merge_results` (the parts of a processor grid into one result), `convert_record` (the one conversion from nats), `NETWORK_KEYS` (what a trained network reports about itself) |
| `permutation.py` | `shift_x` (moving X for a null trial), `permutation_nulls` (running the trials), `attach_nulls` (the null columns and `p_value`) |
| `sweep.py` | `ParameterSweep`, the engine that trains one network per grid point and repeat; `amplification_factor` |
| `task.py` | `run_training_task` (one training run from parameters to a record) and a small cache of static datasets |
| `rigorous.py` | the $\gamma$ ladder, `_find_linear_region`, `_extrapolate_mi` and the fit diagnostics, for `mode='rigorous'` and for the rigorous difference quantities |
| `lag.py`, `precision.py`, `pairwise.py`, `dimensionality.py` | the mode-specific procedures the producers call |
| `conditional.py`, `interaction.py`, `transfer.py` | the component trainings of the difference quantities |
| `offsets.py` | the one path to the rows of every quantity defined by time offsets: `build_rows` (the rows a spec defines, through `utils.build_offset_arrays`, and their geometry), `one_step_series` (streams onto one grid of rows one step wide that tile time, the one route of `processing=`) and `grid_rows` |

**`neural_mi/data/`** holds the preprocessing. `plan.py` has `RowPlan`, the
one plan of every stream a quantity reads, from which every task cuts its rows
and every term selects its streams (see [The row plan](#the-row-plan) below).
`handler.py` has `create_dataset` for any number of streams on one grid,
`create_single_dataset` for one stream, `one_grid` (the grid's one window and
step), and the stream containers (`AlignedStreams`, `StreamBundle`,
`PairedTemporalDataset`, `PairedDataset`). `temporal.py` and `static.py` have
the per-processor datasets, `views.py` the `SubsetView` used for splits, and
`shift_windowing.py` the windowing by sample number behind a plan's moves and
the phase shifter of rows a stride thins.

**`neural_mi/models/`** has the critics (`SeparableCritic`, `HybridCritic`,
`ConcatCritic`), the encoders and `VariationalWrapper` in `embeddings.py`, and
the reconstruction decoders in `decoders.py`. `BranchedEmbedding` is the X
encoder of every term of a difference quantity. `_joint_marginal_difference`
(`analysis/sweep.py`) marks both terms with `_difference_term` and trains the
conditioning term first. The joint term's X is the pair (rest of X,
conditioning set), packed by the plan (`pack` in `data/plan.py`) or by the
quantity that builds its rows by offset, and `build_critic` builds one branch
per part of X from the call's encoder. Every stage that reads X takes the pair:
`PairDecoder`, the augmentations, `StaticDataset`'s corruptions,
`ConcatCritic` and the nonparametric estimator. **`neural_mi/estimators/`** has
the bounds in `bounds.py`, registered by name in `ESTIMATORS`, and KSG in
`ksg.py`, registered in `NONPARAMETRIC` with its conditional form in
`NONPARAMETRIC_CONDITIONAL`.
**`neural_mi/training/trainer.py`** has the training loop, the chunked
evaluation, the epoch selection and the spectral metrics.
**`neural_mi/visualize/`** has the plots and the training animation.
**`neural_mi/generators/`** has the synthetic data.

---

## Design decisions

### One entry point, one pipeline

Every mode goes through `run()` and the same pipeline. The call is expanded into
configurations, axis values, repeats and the networks of each repeat. Each
network is trained and yields one record. The records of a repeat are combined:
unchanged for a single network, as a signed sum for a difference quantity, or by
a weighted extrapolation for a rigorous fit. The repeats are then aggregated
into one `dataframe` row per configuration and axis value. A mode only declares
its networks and its combine step in its producer. The shape of `Results` is the
same for every mode.

### One place for a default

Every setting's default is written once, in `BASE_PARAMS_SCHEMA`
(`neural_mi/defaults.py`). Code below `run()` reads a setting with
`defaults.setting(params, name)`, which falls back to that default, and every
signature default of a setting, in `Trainer`, the encoder and decoder classes,
is `defaults.default_of(name)`. `tests/test_schema_defaults.py` checks the
signatures.

`build_critic` reads only the settings in `utils.ARCHITECTURE_KEYS`, and
`save_network` saves exactly those with a network, so `extract_embeddings`
rebuilds it as it was trained. A value pickle cannot store, such as a class
defined inside a function, is saved as its name, and `extract_embeddings` asks
for it in `base_params`.

### One unit conversion

Every network trains and reports in nats. `assemble.convert_record` converts the
information-valued entries once, after training and before anything is combined.
Components, nulls, ceilings and fits therefore all reach the result in the
caller's units.

### Values that cannot be negative

A network's value is set to 0 in the trainer when its held-out MI never rose
above zero or its training-side value came out negative. The measured value
stays in `raw_train_mi`. A combined value gets the same rule where it is formed,
through `modes._floor_at_zero`: in `_difference_rows` for conditional MI and
transfer entropy, in `_fit_row` for an extrapolation, and in the precision rows.
`modes._floor_at_zero` keeps the measured value as `<key>_raw`. `assemble.aggregate` averages the
measured values (`mi_raw`, or `<col>_raw` for a column in `mean_cols`) wherever a
repeat carries one, and floors only the mean. Elsewhere it leaves the values
reported as 0 out of `mi_mean` and `mi_std` and counts them in `n_zero`. The
permutation test summarises each trial with `aggregate` too. A rung of a
difference's rigorous ladder is its signed measured value (`sweep.rung`), and
`rigorous._drop_rungs` leaves out only the rungs reported as 0, whose networks
learned nothing, the NaN rungs, whose term failed, and the rungs whose network,
or for a difference whose joint term (`sweep.rung`), has a held-out curve that
did not resolve (`curve_diagnosis == 'not resolved'`), with every $\gamma$ that
loses half or more of its rungs to them.
Interaction information is signed and is left as measured.

Every quantity combined from separately trained networks goes through
`sweep.combine_terms`: `_joint_marginal_difference` (conditional MI, transfer
entropy and the quantities built on them), interaction information and the runs
rows of `modes._difference_rows`. The rungs of their rigorous ladders call the
same functions on each chunk. `sweep.term_values` reads each network as
measured, as having learned nothing (it enters as 0) or as failed (its repeat
is NaN). The line between the last two is `epochs.stands_above_zero`, the same
test the trainer uses to word its warning about a negative training-side value,
so a network's warning and its place in a combination agree. Before it reads a
value, `combine_terms` compares the rows each term's task read
(`data.rows.row_identity`, in the task's result under `_row_identity`, which a
plan's dataset carries from `RowPlan.identity`): their count, and the grid's
origin and step, a digest of the kept rows' start times and Y's window wherever
both tasks know them (`data.rows.same_rows`). A repeat whose terms read
different rows is refused: the check that every term read its plan's rows.

### Processor parameters in a grid

A grid key that is a processor parameter, such as `window_size`, changes the
data itself. `sweep` and `lag` window inside each task and take such a key
directly. Every other mode prepares its data once per processor setting in
`_run_processor_grid`, runs the rest of the grid on each preparation and merges
the parts with `merge_results`. Configurations keep the order of the full grid.
A processor key that no stream's processor reads is refused.

### The permutation null

A null trial reruns the mode's producer with X moved in time through the same
`produce()` call as the observed value. The null is built from exactly the rows
the observed value was built from. Moving the source leaves Y and W in place.
Y's history in transfer entropy and the link between Y and W in the conditional
quantities survive. The trials go through `dispatch_tasks` like every other
task.

### Parallelism and seeds

Every loop of independent work, whether configurations, repeats, lags, pairs,
rigorous chunks, splits or permutation trials, runs in a `spawn` pool when
`n_workers` is above 1. Every pool opens through `parallel.worker_pool`; the
named quantities dispatch to one with `parallel.dispatch_tasks`. Each pool wraps its task function in
`logger.CapturedTask` and reads the results through `logger.released`. A worker
then hands back the warnings and log messages of its task, and they are raised
again in the calling process at the caller's line. `logger.collect_repeats`
around every `run()`, named quantity and `create_dataset()` shows each kind of
message once per call and counts the rest. In IPython the count is shared by every call of a cell. On import,
`logger._group_repeats_by_cell` registers `pre_run_cell` and `post_run_cell`
hooks that open and close it. `grouped_warnings()` opens the same shared count
over a block. The count of a repeated warning is raised as a warning of the same
category, and the count of a log message is logged at its level.

A message that needs the user's attention (a result that may be off, a setting
not used as given, data the library changed) is a `warnings.warn` with a
category, and is documented in `reference/MESSAGES.md`. Logging carries only
what the library did, at the info or debug level, and errors. `logger.warning`
is not used, which `test_docs_drift.py` checks.

When an outer loop runs in parallel, any loop inside one of its units runs
with `n_workers=1` so that a pool never starts inside a pool. Each task seeds itself
from the call's `seed` and a key fixed by its position in the grid so that the
result does not depend on which worker ran it or when.

### Tensors are three-dimensional

Inside the library every stream is `(n_samples, n_channels, window)`. Unwindowed
2-D input gains a trailing axis of 1. Image-like 4-D input passes through
unchanged.

### Blocked splits

The default split holds out `n_test_blocks` contiguous stretches spread over the
recording and excludes a gap of `gap_fraction` of a block on either side of each
from training. Neighbouring correlated windows cannot then sit on both sides of
the split. A random split suits independent samples only.

### Preprocessing and the data check

`neural_mi/preprocessing.py` holds every transform of the data: the channel
transforms, the autoregressive whitening and the interface a transform of your
own follows. `utils._whiten` (the embedding spectrum) and `ksg.prepare` use the
same `ZScore`, `ZCA` and `Copula`. `preprocessing.prepare` applies a
`Preprocessing` to the streams of a call and runs the data check on the result,
and every path takes it once, before rows are built: `_run_flat` on the raw
streams for every mode but transfer entropy, the transfer branch on the series
its histories come from (after any grid), and `quantities._by_offsets` on a
named quantity's series, before `offsets.build_rows`. The named quantities take
`preprocessing` out of what they pass to `run()`, which receives rows and skips
its own check when `_rows` says a quantity built them.

### Rows in time order

Whatever builds rows from a time series records their geometry once, in the
parameters every task receives (`neural_mi/data/rows.py`): `row_step`, the
distance between the starts of consecutive rows, and `row_span`, the stretch one
row covers, in one unit. A processor records its windows (`run.py` when it
windows once for the call, `task.py` and `shift_windowing.py` when a task
windows for itself). Rows built by offset get their geometry from the spec
that builds them (`offsets.build_rows` and `offsets.spec_geometry`): a named
quantity or `offset_quantity` hands it to `run()` through the private `_rows`
argument, and transfer entropy records it for its histories (`transfer.te_spec`).
Rows a `stride` above one thins are built at every step when windows shift,
and `_row_phases` tells each task to keep one in every `stride`
(`shift_windowing.RowPhaseShifter`, which moves the phase each epoch). The
record still gives the kept rows' step, `data.rows.kept_rows` their count.
Rows without the record are not
known to be in time order: arrays passed in already built, or samples in no
order. The record is reported per network in `runs` and
`details[config_id]['trainings']`. A plan records its own grid's geometry
(`RowPlan.geometry`).

Everything that depends on the time order reads the record and nothing else.
The trainer's blocked split warns when its gap is shorter than a row's span,
and its random split warns when rows share samples, as does the split
`mode='dimensionality'` shares across its fits. The rigorous ladder cuts
recorded rows into contiguous chunks through `rigorous._Chunker`, the one
chunker of both rigorous paths. A block shuffle moves blocks of the rows one
row's span takes in. Whether rows move each epoch is a different question,
which the trainer answers from the shifter a dataset carries.

### The row plan

`run()` builds one `RowPlan` per call and processor configuration over every
stream the call reads (`data/plan.py`). Each stream carries its processor and
every setting, resolved once from the call's parameters and the schema
(`resolve_stream`), and the grid has the window and step of `one_grid`. A plan
cuts its rows one of two ways (`RowPlan.route`): streams on a regular clock that
start together are windowed by sample number (`shift_windowing.BundleShifter`),
and any other plan puts every stream on one grid in time, from the latest start
to the earliest end, through the processors' own datasets, and keeps a window
only where every stream is valid. A plan of rows built already (`'rows'`) packs
them as given.

A term reads the plan through `TermRows`, the plan and its `Roles`: which
streams make its X (joined along the channels or as a pair) and which its Y.
Every term of a difference therefore reads the same rows, the same kept
windows and the same per-stream fits, and `task.py` builds every raw dataset
this way: a term's from its `TermRows`, a pair's from `RowPlan.of_pair`. A plan
that does not shift is cut once in `run()` (`RowPlan.materialize`) and its rows
reach the tasks built. A plan that shifts reaches each task raw, and its
dataset carries a shifter whose one offset per epoch, drawn within one step,
moves every stream: by sample number, or on the grid in time by windowing
every stream again at the rows' starts plus the offset (`plan.GridShifter`),
where a row that fails a stream's rule at that offset keeps its unmoved
windows. The rigorous ladder cuts a plan into the plans of its chunks
(`RowPlan.cut`), and a permutation trial moves X inside the plan without moving
its grid (`RowPlan.replace`).

### Rigorous fits in $\gamma$

The ladder trains on chunks of $N/\gamma$ samples. For fixed $N$ the bias
$a/N_\text{chunk} = (a/N)\gamma$ is linear in $\gamma$. The fit and the
extrapolation to $\gamma = 0$ therefore work in $\gamma$
([THEORY.md](THEORY.md#correcting-it)). The dependent variable is each chunk's
training-side estimate of the same quantity every other mode reports.

### Evaluation on clean, unshifted data

Augmentations apply inside the training batch loop only. Evaluation reads a
snapshot of the data taken before any window shift. The reported values
therefore never depend on an augmentation or on the last shift drawn. Within a
batch the image augmentations run first, then the others, then the custom ones.

---

## Extending the library

To add an estimator, put the bound in `neural_mi/estimators/bounds.py` and
register its name in `ESTIMATORS` in `neural_mi/estimators/__init__.py`. An
estimator that trains nothing is a function `(x, y, **params)` that returns
nats, registered in `NONPARAMETRIC`. Its conditional form `(x, y, z, **params)`,
registered in `NONPARAMETRIC_CONDITIONAL`, serves the difference quantities.
`run_training_task` returns the value as `train_mi` before any network is built.

To add a processor, subclass `TemporalWindowDataset` in
`neural_mi/data/temporal.py` (`BaseStaticDataset` in `neural_mi/data/static.py`
for unwindowed data). Add a branch for its name in `create_single_dataset()` in
`neural_mi/data/handler.py` and its parameters to `PROCESSOR_PARAMS_SCHEMA` in
`neural_mi/defaults.py`. Document the parameters in
[PARAMETERS.md](PARAMETERS.md) for `tests/test_docs_drift.py` to check against
the schema.

To add an encoder, subclass `BaseEmbedding`, declare the `Model` settings it
reads in its `settings`, register it in `_EMBEDDING_CLASSES` in
`neural_mi/utils.py`, add its name to the allowed values of `embedding_model` in
`neural_mi/validation.py`, and add any new settings to `BASE_PARAMS_SCHEMA`, to
`MODEL_SETTINGS` in `neural_mi/utils.py`, and give them `default_of` defaults.

To add an analysis mode:

1. Write the procedure in a module under `neural_mi/analysis/` when it is more
   than a few lines.
2. Add a producer to `neural_mi/analysis/modes.py` that returns its repeat rows
   through `_produced(...)` with its axis keys and the columns to aggregate.
   Route the mode to it in `produce()`.
3. Add its config dataclass to `neural_mi/config.py`, export it from
   `neural_mi/__init__.py` and add its settings and defaults to
   `MODE_KWARGS_SCHEMA` in `neural_mi/defaults.py`.
4. Register it in `neural_mi/run.py`: in `_MODES` and `_MODE_CONFIG_CLASSES`, in
   `_PERMUTABLE_MODES` if a permutation null means something for it, in
   `_WINDOWS_IN_TASK` if it windows inside its tasks, and in the
   `_SHIFT_*_SAFE_MODES` tuples that apply.
5. Decide what it does with repeats, a configuration grid and `rigorous`. Record
   the decision in the capability table in
   [USING.md](USING.md#what-each-mode-accepts) and add the mode to the check in
   `tests/test_contract.py` that every mode returns the shared `Results` shape or
   refuses clearly.
6. Document its config in [PARAMETERS.md](PARAMETERS.md) and the messages it
   prints in [MESSAGES.md](MESSAGES.md). `tests/test_docs_drift.py` checks both.
