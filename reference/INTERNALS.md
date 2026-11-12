# Internals

This page describes how the library is put together and where to change it. You
need none of it to use NeuralMI. [USING.md](USING.md) explains how to use it.

## Contents

- [Codebase map](#codebase-map)
- [Design decisions](#design-decisions)
- [Extending the library](#extending-the-library)

---

## Codebase map

`neural_mi.run()` validates the settings of a call, prepares the streams and
hands the prepared data to the mode's producer. The producer trains and combines
the networks and returns result rows that the shared assembly layer turns into a
`Results`. A permutation test reruns the same producer on data in which X is
moved in time.

| file | contents |
|---|---|
| `neural_mi/run.py` | `run()` (converts the config objects into keyword arguments for `_run_flat()`); `_run_flat()` (validates, applies defaults, builds the call's row plan and cuts its rows once or leaves them to the tasks, and calls the producer); the call-level checks (`_announce_call`, the loop over preparations `_run_preparation_grid`, `_warn_if_shift_windows_dead`) |
| `neural_mi/config.py` | the config dataclasses, `Processing` through `Pairwise`, and `as_config` |
| `neural_mi/defaults.py` | `BASE_PARAMS_SCHEMA` and `PROCESSOR_DEFAULTS` (with `PROCESSOR_PARAMS_SCHEMA`, the names it declares): every setting's type and default. A mode's own settings are part of its declaration in `analysis/modes.py` |
| `neural_mi/validation.py` | `ParameterValidator`, the allowed values, and the per-stream processor checks |
| `neural_mi/results.py` | the `Results` object: `get`, `summary`, `plot`, `compare`, `animate`, `save`, `load`, `to_dict`, `to_json` |
| `neural_mi/quantities.py` | the named quantities, on one engine: `_named` expands an iterable parameter into configurations, `_call` runs one configuration through `run()`, `_task` is the unit sent to a worker |
| `neural_mi/device.py` | `choose_device` and `with_device`: the device a call trains on when it names none, chosen by timing the first network on the CPU and, when the call is long enough, on the GPU |
| `neural_mi/parallel.py` | `worker_pool`, the one worker pool every parallel loop opens (`spawn` context, the parent's logging level, a share of the CPU threads per worker), and `dispatch_tasks` |
| `neural_mi/seeds.py` | `repeat_seed`, `network_seed` and `generator`: every seed of a call, from the call's `seed`, the repeat (`run_id`) and the role of a network or a draw in its estimate |
| `neural_mi/embeddings_io.py` | `extract_embeddings` (reloads a saved critic) |
| `neural_mi/utils.py` | `build_critic` and `build_optimizer_and_scheduler`, `cross_covariance_svd` (the one whitening and SVD behind the participation ratios and the rotated embeddings), `build_offset_arrays`, `get_device` |

**`neural_mi/analysis/`** contains the modes.

| file | contents |
|---|---|
| `modes.py` | one producer per mode, and `MODES`, which declares each mode once: its producer, its config class, the axis it adds, what it accepts (repeats, a grid, `rigorous`, a permutation null, returned embeddings, custom split indices), the streams that it uses and the settings that it sets itself. `produce()` runs a mode's producer. A producer takes the prepared data and the grid and returns the repeat rows, the per-configuration details and the columns to aggregate |
| `assemble.py` | `build_results` and `aggregate` (repeat rows into `runs` and `dataframe`), `merge_results` (the parts of a grid that is prepared once per value into one result), `convert_record` (the one conversion from nats), `NETWORK_KEYS` (what a trained network reports about itself) |
| `permutation.py` | `shift_x` (moving X for a null trial), `permutation_nulls` (running the trials), `attach_nulls` (the null columns and `p_value`) |
| `sweep.py` | `ParameterSweep`, the engine that trains one network per grid point and repeat; `amplification_factor` |
| `task.py` | `run_training_task` (one training run from parameters to a record) and a small cache of static datasets |
| `rigorous.py` | the $\gamma$ ladder, `_find_linear_region`, `_extrapolate_mi` and the fit diagnostics, for `mode='rigorous'` and for the rigorous difference quantities |
| `lag.py`, `precision.py`, `pairwise.py`, `dimensionality.py` | the mode-specific procedures the producers call |
| `conditional.py`, `interaction.py`, `transfer.py` | the component trainings of the difference quantities |
| `offsets.py` | the one path to the rows of every quantity defined by time offsets: `build_rows` (the rows a spec defines, through `utils.build_offset_arrays`, and their geometry), `one_step_series` (streams onto one grid of rows one step wide that tile time, the one route of `processing=`) and `grid_rows` |

**`neural_mi/data/`** contains the preprocessing. In `plan.py`, `RowPlan` is the
one plan of every stream that a quantity uses. Both the rows of every task and
the streams of every term come from this plan (see [The row plan](#the-row-plan)
below). `handler.py` has `create_dataset` for any number of streams on one grid,
`create_single_dataset` for one stream, `one_grid` (the grid's one window and
step) and the stream containers (`AlignedStreams`, `StreamBundle`,
`PairedTemporalDataset`, `PairedDataset`). `temporal.py` and `static.py` have
the per-processor datasets. `views.py` has the `SubsetView` used for splits.
`shift_windowing.py` has the windowing by sample number that implements the
moves of a plan, and the phase shifter for rows thinned by a stride.

**`neural_mi/models/`** has the critics (`SeparableCritic`, `HybridCritic`,
`ConcatCritic`), the encoders and `VariationalWrapper` in `embeddings.py`, and
the reconstruction decoders in `decoders.py`. `BranchedEmbedding` is the X
encoder of every term of a difference quantity. `_joint_marginal_difference`
(`analysis/sweep.py`) marks both terms with `_difference_term` and trains the
conditioning term first. The X of the joint term is the pair (rest of X,
conditioning set). The plan packs this pair (`pack` in `data/plan.py`), or the
quantity does so when it builds its rows by offset. `build_critic` then builds
one branch per part of X from the call's encoder. Every stage that uses X takes
the pair: `PairDecoder`, the augmentations, `StaticDataset`'s corruptions,
`ConcatCritic` and the nonparametric estimator. **`neural_mi/estimators/`** has
the bounds in `bounds.py`, registered by name in `ESTIMATORS`. KSG is in
`ksg.py` and registered in `NONPARAMETRIC`, with its conditional form in
`NONPARAMETRIC_CONDITIONAL`. **`neural_mi/training/trainer.py`** has the
training loop, the chunked evaluation, the epoch selection and the spectral
metrics. **`neural_mi/visualize/`** has the plots and the training animation.
**`neural_mi/generators/`** has the synthetic data.

---

## Design decisions

### One entry point, one pipeline

Every mode goes through `run()` and the same pipeline. The call is expanded into
configurations, axis values, repeats and the networks of each repeat. Each
network is trained and yields one record. The records of a repeat are combined:
unchanged for a single network, as a signed sum for a difference quantity, or by
a weighted extrapolation for a rigorous fit. The repeats are then aggregated
into one `dataframe` row per configuration and axis value. In its producer, a
mode declares only its networks and its combine step, so the shape of `Results`
is the same for every mode.

### One place for a default

Every setting's default is written once, in `BASE_PARAMS_SCHEMA`
(`neural_mi/defaults.py`). Code below `run()` reads a setting with
`defaults.setting(params, name)`. This function falls back to that default.
Below the public signatures, every signature default of a setting is
`defaults.default_of(name)`. `run()`, the named quantities and `_run_flat` still
have literal defaults in their signatures until design item 1 of the roadmap is
done. `tests/test_schema_defaults.py` checks the signatures of `Trainer` and of
the encoder, decoder and critic classes against the schema and fails on any
default written a second time as a literal.

`build_critic` uses only the settings in `utils.ARCHITECTURE_KEYS`.
`save_network` saves exactly these settings with a network, so
`extract_embeddings` rebuilds the network as it was trained. A value that pickle
cannot store, such as a class defined inside a function, is saved as its name.
`extract_embeddings` then requires that value in `base_params`.

### One unit conversion

Every network trains and reports in nats. `assemble.convert_record` converts the
information-valued entries once, after training and before anything is combined.
Components, nulls, ceilings and fits therefore all appear in the result in the
caller's units.

### Values that cannot be negative

The trainer sets a network's value to 0 when its held-out MI never rose above
zero or its training value came out negative. The measured value stays in
`raw_train_mi`. A combined value follows the same rule where it is formed,
through `modes._floor_at_zero`: in `_difference_rows` for conditional MI and
transfer entropy, in `_fit_row` for an extrapolation, and in the precision rows.
`modes._floor_at_zero` keeps the measured value as `<key>_raw`. Wherever a
repeat has a measured value, `assemble.aggregate` averages the measured values
(`mi_raw`, or `<col>_raw` for a column in `mean_cols`) and sets only the mean to
0 when it is negative. Elsewhere, it leaves the values reported as 0 out of
`mi_mean` and `mi_std` and counts them in `n_zero`. The permutation test also
summarises each trial with `aggregate`. A rung of the rigorous ladder of a
difference takes its signed measured value (`sweep.rung`).
`rigorous._drop_rungs` leaves out only three kinds of rung: the rungs reported
as 0 (their networks learned nothing), the NaN rungs (their term failed) and the
rungs whose held-out curve did not resolve
(`curve_diagnosis == 'not resolved'`). For a difference, this curve is the curve
of the joint term (`sweep.rung`). The function also leaves out every $\gamma$
that loses `left_out_share` or more of its rungs for these reasons. Interaction
information is signed and stays as measured.

Every quantity combined from separately trained networks goes through
`sweep.combine_terms`. Its callers are `_joint_marginal_difference` (conditional
MI, transfer entropy and the quantities built on them), interaction information
and the `runs` rows of `modes._difference_rows`. The rungs of their rigorous
ladders call the same functions on each chunk. `sweep.term_values` classifies
each network as measured, as having learned nothing (it enters as 0) or as
failed (its repeat is NaN). The line between the last two is
`epochs.stands_above_zero`. The trainer uses the same test to word its warning
about a negative training value, so the warning of a network and its place in a
combination agree. Before it uses a value, `combine_terms` compares the rows
that each term's task used (`data.rows.row_identity`). A plan's dataset carries
this identity from `RowPlan.identity`. The task's result keeps it under
`_row_identity`. The comparison covers the count of the rows and, wherever both
tasks know them, the origin and step of the grid, a digest of the start times of
the kept rows and Y's window (`data.rows.same_rows`). A repeat whose terms used
different rows is refused. This is the check that every term used the rows of
its plan.

### Settings that change the preparation of a call

A call prepares its data once. It builds the rows or leaves them to each task,
records their geometry and runs the data check. Every configuration of its grid
then trains on that preparation. Three kinds of grid key change the preparation
itself. A processor parameter, such as `window_size`, changes the rows.
`Model(input_scaling=...)` sets how the encoders scale each stream. `_run_flat`
works out this scaling once from the kind of rows that each stream gives.
`Training(shift_windows=...)` sets whether the rows are cut once for the whole
call or by each task, so that they can move. `defaults.PREPARATION_SETTINGS`
lists these last two settings. For a grid over any of these keys,
`_run_preparation_grid` prepares the data once per combination of their values,
runs the rest of the grid on each preparation and merges the parts with
`merge_results`. Configurations keep the order of the full grid. Each
configuration therefore computes what a separate call with its settings
computes. `tests/test_grid_contract.py` checks this for every setting that a
grid accepts. The named quantities that build their rows by offset before they
call `run()` split their grid in the same way, in `quantities._by_offsets`. A
processor key that no stream's processor uses is refused.

A check or a message that runs once per call reads its settings from every
configuration of the grid (`run._grid_settings`) rather than from the base
configuration alone. Examples are the count of held-out rows in the data
check, the advice for a small dataset and the check that a sequential encoder
has a time axis.

### The permutation null

A null trial reruns the mode's producer, with X moved in time, through the same
`produce()` call as the observed value. The null is therefore built from exactly
the rows that the observed value was built from. Moving the source leaves Y and
W in place, so Y's history in transfer entropy and the link between Y and W in
the conditional quantities are kept. The trials go through `dispatch_tasks` like
every other task.

### Parallelism and seeds

Every loop of independent work runs in a `spawn` pool when `n_workers` is above
1, whether it loops over configurations, repeats, lags, pairs, rigorous chunks,
splits or permutation trials. Every pool opens through `parallel.worker_pool`.
The named quantities dispatch to such a pool with `parallel.dispatch_tasks`.
Each pool wraps its task function in `logger.CapturedTask` and collects the
results through `logger.released`. A worker then hands back the warnings and log
messages of its task so that they are raised again in the calling process at the
caller's line. `logger.collect_repeats` wraps every `run()`, named quantity and
`create_dataset()`. It shows each kind of message once per call and counts the
rest. In IPython, every call of a cell shares the count. On import,
`logger._group_repeats_by_cell` registers `pre_run_cell` and `post_run_cell`
hooks that open and close it. `grouped_warnings()` opens the same shared count
over a block. The count of a repeated warning is raised as a warning of the same
category. The count of a log message is logged at its level.

A message that needs the user's attention (a result that may be off, a setting
not used as given, data that the library changed) is a `warnings.warn` with a
category and is documented in `reference/MESSAGES.md`. Logging is used only for
errors and for what the library did, at the info or debug level.
`test_docs_drift.py` checks that `logger.warning` is not used.

When an outer loop runs in parallel, any loop inside one of its units runs
with `n_workers=1` so that a pool never starts inside a pool.

Every seed comes from three things (`neural_mi/seeds.py`): the call's `seed`,
the repeat, and the role of a network or a draw in its estimate. The repeat is
the task's `run_id`. A call without one has the repeat 0. A role is a rung's
gamma and chunk, a dimensionality fit's channel split and width, or one draw
that a call makes outside its tasks: a channel split, the held-out rows of a
dimensionality call, the noise of `mode='precision'` or a permutation trial's
move of X. A plain network and the terms of a difference have no role. The
repeat alone seeds the shifts of the windows at each epoch and the order in
which the ladder deals its rows into chunks. A network's own seed sets its
initialisation, batch order and dropout, its split and its evaluation rows.
Nothing else enters a seed. A seed does not depend on a configuration's
settings, a task's place in the grid, the worker that runs it or whether a grid
runs as one preparation per value. Every configuration therefore runs a repeat
from the same seeds, as it would if each setting ran on its own with the seeds 0
to R-1. `tests/test_seeds.py` checks the rule.

### Tensors are three-dimensional

Inside the library, every stream is `(n_samples, n_channels, window)`.
Unwindowed 2-D input gains a trailing axis of 1. Image-like 4-D input passes
through unchanged.

### Blocked splits

The default split holds out `n_test_blocks` contiguous stretches spread over the
recording. It excludes from training a gap of `gap_fraction` of a block on
either side of each stretch, so correlated neighbouring windows cannot end up on
both sides of the split. A random split suits only independent samples.

### Preprocessing and the data check

`neural_mi/preprocessing.py` contains every transform of the data: the channel
transforms, the autoregressive whitening and the interface that a transform of
your own follows. `utils._whiten` (the embedding spectrum) and `ksg.prepare` use
the same `ZScore`, `ZCA` and `Copula`. `preprocessing.prepare` applies a
`Preprocessing` to the streams of a call and runs the data check on the result.
Every path calls it once, before rows are built: `_run_flat` on the raw streams
for every mode except transfer entropy, the transfer branch on the series that
its histories come from (after any grid), and `quantities._by_offsets` on a
named quantity's series, before `offsets.build_rows`. The named quantities
remove `preprocessing` from what they pass to `run()`. `run()` then receives
rows and skips its own check when `_rows` shows that a quantity built them. The
part of the check that needs the rows (`run._warn_small_sample`) runs on every
route, at the point where the route counts its rows: once in `_run_flat` for
rows cut there or passed in, at offset 0 for rows that each task cuts, once per
preparation of a grid (`_run_preparation_grid`) and on the histories of transfer
entropy (`tests/test_data_check_routes.py`).

### Rows in time order

Whatever builds rows from a time series records their geometry once, in the
parameters that every task receives (`neural_mi/data/rows.py`). The record
gives, in one unit, the distance between the starts of consecutive rows
(`row_step`) and the stretch that one row covers (`row_span`). A processor
records the geometry of its windows (`run.py` when it windows once for the call,
`task.py` when a task windows for itself). Rows built by offset get their
geometry from the spec that builds them (`offsets.build_rows` and
`offsets.spec_geometry`). A named quantity or `offset_quantity` passes this
geometry to `run()` through the private `_rows` argument. Transfer entropy
records it for its histories (`transfer.te_spec`). When windows shift, rows that
a `stride` above one thins are built at every step. With `_row_phases`, each
task then keeps one row in every `stride` (`shift_windowing.RowPhaseShifter`
moves the phase each epoch). The record still gives the step of the kept rows.
`data.rows.kept_rows` gives their count. Rows without the record, such as arrays
passed in already built or samples in no order, are not known to be in time
order. The record is reported per network in `runs` and
`details[config_id]['trainings']`. A plan records the geometry of its own grid
(`RowPlan.geometry`).

Everything that depends on the time order uses the record and nothing else. The
trainer's blocked split warns when its gap is shorter than the span of a row.
Its random split warns when rows share samples, as does the split that
`mode='dimensionality'` shares across its fits. The rigorous ladder cuts
recorded rows into contiguous chunks through `rigorous._Chunker` (the one
chunker of both rigorous paths). A block shuffle moves blocks of rows, each as
long as the span of one row. Whether rows move each epoch is a separate matter.
The trainer takes it from the shifter that a dataset carries.

### The row plan

`run()` builds one `RowPlan` per call and processor configuration over every
stream that the call uses (`data/plan.py`). Each stream carries its processor
and every setting. `resolve_stream` resolves these settings once from the call's
parameters and the schema. The grid has the window and step of `one_grid`. A
plan cuts its rows in one of two ways (`RowPlan.route`). Streams on a regular
clock that start together are windowed by sample number
(`shift_windowing.BundleShifter`). Any other plan puts every stream on one grid
in time, from the latest start to the earliest end, through the processors' own
datasets, and keeps a window only where every stream is valid. A plan of rows
that were built already (`'rows'`) packs them as given.

`TermRows` pairs the plan with a term's `Roles`. The roles name the streams that
make up the term's X (one stream, or a pair of the rest of X and the
conditioning set) and those that make up its Y. Every term takes its rows from
the plan through `TermRows`. Every term of a difference therefore takes the same
rows, the same kept windows and the same per-stream fits. `task.py` builds every
raw dataset in this way: a term's dataset from its `TermRows` and a pair's
dataset from `RowPlan.of_pair`. A plan that does not shift is cut once in
`run()` (`RowPlan.materialize`), so its rows reach the tasks already built. A
plan that shifts reaches each task raw. Its dataset carries a shifter that draws
one offset per epoch within one step and moves every stream by it. The shifter
moves the streams either by sample number or, on the grid in time, by windowing
every stream again at the starts of the rows plus the offset
(`plan.GridShifter`). On the grid, a row that fails a stream's rule at that
offset keeps its unmoved windows. The rigorous ladder cuts a plan into the plans
of its chunks (`RowPlan.cut`). A permutation trial moves X inside the plan
without moving its grid (`RowPlan.replace`).

The plan contains the rows but not what a task does with them. Each term's task
draws the split, the rows that each epoch evaluates and the shifts from a split
seed and a shift seed that every term of a repeat shares. These draws therefore
match across the terms without the plan holding them. `create_dataset` and the
offset builder (`offsets.build_rows`) still build their rows themselves, the
first for the offset route under `processing=` and the second for every named
quantity.

### Rigorous fits in $\gamma$

The ladder trains on chunks of $N/\gamma$ samples. For fixed $N$, the bias
$a/N_\text{chunk} = (a/N)\gamma$ is linear in $\gamma$. The fit and the
extrapolation to $\gamma = 0$ therefore work in $\gamma$
([THEORY.md](THEORY.md#correcting-it)). The dependent variable is the training
value of each chunk, for the same quantity that every other mode reports.

### Evaluation on clean, unshifted data

Augmentations apply only inside the training batch loop. Evaluation uses a
snapshot of the data taken before any window shift, so the reported values never
depend on an augmentation or on the last shift drawn. Within a batch, the image
augmentations run first, then the others and then the custom ones.

---

## Extending the library

To add an estimator, put the bound in `neural_mi/estimators/bounds.py` and
register its name in `ESTIMATORS` in `neural_mi/estimators/__init__.py`. An
estimator that trains nothing is a function `(x, y, **params)` that returns
nats. Register it in `NONPARAMETRIC`. Its conditional form `(x, y, z, **params)`
is registered in `NONPARAMETRIC_CONDITIONAL` and is used by the difference
quantities. `run_training_task` returns the value as `train_mi` before any
network is built.

To add a processor, subclass `TemporalWindowDataset` in
`neural_mi/data/temporal.py` (`BaseStaticDataset` in `neural_mi/data/static.py`
for unwindowed data). Add a branch for its name in `create_single_dataset()` in
`neural_mi/data/handler.py`, and add its parameters to `PROCESSOR_PARAMS_SCHEMA`
in `neural_mi/defaults.py`. Document the parameters in
[PARAMETERS.md](PARAMETERS.md) so that `tests/test_docs_drift.py` can check them
against the schema.

To add an encoder, subclass `BaseEmbedding` and declare the `Model` settings
that it uses in its `settings`. Register it in `_EMBEDDING_CLASSES` in
`neural_mi/utils.py`, and add its name to the allowed values of
`embedding_model` in `neural_mi/validation.py`. Add any new settings to
`BASE_PARAMS_SCHEMA` and to `MODEL_SETTINGS` in `neural_mi/utils.py`, and give
them `default_of` defaults.

To add an analysis mode:

1. Write the procedure in a module under `neural_mi/analysis/` when it is more
   than a few lines.
2. Add a producer to `neural_mi/analysis/modes.py` that returns its repeat rows
   through `_produced(...)` with the columns to aggregate.
3. Add its config dataclass to `neural_mi/config.py`, export it from
   `neural_mi/__init__.py`, and add a parameter of its name to `run()`.
4. Declare it in `MODES`: its producer, its config, its settings with their
   types and defaults, the axis that it adds, and what it does with repeats, a
   configuration grid, `rigorous`, a permutation null, returned embeddings and
   custom split indices. Give the reason for each request that it refuses. Name
   the function that prepares its own settings for its producer from the call
   (`prepare`), the checks of its settings (`check`), and its figure, summary
   lines and overlay (`draw`, `summarize`, `overlay`, written on the model of
   the functions in `results.py`). `run()`, the checks, the producers and the
   results all use the declaration, but none of them names a mode. Its `table`
   is its row of the capability table in
   [USING.md](USING.md#what-each-mode-accepts). `tests/test_mode_registry.py`
   checks this row. `tests/test_contract.py` runs the mode with every kind of
   grid against the shared `Results` shape.
5. Document its config in [PARAMETERS.md](PARAMETERS.md) and the messages that
   it prints in [MESSAGES.md](MESSAGES.md). `tests/test_docs_drift.py` checks
   both.
