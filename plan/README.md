# Plan to consolidate NeuralMI before its first release

This folder contains the plan to consolidate the `neural_mi` package before its first public release.
The library was built in steps, and several steps left a second copy of code that already existed.
A fix made in one copy is then missing from the other, and most of the wrong behaviours listed in
`1_safety_fixes_deletions.md` arose this way. The plan removes the copies so that each concept of
the library is defined in one place. The public API stays almost as it is.

The plan was written against commit `39f42d1` of `dev`, and line numbers refer to that commit.
The numbers drift once the first steps are merged, so find each symbol by its name.

## How to use this plan

The plan has ten phases in five files. Work through them in this order.

| File | Phases |
|---|---|
| `1_safety_fixes_deletions.md` | S: the safety net. B: wrong behaviours fixed on their own. X: code that nothing reaches. |
| `2_moves_registries.md` | L: moves that put each module in its layer. R: one table for each kind of name. |
| `3_engine.md` | E: one path from a call to its result. |
| `4_data_training.md` | W: one window grid and the data layer. T: the trainer and the task. |
| `5_api_pages_release.md` | A: the public API. P: the pages, the packaging and the release. |

Every step has the same fields. **Why** gives the problem and the evidence. **Change** lists the
edits. **Tests** names the tests to add and the test files that must follow. **Pages** names the
reference pages that must follow, when there are any. **Numbers** appears only when the step
changes a number that a user sees. **Done when** gives checks that pass or fail.

These rules apply to every step.

1. One step is one commit on `dev`. A step with lettered parts is one commit per part.
2. Before each commit, run `pytest -m "not slow"`, the characterisation check of S-02 and the layer
   test of S-03. Run the full suite at the end of each phase.
3. A step changes no number unless it has a **Numbers** field. A step that has one regenerates the
   snapshot of S-02 in the same commit. Its commit message lists the cases that changed.
4. The pages follow the code in the same commit. `tests/test_docs_drift.py` compares
   `reference/PARAMETERS.md` with the schemas, the headings of `reference/MESSAGES.md` with the
   texts of the warnings, every name in backticks in `reference/*.md` and `ROADMAP.md` with the
   code, and the list of test files in `reference/TESTING.md` with `tests/`.
5. A step that moves or renames a symbol updates the tests that import it in the same commit. It
   leaves no alias under the old name, because nothing has been published.
6. Read the code that a step names before you edit it. Where a step says "establish first", write
   what you found in the commit message.
7. Leave the notebooks alone until P-13. They store their outputs, and P-13 runs them all once.

## The model behind the library

Every quantity that the library estimates is `I(A; B | C)`. A and B are the two groups of
variables whose shared information is measured, and C is an optional group that is conditioned
out. A quantity without a C is one estimate, made by one trained network. A quantity with a C is
the difference of two estimates, `I(A,C; B) - I(C; B)`. Interaction information is the only
quantity with three estimates, `I(A,C; B) - I(A; B) - I(C; B)`. We call each estimate a *term*.
All the terms of one quantity must be trained and evaluated on the same rows, with the same split
and the same seed.

The quantities form two families that differ only in how a row is built.

- **Chosen signals.** A row is one snapshot, such as an image, a trial, or one window that a
  processor cut from a recording. The caller chooses the signals that play A, B and C. This family
  contains stimulus information, the pairwise map, informational dimensionality, conditional MI and
  interaction information. `run()` is its front door.
- **Chosen times.** A row is one placement in time along a recording, and A, B and C are streams
  at chosen offsets around the present. This family contains block MI, active information storage,
  predictive information, cross-predictive information, transfer entropy, conditional transfer
  entropy, instantaneous exchange, the directed information rate and the MI rate. The named
  functions are its front door.

An *analysis* decides how many estimates are made and how they are read. It can make one estimate,
or one for each point of a grid of settings, each lag, each pair of channels, each embedding
dimension or each level of jitter in spike times. Two *modifiers* apply to an analysis. The
rigorous correction replaces one fit by a ladder of fits on subsamples of the rows (55 by default)
and extrapolates to infinite data. The permutation test builds a null distribution by running the
analysis again on data in which A is moved in time.

## What the code looks like at the end

One call follows one path.

1. A front door resolves its arguments into one call record. The record contains the mode, the
   settings with the set of keys that the caller set, the settings of the modifiers and the grid.
2. The front door gives the engine a function that builds the rows. `run()` builds them from the
   signals as given, or from windows that the processors cut. A named function builds them by
   offset from series one step wide.
3. The engine runs one preparation for each value of a setting that changes the rows. A
   preparation builds one row plan, checks the data and names the quantity.
4. The mode's procedure asks for estimates. One function trains every term of a repeat on the
   plan's rows, and one function combines the terms.
5. When the call asks for the rigorous correction, the same function is called on each chunk of
   the ladder, and one function fits each ladder.
6. Every list of tasks runs through one runner. No other code chooses the device or opens a
   pool of workers.
7. One module builds the tables of the result. The result records how it is shown, so the
   `Results` class and the figures need nothing from the analysis package.

Each concept has one home.

| Concept | Its home at the end | Where it is written today |
|---|---|---|
| The rows of a call | `data/plan.py`, `RowPlan` | five branches in `run._run_flat`, four forms of data in `analysis/task.py`, a second builder of windows in `data/handler.py` |
| A duration as a number of samples | `data/grid.py` | nine expressions that follow three rules (see W-01) |
| Rows by offset | `data/offsets.py` | `utils.build_offset_arrays`, `analysis/offsets.py`, `modes._prepare_transfer`, `quantities._by_offsets` |
| The terms of a quantity | `analysis/quantity.py` | `MODES[...].terms`, `conditional.py`, `interaction.py`, `transfer.py`, three `*_rigorous_scalar` functions, `_TRANSFER_REVERSE` |
| Training the terms of a repeat | `analysis/estimate.py` | `_joint_marginal_difference`, `_train_single_term`, three `run_*` functions |
| Running a list of tasks | `analysis/runner.py` | five loops, two direct calls of a private method, `parallel.dispatch_tasks` |
| The ladder and its fit | `analysis/ladder.py` | `AnalysisWorkflow`, `run_rigorous_scalar_analysis`, two copies of the fit |
| A grid over a setting that changes the rows | `engine.py` | `run._run_preparation_grid`, `quantities._named_run`, a recursion in `quantities._by_offsets` |
| The settings of a call | `engine.py`, the call record | config exporters with five names, the 80 arguments of `_run_flat`, `_inject`, a namespace of 40 fields |
| A default value | the schemas in `defaults.py` and `MODES[mode].settings` | literals in the signatures of `_run_flat` and of the analysis functions |
| The names a user can pass | one registry for each kind | encoders in six places, estimators in three tables and nine branches |
| How a result is shown | recorded in the `Results` object | fields of `MODES` that import private functions of `results.py` |
| The conversion between nats and bits | `units.py` | five places |

Each layer imports only from the layers below it, and S-03 adds the test that enforces this.

| Layer | Modules |
|---|---|
| 0 Foundations | `exceptions`, `messages`, `seeds`, `units`, `devices`, `defaults`, `config`, `parallel` |
| 1 Containers and data | `results`, `preprocessing`, `data/` |
| 2 Learning core | `estimators/`, `models/`, `training/` |
| 3 Estimation | `analysis/`: `task`, `runner`, `quantity`, `estimate`, `ladder`, `permutation`, `assemble`, `grid` |
| 4 Analyses | `analysis/`: `modes`, `lag`, `pairwise`, `dimensionality`, `precision` |
| 5 Engine | `validation`, `engine` |
| 6 Front doors | `api`, `quantities`, `visualize/`, `generators/` |

The package tree at the end is this one.

```
neural_mi/
  __init__.py        the public names
  api.py             run(), the front door for chosen signals
  quantities.py      the front door for chosen times: the specs and the named functions
  engine.py          the call record, the preparation loop, the path from a plan to a result
  validation.py      the checks of settings
  config.py          the setting classes
  defaults.py        the schemas
  results.py         the Results container and its summary
  preprocessing.py   the transforms
  messages.py  repeats.py  exceptions.py  seeds.py  units.py  devices.py  parallel.py
  analysis/
    modes.py         the declaration of every mode
    quantity.py      Quantity, Term, the three forms, and how terms combine
    estimate.py      the terms of a repeat, trained on one plan
    runner.py        the one runner of tasks, with the choice of device
    task.py          one network on the rows of one term
    ladder.py        the rigorous ladder and its fit
    permutation.py  assemble.py  grid.py
    lag.py  pairwise.py  dimensionality.py  precision.py
  data/
    arrays.py  grid.py  rows.py  offsets.py  plan.py  dataset.py  check.py  streams.py
    corruption.py  shifters.py  create.py
    processors/      continuous.py  spikes.py  binned.py  categorical.py
  models/            embeddings.py  critics.py  decoders.py  factory.py  io.py
  training/          trainer.py  split.py  epochs.py  optim.py  spectrum.py  augmentations.py
  estimators/        __init__.py (the registry)  bounds.py  ksg.py
  visualize/         plot.py  animate.py  figures.py
  generators/
```

## The public API at the end

The front door for chosen signals keeps its shape.

```python
nmi.run(x_data, y_data=None, w_data=None, *, mode='estimate', sweep_grid=None,
        rigorous=None, permutation=None,
        processing=None, preprocessing=None, model=None, training=None, split=None,
        estimator=None, output=None,
        lag=None, pairwise=None, dimensionality=None, precision=None,
        n_workers=1, seed=None, verbose=None, show_progress=True, device=None)
```

The modes are `estimate`, `conditional`, `interaction`, `lag`, `pairwise`, `dimensionality` and
`precision`. Each named function for chosen times lists the same arguments after its own windows,
instead of forwarding `**run_kwargs`.

```python
nmi.transfer_entropy(x_data, y_data, history_window, future_window=1, *, stride=1,
                     bidirectional=False, sweep_grid=None, rigorous=None, permutation=None, ...)
```

The table gives each change with the number of places that call the old form today.

| Today | At the end | Tests | Tutorial cells | Reference pages |
|---|---|---|---|---|
| `mode='rigorous'` | `rigorous=True` or `rigorous=Rigorous(...)` with any mode that can use it | 17 in 11 files | 1 | 15 lines |
| `Conditional(w_data=w)`, `Interaction(w_data=w)` | `run(x, y, w, mode='conditional')` | 117 in 20 files | 1 | 4 lines |
| `mode='transfer'`, `Transfer(...)` | `transfer_entropy(...)`, `conditional_transfer_entropy(...)` | 80 in 9 files | 0 | 5 lines |
| `permutation_test=`, `n_permutations=` and three more | `permutation=True` or `permutation=Permutation(...)` | 31 in 6 files | 0 | 4 lines |
| `k`, `h`, `future_k`, `prediction_horizon` | `history_window`, `future_window` | 83 calls in 9 files | 8 | 14 lines |
| `interaction_information(x, y, w)` | `run(x, y, w, mode='interaction')` | 6 in 2 files | 1 | 1 line |
| `mode='sweep'` | `mode='estimate'` with a `sweep_grid` | 44 in 19 files | 8 in 3 notebooks | 7 lines |

## Decisions

The owner has settled these points.

- The rigorous correction is an argument of `run()` and of every named function, and
  `mode='rigorous'` goes.
- The core of the library is the estimate of one quantity `I(A; B | C)` from one, two or three
  terms on the same rows. Two families build the rows, chosen signals and chosen times.
- The past window is called `history_window`.
- An incomplete piece is dropped.
- Nothing is published, so no step keeps an old name alive.

The plan assumes these points, and the owner has not confirmed them yet. Each one names the steps
that change if the answer differs.

1. `run()` is the front door for chosen signals and the named functions are the front door for
   chosen times. `mode='transfer'` and the `Transfer` config go (A-03), and
   `interaction_information()` goes (A-02).
2. The future window is called `future_window`. It replaces `future_k` and `prediction_horizon`,
   since `te_spec` already reads `prediction_horizon` as the number of future steps (A-05).
3. A window of 12.5 samples contains 12 samples, and the call says so (W-01).
4. `mode='sweep'` is folded into `mode='estimate'` (A-06).
5. The rigorous correction is refused for `precision` and `dimensionality`. It is accepted for
   `lag` and `pairwise`, where each lag or pair gets its own ladder (E-08).
6. These go because nothing uses them: `set_verbose`, `plot_cross_correlation`,
   `analyze_mi_heatmap`, the TUBA, NWJ and JS bounds, the loader of the old network format, and
   `sample_rate` among the settings of a spike stream (X-05 to X-09, B-02).
7. Each quantity keeps the column names that it has today. The plan declares them once and does
   not rename them (E-02).
8. The modules `run.py` and `logger.py` are renamed `api.py` and `messages.py`, so that the
   package attributes `run` and `logger` no longer hide them (L-10, E-14).

## Index of steps

**S. Safety net**
- [ ] S-01 Record the baseline of the test suite
- [ ] S-02 Add a characterisation snapshot of every mode and named function
- [ ] S-03 Add a test of the import layers
- [ ] S-04 Add a snapshot of the public surface
- [ ] S-05 Add a test that no signature repeats a default

**B. Wrong behaviours fixed on their own**
- [ ] B-01 Sort spike times before they are binned
- [ ] B-02 Remove `sample_rate` from the settings of a spike stream
- [ ] B-03 Validate `n_permutations`
- [ ] B-04 Restore the cuDNN flags after a call
- [ ] B-05 Copy a mutable default when it is applied
- [ ] B-06 Let Python report an unknown argument of `run()`
- [ ] B-07 Merge `scheduler_params` with the scheduler's own arguments
- [ ] B-08 Refuse a list `hidden_dim` where an encoder or a decoder cannot use it
- [ ] B-09 Build each decoder from the settings of its own side
- [ ] B-10 Give `plot_embeddings` a `show` argument and `animate` a legend
- [ ] B-11 Compute the directionality index in one place
- [ ] B-12 Stop training the reverse direction inside a ladder
- [ ] B-13 Say in the pages how `run(x, y)` differs from `instantaneous_mi(x, y)`

**X. Code that nothing reaches**
- [ ] X-01 Delete the corruption methods of the datasets and their copy of the data
- [ ] X-02 Replace `SubsetView` by a tensor of indices
- [ ] X-03 Delete the `device` argument of the datasets
- [ ] X-04 Delete the arguments that nothing passes or reads
- [ ] X-05 Delete the functions and tables that nothing calls
- [ ] X-06 Delete the second way to save a network and the loader of its format
- [ ] X-07 Delete the unreachable branches in the models
- [ ] X-08 Delete the two plots that nothing calls
- [ ] X-09 Delete `set_verbose`
- [ ] X-10 Correct the comments that name things that are gone
- [ ] X-11 Replace seaborn by matplotlib (optional)

**L. Moves and layers**
- [ ] L-01 One module for units
- [ ] L-02 The tensor helpers and the device helper leave `utils.py`
- [ ] L-03 The optimiser builder and the spectrum functions move to `training/`
- [ ] L-04 The critic factory and the network files move to `models/`
- [ ] L-05 Rows by offset move to `data/offsets.py`, and `utils.py` is deleted
- [ ] L-06 `augmentations.py` moves to `training/`
- [ ] L-07 The figures leave `results.py`
- [ ] L-08 The setting checks and the data reports leave `run.py`
- [ ] L-09 The data check leaves `preprocessing.py`
- [ ] L-10 `logger.py` becomes `messages.py`, and its three jobs separate
- [ ] L-11 One module for grids
- [ ] L-12 `device.py` moves next to the code that it times
- [ ] L-13 Remove the imports inside functions that only avoided a cycle

**R. Registries**
- [ ] R-01 Encoders
- [ ] R-02 Critics, optimisers, schedulers and activations
- [ ] R-03 Estimators
- [ ] R-04 Processors
- [ ] R-05 The small sets of names
- [ ] R-06 A test that the registries and the validator agree

**E. Engine**
- [ ] E-01 One function runs every list of tasks
- [ ] E-02 A quantity is declared once
- [ ] E-03 One function trains the terms of a quantity
- [ ] E-04 Transfer entropy trains from a row plan
- [ ] E-05 Every mode trains from a row plan (parts a to h)
- [ ] E-06 One fit of the ladder
- [ ] E-07 One ladder for every quantity
- [ ] E-08 The rigorous correction becomes a modifier inside the engine
- [ ] E-09 The named functions call the engine directly
- [ ] E-10 One preparation loop
- [ ] E-11 A result describes itself
- [ ] E-12 One call record
- [ ] E-13 A typed task record
- [ ] E-14 `run.py` becomes `api.py` and `engine.py`, and `modes.py` keeps the declaration only

**W. Window grid and data layer**
- [ ] W-01 One rule converts a duration to samples
- [ ] W-02 `create_dataset` reads the row plan
- [ ] W-03 Delete the second builder of windows
- [ ] W-04 One dataset class
- [ ] W-05 The windowers share their clock, their search and their encoder
- [ ] W-06 The spike processor declares its two forms
- [ ] W-07 The settings of a stream are resolved once
- [ ] W-08 One vocabulary for the kind of a stream
- [ ] W-09 The modules of `data/` in layers
- [ ] W-10 The augmentations read their defaults from the schema

**T. Trainer and task**
- [ ] T-01 The split helpers become functions
- [ ] T-02 `train()` takes the settings as one object
- [ ] T-03 Evaluation in blocks is one function
- [ ] T-04 The checkpoint keeper, the diagnosis and the embedding tracker leave `train()`
- [ ] T-05 Precision and dimensionality use public hooks of the trainer
- [ ] T-06 `run_training_task` becomes a short function
- [ ] T-07 Small repairs in the trainer and the critics
- [ ] T-08 The encoders share what they repeat

**A. Public API**
- [ ] A-01 `rigorous` is an argument of `run()` and of every named function
- [ ] A-02 `run()` takes `w_data`, and the configs `Conditional` and `Interaction` go
- [ ] A-03 Transfer entropy is a named function only
- [ ] A-04 `permutation` is one argument
- [ ] A-05 `history_window` and `future_window` name every window in time
- [ ] A-06 `mode='estimate'` takes a grid
- [ ] A-07 A result names its quantity and lists its keys
- [ ] A-08 The public names of the package
- [ ] A-09 The generators return one form (optional)

**P. Pages, packaging and release**
- [ ] P-01 Write the changelog of the first release
- [ ] P-02 Keep the version in one place
- [ ] P-03 Set the classifiers and test Python 3.13
- [ ] P-04 Check what the wheel and the source distribution contain
- [ ] P-05 Publish from a tag
- [ ] P-06 Make the README work on PyPI
- [ ] P-07 Move the roadmap out of the release
- [ ] P-08 Store the tutorial data as arrays, with their source
- [ ] P-09 Build the documentation site without warnings
- [ ] P-10 Limit the CI runs
- [ ] P-11 Cite the software
- [ ] P-12 Rewrite INTERNALS.md and TESTING.md for the new layout
- [ ] P-13 Rewrite USING.md around the two front doors, and run the tutorials again
- [ ] P-14 Remove this folder
