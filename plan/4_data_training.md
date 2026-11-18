# Phases W and T: the data layer, the trainer and the task

Phase W leaves one builder of windows and one rule for the width of a window. Phase T splits the
two longest functions of the package along their seams. Phase W comes after phase E, because by
then every mode reads its rows from the row plan and the second builder has no other user.

## Phase W: one window grid and the data layer

A window is cut on one of two routes today. Streams on a regular clock that start together are
windowed by sample number (the *unfold* route). Every other set of streams is put on one grid in
time (the *grid* route). `run()` chooses the route from the data, and `create_dataset` always
takes the grid route. The two routes disagree on the width of a window, and W-01 and W-02 remove
that disagreement.

### W-01 One rule converts a duration to samples

**Why.** Nine expressions turn a duration into a number of samples or bins. They follow three
different rules. The table gives what each returns for a duration of 12.5, 2.5 and 13.5 samples,
and for 0.3 divided by 0.1. A script produced it by calling each function, and its output is in
`plan/evidence/conversions.txt`.

| Expression | Where | 12.5 | 2.5 | 13.5 | 0.3 / 0.1 |
|---|---|---|---|---|---|
| `seconds_to_samples` | data/shift_windowing.py:48 | 12 | 2 | 14 | 3 |
| `permutation._window_rows` | analysis/permutation.py:162 | 12 | 2 | 14 | 3 |
| the rows that `BundleShifter` keeps | data/shift_windowing.py:285 | 12 | 2 | 13 | 3 |
| `slots_in_window` | data/temporal.py:69-86 | 13 | 3 | 14 | 3 |
| the samples of a continuous window | data/temporal.py:396 | 13 | 3 | 14 | 3 |
| the samples of a categorical window | data/temporal.py:1057 | 13 | 3 | 14 | 3 |
| the bins of a binned spike window | data/temporal.py:881 | 13 | 3 | 14 | 3 |
| the width in the `full_trajectory` message | data/temporal.py:1073 | 13 | 3 | 14 | 3 |
| `RowGeometry.rows_per_span` | data/rows.py:49 | 13 | 3 | 14 | 3 |

The first two round to the nearest even number, the third rounds down, and the others round up.
For 25 Hz data and a window of 0.5 s, `run()` therefore trains on windows of 12 samples, and
`create_dataset` returns windows of 13 samples.

The owner's rule is that a window contains every sample whose time stamp falls inside it. A
window of 0.5 s that starts on a sample of 25 Hz data contains the samples at 0 s, 0.04 s and so
on up to 0.48 s, so it has 13 samples. Six of the nine expressions already follow this rule.

**Change.**
1. Establish first that the table is still correct by calling the nine expressions again.
2. Add one function to `data/grid.py`.

   ```python
   def samples_in(duration: float, period: float) -> int:
       """The samples stamped inside a window of `duration` that starts on a sample."""
       return max(1, int(math.ceil(duration / period - 1e-9)))
   ```
3. Make all nine expressions call it. On the unfold route the step is converted by the same
   function, and a step that is left unset is one window, so the windows tile the samples. On
   the grid route the step stays a duration.
4. Say once per call, in a log message, when a window or a step is not a whole number of
   samples, and give the number of samples that is used.
5. Keep dropping the incomplete window at the end of a recording. A window needs a fixed width,
   so keeping it would need a rule for padding. With `shift_windows` on, each epoch moves the
   rows by up to one step, so samples beyond the last unmoved window are read in training.

**Tests.** Add a parametrised test of `samples_in` with the four columns of the table. Add a
test that `run()` and `create_dataset` give the same width for 25 Hz data and a window of 0.5 s.

**Pages.** State the rule where `reference/USING.md` introduces `window_size`.

**Numbers.** They change on the unfold route wherever a window or a step is not a whole number
of samples: 12.5 samples gave 12 and now give 13, and 2.5 gave 2 and now give 3. `run()` takes
this route for regular streams that start together. The grid route and `create_dataset` already
round up. Data whose windows are whole numbers of samples are unaffected.

**Done when.** `grep -rn "int(round" neural_mi/data/` shows no conversion of a duration, and the
new tests pass.

### W-02 `create_dataset` reads the row plan

**Why.** `create_dataset` (data/handler.py:736-864) builds its windows with a second builder and
never asks `RowPlan` for the route. `reference/USING.md:1020` tells users to check widths and
retention with it, but it can report other windows than the ones `run()` trains on. The offset
path also depends on it (`offsets.grid_rows`, analysis/offsets.py:102).

**Change.**
1. Establish first every attribute and method that `reference/USING.md` (lines 1018-1044),
   tutorial 04 (12 calls) and the tests (52 calls in 12 files) read from what `create_dataset`
   returns. The audit found `stream(name).data`, `stream_names`, `x_window_width`,
   `y_window_width`, `n_windows_built`, `n_windows_retained` and `window_retention`.
2. Make `create_dataset` build a `RowPlan` and return a small object with those attributes,
   filled from `RowPlan.rows()`.
3. Give `RowPlan` an option that keeps every window, for `grid_rows`. It replaces
   `validate_windows=False` and skips the filter at data/plan.py:279-292.
4. Move the function to `data/create.py`.

**Tests.** Add a test that `create_dataset` and `run()` see the same rows for regular streams
that start together, for a stream that starts late and for a spike stream. Update
`tests/test_data_processors.py`.

**Numbers.** `create_dataset` now returns what `run()` trains on, so its output changes for
regular streams that start together.

**Done when.** `create.py` imports `RowPlan` and no dataset class of `handler.py`.

### W-03 Delete the second builder of windows

**Why.** After W-02 nothing uses `AlignedStreams._initialize_windows` and `_build_windows`
(data/handler.py:331-381). The two methods repeat `RowPlan._rows_grid` (data/plan.py:256-299).

**Change.** Delete `AlignedStreams`, `StreamBundle`, `PairedTemporalDataset`,
`AlignedStaticStreams`, `_NamedStreams`, `_PairAccessors`, the observer part of `WindowManager`,
`update_parameters` and `set_window_size`.

**Tests.** These classes are named 13 times in two test files. Delete the tests that exercise
only them, and keep the tests of the documented behaviour of `create_dataset`.

**Pages.** `reference/INTERNALS.md:53-57` and `:331` describe the classes.

**Done when.** `grep -rn "AlignedStreams\|StreamBundle\|PairedTemporalDataset" neural_mi/` shows
nothing.

### W-04 One dataset class

**Why.** The trainer sees a `PairedDataset` of two `StaticDataset` objects, and other modules
patch attributes onto it: `_window_shifter`, `_row_identity` and the retention counts
(data/plan.py:315-318, data/shift_windowing.py:230). The trainer writes `x_dataset.data` directly
(training/trainer.py:880-883). `BaseStaticDataset` has one subclass and an empty
`_process` stub (data/static.py:47).

**Change.** Add `data/dataset.py` with one class `RowsDataset` whose declared fields are the X
rows, the Y rows, the shifter, the identity and the retention, with one method that replaces the
rows for an epoch. Delete `PairedDataset`, `StaticDataset` and `BaseStaticDataset`.

**Tests.** Update `tests/test_trainer.py`, `tests/test_plan.py` and
`tests/test_shift_windowing.py`.

**Done when.** No module sets an attribute of a dataset from outside its class.

### W-05 The windowers share their clock, their search and their encoder

**Why.** `data/temporal.py` (1,265 lines) contains four windowers that repeat each other. The clock
and period are set up for continuous streams (lines 341-368) and again for categorical streams
(1018-1033), and `plan._clock` (data/plan.py:502) and `stream_period` are two more versions. Only
the continuous copy checks `period_tolerance`. The three categorical movers repeat one loop
(1098-1218) and duplicate `make_categorical_encoder`. The two spike classes repeat the search
and the scatter into windows (775-790 and 921-938). The resolution of `data_device` is written in
`temporal.py:147` and `static.py:27`.

**Change.**
1. Split the file into `data/processors/continuous.py`, `spikes.py`, `binned.py` and
   `categorical.py`, with their base class.
2. Keep one function for the clock and the period in `data/grid.py`. Establish first what the
   `period_tolerance` check does, and decide whether a categorical stream must pass it.
3. Keep one encoder for categories that both routes call, and one helper for the search and the
   scatter of spikes.

**Tests.** `tests/test_data_processors.py`, `tests/test_empty_windows.py` and the snapshot guard
this step.

**Done when.** `data/temporal.py` does not exist and each of the four patterns is written once.

### W-06 The spike processor declares its two forms

**Why.** The name `'spike'` covers two processors. The audit read that with `bin_size` set, the
settings `n_seconds`, `max_spikes_per_window`, `exclude_bursty_neurons` and `no_spike_value` are
ignored, and that without it `normalize_bins` is ignored. A setting that is accepted and then
ignored is the kind of fault that this plan removes.

**Change.**
1. Establish the two lists first, by reading which settings each of the two classes takes
   (data/handler.py:626-637).
2. Declare the two forms in the processor's record of R-04, each with its own settings. Keep
   the one name `'spike'`, with `bin_size` choosing the form.
3. Make `_warn_ineffective_settings` warn about a setting that the chosen form does not read.

**Pages.** Mark in the spike table of `reference/PARAMETERS.md` which form reads each setting.
Add the warning to `reference/MESSAGES.md`.

**Done when.** A call with `bin_size` and `max_spikes_per_window` warns.

### W-07 The settings of a stream are resolved once

**Why.** The rule that Y and W read with X's processor and settings is in
`defaults.stream_processing`, but six places apply it: `run.py:1057-1061`,
`analysis/lag.py:60-64`, `analysis/sweep.py:50-52`, `analysis/offsets.py:40-58`,
`handler._stream_specs_from_pair` and `RowPlan.of_pair`. The dataset constructors also repeat
defaults as literals (`min_coverage_fraction=0.2`, `encoding`).

**Change.**
1. Resolve every stream once in `resolve_call` (E-12) with `resolve_stream`
   (data/plan.py:83-90), and pass resolved `Stream` objects from there on.
2. Remove the other calls of `stream_processing`.
3. Make each windower take a resolved `Stream` instead of keyword arguments with defaults.

**Done when.** `stream_processing` is called from one place.

### W-08 One vocabulary for the kind of a stream

**Why.** `preprocessing.stream_kind` (preprocessing.py:324) answers series, rows or spike.
`defaults.stream_kind_of` answers continuous, spike counts and other values. Both describe the
same property of a stream.

**Change.** Keep the vocabulary of `stream_kind_of` as the `kind` of the processor's record
(R-04) and derive what the data check needs from it.

**Done when.** `stream_kind` is defined once.

### W-09 The modules of `data/` in layers

**Why.** `handler.py` and `plan.py` import each other (handler.py:561 and 842, plan.py:122, 257
and 308), and `shift_windowing.py` imports `handler` and `static` inside a function (221-222).

**Change.** Arrange the modules from the bottom up as `arrays`, `grid`, `processors/`,
`shifters`, `rows`, `dataset`, `plan`, `offsets`, `check`, `streams`, `create`. Move `one_grid`,
`resolve_step_size`, `stream_period`, `refuse_mixed_streams` and `rows_line_up` to `grid.py`, and
the shifters (`WindowShifter`, `BundleShifter`, `RowPhaseShifter`, `GridShifter`, `RoleShifter`)
to `shifters.py`. Move every import in `data/` to the top of its module.

**Done when.** The allowlists of S-03 hold no entry for `data/`.

### W-10 The augmentations read their defaults from the schema

**Why.** `apply_augmentations` writes its defaults inline. Only `kernel_size` is read
from the schema (augmentations.py:158). The audit also read that `random_flip_h` flips the
second axis and calls it the height.

**Change.**
1. Read every default of an augmentation from the schema.
2. Establish first which axis `random_flip_h` flips for each input shape, then correct its name
   or its docstring.

**Done when.** The test of S-05 lists no augmentation default.

## Phase T: the trainer and the task

`Trainer.train` is one method of 996 lines with 45 keyword arguments
(training/trainer.py:242-1237), and `run_training_task` is one function of 435 lines
(analysis/task.py:152-586). Both have clear seams. Each step below moves one part out without
changing what it computes, and the snapshot of S-02 guards every step. A step may claim that the
numbers are unchanged only after it has checked that the random draws keep their order.

### T-01 The split helpers become functions

**Why.** `_create_random_split`, `_create_blocked_split`, `_select_eval_indices`,
`held_out_rows`, `rows_to_fill_a_batch` and `too_few_held_out` are methods of the trainer, but
`analysis/dimensionality.py:110-118` calls them on a stand-in object that contains only a
generator. `_rng_of` (trainer.py:57) exists for that call.

**Change.** Move the six to `training/split.py` as functions that take a generator. Update
`dimensionality.py`, `analysis/rigorous.py:667` and the trainer, and delete `_rng_of`.

**Tests.** `tests/test_held_out.py` and `tests/test_trainer.py` guard this step.

**Done when.** `grep -n "SimpleNamespace" neural_mi/analysis/dimensionality.py` shows nothing.

### T-02 `train()` takes the settings as one object

**Why.** `train()` has 45 keyword arguments with literal defaults, and `task.py:384-449`
forwards each one and resolves again the defaults that the trainer already has. The docstrings
repeat default values that can drift from the schema.

**Change.**
1. Change the signature to `train(dataset, settings)`, where `settings` is the resolved mapping
   of the task. The trainer reads each value with `setting(settings, name)`.
2. Replace the forwarding in `task.py` by one argument.
3. Add a helper for the tests that returns the schema's defaults with overrides.

**Tests.** `Trainer(` is built 16 times in five test files. Update the calls of `train`.

**Done when.** `train` has two parameters, and the test of S-05 lists no default of the trainer.

### T-03 Evaluation in blocks is one function

**Why.** Two nearly identical blocks parse `eval_train` and `track_embeddings`
(trainer.py:616-684), and the evaluation of a partition in blocks is written for the training
side and for the held-out side.

**Change.** Write one parser for the two settings and one function that evaluates a set of rows
in blocks of at most `max_eval_samples`.

**Tests.** `tests/test_blocked_evaluation.py` guards this step.

**Done when.** The trainer has one loop over evaluation blocks.

### T-04 The checkpoint keeper, the diagnosis and the embedding tracker leave `train()`

**Why.** `train()` also keeps and drops checkpoint copies (lines 731-861), diagnoses the curve
and gives its warnings (943-1115), and tracks and rotates embeddings (803-825 and 1171-1201).

**Change.** Move each part to its own function or small class, one commit each. Put the
diagnosis in `training/epochs.py`, beside the rule for the reported epoch.

**Tests.** `tests/test_best_epoch.py`, `tests/test_epoch_rule.py` and
`tests/test_embedding_extraction.py` guard this step.

**Done when.** `train()` is the epoch loop with calls to these parts, in under about 200 lines.

### T-05 Precision and dimensionality use public hooks of the trainer

**Why.** `run_precision_analysis` reads `trainer._safe_eval_mi` and
`trainer.last_train_indices`, and takes the trainer, the dataset and the frozen evaluation rows
out of the result dict under the keys `_trainer`, `_dataset`, `_frozen_eval_x` and
`_frozen_eval_y` (analysis/precision.py).

**Change.**
1. Give the trainer a public method `evaluate(x, y, max_rows)`.
2. Return from `train()` a small record with declared fields, among them the indices of the
   training rows and the frozen evaluation rows.
3. Return the trainer itself in a declared field when the task asks for it.

**Tests.** `tests/test_precision.py` guards this step.

**Done when.** `grep -rn "trainer\._" neural_mi/analysis/` shows nothing.

### T-06 `run_training_task` becomes a short function

**Why.** After E-05 and E-13 the function still infers the sizes of the inputs (task.py:244-289),
runs the KSG branch (291-292), builds the critic and the decoders (294-367), trains, extracts
embeddings and assembles the result, all inline.

**Change.** Write one function for each of these parts and call them in order. Delete the
`del dataset` and the cache check at task.py:566-571, since both do nothing after E-05.

**Done when.** `run_training_task` is under about 60 lines.

### T-07 Small repairs in the trainer and the critics

**Why.** These are single faults that the audit found.

**Change.**
1. Move the docstring of `Trainer.__init__` above its first statement (trainer.py:161-162), so
   that `help()` shows it.
2. Move the decoders to the device once (they are moved at trainer.py:232 and again at 495).
3. Keep one method for the embeddings of a critic. `get_embeddings` and
   `get_training_embeddings` run the same lines (models/critics.py:250-296).
4. Give `max_n_batches` one meaning. It is used as a number of rows per encoder chunk and, as
   `max_n_batches // batch_size`, as a number of rows of pairs. Establish first whether chunking
   during training splits the statistics of a batch-norm layer, as the audit read.
5. Say in `TransformerDecoder` that it rounds `hidden_dim` (models/decoders.py:407), or refuse a
   value that needs rounding.

**Pages.** Correct the row of `max_n_batches` in `reference/PARAMETERS.md`.

**Done when.** `Trainer.__init__.__doc__` is not `None`.

### T-08 The encoders share what they repeat

**Why.** `GRU` and `LSTM` are nearly identical (models/embeddings.py:404-502). The Xavier
initialisation is written five times. Every built-in encoder except `MLP` and `CNN1D` accepts
`**kwargs` that it ignores, so a mistyped setting goes unnoticed.

**Change.**
1. Write one recurrent encoder with the cell type as an argument, and keep `GRU` and `LSTM` as
   its two registered names. Keep the names of the parameters and the order in which they are
   created, so that the initial weights do not change.
2. Write one helper for the initialisation.
3. Remove the `**kwargs` from the built-in encoders.

**Tests.** `tests/test_models.py` and the snapshot guard this step.

**Done when.** `grep -n "\*\*kwargs" neural_mi/models/embeddings.py` shows only the custom
classes that need it.
