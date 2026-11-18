# Phases S, B and X: the safety net, the fixes and the deletions

These three phases prepare the ground. Phase S adds the checks that let every later step show
that it preserved behaviour. Phase B fixes the behaviours that are wrong today and that can be
fixed without restructuring anything. Phase X deletes the code that nothing reaches, so that the
later phases move and merge less code.

## Phase S: the safety net

The existing suite has about 1,490 tests in 76 files, with oracle tests that compare estimates
with exact values. It does not show that a refactoring left every number where it was. The steps
below add that check and three guards against the kinds of drift that this plan removes.

### S-01 Record the baseline of the test suite

**Why.** Later steps need a reference for what passed and how long it took.

**Change.**
1. Run `pytest -q` and `pytest -m "not slow" -q` on the starting commit.
2. Write `plan/BASELINE.md` with the commit, the versions of Python, torch and numpy, the number
   of tests, any failure and the wall time of both runs.

**Done when.** `plan/BASELINE.md` exists and the suite passes, or each failure is listed in it.

### S-02 Add a characterisation snapshot of every mode and named function

**Why.** A step that only moves or merges code must leave every result identical. The oracle
tests allow a tolerance, so they cannot show this.

**Change.**
1. Add `tests/characterisation/cases.py` with one small function per case. Each case calls the
   library on synthetic data with a fixed seed, `device='cpu'` and a few epochs, and returns the
   result. The cases cover every path that the plan touches:
   - each of the ten modes, with `mode='transfer'` once in both directions
   - each of the twelve named functions, one of them with a list for its own window
   - the estimators `infonce`, `smile` and `ksg`, with `ksg` also on a conditional quantity
   - rows given already, windows cut on a regular clock, a stream that starts late, spike times,
     binned spikes and a categorical stream
   - `shift_windows` on and off
   - a `sweep_grid` over `run_id` and over `window_size`
   - the rigorous correction on plain MI, on conditional MI and on one named function, with
     `gamma_range=range(1, 4)`
   - a permutation test with three trials
   - `use_decoder=True` and `return_embeddings=True`
   - one case run with `n_workers=2`
2. Add `tests/characterisation/record.py`. It runs every case and writes
   `tests/characterisation/snapshot.json`. For each case it stores the mode, the sorted column
   names of `dataframe` and `runs`, every numeric column of both tables, the scalar entries of
   `details`, and the shapes and sums of any returned embeddings. It also stores a fingerprint of
   the machine: the platform and the versions of Python, torch and numpy.
3. Add `tests/test_characterisation.py`. It runs every case and compares it with the snapshot,
   with a relative tolerance of 1e-6. It skips with a clear message when the fingerprint differs,
   because training amplifies the rounding differences between machines.
4. Establish first that each case gives the same numbers in two runs in one process and in a
   fresh process. Leave out any case that does not, and report it.
5. Keep the whole module under about three minutes with `-n auto` by shrinking the data and the
   epochs. Add the file to `reference/TESTING.md`.

**Done when.** `python -m tests.characterisation.record` followed by
`pytest tests/test_characterisation.py` passes twice in a row.

### S-03 Add a test of the import layers

**Why.** Seven groups of imports point upward or form a cycle today. They work only because
the imports sit inside functions (37 of them in `analysis/`). Without a test the cycles return.

**Change.**
1. Add `tests/test_layers.py`. It parses every module of `neural_mi` with `ast` and collects the
   imports of other `neural_mi` modules at module level and inside functions.
2. The test contains the layer table of `plan/README.md`, mapped onto today's module names, and
   fails on an import that points to a higher layer.
3. The test contains an allowlist of today's violations. Each entry names the step that removes it.
   An entry that no longer matches an import fails the test, so the list can only shrink.
4. The test fails on any import of a `neural_mi` module inside a function, except on a second
   allowlist. That list contains the imports that guard an optional package (torchvision,
   scikit-learn, umap) and, for now, the ones that avoid a cycle, each with its step.

**Done when.** The test passes, and removing one entry of the allowlist makes it fail.

### S-04 Add a snapshot of the public surface

**Why.** Phase A changes the API on purpose. Every other phase must leave it alone.

**Change.**
1. Read `tests/test_contract.py` and `tests/test_run_config_api.py` first and extend them where
   they already pin part of the surface.
2. Add `tests/public_surface.txt` with the names in `neural_mi.__all__`, the signature of `run`
   and of each named function, and the fields of each config class.
3. Add a test that rebuilds the text and compares it with the file.

**Done when.** The test passes, and renaming one argument of `run` makes it fail.

### S-05 Add a test that no signature repeats a default

**Why.** `reference/INTERNALS.md` says that a default has one home, but the signatures of
`_run_flat` (run.py:952-1035), `run_rigorous_scalar_analysis` (rigorous.py:964-970),
`run_precision_analysis` (precision.py:45-47) and `run_dimensionality_analysis`
(dimensionality.py:428-442) repeat schema defaults as literals.

**Change.**
1. Add a test that reads every function signature in the package with `ast`. A keyword argument
   whose name is a key of `BASE_PARAMS_SCHEMA` or of a mode's settings must default to `None` or
   to a call of `default_of`.
2. Start with an allowlist of today's offenders, each with the step that removes it (most are
   removed by E-12).

**Done when.** The test passes with the allowlist and fails without it.

## Phase B: wrong behaviours fixed on their own

Each step fixes one behaviour that a second pass of the audit reproduced by running the library.
Each step starts with a test that fails before the fix. Five other wrong behaviours are fixed by
the consolidation itself and are listed at the end of this phase.

### B-01 Sort spike times before they are binned

**Why.** `BinnedSpikeDataset.__init__` (data/temporal.py:865-878) stores each neuron's spike
times unsorted, but its extent (884-888) and its counts (921-922) assume that they are sorted. In
the reproduction, 400 shuffled spikes gave 7 windows instead of 20, with spikes counted in the
wrong bins. `SpikeWindowDataset` sorts them (640-646), and `validation.py:148-152` tells the user
that the dataset sorts them in both cases.

**Change.**
1. Sort each neuron's times in `BinnedSpikeDataset.__init__`.
2. Reword the message at `validation.py:148-152` so that it names no class.

**Tests.** Add a test that shuffled and sorted spike times give equal tensors with `bin_size`
set, through `create_dataset` and through `run()`.

**Numbers.** They change only for unsorted spike times with `bin_size`.

**Done when.** The new test passes.

### B-02 Remove `sample_rate` from the settings of a spike stream

**Why.** The schema accepts `sample_rate` for a spike stream (defaults.py:305-306), but
`create_single_dataset` passes it to neither spike dataset (data/handler.py:626-637). For a spike
stream it reaches only a log line in `analysis/lag.py:78`. `reference/PARAMETERS.md:106` says
that `mode='lag'` uses it.

**Change.**
1. Remove the key from the spike entry of the processor schema.
2. Make the log line in `lag.py` say that the lags of a spike stream are in seconds.

**Tests.** Add a test that a spike stream with `sample_rate` is reported as every other unknown
processor setting is. Remove the key from the tests that pass it (grep `sample_rate` in `tests/`).

**Pages.** Remove the row at `reference/PARAMETERS.md:106`.

**Done when.** `grep -n sample_rate neural_mi/defaults.py` shows no line in the spike entry.

### B-03 Validate `n_permutations`

**Why.** Only `min_permutations` is validated (run.py:1047). `n_permutations=0` returns a NaN
p-value, and `-1` raises `ZeroDivisionError` from a log message (run.py:512).

**Change.** Refuse a value that is not a positive integer, next to the check at run.py:1047.
A-04 later moves the rule into the schema of the permutation settings.

**Tests.** Add a test in `tests/test_permutation.py` that 0, -1 and 2.5 raise a `ValueError` that
names the setting.

**Done when.** The new test passes.

### B-04 Restore the cuDNN flags after a call

**Why.** `_run_flat` sets `torch.backends.cudnn.deterministic` and `benchmark` (run.py:1084-1085)
and never restores them, so one call slows the user's other GPU code for the rest of the session.

**Change.** Save both flags before they are set and restore them in the `finally` block at
run.py:1502-1505.

**Tests.** Add a test that both flags have their earlier values after `run()` returns and after
it raises.

**Done when.** The new test passes.

### B-05 Copy a mutable default when it is applied

**Why.** `apply_defaults` assigns the schema's default object itself (validation.py:409). The
four dict defaults (`estimator_params`, `augmentation_params`, `optimizer_params`,
`scheduler_params`) are then shared with `result.params`, and editing a result changes the
default of every later call.

**Change.** Copy the default with `copy.deepcopy`, as `default_of` already does.

**Tests.** Add a test that edits `result.params['base_params']['optimizer_params']` and then
checks that the schema's default is still empty.

**Done when.** The new test passes.

### B-06 Let Python report an unknown argument of `run()`

**Why.** `run()` collects unknown keywords in `**_removed` (run.py:672) and answers with a
paragraph about config objects (794-803). The paragraph was written for an older API that was
never released. A typo such as `seeds=1` gets this paragraph as its answer.

**Change.** Remove `**_removed` and the block that raises. Python then names the argument.

**Tests.** Replace the tests that match the old text (grep `unexpected keyword` in `tests/`) by
one that checks a `TypeError` naming the argument.

**Pages.** Remove the entry from `reference/MESSAGES.md` if it has one.

**Done when.** `grep -n _removed neural_mi/run.py` shows nothing.

### B-07 Merge `scheduler_params` with the scheduler's own arguments

**Why.** `build_optimizer_and_scheduler` passes `step_size` to the `step` scheduler and then
passes `**scheduler_params` again (utils.py:1038-1050), so `scheduler_params={'step_size': 5}`
raises "multiple values". The same happens for `T_max` with `cosine` and `mode` with `plateau`.
The `cosine_warmup` branch never reads `scheduler_params`.

**Change.**
1. Build each scheduler's arguments as the library's values overridden by `scheduler_params`.
2. Read `scheduler_params` in the `cosine_warmup` branch, and refuse a key that the branch does
   not take.

**Tests.** Add a test for each of the four schedulers with its natural key.

**Done when.** The new tests pass.

### B-08 Refuse a list `hidden_dim` where an encoder or a decoder cannot use it

**Why.** The schema accepts a list for `hidden_dim` (defaults.py:124). The list works for `mlp`,
`cnn`, `cnn2d` and `tcn`. For `gru`, `lstm`, `lru` and `transformer` it fails with an unrelated
torch error. `deepsets` silently uses only the first width (models/embeddings.py:871). With
`use_decoder=True` every encoder crashes at `models/decoders.py:124`.

**Change.**
1. In the validation of the model settings, refuse a list for an encoder that takes one width,
   with a message that names the encoder. R-01 later reads this from the encoder's registry entry.
2. Refuse a list in `DeepSets` instead of cutting it.
3. Refuse a list together with `use_decoder=True`, with a message that says a decoder takes one
   width.

**Tests.** Add one parametrised test over the ten encoder names, and one for the decoder.

**Numbers.** A `deepsets` call with a list now raises instead of using the first width.

**Done when.** The new tests pass.

### B-09 Build each decoder from the settings of its own side

**Why.** With `use_decoder=True`, `run_training_task` builds both decoders from X's
`embedding_model`, `hidden_dim` and `n_layers` (analysis/task.py:333-348). The settings
`embedding_model_y`, `hidden_dim_y` and `n_layers_y` are ignored. A custom or `deepsets` encoder
gets an MLP decoder, and the only notice is a log line at information level
(models/decoders.py:584-593).

**Change.**
1. Build the Y decoder from the Y settings where they are set, and from X's otherwise. This is
   the rule that the encoders follow.
2. Turn the notice of the MLP fallback into a warning, and give it for a custom encoder too.

**Tests.** Add a test that spies on `build_decoder` with different X and Y encoders.

**Pages.** Add the new warning to `reference/MESSAGES.md`.

**Numbers.** They change for a call with `use_decoder=True` whose Y settings differ from X's.

**Done when.** The new test passes.

### B-10 Give `plot_embeddings` a `show` argument and `animate` a legend

**Why.** `reference/USING.md:1686` says that every plot takes `show`, but `plot_embeddings`
(visualize/plot.py:424-432) forwards it to matplotlib and crashes. In `animate`,
`_resolve_scatter_color` turns category codes into floats (visualize/animate.py:487-488), so the
check at line 317 is always true and the legend for categories is never drawn. The embeddings of
Y are reduced (171-172) and never shown.

**Change.**
1. Add `show: bool = True` to `plot_embeddings`, as in the four other plot functions.
2. Make `_resolve_scatter_color` report whether the labels are categories, and draw the legend
   for them instead of the colour bar.
3. Stop reducing the embeddings of Y.

**Tests.** Add tests in `tests/test_visualize.py` and `tests/test_animate.py`.

**Done when.** The new tests pass.

### B-11 Compute the directionality index in one place

**Why.** `run_transfer_entropy` computes the index for its log (analysis/transfer.py:207-208) and
`_difference_rows` computes it for the result (analysis/modes.py:224-228). The first returns 0.0
when a direction is NaN and the second returns NaN, so the log can contradict the result.

**Change.** Add one function `directionality_index(te_xy, te_yx)` with the rule of `modes.py`
and call it from both places. E-04 later removes the first call.

**Tests.** Add a unit test of the function with a NaN, with two zeros and with two values.

**Done when.** `grep -n "te_sum" neural_mi/` shows nothing.

### B-12 Record with a test that a ladder trains the reverse direction for nothing

**Why.** With `rigorous=True` and `bidirectional=True`, `_te_rigorous_scalar` forwards
`bidirectional` to `run_transfer_entropy` at every rung (analysis/transfer.py:229-255), but
`rung` combines only the forward terms and `produce_difference_rigorous`
(analysis/modes.py:290-317) reports only the forward ladder. The audit read this in the code and
did not run it.

**Change.** Add a test that counts the calls of `run_training_task` in such a call and checks
that the result reports both directions. Mark it as an expected failure. E-08 gives each
direction its own ladder and removes the mark.

**Done when.** The test exists and fails for the reason that it documents.

### B-13 Say in the pages how `run(x, y)` differs from `instantaneous_mi(x, y)`

**Why.** On the same arrays and seed the two calls agree by default, but they gave 0.73 and 0.61
bits with a random split and an evaluation subset. The cause is by design. Rows given to `run()`
without a processor carry no order in time, so a ladder cuts them in a random order
(analysis/rigorous.py:624-643). The named functions record that consecutive rows are consecutive
steps (quantities.py:384-388), so chunks are contiguous and the split's leak checks apply.

**Change.** State this difference in the docstrings of `run` and `instantaneous_mi`, where
quantities.py:544 now says only that `mode='estimate'` handles rows built in another way.

**Pages.** Add the same paragraph to the part of `reference/USING.md` that introduces the named
quantities.

**Done when.** Both docstrings and the page say which call treats rows as ordered in time.

### Wrong behaviours that the consolidation fixes

- `mode='lag'` and `mode='dimensionality'` never consider the GPU, and `mode='lag'` never gives
  the memory warning. E-01 fixes both.
- The same short ladder warns in `mode='rigorous'` and stays silent or raises in a rigorous
  difference quantity. E-06 fixes this.
- A `mi_rate` result calls itself conditional and labels its terms with X, W and Y. E-11 fixes
  this.
- `transfer_entropy()` cannot set `prediction_horizon`. A-05 fixes this.
- The two windowing routes cut windows of different widths. W-01 fixes this.

## Phase X: code that nothing reaches

Each step proves again that nothing in `neural_mi/` reaches the code, and then removes it together
with its tests and its mentions in the pages. The commands under **Done when** show that a name
is gone.

### X-01 Delete the corruption methods of the datasets and their copy of the data

**Why.** `apply_noise`, `apply_precision`, their `*_by_name` forms and `reset` exist on every
dataset and container (data/handler.py:265, 272, 475, 478, data/views.py:223, 227,
data/static.py:157-163, and data/temporal.py:574-591, 817-834, 964-978, 1235). Nothing in the
package calls them, because `mode='precision'` calls `corrupt()` on tensors. Each temporal dataset
also keeps a full copy of its data for them (`data_master`, temporal.py:499, 796, 944, 1096), and
the trainer has to clear it (training/trainer.py:882-883).

**Change.** Delete the methods, the `data_master` copies and the two lines in the trainer.

**Tests.** Delete the tests of these methods (25 mentions in two test files).

**Done when.** `grep -rn "apply_noise\|apply_precision\|data_master" neural_mi/` shows nothing.

### X-02 Replace `SubsetView` by a tensor of indices

**Why.** The trainer builds `SubsetView(indices=...)` (training/trainer.py:549, 576, 613, 642)
and reads only `.indices`. The mode that selects by time, the registry of views and
`_on_dataset_updated` cannot be reached, and `AlignedStreams._notify_subset_views`
(data/handler.py:381-386) is their only reader.

**Change.**
1. Keep each partition as a sorted `LongTensor` in the trainer.
2. Delete `data/views.py`, its export in `data/__init__.py` and `_notify_subset_views`.

**Tests.** Delete `tests/test_views.py` and the other tests of the class (15 mentions in two
files). Remove the file from `reference/TESTING.md`.

**Pages.** Remove the entry at `reference/MESSAGES.md:992` and the mentions in
`reference/INTERNALS.md:53-57` and `:331`.

**Done when.** `grep -rn SubsetView neural_mi/ reference/` shows nothing.

### X-03 Delete the `device` argument of the datasets

**Why.** Every dataset constructor takes a `device` that it keeps only for reference. For this
argument `data/static.py:6` and `data/temporal.py:6` import `utils.get_device`, and `utils`
imports every model class, so the data layer depends on the models.

**Change.** Delete the argument and the two imports. Keep `data_device`.

**Done when.** `grep -n "get_device" neural_mi/data/` shows nothing.

### X-04 Delete the arguments that nothing passes or reads

**Why.** These arguments have no effect, and each one suggests a behaviour that does not exist.

**Change.** Delete each of these.
1. `windows_in_task` of `ParameterSweep.run` and `_prepare_tasks`. Both branches are identical
   (analysis/sweep.py:214-221). Remove it from the callers at `pairwise.py:74` and
   `dimensionality.py:307`. Keep `Mode.windows_in_task` until E-05, because `run.py:1379` reads it.
2. `**kwargs` of `ParameterSweep.__init__`, `run` and `_prepare_tasks`, and of `run_lag_analysis`
   (analysis/lag.py:24).
3. `std_cols` in `modes._produced`, at `run.py:1492` and in `assemble.build_results`. No producer
   sets it.
4. The `mode` argument of `permutation.row_values` and of `modes.produce_single_network`, the
   `ctx` argument of `permutation._window_rows`, and the `n_workers` argument of
   `run_precision_analysis`.
5. The line `info.pop('raw_results', None)` at `modes.py:439` and the docstring line in
   `precision.py` that documents the key.
6. The `pack` argument of `BundleShifter`, the `window_manager` argument of the four temporal
   datasets, and the pass-through `StreamBundle.__init__` (data/handler.py:489).

**Done when.** `grep -rn "windows_in_task" neural_mi/analysis/sweep.py` shows nothing.

### X-05 Delete the functions and tables that nothing calls

**Why.** Each of these has no caller in the package.

**Change.** Delete each of these.
1. `assemble.expand` (analysis/assemble.py:154).
2. `Estimator.to_base_params` (config.py:213) and its test at `tests/test_config.py:61`.
   `run.py:859-863` builds the same two keys by hand.
3. `ESTIMATOR_DEFAULTS` (estimators/__init__.py:30).
4. `utils._names` (utils.py:644), in favour of `utils._accepts_kwarg` (utils.py:468).
5. The loop over a per-side bias at `utils.py:839-850`, where both values are always equal.

The TUBA and NWJ bounds of `estimators/bounds.py` stay, by the owner's decision. They cannot be
selected today, and R-03 registers them.

**Done when.** `grep -rn "ESTIMATOR_DEFAULTS\|def expand" neural_mi/` shows nothing.

### X-06 Delete the second way to save a network and the loader of its format

**Why.** `Trainer.train(save_best_model_path=...)` writes a bare state dict
(training/trainer.py:935-936). Its only caller passes `None` (analysis/task.py:433) and saves the
bundled format through `save_network`. The loader keeps a branch for the bare format
(embeddings_io.py:223-239, with `_REQUIRED_BUILD_KEYS`), and only a simulated file tests it
(tests/test_embedding_extraction.py:176-178). No release has written that format.

**Change.** Delete the argument of `train`, the branch of the loader and the test.

**Done when.** `grep -n "save_best_model_path" neural_mi/training/trainer.py` shows nothing.

### X-07 Delete the unreachable branches in the models

**Why.** These branches cannot run.

**Change.**
1. In `models/critics.py`, delete the check `isinstance(kl, float)` at line 231 and the fallback
   for the concat critic at 290-292. Correct the docstring at 274-275, since neither method wraps
   `no_grad`. Keep one copy of the comment block that is pasted at 303-306, 352-355 and 428-431.
2. In `models/embeddings.py`, delete the fallback for an older torchvision at 1066-1077. It
   catches `AttributeError`, but `get_model` raises `ValueError`, and `pyproject.toml` asks for
   torchvision 0.19 or later.

**Done when.** The tests of the models pass and the two blocks are gone.

### X-08 Delete the two plots that nothing calls

**Why.** `plot_cross_correlation` and `analyze_mi_heatmap` (visualize/plot.py) are called by
nothing in the library, the tutorials or the README. `reference/USING.md:1695-1696` lists them,
and `reference/MESSAGES.md:867` documents one of their messages. `analyze_mi_heatmap` chooses a
configuration by a rule that the owner no longer supports, and `Results.best` is the tool that
ranks configurations (A-07). The cross-correlation returns in A-10, as the Gaussian estimator
with `mode='lag'`.

**Change.** Delete both functions, their exports in `visualize/__init__.py` and their tests in
`tests/test_visualize.py`.

**Pages.** Remove the two rows of `USING.md` and the entry of `MESSAGES.md`.

**Done when.** `grep -rn "plot_cross_correlation\|analyze_mi_heatmap" neural_mi/ reference/`
shows nothing.

### X-09 One switch for messages, and a warning for what needs attention

**Why.** The library reports through two channels, and the rule that separates them is written
nowhere. The logger shows what the library is doing, and `set_verbosity` sets its level for the
session (logger.py:109-136). `set_verbose` is a second function for the same switch
(logger.py:150-160). It calls `set_verbosity` with one of two levels, no code or tutorial calls
it, and three pages document it. `run(verbose=...)` sets the level for one call. Python warnings
are the second channel. Some messages are on the wrong channel today. The fallback to an MLP
decoder is a log line that the default level hides (B-09), and a ladder that cannot be fitted is
a log line at error level followed by a silent drop (analysis/rigorous.py:577-578).

**Change.**
1. Write the rule in the docstring of the messages module, as principle 4 of `plan/README.md`
   states it. What the library is doing is a log message. Anything that may change how a result
   should be read is a Python warning. A call that cannot proceed raises.
2. Keep `set_verbosity(level)` for the session and `verbose=` for one call. Delete `set_verbose`
   and its export in `neural_mi/__init__.py`.
3. List every call of `logger.warning` and `logger.error` in the package, and turn each one into
   a Python warning or an exception by the rule.
4. Correct the docstring of `run`. It says that `verbose=False` shows only errors, while the
   code sets the level to warnings (run.py:1074).

**Pages.** Remove `set_verbose` from `reference/MESSAGES.md:875`, `reference/USING.md:1734` and
`docs/source/api_reference.rst:107`, and state the rule once in `reference/USING.md`.

**Done when.** `grep -rn "set_verbose\b\|logger.warning\|logger.error" neural_mi/` shows nothing.

### X-10 Correct the comments that name things that are gone

**Why.** These lines describe code that no longer exists, and a reader follows them to nothing.

**Change.** Correct or delete each of these.
1. The docstring of `data/__init__.py`, with its `ContinuousDataset` and `SpikeDataset`.
2. The orphan comment at `data/handler.py:540-541` and the log text "Created PairedDataset" at
   line 556.
3. The docstring at `data/views.py:18-19`, if X-02 has not removed the file.
4. `analysis/pairwise.py:109`, with its `_dispatch_splits`.
5. `utils.py:72` (`workflow.py`), `utils.py:138` and `utils.py:146` (a `step_size` of `block_mi`).
6. The mentions of `VarMLP` and `base_params` in the docstring of `VariationalWrapper`, at
   `models/decoders.py:9` and at `training/trainer.py:464`.
7. The `**kwargs` in the docstring of `build_decoder`, and the claim in the docstring of
   `build_optimizer_and_scheduler` that `precision.py` uses it.
8. The module docstring of `config.py`. It lists only some of the configs.
9. The sentence at `reference/INTERNALS.md:191` that says permutation trials use
   `dispatch_tasks`, and the one at `reference/INTERNALS.md:50` that says `data/` contains the
   preprocessing.

**Done when.** `grep -rn "VarMLP\|_dispatch_splits\|ContinuousDataset" neural_mi/` shows nothing.

### X-11 Replace seaborn by matplotlib (optional)

**Why.** seaborn is a required package, and the library uses four of its functions: `heatmap`
(results.py:275, 295 and visualize/plot.py:168), `stripplot` (plot.py:380), `despine` (plot.py:105,
245, 320, 416, 599) and `color_palette` (plot.py:293).

**Change.** Write each of the four with matplotlib and remove seaborn from `pyproject.toml`.

**Numbers.** No number changes, but every figure looks slightly different, so the README figures
and the tutorials must be drawn again in P-13.

**Done when.** `grep -rn seaborn neural_mi/ pyproject.toml` shows nothing.
