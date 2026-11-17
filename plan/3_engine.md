# Phase E: one path from a call to its result

This phase is the centre of the plan. It starts from an engine in which each mode reaches the
trainer in its own way. It ends with the path that `plan/README.md` describes: one row plan,
one declaration of each quantity, one function that trains the terms, one runner, one ladder, one
preparation loop and one call record.

The steps are ordered so that each one removes a copy while the public API stays as it is. Phase A
changes the API afterwards. Every step is checked with the characterisation snapshot of S-02. A
step says under **Numbers** where a number may change, and such a step is merged only once the
change is understood and written in its commit message.

Three terms are used throughout. A *row plan* is the `RowPlan` of `data/plan.py`: every stream of
a call on one grid of rows. A *term* is one estimate `I(X side; Y side)`, made by one trained
network or by one KSG estimate. A *rung* is the estimate of a quantity on one chunk of the rows
at one value of gamma, and the rungs of one configuration and repeat form its *ladder*.

### E-01 One function runs every list of tasks

**Why.** The loop that runs tasks in sequence or in a pool is written five times
(analysis/sweep.py:111-160, analysis/rigorous.py:787-799 and 1096-1111, analysis/pairwise.py:106-132,
analysis/permutation.py:274-280). `lag.py:117-119` and `dimensionality.py:307-313` borrow a
private method of `ParameterSweep` instead, and `quantities.py:221` uses `parallel.dispatch_tasks`.
The copies differ. The two borrowed calls skip the choice of device, so `mode='lag'` and
`mode='dimensionality'` train on the CPU whenever `device` is unset, even on a machine with a GPU.
`lag.py` builds `ParameterSweep(x_data=None, ...)`, so the memory warning of sweep.py:127-151
never fires for it. The device is timed again for each term and each chunk.

**Change.**
1. Add `analysis/runner.py`.

   ```python
   def run_tasks(tasks, worker, *, n_workers=1, desc='', show_progress=True, device=None):
       """Run worker(task) for every task and return the results in the order of the tasks."""
   ```
   It chooses the device once per call and reuses the choice, gives the memory warning from the
   data of the tasks, runs in sequence or in `worker_pool`, captures the workers' warnings with
   `CapturedTask` and `released`, and shows one progress bar.
2. Put the rule for nested loops in one function of the same module. The outermost loop with
   more than one item gets the workers, and the loops inside it run in sequence.
3. Switch each of the eight places to `run_tasks`, one commit per place.
4. Fold `analysis/device.py` into the runner.

**Tests.** Update `tests/test_device.py`, `tests/test_parallel.py` and `tests/test_sweep.py`. Add
a test that `mode='lag'` with 22 tasks and `dataset_device='auto'` gives the memory warning. Add
a test, with the accelerator patched, that the device is chosen once in a conditional call and
once in a lag call.

**Numbers.** They are unchanged on a CPU. On a machine with a GPU, `lag` and `dimensionality`
may now train on it, as the other modes do.

**Done when.** `grep -rn "worker_pool(" neural_mi/` shows only `runner.py` and `parallel.py`.

### E-02 A quantity is declared once

**Why.** The terms of each quantity are written in several places. For conditional MI they are
in `MODES['conditional']` (analysis/modes.py:814-820, the fields `terms`, `components`, `symbol`,
`title` and `label`), in `run_conditional_mi` (analysis/conditional.py:65-70) and in
`_cmi_rigorous_scalar` (conditional.py:94). Interaction information and transfer entropy repeat
the pattern, and the reverse direction of transfer entropy is a separate constant
(`_TRANSFER_REVERSE`, modes.py:51-52).

**Change.**
1. Add `analysis/quantity.py` with two records.

   ```python
   @dataclass(frozen=True)
   class Term:
       key: str                  # its column in result.runs, such as 'mi_xw_y'
       sign: int                 # +1 or -1 in the combination
       x: Tuple[str, ...]        # the plan's streams on the X side
       y: str = 'y'
       label: str = ''           # its name in messages, such as 'X,W;Y'

   @dataclass(frozen=True)
   class Quantity:
       name: str
       symbol: str
       terms: Tuple[Term, ...]
       signed: bool = False      # the value may be negative
       title: str = ''
       joint: Optional[str] = None   # the key of the term that contains every variable
   ```
2. Declare `MI` with one term, `CONDITIONAL` with two, `INTERACTION` with three, and `TRANSFER`
   with its reverse. Keep every key, sign and label as it is today. Keep the order of the streams
   of each joint term: conditional MI reads `('x', 'w')` (conditional.py:66) and interaction
   information reads `('w', 'x')` (interaction.py:94). The second stream of the pair is the
   conditioning set of its difference, and KSG reads the pair in that order (task.py:140-142).
3. Move to the same module the functions that combine terms: `amplification_factor`,
   `warn_combination`, `rung`, `combined_spread`, `term_values`, `combine_terms` and
   `_refuse_different_rows` (analysis/sweep.py:240-572), and `directionality_index` of B-11.
4. Give `Mode` one field `quantity` in place of `terms`, `signed`, `label`, `symbol`,
   `components` and `title`. Derive `_DIFFERENCES`, `_SIGNED` and `_QUANTITY_NAMES`
   (modes.py:894-896) from it.

**Tests.** Update `tests/test_mode_registry.py`, `tests/test_combining_terms.py`,
`tests/test_amplification_factor.py` and `tests/test_paired_terms.py`.

**Done when.** Inside `neural_mi/analysis/`, the strings `'mi_xw_y'` and `'i_xypast_yfuture'`
appear only in `quantity.py`.

### E-03 One function trains the terms of a quantity

**Why.** Three functions train the terms of a difference. `_joint_marginal_difference`
(analysis/sweep.py:575-678) trains two terms with two `ParameterSweep` runs.
`_train_single_term` (analysis/interaction.py:27-44) trains the third term of interaction
information. `ParameterSweep.run` trains the one term of plain MI. Each quantity then computes
its value twice: once inside `run_conditional_mi`, `run_interaction_information` or
`run_transfer_entropy`, and again in `modes._difference_rows` (modes.py:199-236). Only the
second value is used.

**Change.**
1. Add `analysis/estimate.py`.

   ```python
   def estimate_terms(plan, quantity, params, grid=None, n_workers=1):
       """Train every term of `quantity` on the rows of `plan`, for every point of `grid`.
       Return the task results by term key, in the order of the grid."""
   ```
   It builds one task for each term and each grid point with
   `TermRows(plan, Roles(term.x, term.y))`, labels each network with its term, and runs all the
   tasks in one call of `run_tasks`. It takes over the grid handling of
   `ParameterSweep._prepare_tasks` (sweep.py:162-229).
2. Make `produce_difference` (modes.py:266-287) call `estimate_terms` and combine once. Move the
   log line and the call of `warn_combination` (sweep.py:661-677) into the producer, with their
   texts unchanged, so that each warning is still given once.
3. Replace `_cmi_rigorous_scalar` and `_ii_rigorous_scalar` by one module-level function that
   calls `estimate_terms` and `rung` for a quantity passed as an argument.
4. Delete `_joint_marginal_difference`, `_train_single_term`, `run_conditional_mi`,
   `run_interaction_information`, `modes._call_difference`, `_conditional_terms`,
   `_interaction_terms` and the field `Mode.run_terms`. The files `conditional.py` and
   `interaction.py` go.

**Tests.** The deleted functions are named 24 times in five test files
(`tests/test_conditional.py`, `test_interaction.py`, `test_combining_terms.py`,
`test_paired_terms.py`, `test_transfer.py`). Rewrite those tests against `estimate_terms` with
`combine_terms`, or through `run()`.

**Numbers.** They should not change. The terms of a repeat keep their role and their repeat
seed (seeds.py, task.py:120-123), so the order in which they train does not matter. The snapshot
confirms it.

**Done when.**
`grep -rn "_joint_marginal_difference\|run_conditional_mi\|run_interaction_information" neural_mi/`
shows nothing.

### E-04 Transfer entropy trains from a row plan

**Why.** `run_transfer_entropy` builds its arrays with `build_rows(series, te_spec(...))` and
passes the X side of the joint term as a tuple of arrays, with `joint_streams` and
`marginal_streams` naming the streams behind each part (analysis/transfer.py:159-175). The eight
other time quantities build the same groups and pass them through a row plan
(quantities.py:384-388). The library contains two constructions of one thing.

**Change.**
1. Give `RowPlan` three optional fields for rows that are built already: `geometry` (a
   `RowGeometry`), `phases` (the stride phases, held today under `PHASES_KEY` in `base_params`)
   and `kinds` (each stream's kind for input scaling, held today in `_stream_kinds` and
   `_term_streams`). `RowPlan.geometry()` returns the first. `RowPlan.build()` attaches the
   shifter that `try_build_row_phase_dataset` builds today (task.py:191). The task reads the
   kinds from the plan (task.py:297-299).
2. Add `plan_by_offsets(series, spec, stride, shift, kinds)` to `data/offsets.py`. It calls
   `build_rows` and returns the plan of the three groups under the names `x` (group A), `y`
   (group B) and `w` (group C).
3. Make `run_transfer_entropy` build that plan and call `estimate_terms(plan, TRANSFER, ...)`.
   For `bidirectional=True` it builds a second plan from `te_spec(source='y', target='x')` and
   trains the reverse quantity on it.
4. Delete `joint_streams`, `marginal_streams`, `_term_streams` and the tuple branch of
   `ParameterSweep.__init__` (sweep.py:95-107).

**Tests.** Update `tests/test_transfer.py`, `tests/test_conditional_transfer_rigorous.py` and
`tests/test_input_scaling.py`.

**Numbers.** Establish first. For streams of one kind, both constructions hold the same rows in
the same order, so the numbers should not change. A conditioning group that mixes kinds, such as
a continuous Y with a binned W, is scaled part by part today and as one mixed group on the plan
path (quantities.py:367-371). Run both on one such case and record the outcome.

**Done when.** `grep -rn "_term_streams\|joint_streams" neural_mi/` shows nothing.

### E-05 Every mode trains from a row plan

**Why.** `_run_flat` builds the rows in five branches (run.py:1353-1448), and
`run_training_task` accepts four forms of data (analysis/task.py:182-224). Only conditional MI
and interaction information use the plan. Each branch repeats the same bookkeeping: the
`'preprocessed'` flags (run.py:1405-1412 and 1429-1434), the retention keys (1369-1371 and
1442-1444) and the geometry (1372 and 1436). One branch counts the rows outside the plan
(`_deferred_window_count`, run.py:1520-1551).

**Change.** At the end, `_run_flat` builds one `RowPlan` for every mode, calls `materialize()`
when the rows do not shift, and every task receives `TermRows(plan, PAIR)`. Each part below is
one commit.

a. **Estimate and sweep.** `produce_single_network` (modes.py:87-91) calls
   `estimate_terms(plan, MI, ...)`. Check with the snapshot each of the three forms that it
   replaces: rows given already (run.py:1398-1416), rows cut once in `run()` (1417-1448) and rows
   cut in each task (1379-1397).

b. **The ladder of plain MI.** `_Chunker` already cuts a plan (analysis/rigorous.py:614-654).
   Pass the plan, so that `AnalysisWorkflow._prepare_tasks` always builds
   `TermRows(cut, PAIR)`, as it does today only when the cut is a plan (lines 851-852).

c. **Lag.** `run_lag_analysis` shifts the raw arrays with `_shift_data` and every task windows
   them (analysis/lag.py:88-115). Build one plan per lag from the shifted streams and leave it
   unbuilt, so that each task cuts its rows in its own process as today. Truncate for
   `equalize_n` (lag.py:96-106) before the plan is built.

d. **Pairwise.** `_slice_channel` (analysis/pairwise.py:45-57) slices raw lists, raw arrays and
   built rows. Add `select_channels(stream, indices)` to `data/` and build one plan of two
   streams per pair. Put the tasks of every pair and repeat in one list for `run_tasks`. This
   replaces `_dispatch_pairs` (pairwise.py:106-132) and the two copied branches for the self and
   the cross map (lines 205-280).

e. **Dimensionality.** Establish first what the mode receives when `shift_windows` is on
   (run.py:1379). `_halves` (analysis/dimensionality.py:146-262) splits built rows into two
   views. Wrap each pair of views in a plan of rows given already, and replace the private calls
   of `ParameterSweep` (dimensionality.py:307-313) by `estimate_terms`.

f. **Precision.** `run_precision_analysis` trains one network and then evaluates it on corrupted
   rows (analysis/precision.py). Give the baseline task `TermRows(plan.materialize(), PAIR)`. The
   corruption keeps reading the rows of the trained dataset.

g. **The permutation test.** `shift_x` already moves X inside a plan
   (analysis/permutation.py:186-193). Its branches for a bare tensor and for a spike list remain
   only as the helpers that move the data of one stream. `_window_rows` (permutation.py:131-163)
   reads the size of a block from `plan.geometry()` and the plan's phases.

h. **Delete what the old forms needed.** Delete the branches of `run_training_task` for data
   that is not a `TermRows` (task.py:187-224) with the dataset cache (task.py:51-99),
   `RowPlan.of_pair` and `count_rows`, the size inference in `ParameterSweep.__init__`
   (sweep.py:86-109) and `AnalysisWorkflow.__init__` (rigorous.py:713-719), the `'preprocessed'`
   flags and their readers (sweep.py:193-196, modes.py:599-602, permutation.py:150), the
   retention keys in `base_params`, `_deferred_window_count`, `Mode.windows_in_task`, and the
   class `ParameterSweep`.

**Tests.** `ParameterSweep` is named 15 times in four test files. Update them, with
`tests/test_sweep.py`, `test_workflow_internals.py`, `test_pairwise.py`, `test_dimensionality.py`,
`test_precision.py`, `test_permutation.py` and `test_plan.py`.

**Numbers.** Each part must leave the snapshot unchanged. Report a part that cannot, with the
cause, before it is merged.

**Done when.**
`grep -n "create_dataset\|of_pair\|try_build_row_phase_dataset" neural_mi/analysis/task.py` shows
nothing, and the part of `_run_flat` that builds rows has one branch.

### E-06 One fit of the ladder

**Why.** The fit that extrapolates a ladder is written twice, in `_post_process_and_correct`
(analysis/rigorous.py:458-581) and inline in `run_rigorous_scalar_analysis` (lines 1125-1216).
The copies have drifted, and a reproduction showed three differences. The first copy counts the
gammas that are left, and the second counts rows (line 1125). The first warns when too few
gammas are left (503-506), and the second does not (1169). On too few points the first logs an
error and drops the configuration (577-578), and the second raises or falls back to every gamma
(1151-1163). The defaults of the fit are literals in three places (rigorous.py:806-811 and
964-970, modes.py:466-473).

**Change.**
1. Write one function in `rigorous.py`.

   ```python
   def fit_ladder(ladder: pd.DataFrame, settings: Mapping[str, Any], where: str) -> Dict[str, Any]:
       """Fit the rungs of one configuration and repeat, and return the fit with its diagnostics."""
   ```
   It calls `_drop_rungs`, `_unfittable_ladder`, `_find_linear_region`, `_extrapolate_mi`,
   `_compute_fit_diagnostics` and `_compute_per_gamma_diagnostics`, as both copies do.
2. Give the three differences one behaviour. Count the distinct gammas that have a value. Give
   the warning for too few gammas in every case. On too few points return a fit that is marked
   unreliable, and raise `InsufficientDataError` only when no rung has a value.
3. Add one schema `RIGOROUS_SETTINGS` to `defaults.py` with the content of `modes._FIT`,
   `gamma_range` and `temporal_chunking`. `fit_ladder` reads its settings from the resolved
   mapping, and no signature repeats a default.
4. Keep `per_gamma_train_mi_spread`, `per_gamma_ceiling_mi` and `per_gamma_saturation` in the
   result, under `details[config_id]`. They are computed today (rigorous.py:397-455) and then
   dropped by `modes._FIT_KEYS` (modes.py:43-49), although the docstring of `_Chunker` points
   the reader to them.
5. Use one set of warning texts, with `where` naming the call.

**Tests.** Update `tests/test_rigorous_diagnostics.py`, `tests/test_workflow_internals.py` and
`tests/test_conditional_transfer_rigorous.py`. Add a test that the same short ladder gives the
same warning and the same flags for plain MI and for conditional MI.

**Pages.** `reference/MESSAGES.md` has two families of headings for the fit. They become one.

**Numbers.** A fit is unchanged where the two copies agreed. A ladder with too few gammas now
returns an unreliable fit in every case.

**Done when.** `_extrapolate_mi` is called from one function.

### E-07 One ladder for every quantity

**Why.** `AnalysisWorkflow` (analysis/rigorous.py:690-867) builds one flat list of tasks for
every configuration of plain MI. `run_rigorous_scalar_analysis` (lines 955-1216) builds one
ladder per configuration and repeat for a difference quantity, runs the chunks in a pool, and
trains the terms of a rung in sequence inside a worker. Only `_Chunker` is shared.

**Change.**
1. Write one function.

   ```python
   def ladder(plan, quantity, params, grid, settings, n_workers=1) -> List[Dict[str, Any]]:
       """One fit for every configuration and repeat of `grid`."""
   ```
   For every configuration, repeat, gamma and chunk, it cuts the plan with `_Chunker.cut` and
   builds the tasks of the rung's terms with the role `gamma{g}chunk{c}`. Both copies use this
   role today (rigorous.py:856 and 1090). All the tasks go through one call of `run_tasks`.
2. Combine the terms of each rung with `combine_terms` and `rung`. `rung` keeps the rule that a
   rung whose joint term did not resolve is left out of the fit (sweep.py:384-402).
3. Fit each ladder with `fit_ladder`.
4. Merge `produce_rigorous` and `produce_difference_rigorous` (modes.py:140-164 and 290-317)
   into one producer.
5. Delete `AnalysisWorkflow`, `run_rigorous_analysis`, `run_rigorous_scalar_analysis`,
   `_run_scalar_fn_task`, `_post_process_and_correct`, the scalar function of E-03,
   `_te_rigorous_scalar` and `Mode.rigorous_scalar`. Rename `rigorous.py` to `ladder.py`.

**Tests.** The old entry points are named 30 times in eight test files. Rewrite those tests
against `ladder` and `fit_ladder`.

**Numbers.** Plain MI and the quantities on a plan keep their chunks, roles and seeds, so their
numbers should not change, and the snapshot confirms it. The ladder of transfer entropy changes.
Today it cuts the series and builds the histories inside each chunk (rigorous.py:1075-1091).
After E-04 it cuts the built rows, as the eight other time quantities already do, so each chunk
keeps the rows at its edges.

**Done when.** `grep -rn "AnalysisWorkflow\|run_rigorous_" neural_mi/` shows nothing.

### E-08 The rigorous correction becomes a modifier inside the engine

**Why.** The correction is a mode for plain MI and a field of three configs for the difference
quantities. `modes._difference` branches on it (modes.py:748-752), and three functions prepare
its settings in two ways (`_prepare_rigorous`, `_rigorous_terms` and `_scalar_rigorous_kwargs`,
modes.py:478-545).

**Change.**
1. The engine resolves the settings of the correction once, for every mode, into
   `ctx['rigorous']`. The value is `None` or the resolved `RIGOROUS_SETTINGS`.
2. Give `Mode` one field `rigorous_refused` with the reason, like the field `permutation`.
   `precision` and `dimensionality` refuse, because each reads a curve from trained networks and
   has no single value to extrapolate.
3. Add one function `estimates(plan, quantity, params, grid, ctx)` that returns the rows of a
   plain estimate or of a ladder. The producers of `estimate`, `sweep`, `conditional`,
   `interaction` and `transfer` call it, and so do `lag` and `pairwise`. For those two, each lag
   or pair gets its own ladder, and the call announces the number of networks.
4. Keep `mode='rigorous'` as a name that sets the modifier on a sweep, until A-01 removes it.
5. Lift the refusal of B-12, so that each direction of transfer entropy gets its own ladder.
6. `Results._is_rigorous` (results.py:503) reads the recorded modifier. The figures of the lag
   curve and of the pairwise map draw the error of the fit when the modifier is on.

**Tests.** Add one case each for `lag` and `pairwise` with `gamma_range=range(1, 4)`. Update the
table check in `tests/test_mode_registry.py`.

**Pages.** The table of `reference/USING.md` that lists what each mode accepts is checked
against `MODES`.

**Done when.** `ctx['rigorous']` is written in one place and no producer has a rigorous twin.

### E-09 The named functions call the engine directly

**Why.** `quantities._by_offsets` builds rows and then calls `run()` on rows that are built
already. It passes their geometry, stride phases and stream kinds through three private
arguments of the public function (run.py:669-671). It also repeats decisions that `run()` makes:
the input scaling (quantities.py:352-371), the split of the grid (332-345) and the paths of
saved networks (138-150). `transfer_entropy` calls `run(mode='transfer')` instead, and the hook
of that mode (modes.py:548-623) prepares the streams in a second way.

**Change.**
1. Split `_run_flat` where the rows exist. The part after that point becomes
   `execute(call, plan, quantity)`: it runs the producer, builds the result and attaches the
   nulls of a permutation test.
2. `_by_offsets` prepares its series, builds the plan with `plan_by_offsets` and calls
   `execute`. The quantity is `MI` or `CONDITIONAL` under the function's own name, symbol and
   labels.
3. `transfer_entropy` and `conditional_transfer_entropy` use `_by_offsets` with `te_spec`, like
   the other time quantities. `preprocessing.RATE_QUANTITIES` (preprocessing.py:41-42) already lists both
   functions, so `_prepared_series` (quantities.py:294-295) passes the same `rate=True` that the
   hook passes today (modes.py:593-595).
4. Delete `quantities._call`, `_task`, `_rigorous_settings`, `_difference_fit` and `_FIT_FIELDS`,
   `modes._prepare_transfer`, `_transfer_terms` and `_to_2d`, `run_transfer_entropy`,
   `Mode.builds_own_rows`, and the arguments `_rows`, `_row_phases` and `_stream_kinds` of
   `run()` and `_run_flat` with the code that reads them (run.py:915-923, 1215-1217, 1300).
5. Keep `mode='transfer'` as a call of `quantities.transfer_entropy` until A-03 removes it.
6. `block_mi` keeps building its rows through the processors, so it keeps calling `run()`.

**Tests.** Update `tests/test_quantities.py`, `test_quantities_sweep.py`, `test_transfer.py`,
`test_known_answers.py` and `test_oracle.py`.

**Numbers.** They are unchanged for the eight functions that are on the offset path today. For
transfer entropy they follow from E-04.

**Done when.** `grep -n "_row_phases\|_stream_kinds" neural_mi/run.py` shows nothing.

### E-10 One preparation loop

**Why.** A grid key that changes the rows needs one preparation per value. This loop is written
three times. `run._preparation_grid` and `_run_preparation_grid` (run.py:468-491 and 530-569)
handle the settings of a processor. `quantities._named_run` (quantities.py:194-225) handles a
function's own window, with `parallel.dispatch_tasks` and `merge_results`. A recursion in
`_by_offsets` (quantities.py:332-345) handles the first kind inside the second.

**Change.**
1. Add `run_preparations(call, build)` to the engine. It splits the grid with
   `split_preparation` into the keys that change the rows and the others. For each combination
   of the first, it calls `build(values)` for the plan and the quantity, then `execute`. It
   merges the parts with `assemble.merge_results`.
2. `run()` passes a `build` that makes the plan from the processors. A named function passes one
   that makes it by offset, and declares its own window as a key that changes the rows.
3. Use the rule of E-01 for the workers. Establish first what `_run_preparation_grid` does
   today, and keep one behaviour for both front doors.
4. Label the saved networks in one place (today `_with_save_path` at quantities.py:138-150 and
   `with_model_labels`).
5. Delete the three copies and `parallel.dispatch_tasks`.

**Tests.** Update `tests/test_grid_contract.py`, `tests/test_quantities_sweep.py` and
`tests/test_run.py`.

**Done when.** `merge_results` is called from one place.

### E-11 A result describes itself

**Why.** `Results` looks up `MODES` to learn how to draw and summarise itself (results.py:151,
210, 222, 500 and 586, through imports inside functions), and `MODES` contains the drawing
functions (modes.py:35-38). The analysis package therefore imports the figures, and the results
import the analysis package. A result loaded from disk depends on the registry of the version
that loads it. A `mi_rate` result is drawn under the title "Conditional MI components" with the
bars "I(X,W;Y)" and "I(W;Y)".

**Change.**
1. Give `Results` one field `view`, a plain dict that `build_results` writes. It contains the names
   of the figure, the summary and the overlay, the quantity's name, symbol and title, the keys
   and labels of its terms, the axis keys, and the flags that `Mode` contains today (`signed`,
   `extrapolates`, `ranks`, `own_figure`, `draws_per_config`, `groups_by_axis`,
   `axis_is_networks`).
2. Keep one table from figure name to function in `visualize/figures.py` and one from summary
   name to function in `results.py`. `Results._spec` (results.py:498) and the five lookups read
   `self.view`.
3. `Mode` contains the three names instead of the functions, and `modes.py` imports nothing from
   `results` or `visualize`.
4. Give transfer entropy a summary function of its own, in place of the columns that
   `Results.summary` names itself (results.py:706-710). Make `animate.py` read the axis names
   from `view`.
5. Store the rungs that a fit left out in `details` when the result is built, and remove the
   import of `analysis.rigorous.rungs_left_out` from `visualize/plot.py:384`.
6. `save`, `load`, `to_dict` and `to_json` carry `view`.

**Tests.** Update `tests/test_mode_registry.py`, `tests/test_results.py`,
`tests/test_visualize.py` and `tests/results_factory.py`. Add a test that a `mi_rate` result
names itself and its terms. Add a test, in a subprocess, that a saved result is loaded and drawn
without `neural_mi.analysis` in `sys.modules`.

**Done when.** `grep -rn "neural_mi.analysis" neural_mi/results.py neural_mi/visualize/` shows
nothing.

### E-12 One call record

**Why.** `run()` flattens the configs into three dicts (run.py:838-924). `_run_flat` takes about
80 named arguments (952-1035) and puts about 40 of them back into `base_params` with `_inject`
(1105-1180). Its signature repeats schema defaults as literals, so the set of keys that the
caller set (run.py:1204) wrongly contains 16 keys. `ParameterValidator` receives `locals()` (line
1205), and the hooks of the modes read a namespace of 40 fields (1465-1480). A mode's settings
are defined four times: in its config class, in `MODES[mode].settings`, in the signature of
`_run_flat` and in the signature of its analysis function.

**Change.**
1. Add the record to the engine.

   ```python
   @dataclass
   class Call:
       mode: str
       settings: Dict[str, Any]             # every base setting, resolved
       set_by_caller: FrozenSet[str]        # the keys given in a config or in the grid
       mode_settings: Dict[str, Any]        # the mode's own settings, resolved
       rigorous: Optional[Dict[str, Any]]
       permutation: Optional[Dict[str, Any]]
       grid: Dict[str, list]
       streams: Dict[str, Any]              # for each stream: processor, parameters, clock
       preprocessing: Optional[Preprocessing]
       labels: Dict[str, Any]
       n_workers: int
       seed: int
       verbose: Optional[bool]
       show_progress: bool
       device: Optional[str]
   ```
2. One function `resolve_call(...)` builds the record for both front doors. It reads each config
   through one method `settings()`. The method returns the fields that are set, under their
   schema names, and replaces `to_base_params`, `to_kwargs`, `to_analysis_kwargs`,
   `to_w_kwargs` and `to_labels` (config.py).
3. `ParameterValidator` validates the record. `apply_defaults` fills `settings` from
   `BASE_PARAMS_SCHEMA` and `mode_settings` from `MODES[mode].settings`. `Mode.overrides`
   receives `settings` and `set_by_caller`.
4. The hooks and the producers read the record. Delete the namespace, `_inject`,
   `_ENGINE_PARAMS` (run.py:1511-1515), the named arguments of `_run_flat` and the second copy
   of the code that sets the log level (run.py:1066-1077, since `call_verbosity` at line 933
   already does it).
5. Remove every literal default that the test of S-05 lists. Each analysis function receives its
   resolved settings. This covers `precision.py:45-47`, `dimensionality.py:428-442` and the dead
   fallbacks at `modes.py:483-485`.

**Tests.** `_run_flat` is called 7 times in `tests/test_run_config_api.py`,
`tests/test_schema_defaults.py` and `tests/test_settings_are_read.py`. Rewrite those tests
against `resolve_call`. Add a test that `set_by_caller` contains exactly the keys that the caller
gave.

**Pages.** `reference/INTERNALS.md` describes `_run_flat` in six places.

**Done when.** `_run_flat` does not exist and the allowlist of S-05 is empty.

### E-13 A typed task record

**Why.** A task is the tuple `(x, y, params, run_id)`. Its `params` contains the settings together
with keys that are not settings: `_role`, `_difference_term`, `_model_labels`, `_keep_trained`,
the geometry of the rows, and the sizes `input_dim_x` and `n_channels_x` that the task writes
itself (analysis/task.py:244-289). The schema lists some of these beside the user's settings
(defaults.py:155-170).

**Change.**
1. Add a record `Task` with the fields `rows`, `settings`, `role`, `run_id`, `labels`, `term`
   and `keep_trained`.
2. `run_training_task(task)` reads these fields, and the sizes become local values that it
   passes to `build_critic`.
3. Remove the internal keys from `BASE_PARAMS_SCHEMA` and the underscore keys from `params`.

**Tests.** `run_training_task` is named 18 times in five test files. Update them.

**Done when.** `grep -rn "'_role'\|'_difference_term'\|'_keep_trained'" neural_mi/` shows nothing.

### E-14 `run.py` becomes `api.py` and `engine.py`, and `modes.py` keeps the declaration only

**Why.** `from .run import run` makes the package attribute `run` hide the module `neural_mi.run`,
and `tests/test_run_config_api.py:13` and `:196` work around it with `importlib`. `modes.py` is
at once a registry, a set of producers and a set of hooks.

**Change.**
1. Move the public `run()` to `neural_mi/api.py`. Move `Call`, `resolve_call`,
   `run_preparations` and `execute` to `neural_mi/engine.py`. Delete `run.py`.
2. Keep `Mode` and `MODES` in `analysis/modes.py`. Move each producer and each hook to the module
   of its procedure (`lag.py`, `pairwise.py`, `dimensionality.py`, `precision.py`), and the
   producers of plain estimates and ladders to `analysis/estimate.py`.
3. Move the last imports out of functions and empty both allowlists of S-03.

**Tests.** Update every test that imports `neural_mi.run`, and remove the two workarounds.

**Pages.** Update `docs/source/api_reference.rst`.

**Done when.** `python -c "import neural_mi.api, neural_mi.engine"` works, and the allowlists of
S-03 hold only the optional packages.
