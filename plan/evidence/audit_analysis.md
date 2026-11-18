1. MAP (neural_mi/analysis/, 6203 lines)

- modes.py (909): three jobs in one file. (a) It is the registry: the `Mode` dataclass, which has 37 fields covering production, validation, plotting and docs, and the `MODES` dict. (b) It holds the producers (`produce_*`) and `produce()`, which turn a mode's output into repeat rows. (c) It holds the per-mode "prepare" hooks that build ctx. Callers: run._run_flat, permutation._trial, and results/validation (they read MODES).
- assemble.py: grid helpers (split_grid, config_lookup, split_preparation), unit conversion (convert_record), `aggregate`, `build_results` and `merge_results`. This is the only place dataframes are built; results.py only holds the Results class and drawing. Called by run.py, quantities.py, modes and permutation.
- sweep.py: `ParameterSweep` (build tasks, then run them in sequence or in a pool) and `merge_grid_values`. It also holds the whole difference-quantity combine engine: amplification_factor, warn_combination, term_values, combine_terms, combined_spread, rung and _joint_marginal_difference. That engine is about 440 of the file's 678 lines.
- task.py: `run_training_task` (a 435-line function) and a dataset cache. Called by sweep, rigorous, precision and device.
- rigorous.py: the gamma-ladder fit, `_Chunker`, `AnalysisWorkflow` with `run_rigorous_analysis`, and `run_rigorous_scalar_analysis`.
- permutation.py: shift_x, permutation_nulls, attach_nulls. Called by run.py.
- offsets.py: build_rows, one_step_series, resolve_stream_processing. Called by transfer, modes and quantities.py.
- lag, precision, pairwise, dimensionality, conditional, interaction, transfer: the per-mode procedures, called only by modes.
- __init__.py: exports 9 names from 8 modules. Nothing outside tests imports them. `run_interaction_information` is missing while its siblings are there, and `AnalysisWorkflow` is an internal class. The export list is not coherent.

Modes (the layers are: Mode.produce adapter → produce_X → run_X):

| mode | entry function | signature | returns (nats) | machinery |
|---|---|---|---|---|
| estimate, sweep | ParameterSweep.run | (x, y, bp).run(grid) | list of task dicts | ParameterSweep + with_device |
| rigorous | run_rigorous_analysis | (x, y, bp, sweep_grid, gamma_range, n_workers, **fit) | {corrected_results, raw_results_df} | its own pool loop, with_device, _Chunker |
| lag | run_lag_analysis | (x, y, bp, lag_range, sweep_grid, n_workers, equalize_n, **unused) | list of task dicts | builds its own tasks; ParameterSweep(None, None)._run_parallel; no with_device |
| conditional, interaction | run_conditional_mi, run_interaction_information | (plan, bp, sweep_grid, n_workers) | summary dict + raw_* | _joint_marginal_difference → ParameterSweep ×2 or ×3 |
| transfer | run_transfer_entropy | (x, y, bp, history_window, prediction_horizon, sweep_grid, n_workers, bidirectional, w_data, stride) | summary dict + raw_* + n_samples | same as above |
| difference with rigorous=True | run_rigorous_scalar_analysis | (scalar_fn, x, y, bp, extra_data, …, 7 fit kwargs) | flat fit dict | its own pool loop, _Chunker |
| pairwise | run_pairwise_mi | (x, bp, y_data, sweep_grid, n_workers, pairs) | {mi_matrix, dataframe, records, n_channels} | its own pool across pairs, with a nested ParameterSweep |
| dimensionality | run_dimensionality_analysis | (x, bp, y_data, *, 17 kwargs) | tuple (fits, info) | private ParameterSweep._prepare_tasks/_run_parallel; no with_device |
| precision | run_precision_analysis | (x, y, bp, tau_grid, …, n_workers unused) | {samples, dataframe, details} | run_training_task + with_device; ignores the grid |

There is no shared contract. Argument order, return type, the name of the grid argument (sweep_grid vs AnalysisWorkflow's param_grid) and whether the function loops over configurations all vary by mode.

2. FINDINGS (most valuable first)

F1. The "run a batch of tasks" loop is written 5 times, and 2 modes skip the device choice. Effort M, risk low.
- Copies: sweep.py:111-160, rigorous.py:787-799, rigorous.py:1096-1111, pairwise.py:106-132, permutation.py:274-280.
- lag.py:117-119 and dimensionality.py:307-313 call `_run_parallel` directly and never go through `with_device`. When no device is named, those two modes always train on the CPU, while the other modes time the GPU and may pick it.
- lag also builds a `ParameterSweep(x_data=None, …)` only to reach that method. Its pre-run memory warning reads `self.x_data`, which is None there, so the warning never fires for lag.
- `with_device` runs once per `ParameterSweep.run` call. A rigorous difference run on one process therefore re-times the device for each chunk and each term (only matters when a GPU is present).
- Change: write one `run_tasks(tasks, n_workers, desc)` that calls with_device and then the pool, and choose the device once per call in produce().

F2. Difference quantities are combined twice, and each quantity's terms are declared 3 times. Effort M, risk low.
- run_conditional_mi, run_interaction_information and run_transfer_entropy compute cmi_estimate, te_xy, te_estimate, interaction_info, amplification_factor and mi_estimate_std. modes.produce_difference reads only `raw_*` and `n_samples` and recombines with combine_terms again (modes.py:186-236). A grep confirms only tests read those summary keys.
- The directionality index is computed in two places: transfer.py:208 and modes.py:224. They handle NaN differently.
- Term specs live in conditional.py:65-94, in MODES.terms (modes.py:816/823/831) and in the rung specs of the *_rigorous_scalar functions. The reverse TE spec is a stray constant (modes.py:51).
- Change: run_X returns only the raw term results. MODES.terms is the only spec, and the rigorous scalar functions read it.

F3. The rigorous fit is implemented twice and the copies have drifted. Effort M, risk medium.
- The two copies are `_post_process_and_correct` (rigorous.py:458-581) and the inline fit at rigorous.py:1125-1216. Differences:
  - When too few points are left, the grid path logs an error and drops the configuration silently (578). The scalar path falls back to all gammas instead (1151). Uncertain: both branches appear unreachable after `_unfittable_ladder`.
  - The scalar path never warns when `enough_gamma_points` is false.
  - The scalar path checks the number of rows against min_gamma_points (1125), not the number of gammas.
- Change: one `fit_ladder(ladder_df, settings) -> fit dict`. `AnalysisWorkflow` should become a function, which also removes the coupling where `resolved_chunking_mode` is set in `_prepare_tasks` and read in `run`.

F4. Fit and mode defaults are written as literals in several places, against the "one place for a default" rule in INTERNALS.md. Effort S, risk low.
- Rigorous: modes.py:466-473, rigorous.py:806-811 and rigorous.py:964-970.
- Lists of fit-setting names: modes._RIGOROUS_FIT_KEYS, modes._FIT, _scalar_rigorous_kwargs and quantities._FIT_FIELDS.
- Dimensionality: every default at dimensionality.py:428-442 duplicates MODES settings (modes.py:841-879).
- Precision: precision.py:45-47.
- Change: function signatures use `default_of`-style lookups on MODES[...].settings.

F5. modes.py is a registry, a dispatcher and data preparation at once, and it creates import cycles. Effort M, risk medium.
- Cycles: run → modes at top level (run.py:31); modes → run lazily (modes.py:557); modes → results and validation at top level, while those import modes lazily. They work only because of function-level imports; the analysis package has 37 of them.
- `_prepare_transfer` (modes.py:548-623) is 75 lines of stream preparation. It imports run._warn_small_sample and run._encoders_scale and mutates base_params in place.
- The Mode dataclass pulls private plotting functions (`results._draw_*`) into analysis.
- Change: split into modes.py (the declaration only), producers.py, and move transfer preparation into run or data. Plotting hooks could be registered from results.py.

F6. Each mode goes through two layers of pure adapters. Effort S, risk low.
- modes.py:734-768 has 7 adapters (`_single`, `_lag`, …) that only reshape arguments for produce_X, which in turn reshape them for run_X. `_call_difference` (261) is a one-line indirection, and `produce_single_network` takes an unused `mode`.
- Configurations are iterated two ways: estimate, sweep, lag and rigorous pass the full grid and map results back with config_lookup; difference, pairwise and dimensionality loop over split_grid themselves. Pick one.
- Change: give each producer the uniform signature directly, and replace the `_produced` dict with a dataclass that build_results takes whole (run.py:1488-1493 unpacks 7 keys by hand).

F7. sweep.py is misnamed for most of its content. Effort S, risk low.
- Move the combination engine to analysis/combine.py.
- `interaction._train_single_term` duplicates half of `_joint_marginal_difference`; one `train_term()` should serve both.

F8. Dead code and leftovers. Effort S, risk low.
- `windows_in_task` in ParameterSweep: both branches are identical (sweep.py:214-221). pairwise.py:74 still passes it.
- `ParameterSweep.__init__` and `AnalysisWorkflow.__init__` infer input dimensions (sweep.py:86-109, rigorous.py:713-719). task.py:244-289 always overwrites them.
- Unused parameters: `**kwargs` on ParameterSweep and run_lag_analysis, `std_cols` (no producer sets it, but it is plumbed through run/assemble), `row_values(mode, …)`, `_window_rows(…, ctx)`, precision's n_workers.
- `assemble.expand` has no callers.
- modes.py:439 pops 'raw_results', which precision never sets; precision's docstring (113) documents it anyway.
- `per_gamma_train_mi_spread`, `per_gamma_ceiling_mi` and `per_gamma_saturation` are computed (rigorous.py:397-455) and then dropped by `_FIT_KEYS`, so users never see them, although the `_Chunker` docstring points them to it.
- The `dataframe` returned by pairwise and precision is unused.
- pairwise.py:109 cites a `_dispatch_splits` that does not exist.

F9. Grid product and hashing helpers are copied. Effort S, risk low.
- Grid product appears 4 times: sweep._product_dict, assemble._product, rigorous.py:832-833 and rigorous.py:775-776 (an unreadable `n_combos` expression).
- `_hashable` (assemble.py:170) and `task._make_hashable` (task.py:69) do the same job.

F10. `run_training_task` has natural seams. Effort M, risk low.
- At 435 lines it splits into: building or caching the dataset, inferring dimensions, building critic and decoders, training, extracting embeddings, and assembling the result.
- The conditional `del dataset` and cache check (task.py:566-571) do nothing useful.

F11. Pairwise inconsistencies. Effort S.
- The self and cross branches are copy-paste (pairwise.py:205-280).
- `variable_names_x`/`variable_names_y` (modes.py:351-357) mean plot columns and rows, not the X and Y data, so cross mode stores X's names under `_y`. This is correct but confusing.
- Uncertain: pairwise rows take 'mi' as train_mi without the `all_mi_negative`→0 rule that single_network_rows and dimensionality apply.

F12. Things found outside my area:
- run.py:1529 imports the private `permutation._spike_population_extent`.
- plot.py imports `rigorous.rungs_left_out`.
- run() builds a 40-field untyped SimpleNamespace `call` for the prepare hooks (run.py:1465).

3. Keep as is
- One `produce()` path shared by the observed call and the permutation trials.
- `aggregate`, `build_results` and `merge_results` as the only builder of dataframes.
- `convert_record` as the one unit conversion.
- The `combine_terms` rule (failed vs nothing) and `rung`.
- `_Chunker` as the one chunking path.
- Declaring each mode once in MODES (slim it down, but keep the idea).
- Seeds by role (neural_mi/seeds.py) and offsets.build_rows as the one builder of rows by offset.