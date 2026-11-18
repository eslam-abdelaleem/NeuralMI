Read-only audit of the entry and configuration layer. Nothing was modified. torch is not installed here, so I could not import the package; everything below comes from reading the code, AST checks and grep, plus one mock-package test.

**1. MAP**

- **`__init__.py`**: the public surface. Re-exports `run`, the 15 config classes, `Results`, the exceptions, the quantities and the subpackages. On import it registers IPython cell hooks and installs a global tqdm warning filter.
- **`run.py`**: three layers in one file.
  - Public `run()`: coerces configs and lowers them.
  - `_run_flat()`: the engine.
  - About 400 lines of setting checks (`_check_grid`, `_unread_settings`, `_check_requests`, …) and about 180 lines of data-check reporting (`_warn_small_sample`, `_report_rows`, `_encoders_scale`).
  - Used by `__init__`, `quantities.py`, and `analysis/modes.py:557` (lazy import of run's private functions).
- **`config.py`**: dataclasses whose fields are all None-means-unset, plus `as_config`. Used by `run`, `analysis/modes`, `quantities`.
- **`defaults.py`**: `BASE_PARAMS_SCHEMA` (123 keys: types, minimums, defaults), the processor defaults, and `setting`/`default_of`. Also stream-scaling helpers that belong to data/models, not defaults. Used everywhere.
- **`validation.py`**: `ParameterValidator` (fed `_run_flat`'s `locals()`), `DataValidator`, `ALLOWED_VALUES`, and the per-mode `check_*_settings` functions that `MODES` references.
- **`exceptions.py`**: 4 errors and `CombinationWarning`.
- **`logger.py`**: the logger, verbosity control, repeated-warning grouping, IPython hooks, and worker warning capture (`CapturedTask`/`released`).
- **`seeds.py`**: derives seeds from (call seed, repeat, role). Used by `analysis/task`, `precision`, `permutation`, `dimensionality`, `rigorous`.
- **`device.py`**: automatic CPU/GPU choice by timing a few training steps. Used by `analysis/{sweep,rigorous,precision}`; calls back into `analysis.task`.
- **`parallel.py`**: `worker_pool`, `threads_per_worker`, `dispatch_tasks`. `dispatch_tasks` is used only by `quantities.py`.

**How `run()` gets from arguments to `Results`:**
1. `run()` checks the mode, rejects `**_removed`, and coerces every config with `as_config`.
2. It runs `_normalise_grid`, `_warn_ineffective_settings`, `_check_estimator_params` and `_check_grid`.
3. It lowers the configs (`to_base_params` / `to_kwargs` / `to_labels` / `to_w_kwargs` / `to_analysis_kwargs`) into three dicts. The split is decided by `_ENGINE_PARAMS`, which is read from `_run_flat`'s signature. Then `_check_requests`, seed draw, `resolve_model_path`.
4. Under `call_verbosity` + `collect_repeats` it calls `_run_flat`.
5. `_run_flat`: `stream_processing` for Y and W, sets the log level again, seeds the global RNGs. Then `_preparation_grid` and `_announce_call`; if needed, `_run_preparation_grid` calls `_run_flat` once per part and merges with `merge_results`.
6. It `_inject`s ~40 named arguments back into `base_params`, then runs `_check_custom_split`, the permutation checks, `ParameterValidator.validate` / `apply_defaults`, `MODES[mode].overrides`, `record_rows`, `_announce_loss`, `DataValidator`, and the recurrent-encoder (GRU/LSTM) time-axis check.
7. It runs `scaled_streams` and `preprocessing.prepare`, then builds rows through one of five branches (`RowPlan` / `resolve_stream`), then `_warn_small_sample` and `_warn_if_shift_windows_dead`.
8. It builds the `call` `SimpleNamespace`, and `MODES[mode].prepare(call)` returns `ctx`.
9. `analysis.modes.produce` → `build_results` → `Results`.
10. If `permutation_test`: `permutation_nulls` + `attach_nulls`. Back in `run()`, `_report_saved_networks`.

**2. FINDINGS (most valuable first)**

1. **Configs are flattened, then rebuilt (lowering round trip).** (L effort, M risk)
   - `_run_flat`'s ~80 named parameters (`run.py:952-1035`) are a leftover of the old flat API. Configs are flattened, sorted by signature inspection (`run.py:1511`), then `_inject`ed back into `base_params` (`run.py:1105-1180`).
   - The signature repeats schema defaults as literals (`split_mode='blocked'`, `train_fraction=0.9`, `optimizer='adam'`, `use_amp='auto'`, …), so those keys are always injected. As a result `_pre_default_keys`, passed on as `user_set_keys` (`run.py:1204, 1451, 1479`), wrongly counts ~15 keys as set by the user.
   - Today's consumers only read `shift_windows`, `critic_type`, `n_epochs`, `patience` and `shared_encoder`, so this is a latent bug, not an active one.
   - Change: `run()` builds one resolved call spec (`base_params` plus mode settings plus streams) directly, and the engine takes that.
   - Tests that pin the current routing will need updating: `test_run_config_api.py`, `test_schema_defaults.py`.
2. **Mode settings are defined four times.** (M effort)
   - Config dataclasses, `MODES[...].settings` (`modes.py:466-469, 805`), the `_run_flat` signature (`run.py:980-1000`) and the analysis function signatures (`rigorous.py:964-966`, `precision.py:46`). The defaults 2.0, 0.68, 5, 50 and 0.9 are each written about three times.
   - Dead fallback: `analysis_kwargs.get('curvature_t_threshold', curvature_t_threshold)` (`modes.py:483-485`). The named parameter never lands in `analysis_kwargs`.
   - `ParameterValidator._mode_kwarg`'s two-place lookup (`validation.py:343-359`) exists only because of this split.
   - Change: make each config's fields (with metadata) the schema, and pass the config instance to `prepare`.
3. **Config classes are copy-pasted and the W stream is scattered.** (S–M effort)
   - The 7 rigorous-fit fields appear 4 times (`config.py:315-322, 369-376, 434-441, 467-474`). `Conditional` and `Interaction` are identical. `_W_FIELDS` / `to_w_kwargs` is copied 3 times.
   - W's data lives in mode configs, but its processor and time live in `Processing`.
   - The exporters have five different names. `Estimator.to_base_params` is unused outside tests: `run.py:859-863` builds it by hand under the key `'estimator'`.
   - `Transfer.bidirectional` → `bidirectional_te` (`config.py:390`) exists only because of the flat namespace collision with `Model.bidirectional`.
   - Change: take `w_data` as a `run()` argument; nest `rigorous: Rigorous|bool`; give every config one `lower()` method.
4. **Private things in the public signature.** (S effort)
   - `_rows`, `_row_phases`, `_stream_kinds` (`run.py:669-671`) are passed only by `quantities.py:113-114, 371`. Change: public `run()` delegates to a private `_run()` that `quantities` calls.
   - `**_removed` (`run.py:672, 794`) is a migration shim for an API that was never published. It also tells someone who mistyped an argument (e.g. `seeds=1`) to use config objects. Drop it.
   - The permutation settings are flat arguments (`run.py:664-668`). Change: one `permutation=Permutation(...)` config, or True.
   - Their validation is scattered: `run.py:1047` and `1182-1190`, inside `_run_flat` after global seeding. `n_permutations` is validated nowhere; I grepped and found no check. What a 0 or negative value does is uncertain.
5. **`run.py` holds the wrong responsibilities and creates import cycles.** (M effort, low risk: these are moves)
   - The setting checks (`run.py:93-466`) belong in `validation.py`.
   - The data-check reporting (`run.py:1520-1701`) belongs with `preprocessing` or the data layer. Moving it removes the `analysis.modes` → `run` back-import (`modes.py:557`).
   - Second cycle: `validation` lazily imports `MODES` (`validation.py:363`) while `modes` imports `validation`'s checks (`modes.py:39`). Fix: put each check on the mode spec, or move them into `modes`.
   - `results.py` imports `MODES` lazily in 5 places (151, 210, 222, 500, 586) — this crosses into another agent's area.
6. **`_run_flat` is 554 lines (952-1505).** Natural seams: stream resolution (1054-1064); log level and seeding (1066-1085); `base_params` assembly (1101-1180); validation and defaults (1182-1274); row building, five branches with a duplicated "mark as preprocessed" block (1405-1412 vs 1429-1434); produce and permutation (1465-1501). (M effort)
   - The log-level code duplicates `call_verbosity`, which already wraps the call (`run.py:933` vs `1071-1077, 1502-1505`).
   - It sets `cudnn.deterministic` / `cudnn.benchmark` permanently and never restores them (1084-1085).
   - Tasks reseed the global RNGs themselves (`analysis/task.py:162-164`), so the global seeding may be redundant (uncertain).
7. **Defaults hazards.** (S effort)
   - `apply_defaults` assigns mutable schema defaults (`{}`) by reference (`validation.py:409`), unlike `default_of`, which copies. I found no code that mutates them in place yet, but any future in-place change would corrupt the schema for the whole process.
   - With `verbose=True` it logs one INFO line per unset key (107 keys have defaults).
   - The schema mixes user settings with internal keys (`input_dim_x`, `n_channels_x`, `gamma`, `lag`, `defaults.py:155-170`). Unvalidated `_`-prefixed keys are also put into `base_params` (`_scaled_streams`, `_window_retention`, …).
8. **`ALLOWED_VALUES` repeats the registries** (`validation.py:43-55`): `utils._EMBEDDING_CLASSES` (`utils.py:30`), `critics.CRITICS` (`critics.py:505`), the optimizer/scheduler maps (`utils.py:955-958`). Derive it from them. (S effort)
9. **`__init__` shadows two submodules.**
   - `from .run import run` and `from .logger import logger` replace those submodules as package attributes. I confirmed with a mock package that `import pkg.run as m` then gives the function. `mock.patch('neural_mi.run.X')` would patch an attribute on the function.
   - Change: rename the modules (e.g. `_engine.py`, `_messages.py`).
   - `set_verbose` duplicates `set_verbosity` and nothing in the library uses it. Remove it.
   - `__init__` imports the private `_group_repeats_by_cell` into the package namespace (`__init__.py:18, 43`). Call it at the bottom of `logger.py` instead.
   - `validation` and `utils` are exported as public modules.
10. **`logger.py` is three modules in one.** (M effort)
    - Logging setup, warning grouping, and worker capture.
    - At import it installs its own StreamHandler (`logger.py:45-56`), where libraries normally add a NullHandler.
    - It requires torch and tqdm at import (`logger.py:67, 467`).
11. **The parallel paths are not unified.** (M effort)
    - `sweep`, `rigorous`, `permutation` and `pairwise` each hand-roll `worker_pool` + `CapturedTask` + `released` + tqdm. Only `quantities` uses `dispatch_tasks`.
    - `INTERNALS.md:191` says permutation trials go through `dispatch_tasks`; they do not (`permutation.py:275`).
    - `parallel.py` imports the private `utils._configure_multiprocessing`.
12. **`device.py` sits at top level but calls `analysis.task.run_training_task`** (`device.py:75`), a lower layer calling a higher one. Move it into `analysis/` or `training/`. (S effort)
13. **Minor.**
    - `NeuralMIError` is never raised directly. Most errors are plain `ValueError`/`TypeError`, and `InsufficientDataError` is raised only in `rigorous`.
    - `Training(small_sample)` is a data-check setting, but the other data-check settings live on `Preprocessing` (special-cased at `run.py:41`).
    - `Training` carries 15 curve-judging thresholds; consider a `Diagnostics` config.
    - The `config.py` docstring lists only some of the configs.

**3. Keep as is**

- The declarative `MODES` registry.
- `seeds.py`.
- The "None = unset, one schema default" principle (enforce it, don't abandon it).
- `as_config` accepting a dict.
- `user_stacklevel`.
- The `collect_repeats` grouping design.
- `_check_grid` refusing grids that change nothing.
- The specific, actionable error messages.