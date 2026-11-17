# Phases L and R: the moves and the registries

Phase L changes where code lives and nothing about what it does. At its end each module sits in
the layer that `plan/README.md` gives it, `utils.py` is gone, and most imports inside functions
are gone with the cycles that they avoided. Phase R gives every kind of name that a user can pass
one table, and validation, the builders and the pages read that table.

No step of these two phases changes a number. Each step updates the imports of the tests that it
touches in the same commit, and shrinks the allowlists of S-03.

## Phase L: moves and layers

The table gives the destination of every module that moves or splits. Phase L does the moves
marked L. The moves marked E, W or T happen in those phases, because the code is rewritten there.

| Today | At the end | Step |
|---|---|---|
| `utils.py` | `units.py`, `devices.py`, `data/arrays.py`, `data/offsets.py`, `data/rows.py`, `models/factory.py`, `training/optim.py`, `training/spectrum.py`, `analysis/lag.py`, `parallel.py` | L-01 to L-05 |
| `embeddings_io.py` | `models/io.py` | L-04 |
| `augmentations.py` | `training/augmentations.py` | L-06 |
| the figures in `results.py` | `visualize/figures.py` | L-07 |
| the checks in `run.py` | `validation.py`, `data/check.py` | L-08 |
| the data check in `preprocessing.py` | `data/check.py`, `data/streams.py` | L-09 |
| `logger.py` | `messages.py`, `repeats.py`, `parallel.py` | L-10 |
| `analysis/offsets.py` | `data/offsets.py` | L-05 |
| `device.py` | `analysis/device.py`, then `analysis/runner.py` | L-12, E-01 |
| `analysis/sweep.py` | `analysis/quantity.py`, `analysis/estimate.py`, `analysis/runner.py`, `analysis/grid.py` | L-11, E-01 to E-03 |
| `analysis/conditional.py`, `interaction.py`, `transfer.py` | `analysis/quantity.py`, `analysis/estimate.py`, `quantities.py` | E-02 to E-04 |
| `analysis/rigorous.py` | `analysis/ladder.py` | E-06, E-07 |
| `run.py` | `api.py`, `engine.py` | E-14 |
| `data/handler.py`, `data/static.py` | `data/create.py`, `data/dataset.py` | W-02 to W-04 |
| `data/temporal.py` | `data/processors/` | W-05 |
| `data/shift_windowing.py` | `data/shifters.py`, `data/grid.py` | W-01, W-09 |
| `data/views.py` | deleted | X-02 |

### L-01 One module for units

**Why.** The conversion between nats and bits is written in five places: `utils.py:510`,
`analysis/assemble.py:58`, `results.py:596`, `training/trainer.py:491` and
`generators/oracle.py:41` (again at line 508). The constants agree today. The rule for when to
convert is coded three times and can drift.

**Change.**
1. Add `neural_mi/units.py` with the constant `NATS_TO_BITS` and one function that returns the
   scale and the name for a unit.
2. Move `mi_report_units` (utils.py:479-538) there, and make the four other places import the
   constant.

**Tests.** Update the tests that import `mi_report_units` from `utils`.

**Done when.** `grep -rn "log(2" neural_mi/` shows only `units.py`.

### L-02 The tensor helpers and the device helper leave `utils.py`

**Why.** `float_tensor`, `with_window_axis`, `_as_tensor`, `_is_spike_input` and `_ensure_cpu`
(utils.py:43-128) are data helpers. A module that imports one of them from `utils` also imports
every model class (utils.py:19-25).

**Change.**
1. Move the five functions to `data/arrays.py`.
2. Move `get_device` (utils.py:61) to a new `neural_mi/devices.py`.
3. Update the importers: `run.py`, `quantities.py:55`, `data/plan.py:224`, `analysis/modes.py:558`,
   `analysis/sweep.py:26`, `analysis/pairwise.py`, `analysis/rigorous.py` and `analysis/task.py`.

**Done when.** No module of `data/` imports `neural_mi.utils`.

### L-03 The optimiser builder and the spectrum functions move to `training/`

**Why.** `build_optimizer_and_scheduler` (utils.py:961-1059) and the spectrum functions
`_whiten`, `cross_covariance_svd`, `compute_cross_covariance_spectrum`,
`compute_cross_covariance_rotation` and `compute_spectral_metrics` (utils.py:1060-1180) are
called only by the trainer and the task.

**Change.** Move the first, with `_OPTIMIZERS` and `_SCHEDULER_NAMES` (utils.py:955-958), to
`training/optim.py`. Move the others to `training/spectrum.py`.

**Done when.** The two new modules exist and `utils.py` no longer defines these names.

### L-04 The critic factory and the network files move to `models/`

**Why.** The functions that build a critic sit in `utils.py:468-960`, outside the package that
holds the classes they build. `run.py:202` imports three private names of that block.

**Change.**
1. Move to `models/factory.py`: `_accepts_kwarg`, `check_custom_critic`, `critic_recipe_takes`,
   `_scaling_modes`, `_scaled_input`, `encoders_scale`, `_custom_critic`, `build_critic`,
   `_library_critic`, `_EMBEDDING_CLASSES` (utils.py:30), `MODEL_SETTINGS` (utils.py:515),
   `ARCHITECTURE_KEYS` and the `CRITIC_*` constants.
2. Move `embeddings_io.py` to `models/io.py`. Keep `nmi.extract_embeddings` exported.
3. Remove the imports inside functions in `models/critics.py` that have no cycle to avoid.

**Tests.** `build_critic` is imported in 16 test files. Update the import line in each.

**Done when.** `utils.py` contains none of these names and `embeddings_io.py` is gone.

### L-05 Rows by offset move to `data/offsets.py`, and `utils.py` is deleted

**Why.** The builder of rows by offset is split between `utils.build_offset_arrays`
(utils.py:171-316) and `analysis/offsets.py`. It is a data concern that the named functions and
the row plan both need.

**Change.**
1. Create `data/offsets.py` from `build_offset_arrays`, `validate_stride`, `validate_window`
   (utils.py:129-170) and the whole of `analysis/offsets.py`.
2. Move `_shift_data` (utils.py:317-467) to `analysis/lag.py`, its only caller.
3. Move `warn_if_blocked_split_leaks` and `warn_if_random_split_leaks` (utils.py:1181-1231) to
   `data/rows.py`.
4. Move `_configure_multiprocessing` (utils.py:68) to `parallel.py`, its only caller.
5. Delete `utils.py`, and remove `utils` from the imports and from `__all__` in
   `neural_mi/__init__.py`.

**Tests.** `neural_mi.utils` is named in 15 test files and `build_offset_arrays` 18 times in one.
Update each. Keep `tests/test_utils.py` and change its imports.

**Pages.** `reference/USING.md:1287` and two docstrings name `neural_mi.utils.build_offset_arrays`.
Point them to `neural_mi.data.offsets.build_offset_arrays`.

**Done when.** `neural_mi/utils.py` does not exist and `grep -rn "neural_mi.utils\|nmi.utils"`
over the repository shows nothing.

### L-06 `augmentations.py` moves to `training/`

**Why.** Only the trainer applies augmentations, in `Trainer._step`.

**Change.** Move the file to `training/augmentations.py` and update the trainer and
`tests/test_augmentations.py`.

**Done when.** `neural_mi/augmentations.py` does not exist.

### L-07 The figures leave `results.py`

**Why.** `results.py` contains thirteen drawing functions beside the `Results` class, and
`analysis/modes.py:35-38` imports them by their private names.

**Change.**
1. Move `_draw_training`, `_draw_axis_curve`, `_draw_extrapolation`, `_bars`, `_draw_components`,
   `_draw_directions`, `_draw_precision`, `_draw_matrix`, `_draw_dimensionality`,
   `_one_repeat_each`, `_overlay_training`, `_overlay_extrapolations` and `_overlay_curves`
   (results.py:106-446) to `visualize/figures.py`.
2. Keep the summary functions (`_summary_*`, `_fit_lines`, `_unreliable_reasons`) in
   `results.py`, since they produce text.
3. Give `_draw_extrapolation` and `_overlay_extrapolations` one helper for the code that they
   repeat (results.py:168-173 and 412-415).
4. Remove the module-level import of pyplot from `results.py` (line 21).
5. Change the import in `modes.py`. It still points upward, and E-11 removes it.

**Tests.** Update `tests/test_results.py` and `tests/test_visualize.py` where they import a
drawing function.

**Done when.** `grep -n "^def _draw\|^def _overlay" neural_mi/results.py` shows nothing.

### L-08 The setting checks and the data reports leave `run.py`

**Why.** `run.py` contains about 400 lines of setting checks (49-466) and about 180 lines of
reports on the data (1520-1701). `analysis/modes.py:557` imports two of the reports back from
`run`, and `validation.py:363` and `modes.py:39` import each other.

**Change.**
1. Move to `validation.py`: `_stream_processors`, `_normalise_grid`, `_check_grid`,
   `_unread_model_settings`, `_rotation_settings_ignored`, `_default_critic`, `_unread_settings`,
   `_warn_ineffective_settings`, `_check_requests`, `_check_estimator_params` and
   `_check_custom_split`.
2. Move to `data/check.py`: `_deferred_window_count`, `_held_out_count`, `_report_rows`,
   `_warn_small_sample` and `_warn_if_shift_windows_dead`. Keep `_encoders_scale` with the
   caller and pass its value in, because it reads the model settings.
3. Move `permutation._spike_population_extent` to `data/`, since `run.py:1529` imports it from
   `permutation`.
4. Move each mode's own check (`check_lag_settings`, `check_precision_settings`,
   `check_dimensionality_settings`, `check_rigorous_settings`) from `validation.py` to the module
   of its procedure. `validation.py` can then import `MODES` at module level.

**Tests.** Three test files import a report from `neural_mi.run`
(`tests/test_preprocessing.py:300`, `tests/test_one_reaction.py:17`,
`tests/test_shared_encoder.py:71`). Update them.

**Done when.** `modes.py` has no import of `neural_mi.run`, and `validation.py` imports `MODES`
at module level.

### L-09 The data check leaves `preprocessing.py`

**Why.** `preprocessing.py` contains the transforms and also the data check, with about 230
lines of warning text. `estimators/ksg.py:66`, `models/embeddings.py:1242` and the spectrum
functions reuse the transforms.

**Change.**
1. Move `check_streams`, `check_settings` and `stream_kind` to `data/check.py`.
2. Move `prepare`, the function that applies the transforms and then runs the check, to
   `data/streams.py`.
3. Keep the transforms, `resolve`, `describe` and `constant_channels` in `preprocessing.py`. It
   stays a public module, because `reference/USING.md:943` names `neural_mi.preprocessing.PCA`.

**Tests.** Update the imports in `tests/test_preprocessing.py` and
`tests/test_data_check_routes.py`.

**Done when.** `preprocessing.py` imports nothing from `data/`.

### L-10 `logger.py` becomes `messages.py`, and its three jobs separate

**Why.** `from .logger import logger` in `neural_mi/__init__.py:17` makes the package attribute
`logger` hide the module of the same name, so `import neural_mi.logger as m` returns the
`Logger` object. `tests/test_run.py:336` works around this with `importlib`. The module also
does three jobs and needs torch and tqdm to be imported (logger.py:67, 467).

**Change.**
1. Rename the module to `messages.py`. It keeps the logger, `set_verbosity`, `call_verbosity`
   and `user_stacklevel`.
2. Move `collect_repeats`, `grouped_warnings` and `_group_repeats_by_cell` to `repeats.py`. Call
   `_group_repeats_by_cell()` at the end of that module, and remove the private import and the
   call from `neural_mi/__init__.py:18` and `:43`.
3. Move `CapturedTask` and `released` to `parallel.py`.
4. Replace `neural_mi.logger` by `neural_mi.messages` in every importer.

**Tests.** Remove the `importlib` workaround in `tests/test_run.py:336`. Update
`tests/test_repeated_messages.py`.

**Pages.** `docs/source/api_reference.rst:107-109` names functions of `neural_mi.logger`.

**Done when.** `import neural_mi.messages` returns a module and `neural_mi/logger.py` does not
exist.

### L-11 One module for grids

**Why.** The product of a grid is written four times (`sweep._product_dict` at sweep.py:29,
`assemble._product` at assemble.py:124, rigorous.py:832-833 and rigorous.py:775-776). The hashing
of a grid value is written twice (`assemble._hashable` at line 170 and `task._make_hashable` at
line 69).

**Change.** Create `analysis/grid.py` with one product and one hashing function, and move
`split_grid`, `config_lookup`, `split_preparation` (assemble.py:78-153) and `merge_grid_values`
(sweep.py:36-62) there.

**Done when.** `grep -rn "itertools.product" neural_mi/analysis/` shows only `grid.py`.

### L-12 `device.py` moves next to the code that it times

**Why.** `device.py` sits at the top level but calls `analysis.task.run_training_task`
(device.py:75), so a low module calls a higher one.

**Change.** Move the file to `analysis/device.py` and update `analysis/sweep.py:21`,
`analysis/rigorous.py` and `analysis/precision.py`. E-01 later folds it into the runner.

**Done when.** `neural_mi/device.py` does not exist.

### L-13 Remove the imports inside functions that only avoided a cycle

**Why.** After the moves above, many imports inside functions avoid nothing.

**Change.** For each entry of the second allowlist of S-03 that is tagged with a step of this
phase, move the import to the top of its module and delete the entry. Leave the entries that
phases E and W remove.

**Done when.** The allowlist contains only entries tagged E or W, and the optional packages.

## Phase R: registries

A registry is one table that maps each name a user can pass to what implements it. Today the
names of the encoders are listed in six places, so adding an encoder takes six edits. Each step
below leaves one table and makes every other place read it.

### R-01 Encoders

**Why.** The encoder names are written in `utils._EMBEDDING_CLASSES` (utils.py:30),
`validation.ALLOWED_VALUES` (validation.py:45), the chain in `decoders.build_decoder`
(models/decoders.py:563-594), `_4D_NATIVE` (analysis/task.py:242), a list at
`analysis/dimensionality.py:241`, the checks for a time axis (run.py:1266, analysis/sweep.py:200)
and comments in `config.py:86` and `defaults.py:129`.

**Change.**
1. Keep one table `ENCODERS` in `models/`, from name to class.
2. Give `BaseEmbedding` the class attributes that replace each special case: `image_input` (true
   for `cnn2d`, `mlp` and `pretrained_backbone`), `needs_time_axis` (true for `gru` and `lstm`)
   and `hidden_dim_list` (true for `mlp`, `cnn`, `cnn2d` and `tcn`). Establish first what the
   list at `dimensionality.py:241` asks, and add an attribute for it.
3. Replace the chain in `build_decoder` by a table `DECODERS` from encoder name to decoder
   class, in `models/decoders.py`.
4. Derive the allowed values in `validation.py` from `ENCODERS`.
5. Turn each `settings` dict into a frozenset, since every one maps each key to itself, and
   compute `MODEL_SETTINGS` as their union.

**Tests.** Add a test that every key of `DECODERS` is a key of `ENCODERS`.

**Pages.** `reference/INTERNALS.md:371` tells a contributor to edit `utils` and `validation` by
hand. Rewrite it for the table.

**Done when.** `grep -rn "'lstm'" neural_mi/ --include=*.py` shows the table and the class only.

### R-02 Critics, optimisers, schedulers and activations

**Why.** `validation.py:44` repeats `critics.CRITICS` (models/critics.py:505), and
`validation.py:50-51` repeats the optimiser and scheduler maps (utils.py:955-958). The
activations are already derived from `ACTIVATIONS`.

**Change.** Derive the three lists in `validation.py` from the tables.

**Done when.** `ALLOWED_VALUES` contains no literal list for these three kinds.

### R-03 Estimators

**Why.** The estimators are spread over three tables (`ESTIMATORS`, `NONPARAMETRIC`,
`NONPARAMETRIC_CONDITIONAL`), an identity check in the trainer
(`self.estimator_fn is infonce_lower_bound`, training/trainer.py:944 and 1109) and nine branches
on `'ksg'` or on `NONPARAMETRIC` (run.py:309, 428, 827 and 1588, analysis/task.py:291,
analysis/rigorous.py:665, analysis/sweep.py:317, device.py:54, validation.py:49).

**Change.**
1. Declare one record for each estimator in `estimators/__init__.py`.

   ```python
   @dataclass(frozen=True)
   class EstimatorSpec:
       name: str
       fn: Callable
       trains_network: bool
       conditional_fn: Optional[Callable] = None
       log_n_ceiling: bool = False
   ```
2. Replace each branch by the property that it asks about. The trainer's identity check reads
   `log_n_ceiling`. The branches on KSG read `trains_network` or `conditional_fn`.

**Tests.** Update `tests/test_estimators.py` and `tests/test_ksg.py` where they read the three
tables.

**Done when.** `grep -rn "NONPARAMETRIC\|is infonce_lower_bound" neural_mi/` shows nothing.

### R-04 Processors

**Why.** What a processor is, is written in `PROCESSOR_DEFAULTS` and `PROCESSOR_PARAMS_SCHEMA`
(defaults.py), the chain in `create_single_dataset` (data/handler.py:605-637), `REGULAR`
(data/plan.py:43), and `stream_kind_of` with `SCALING_OF_KIND` (defaults.py).

**Change.** Declare one record for each processor, with its dataset class, its settings with
their defaults, whether its clock is regular, and the kind of rows that it gives. Make the five
places read the record.

**Done when.** `create_single_dataset` contains no chain over processor names.

### R-05 The small sets of names

**Why.** `ALLOWED_VALUES` (validation.py:43-55) also repeats the split modes, the values of
`split_method`, the corruption methods, the epoch rules, the names of the transforms, the
permutation shuffles, the units, the norm layers and the pooling of the CNN.

**Change.** Keep each set as a constant next to its implementation and import it in
`validation.py`.

**Done when.** `ALLOWED_VALUES` is built only from imported constants.

### R-06 A test that the registries and the validator agree

**Why.** The registries only help if nothing lists the names again.

**Change.** Add a test that every name of a registry passes validation, and that every value the
validator allows comes from a registry. Establish first how `tests/test_docs_drift.py` reads the
allowed values for `reference/PARAMETERS.md`, and make it read the registries.

**Done when.** The new test passes, and adding a name to a registry needs no other edit for
validation to accept it.
