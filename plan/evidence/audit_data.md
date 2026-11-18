I didn't run any code: torch and tqdm aren't installed here, so the package won't import. Every finding comes from reading the code, plus plain-Python arithmetic for finding 2.

1. MAP

- data/plan.py: the real core. `RowPlan` (route = rows/unfold/grid, `rows`, `build`, `materialize`, `cut`, `replace`), `Stream`/`resolve_stream`, `Roles`/`PAIR`, `TermRows`, `GridShifter`, `RoleShifter`, `pack`, `count_rows`, `refuse_mixed_streams`, `rows_line_up`. Called by run.py, analysis/{task,conditional,interaction,rigorous,permutation,sweep,dimensionality}.
- data/handler.py: the public `create_dataset`/`create_single_dataset`, plus `WindowManager`, `one_grid`, `report_retention` and seven container classes (`_NamedStreams`, `AlignedStreams`, `_PairAccessors`, `StreamBundle`, `PairedTemporalDataset`, `AlignedStaticStreams`, `PairedDataset`). Called by plan, offsets.grid_rows, task.py (static cache path), trainer (type hint).
- data/temporal.py: the per-processor windowers (`ContinuousWindowDataset`, `SpikeWindowDataset`, `BinnedSpikeDataset`, `CategoricalWindowDataset`) and helpers (`relabel_categorical_data`, `slots_in_window`, `max_events_in_window`). Only built through `create_single_dataset`.
- data/static.py: `StaticDataset` (rows already built, or the (a, c) pair) and `BaseStaticDataset`, which has one subclass.
- data/shift_windowing.py: the unfold route (`WindowShifter`, `BundleShifter`, `make_categorical_encoder`, `stream_period`, `seconds_to_samples`, `resolve_step_size`, `safe_n_windows`) and the stride-phase shifter (`RowPhaseShifter`, `try_build_row_phase_dataset`).
- data/rows.py: the row-geometry record and row identity (`RowGeometry`, `record_rows`, `row_identity`, `same_rows`). Read by trainer, rigorous, permutation, sweep.
- data/views.py: `SubsetView`, used by trainer only.
- data/corruption.py: `corrupt()`, used by analysis/precision and the datasets.
- preprocessing.py: `Transform`s (ZScore, ZCA, PCA, Copula, ARWhitening), `prepare` = `apply_preprocessing` + `check_streams`. Called by run, quantities, modes; its transforms are reused by ksg, utils._whiten and embeddings.
- augmentations.py: `apply_augmentations`, applied per batch in Trainer._step.

How raw x/y become a trainer batch:
1. run._run_flat → `preprocessing.prepare` (transforms, then the data check) (run.py:1320)
2. `resolve_stream` per stream; Y inherits X's processor and settings via `defaults.stream_processing`
3. `RowPlan(streams, shift)` → `one_grid` → `resolve_step_size`
4. `RowPlan.route`:
   - unfold: `_prep_shift_side` → `BundleShifter` → `WindowShifter.windows_at(0)`
   - grid: `create_single_dataset` → `*WindowDataset` + `WindowManager` → `_attach` → `validate_window_coverage` fold → `report_retention` → `remove_invalid_windows`
5. If the rows don't shift, `materialize()` builds them once in run(); otherwise the plan travels raw to each task
6. task.run_training_task → `TermRows.dataset` → `RowPlan.build` → `pack` → `PairedDataset(StaticDataset(x), StaticDataset(y))`, with `_window_shifter`/`_row_identity` set as private attributes (offset quantities go through `try_build_row_phase_dataset`)
7. Trainer: `SubsetView(indices)` per split; each epoch `windows_at(shift)` overwrites `dataset.x_dataset.data`
8. `_step`: `x_dataset[batch_idx]` → device → `apply_augmentations` → model

2. FINDINGS (most valuable first)

1. **There are two grid builders, and `create_dataset` can give different rows than `run()`.**
   - Evidence: `AlignedStreams._initialize_windows`/`_build_windows` (handler.py:331-381) re-implement `RowPlan._rows_grid` (plan.py:256-299). The training path never builds a `PairedTemporalDataset` or `StreamBundle`; only `create_dataset` and `offsets.grid_rows` (offsets.py:102, `validate_windows=False`) do. `create_dataset` always takes the grid route (np.interp), even where `run()` would take unfold.
   - Change: make `create_dataset` a thin wrapper over `RowPlan`, and add a "keep all windows" flag to `RowPlan` for offsets. Then delete AlignedStreams, StreamBundle, PairedTemporalDataset, AlignedStaticStreams, _NamedStreams, _PairAccessors, the WindowManager observer machinery, `update_parameters` and `set_window_size` (no library caller).
   - Effort M–L. Risk M (tests/test_data_processors.py and tests/test_views.py use them).
2. **The two routes disagree on window width.**
   - Evidence: unfold uses `seconds_to_samples` = `round()` (shift_windowing.py:48, banker's rounding); grid uses `slots_in_window` = ceil with tolerance (temporal.py:69-86). For 0.5 s at 25 Hz: 12 vs 13 samples. For 0.25 at period 0.1: 2 vs 3. This contradicts `_SampledStream`'s claim at temporal.py:232-234 that every route cuts the same rows. The full_trajectory width warning uses a third rule, `np.ceil` (temporal.py:1073).
   - Change: one function for both routes.
   - Effort S. Risk: numbers change wherever window/period isn't a whole number.
3. **The datasets' corruption API is dead.**
   - Evidence: `apply_noise`/`apply_precision` on every dataset, `apply_*_by_name`, the pair accessors and SubsetView have no library caller; precision mode calls `corrupt()` on tensors directly (analysis/precision.py:174). `reset()` (×4) is never called. Each temporal dataset keeps a full `data_master` clone (temporal.py:499,796,944,1096) that only this API reads, and trainer.py:882-883 has to null it.
   - Change: remove all of it.
   - Effort S. Risk low.
4. **SubsetView is only used as an index holder.**
   - Evidence: the trainer reads only `.indices` (trainer.py:549-772). The time-based mode, weakref registry, `channels_x`/`channels_y` and `_on_dataset_updated` can't be reached from the library. The docstring's "time shift updates `times`" (views.py:18-19) describes code that no longer exists.
   - Change: replace with a sorted LongTensor and delete views.py.
   - Effort S. Risk low.
5. **Import cycles and layering.**
   - Verified cycles, both dodged with function-level imports: handler↔plan (handler.py:561,842 / plan.py:122,257,308), and handler→shift_windowing (top-level) against shift_windowing→handler/static (lazy, :221-222, only for `try_build_row_phase_dataset`).
   - These lazy imports have no back edge and could be top-level: plan→shift_windowing, plan→rows, shift_windowing→temporal.
   - Worse layering problem: static.py:6 and temporal.py:6 import `neural_mi.utils.get_device`, and utils imports `models.embeddings` and `critics` at top level, so data/ depends on models/. The `device` argument exists only "for reference" in every dataset constructor.
   - Change, as layers:
     - grid.py at the bottom: `resolve_step_size`, `seconds_to_samples`/`slots_in_window` merged, `one_grid`, `stream_period`, `refuse_mixed_streams`, `rows_line_up`
     - processors next
     - shifters next
     - plan.py: owns the dataset class and `try_build_row_phase_dataset`
     - public `create_dataset` on top
   - Also drop `device`.
   - Effort M. Risk low.
6. **temporal.py repeats itself.** It holds five separable pieces: base/clock, continuous, two spike classes, categorical.
   - The three categorical `_move_*` methods repeat one two-pointer loop (1098-1218). They also duplicate `make_categorical_encoder`.
   - Spike and binned spike duplicate the searchsorted plus ragged scatter (775-790 vs 921-938).
   - Continuous and categorical duplicate the clock and period setup (341-368 vs 1018-1033); only continuous checks `period_tolerance`. `plan._clock` (502) and `stream_period` are a third and fourth version.
   - `data_device` resolution is duplicated (temporal.py:147 / static.py:27).
   - Change: compute each window's [L, R) once, then use one encoder for both routes.
   - Effort M. Risk low–M.
7. **The 'spike' processor is really two processors.**
   - Possible bug: `BinnedSpikeDataset` doesn't sort spike times but relies on searchsorted (870, 921), unlike `SpikeWindowDataset` (640-646). Unsorted input would miscount silently.
   - With `bin_size` set, `n_seconds`, `max_spikes_per_window`, `exclude_bursty_neurons` and `no_spike_value` are silently ignored. Without it, `normalize_bins` is ignored.
   - The spike `sample_rate` in the schema (defaults.py:306) is never passed to a spike dataset (handler.py:626-637). Uncertain whether something else reads it.
   - `SCALING_OF_KIND` already distinguishes spike_counts from spike_times.
   - Change: split into two processors, each with its own schema.
   - Effort S–M.
8. **Settings resolution is duplicated.**
   - Processor settings are resolved twice (handler.py:605-609, plan.py:87-90). The rule that Y inherits X's settings exists twice (`handler._stream_specs_from_pair`, `RowPlan.of_pair`). Constructors also hard-code defaults (`min_coverage_fraction=0.2`, `encoding`) beside `PROCESSOR_DEFAULTS`.
   - Change: constructors take a resolved `Stream`.
   - Effort S.
9. **The trainer reaches into dataset internals.**
   - Evidence: trainer.py:880-883 writes `x_dataset.data`. `_window_shifter`, `_row_identity` and `n_windows_*` are attributes patched onto `PairedDataset` (plan.py:315-318, shift_windowing.py:230). `BaseStaticDataset` has one subclass, and its `_process` is an empty stub (static.py:47).
   - Change: one `RowsDataset(x, y, shifter, identity, retention)` replacing PairedDataset/StaticDataset/BaseStaticDataset.
   - Effort M.
10. **Leftovers.**
    - data/__init__.py docstring names `ContinuousDataset`/`SpikeDataset`, which don't exist.
    - handler.py:540-541 is an orphan comment.
    - `StreamBundle.__init__` is a pass-through (489).
    - `AlignedStaticStreams` logs "Created PairedDataset" (556).
    - `x_window_width`/`y_window_width` have no callers, and their 20-line docstrings are duplicated.
    - The `window_manager` constructor argument on all four temporal datasets is never passed.
    - `BundleShifter(pack=)` is never passed.
    - `CategoricalWindowDataset.reset` says "Does not undo time shifts".
    - Three names for two routes: grid/eager/WindowManager path vs unfold/reslice.
    - Effort S.
11. **preprocessing.py's boundaries.**
    - Its role (optional stream transforms plus a data check, before the plan) is principled. INTERNALS.md:50 nevertheless says "data/ contains the preprocessing".
    - ksg.py:66, utils.py:1064 and embeddings.py:1242 import it only for `ZScore`/`ZCA`/`Copula`/`constant_channels`, which are generic numeric transforms.
    - Change: split into transforms.py (dependency-free) and data/check.py (`check_streams` plus about 230 lines of warning prose).
    - `stream_kind` (preprocessing.py:324: series/rows/spike) and `defaults.stream_kind_of` (continuous/spike_counts/…) are two "kind" vocabularies for the same thing.
    - Effort S–M.
12. **utils.py holds data helpers.**
    - Move `float_tensor`, `with_window_axis`, `_as_tensor` and `_is_spike_input` to data/arrays.py.
    - `build_offset_arrays` (~145 lines) belongs in analysis/offsets.py.
    - `_shift_data` belongs in analysis/lag.py, its only caller.
    - `warn_if_*_split_leaks` read `RowGeometry` and belong in rows.py or training.
    - Effort S. This also removes data's dependency on utils.
13. **Row counts are re-derived outside the plan (cross-area, run).** `run._deferred_window_count` (run.py:1520-1551) re-derives counts outside the plan. The spike branch ignores `drop_empty_windows`. Whether its counts actually diverge from the plan's is uncertain.
14. **augmentations.py convention drift.**
    - Defaults are inline except `kernel_size`, which goes through the schema (158).
    - `random_flip_h` flips dim 2 ("height"), which is unconventional.
    - Effort S.

3. KEEP AS IS
- `corrupt()`: one function, empty-value aware.
- rows.py: small and clean, a single record of row geometry and identity.
- `RowPlan`, `Roles`, `TermRows` as the one central abstraction, with `cut`/`replace`/`materialize`.
- `BundleShifter` keeping a constant row count across shifts.
- The `Transform` interface in preprocessing.
- The specific error messages, such as the string-as-`y_data` check.