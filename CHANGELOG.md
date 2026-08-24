# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed: `return_embeddings=True` could read live-shifted data instead of the frozen eval snapshot

When `shift_windows`/`shift_time` is active, `dataset.x_data`/`.y_data`
reflect whichever shift was last applied during training. Embedding
extraction (`task.py`) read them directly, diverging from the canonical,
frozen pre-shift view that `best_model_state` was actually scored against
(everything else — `test_mi`/`train_mi`, spectral metrics, decoder loss —
already reads through that frozen snapshot). `Trainer.train()` now exposes
the snapshot in its results dict when shifting was active; embedding
extraction reads through it when present.

### Temporal window shifting: renamed, on by default, ramp removed

`epoch_window_shift`/`random_time_shifting` are renamed to `shift_windows`/
`shift_time` (matching `Training`'s other verb+noun boolean flags, e.g.
`use_amp`/`track_embeddings`) and both now **default to `True`** wherever
they apply, since extensive testing showed genuinely better-generalizing
models (better held-out test MI, especially in the small-sample regime,
with the un-shifted baseline hitting a hard ceiling no amount of extra
training crosses) with no case where it hurts. If `n_epochs`/`patience`
need adjusting for a given dataset, the existing "under-trained lower
bound" warning already flags it; it isn't auto-scaled.

- **Breaking rename**: `epoch_window_shift` → `shift_windows`,
  `random_time_shifting` → `shift_time`, throughout the public API,
  internal helpers, and tests. `EpochWindowShifter`/
  `PairedEpochWindowShifter` (`neural_mi/data/shift_windowing.py`) are
  renamed to `WindowShifter`/`PairedWindowShifter`.
- **`shift_time`'s `epochs_to_max_shift` warm-up ramp is removed
  entirely**, not deprecated — it measured no benefit across several ramp
  lengths and added a parameter with nothing left for it to control once
  gone. Both mechanisms now shift at full magnitude from epoch 0, matching
  `shift_windows`'s existing (unramped) behavior.
- **`shift_windows`'s shift range no longer depends on `step_size`**: it's
  drawn from `[0, window_size)` rather than `[0, step_size)`. Byte-identical
  to before whenever `step_size == window_size` (the common case, since
  `step_size` usually isn't set); only changes behavior (more conservative,
  fewer windows) when `step_size` is explicitly set smaller than
  `window_size`.
- Flipping the default required a companion fix in `run.py`: the
  "has no effect for this configuration" warnings now only fire when the
  user explicitly set the flag (reusing the existing `_pre_default_keys`
  explicit-vs-defaulted tracking), so a schema-defaulted `True` that
  doesn't apply to a given pair stays silent instead of warning.

### Fixed: `random_time_shifting`/`epoch_window_shift` could leak into the reported estimate

Both mechanisms are meant to affect training dynamics only. They didn't:
the final `test_mi`/`train_mi` (and per-epoch history used for early
stopping and best-epoch selection, decoder reconstruction loss, spectral
metrics, per-epoch embedding tracking) was evaluated against
`dataset.x_dataset`/`.y_dataset` as they stood *after* the epoch loop
ended — i.e. whichever shift happened to be applied at the end of the last
epoch trained, which is essentially never the same shift that was in
effect when the best-epoch checkpoint was actually selected. The reported
number therefore depended partly on incidental timing, not just model
quality. Live comparison for one case: -0.0554 nats (stale-shift eval,
what the old code produced) vs. -0.0645 nats (the correct, canonical-view
number) — a real, non-rounding difference.

Fixed in `neural_mi/training/trainer.py`: a frozen snapshot of the data
(taken before any shift) plus the original, non-drifting test/train-eval
index arrays are now used for every evaluation/diagnostic read; training
batches still read the live, currently-shifted data. For
`random_time_shifting` this closes a second, related gap:
`SubsetView.indices` for a real temporal dataset re-derives from stored
time ranges on every rebuild (the ordinary, now-bounded edge effect from
the SubsetView fix below), so even with frozen *content* the *index set*
itself could still drift — evaluation now uses the original index arrays
directly, sidestepping `SubsetView`'s live-tracking machinery entirely for
this purpose. Verified via a live before/after re-evaluation matching the
trained model against an independently-reconstructed canonical view, for
both mechanisms, down to floating-point equality.

### Temporal window shifting: plain-sweep reachability, cross-unit warning, honest default

Small follow-ups to the shift-coverage work below.

- **`epoch_window_shift`/`random_time_shifting` now reachable at plain
  (non-processor-swept) `mode='sweep'`**, not just `mode='estimate'` — any
  mode dispatching independent training runs from the same raw data
  qualifies equally. `mode='pairwise'` is deliberately excluded: despite
  also being "independent sub-runs," its per-channel dispatch needs an
  already-windowed array to slice channels from, a real restructuring of
  `pairwise.py` rather than a gating change — flagged, not attempted.
- **New warning in `create_dataset`**: pairing `spike` with `continuous`/
  `categorical` when the regular-grid side lacks `sample_rate` now warns
  that window alignment may be meaningless, independent of whether any
  shift is requested — `PairedTemporalDataset` combines both sides'
  temporal extents via plain numeric `min`/`max`, which silently produces
  wrong results if the two sides aren't already in the same time unit.
- **Fixed a misleading default**: `Trainer.train()`'s own signature said
  `random_time_shifting: bool = True`, while the real, user-facing default
  (enforced by `task.py`, always passed explicitly) was already `False`.
  No caller relied on the naked default taking effect on temporal data
  (verified) — this is a same-behavior, honesty-only fix.

### Temporal window shifting: categorical/mixed-pair/spike coverage, and a SubsetView correctness fix

Extends the `epoch_window_shift`/`random_time_shifting` work below to more
`processor_type` pairs, and fixes a real bug found while doing so. See
`NEURALMI_REFERENCE.md`'s "Temporal Window Shifting" section for the full
per-pair reachability table.

- **`epoch_window_shift` now covers the whole "regular grid" family**:
  `categorical`+`categorical` and `continuous`+`categorical` (either order),
  not just `continuous`+`continuous`. Categorical data is re-encoded
  (`majority_vote`/`probability`/`full_trajectory`) via a new vectorized
  step in `neural_mi/data/shift_windowing.py` after each reslice, matching
  `CategoricalWindowDataset`'s three encodings exactly (verified
  byte-identical at shift=0). Also fixes a latent bug: `window_size`/
  `step_size` in `processor_params` are in seconds whenever `sample_rate`
  is set, but were previously passed straight to `torch.unfold` (which
  requires an integer sample count) — would have crashed with a confusing
  `TypeError`. Now converted per-side via each side's own `sample_rate`,
  which also makes X and Y correctly handle *different* sample rates.
- **`random_time_shifting` is now reachable at `mode='estimate'`** for
  `spike`+`spike` pairs (previously only `mode='lag'`/processor-swept
  `mode='sweep'`), and for mixed `spike`+continuous/categorical pairs when
  the non-spike side has `sample_rate` set (required so a shift value means
  the same real time on both sides; without it, the pair is left unshifted
  with an explanatory warning rather than silently misaligned).
- **`SpikeWindowDataset`/`BinnedSpikeDataset` windowing vectorized**: the
  Python-level `for w in range(n_windows)` two-pointer loop is replaced
  with two `np.searchsorted` calls plus a vectorized ragged-to-flat scatter
  — verified byte-identical to the original loop across overlapping,
  non-overlapping, sparse, and empty-neuron cases.
- **Fixed: `SubsetView` eval-subset bug** (`neural_mi/training/trainer.py`).
  The training-evaluation subsample (`train_eval_idx`, used for the
  reported train MI and `eval_train`'s per-epoch tracking) was built via
  `np.random.choice` — a scattered sample that `SubsetView` can only
  represent as thousands of degenerate zero-width time ranges, which
  collide or drop en masse on the very next window rebuild (observed: up
  to 100% loss on a single shift). Root-cause fixed, not gated around: the
  eval subsample is now built from proportional *contiguous* sub-chunks of
  the train split's own contiguous segments (`Trainer._select_train_eval_indices`),
  the same representation `train_view`/`test_view` already handle safely.
  Also closes a separate, sharper latent risk where the `eval_train`
  per-epoch subset bypassed `SubsetView` entirely and could index past the
  end of a rebuilt dataset. This was a pre-existing bug affecting *any*
  processor type through the already-shipped `mode='lag'`/processor-swept
  `mode='sweep'` paths, not something this pass introduced — fixing it
  benefits existing callers too.

### `Training(epoch_window_shift=True)`: cheap per-epoch window-tiling shift for regularly-sampled data

New, narrower alternative to `random_time_shifting` for the common case:
regularly-sampled data windowed via a fixed `window_size`/`step_size`.
`random_time_shifting` (gated on `PairedTemporalDataset`) rebuilds the
entire windowed array via `np.interp` on every shift, at a cost independent
of shift size (confirmed: shift=0 costs the same as shift=10), and has a
separate bug where the training-evaluation subset loses over half its
windows on the first rebuild regardless of shift magnitude. `epoch_window_shift`
sidesteps both: for a regular grid, shifting which sample starts the tiling
is a plain re-slice (`neural_mi/data/shift_windowing.py`'s
`EpochWindowShifter`, `torch.Tensor.unfold`-based, no interpolation, no
time-vector, no `SubsetView`), with the window count held fixed across
every shift so train/test/eval splits stay valid throughout. No ramp-up
schedule (shifts at full magnitude from epoch 0) — `epochs_to_max_shift`-style
ramping showed no measurable accuracy benefit in direct testing, across
ramp settings from 1 to 40 epochs. Benchmarked against
`random_time_shifting` (5-seed shared-latent Gaussian oracle,
`mode='estimate'`): comparable or faster wall-clock (39.1s ± 1.2s vs. the
old mechanism's 50.4s ± 16.7s baseline — also far more consistent, no
interpolation-cost variance), a real train/test-gap reduction (+0.196 ->
+0.138), and zero eval-subset warnings. Wired up for `mode='estimate'` with
a `continuous` processor only for this pass; extending reach to more modes
is a natural follow-up, not bundled in. New
`_warn_if_epoch_window_shift_dead` (`run.py`) mirrors the existing
`random_time_shifting` reachability warning.

### `DualBranchEmbedding` and `Conditional(align='dual_branch')`: MI rate, instantaneous exchange, directed information rate

The three remaining quantities in the taxonomy (`THEORY.md` §11) all need
$A$ and $C$ at genuinely different window lengths, beyond
`mode='conditional''`'s existing small trim tolerance. New
`DualBranchEmbedding` (`neural_mi/models/embeddings.py`), used via
`custom_embedding_cls=DualBranchEmbedding`: two independent sub-embedding
networks, one per input length, fused by a small MLP, replacing what would
otherwise require zero-padding the shorter array. New
`Conditional(align='dual_branch')` opt-in (`neural_mi/config.py`,
`neural_mi/analysis/conditional.py`) builds the "X-role" data as a tuple
`(a_data, c_data)` instead of concatenating; `align=None` (the default)
keeps today's concatenation-and-trim behavior byte-for-byte unchanged.

Threading a tuple through the pipeline touched every layer between data and
training: `StaticDataset`, `PairedDataset._align_datasets`, `ParameterSweep`,
`analysis/task.py`'s dim computation, `Trainer`'s batch-slicing (new
`_batch_size_of`/`_to_device`/`_index_batch` helpers), and the critic layer's
`_compute_embeddings_chunked`/`HybridCritic.forward` (new
`_batch_size_of`/`_device_of`/`_slice_batch` helpers), each a small,
mechanical `isinstance(x, tuple)` branch. `rigorous.py` itself needed no
changes: Z rides along via the existing `extra_data` mechanism, and the
tuple is assembled only at the `_cmi_rigorous_scalar` boundary. `_ensure_cpu`
(`neural_mi/utils.py`) now recurses into tuples, closing a real
`n_workers > 1` spawn-boundary risk (a GPU/MPS-resident tuple element would
otherwise silently never reach CPU before pickling).

New `neural_mi.mi_rate`, `neural_mi.instantaneous_exchange`,
`neural_mi.directed_information_rate` convenience functions
(`neural_mi/quantities.py`), each building the right $A$/$B$/$C$ arrays and
requiring a `DualBranchEmbedding`-configured `model=` (raises a clear
`ValueError` upfront otherwise). Directed information rate is estimated
directly rather than via its exact $\text{TE} + $ instantaneous-exchange
decomposition, to avoid compounding transfer entropy's small-residual
fragility into an otherwise well-behaved quantity; the identity is a
test-suite cross-check only. `use_variational=True` and `mode='sweep'`'s
`max_samples_per_task` are explicitly unsupported with `DualBranchEmbedding`,
raising a clear error rather than a silent wrong result.

### New `mode='interaction'`: interaction information

Computes `II = I(X,W;Y) - I(X;Y) - I(W;Y)` via `Interaction(w_data=...)`, new
`neural_mi/analysis/interaction.py`. The one quantity in the taxonomy
(`THEORY.md` §12) that isn't a single conditional-MI call, three separate MI
estimates combined by a formula, reusing `_joint_marginal_difference` once
plus a new `_single_mi_mean` helper for the standalone `I(W;Y)` term (three
sweeps total, not four). New `Interaction` config dataclass
(`neural_mi/config.py`, mirrors `Conditional`'s `z_*` split), new
`MODE_KWARGS_SCHEMA['interaction']` entry, new `Results.plot()`/`summary()`
branches (a 4-bar chart of the three components plus II, correctly handling
II's sign since it's the one quantity here that can legitimately be
negative). Works with `rigorous=True` and `permutation_test=True`, same
mechanics as `mode='conditional'`. New
`neural_mi.interaction_information(...)` convenience function.

### `mode='transfer'`: conditional transfer entropy via `Transfer(w_data=...)`

`Transfer` (`neural_mi/config.py`) gains `w_data`/`w_time`/`w_processor_type`/
`w_processor_params`, mirroring `Conditional`'s existing `z_*` split. When
`w_data` is provided, `run_transfer_entropy` (`neural_mi/analysis/transfer.py`)
folds a third signal's history ($W_{past}$, built the same way as
$X_{past}$/$Y_{past}$) into both the joint and marginal conditioning arrays,
computing $\text{TE}_{X\to Y}(W) = I(X_{past}; Y_0 \mid Y_{past}, W_{past})$
instead of plain TE, see `THEORY.md` §11. `w_data=None` (the default) is
byte-for-byte identical to before this parameter existed. Works with
`rigorous=True`: `_te_rigorous_scalar` now accepts and forwards `w_data`,
chunked per-gamma via the same `extra_data` mechanism `z_data` already uses
for `mode='conditional'`'s rigorous path. New
`neural_mi.conditional_transfer_entropy(...)` convenience function.

### `neural_mi/quantities.py`: named convenience functions for temporal information quantities

Five unconditioned $I(A;B)$ quantities on offset slices of one or two raw
time series (`active_information_storage`, `excess_entropy`,
`instantaneous_mi`, `cross_predictive_information`, `block_mi`), see
`THEORY.md` §11 for definitions and `NEURALMI_REFERENCE.md` §13 for the API.
Each is a thin wrapper around `mode='estimate'`, no new estimation mechanism;
discoverability lives at this convenience-function layer rather than growing
the `mode=` enum, since `mode=` tracks estimation mechanism, not scientific
question. New `neural_mi/analysis/offsets.py` generalizes the sliding-window
construction already used internally by `mode='transfer'`
(`build_past_future`, `build_cross_offset`) for the single- and
unconditioned-two-signal cases. Each function's construction parameter (`k`,
`past_k`, `window_size`) accepts an iterable to dispatch a parallel sweep
(new `neural_mi/parallel.py`'s `dispatch_tasks`, the same `spawn`-pool idiom
already used independently in `rigorous.py`/`dimensionality.py`/`pairwise.py`/
`sweep.py`, factored into one shared helper) and returns a `DataFrame`.

### `mode='dimensionality'`: cross-run-stable directions of shared structure

`mode='dimensionality'` (`neural_mi/analysis/dimensionality.py`, `neural_mi/config.py`,
`neural_mi/defaults.py`, `neural_mi/utils.py`, `neural_mi/visualize/plot.py`,
`neural_mi/results.py`) does not report a scalar dimensionality count: a nonlinear
encoder given more embedding capacity than the number of genuinely shared latent
factors can construct combinations of them that are spectrally indistinguishable from
genuine factors, so no measure of a single trained spectrum can be trusted as an exact
count. See `THEORY.md` §6 for the full reasoning. The mode instead reports:

- **A no-training regime diagnostic** (`compute_regime_diagnostic` in `utils.py`):
  whether each view's raw channel correlation structure looks separable-like or
  entangled-like, attached as `result.details['regime_x']`/`['regime_y']`.
- **Cross-run-stable directions**: a modest-capacity (`embedding_dim=8` default) Hybrid
  Critic embedding is trained `n_splits` independent times (default 3, min 2); each
  rank of the resulting cross-covariance rotation is checked for reproducibility across
  every pair of runs on held-out data (`stability_threshold`, default `0.7`) and for
  strength above a noise floor relative to the top rank (`min_strength_fraction`,
  default `0.05`, independent of and catching what the correlation check alone misses).
  Adjacent ranks too close in strength to individually order (`degeneracy_ratio_threshold`,
  default `1.3`) are reported as a group. Output: `result.details['stable_directions']`,
  `['stable_but_degenerate_groups']`, `['n_stable_total']`, `['stability_per_rank']`.
- **Convergence gating**: `result.details['converged']` is `True` only if every
  independent run actually converged; a `UserWarning` names any that didn't.
- **`pr_eig`/`pr_singular`** remain in `result.dataframe`/`result.details['raw_results']`
  as a labeled secondary diagnostic, not the mode's answer.
- **A lightweight ceiling-proximity warning** (`ceiling_mi_fraction`, default `0.85`,
  informational only) when the underlying MI estimate nears its InfoNCE evaluation
  ceiling (`log(eval_size)`); no automatic remediation is applied.
- **`shared_encoder`** defaults to `True` only for intrinsic mode (`y_data=None`,
  splitting one dataset in half) and `False` for interaction mode (two views) —
  previously defaulted `True` unconditionally regardless of sub-mode.
- **`Dimensionality` config**: `split_method`, `n_splits`, `lag`, `channel_indices_x`,
  `stability_threshold`, `degeneracy_ratio_threshold`, `min_strength_fraction`,
  `ceiling_mi_fraction`.
- **Plotting**: `plot_dimensionality_curve(details, ax, show, **kwargs)` — a per-rank
  bar chart (stable / stable-but-degenerate / not-stable), replacing the previous
  sweep-curve-shaped plot. Takes `result.details`, not `result.dataframe`. Not supported
  by `Results.compare()` (overlaying several isn't well-defined for this chart shape).

### Smaller fixes from the same broad-audit pass

- `validation.py`'s `DataValidator._validate_type` crashed with an unrelated `AttributeError:
  'list' object has no attribute 'dtype'` for `x_data`/`y_data` passed as a plain Python list
  with `processor_type='continuous'`/`'categorical'` — lists have no `.dtype`, and `run()`'s own
  docstring documents `list` as an accepted input type. Fixed by converting the list to an array
  once for the numeric-dtype check (mirroring what `create_dataset` does downstream anyway), with
  a safe fallback if the list is genuinely malformed/ragged.
- `data/handler.py`'s `create_dataset`: when `processor_type_x=None` (X pre-processed) is paired
  with a windowed `processor_type_y`, `window_size` was always read from (nonexistent)
  `processor_params_x`, silently resolving to `None` and raising a misleading
  `"window_size must be provided"` even when the user correctly supplied it in
  `processor_params_y`. Fixed to fall back to Y's `window_size` when X has none. Fixing that
  surfaced a separate, more fundamental issue it had been masking: pairing a genuinely
  pre-processed/static X with any windowed Y can never work (`StaticDataset` has no temporal
  extent for `PairedTemporalDataset` to align windows against), which previously surfaced as an
  opaque `AttributeError: 'StaticDataset' object has no attribute 'get_temporal_extent'` deep in
  `PairedTemporalDataset._initialize_windows`. Added an explicit, actionable `ValueError` for this
  combination instead. (The common case — `processor_type_y` left unset so it inherits
  `processor_type_x` — is unaffected either way.)
- `Results.plot()` for `mode='pairwise'`: the channel-count-aware default figure sizing
  (`visualize.plot`'s formula, ~0.65in/channel) was dead code — the generic top-level axes
  creation shared by most other modes ran first, consuming `figsize` and setting `ax` before the
  pairwise-specific branch ever checked either, so every pairwise heatmap used the generic
  `(10, 6)` figure regardless of matrix size. Fixed by adding `'pairwise'` to the set of modes
  that build their own figure (previously only `'dimensionality'`).
- `analysis/precision.py`: corrected a comment claiming `dataset_device` defaults to `'auto'`
  (co-located with the compute device) for `mode='precision'` — `ParameterValidator.apply_defaults()`
  always pre-fills it with the schema's global default (`'cpu'`) before this code runs, so that
  fallback was unreachable through the public `run()` API. Performance-only, no correctness impact;
  comment now describes the actual behavior and how to opt into the co-located mode explicitly.

### `random_seed` did not actually reproduce results, even with `n_workers=1`

Found during a follow-up broad-audit pass — a significant bug for a library whose docs promise
"full reproducibility only with n_workers=1". `ParameterSweep._prepare_tasks` (`analysis/sweep.py`)
and `AnalysisWorkflow._prepare_tasks` (`analysis/rigorous.py`, plain `mode='rigorous'`) built each
task's `run_id` with a fresh `str(uuid.uuid4())` prefix on *every* call. `analysis/task.py`'s
per-task seeding then hashed that `run_id` string to derive the actual training seed — so the
effective seed differed on every single `run()` invocation regardless of an explicit
`random_seed`, even single-process (`n_workers=1`). Verified: two back-to-back
`nmi.run(..., seed=42, n_workers=1)` calls on identical data produced different `mi_estimate`
values. Affects every mode that builds tasks via `ParameterSweep`/`AnalysisWorkflow` (`estimate`,
`sweep`, `dimensionality`, `rigorous`, `conditional`, `transfer`, `pairwise`); `mode='lag'` was
unaffected, since it already builds fully deterministic ids itself.

Fixed by seeding from a separate, purely deterministic `_seed_key` (built from each task's
combination/gamma/subset index, stripped before the task's result dict is returned) instead of
the display-only `run_id`. Different tasks within one sweep/rigorous run still get distinct
seeds — only the *same* task across separate `run()` calls now reproduces identically.
Regression tests in `tests/test_reproducibility.py` (confirmed to fail against the pre-fix code).

### Rigorous conditional/transfer: nats-to-bits unit conversion bug

Found during a follow-up broad-audit pass. `mode='conditional'`/`mode='transfer'` with
`rigorous=True` had two unit-conversion bugs stacked in the same few lines of `run.py`:

1. `_convert_mi_units`'s dict branch only converted a fixed list of scalar keys
   (`_MI_SCALAR_KEYS`) that did **not** include `mi_corrected`/`mi_error`/`mi_error_pred`/`slope`
   — so `result.mi_estimate` and `result.details['mi_error']` (etc.) stayed in **nats** even with
   the default `output_units='bits'`. (These same keys were already converted correctly in the
   DataFrame and list-of-dicts branches, and in plain `mode='rigorous'` via a different code path
   — this only affected the flat scalar-rigorous dict returned by
   `run_rigorous_scalar_analysis`.)
2. Separately, `raw_results_df` was converted once *inside* that same `_convert_mi_units` call
   (the dict branch recurses into it), then popped and converted a **second** time — silently
   inflating `result.dataframe['train_mi']` by an extra factor of `NATS_TO_BITS` (≈44% too high)
   instead of the correct single conversion.

Fixed by adding the four missing keys to `_MI_SCALAR_KEYS` and removing the redundant second
`_convert_mi_units` call at both call sites (`mode='conditional'` and `mode='transfer'`).
Verified via a mocked-training end-to-end test through `nmi.run()` confirming an exact
single-factor nats→bits ratio; regression tests added in
`tests/test_conditional_transfer_rigorous.py::TestRigorousConditionalTransferUnitConversion`
(confirmed to fail against the pre-fix code before landing the fix).

### Categorical processor no longer requires pre-cast integer data

`processor_type='categorical'` raised `TypeError: ... must be integer type` for any non-integer
numeric input (e.g. `float64` category labels), even though the underlying
`CategoricalWindowDataset` already auto-relabels non-integer data to consecutive integer category
codes via `np.unique` — the block was a separate, earlier `DataValidator` check
(`validation.py`) that pre-empted the working code path before it ever ran. (`z_data` for
`mode='conditional'` was unaffected either way — it bypasses `DataValidator` entirely.) Removed
the redundant check; `CategoricalWindowDataset` now also logs a warning when it performs the
relabeling, so silently-remapped category codes aren't a surprise. Non-numeric input (e.g.
strings) is still rejected, as before.

### `results.plot()` support for multi-parameter sweeps

`mode='sweep'`/`mode='lag'` with 2+ swept parameters (excluding `run_id`) previously plotted MI
against only the *first* swept parameter, silently discarding the rest — the aggregated
dataframe had a column per parameter, but `result.plot()` always resolved a single
`sweep_var` and fed it to a 1-D line plot, producing a connected line with multiple y-values
stacked at the same x position and no indication a second parameter existed.

`Results.plot()` now auto-selects a plot kind from how many parameters were actually swept
(`result.params['sweep_group_vars']`, newly threaded through from `run.py`'s sweep/lag
aggregation): 1 parameter keeps the existing line plot unchanged; 2 parameters default to a new
heatmap (`visualize.plot.plot_sweep_heatmap`, one param per axis, MI as colour); 3+ default to a
new grouped bar chart (`visualize.plot.plot_sweep_bar`, one bar per parameter combination). All
three are selectable explicitly via `result.plot(kind='line'|'heatmap'|'bar')`; `kind='heatmap'`
raises a clear error if the result didn't sweep exactly 2 parameters. `Results.compare()` now
raises a clear error for multi-parameter sweep results (overlaying is only well-defined for a
shared 1-D x-axis) instead of silently repeating the same first-parameter-only behavior.

### Parallelization fixes: rigorous conditional/transfer and pairwise modes

Two real bugs where `n_workers > 1` silently had no effect, both found while auditing
"does every mode with multiple independent runs actually parallelize them":

- `mode='conditional'`/`mode='transfer'` with `rigorous=True` (bias-corrected CMI/TE with a
  confidence interval) ran its ~55-task gamma-subset loop as a plain sequential Python `for`
  loop in `run_rigorous_scalar_analysis` (`analysis/rigorous.py`) — it didn't even accept an
  `n_workers` argument, unlike plain `mode='rigorous'`, which already dispatched the same
  55 tasks to a multiprocessing pool. Root cause: the per-task callables were local closures
  defined inside `run.py`, which can't be pickled for a `multiprocessing` 'spawn' pool. Fixed
  by promoting them to top-level, picklable functions (`_cmi_rigorous_scalar` in
  `analysis/conditional.py`, `_te_rigorous_scalar` in `analysis/transfer.py`) and giving
  `run_rigorous_scalar_analysis` a real `n_workers` parameter that dispatches the gamma-chunk
  tasks to a `Pool`, exactly like `AnalysisWorkflow.run()` already does for plain rigorous mode.
- `mode='pairwise'` looped over channel pairs sequentially in Python; `n_workers` only ever
  affected a *single pair's* internal sweep (normally 1 task with no `sweep_grid`), so the
  whole MI matrix computed serially regardless of `n_workers` — even though
  `NEURALMI_REFERENCE.md`'s own example already used `n_workers=8`. Fixed in
  `analysis/pairwise.py`: channel pairs are now dispatched to a `Pool(n_workers)` (one pair per
  worker) when there's more than one pair, with each pair's own inner sweep forced to
  `n_workers=1` to avoid nested pools — the same outer-loop-gets-workers convention already
  used for `dimensionality` mode's channel-split parallelization.

Also corrected a stale doc claim in `NEURALMI_REFERENCE.md` that sweeps parallelize via
`concurrent.futures.ProcessPoolExecutor` — the actual (and unchanged) mechanism is a
`torch.multiprocessing` 'spawn'-context `Pool`.

### Pre-submission cleanup pass

A full correctness, documentation, and tutorial-accuracy audit ahead of submission: every
source module reviewed against its own tests, every tutorial re-executed end-to-end, and all
front-facing docs (README, `NEURALMI_REFERENCE.md`, the Sphinx tutorials index) reconciled
against the live API. Full suite green throughout (638 passed / 1 skipped).

**Real bugs found and fixed**, all pre-existing and surfaced by actually re-running the
tutorials rather than trusting stale cached notebook outputs:
- `analyze_mi_heatmap` (`neural_mi/visualize/plot.py`) crashed when the significant-MI
  contour list was non-empty but contained only degenerate (empty or single-point) segments.
- `analysis/precision.py`'s `details['precision_tau']` uses `None`, never `np.nan`, as its
  "not found" sentinel (confirmed against its only other consumer, `results.py`, which has
  always checked `is not None`). Tutorial 6 and `NEURALMI_REFERENCE.md` both assumed
  `np.nan` and called `np.isnan()` on it, which raises on `None` — fixed in both, plus the
  `precision.py` docstring that made the same wrong claim.
- `mode='conditional'` concatenates X and Z's windowed tensors along the channel axis,
  which requires matching window-size dimensions — but every `z_processor_type='categorical'`
  encoding collapses that axis to the category count, not `window_size`, so it could never
  satisfy the shape check for a Z with more than one category (using the same `window_size`
  across X/Y/Z, as Tutorial 7 previously advised, did not fix this). Fixed in
  `run.py`/`analysis/conditional.py`: `'majority_vote'`/`'probability'` (window-constant
  per-category summaries) are folded into channels and broadcast across X's window;
  `'full_trajectory'` (genuine per-timepoint resolution) is folded into channels with its
  real window axis restored. Surfaced two further, separate processor-level discrepancies
  along the way, both reconciled with a tight, warned, exactly-1-sample trim in
  `analysis/conditional.py` rather than raising: `ContinuousWindowDataset` reserves a
  deliberate "+1" interpolation-edge buffer sample that `CategoricalWindowDataset` doesn't
  need, so a full-resolution categorical Z's window *size* can be 1 short of X's; and the two
  processors' different window-*coverage*-validation implementations (searchsorted-based vs.
  a two-pointer scan) can disagree on a boundary window's validity, so their window *counts*
  can differ by 1 too (the same truncate-to-shorter precedent `create_dataset` already uses
  when aligning X/Y streams of different duration — see Tutorial 2). A genuinely different
  `window_size` or duration still raises in both cases.
- Several `nmi.run()` calls across Tutorials 1, 3, and 8 were missing `show_progress=False`
  (Tutorials 4-7 were already fixed in an earlier pass), so their tqdm progress bars got
  captured as inert, non-updating text/widget clutter in the saved notebook — worst case one
  frozen bar per loop iteration (Tutorial 3's 7-value `window_size` sweep alone had 7). Added
  across all 9 remaining call sites.
- `permutation_test=True`'s own internal progress bar (`run.py`'s `_run_permutation_test`,
  `desc="Permutation test"`) never read `show_progress` at all, so it rendered regardless of
  the outer `nmi.run(..., show_progress=False)` setting — the gap that made Tutorial 8's two
  permutation-test demo cells keep showing widget clutter even after the fix above. Both
  `tqdm(...)` call sites now pass `disable=not show_progress`; regression coverage added in
  `tests/test_permutation.py::TestPermutationTestProgressBar`.
- `mode='rigorous'` plotting had three related bugs in `results.py`/`visualize/plot.py`.
  `Results.plot(show=False)` didn't actually suppress the render — `show` was never forwarded
  to `plot_bias_correction_fit`, which defaulted to `show=True` and called `plt.show()`
  regardless, closing the figure in Jupyter's inline backend before any post-hoc edits to the
  returned `ax` could take effect. Fixed by forwarding `show=show`. Reliability was only ever
  annotated on the plot when `is_reliable=False`, with the reason hardcoded to always say
  `leverage_warning` even when `fit_quality_warning` (or too few surviving gamma points, which
  sets neither flag) was the actual cause — now annotated symmetrically (a green "reliable" box
  when `True`) with the reason derived from the real flags. `plot_bias_correction_fit` accepted
  `label`/`color` kwargs but silently ignored them (swallowed into unused `**kwargs`), so
  `Results.compare()` for rigorous mode couldn't visually distinguish overlaid results — every
  result rendered in the same hardcoded gray/black/red with duplicate generic legend entries.
  Both kwargs are now respected; `compare()` also gained a per-result reliability text box and
  now forces `show=False` on each per-result call regardless of the outer flag (letting it
  default to `True` mid-loop closed the shared axes after the first result, truncating the
  overlay). Regression coverage added in `tests/test_visualize_extended.py`.
- Spectral/participation-ratio tracking was scattered across five knobs at three
  different reachability levels (`Output.spectral_mode`, plus internal-only
  `track_spectral_metrics`/`spectral_output`/`return_spectrum`/`spectral_whitening`
  keys with no dataclass field to set them directly) — `return_spectrum` in
  particular was real and documented, just stranded in a flat `base_params` table
  disconnected from the `Output`/`Training` reference a user would actually check.
  `mode='dimensionality'` also had its own separate, completely unreachable
  `spectral_mode` parameter (`analysis/dimensionality.py`) that the public
  `Dimensionality` config rejected outright — dead code. Collapsed to one knob,
  `Output.track_spectral_history` (default `False`): `pr_eig`, `pr_singular`, and
  the raw cross-covariance spectrum (singular values) are now always available in
  `result.details` from the best epoch, at no extra cost — they were already being
  computed as a byproduct of the participation-ratio calculation and simply weren't
  being kept. `track_spectral_history=True` additionally records the same three
  values at every epoch in `spectral_metrics_history`. `effective_rank`/
  `spectral_entropy` were dropped from both paths (cheaply derivable from the raw
  spectrum if ever needed). `spectral_whitening` (how the covariance is whitened
  before SVD) is unrelated and untouched.
### Added

- **`benchmarks/vs_classical_estimators.ipynb`**: compares `NeuralMI` against the KSG
  estimator and geometric intrinsic-dimension estimators (MLE, TwoNN) on problems chosen to
  be hard for them — not a tutorial, but useful for deciding whether a neural estimator is
  the right tool for a given dataset.

### API redesign — flat keywords → typed config objects (`run()`)

`nmi.run()` moved from a ~74-parameter flat signature to a small set of grouped,
typed config objects (`neural_mi/config.py`). This is a **hard break**: the flat
keywords are removed and now raise a `TypeError`. Every config field defaults to
`None` (unset → dropped), so `BASE_PARAMS_SCHEMA` remains the single source of
defaults and `run(x, y)` still behaves identically to before. Kept as a private
`_run_flat` engine; the new `run()` lowers configs onto it.

**Migration reference (old flat keyword → new location):**

| Old (flat) | New |
|---|---|
| `base_params={'embedding_dim':.., 'hidden_dim':.., 'n_layers':.., 'critic_type':.., 'dropout':.., 'norm_layer':.., 'use_spectral_norm':.., 'shared_encoder':.., 'embedding_model':.., 'custom_critic':.., 'custom_embedding_cls':.., 'use_decoder':.., 'pytorch_predefined':.., ...}` | `model=Model(...)` |
| `base_params={'n_epochs':.., 'learning_rate':.., 'batch_size':.., 'patience':.., 'optimizer':.., 'optimizer_params':.., 'scheduler':.., 'scheduler_params':.., 'gradient_clip_val':.., 'use_amp':.., 'eval_train':.., 'peak_fraction':.., 'max_eval_samples':.., 'train_subset_size':.., 'save_best_model_path':.., 'augmentation_params':.., 'dataset_device':.., ...}` | `training=Training(...)` |
| `n_epochs=`, `batch_size=`, `shared_encoder=`, `dropout=`, `norm_layer=`, `optimizer=`, `optimizer_params=`, `scheduler=`, `scheduler_params=`, `use_amp=`, `use_spectral_norm=`, `gradient_clip_val=`, `eval_train=`, `peak_fraction=`, `max_eval_samples=`, `train_subset_size=`, `save_best_model_path=`, `custom_critic=`, `custom_embedding_cls=` (top-level shortcuts) | the matching field on `model=Model(...)` / `training=Training(...)` |
| `processor_type_x=`, `processor_params_x=`, `processor_type_y=`, `processor_params_y=`, `x_time=`, `y_time=` | `processing=Processing(x=, x_params=, y=, y_params=, x_time=, y_time=)` |
| `split_mode=`, `train_fraction=`, `n_test_blocks=`, `split_gap_fraction=`, `train_indices=`, `test_indices=` | `split=Split(mode=, train_fraction=, n_test_blocks=, gap_fraction=, train_indices=, test_indices=)` |
| `estimator=`, `estimator_params=` | `estimator='name'` **or** `estimator=Estimator(name=, params=)` |
| `output_units=`, `spectral_mode=`, `return_embeddings=`, `track_embeddings=`, `return_rotated_embeddings=`, `rotated_embeddings_whitening=`, `rotated_embeddings_per_epoch=`, `return_rotation_matrices=`, `max_index_reduction=`, `x_name=`, `y_name=`, `channel_names_x=`, `channel_names_y=` | `output=Output(units=, spectral_mode=, return_embeddings=, ..., x_name=, channel_names_x=)` |
| `random_seed=` | `seed=` |
| `delta_threshold=`, `min_gamma_points=`, `confidence_level=`, `gamma_range=`, `residual_threshold=`, `r2_threshold=`, `leverage_threshold=` (rigorous) | `rigorous=Rigorous(...)` |
| `tau_grid=`, `corrupt_target=`, `corruption_method=`, `n_noise_samples=`, `threshold_ratio=` (precision) | `precision=Precision(...)` |
| `lag_range=`, `equalize_n=` (lag) | `lag=Lag(...)` |
| `history_window=`, `prediction_horizon=`, `bidirectional_te=` (→ `bidirectional`), `rigorous=`, `gamma_range=` (transfer) | `transfer=Transfer(...)` |
| `split_method=`, `n_splits=`, `channel_indices_x=` (dimensionality) | `dimensionality=Dimensionality(...)` |
| `z_data=`, `z_time=`, `z_processor_type=`, `z_processor_params=`, `rigorous=`, `gamma_range=` (conditional) | `conditional=Conditional(...)` |
| `mode=`, `sweep_grid=`, `n_workers=`, `device=`, `verbose=`, `show_progress=`, `permutation_test=`, `n_permutations=` | **unchanged** (still top-level) |

Notes: mode-specific configs are named to match `mode` and only the matching one is
used (a stray one warns). `bidirectional_te` was renamed to `Transfer(bidirectional=)`.
Anywhere a config is accepted, a plain `dict` with the same field names also works
(e.g. `training={'n_epochs': 50}`). The 12 config classes are exported from
`neural_mi`. Full suite green (541 passed / 1 skipped) on the new API.

### Changed

- **Participation-ratio metrics renamed: `pr_eig` / `pr_singular`** (repo-wide): the vague, inconsistently-named `participation_ratio` / `pr_covariance` / `participation_ratio_singular` are gone. `pr_eig` = `(Σσᵢ²)²/Σσᵢ⁴` (eigenvalue/covariance-spectrum variant), `pr_singular` = `(Σσᵢ)²/Σσᵢ²` (singular-spectrum variant). `dimensionality` mode's lean/default spectral output now reports **both** variants (previously only one, under the vague name) — a real behavior change, not just a rename. This is a breaking change with no deprecated aliases (library is pre-publication). Downstream notebooks/scripts outside this repo that reference the old names need updating.
- **`HybridCritic.forward` now row-chunks pair scoring** (`neural_mi/models/critics.py`), matching `ConcatCritic`'s existing pattern: the full `(N², 2d)` pair tensor is never materialized at once, bounding peak memory during large-N evaluation (e.g. `dimensionality` mode's multi-split evaluation).
- **`PretrainedBackboneEmbedding` gradient/BatchNorm fix** (`neural_mi/models/embeddings.py`): removed a stray `torch.no_grad()` around the frozen backbone's forward pass that was silently severing gradient to the trainable channel adapter whenever `input_dim != backbone_in_ch` (the adapter was frozen at random init and never trained). Backbone freezing is already handled via `requires_grad=False` and doesn't need `no_grad`. Also added a `train()` override so the frozen backbone's BatchNorm layers stay in eval mode regardless of the outer model's train/eval state.
- **`min_coverage_fraction` semantics documented, not changed** (`neural_mi/data/temporal.py`): coverage is a source-*timestamp* count, not a value-validity check (NaN-valued-but-present-timestamp windows are not dropped), and gap interpolation is not bounded by the coverage fraction. Docstring-only.
- **`AnalysisWorkflow.__init__` input_dim now uses the full flattened shape** (`neural_mi/analysis/rigorous.py`): `int(np.prod(shape[1:]))` instead of `shape[1]*shape[2]`, which silently dropped the width axis for 4-D (`cnn2d`) inputs. No-op for existing 3-D callers.
- **`estimators/bounds.py`, `logmeanexp_nodiag`**: `dim=0` was falsy and silently fell through `dim or (0,1)` to reduce over both axes instead of just dim 0. Fixed to `dim if dim is not None else (0,1)`. Never fired in practice (only `None` and `(0,1)` are passed anywhere in the codebase today) — pure future-proofing.
- **`analysis/transfer.py`, `_build_te_arrays`**: replaced a Python list-comprehension + `torch.stack` with `tensor.unfold()`, which produces the same layout as a view instead of materializing three large window-array copies. Verified bit-exact equivalent before applying.
- **`data/temporal.py`, `SpikeWindowDataset`**: the `max_spikes_per_window` truncation message (data is silently dropped) is now a `logger.warning`, not `logger.info`.

### Fixed

- **`analysis/conditional.py`**: X and Z with mismatched window sizes now raise a clear `ValueError` before the `torch.cat` into XZ, instead of a bare shape-mismatch error.
- **`data/temporal.py`, `CategoricalWindowDataset.__init__`**: integer-typed labels with negative values now raise a clear `ValueError` instead of silently reaching `np.bincount` via `n_categories = data.max() + 1`. Non-integer labels (floats/strings) are unaffected — still auto-relabeled to consecutive non-negative integers as before.
- **`run()` silently dropped `optimizer_params`/`estimator_params`/`scheduler_params` set only in `base_params`** (`neural_mi/run.py`): the top-level kwargs default to `None`, were converted to `{}` via `X or {}` before being passed to `_inject()`, and `_inject()` only skips overwriting on a literal `None` — so the `{}` unconditionally clobbered a caller-supplied `base_params['optimizer_params']` whenever the matching top-level kwarg wasn't also passed. This silently zeroed any `weight_decay` set via `base_params` alone, including via this library's own documented call pattern. Fixed by passing the raw value (no `or {}`) so `_inject`'s None-guard and `apply_defaults()`'s missing-key backstop behave the same as every other base_params key (e.g. `dropout`). Covered by a new regression test (`tests/test_validation.py`).
- **`embedding_model='gru'`/`'lstm'` unconditionally rejected pre-processed 3-D input** (`neural_mi/run.py`): a top-level validation check raised `ValueError` whenever `processor_type_x=None`, even when `x_data` was already a legitimate pre-windowed `(N, C, W)` tensor — a case the rest of the pipeline (`ParameterSweep`'s own `is_proc_sweep` auto-detection) already supports. Fixed to skip the check when the array already has a time dimension (`x_data.ndim == 3`).

### Removed

- **`CalciumEmbedding` / `embedding_model='calcium_cnn'`** (`neural_mi/models/embeddings.py`) and its generator `generate_windowed_calcium`: cut rather than fixed. `_deconv_kernel` built the time-reversed, unit-normalized indicator impulse response — a **matched filter**, which further low-passes the signal, not a deconvolution (which would sharpen/invert the blur). The docstring's "FIR deconvolution" claim did not match what the code did. Independently, the only generator for it carries its shared information in firing rate, for which mean fluorescence is already near-sufficient, so a correct deconvolution would not have bought anything there either. All registry entries, base-params schema keys (`tau_rise`, `tau_decay`, `learn_calcium_kernel`), tests, tutorial sections, and reference-doc rows removed with it.
- **`SpikePhysicsEmbedding` / `embedding_model='spike_physics'`** (`neural_mi/models/embeddings.py`): removed after an empirical gate against a regularized generic MLP on rate-code spike data (Regime C, `results/gate/decision_log.md`) came back NO_HEADROOM under the gate's strict criterion (10x converged-N ratio, but overlapping ±1 std bands at the discriminating N). All registry entries, the `'features'`/`'concat'` fusion code path, base-params schema keys, tests, tutorial sections (Tutorials 8, 10, 11), and reference-doc rows removed with it.
- **Depthwise-separable first layer / `use_depthwise` on `embedding_model='cnn'`** (`neural_mi/models/embeddings.py`): removed after an empirical gate on a favorable multichannel regime (per-channel distinct carriers) showed no advantage over a plain `CNN1D` — at N=10000 plain CNN (0.62 bits) actually exceeded depthwise (0.47 bits); see `results/gate/decision_log.md`. The `use_depthwise` flag, base-params schema key, tests, tutorial sections (Tutorials 10, 11), and reference-doc rows removed with it. `generate_windowed_multichannel` (its only consumer) is retained since it still feeds the gate's own evidence chain.
- **`SincEmbedding` / `embedding_model='sinc_cnn'`** (`neural_mi/models/embeddings.py`): removed after the initial win against a generic CNN turned out to be confounded — the generic baseline ran at the library-default `kernel_size=3` with mean pooling, neither of which can build a frequency-selective filter (a log-band-power pooling fix along the way was real, but did not settle the confound). A fair comparison (kernel-size-matched CNN, {3, 15, 51}, with and without matched log-band-power pooling) on the same band-power-vs-broadband-interference regime showed `sinc_cnn` never converged within N=3000 across 3 seeds (0.872±0.323 bits vs. true MI 0.996) while the matched-kernel generic CNN did (1.074±0.052 bits); see `results/gate/decision_log.md`. All registry entries, base-params schema keys (`n_sinc_filters`, `feature_fusion`), the modality-metadata (`sample_rate_x/y`) injection that fed it, tests, tutorial sections (Tutorials 8, 10, 11 — the latter two retired entirely), and reference-doc rows removed with it.

### Added

- **SVD-aligned rotated embeddings (`return_rotated_embeddings`)** (`neural_mi/utils.py`, `neural_mi/training/trainer.py`, `neural_mi/analysis/task.py`):
  A new `compute_cross_covariance_rotation()` utility and four new `base_params` keys enable returning embeddings re-projected so that dimension 0 captures the most shared variance between the two modalities, dimension 1 the next most, and so on — consistent with the Participation Ratio ordering. This makes the first *k* dimensions directly interpretable without separately inspecting the SVD.

  New parameters:
  - `return_rotated_embeddings` (bool, default `False`) — enable the feature. Works alongside `return_embeddings` and/or `track_embeddings`; has no effect for `concat` critics (emits a `UserWarning`).
  - `rotated_embeddings_whitening` (str or None, default `'std'`) — whitening applied to the cross-covariance before SVD to derive the rotation axes. Does **not** affect the scale of the returned embeddings (which are always `ZX_centered @ U`/`ZY_centered @ V` in original embedding space). Matches the default used by `compute_cross_covariance_spectrum` for consistency with PR estimates.
  - `rotated_embeddings_per_epoch` (bool, default `False`) — when `track_embeddings` is also enabled: `False` (default) derives one global rotation from the best epoch's embeddings and applies it to all tracked epochs (consistent coordinate system across epochs); `True` computes a fresh SVD per tracked epoch (shows how latent structure emerges).
  - `return_rotation_matrices` (bool, default `False`) — include U and V in `result.details` so new data can be projected into the same aligned basis.

  New `result.details` keys:
  - `embeddings_x_rotated`, `embeddings_y_rotated`, `embeddings_rotation_singular_values` (+ optional `embeddings_rotation_x/y`) — from `return_embeddings` path.
  - `embedding_history_x_rotated`, `embedding_history_y_rotated`, `embedding_rotation_singular_values` (+ optional rotation matrices) — from `track_embeddings` path.

- **Physics parameter tracking (`get_physics_params()` extensibility hook)** (`neural_mi/models/embeddings.py`, `neural_mi/training/trainer.py`):
  the trainer calls `get_physics_params()` after every evaluation epoch, if the
  embedding implements it, and stores results in `result.details`:
  - `result.details['physics_params_history']` — dict of per-epoch parameter lists (keys prefixed by
    `x_` or `y_`).
  - `result.details['physics_params_final']` — same keys with values from the best epoch.
  Both keys are absent when the embedding does not implement `get_physics_params()` — true
  of every currently-shipped embedding (this was originally exercised by `SincEmbedding`,
  since removed; see `results/gate/decision_log.md`). Kept as a hook for future custom
  embeddings supplied via `custom_embedding_cls`.

- **Pretrained backbone spatial dimension mismatch handling** (`neural_mi/models/embeddings.py`):
  `PretrainedBackboneEmbedding` now automatically inserts a bilinear `nn.Upsample` layer when input
  images are not 224×224 (standard ImageNet resolution). The upsample is created lazily on the first
  forward pass and emits a `UserWarning` with the input and expected sizes.

- **Two new windowed generators with analytically known MI** (`neural_mi/generators/synthetic.py`):
  - `generate_windowed_oscillatory(n_windows, n_channels, window_size, f_carrier_hz, sample_rate, latent_mi, snr)` —
    windowed oscillatory LFP with shared latent carrier; MI computed from the linear-Gaussian
    `ρ_obs = ρ_latent × SNR² / (SNR² + 1)` formula.
  - `generate_windowed_multichannel(n_windows, n_channels, window_size, f_min_hz, f_max_hz, sample_rate, latent_mi, snr)` —
    same model with per-channel carrier frequencies uniformly spaced in `[f_min_hz, f_max_hz]`; total
    MI = sum over channels.
  Both return `(X, Y, true_mi)` and are exported from `neural_mi.generators`.

- **`generate_timing_code_spike_trains` generator** (`neural_mi/generators/synthetic.py`):
  new function for generating a precise-timing spike code embedded in high-rate
  independent background Poisson noise.  Each neuron pair shares signal spikes
  (`signal_rate` Hz) that Y fires with a fixed `delay` + Gaussian `jitter`;
  both populations are additionally driven by `background_rate` Hz background.
  With `background_rate >> signal_rate`, summary statistics of the spike counts
  are dominated by noise, so GRU's ability to process actual spike timestamps
  gives it a detectable advantage.  Exported from `neural_mi.generators`.

- **`torchvision` optional dependency** (`setup.py`, `pyproject.toml`):
  added `vision` extras group (`pip install neural_mi[vision]`) for
  `PretrainedBackboneEmbedding`.  Was previously an undeclared dependency.

### Removed

- **`generate_oscillatory_lfp`** (`neural_mi/generators/synthetic.py`): replaced by
  `generate_windowed_oscillatory`, which returns IID pre-windowed arrays and an analytically
  computed true MI value.

### Fixed

- **`ContinuousWindowDataset` / `CategoricalWindowDataset` time-vector units**
  (`neural_mi/data/temporal.py`): when `sample_rate` is given but no
  `time_vector`, both datasets now construct a seconds-based time vector
  (`np.arange(N) / sample_rate`) instead of an integer-index vector.  With
  integer indices, a `window_size` in seconds (e.g., 0.5 s) was less than one
  sample, producing zero valid windows.

- **GRU/LSTM validation false-positive in `ParameterSweep`**
  (`neural_mi/analysis/sweep.py`): the check that errors when
  `embedding_model='gru'` and `processor_type_x=None` no longer fires when
  data has already been pre-processed upstream (detected via
  `processor_params_x['preprocessed'] == True`).

- **`'cnn2d'` missing from `ALLOWED_VALUES`** (`neural_mi/validation.py`):
  `embedding_model='cnn2d'` raised a validation error; added to the allowed
  list.

### Changed

- **Online data augmentations**: per-batch augmentations applied
  during training only (never at eval time).  Three new `base_params` keys:
  - `augmentation_params` — shared augmentation spec for both X and Y.
  - `augmentation_params_x` — per-variable override for X (`None` = use shared,
    `{}` = explicitly disable).
  - `augmentation_params_y` — per-variable override for Y (same semantics).
  Available augmentation keys (`neural_mi/augmentations.py`):
  - *Spatial (4-D input only)*: `random_flip_h`, `random_flip_v`,
    `random_rotation_90`, `random_crop`, `random_erase`, `time_mask`,
    `freq_mask`, `gaussian_blur`.
  - *Non-spatial (any ndim)*: `gaussian_noise`, `intensity_scale`,
    `channel_dropout`.
  - *Custom*: `custom` — a single callable or list of callables, each
    accepting an `(N, ...)` tensor and returning a tensor of the same shape.
  Application order is always: spatial → non-spatial → custom.  Spatial
  augmentations requested on non-4-D input emit a `UserWarning` and are
  skipped gracefully.

- **Plotting — `estimate` mode: `conservative_epoch` marker**: when
  `peak_fraction < 1.0` is used, `result.details` contains
  `'conservative_epoch'` (the epoch whose train MI is reported as the final
  estimate).  `Results.plot()` now draws a green dotted vertical line and a
  diamond scatter marker at that epoch alongside the existing red best-epoch
  marker.  Without `peak_fraction`, the plot is unchanged.

- **Plotting — `conditional` mode**: `Results.plot()` now renders a vertical
  bar chart showing the three CMI components: `I(XZ;Y)`, `I(Z;Y)`, and
  `CMI I(X;Y|Z)`.  Bars are labelled with numeric values.  Previously raised
  `NotImplementedError`.

- **Plotting — `transfer` mode**: `Results.plot()` now renders a bar chart
  showing `TE(X→Y)` and (when available) `TE(Y→X)`.  The plot title includes
  the directionality index and a plain-English direction label when present.
  Previously raised `NotImplementedError`.

- **Plotting — `Results.compare()` for `estimate` mode**: overlay of test-MI
  training curves across multiple runs.  Each curve is drawn in a distinct
  colour with best-epoch markers as faint dashed vertical lines.  Previously
  raised `NotImplementedError`.

- **`plot_bias_correction_fit` now returns `ax`**: the function previously
  returned `None`; it now returns the `matplotlib.axes.Axes` used for the
  plot, enabling composability.

- **`plot_cross_correlation` composability**: added `ax`, `show`, and `xlim`
  parameters; function now returns the axes.  The previously hard-coded
  `xlim=(-100, 100)` is gone — the full lag range is shown by default.

- **`analyze_mi_heatmap` composability**: added `ax` and `show` parameters;
  function now returns the axes.  All `print()` statements replaced with
  `logger.info()` / `logger.warning()` calls so the function is silent in
  library use.

- **`_RESULT_COLS` extended**: `pr_eig`, `pr_eig_mean`, `pr_eig_std`,
  `pr_singular`, `pr_singular_mean`, `pr_singular_std`, and `split_id`
  added to the frozenset.  These were previously missing, causing the
  dimensionality sweep-variable inference to consider them as candidate
  x-axis columns and fail silently.

- **Rigorous plot `is_reliable=False` annotation**: when `is_reliable` is
  `False` in `result.details`, `Results.plot()` adds a red text box to the
  bias-correction figure so unreliable extrapolations are immediately visible
  without checking `result.summary()`.

- **`use_amp` parameter** (`'auto'` / `True` / `False`): mixed-precision (AMP)
  training via `torch.cuda.amp.autocast` + `GradScaler`. Enabled automatically
  on CUDA devices (`'auto'`); a no-op on CPU and MPS so all existing workflows
  are unaffected. Added to `BASE_PARAMS_SCHEMA` in `defaults.py`; wired through
  `run()`, `task.py`, and `Trainer`. On CUDA the forward pass runs in float16,
  reducing memory by ~2× and improving throughput on Ampere+ GPUs.

- **`Results.to_dict()`**: returns a fully JSON-serialisable `dict` with keys
  `mode`, `mi_estimate`, `params`, `details`, and `dataframe`. All numpy arrays
  are converted to nested Python lists; DataFrames are exported in `records`
  orientation; non-serialisable objects fall back to a `"<TypeName>"` string.

- **`Results.to_json()` now uses `to_dict()`**: arrays are serialised as nested
  lists (previously as `"<array shape=... dtype=...>"` strings), making the JSON
  output both human-readable and round-trippable. Existing call signatures and
  the auto-naming / no-overwrite behaviour are unchanged.

- **Named variable support in `run()`**: four new optional top-level arguments —
  `x_name` (str), `y_name` (str), `channel_names_x` (list of str),
  `channel_names_y` (list of str). Stored in `result.params` for use in plot
  axis labels. In pairwise mode, `channel_names_x/y` are injected into
  `result.details['variable_names_x/y']`, which drives the MI-matrix heatmap
  tick labels. Fallback when omitted is the current integer-index behaviour.

- **`return_embeddings` — full dataset, original order**: `result.details['embeddings_x']`
  and `result.details['embeddings_y']` now contain embeddings for **all** windows in
  original sample order, with no subsampling or shuffling. Previously the extraction
  block reused `max_eval_samples` (default 5000) and drew a random subset via
  `np.random.choice`, making the returned arrays impossible to align with
  time-indexed behavioural signals. Inference is now performed in mini-batches of
  512 (internal constant `_EMBEDDING_BATCH`) to avoid OOM on large datasets.
  `max_eval_samples` continues to control only the epoch-level evaluation MI estimate
  and has no effect on embedding extraction. Applies to all modes: `estimate`,
  `sweep`, and `dimensionality`.

- **`extract_embeddings()` — full dataset, original order**: same fix applied to the
  standalone function in `embeddings_io.py`. The `max_samples` parameter has been
  removed entirely; inference uses mini-batched ordered iteration over the full
  input. Code that previously passed `max_samples=N` will receive a `TypeError` and
  should be updated (pass the desired subset of the data directly).

- **Dimensionality mode — embedding arrays no longer corrupt the results DataFrame**:
  `run_dimensionality_analysis()` now strips `embeddings_x`/`embeddings_y` from the
  per-split result dicts before constructing the `pd.DataFrame`. Previously, if
  `return_embeddings=True` was set, 2-D numpy arrays would end up as object-dtype
  columns in the aggregated DataFrame, breaking groupby aggregation. The embeddings
  are now returned as a second value `(df, embeddings_dict_or_None)` from
  `run_dimensionality_analysis()`; `run()` unpacks this and places the arrays in
  `result.details['embeddings_x/y']`. With `n_splits > 1`, embeddings come from the
  last split's model (logged explicitly).

- **`show` parameter for plot utilities**: `plot_sweep_curve`, `plot_dimensionality_curve`,
  and `plot_bias_correction_fit` now accept `show: bool = True`. When `False`,
  `plt.show()` is suppressed, enabling these functions to be embedded in multi-panel
  figures or called in Jupyter notebooks without blocking execution.

- **Dimensionality mode — `split_method='index'`**: new channel-split option for
  `run_dimensionality_analysis()` / `run(..., mode='dimensionality')`.  Pass
  `channel_indices_x=[0, 1, 4, 5, 7]` as a keyword argument; Y is automatically the
  complement set.  Works for both 2-D `(N, C)` and 3-D `(N, C, W)` data.  When X and
  Y have different channel counts, `shared_encoder=True` is automatically disabled with
  a logger warning; this can be suppressed by explicitly setting `shared_encoder=False`
  in `base_params`.  Multiple `n_splits` independent model initialisations are still
  performed (same fixed channel assignment, different weight initialisation) so the
  output DataFrame retains the same mean/std structure as other split methods.

- **`track_embeddings` parameter**: controls per-epoch embedding extraction during
  training.  Accepted in `base_params` for all analysis modes; in `dimensionality` mode
  the default is `512` (track the first 512 samples each epoch); in all other modes the
  default is `False` (disabled).  Accepted values mirror the existing `eval_train`
  syntax: `False` (off), `True` (first 512 samples), a positive `int` (exact sample
  count), a `float` in `(0, 1)` (fraction of dataset), or `'full'` (entire dataset,
  emits a `UserWarning` about memory cost).  Embeddings are always extracted from the
  first N samples in original order (deterministic, aligns with user-provided labels).
  Per-epoch arrays are stored in `result.details['embedding_history_x']` and
  `result.details['embedding_history_y']` (each a list of `(n_tracked, embed_dim)`
  numpy arrays, one per epoch).

- **`animate_training()` and `result.animate()`**: new animation utility in
  `neural_mi.visualize.animate` that creates frame-by-frame GIF / MP4 animations from
  training history stored in `result.details`.  Panels are auto-detected from available
  data or specified explicitly via `panels=['mi', 'spectral_metrics', 'spectrum', 'embeddings']`.
  The `'mi'` panel always shows test MI vs epoch; train MI is overlaid when present.
  The `'spectral_metrics'` panel plots participation ratio vs epoch.  The `'spectrum'`
  panel shows an animated bar chart of singular values (requires `spectral_mode='full'`).
  The `'embeddings'` panel renders a 2-D or 3-D scatter of learned embeddings at each
  epoch, with PCA or UMAP reduction fitted jointly on all frames for consistent
  coordinates.  Pass `embedding_labels` as a 1-D array or a `dict` of name → array for
  categorical (tab10 palette) or continuous (viridis) point colouring; each dict entry
  produces its own subplot column.  Output is saved as a GIF via `PillowWriter` or MP4
  via `FFMpegWriter` when `output_path` is supplied.  `result.animate(**kwargs)` is a
  thin wrapper around `animate_training(result, **kwargs)`.

- **`umap-learn >= 0.5.0` added as a hard dependency**: required for UMAP dimensionality
  reduction in `animate_training()`.  Added to both `setup.py` and `pyproject.toml`.

- **`CNN2D` encoder — `embedding_model='cnn2d'`**: new 2-D convolutional encoder for
  image-like input of shape `(N, C, H, W)`. Architecture: stacked `Conv2d` blocks (same
  padding) → `AdaptiveAvgPool2d(1)` → `Flatten` → two `Linear` layers. The adaptive
  pooling head collapses any spatial size to a fixed `(1, 1)` representation so no
  `input_shape` parameter is needed — only `n_channels` is used, exactly as for `CNN1D`.
  Reuses the existing `kernel_size` base parameter (must be odd, default 3).  Exported
  from `neural_mi.models` and selectable via `embedding_model='cnn2d'` in any analysis
  mode.

- **4-D input support in `task.py`**: `run_training_task()` now handles 4-D tensors
  `(N, C, H, W)` when computing `input_dim_x/y` (previously assumed `(N, C, W)`).
  Behaviour by model type:
  - `'cnn2d'` — handled natively; no warning.
  - `'mlp'` — flattened to `C×H×W` features silently; no warning.
  - `'cnn'` (CNN1D) — raises `ValueError` (spatial axes are ambiguous for a 1-D kernel).
  - all other sequence/graph models (`'gru'`, `'lstm'`, `'tcn'`, `'transformer'`) —
    emit a `UserWarning` noting that spatial dimensions are not preserved.

- **Dimensionality mode — spatial split methods for 4-D data**: six new/updated
  `split_method` values for `run_dimensionality_analysis()` / `run(..., mode='dimensionality')`:
  - `'horizontal'` — top half vs. bottom half (height axis).
  - `'vertical'` — left half vs. right half (width axis).
  - `'row_interleaved'` — even-indexed rows → X, odd-indexed rows → Y. Fine-grained
    horizontal stripes; avoids contiguous spatial bias.
  - `'col_interleaved'` — even-indexed columns → X, odd-indexed columns → Y.
    Column-wise counterpart to `'row_interleaved'`.
  - `'diagonal'` — true geometric split: upper-left triangle + main diagonal → X
    (`row ≤ col`), lower-right triangle → Y. Works with `'mlp'` and sequence models;
    raises `ValueError` for `'cnn2d'` / `'cnn'`. Rectangular input (H ≠ W) is allowed
    with a warning; `shared_encoder` is always disabled (diagonal pixels go to X).
  - `'antidiagonal'` — true geometric split: upper-right triangle + anti-diagonal → X
    (`row + col ≤ W−1`), lower-left triangle → Y. Same constraints as `'diagonal'`.
  All six require 4-D input `(N, C, H, W)`. Unequal flat sizes disable `shared_encoder`
  automatically.

  > **Note:** what was previously called `'diagonal'` (interleaved rows) has been renamed
  > `'row_interleaved'`, and `'antidiagonal'` (interleaved columns) has been renamed
  > `'col_interleaved'`.  The names `'diagonal'` and `'antidiagonal'` now refer to the
  > true geometric triangular splits.

- **Dimensionality mode — `split_method='index'` extended to 4-D**: the existing index
  split now handles 4-D tensors `(N, C, H, W)` in addition to 2-D `(N, C)` and 3-D
  `(N, C, W)` data.

- **`PretrainedBackboneEmbedding` — `embedding_model='pretrained_backbone'`**: frozen
  torchvision backbone (e.g. ResNet18, EfficientNet-B0) used as a fixed feature
  extractor, followed by a trainable MLP head mapping to `embedding_dim`.  Set
  `pytorch_predefined` to the torchvision model name and `pretrained=True` to load
  ImageNet weights.  Expects 4-D input `(N, C, H, W)`.

- **New synthetic data generators** (`neural_mi/generators/synthetic.py`):
  `generate_oscillatory_lfp`, `generate_modulated_spike_trains`,
  `generate_noisy_image_pairs`.  All three are exported from `neural_mi.generators`.

### Fixed

- **`test_critic_chunking_equivalency[Separable]` flaky failure**: the test used
  unseeded `x_data`/`y_data` fixtures; in the full suite their values depended on
  cumulative RNG state, occasionally producing bilinear critic scores of magnitude
  10⁵+ where float32 differences between chunked and non-chunked forward passes
  exceeded the `atol=1e-4` tolerance.  Fix: move `torch.manual_seed(42)` to the
  very first line of the test body and construct `x_data`/`y_data` locally,
  making the test fully deterministic regardless of execution order.

- **`test_paired_time_shift_positive` flaky failure**: after undoing a time shift
  (`+d` then `−d`) the spike-time float64 round-trip can shift the reconstructed
  window range by ε, yielding ±1 window vs. the original and an index-offset in
  the window tensor (so `after_undo[i] ≈ original[i−1]`).  Fix: replace the
  fragile `torch.allclose` data comparison with a check that (1) the continuous
  time vector is approximately restored (`np.allclose(..., atol=1e-6)`) and (2)
  the window count is within ±1 of the original.

- **`_BUILD_PARAMS_KEYS` consolidation** (`task.py`): the module-level constant was missing
  all six decoder keys (`use_decoder`, `decoder_weight`, `decoder_weight_x`,
  `decoder_weight_y`, `decoder_output_activation_x`, `decoder_output_activation_y`). A
  redundant local redefinition inside `run_training_task()` held the complete list. The
  local redefinition has been removed; the module-level constant is now the single
  authoritative source used by both `run_training_task()` and `extract_embeddings()`.

- **`spectral_output` docstring** (`trainer.py`): the docstring incorrectly stated `'full'`
  as the value that returns all spectral metrics. The actual code checks for `== 'all'`;
  the docstring has been corrected to match.

- **Rigorous mode `is_reliable` false-positive for large datasets (R² gate)**:
  R² of the WLS linear fit was previously a condition for `fit_quality_warning`
  (and therefore `is_reliable=False`). With large N, the finite-sampling bias
  across gamma values is inherently small, producing a near-flat MI vs. gamma
  curve where R² collapses toward zero even when the fit and extrapolation are
  sound (observed: R²=0.10 at N=10 000, R²=0.00 at N=1 000 on well-behaved
  data). R² is now computed and stored in `result.details['r_squared']` for
  transparency but no longer affects `fit_quality_warning`. The `r2_threshold`
  parameter is retained in all public APIs for backward compatibility but is a
  no-op.

- **Rigorous mode `is_reliable` false-positive for large datasets (residual
  gate)**: `fit_quality_warning` (max externally studentized residual >
  threshold) was also a condition for `is_reliable=False`. The heteroscedastic
  WLS structure of rigorous mode — where low-gamma rows dominate the MSE while
  high-gamma training runs have natural noise — routinely produces large
  studentized residuals even for perfectly valid fits (observed: residuals of
  4.63, 9.03, and 3.22 at N=1 000/5 000/10 000 with threshold 2.5).
  `fit_quality_warning` is now **informational only** and does not affect
  `is_reliable`. `is_reliable` is now governed solely by (1) sufficient gamma
  points and (2) `leverage_warning` (LOO γ=1 intercept-stability check), which
  is scale-invariant and directly tests whether the extrapolation anchor is
  stable.

### Changed

- **`NEURALMI_REFERENCE.md`** updated to document recently added parameters: `peak_fraction`
  added to Training table; `track_spectral_metrics` and `return_spectrum` added to
  Spectral/Whitening table; new Decoder section covering `use_decoder`, `decoder_weight`,
  `decoder_weight_x/y`, and `decoder_output_activation_x/y`; `use_amp` added to Memory &
  Device table; `plot_dimensionality_curve` and `plot_bias_correction_fit` signatures in
  the Low-Level Utilities section updated to reflect the new `show` parameter; new Online
  Data Augmentations section covering all 11 built-in keys, the three `base_params`
  augmentation keys, application order, and usage examples; §4 extended with 4-D input
  note for CNN2D and spatial augmentations; §5 `run()` signature annotated with
  augmentation params; §10 CNN2D input shape corrected to 4-D; Quick Reference Card
  updated; §12 Design Decisions augmented with augmentation note.

### Tests Added

- `tests/test_augmentations.py`: 32 tests covering all 11 built-in augmentation
  types (shape preservation for 3-D and 4-D inputs, spatial-on-3D `UserWarning`,
  Gaussian noise / intensity scale / channel dropout semantics, time mask / freq
  mask / random erase / Gaussian blur correctness, custom callable and list of
  callables, `custom` invalid-type error, application order, and `True` shortcut
  defaults).

- `tests/test_amp_and_names.py`: 14 tests covering `use_amp` (`'auto'`, `True`,
  `False` on CPU; schema presence; sweep-mode passthrough) and named variables
  (`x_name`/`y_name` stored in params; pairwise `channel_names_x/y` injected
  into details; clean params when names are omitted; signature reflection).
- `tests/test_results_extended.py`: 9 new tests in `TestToDict` covering
  `to_dict()` return type, required keys, 1-D and 2-D numpy arrays as nested
  lists, training-history inclusion, DataFrame as records, `None` DataFrame, and
  `to_json()` round-trip fidelity.
- `tests/test_rigorous_diagnostics.py`: added `test_low_r2_does_not_trigger_fit_quality_warning`
  — regression test confirming that a near-flat MI curve (large N, tiny bias)
  with low R² does not set `fit_quality_warning=True`. Also added
  `test_gamma1_outlier_sets_is_reliable_false` — confirms that `leverage_warning`
  (not `fit_quality_warning`) is the gate for `is_reliable`.
- `tests/test_dimensionality.py`: coverage for `_extract_embedding_history`,
  `_strip_embeddings`, `split_method='index'` input validation (missing/empty/out-of-range
  `channel_indices_x`, all-channels-to-X, unknown split_method), the `shared_encoder`
  auto-disable guard (unequal vs equal channel counts), and correct 2-D and 3-D channel
  slicing.
- `tests/test_animate.py`: 27 new tests covering `_auto_panels`, `_fit_reducer` (no-op when
  `embed_dim ≤ n_components`; `reduction='none'`; PCA; empty input; unknown reduction),
  `_resolve_scatter_color` (None, float, categorical int, categorical str), and
  `animate_training()` smoke tests for MI-only, auto-panels, spectral, spectrum, train MI
  overlay, missing `test_mi_history` error, empty panel list error, single-array and dict
  embedding labels, missing embedding history warning, 3-D embeddings, `reduction='none'`,
  and the `result.animate()` delegate.

---

## [Unreleased]

### Added

#### Generic Variational Wrapper (`use_variational=True` for all encoders)
- **Removed `VarMLP`** — the purpose-built variational MLP is gone entirely.
  All `use_variational=True` runs now use `VariationalWrapper` instead.
- **New `VariationalWrapper` class** in `neural_mi/models/embeddings.py`:
  wraps *any* base encoder (MLP, CNN1D, GRU, LSTM, TCN, Transformer, or a
  custom module) with μ and log σ² projection heads plus the reparameterization
  trick.  At training time returns `(z_sampled, kl_loss / batch_size)`; at eval
  time returns `(μ, 0.0)` — identical protocol to the former `VarMLP`.
- **`build_critic()` updated**: when `use_variational=True`, `build_critic`
  builds the selected base encoder normally and then wraps it with
  `VariationalWrapper(base_encoder, embed_dim)`.  This applies to all six
  `embedding_model` choices: `'mlp'`, `'cnn'`, `'gru'`, `'lstm'`, `'tcn'`,
  `'transformer'`.
- **`shared_encoder` remains fully compatible** with variational mode: the
  shared encoder is built once and wrapped once; both `net_x` and `net_y`
  point to the same `VariationalWrapper` instance.
- `neural_mi/models/__init__.py`: exports `VariationalWrapper`; no longer
  exports `VarMLP`.

### Changed

#### Generic Variational Wrapper
- `neural_mi/utils.py`: `build_critic()` no longer has a special `VarMLP`
  branch for variational mode.  The model-selection tree is now strictly
  by `embedding_model` name; variational wrapping is a post-construction step.
- `neural_mi/models/decoders.py`: removed `'var_mlp'` from the name aliases in
  `build_decoder()` — it was never needed in practice.
- `neural_mi/run.py`: updated `use_spectral_norm`, `dropout`, and `norm_layer`
  docstrings to reference "MLP" rather than "MLP/VarMLP".

### Tests Added

#### Generic Variational Wrapper
- `tests/test_models.py` fully updated:
  - `test_varmlp_embedding` → `test_variational_wrapper_embedding`
  - `test_varmlp_kl_loss` → `test_variational_wrapper_kl_loss`
  - New `test_variational_wrapper_eval_returns_mu` — checks determinism in eval mode.
  - New `test_variational_wrapper_gradients_flow` — verifies gradients reach
    both the mu/log_var heads and the base encoder.
  - New class `TestVariationalWrapperAllEncoders` — 6 parametrized tests
    (one per encoder type) each checking output shape, positive KL in training
    mode, and zero KL in eval mode.
  - Critic tests updated: `test_separable_critic_with_varmlp` →
    `test_separable_critic_with_variational_wrapper`,
    `test_concat_critic_with_varmlp` →
    `test_concat_critic_with_variational_wrapper`.
  - `test_critic_chunking_equivalency` now builds `VariationalWrapper(MLP(…), embed_dim)`.
  - `critic_and_data` fixture: `"SeparableVarMLP"` renamed to `"SeparableVariational"`.

#### Enhanced Rigorous Mode Diagnostics
- **Standardized-residual check:** After the WLS bias-correction fit,
  `rigorous` mode now computes externally studentized residuals.  If
  `max(|rᵢ|) > residual_threshold` (default 2.5) **or** R² < `r2_threshold`
  (default 0.90), `fit_quality_warning=True` is stored in `result.details` and
  `is_reliable` is set to `False`.
- **LOO γ=1 intercept-stability check:** Refits WLS excluding all γ=1
  rows and measures the relative intercept shift
  `δ = |I_full − I_loo| / (|I_full| + ε)`.  If `δ > leverage_threshold`
  (default 0.20), `leverage_warning=True` is stored in `result.details` and
  `is_reliable` is set to `False`.
- Both checks store their source in `result.details`:
  `fit_quality_warning`, `leverage_warning`, `r_squared`, `max_abs_residual`,
  `loo_intercept_shift`.
- New configurable thresholds in `base_params` / `analysis_kwargs`:
  `residual_threshold` (default 2.5), `r2_threshold` (default 0.90),
  `leverage_threshold` (default 0.20).
- `Results.summary()` now prints diagnostic reasons when `is_reliable=False`.

#### Optional Decoder (Deep Symmetric Information Bottleneck)
- New `use_decoder=True` flag in `base_params` enables decoder-augmented training
  for all analysis modes.
- New `base_params` keys:
  - `use_decoder` (bool, default `False`)
  - `decoder_weight` (float, default 1.0) — reconstruction weight applied to both X and Y.
  - `decoder_weight_x` / `decoder_weight_y` (float | None, default `None`) —
    per-channel overrides; when `None` the shared `decoder_weight` is used.
  - `decoder_output_activation_x` / `decoder_output_activation_y` (str,
    default `'linear'`) — `'linear'` for continuous, `'sigmoid'` for binary/spike,
    `'softmax'` for categorical.
- New module `neural_mi/models/decoders.py` with decoder variants for all six
  embedding architectures: `MLPDecoder`, `CNN1DDecoder`, `GRUDecoder`,
  `LSTMDecoder`, `TCNDecoder`, `TransformerDecoder`, and a `build_decoder()`
  factory function.
- Training objective:
  - Deterministic: `L = −MI(Z_X; Z_Y) + w_x·MSE(X, X̂) + w_y·MSE(Y, Ŷ)`
  - Variational: `L = KL_X + KL_Y − β·MI(Z_X; Z_Y) + w_x·MSE + w_y·MSE`
- Decoder and encoder parameters are optimised jointly by the same optimizer.
- `result.details['decoder_recon_loss']` reports the weighted reconstruction loss.
- `Results.summary()` prints decoder reconstruction loss when present.

#### Rigorous Bias Correction for Conditional and Transfer Modes
- `mode='conditional'` and `mode='transfer'` now accept `rigorous=True` in
  `analysis_kwargs` (or as a top-level `run()` keyword) to produce a
  bias-corrected, extrapolated estimate.
- Uses correlated subsampling (same master permutation index for all component
  estimates at each γ) so noise partially cancels in the difference.
- New parameters for conditional/transfer rigorous mode: `gamma_range`,
  `delta_threshold`, `min_gamma_points`, `confidence_level`, `residual_threshold`,
  `r2_threshold`, `leverage_threshold`.
- Returns a full rigorous details dict: `mi_corrected`, `mi_error`, `slope`,
  `is_reliable`, `gammas_used`, `fit_quality_warning`, `leverage_warning`,
  `r_squared`, `max_abs_residual`, `loo_intercept_shift`, `raw_results_df`.
- Graceful fallback: when the linear-region finder prunes too aggressively (noisy
  data with no clear γ trend), `run_rigorous_scalar_analysis` falls back to
  using all available γ values and sets `is_reliable=False`.
- New public function `run_rigorous_scalar_analysis()` in
  `neural_mi/analysis/rigorous.py` for use with any scalar MI-derived quantity.
- Pairwise mode: per-pair rigorous estimation will be addressed in a future release.

### Changed
- `neural_mi/models/critics.py`: Added `get_training_embeddings(x, y)` method to
  `BaseCritic` — returns embeddings with gradient flow (used by decoders during
  training).
- `neural_mi/training/trainer.py`: Added `decoder_x`, `decoder_y`,
  `decoder_weight_x`, `decoder_weight_y` constructor parameters; training loop
  now incorporates decoder reconstruction loss when decoders are present.
- `neural_mi/analysis/task.py`: Builds and passes decoders to `Trainer`; added
  decoder keys to `_BUILD_PARAMS_KEYS` for model serialisation.
- `neural_mi/run.py`: Pairwise mode unit conversion now correctly handles
  `mi_mean`/`mi_std` columns in addition to legacy `mi_estimate`.
- `neural_mi/defaults.py`: `MODE_KWARGS_SCHEMA` now includes a `'conditional'`
  entry (previously missing); `'transfer'` entry extended with rigorous params.
- `neural_mi/results.py`: `Results.summary()` extended to display rigorous
  diagnostic reasons and decoder reconstruction loss.

### Tests Added
- `tests/test_rigorous_diagnostics.py` — unit and integration tests for
  `_compute_fit_diagnostics`, `_post_process_and_correct`, decoder shapes,
  output activations, end-to-end decoder training, and
  `run_rigorous_scalar_analysis`.
- `tests/test_conditional_transfer_rigorous.py` — end-to-end tests for
  `rigorous=True` in conditional and transfer modes.

## [Unreleased]

### Added

- **`dataset_device` parameter**: controls where dataset tensors are stored in
  memory, independent of the compute device (`device` param).  Default is
  `'cpu'` for all modes, which keeps large arrays in pageable system RAM so the
  OS can reclaim memory freely between tasks.  Pass `'auto'` to co-locate data
  with the compute device (MPS / CUDA), which avoids repeated host→device
  transfers when the same dataset is evaluated many times — precision analysis
  uses `'auto'` by default for exactly this reason.  Any explicit device string
  is also accepted.  Added to `BASE_PARAMS_SCHEMA` in `defaults.py`.
- **Module-level dataset cache in `task.py`**: sequential sweep tasks that share
  identical data and dataset-construction parameters (processor type / params /
  `dataset_device`) now reuse a single pre-built `PairedDataset` object instead
  of re-running `create_dataset()` for every task.  The cache is keyed by data
  memory address and construction fingerprint; LRU eviction keeps at most four
  entries.  Temporal datasets (`PairedTemporalDataset`) are intentionally
  excluded from caching because they are mutated in-place by `time_shift()`
  during training.  The cache is process-local and also benefits
  `multiprocessing` workers that handle more than one task.
- **Memory-pressure warning in `ParameterSweep`**: when `dataset_device` is not
  `'cpu'` and a sequential sweep has more than 20 tasks, a `UserWarning` is
  emitted before training starts, estimating dataset size and advising the user
  to set `dataset_device='cpu'` if memory pressure is a concern.

### Fixed

- **Root-cause fix for MPS/CUDA memory exhaustion during long sweeps**: all
  dataset classes (`StaticDataset`, `ContinuousWindowDataset`,
  `SpikeWindowDataset`, `BinnedSpikeDataset`, `CategoricalWindowDataset`)
  previously stored `self.data` on the accelerator device by default (via
  `get_device()`).  On Apple Silicon (unified DRAM) this caused the full
  dataset to be allocated on MPS for every training task in a sequential sweep,
  and PyTorch's MPS allocator does not return freed tensors to the OS without
  an explicit `torch.mps.empty_cache()` call.  With 300 tasks this caused
  monotonic memory growth and system crashes.  The fix moves all dataset tensor
  storage to CPU by default; batch loops in the `Trainer` already call
  `.to(device)` per batch so no training logic changed.
- **`SubsetView`: device-agnostic indexing**: index tensors are now always
  created as CPU LongTensors, and `__getitem__` converts any 0-dim index
  tensor to a Python `int` before delegating to the dataset.  Python `int`
  indices work on any device tensor (CPU or accelerator), eliminating the
  previous `RuntimeError` when dataset data was on CPU but index tensors were
  on MPS, and making `SubsetView` safe for use with `dataset_device='auto'`.
- **`SpikeWindowDataset.apply_precision()` now reads from `data_master`**:
  previously the method rounded `self.data` in-place while reading from
  `self.data` as the source.  Calling it twice at different precision levels
  would compound the rounding error rather than starting from the original
  spike times.  The fix mirrors the implementation in
  `ContinuousWindowDataset` and `BinnedSpikeDataset`, which already read from
  `self.data_master`.
- **`PairedDataset._align_datasets()` now performs effective truncation**:
  when X and Y datasets have different sample counts, the method now slices
  `self.data` on both sides so that `__len__()` (which reads
  `self.data.shape[0]`) reports the correct length.  The previous
  implementation set a phantom `n_windows` attribute that `StaticDataset` does
  not use, leaving mismatched datasets that would crash during training.  Any
  lazily-allocated `data_master` is also invalidated so it is re-cloned from
  the truncated data on next use.
- **`CategoricalWindowDataset._move_full_trajectory()` no longer assigns
  `data_master` twice**: the method previously set `self.data_master` at the
  end of its body while `move_data_to_windows()` also set it unconditionally
  after every encoding method returned.  The redundant internal assignment is
  removed; all three encoding paths now consistently delegate `data_master`
  initialization to their single caller.
- **`DEVELOPERS_GUIDE.md` processor file reference corrected**: entries
  referring to a non-existent `processors.py` file have been updated to point
  to the correct files (`handler.py`, `temporal.py`, `static.py`) and the
  step-by-step guide for adding a new data processor now reflects the actual
  codebase structure.
- **`z_time` parameter in `run()`**: a time vector can now be passed for the
  conditioning variable Z in `mode='conditional'` when `z_processor_type` is a
  temporal processor (e.g. `'continuous'`). Forwarded to `create_dataset` as
  `x_time=z_time`.
- **`Results.save(path=None)`**: serialises a Results object to a pickle file.
  Auto-generates a timestamped filename (`neuralmi_{mode}_{YYYYMMDD_HHMMSS}.pkl`)
  in the current directory when no path is given; never overwrites existing files
  (appends a numeric suffix). Returns the absolute path of the saved file.
- **`Results.load(path)`**: classmethod that deserialises a Results object
  previously saved with `save()`.
- **`Results.to_json(path=None)`**: exports a human-readable JSON snapshot of
  scalar fields (`mode`, `mi_estimate`, `params`) and the DataFrame. Large objects
  in `details` (numpy arrays, raw result lists) are summarised by type and shape.
  Auto-naming and no-overwrite logic follow the same convention as `save()`.
- **`sample_rate` parameter wired** into `ContinuousWindowDataset` and
  `CategoricalWindowDataset`. When provided it overrides the period inferred from
  the time vector; now propagated from `processor_params_x/y` via `handler.py`.
- **`max_spikes_per_window` and `n_seconds` parameters wired** into
  `SpikeWindowDataset`. `max_spikes_per_window` caps the allocated spike slot
  count; `n_seconds` sets an explicit recording duration for temporal extent
  inference. Both are now propagated from `processor_params_x/y` via `handler.py`.

### Changed
- **Precision mode `Results.mi_estimate`**: now holds the baseline MI (at zero
  corruption) rather than the precision threshold τ. The threshold τ remains in
  `Results.details['precision_tau']`. `Results.summary()` for precision mode
  now shows baseline MI, τ, and the threshold MI value explicitly.
- **Pairwise mode DataFrame columns**: the `mi_estimate` column is replaced by
  `mi_mean` and `mi_std` (consistent with sweep/lag modes). The MI matrix
  continues to hold per-pair means.
- **`Results.summary()`** for `conditional`, `transfer`, and `pairwise` modes
  now prints mode-relevant detail (component MI values, directionality index,
  matrix range) in addition to the generic DataFrame shape.
- **`bidirectional_te` is exclusively a top-level `run()` parameter**. It has
  been removed from `MODE_KWARGS_SCHEMA['transfer']` to eliminate the
  dual-pathway inconsistency where the same parameter could be set in two places.
- **Non-processor, non-lag calls skip an intermediate `PairedDataset`**
  allocation when both `processor_type_x` and `processor_type_y` are `None`.
  Tensor conversion and length alignment now happen inline in `run._run_inner`,
  removing a redundant object construction on every non-temporal call.

### Fixed
- **`_initialize_windows` min/max inversion** (`handler.py`): the `min` and `max`
  operators were swapped, causing the temporal window range to *expand* beyond the
  original recording after `time_shift()` instead of being clamped to it.
- **`_BUILD_PARAMS_KEYS`** (`task.py`): `dropout`, `norm_layer`, and
  `use_spectral_norm` were missing. Models saved with `norm_layer='batch'` or
  `norm_layer='layer'` can now be reloaded correctly via `extract_embeddings()`.
- **Precision mode `n_test_blocks`** (`precision.py`): was read from `**kwargs`
  (ignored when set in `base_params`). Now reads from `base_params`.
- **Noise mask in `apply_noise`**: `ContinuousWindowDataset` and
  `SpikeWindowDataset` derived the non-zero position mask from `self.data` (the
  working copy), causing repeated noise applications to compound. Both now derive
  the mask from `self.data_master`.
- **`window_size` in `processor_params_y`** was silently ignored when Y specified
  a different value. Now emits a clear warning and uses X's `window_size` (shared
  `WindowManager` constraint).
- **Pairwise permutation test** was silently discarded when `permutation_test=True`.
  Now emits a `UserWarning` about computation cost and populates
  `results.details['null_distribution']` with the null MI samples.
- **Conditional MI log message** incorrectly annotated MI values as being in
  `output_units`; values are always in nats at that point in the code.
- **Transfer entropy `bidirectional=False` log level**: demoted from `warning`
  to `info` (this is the normal, expected default — not a user-facing warning).
- **`trainer.py` variable name**: `is_first_valid_epoch` renamed to
  `has_valid_baseline` to reflect its actual semantics (True once a baseline MI
  has been established, not on the first valid epoch).

## [2.1.0] - 2026-03-13

### Added

- **Optimizer flexibility**: `optimizer` parameter accepts a string name
  (`'adam'`, `'adamw'`, `'sgd'`, `'rmsprop'`, `'adagrad'`) or any
  `torch.optim.Optimizer` subclass; `optimizer_params` forwards extra
  constructor kwargs (e.g. `weight_decay`).
- **Learning-rate schedulers**: new `scheduler` / `scheduler_params` parameters
  in `run()` and `base_params`. Supported names: `'cosine'` (CosineAnnealingLR),
  `'cosine_warmup'` (linear warm-up + cosine), `'step'` (StepLR),
  `'plateau'` (ReduceLROnPlateau, monitors test MI). Custom
  `torch.optim.lr_scheduler` subclasses also accepted. Scheduler steps at the
  end of each training epoch; `ReduceLROnPlateau` receives the current test MI.
- **MLP regularisation**: `dropout` (float, default 0.0) and `norm_layer`
  (`None`/`'layer'`/`'batch'`, default `None`) parameters for MLP
  embedding networks, applied in the order Linear → Norm → Activation → Dropout.
- **Per-epoch train MI tracking**: `eval_train` parameter (`False` / `True` /
  fraction / sample count) records train-set MI at every epoch, populating
  `result.details['train_mi_history']`. The training curve plot overlays the
  dashed orange train curve automatically when this key is present.
- **Raw train MI**: `result.details['raw_train_mi']` is always populated with
  the true computed train-set MI regardless of whether the model generalised
  (when the model fails, `train_mi` is set to 0 while `raw_train_mi` preserves
  the actual value for diagnostic purposes).
- **Small-sample warnings**: after dataset creation, `run()` emits a `UserWarning`
  when the processed dataset has fewer than 200 windows (strong) or fewer than
  500 windows (mild) with specific regularisation suggestions.
- **Tutorial 09**: new end-to-end pipeline tutorial covering sanity check,
  window sweep, architecture sweep, training diagnostics, rigorous estimation,
  lag analysis, conditional MI, and summary reporting on synthetic hippocampal
  data.

### Fixed

- **Sweep MPS/CUDA bug**: `ParameterSweep._prepare_tasks` incorrectly called
  `.is_mps` on numpy arrays when `torch.backends.mps.is_available()` was True,
  causing `AttributeError` during parallel sweeps. Fixed operator precedence
  and added `isinstance(tensor)` guard.
- **`is_proc_sweep` auto-detection**: `ParameterSweep.run()` and
  `_prepare_tasks()` now infer `is_proc_sweep` automatically from data type
  (3-D `torch.Tensor` → pre-processed; everything else → raw). The parameter
  still accepts an explicit `bool` for backward compatibility.
- **Unit conversion in `estimate` mode**: `train_mi`, `raw_train_mi`,
  `test_mi_history`, and `train_mi_history` were returned in nats even when
  `output_units='bits'`. All four keys are now correctly converted.
- **All-MI-non-positive threshold**: changed from `< 0` to `<= 0` so that
  exactly-zero MI values correctly trigger the model-failure warning.

### Changed

- Tutorial 08 updated with new sections on optimizer choice, MLP regularisation,
  training diagnostics (`eval_train`), and LR schedulers.
- `NEURALMI_REFERENCE.md` updated with new parameters in the `run()` reference
  block and Base Parameters table (Optimizer/Scheduler section; dropout,
  norm_layer, eval_train).

## [2.0.0] - 2026-03-10

### Added

- Unified `run()` entry point supporting nine analysis modes: `estimate`, `sweep`,
  `rigorous`, `lag`, `dimensionality`, `precision`, `conditional`, `transfer`,
  `pairwise`.
- `rigorous` mode: automated finite-sampling bias correction via subsampling and
  linear extrapolation to the infinite-data limit, with confidence intervals.
- `dimensionality` mode: latent dimensionality estimation using a Hybrid Critic and
  cross-covariance SVD (Participation Ratio).
- `precision` mode: spike-timing precision analysis via "Train Once, Evaluate Many"
  with deterministic rounding and additive noise corruption methods.
- `transfer` mode: transfer entropy estimation using the chain-rule decomposition.
- `conditional` mode: conditional mutual information I(X;Y|Z).
- `pairwise` mode: channel-to-channel MI matrix (self-pairwise and cross-pairwise).
- Data processors: `ContinuousProcessor`, `SpikeProcessor`, `CategoricalProcessor`
  for LFP/EEG, spike-train, and categorical state data respectively.
- Embedding models: MLP, CNN1D, GRU, LSTM, TCN, Transformer.
- Critic architectures: `SeparableCritic`, `ConcatCritic`, `HybridCritic`.
- MI estimators: InfoNCE, SMILE.
- Blocked and random train/test splitting strategies for temporal and IID data.
- Built-in synthetic data generators (`generate_correlated_gaussians`,
  `generate_nonlinear_from_latent`, `generate_temporally_convolved_data`,
  `generate_correlated_spike_trains`, `generate_xor_data`,
  `generate_correlated_categorical_series`, `generate_event_related_data`).
- Permutation test support for null-distribution estimation.
- `extract_embeddings` utility for extracting learned latent representations.
- `spectral_mode` parameter for controlling spectral metric computation.
- Comprehensive tutorial series (Tutorials 01–08) covering basic estimation,
  neural data formats, temporal analysis, sweeps, rigorous estimation,
  population-level questions, and model/estimator selection.
  Tutorial 09 (full end-to-end pipeline) added in v2.1.0.
- `NEURALMI_REFERENCE.md`: complete library reference document.
- `THEORY.md`: theoretical background for all core methods.
- `CONCEPTS.md`: code-based walkthrough of MI estimator internals.
- `DEVELOPERS_GUIDE.md`: contributor guide to the codebase architecture.
