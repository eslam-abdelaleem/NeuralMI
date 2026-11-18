1. MAP

- `quantities.py`: 12 named functions. Each builds offset rows (or passes a config) and calls `run()`; none has estimation logic of its own. There are three dispatch paths inside the file:
  - 8 functions go `_by_offsets` → `_named` → `_call` (quantities.py:321, 172, 97).
  - TE and CTE go `_named` directly.
  - `block_mi` and `interaction_information` call `run` directly (they skip `_with_save_path` and `call_verbosity`/`collect_repeats`).
  - Callers: the user only.
- `results.py`: the `Results` dataclass. Its methods are `get`, `best`, `summary`, `plot`, `compare`, `animate`, `save`/`load`, `to_dict`/`to_json`. The file also holds about 20 module-level `_draw_*`, `_summary_*` and `_overlay_*` helpers. Those helpers are imported by name (private) in `analysis/modes.py:35-38` into `MODES`, while `results.py` lazy-imports `MODES` back. That is a cycle hidden by function-local imports.
- `utils.py`: a grab bag; redistribution in F3. Its only documented user-facing name is `build_offset_arrays` (USING.md:1287); everything else is internal.
- `visualize/plot.py`: 9 plot functions. `plot_sweep_heatmap` and `plot_sweep_bar` are used by `Results.plot` but are not exported from `visualize/__init__`. `plot_cross_correlation` and `analyze_mi_heatmap` are used by nothing in the library or tutorials, only by tests.
- `visualize/animate.py`: `animate_training`, behind `Results.animate`.
- `generators/oracle.py`: `LatentGaussian` (exact I(A;B|C) for any offset spec, `sample`, `mi_rate`, `storage_transfer`) plus 10 closed-form generators. Nothing in the library uses them; they are used by tests, all 5 tutorials, README and USING. They are user-facing, so keep them.

Quantity vs mode

| quantity | runs | same code path? |
|---|---|---|
| `interaction_information` | `interaction` | Identical call; only adds `params['quantity']`. A pure alias. |
| `transfer_entropy` / `conditional_transfer_entropy` | `transfer` | Same call. Adds an iterable `history_window`, which `run` refuses in `sweep_grid` and points users to TE instead (run.py:139-143). Cannot set `prediction_horizon`. |
| `instantaneous_mi` | `estimate` | Same producer, but it records row geometry: leak checks, permutation blocks and contiguous rigorous chunks all differ from `run(x,y)` on the same (T,C) input. Whether the numbers differ is unverified. |
| `block_mi` | `estimate`/`sweep` | Literally `run` with a windowed `Processing`. |
| `mi_rate`, `instantaneous_exchange`, `directed_information_rate` | `conditional` | Offset rows, then `conditional`. |
| `offset_quantity` | `conditional`/`estimate` | A TE spec here (USING.md's own example) runs `mode='conditional'`, not `'transfer'`. Same quantity, two modes. |
| AIS, PI, CPI | `estimate`/`sweep`/`rigorous` | — |

`Results.mode` is the run mode, so `mi_rate` reports `'conditional'` and its plot and summary label the bars "I(X,W;Y)" and "I(W;Y)". `AIS(k=5)` gives `'estimate'` but `k=[5]` gives `'sweep'`.

Signatures (all take `rigorous=False, n_workers=1, show_progress=True, **run_kwargs`)

| function | data | own param | stride |
|---|---|---|---|
| `active_information_storage` | x | `k`, `future_k` | yes |
| `predictive_information` | x | `k` | yes |
| `instantaneous_mi` | x, y | – | yes |
| `cross_predictive_information` | x, y | `k` | yes |
| `block_mi` | x, y | `window_size` | no |
| `transfer_entropy` | x, y | `history_window` (+`bidirectional`) | yes |
| `conditional_transfer_entropy` | x, y, w | `history_window` (+`bidirectional`) | yes |
| `interaction_information` | x, y, w | – | no |
| `mi_rate` | x, y | `h`, `half_width` | yes |
| `instantaneous_exchange` | x, y | `k` | yes |
| `directed_information_rate` | x, y | `k` | yes |
| `offset_quantity` | dict | `spec` | yes |

2. FINDINGS (ranked)

F1. Rigour is requested four different ways.
- Quantities take `rigorous=True|Rigorous`.
- `run` uses `mode='rigorous'` plus a `rigorous=Rigorous(...)` config.
- The difference configs take a `rigorous=True` field.
- The 7 fit fields are re-declared in `Rigorous`, `Transfer`, `Conditional` and `Interaction` (config.py:315, 370, 435, 468), and `quantities._FIT_FIELDS`/`_difference_fit` (quantities.py:62, 82) translates between them.

Change: one `rigorous: bool|Rigorous` argument on `run` that applies to any mode; drop `mode='rigorous'` and the duplicated fields. Effort M, medium risk.

F2. History length has three names: `k`, `history_window` and `h` (quantities.py:447, 677, 809). It also has no `prediction_horizon` on TE, and passing `transfer=` through `run_kwargs` raises a duplicate-kwarg TypeError (quantities.py:122). Change: name it `k` everywhere (or `history`), and expose `prediction_horizon`. Effort S.

F3. `utils.py` should be split up.
- `build_critic`, `_library_critic`, `_custom_critic`, `_scaled_input`, `_scaling_modes`, `encoders_scale`, `check_custom_critic`, `critic_recipe_takes`, `ARCHITECTURE_KEYS`, `MODEL_SETTINGS`, `CRITIC_*`, `_EMBEDDING_CLASSES`, `_names`, `_accepts_kwarg` → `models/factory.py`.
- `build_optimizer_and_scheduler` → `training/optim.py`.
- `cross_covariance_svd`, `compute_*`, `_whiten` → `training/spectrum.py` (only `trainer.py` and `task.py` call them).
- `warn_if_*_split_leaks` → `data/rows.py`.
- `_shift_data` → `analysis/lag.py` (its only caller).
- `build_offset_arrays` → `analysis/offsets.py` (its only caller besides docs).
- `validate_stride`, `validate_window` → `validation.py`.
- `float_tensor`, `with_window_axis`, `_as_tensor`, `_is_spike_input`, `_ensure_cpu` → `data/tensors.py`.
- `get_device` → `device.py`.
- `_configure_multiprocessing` → `parallel.py`, its only caller.
- `mi_report_units` → `defaults` or a units module. Unit conversion is currently re-implemented 5 times: utils.py:510, assemble.py:58, results.py:596, trainer.py:491, oracle.py:41.

Today, importing a tensor helper drags in every model class (utils.py:19-25). Leftovers: the docstring at utils.py:72 cites a nonexistent `workflow.py`; utils.py:138 says "mi_rate has h and W" (now `half_width`); utils.py:146 points to "block_mi's step_size", which does not exist. Effort M, low risk.

F4. Drawing code lives in `results.py` but is wired by `analysis/modes.py`. Move every `_draw_*`, `_overlay_*` and `_summary_*` into `visualize/` (say `visualize/modes.py`), and let `MODES` import from there. `results.py` then drops its top-level pyplot import (results.py:21). Also:
- `Results.summary` hard-codes TE columns (`te_yx_mean`, `directionality_index`; results.py:706-710) despite the registry. Move that into `transfer`'s `summarize`.
- `_draw_extrapolation` and `_overlay_extrapolations` duplicate the fit-dict and ladder code (results.py:168-173 vs 412-415).

Effort M, low risk.

F5. The quantity layer re-implements `run` internals.
- `_by_offsets` re-derives input-scaling and stream kinds "as run() decides" and passes them through the private `_stream_kinds`/`_rows`/`_row_phases` parameters on the public `run` signature (quantities.py:352-371, run.py:669-671).
- It has its own iterable-parameter grid: `dispatch_tasks` plus `merge_results`, each value run with `n_workers=1`, so a `sweep_grid` of repeats runs serially per value (quantities.py:156, 219-225).
- It has a third, recursive grid for preparation keys (quantities.py:332-345).

Change: let `run` accept offset specs (rows plus geometry) and grid keys like `history_window`. The quantities then become spec constructors only. Effort L, medium risk.

F6. Merge `transfer` into `conditional` (cross-area). TE is `te_spec` fed to the same builder (analysis/transfer.py:27, 159). Keep TE-specific bidirectionality as an option of the spec route. Also, `interaction_information` adds nothing over `run(mode='interaction')`: keep one of the two. Effort M.

F7. `Results` key discoverability. There is no `keys()` listing; `get()` searches 4 tables. Columns are inconsistently named: `i_xypast_yfuture` for transfer vs `mi_xw_y` for conditional (modes.py:830 vs 815). Change: add `keys()` (or `__getitem__`), unify the column names, and add `quantity` to `__repr__`/`summary`. Effort S.

F8. The two serialisation paths are fine as designed, but `to_json` has no loader, and `summary()` only prints (it returns no string, and there is no `__str__`). Uncertain: a pickled `params` holding a `custom_critic` or a local class may fail to load. Effort S.

F9. Bugs and inconsistencies in `visualize`.
- In `animate`, categorical `embedding_labels` always get a colourbar and never a legend: `_resolve_scatter_color` returns a float `c`, so the check at animate.py:317 is always true.
- `animate` shows Y's embeddings titled "Embeddings X" and never shows both (animate.py:281, 314).
- `animate` duplicates `plot_embeddings`' reducer and palette code.
- `animate_training` hard-codes the axis names `lag`, `ch_x`, …, which `MODES` already declares.
- `plot_embeddings` has no `show=`, contrary to USING.md:1686.
- `plot_cross_correlation` (uses `x[0]`, requires `true_lag`) and `analyze_mi_heatmap` (a lag×window heuristic that `run` cannot produce in one call) are tutorial leftovers. Delete them.
- Export `plot_sweep_heatmap` and `plot_sweep_bar`.
- `visualize` imports `analysis.rigorous` (plot.py:384).
- `seaborn` is a core dependency used only for `stripplot`, `heatmap` and `despine`; it could go.

Effort S.

F10. Public submodules in `__all__`.
- User-facing: `generators`, `preprocessing` (`Transform`, `PCA`), `models` (`BaseEmbedding`, `BaseCritic`), `data` (`create_dataset`; the docs use the deep path `data.handler`), `estimators` (`ksg_mi`, once).
- Internal: `utils` (except `build_offset_arrays`), `validation` (never documented) and `results` (redundant with `nmi.Results`).
- Also, `__init__.py` imports private `_group_repeats_by_cell`.

Change: drop `utils`, `validation` and `results` from `__all__`, and re-export `build_offset_arrays` (or make `offset_quantity` the only route). Effort S.

F11. Inconsistencies in `generators`.
- Returns differ: `(x, y)` without the MI (correlated, nonlinear) vs `(x, y, mi)` vs `(data, oracle)`.
- Default types differ: torch for 4 generators, numpy for 4.
- Size names differ: `n_samples`, `n_windows`, `T`.
- Dead code: `seed is not None` fallbacks in xor (oracle.py:1002-1006) with seed docstrings saying "Defaults to None" (987, 1056); a no-op `x[:, ch] = x[:, ch]` (1081).
- The return annotation at oracle.py:514 says numpy while the default is torch.

Change: always return `(x, y, exact_mi_bits)`, numpy by default. Effort S.

F12. `instantaneous_mi` vs `run(x, y)` on identical (T,C) data behave differently in their time-order handling, and the docstring (quantities.py:544) does not make the consequence clear. Either record geometry for raw (T,C) input in `run`, or document it. Effort S (uncertain how many results it changes).

3. KEEP
- Offset specs `{'A', 'B', 'C'}` shared by `offset_quantity`, `build_offset_arrays` and `LatentGaussian.exact(**spec)`, so every estimate has an oracle.
- One `Results` type across all modes, with `runs`/`dataframe`/`details`.
- The `MODES` registry driving `plot`/`summary`/`compare`.
- `get()` refusing ambiguous reads.
- The validation messages and the amplification-factor reporting.
- `LatentGaussian` itself.