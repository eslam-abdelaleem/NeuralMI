# NeuralMI Consolidated Upgrade — Agent Handoff

## 0. How to read this document

You (the agent) are working inside the full NeuralMI repository, with access to the
source, tests, tutorials, and these companion specs already in the project:

- `noise_injection_dimensionality_spec.md` — the ceiling-escape noise-injection feature for
  `dimensionality` mode. Authoritative for that feature's behavior and invariants.
- `inductive_bias_gate_job.md` — the empirical gate/fix/remove job for the physics-informed
  embeddings. Authoritative for that workstream.

This document is the **conductor layer**. It does three things the two specs cannot do on
their own: it folds in a set of independent library fixes found during code review, it
resolves the seams and ordering between the three workstreams, and it gives you a
propagation map so wide changes do not leave the repo half-migrated. Where this document and
a companion spec agree, follow the spec's detail. Where this document adds sequencing,
propagation, or conflict-resolution instructions, those govern.

**Do not** treat any spec's internal steps as re-negotiable on technical grounds already
settled there; the theory behind the noise feature and the decision rule behind the gate job
were worked out separately. Your job is correct, clean, fully-propagated implementation.

### 0.1 Files that must be present in the repo for this agent

This agent works only inside the codebase; it cannot see the human's project files or the
chats where these were designed. The following must be dropped into the repo before the run,
because this document references them by name:

- **Required (this document depends on them):** `AGENT_HANDOFF_neuralmi_upgrade.md` (this
  file), `noise_injection_dimensionality_spec.md`, and `inductive_bias_gate_job.md`. Suggested
  location: a `docs/upgrade/` or `handoff/` directory at the repo root.
- **Optional (theory cross-check only):** `dim_estimation_theory.md`. The noise spec cites its
  Sections 5.5 and 7.8 for *why* the correction works, but the noise spec is self-contained for
  *implementation* and its acceptance tests (spec Section 9) are described operationally. Add it
  only if you want the agent to verify against the theory; it is not needed to build the feature.
- **Assumed already in the repo (library's own docs, do not re-add):** `NEURALMI_REFERENCE.md`,
  the tutorials, and the test suite. This document and the gate job instruct you to update
  these in place, so they must exist in the repo; if any is missing, stop and report rather
  than guessing.

If while executing you find you need a file that is not present, stop and list it for the
human to add, rather than inventing its contents.

---

## 1. Two batches, and why

Execute in two separate sessions. This is deliberate; do not collapse them.

**Batch 1 — Deterministic upgrade (this is the cheap, test-verifiable one).**
All library fixes in Section 3, the repo-wide PR rename in Section 4, the noise-injection
feature (the whole `noise_injection_dimensionality_spec.md`), and the two Phase-0 correctness
actions from the gate job (`inductive_bias_gate_job.md` Sections 0.1 and 0.2). No
benchmarking. Everything here is verifiable by unit and acceptance tests. Batch 1 must end
with the full test suite green.

**Batch 2 — Empirical embeddings gate (the expensive, outcome-gated one).**
The gate job's Phases 1 through 3, the conditional survivals/removals they imply, and the
generator/doc cleanup that follows. This runs hundreds of trainings and only then edits code.

Why split:
- The backbone correctness fix (gate 0.2) must land **before** the backbone is benchmarked
  (gate Phase 3), and the whole gate study should run against already-fixed code. Batch 1
  before Batch 2 satisfies this automatically.
- Batch 1 is deterministic and gated by tests; Batch 2's code edits are contingent on
  benchmark verdicts. Fusing them risks deleting encoders before the evidence justifies it,
  or blocking cheap fixes behind a long study.
- The human approves the Batch 2 compute spend separately.

Within Batch 1 you have latitude to group and order the mechanical changes as you see fit,
with two hard constraints: the PR rename (Section 4) is a single atomic pass, and the
noise-injection feature is implemented **after** the PR rename and **after** the HybridCritic
chunking fix (3.1), because it depends on both.

---

## 2. Resolved decision: spike-count stabilization is a documented default-on toggle

This **overrides** `noise_injection_dimensionality_spec.md` Section 3, which specifies
stabilization as automatic and non-optional. The human has decided otherwise. No library
results are committed yet, so adopting this default is safe.

Binned-spike variance stabilization (Anscombe / square-root) is a **toggle that defaults to
on** for binned-spike `dimensionality` runs, e.g. `stabilize_counts=True`. Document it in the
mode's docstring and reference entry. Setting `stabilize_counts=False` yields the plain,
un-stabilized counts so the human can obtain and compare raw results.

The toggle is respected on both paths:
- **No-noise path** (`sigma_add` unset): stabilize by default; off gives plain counts.
- **Noise path** (`sigma_add` engaged): when on, keep the fixed order of operations from spec
  Section 3 (stabilize → measure per-channel std on stabilized values → inject in
  stabilized-std units). When the human turns it off while injecting noise on binned spikes,
  proceed but **emit a warning** that additive noise on raw counts is heteroscedastic and the
  per-channel-std unit is not portable across channels, so the ladder may be uneven. Do not
  silently inject on raw counts without that warning.

Log, on every binned-spike dimensionality run, whether stabilization was applied and which
transform, so units are never misread.

---

## 3. Batch 1 — Library fixes (from code review)

Grouped must-do vs suggested. Each item gives the file and the intent; you have the code, so
implement idiomatically and add/adjust tests.

### 3.1 MUST — HybridCritic pair-row chunking (`models/critics.py`)
`HybridCritic.forward` materializes the full `(N², 2d)` pair tensor in one reshape before the
decision head, with no chunking over pair-rows. At evaluation the trainer runs the critic
over the whole eval set (up to `max_eval_samples`, default 5000), so this is ~13 GB at N=5000
and is exercised every epoch in `dimensionality` mode. `ConcatCritic.forward` already solves
this with a `chunk_rows` loop; port the same pattern to `HybridCritic` so the decision head is
applied in row-chunks and the full pair tensor is never allocated. This is load-bearing for
the noise feature, whose ladder multiplies eval cost by (levels × splits). Verify parity:
chunked and unchunked scores must match to floating-point tolerance on a small case.

### 3.2 MUST — Document `min_coverage_fraction` gap semantics (`data/temporal.py`)
No behavior change; documentation only, in `ContinuousWindowDataset` (docstring),
`move_data_to_windows`, and `validate_window_coverage`. State all three behaviors precisely:
coverage is measured by **source-timestamp count** inside each window, not by value validity;
a window whose timestamps are present but whose values are NaN is **not** dropped by this
check; and a partial-gap window that is retained carries `np.interp`-bridged content up to
`(1 - coverage)` of its length because mid-recording gaps are linearly interpolated while only
before-first / after-last target times are zeroed. Recommend gap pre-collapsing (the
reindexed-timeline approach) or raising the fraction for gappy real-timeline runs.

### 3.3 SUGGESTED (correctness hygiene, latent today)
- `estimators/bounds.py`, `logmeanexp_nodiag`: replace `dim or (0,1)` with
  `dim if dim is not None else (0,1)`, and make the `num_elem` branch consistent for a
  single-int `dim`. Never fires today (only `None` and `(0,1)` are passed); pure
  future-proofing against the `0`-is-falsy trap.
- `analysis/rigorous.py`, `_prepare_tasks`: compute `input_dim` as `prod(shape[1:])` instead
  of `shape[1] * shape[2]` so 4-D (cnn2d) inputs do not misconfigure.
- `data/temporal.py`, `CategoricalWindowDataset.__init__`: validate that labels are
  non-negative integers before `n_categories = data.max() + 1` (it feeds `np.bincount`), with
  a clear error for float/negative labels.

### 3.4 SUGGESTED (developer experience)
- `analysis/conditional.py`: raise a clear error when X and Z window sizes differ, instead of
  a bare `torch.cat` shape failure.
- `data/temporal.py`, `SpikeWindowDataset._compute_max_samples_per_window`: raise the
  `max_spikes_per_window` truncation message from info to warning.
- `estimators/bounds.py`, `smile`: add a code comment noting JS is computed on clipped scores,
  a minor deviation from canonical SMILE (comment only, no behavior change).

### 3.5 SUGGESTED (performance; safe to defer, do only if cheap)
- `analysis/pairwise.py`: each channel pair trains a full estimator; for single-time-bin
  channels this is wasteful and higher-variance than a classical estimator. Leave as-is unless
  trivially switchable; note in docstring.
- `analysis/rigorous.py`, `run_rigorous_scalar_analysis`: the gamma-chunk loop is sequential
  with parallelism only nested inside each `scalar_fn`. A top-level parallel map over chunks
  would speed rigorous CMI/TE substantially. Optional.
- `analysis/transfer.py`, `_build_te_arrays`: replace the Python list-comprehension
  `torch.stack` with `unfold` to avoid materializing three large window arrays. Optional.

---

## 4. Batch 1 — PR rename (atomic, repo-wide, highest propagation risk)

Per `noise_injection_dimensionality_spec.md` Section 8. Do this as one pass and run the full
suite immediately after. Definitions (already matching the code):

- `pr_eig` = `(Σ s_i²)² / Σ s_i⁴`  — currently named `pr_covariance` in `utils.py`.
- `pr_singular` = `(Σ s_i)² / Σ s_i²` — unchanged name.

The naming is already tangled, so rationalize while renaming so the end state is clean:
canonical `pr_eig` and `pr_singular`, with `_mean`/`_std` variants, and no vague
`participation_ratio` / `participation_ratio_singular` left over. The library is
pre-publication; do **not** keep deprecated aliases.

Confirmed call sites to migrate (grep for the old strings again afterward to be sure none
remain, including in tutorials, notebooks, and markdown):

- `utils.py` ~L475: `metrics["pr_covariance"]` → `metrics["pr_eig"]`.
- `training/trainer.py` ~L906-911 and docstring ~L227-230: the `spectral_output` readout that
  emits only `participation_ratio` (= `pr_singular`); the `'all'` branch comment listing
  `pr_covariance`. Make dimensionality output surface **both** `pr_eig` and `pr_singular`
  (the noise spec's output contract, Section 6, requires both per rung), not just one.
- `defaults.py` ~L34 and `validation.py` ~L24: `spectral_output` accepted values. Note the
  existing mismatch (`validation.py` lists `['default','full','all']` while the trainer uses
  `'default'`/`'all'`); reconcile to one vocabulary and include whatever value yields "both PR
  variants" for the dimensionality/noise output.
- `results.py` ~L26-27: `NUMERIC_COLUMNS` entries `participation_ratio`,
  `participation_ratio_mean/std`, `participation_ratio_singular` → explicit `pr_eig`,
  `pr_singular` (+ `_mean`/`_std`).
- `run.py` ~L952: sweep metrics list `['train_mi', 'participation_ratio']` → the renamed
  column(s).
- `analysis/dimensionality.py` ~L173, 195-198, 597-599: readout and the
  `if 'participation_ratio' not in df.columns` guard.
- `visualize/plot.py` ~L130,155,221-239 and `visualize/animate.py` ~L228-229,401: plot column
  names and labels.
- `NEURALMI_REFERENCE.md`: Section 3.3-3.5 and any spectral-metric table.
- Tests and tutorials asserting on the old column names.

**Human-side note to pass along:** downstream notebooks that consume the library output
(`fig2_dimensionality.ipynb` and any handoff that reads `participation_ratio` or
`pr_covariance`) live outside the repo and must be updated by the human after this lands.

---

## 5. Batch 1 — Noise-injection feature

Implement `noise_injection_dimensionality_spec.md` in full. It is self-contained and
detailed; below are only the seams with the rest of the repo, plus the invariants most likely
to be violated by an implementation that does not know the codebase.

### 5.1 Sequencing
Implement after Section 4 (so the output contract reports the renamed `pr_eig`/`pr_singular`)
and after 3.1 (so the ladder's repeated large-N evaluations are affordable).

### 5.2 Seams with existing code
- **Ceiling is `log(eval_size)`, not `log(batch_size)`.** The spec (Section 5) is emphatic and
  the code agrees: the reported InfoNCE estimate is evaluated over the eval denominator, which
  in the trainer is `eval_size = min(len(test_idx), max_eval_samples)` (see `_safe_eval_mi` /
  `_eval_mi` and the ceiling-warning block in `training/trainer.py`). Every ceiling comparison
  in `'auto'` titration must use `log(eval_size)`. Add an acceptance test that moves the
  detached/pinned boundary by changing eval size and confirms batch size alone does not move
  it (spec test 7).
- **Do not route noise through the per-batch augmentation.** The dataset already has an
  `apply_noise(amplitude)` path driven by the trainer's `amplitude_x/amplitude_y`, which
  resamples every batch and is off at eval. The injected `sigma_add` noise is a **one-time**
  observation-space perturbation that becomes part of the dataset for both train and eval of a
  fit. Add it at dataset construction for the dimensionality path, independent of
  `apply_noise` (spec Section 1, "fixed realization"). A test must confirm it is present at
  eval and identical across epochs.
- **Independence per view** (spec Section 1): draw independent noise for X and Y. In intrinsic
  mode the two halves are disjoint channels so per-channel i.i.d. noise on the full tensor
  before the split is already independent; in interaction mode draw separate tensors.
- **Reproducible draw under parallelism** (spec Section 1): regenerate the base normal tensor
  `E` deterministically from `(global_seed, split_id)` inside each worker; `sigma_add` must
  not enter the seed, so every ladder rung scales the same `E`. Check how splits are currently
  seeded and how `ParameterSweep` distributes work (`analysis/sweep.py`, `analysis/task.py`)
  and thread the seed through so workers reconstruct `E` rather than pulling from a live RNG.

### 5.3 Variance stabilization (the one cross-spec resolution)
Implement a **single** canonical Anscombe / square-root stabilizer as a shared utility (e.g.
in `utils.py` or `data/`), and have the binned-spike dimensionality path call it with the
fixed order of operations from spec Section 3 (stabilize counts → measure per-channel std on
stabilized values → inject in stabilized-std units). The gate job does **not** build a
stabilizer, so there is no competing implementation to reconcile; this one is canonical for
any future use. Gate it behind the `stabilize_counts` toggle from Section 2 (default on;
off gives plain counts; warn if off while noise is injected).

### 5.4 Modality guards, output, logging, tests
Implement the processor guards (continuous supported; binned-spike supported with
stabilization; raw-spike and categorical rejected with specific errors), the full per-rung
output contract, the `'auto'` bracketing, the logging requirements, and all eight acceptance
tests exactly as the spec lists them (Sections 3-9). The categorical rejection here composes
with fix 3.3's label validation; keep the two error messages distinct and both clear.

---

## 6. Batch 1 — Gate job Phase 0 only (correctness, no benchmarking)

From `inductive_bias_gate_job.md`. Only Sections 0.1 and 0.2 belong in Batch 1; everything
gated by `RUN_PHASE2`/`RUN_PHASE3` or by Phase-1 outcomes is Batch 2.

- **0.1 Remove `calcium_cnn` entirely.** Follow the spec's deletion list. One interdependency
  the spec implies but is worth stating: remove only the **calcium branch** of the
  physics-params tracking (`physics_params_history` / `physics_params_final`), not the tracking
  framework itself, because surviving physics encoders (sinc, backbone) still use it. After
  removal, grep the whole repo for `calcium_cnn`, `learn_calcium_kernel`, `tau_rise`,
  `tau_decay`, `generate_windowed_calcium`, and `CalciumEmbedding`, and fix every dangling
  import, registry entry, doc row, and tutorial cell.
- **0.2 Fix `PretrainedBackboneEmbedding`** (keep the class). Remove the `torch.no_grad()`
  wrapper around the backbone forward so gradient reaches the channel adapter (the backbone is
  already frozen via `requires_grad=False`), and keep backbone BatchNorm in eval mode by
  overriding `train()`. Add the regression test the spec specifies (adapter gets non-zero grad
  when `input_dim != backbone_in_ch`; backbone params stay frozen and unchanged; backbone BN
  in eval during forward). This fix must land in Batch 1 because Batch 2's Phase 3 benchmarks
  are uninterpretable without it.

Do **not** touch `SincEmbedding`, `SpikePhysicsEmbedding`, or depthwise `CNN1D` in Batch 1
(spec 0.3); they are Batch 2, gated by Phase 1.

---

## 7. Batch 2 — Empirical embeddings gate

Run `inductive_bias_gate_job.md` Phases 1-3 in full, against the Batch-1 code. Nothing here is
pre-scriptable beyond the spec, because the code edits (Phase 2/3 keeps and removes, Section 5
generator cleanup) are contingent on the benchmark verdicts. Honor the flags (`RUN_PHASE2`,
`RUN_PHASE3`), the ceiling guard (`true_mi ≤ 3 bits` except the raised-batch image case), the
oracle-headroom logic, and the "regularize don't bias" decision rule. Deliver the artifacts in
the spec's Section 7, including a one-line decision-number per bias and an explicit call-out of
anything that could not be run (e.g. missing Allen preprocessing) rather than a silent skip.

Compute warning to the human before this batch: it trains across
regimes × encoders × N-grid × seeds with capacity/regularization sweeps; this is the expensive
part of the whole upgrade. Confirm the budget before launching.

---

## 8. Cross-cutting invariants and the propagation checklist

Apply on every change, and especially after the PR rename and the calcium removal:

1. After any rename or removal, grep the **entire** repo for the old symbol, including
   `tests/`, `tutorials/`, notebooks, and all markdown, and fix imports, columns, labels,
   docstrings, and prose. No dangling references.
2. Any generator or class whose only consumer was removed, and that no figure uses, is removed
   (gate Section 5 inventory). Produce the inventory before deleting.
3. Behavior-preserving items (3.2 docs, 3.4 messages, 3.5 comment) must not change any
   numerical output; guard with the existing tests.
4. New numerical behavior (noise feature, HybridCritic chunking) must be covered by tests that
   assert the invariant, not just that code runs: chunked==unchunked scores; noise fixed across
   epochs and present at eval; ceiling keys on `log(eval_size)`; independence produces no
   spurious shared dimension.
5. Run the full suite at the end of Batch 1 and require green before Batch 2 starts.

---

## 9. Questions to raise if you hit them
- If the `spectral_output` vocabulary reconciliation (Section 4) forces a user-facing value
  change beyond `pr_covariance→pr_eig`, surface the chosen vocabulary rather than guessing.
- If threading the `(global_seed, split_id)` noise seed through `ParameterSweep` requires an
  API change to the worker task, propose it rather than smuggling state through a global.
