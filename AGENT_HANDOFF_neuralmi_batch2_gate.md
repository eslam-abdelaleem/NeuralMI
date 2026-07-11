# NeuralMI Batch 2 — Empirical Embeddings Gate — Agent Handoff

## 0. How to read this document

You are picking up a fresh session with no memory of the Batch 1 work. This document is a
thin conductor layer on top of `inductive_bias_gate_job.md`, which remains **authoritative**
for the gate's mechanics (regimes, generators, thresholds, decision rules, Phase 2/3 details).
Do not re-litigate anything settled there. This document adds only: what Batch 1 already
did (so you don't redo it or get confused by renamed symbols), empirically-grounded compute
numbers gathered in the Batch 1 session, a recommended staged execution plan, and open
questions to raise with the human before certain sub-steps.

Required files in the repo for this session: `inductive_bias_gate_job.md` (authoritative),
this document, and the working `neural_mi` codebase with Batch 1 already merged. If any of
these are missing, stop and ask rather than guessing.

---

## 1. Batch 1 status (already done — do not redo)

Batch 1 (`AGENT_HANDOFF_neuralmi_upgrade.md`) is complete, tested, and pushed to the `paper`
branch on `git@github.com:eslam-abdelaleem/NeuralMI.git`, commits `402a6b4` → `66e5b5c` →
`4924f01` → `323be2e`. Full suite green at 527 passed / 1 skipped (CUDA unavailable) / 0
failed. Relevant to Batch 2:

- **Gate job Phase 0 is already done**: `CalciumEmbedding`/`calcium_cnn` is fully removed
  (class, registry, schema keys, tests, tutorials, docs). `PretrainedBackboneEmbedding`'s
  `no_grad`/BatchNorm bug is fixed (this matters for Phase 3 — those benchmarks are now
  interpretable). Do **not** re-open Phase 0.
- **Participation ratio was renamed**: `participation_ratio`/`pr_covariance` →
  `pr_eig`/`pr_singular` (both now always reported by `dimensionality` mode). If you read
  older notes, code comments, or your own training data referencing `participation_ratio`,
  translate mentally — the current codebase only has `pr_eig`/`pr_singular`.
  `SincEmbedding`, `SpikePhysicsEmbedding`, and depthwise `CNN1D` were **not** touched in
  Batch 1 (correctly deferred to you, per gate job Section 0.3).
  - **`dimensionality` mode gained a `sigma_add` ceiling-escape noise-injection feature.**
    Not directly needed for the gate (Phase 1-3 mostly use `mode='estimate'`/`'sweep'`), but
    if any regime's true MI ends up saturating the InfoNCE ceiling unexpectedly, this exists
    as an escape hatch — see `noise_injection_dimensionality_spec.md` and
    `NEURALMI_REFERENCE.md`'s dimensionality section.
- Several small correctness/DX fixes landed (see `CHANGELOG.md`'s `[Unreleased]` section,
  top entries) — none of them change gate-job semantics, just noting they exist.

---

## 2. Empirical compute grounding (measured in the Batch 1 session, this machine)

Environment: macOS, no CUDA, MPS available but untested for this workload (PyTorch MPS
support is inconsistent across ops — validate before relying on it, don't assume it's
faster). 12 CPU cores. Conda env `neuralmipaper` at `~/anaconda3/envs/neuralmipaper` is the
correct interpreter — it's an editable install pointing at this exact repo (confirm with
`~/anaconda3/envs/neuralmipaper/bin/python -c "import neural_mi; print(neural_mi.__file__)"`
before trusting it hasn't drifted). System `python3` does NOT have torch installed.

Measured with the **exact Phase 1 global config** (`n_epochs=200, patience=40,
batch_size=256, embedding_dim=32, hidden_dim=64, n_layers=2, lr=5e-4, adam`):

| Config | Result |
|---|---|
| MLP, N=1000, single run, CPU | 1.5s (early-stopped at epoch 87) |
| MLP, N=10000, single run, CPU | 5.5s (early-stopped at epoch 66) |
| MLP, 4 configs parallel, `n_workers=8`, N=2000 | 8.7s total (~2.2s/run amortized) |
| **GRU**, ~8000-window spike data (Regime-D-like), CPU | **239s** (early-stopped at epoch 45 of a possible 86) |

**Takeaway: MLP/CNN-based work (Regimes A, B, C) is cheap** — parallelizes cleanly, low
single-digit seconds per run even at N=10000. **GRU (Regime D only) is the real cost
driver** — roughly 150x slower than MLP at comparable scale. This changes the "confirm
budget" conversation from "hours" to something much more tractable; see Section 3.

**Operational gotcha discovered**: `n_workers > 1` uses `multiprocessing` with the `spawn`
start method, which re-imports `__main__` in child processes. This **requires a real `.py`
script file** — running training code via `python -c "..."` or a heredoc/stdin script
fails with `FileNotFoundError: ... <stdin>`. Always write sweep/gate scripts to actual files
before running them with `n_workers > 1`.

---

## 3. Recommended staged execution for Phase 1

Given the numbers above, Phase 1 as specified in `inductive_bias_gate_job.md` Section 2
(all 4 regimes, full `N_grid`, full capacity/regularization sweep, `seeds=[0,1,2,3,4]`) is
estimated at roughly **30-60 minutes total wall-clock**, not the "hours" a naive per-run
estimate would suggest — because the MLP/CNN regimes are fast and parallelize well, and
even Regime D's GRU cost (~30 runs × up to ~240s worst-case, likely less on average since
`N_grid` spans 200-10000) is bounded. This is a comfortable single background job.

**Suggested two-pass approach** (agree on this with the human before running, don't just
launch it):

1. **Validation pass — `seeds=[0]` only, full regimes/N_grid/configs.** Estimated well
   under 15 minutes. Purpose: confirm every generator, encoder, and metric path in
   `inductive_bias_gate_job.md` Section 2 actually runs end-to-end without errors, and that
   the `true_mi <= 3.0 bits` ceiling guard holds for each regime as specified (Section 2.1).
   Fix anything that breaks before spending the full seed budget.
2. **Full pass — `seeds=[0,1,2,3,4]`, full regimes/N_grid/configs.** Estimated 30-60
   minutes. This produces the actual deliverable table (Section 2.5): converged-N per
   encoder, headroom verdicts, "regularize vs bias" verdicts.

Run Regime D's GRU sweep with a **capped `n_workers`** (e.g. 4, not 8-12) — running many
GRU processes concurrently on a 12-core machine risks CPU contention that inflates per-run
time non-linearly (not directly measured, but worth guarding against given how much slower
GRU already is per-run).

Save raw arrays to `results/gate/` incrementally per regime as you go (per the gate job's
own Section 2.5 deliverable), not only at the very end — if something crashes partway
through the full pass, you don't want to lose completed regimes.

After Phase 1 completes and you have the headroom verdicts, **stop and report to the human**
before proceeding to Phase 2/3 — those are separately gated (`RUN_PHASE2`, `RUN_PHASE3`) and
their scope depends on what Phase 1 actually found, not on a plan made before seeing the
data.

---

## 4. Open question to raise before Phase 3.2 (Allen stimulus-to-response)

`inductive_bias_gate_job.md` Section 4.2 needs raw Allen Brain Observatory **stimulus**
frames (natural movie/scene pixels) to feed `pretrained_backbone`/`cnn2d`. The files present
in `tutorials/data/` (`nat_movie_visp_20.npy`, `nat_movie_ca1_20.npy`, etc.) are pre-extracted
**neural response** arrays only (shape `(60051, n_neurons)`, per `tutorials/raw
tutorials/7.py`) — not stimulus pixels. This is a real, currently-unmet data dependency, not
a bug to fix in code.

Before attempting 4.2: ask the human whether raw stimulus data is available (there is a
separate `allensdk` conda env on this machine, suggesting some prior Allen data workflow
exists somewhere — worth asking about specifically). If it's not readily available, run
Phase 3.1 (MNIST reproduction — no missing dependency, torchvision downloads it directly)
and flag 4.2 as blocked in the deliverable, per the gate job's own Section 7 instruction to
flag rather than silently skip missing pieces.

---

## 5. What to hand back

Follow `inductive_bias_gate_job.md` Section 7 exactly for deliverables. Additionally:
confirm with the human before spending compute on Phase 2 or Phase 3, and before treating
any encoder as "REMOVE" — those are code-deletion decisions on a shared branch, not
something to execute unilaterally even after the empirical verdict is clear (surface the
decision log, let the human confirm, then make the removal a separate, reviewable commit).
