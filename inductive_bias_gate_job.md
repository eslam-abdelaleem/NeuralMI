# Agent Job: Inductive-Bias Embeddings — Gate, Fix, or Remove

## 0. Objective and the decision rule

We have five physics/architecture-informed embeddings in `neural_mi/models/embeddings.py`:
depthwise-separable CNN (`cnn` + `use_depthwise`), `sinc_cnn`, `calcium_cnn`,
`spike_physics`, `pretrained_backbone`. In prior informal testing they did not beat
generic MLP/CNN encoders on MI estimation, and sometimes hurt. This job settles each one
through a single door:

> **An encoder bias earns its place only if it moves the MI-vs-N curve measurably closer
> to the true MI than a capacity-matched, regularized generic encoder, in a regime where a
> known oracle proves the headroom exists. Otherwise it is removed.**

Grounding principle (do not re-litigate, just apply it): by the data processing inequality
no encoder can raise the achievable MI. A bias can only (a) shift MI-vs-N leftward toward
the true value, or (b) if it is a lossy projection, lower the ceiling and bias the estimate
down. So we first measure how much leftward shift is even possible (the "oracle headroom"),
then ask whether each candidate captures a meaningful fraction of it without a plain
regularized generic encoder already capturing it for free.

The library is unpublished; there are no external users to preserve. Removing classes,
generators, params, and tests outright is fine and expected. The agent has access to the
full repo including tutorials and must update all of them consistently.

Work in three ordered phases. Phase 0 and Phase 1 always run. Phases 2 and 3 are gated by
flags and by Phase 1 outcomes.

---

## 1. Phase 0 — correctness actions (no experiment; do first)

These are decided on correctness, not performance. Do them before any benchmarking.

### 0.1 REMOVE `calcium_cnn` entirely
Reason: `CalciumEmbedding._deconv_kernel` builds `h_inv = flip(h)` normalized to unit L2.
That is the **matched filter** of the indicator response, which further low-passes the
signal. It is not a deconvolution (which would sharpen / invert the blur). The docstring and
the reference table claim "FIR deconvolution of the GCaMP impulse response," which the code
does not do. Independently, the only generator for it (`generate_windowed_calcium`) carries
its shared information in firing *rate*, for which mean fluorescence is already near-sufficient,
so even a correct deconvolution would buy nothing there. Cut it rather than fix it.

Delete:
- class `CalciumEmbedding` in `embeddings.py`
- `'calcium_cnn'` from the embedding-model registry/factory and from the `embedding_model`
  options list wherever validated
- params `tau_rise`, `tau_decay`, `learn_calcium_kernel` from the base-params reference and
  any injection logic
- the calcium branch of the physics-params tracking (`physics_params_history` /
  `physics_params_final`) in the trainer/results code
- `generate_windowed_calcium` in `synthetic.py`
- calcium tests: `test_calcium_physics_params_not_tracked_when_fixed`,
  `test_calcium_physics_params_tracked_when_learnable`, and any calcium cases in Tutorial 10
- calcium row in `NEURALMI_REFERENCE.md` Sections 3.3 and the class-reference table

### 0.2 FIX `PretrainedBackboneEmbedding` (keep the class; it goes to Phase 3)
Two correctness bugs that would make any prior null uninterpretable:

1. **Dead channel adapter.** In `forward`, `self._channel_adapt(x)` runs, then the backbone
   runs inside `with torch.no_grad():`. Gradient never reaches the adapter, so for any
   `input_dim != backbone_in_ch` the adapter is frozen at random init and never trains.
   Fix: remove the `torch.no_grad()` wrapper around the backbone forward. The backbone is
   already frozen via `requires_grad=False` on its params in `__init__`; freezing does not
   require `no_grad`, and `no_grad` additionally severs the adapter's gradient path. After
   the fix, gradient flows through the frozen backbone (params still not updated) to the
   trainable adapter and head.

2. **BatchNorm in train mode.** A frozen pretrained backbone with BatchNorm must stay in
   eval mode so running stats and normalization are fixed; otherwise the trainer's
   `.train()` call puts BN into batch-statistic mode and corrupts the "frozen features"
   claim. Fix: override `train()` on the module to keep `self.backbone` in `eval()`
   regardless (call `self.backbone.eval()` at the end of `train()`), or set it in `forward`.

Add a regression test asserting: (a) after a backward pass, `self._channel_adapt` has a
non-None, non-zero gradient when `input_dim != backbone_in_ch`; (b) backbone params remain
`requires_grad=False` and unchanged; (c) backbone BN layers are in eval mode during forward.

### 0.3 Keep for now, pending Phase 1
`SincEmbedding`, `SpikePhysicsEmbedding`, depthwise `CNN1D`. Do not modify them yet.
Do not build the side-channel fusion rewrite; it is only worth doing if Phase 1 shows
headroom (see Phase 2).

---

## 2. Phase 1 — the gate (always run)

### 2.1 Global training config (fixed across every run in Phase 1)
Use these constants for **every** encoder and every regime so the only thing that varies is
the input and the encoder architecture:

```
batch_size      = 256
n_epochs        = 200
patience        = 40
learning_rate   = 5e-4
optimizer       = 'adam'
critic_type     = 'separable'
estimator       = default InfoNCE
embedding_dim   = 32
hidden_dim      = 64
n_layers        = 2
split_mode      = 'random'      # IID synthetic
seeds           = [0, 1, 2, 3, 4]
N_grid          = [200, 500, 1000, 2000, 5000, 10000]
```

**InfoNCE ceiling guard.** Keep every generator's true MI at or below 3 bits so we never
sit near the `log2(batch_size)=8` bit ceiling (that regime is a separate, already-studied
problem and would confound this test). Assert `true_mi <= 3.0` for each regime before running.

**Capacity/regularization sweep for the generic baseline.** The honest competitor is not one
plain encoder but a small regularized sweep. For the generic MLP (and CNN where used), run
the cross product:
```
hidden_dim ∈ {32, 64}
dropout    ∈ {0.0, 0.1}      # via base_params 'dropout'
weight_decay ∈ {0.0, 1e-4}  # via base_params optimizer_params={'weight_decay': wd}
```
Report the **best** generic configuration per (regime, N) as the baseline to beat.

### 2.2 Call templates
Generic encoder:
```python
res = nmi.run(
    x_data=X, y_data=Y, mode='estimate', split_mode='random',
    processor_type_x=PROC, processor_type_y=PROC,
    processor_params_x=PPARAMS, processor_params_y=PPARAMS,
    base_params={
        'n_epochs': 200, 'patience': 40, 'batch_size': 256,
        'learning_rate': 5e-4, 'embedding_dim': 32,
        'hidden_dim': H, 'n_layers': 2, 'dropout': DO,
        'optimizer_params': {'weight_decay': WD},
        'embedding_model': 'mlp',   # or 'cnn' / 'gru'
    },
    random_seed=s, show_progress=False)
mi = res.mi_estimate
```
Oracle: identical call but `x_data=S_x, y_data=S_y`, `processor_type=None` (feed the
sufficient statistic as a plain `(N, d_S)` array), `embedding_model='mlp'`, `dropout=0`,
`weight_decay=0`. Same `hidden_dim=64, n_layers=2, embedding_dim=32`. The oracle is literally
the generic MLP fed the true sufficient statistic, so the only difference from the generic
baseline is the input.

### 2.3 Regimes (build S(X) exactly as specified)

For MI-vs-N: generate one large dataset per seed (N=10000), then evaluate on nested
subsamples from `N_grid` so curves are monotone in the same data.

**Regime A — smooth low-dim latent (prediction: near-zero headroom; the binned-population case).**
- Generator: `generate_nonlinear_from_latent(n_samples, latent_dim=5, observed_dim=100, mi=2.0, hidden_dim=64)`.
- Add observation noise the generator does not: `X = X + sigma * randn`, `Y = Y + sigma * randn`
  with `sigma` set so per-dim SNR ≈ 3 (compute from X std). This makes it realistic and makes
  the sufficient statistic the MMSE estimate of Z rather than an exact inverse.
- `S_x = z_x`, `S_y = z_y` (the true 5-dim latents; the generator exposes them internally,
  so add a `return_latents=True` path or regenerate with fixed seed to recover them).
- true_mi = 2.0 bits.
- Encoders: generic MLP (with sweep) on the 100-dim observable; oracle MLP on the 5-dim latent.
- Purpose: if oracle ≈ generic here, the encoder was never the bottleneck for binned
  population data and **all** candidate biases are cut for that regime.

**Regime B — single-band amplitude code (LFP-like; phase is nuisance).**
- Generator: `generate_windowed_oscillatory(n_windows, n_channels=4, window_size=128, f_carrier_hz=10.0, sample_rate=256.0, latent_mi=0.5, snr=3.0)` → true_mi returned (≈ 4 × per-channel; verify ≤ 3 bits, lower `latent_mi` or `n_channels` if not).
- `S_x[:,c] = <X[:,c,:], carrier> / ||carrier||^2` (matched-filter amplitude per channel);
  same for `S_y`. Carrier = `sin(2π·10·t)`, `t = arange(128)/256`.
- Encoders: generic MLP and generic CNN (`embedding_model='cnn'`, sweep) on raw windows;
  oracle MLP on the per-channel amplitudes.
- Purpose: does raw-LFP MI estimation leave headroom a matched filter would recover? If
  oracle ≈ best generic (CNN likely wins because a conv can approximate a matched filter),
  then `sinc_cnn` is dead without even building an interference generator (see Phase 2 gate).

**Regime C — rate-code spikes.**
- Generator: `generate_modulated_spike_trains(n_neurons=8, duration=..., baseline_rate=5.0, modulation_depth=0.7, modulation_freq=1.0)`; window into `processor_type='spike'` format. Set duration so N windows cover `N_grid`. Verify true_mi ≤ 3 bits (tune `modulation_depth`); if no analytic MI, compute a large-N CCA lower bound on per-neuron counts as the reference, and note it is a lower bound.
- `S_x[:,i] = spike count of neuron i in the window`; same for `S_y`.
- Encoders: generic MLP on the padded `(N, n_neurons, max_spikes)` tensor; oracle MLP on
  per-neuron counts.
- Purpose: does the padded-timestamp representation leave headroom that a count statistic
  recovers? Gates `spike_physics`.

**Regime D — timing-code spikes (rich sufficient statistic; oracle N/A, architecture is the only bias).**
- Generator: `generate_timing_code_spike_trains(n_neurons=8, duration=200.0, signal_rate=5.0, background_rate=15.0, delay=0.015, jitter=0.003)`; window into `spike` format. Reference MI: large-N estimate from the best available model (report as approximate).
- No oracle (S ≈ the full train). Compare generic MLP vs `gru` (`embedding_model='gru'`) on
  the padded raw tensor across `N_grid`.
- Purpose: this is the "known architecture is the bias" control. If GRU beats MLP here and
  `spike_physics` (tested in Phase 2) does not, the conclusion is "use a sequence model, not
  physics features," which is a clean paper sentence.

### 2.4 Metrics and thresholds
For each (regime, encoder), across seeds, at each N: record mean and std of `mi_estimate`.

- **Converged-N**: smallest N in the grid at which mean estimate is within
  `max(0.1 bit, 10% of true_mi)` of true_mi and stays within band for all larger N.
- **Headroom present** in a regime iff `convN(generic_best) >= 2 × convN(oracle)` with
  non-overlapping ±1 std bands at the discriminating N. (Oracle needs at least 2× fewer
  samples.)
- **"Regularize, don't bias" outcome**: if the regularized generic sweep reaches within
  `1.3 ×` of the oracle's converged-N, declare no room for an architectural bias in that
  regime and do not proceed to Phase 2 for the corresponding candidate.
- **Fraction of headroom recovered** by a candidate encoder E (Phase 2):
  `(convN(generic) − convN(E)) / (convN(generic) − convN(oracle))`, clipped to [0, 1].

### 2.5 Phase 1 deliverable
A results table `regime × encoder × N → (mi_mean, mi_std)`, plus a per-regime summary with
converged-N for generic-best / oracle / (GRU in D), the headroom verdict, and the
"regularize vs bias" verdict. Save raw arrays to `results/gate/` and a markdown summary.

---

## 3. Phase 2 — conditional survivors (flag `RUN_PHASE2`, and only where Phase 1 shows headroom)

For each candidate, run **only if** its gating regime showed headroom AND the regularized
generic did not already close it. Test each candidate in a **non-lossy** form (physics
feature available to the critic *in addition to* a generic path) so a null is interpretable;
if it still loses, remove it.

### 3.1 `spike_physics` — gated by Regime C headroom
- If C shows headroom: test `spike_physics` with `feature_fusion='concat'` (features plus raw
  timing available) on Regime C (favorable, rate code) and Regime D (control, timing code).
- SURVIVES iff: recovers ≥ 50% of Regime-C headroom **and** does not degrade below the
  generic MLP on Regime D. Otherwise REMOVE the class, the `'features'`/`'concat'` fusion
  code path, and `spike_physics` from the registry, params, tutorials, and reference.
- If C shows no headroom: REMOVE without further testing.

### 3.2 depthwise `CNN1D` — gated by Regime B (or a multichannel variant) headroom
- If B shows headroom: test `cnn` + `use_depthwise=True` vs plain `cnn` under matched
  capacity on `generate_windowed_multichannel(n_channels=8, ...)` (favorable: per-channel
  distinct carriers) with an N sweep, plus the single-shared-latent oscillatory case as the
  control where depthwise should show no advantage.
- SURVIVES iff: strictly fewer samples than plain CNN on multichannel **and** no advantage
  on the shared-latent control (advantage there would mean it is just parameter-count
  regularization, report as such). Otherwise keep `use_depthwise` in the library only if it
  helps, else REMOVE the flag and its code path.
- If B shows no headroom: REMOVE the depthwise path.

### 3.3 `sinc_cnn` — double-gated by Regime B headroom
`sinc_cnn` is only worth touching if Regime B shows headroom that a generic CNN does *not*
already capture. If so, it needs two rewrites before it can be tested fairly:
1. **Power pooling, not mean pooling.** Replace `features.mean(dim=-1)` readout with a band
   power readout (mean of squares, or `log(mean(square) + eps)`) so the band amplitude the
   filterbank exists to expose survives pooling. A bandpassed oscillation is ~zero-mean, so
   the current mean-pool discards most of the signal.
2. **A favorable generator that does not currently exist**: narrowband signal (e.g. 8 Hz)
   plus strong **out-of-band broadband interference** (e.g. 1/f noise or a 40 Hz distractor),
   so a bandpass front-end can structurally reject interference a short-kernel generic conv
   cannot. Build `generate_windowed_bandpower_interference(...)` for this and no other reason.
- SURVIVES iff: with power pooling, beats the best generic CNN on the interference regime by
  the headroom criterion, **and** the learned cutoffs migrate to the true signal band (assert
  the final `f_low_hz/f_high_hz` bracket 8 Hz, not the EEG init). Otherwise REMOVE `sinc_cnn`
  from the paper. Optionally retain in the library **only** as an interpretability readout if
  and only if the cutoff-recovery test passes; document it as interpretability, not accuracy.
- If B shows no headroom: REMOVE `sinc_cnn`, and do **not** build the interference generator.

---

## 4. Phase 3 — stimulus-side pretrained backbone (flag `RUN_PHASE3`; run regardless of Phase 1)

This is the one place the original MNIST insight transfers intact: on the **stimulus side**,
where one variable is natural images and an ImageNet backbone genuinely transfers. Do not
apply the backbone to the neural side.

### 4.1 Reproduce the MNIST result as an in-repo regression
- Build a class-identity image-pair generator (or use torchvision MNIST directly): each pair
  `(X_i, Y_i)` is two different instances of the same digit class. True MI = `log2(10) ≈ 3.32`
  bits (this exceeds the 3-bit guard; for this image experiment only, raise `batch_size` to
  1024 so the InfoNCE ceiling `log2(1024)=10` clears it).
- Compare three image encoders across `N_grid`: `pretrained_backbone` with `pretrained=True`
  (fixed per 0.2), `pretrained_backbone` with `pretrained=False` (**frozen-random control**,
  same capacity, isolates transfer from capacity reduction), and a from-scratch `cnn2d`.
- Expect: pretrained converges at far fewer samples than both. The random-frozen control is
  essential; if random-frozen also converges fast, the win is capacity/regularization, not
  transfer, and the claim must be softened accordingly.
- Replace `generate_noisy_image_pairs` (a Gaussian blob in noise is not ImageNet-like and does
  not test transfer) with this class-identity generator, or remove it if torchvision is used.

### 4.2 Allen stimulus-to-response MI
- Estimate `I(natural scene stimulus; VISp population response)` on the Allen data already
  loaded in the project. Image side: `pretrained_backbone` (pretrained=True) vs from-scratch
  `cnn2d` vs frozen-random control. Neural side: generic MLP/CNN, held fixed across the three.
- Success = pretrained image encoder reaches the same MI at fewer samples than from-scratch,
  with the random-frozen control ruling out the capacity confound.
- This is the positive result that reframes the whole post-mortem: a principled boundary
  ("weight priors buy sample efficiency on the stimulus side where a natural-image backbone
  transfers; they do not on the neural side"), rather than only a list of cuts.

---

## 5. Generator cleanup (apply after Phases 1–3 resolve)

Rule: a generator survives iff it feeds a surviving test/embedding or a paper figure.
Produce a generator inventory mapping each function → consumer(s) → keep/remove, then execute:

| Generator | Decision | Reason |
|---|---|---|
| `generate_windowed_calcium` | REMOVE | calcium cut in 0.1 |
| `generate_noisy_image_pairs` | REMOVE / REPLACE | blob-in-noise does not test transfer; replaced by class-identity generator in 4.1 |
| `generate_windowed_oscillatory` | KEEP | Regime B gate |
| `generate_modulated_spike_trains` | KEEP | Regime C gate |
| `generate_timing_code_spike_trains` | KEEP | Regime D control |
| `generate_nonlinear_from_latent`, `generate_correlated_gaussians`, `generate_linear_data` | KEEP | Regime A gate + Figure 2 |
| `generate_windowed_multichannel` | KEEP iff depthwise tested (Regime B headroom); else REMOVE | only consumer is depthwise Phase 2 |
| `generate_windowed_bandpower_interference` | CREATE iff sinc reaches Phase 2; else do not create | only consumer is sinc Phase 2 |

Any generator whose only consumer was a removed embedding and that is not used by a figure
gets removed. Report the final inventory.

---

## 6. Test-suite and docs cleanup
- Remove tests for cut classes (`test_physics_params_and_backbone.py`: keep and update the
  backbone tests per 0.2; drop the calcium tests).
- Update `NEURALMI_REFERENCE.md` Section 3.3, the class-reference table, the base-params
  reference, and Tutorial 10 to reflect exactly what remains.
- Grep the whole repo (`calcium_cnn`, `learn_calcium_kernel`, and any removed names) and
  remove dangling references so imports and docs stay consistent.

---

## 7. What to hand back
1. `results/gate/` raw arrays + a markdown summary table (regime × encoder × N, converged-N,
   headroom verdicts, "regularize vs bias" verdicts, fraction-of-headroom per candidate).
2. Phase 3 MI-vs-N curves for the three image encoders (MNIST reproduction + Allen), with the
   random-frozen control called out explicitly.
3. A one-page decision log: for each of the five biases, SURVIVE or REMOVE, and the single
   number (converged-N ratio or fraction-of-headroom) that decided it.
4. The cleaned `embeddings.py`, `synthetic.py`, test suite, reference doc, and tutorials.
5. Anything the agent could not run (e.g. missing Allen preprocessing) flagged rather than
   silently skipped.

## 8. Expected outcome (state as a prior to be tested, not assumed)
Near-zero headroom in Regime A (so all encoder biases are cut for binned population data).
Small or no headroom in B and C once the generic encoder is regularized (so `sinc_cnn`,
`spike_physics`, and depthwise likely do not survive, or survive only as minor
sample-efficiency notes). Architecture-as-bias (GRU) is the only neural-side win, in Regime D.
The real positive result is Phase 3 on the stimulus side. If the data contradict this prior,
follow the data; the gate is built so that "cut everything on the neural side" is a
first-class, reportable outcome, not a failure.
