# Behavior Spec: Ceiling-Escape via Observation-Space Noise Injection in `dimensionality` Mode

**Audience.** A coding agent implementing this in NeuralMI, after this spec has been consolidated with other pending library fixes. The agent has library source access; this document specifies behavior and invariants, not signatures.

**Purpose.** When the true MI between the two views exceeds the InfoNCE ceiling, the estimator saturates and the learned cross-covariance spectrum becomes unreliable for the participation-ratio (PR) readout. This feature lets `dimensionality` mode add a controlled amount of independent observation-space noise to lower the MI below the ceiling while leaving the saturation dimension unchanged, then read the dimension in that well-behaved regime. The theory, including why the saturation point is preserved and why the readout is corrupted at the ceiling, is in `dim_estimation_theory.md`, Sections 5.5 and 7.8. This spec operationalizes that theory.

**One-line summary of the change.** `dimensionality` mode gains an optional `sigma_add` control. With it unset, the mode behaves exactly as today. With it set, the mode injects fixed, independent, per-channel Gaussian noise into the observations once, in units of measured per-channel standard deviation, and can sweep a ladder of noise levels to locate the regime where the MI has detached from the ceiling.

---

## 1. The core mechanism and its invariants (no latitude)

These four properties are what make the correction valid. Each has a specific failure mode if implemented differently, noted inline.

**Observation space, before embedding.** The noise is added to the raw observations that the processor produces, before the embedding network sees them, and it stays part of the data for the rest of the run. It is not added in embedding space and not added inside the critic. The embedding whitening the mode already does for the spectral readout is a separate, downstream operation and is unchanged by this feature.

**Independent per view.** The added noise on the X-half and the Y-half must be independent draws. Independence is the entire reason the noise cancels from the cross-covariance and only lowers the canonical correlations rather than moving the saturation dimension. Implementation: add per-channel i.i.d. Gaussian to the full observation tensor before the channel split. In intrinsic mode this guarantees independence automatically, since the two halves are disjoint channel sets and each channel carries its own independent noise. In interaction mode (both `x` and `y` provided) draw independent noise tensors for `x` and `y`. Failure mode if a single draw is shared across the two views: the cross-noise no longer cancels, and the injected noise creates or distorts shared structure.

**Fixed realization, not resampled per batch.** The noise is drawn once and becomes part of the dataset for both training and evaluation of a given fit. It must not route through the existing per-batch `gaussian_noise` augmentation, which resamples every batch and is disabled at eval. Failure mode if resampled per batch: the critic averages the noise away across epochs and learns a partially denoised representation, which silently under-noises and defeats the ceiling escape.

**Scaled to measured per-channel standard deviation.** The theory's noise level is in units of the observations, which are unit-variance there but are counts, rates, or other native units on real data. A raw noise level is therefore not portable across datasets. The portable unit is per-channel standard deviation measured from the data. See Section 4 for the exact meaning and the interaction with variance stabilization.

**Reproducibility structure of the draw (critical under parallelism).** The base noise for a given split must be regenerated deterministically from `(global_seed, split_id)`, not pulled from a shared live RNG. Concretely:

- For each split, draw one base standard-normal tensor `E` of the observation shape, seeded by `(global_seed, split_id)`. `sigma_add` must NOT enter this seed.
- For a noise level `s`, the injected noise is `s * per_channel_std * E`. The same `E` is scaled by every level on the ladder, so adjacent ladder points differ only by the scale, not by an independent draw. This removes draw-to-draw variance between rungs and makes the ladder clean and monotone.
- Across the `n_splits` splits, `E` is redrawn (different `split_id`), which is what produces proper mean and standard deviation across splits in the output.

Because levels within a split share one `E`, and because runs are parallelized across workers (Section 5), each worker must reconstruct `E` deterministically from `(global_seed, split_id)` rather than draw it live. Failure mode if drawn from a live shared RNG: the same-`E`-across-levels property breaks under parallel execution and the ladder becomes noisy and non-reproducible.

---

## 2. What "dimensionality is unaffected by standardization" means for the readout

Per-channel standardization of the observations is a diagonal invertible linear map, so it leaves the cross-covariance rank, and therefore the saturation dimension `d`, exactly unchanged. Implement per-channel scaling freely. There is one consequence to document, not to fix: on real data the standardized signal spectrum is mildly anisotropic even before any nonlinearity, because the signal directions are only exactly isotropic in the model's raw orthonormal coordinates. Mild anisotropy pulls `pr_eig` slightly below the true rank (this is the population effect in Section 7.8, equation 25e). This is expected and is the reason both PR variants are reported rather than one. No code action beyond reporting both.

---

## 3. Modality handling: which processors are supported

The added noise must be an independent additive perturbation in the space the embedding consumes, one that preserves the cross-covariance rank and only lowers the canonical correlations. That criterion cleanly partitions the four processors.

**Continuous: supported directly.** Add per-channel Gaussian in measured-std units. No special handling.

**Binned spikes: supported, with automatic variance stabilization.** Binned counts are heteroscedastic (count variance grows with rate), so plain additive Gaussian would be swamped in high-rate channels and dominant in low-rate ones, and the per-channel-std unit would not be portable across channels. Therefore, when mode is `dimensionality` and the processor is binned spikes, apply an Anscombe (or square-root) variance-stabilizing transform to the counts automatically, before scaling and injection. This is automatic, not a toggle, and it fires on every binned-spike dimensionality run, including the plain no-noise (`sigma_add` unset) path, so that the mode's behavior on this processor is uniform whether or not noise is added. It must be logged (Section 7). Order of operations is fixed and is the single most error-prone part of this feature:

1. Stabilize the raw counts (Anscombe / square-root).
2. Measure per-channel standard deviation on the stabilized values.
3. Inject `sigma_add` in those stabilized-per-channel-std units.

Measuring the scale on raw counts and injecting on the stabilized scale, or any other permutation of these three steps, detaches the units and destroys portability. The logged line must state that `sigma_add` is in units of stabilized per-channel standard deviation.

**Raw spikes (timestamps): not supported.** The information is in spike timing, and additive observation noise there is not a perturbation that maps to lowering a canonical correlation; jittering timing changes precision content, which is a different axis (and the substrate of `precision` mode). Reject with a clear error that directs the user to bin the spikes first, which converts the problem to the supported binned case.

**Categorical: not supported.** A label has no metric, so "add noise" can only mean label flipping, which alters which off-diagonal cross-covariance mass exists rather than merely scaling it, breaking the rank-preservation argument and the guarantee that the saturation point is fixed. Reject with a clear error stating that observation-space noise injection is undefined for categorical data.

---

## 4. `sigma_add`: units and accepted forms

Keep everything under `dimensionality` mode. With `sigma_add` unset (the default), behavior is identical to the current mode.

**Units must be unambiguous in every branch.** The primary, portable unit is a multiple of measured per-channel standard deviation (for binned spikes, stabilized per-channel std per Section 3). A companion mechanism (for example a `sigma_add_units` parameter) selects between:

- `relative` (default and recommended): `sigma_add` is a multiple of measured per-channel std. This is the portable form and the one `auto` uses.
- `absolute`: `sigma_add` is the actual noise standard deviation in the data's native units, an expert override for users who know their units.

Whichever branch is taken, the resolved absolute per-channel noise scale must be logged so downstream numbers are never misread.

**Accepted forms for `sigma_add`:**

- A scalar: inject that single level, run the mode as usual on the noised data, and return the vanilla output plus the resolved noise scale.
- A list or explicit range: run the full ladder of levels and return one result row per level (Section 6).
- `'auto'`: infer a reasonable ladder and locate the detached regime (Section 5).

---

## 5. `'auto'` behavior

The ceiling in this library is `log(eval_size)` in nats, where `eval_size` is the number of samples in the InfoNCE evaluation denominator. It is NOT `log(batch_size)`. Every ceiling comparison and any ceiling-derived quantity in this feature must use `log(eval_size)`. (Note for the consolidating chat: the theory document writes the ceiling as `log(N_batch)`; substitute `log(eval_size)` when porting any formula.)

`'auto'` runs a geometric ladder over `relative` units spanning roughly `0.25` to `5` times measured per-channel std, parallelized across `n_workers` since the rungs are independent fits. Running the full geometric grid in parallel is preferred over serial early-stopping, because it uses the parallelism already present and yields the whole picture at once; the intelligence is in classification and bracketing, not in stopping early. At each rung, classify the estimated MI (the saturated MI at the large `k_z` used by the mode, aggregated over splits) into one of three regimes relative to the ceiling `C = log(eval_size)`:

- Pinned: MI within a small margin of `C`. Under-noised; the true regime is above this rung.
- Collapsed: MI at or below a small floor near zero. Over-noised; signal has sunk into the noise floor.
- Detached: MI in the open interval between the floor and `C` minus the margin. This is the target band.

Use a ceiling margin of roughly 0.5 to 1 nat (the same `c` as in the theory's equation 32a) and a small MI floor (a few percent of `C`, or a small absolute nat value; leave the exact value tunable). If the initial grid does not bracket the detached band (all rungs pinned, or all collapsed), widen or shift the grid once and, if it still does not bracket, return what was found and emit a warning rather than looping.

The suggested operating level is the geometric midpoint of the contiguous detached band, returned as a suggestion, never as a silent overwrite of the estimate (Section 6). Do not attempt automatic plateau detection beyond this; the final read-off is the user's, and the output must surface enough for them to make it.

---

## 6. Output contract

Every run, whether a single level or a ladder, returns the full set of rungs with their parameters, so nothing is hidden. Per rung, report at minimum:

- The noise level in both forms: the `relative` multiple and the resolved `absolute` per-channel scale.
- MI mean and standard deviation across splits.
- `pr_eig` mean and standard deviation (the eigenvalue / covariance-spectrum PR).
- `pr_singular` mean and standard deviation (the singular-spectrum PR).
- The permutation p-value, only if permutation significance is enabled (Section 8); otherwise omit it, do not fabricate it.
- The regime label (pinned, detached, collapsed) and a boolean detached flag.

Preserve the existing per-split raw rows as the mode does today. Add a top-level suggestion field carrying the suggested operating level and its regime, clearly marked as a suggestion. The suggestion never replaces or reorders the reported estimates.

**Plotting.** Propagate a plot option consistent with the mode's existing plotting. The useful plot for this workflow is `d_hat` from both PR variants against `log(sigma_add)`, with the detached band shaded, which renders the plateau-flanked-by-two-ramps picture from the theory. Optionally overlay MI versus `sigma_add`. This is not the per-`k_z` MI curve.

---

## 7. Logging requirements

The following must be logged (not silent, not merely returned):

- When binned-spike variance stabilization fires: state that it fired, the transform used, that it applies to this dimensionality run regardless of noise, and that `sigma_add` is henceforth in units of stabilized per-channel standard deviation.
- When `sigma_add` is engaged: state the resolved absolute per-channel noise scale and the units branch (`relative` or `absolute`) that produced it.
- When `'auto'` fails to bracket the detached band after widening: warn, and state what was found.

---

## 8. Preconditions, scope boundaries, and explicit non-goals

**InfoNCE assumption.** The titration and the detached-flag logic key on the `log(eval_size)` ceiling, which is specific to InfoNCE. If the calibration is engaged with a non-InfoNCE estimator (for example SMILE), proceed but emit a clear warning that the ceiling-based classification is calibrated for InfoNCE and that other estimators have a different, non-constant bias, so the `auto` bracketing and detached flags may not be meaningful. Warn, do not block.

**Lower edge / gap check: out of scope.** Do not build an analytic bulk-edge (`kappa'`) check for the lower detectability edge. Runs are assumed to be comfortably above it, and the existing permutation mode already provides a per-rung significance value that stands in for a gap check when enabled. No new significance machinery.

**Edge-count estimator: out of scope for now.** Do not add the edge-thresholded rank count as a reported estimator. It is contingent on permutations being enabled, and a reported estimator that appears and disappears with an unrelated flag is undesirable. Reported dimensionality estimators remain `pr_eig` and `pr_singular` only.

**PR rename (breaking, intended).** Rename the two participation-ratio outputs everywhere in the library to `pr_eig` (the eigenvalue / covariance-spectrum variant, `(sum s_i^2)^2 / sum s_i^4`) and `pr_singular` (the singular-spectrum variant, `(sum s_i)^2 / sum s_i^2`). The library is pre-publication; do not keep deprecated aliases. Update all call sites, column names, plot labels, and docstrings.

---

## 9. Acceptance tests

All on a synthetic linear-Gaussian two-view model with known latent dimension `d`, unless noted. These map one-to-one onto the theory and are hard pass/fail criteria.

1. **Monotone MI reduction.** Measured MI decreases monotonically as `sigma_add` increases along the ladder.
2. **Plateau in the interior.** `pr_eig` stays near `d` across the detached band, and the detached band appears once and only once MI drops below the ceiling.
3. **Over-noising inflation.** As `sigma_add` is pushed into the collapsed regime, `d_hat` inflates toward `k_z` as signal sinks into the noise floor.
4. **Independence preserved.** Injecting pure noise into two views with no shared latent does not create a spurious shared dimension (the cross-covariance signal rank stays at zero).
5. **Reproducibility.** A fixed `(global_seed, split_id)` reproduces identical results, and the same base `E` is reused across every level within a split, verified by the ladder being noise-free between adjacent levels up to the scale factor. This must hold under `n_workers > 1`.
6. **Modality guards.** Raw-spike and categorical inputs raise clear, specific errors. A binned-spike dimensionality run triggers the logged stabilization on both the no-noise and the noised paths.
7. **Ceiling source.** The ceiling used in classification equals `log(eval_size)`, verified by changing eval size and confirming the detached/pinned boundary moves accordingly, and that batch size alone does not move it.
8. **Estimator precondition.** Engaging calibration with a non-InfoNCE estimator emits the warning and still runs.

---

## 10. Open dependency for the consolidating chat

The binned-spike variance stabilization here (automatic Anscombe on the dimensionality path) overlaps with the variance-stabilization work in the embeddings fix workstream. Coordinate so a single stabilization implementation is shared rather than two colliding ones. If that workstream lands a canonical stabilizer, this feature should call it rather than reimplement it, keeping the order-of-operations guarantee from Section 3.
