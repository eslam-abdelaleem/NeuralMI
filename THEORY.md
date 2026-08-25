# Theoretical Foundations of NeuralMI

This document provides a concise theoretical background for the core methods used in the `NeuralMI` library. It is intended as a formal reference for users who wish to understand the mathematical principles behind the code.

## 1. The Challenge of Estimating Mutual Information

Mutual Information (MI) is formally defined as the Kullback-Leibler (KL) divergence between the joint distribution $p(x, y)$ and the product of the marginal distributions $p(x)p(y)$:

$$
I(X; Y) = \int p(x, y) \log \frac{p(x, y)}{p(x)p(y)} \, dx \, dy
$$

Calculating this directly requires knowing these probability distributions. For high-dimensional and continuous data, like that often found in neuroscience, these distributions are unknown and practically impossible to estimate accurately. Traditional methods like binning or kernel density estimation fail due to the "curse of dimensionality."

To overcome this, `NeuralMI` uses a modern approach called **neural estimation**, which reframes MI estimation as a neural network optimization problem.

---

## 2. Neural MI Estimators: A Bias-Variance Trade-off

Instead of estimating the probability densities, we can use a neural network, called a **critic** $f(x, y)$, to help us estimate a lower bound on the true MI. The core idea is to train the critic to distinguish between "positive" samples (pairs `(x_i, y_i)` that genuinely occurred together) and "negative" samples (pairs `(x_i, y_j)` from the same batch that did not).

Different mathematical formulations, or estimators, can be used for this task. They represent a trade-off between the **bias** of the estimate (how far it is from the true value on average) and its **variance** (how much it fluctuates on different runs). As argued in recent literature, choosing the right estimator depends on the scientific question.

:::{admonition} References
:class: note

- "Understanding the Limitations of Variational Mutual Information Estimators" (ICLR 2020)
- "On Variational Bounds of Mutual Information" (PMLR 2019)
- "Accurate Estimation of Mutual Information in High Dimensional Data" (ArXiv 2025)
:::

`NeuralMI` focuses on two particularly effective estimators that cover the most common use cases.

### 2.1 InfoNCE (Low Variance, High Bias)

The **InfoNCE** (Noise-Contrastive Estimation) estimator is the workhorse of `NeuralMI`. Its formula is:

$$
I(X;Y) \ge \mathbb{E}\left[ f(x,y) - \log\left(\frac{1}{N}\sum_{j=1}^N e^{f(x,y_j)}\right) \right]
$$

**Intuition:** For each positive pair `(x, y)`, the critic `f(x,y)` tries to maximize its score relative to the scores of `N-1` negative pairs `(x, y_j)`. This is effectively a classification problem where the model tries to pick the "real" partner `y` for a given `x` out of a lineup of `N` candidates.

**Properties:**

- **Low Variance:** InfoNCE is known to be a very stable estimator, producing consistent results across different random seeds.
- **Biased:** It is a lower bound on the true MI. Crucially, this bound is **theoretically limited by $\log(N)$**, where $N$ is the size of the score matrix the bound is evaluated on. This means InfoNCE can never report an MI value higher than $\log(N)$. Its stability is a major advantage, which is why it's the default in `NeuralMI` — but the ceiling is not automatically safe to ignore, and it does not scale with `max_eval_samples`. At **training** time, $N$ is `batch_size`. At **evaluation** time — where `train_mi` and `test_mi` are actually computed — $N$ is `train_eval_size` and `eval_size` respectively (see §9), each capped by `max_eval_samples` but *also* by how many samples the train/test split actually produced. A short recording, a large `window_size` (fewer windows fit), or a small `train_fraction`/`n_test_blocks` choice can all shrink $N$ well below `max_eval_samples` — and because temporal MI is *extensive* (block MI over a `window_size`-sample window grows with `window_size`, not per-sample), the true value you're trying to estimate can grow faster than the ceiling does as you sweep `window_size`. Check `result.details['eval_size']` / `['train_eval_size']` and their derived `['test_ceiling_mi']` / `['train_ceiling_mi']` / `['test_saturation']` / `['train_saturation']` rather than assuming the ceiling rarely binds.

### 2.2 SMILE (Low Bias, Moderate Variance)

The **SMILE** (Smoothed Mutual Information Lower-bound Estimator) is designed to provide a less biased estimate, which is critical in scenarios where the true MI might be high.

$$
I(X;Y) \ge \mathbb{E}\left[ f(x,y) \right] - \log \mathbb{E}\left[ e^{\text{clip}(f(x,y'), \tau)} \right]
$$

**Intuition:** SMILE is similar to other classical estimators—like MINE (Mutual Information Neural Estimator)—but it introduces a clipping function on the normalization factor. By clipping the scores at a value $\tau$, it prevents a few "easy" samples from dominating the loss function, which is a major source of bias.

**Properties:**

- **Low Bias:** By mitigating the impact of easy negatives, SMILE can provide estimates that are much closer to the true MI, especially when the MI is high. It is not strictly bounded by $\log(N)$ in the same way as InfoNCE.
- **Moderate Variance:** This reduction in bias comes at the cost of slightly higher variance compared to InfoNCE.
- **The `clip` parameter ($\tau$):** A clipping value of $\tau=5$ is often a robust default choice.

:::{admonition} Recommendation
:class: tip

Use **InfoNCE** for general-purpose, stable MI estimation. Use **SMILE** when the true MI may be high relative to your evaluation set size and you need a less biased estimator.
:::

---

## 3. The Variational Approach

Standard neural estimators learn a single embedding vector, $z = g(x)$, for each input. A variational approach, in contrast, learns a posterior distribution over the embeddings, $q(z|x)$. This is typically a Gaussian distribution parameterized by a mean and a variance vector, $(\mu_x, \sigma_x) = g(x)$.

The total loss function is modified to include a KL divergence term that acts as a regularizer, encouraging the learned posterior distributions to be close to a prior (usually a standard normal distribution):

$$
\mathcal{L} = \left(D_{\text{KL}}(p(z_x|x) \| q(z_x)) + D_{\text{KL}}(p(z_y|y) \| p(z_y))\right) - \beta \, \hat{I}_{\text{estimator}}(Z_X;Z_Y)
$$

This regularization can improve the quality of the learned representations and lead to more stable and robust MI estimates, particularly in complex, high-dimensional settings.

**Implementation note:** In NeuralMI, variational training is enabled by setting
`use_variational=True` in `base_params`.  Internally, a `VariationalWrapper` is
placed *on top of* the chosen base encoder: the base encoder first maps the input
to a deterministic embedding of shape `(batch, embed_dim)`, and the wrapper then
applies two linear heads (μ and log σ²) plus the reparameterization trick.  This
design generalises the variational approach to **all** embedding architectures —
MLP, CNN, GRU, LSTM, TCN, and Transformer — without requiring a separate
architecture variant for each.

**Choosing $\beta$:** The default value of $\beta = 1024$ reflects the typical use-case where MI maximisation should strongly dominate over KL regularisation. With this setting the loss is effectively $\mathcal{L} \approx -1024\,\hat{I}$, which drives the embeddings to extract maximal shared information while the KL term still gently penalises degenerate distributions. Decreasing $\beta$ increases the relative influence of the KL prior; setting $\beta \ll 1$ can collapse the embeddings toward the prior and reduce estimated MI.

> **Implementation note (normalization):** Internally, the `VariationalWrapper` returns the KL divergence already normalized per sample (i.e., $\frac{1}{B}\sum_{i=1}^{B} D_\text{KL}^{(i)}$). The Trainer then computes $\mathcal{L} = \overline{D}_\text{KL} - \beta\,\hat{I}$ directly. As a result, $\beta$ has a direct and stable interpretation across different batch sizes: a tenfold change in $\beta$ always produces a tenfold change in the relative weight of MI, regardless of how many samples are in the batch.

---

## 4. The Problem of Finite-Sampling Bias

Even with a perfect estimator, any analysis performed on a finite dataset of $N$ samples will be biased. Two effects compete: the classical limited-sampling bias, in which finite samples manufacture spurious dependence and *inflate* the estimate, and the opposing tendency of a variational *lower-bound* estimator (see §2.1) to *under*-shoot when its critic is trained on little data. Which one dominates is regime-dependent, but in either case the deviation is **systematic in the sample size** — and that is what makes it correctable.

Theoretically, for a large number of samples $N$, this bias has a clear relationship with the sample size:

$$
I_{\text{estimated}}(N) \approx I_{\text{true}} + \frac{a}{N} + O\left(\frac{1}{N^2}\right)
$$

This means the estimated MI is approximately linear in $1/N$. This is the key insight that `NeuralMI` uses to correct for the bias.

---

## 5. The Solution: Rigorous Bias Correction

The `mode='rigorous'` in `NeuralMI` automates a principled, multi-step workflow based on this theoretical relationship:

1. **Subsampling:** The library repeatedly runs the MI estimation on different fractions of the data. For example, it might split the data into $\gamma=2$ halves, then $\gamma=3$ thirds, and so on.

2. **Fitting:** Substituting $N_{\text{chunk}} = N/\gamma$ into the bias formula gives $I_{\text{estimated}} \approx I_{\text{true}} + \frac{a}{N}\,\gamma$, so the estimated MI is **linear in $\gamma$** (the number of subsets). The library fits a weighted linear regression of MI vs. $\gamma$ to these points.

3. **Extrapolation:** It extrapolates the fitted line back to $\gamma = 0$, which corresponds to using the entire dataset as a single chunk ($N_{\text{chunk}} \to \infty$, $1/N \to 0$). The y-intercept at $\gamma = 0$ is the final, bias-corrected MI estimate. The confidence interval of this intercept provides the error bars.

This procedure effectively subtracts the bias that is dependent on sample size, yielding a more accurate and scientifically rigorous result.

### Quadratic Curvature Filtering

In practice, the MI-vs-$\gamma$ relationship is only approximately linear; at very large $\gamma$ (very small chunk sizes), finite-sample effects and network under-fitting introduce measurable curvature. `NeuralMI` applies an automatic **quadratic curvature filter**: it fits a quadratic polynomial to the MI-vs-$\gamma$ curve and excludes any $\gamma$ point whose estimated quadratic coefficient exceeds the `delta_threshold` parameter (default 0.1). Only the remaining approximately-linear points are used for the final regression. A minimum of `min_gamma_points` (default 5) such points must survive for the estimate to be considered reliable; if fewer remain the result is flagged as unreliable.

:::{admonition} References
:class: note

- "Estimation of mutual information for real-valued data with error bars and controlled bias" (PRE 2019)
- "Accurate Estimation of Mutual Information in High Dimensional Data" (ArXiv 2025)
:::

---

## 6. Cross-Run-Stable Directions of Shared Structure (`mode='dimensionality'`)

A nonlinear encoder given more embedding capacity than the number of genuinely shared latent factors between two views does not only find those factors — it can also construct new combinations of them (products, higher-order mixtures) that are, once trained, spectrally indistinguishable from independent factors. Every measure computable from a single trained embedding's spectrum — participation ratio, eigengap, singular-value profile — is blind to this distinction: a genuine shared factor and an encoder-constructed combination look identical to all of them. There is consequently no general way to read an exact count of "the" dimensionality off a single trained spectrum, and `mode='dimensionality'` does not attempt to: `result.mi_estimate` stays `None` for this mode, and no field claims to be a dimensionality count.

What the mode reports instead is three narrower, complementary pieces of evidence.

### A cheap regime read, before any training

`compute_regime_diagnostic` centers each view's raw channel data, computes the eigenvalues of its within-view channel correlation matrix, and looks at the ratio between consecutive eigenvalues. An isolated large ratio marks a **separable-like** regime — each channel driven mostly by a single underlying factor. A flat, gradual ratio curve marks an **entangled-like** regime — every channel reflecting several factors jointly (mixed selectivity). This runs once, with no training, and is attached to the output as `regime_x`/`regime_y` context; it does not gate or discount anything else the mode reports. The default threshold (peak ratio ≥ 3.0) is a rough heuristic calibrated on two validated example cases (~14–16× for a clean separable case, ~1.7× for a genuinely entangled one) — treat it as a guide, not a precise cutoff; `peak_val` is always returned alongside the label so borderline cases can be judged directly.

### Cross-run-stable directions

NeuralMI uses a **Hybrid Critic**: it embeds $X$ and $Y$ independently, then processes their concatenation through a final MLP decision head, avoiding the rigid geometry a dot-product critic would impose. The default embedding size is modest ($k_z = 8$, user-overridable via `Model(embedding_dim=...)`) — over-provisioning capacity is exactly what lets encoder-constructed combinations masquerade as genuine factors, so the default deliberately does not chase a large bottleneck.

Before computing the cross-covariance, embeddings are whitened (`spectral_whitening='std'` by default), standardising each dimension by its empirical standard deviation so that accidentally large variance in one dimension doesn't dominate the spectrum:

$$\tilde{Z}_{X,i} = \frac{Z_{X,i}}{\text{std}(Z_{X,i})}, \qquad \tilde{Z}_{Y,i} = \frac{Z_{Y,i}}{\text{std}(Z_{Y,i})}$$

The cross-covariance of the whitened, held-out test embeddings,

$$C_{XY} = \frac{1}{N-1} (\tilde{Z}_X - \bar{\tilde{Z}}_X)^T (\tilde{Z}_Y - \bar{\tilde{Z}}_Y)$$

is decomposed by SVD into a rotation (dimension 0 captures the most shared variance, dimension 1 the next most, and so on) and a spectrum of singular values $\sigma_i$. A single such spectrum still can't distinguish genuine factors from encoder-constructed combinations, so instead of reading it once, the whole fit is independently repeated `n_splits` times (default 3), and each rank's rotated direction is compared *across* repeats, on held-out data only:

* **Stability** — a rank is reported as stable only if its direction reproduces closely across repeats (minimum pairwise correlation over every pair of repeats ≥ `stability_threshold`, default 0.7). An encoder-constructed combination is a training-specific artifact of one fit and is not expected to reproduce this way.
* **Noise floor** — independently of the correlation check, a rank's mean singular-value strength must clear `min_strength_fraction` (default 0.05) of the top rank's strength. This catches a failure mode the correlation check alone misses: a pure-noise channel can show a spuriously high cross-run correlation by chance despite carrying essentially no shared signal, and the strength floor excludes it anyway.
* **Near-degeneracy** — adjacent ranks whose strength ratio is under `degeneracy_ratio_threshold` (default 1.3) are too close to individually order. They're reported as a group — existence confirmed, individual identity not claimed — rather than an arbitrary ranking.

The result is `stable_directions` (individually-trustworthy ranks), `stable_but_degenerate_groups` (trustworthy as a set only), and `n_stable_total`: always a lower bound on the number of genuine shared directions, never an exact count, and never inflated by capacity artifacts that a single fit can't rule out.

### Convergence gating

None of the above means anything if a fit hasn't actually converged — an under-trained embedding's spectrum reflects an incomplete optimization, not shared structure. Each independent repeat is checked (`best_epoch` short of the final training epoch); `result.details['converged']` is `True` only if every repeat converged, and a repeat that didn't is flagged rather than silently folded into the stability report.

### Participation ratio — kept as a secondary, non-headline diagnostic

Each per-split row of `result.dataframe` still reports two participation-ratio variants computed from that split's own spectrum:

$$PR_{\text{singular}} = \frac{\left(\sum_i \sigma_i\right)^2}{\sum_i \sigma_i^2}, \qquad PR_{\text{covariance}} = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4}$$

Both describe the effective spread of that one spectrum — $PR_{\text{covariance}}$ weights by eigenvalue $\lambda_i = \sigma_i^2$ and so is more sensitive to the true rank of the representation than $PR_{\text{singular}}$, which weights linearly by $\sigma_i$. They remain useful, free-to-compute descriptions of how concentrated or spread out a given trained spectrum is — but, per the opening of this section, a single spectrum's PR cannot itself distinguish genuine shared factors from encoder-constructed combinations, so it is reported per-split as context alongside the cross-run stability report, not as the mode's answer.

### Ceiling proximity

The InfoNCE-family critic this mode trains shares the same $\log(N)$-style ceiling described in §2.1, evaluated here on the held-out set actually used to compute `test_mi`: $\log(\text{eval\_size})$, where `eval_size = min(test_size, max_eval_samples)`. When the underlying MI estimate sits close to that ceiling (`ceiling_mi_fraction`, default 0.85), `mode='dimensionality'` emits a warning, since any spectral reading built on a saturated estimate deserves extra scrutiny. This is informational only, not a remediation: convergence gating already excludes the large majority of near-ceiling cases in practice (an estimate rarely reaches the ceiling without also still being mid-training), and a genuinely converged run that is near-ceiling degrades the stable-direction count conservatively — fewer directions reported, near-degeneracy flags absorbing the ambiguity — rather than misleadingly.

### Interaction vs. intrinsic

* **Interaction** (`y_data` provided): the cross-run-stable-directions analysis above is applied directly between $X$ and $Y$; each repeat is an independent weight initialisation on the same data.
* **Intrinsic** (`y_data=None`): NeuralMI splits a single dataset into two non-overlapping halves (randomly across channels/neurons by default, or spatially/temporally) and applies the same analysis between the halves; each repeat uses a different split, not just a different initialisation — a related but distinct notion of "stability" from the interaction case, since it also has to be robust to which channels landed on which side.

---

## 7. Spike Timing Precision

In many biological systems, neural codes rely on precise timing down to the millisecond scale. Measuring the exact temporal precision at which a representation carries information requires determining how much that information degrades when the timing is perturbed.

NeuralMI implements a highly efficient **"Train Once, Evaluate Many"** paradigm to establish this precision threshold without the massive computational overhead of retraining models for every noise level.

### The Baseline and Corruption Methodology
First, a baseline Mutual Information estimate is established by training a critic on the raw, uncorrupted data ($X$ and $Y$). Once the model converges, its weights are frozen.

To evaluate precision, the test data is iteratively corrupted across a grid of precision levels, denoted as $\tau$. NeuralMI supports two primary methods for corruption:

1.  **Deterministic Rounding (Default):** The data is explicitly quantized, forcing continuous times to snap to a discrete grid defined by the precision level $\tau$. Because this operation is deterministic, it requires only a single forward pass through the frozen network per precision level. The rounding operation is defined as:
    $$\tilde{X} = \tau * \left \lfloor{\frac{X}{\tau}}\right \rceil$$
    where $\left \lfloor{\cdot}\right \rceil$ denotes rounding to the nearest integer.
2.  **Additive Uniform Noise:** A stochastic alternative where noise sampled from a uniform distribution $U(-\frac{\tau}{2}, \frac{\tau}{2})$ is added to the data. Because this is probabilistic, the evaluation must be repeated multiple times (e.g., $N=50$) and averaged to get a stable estimate of the degraded Mutual Information.

### Defining the Precision Threshold

As $\tau$ increases, the severity of the corruption increases, and the Mutual Information estimated by the frozen network will inevitably drop. The spike timing precision of the representation is formally defined as the smallest $\tau^*$ at which the degraded Mutual Information falls below a fixed fraction $\rho$ of the baseline:

$$I(\tilde{X}^{\tau^*}; Y) < \rho \cdot I(X; Y)$$

The default ratio $\rho = 0.9$ (90%) is deliberately conservative: it identifies the coarsest timing resolution at which 90% of the available information is still preserved, providing an upper bound on the temporal precision required for faithful information transmission. This approach mirrors methods established in prior work on neural coding precision (Abdelaleem et al., "An information theoretic method to resolve millisecond-scale spike timing precision in a comprehensive motor program").

Multiple threshold ratios can be specified simultaneously — e.g., `threshold_ratio=[0.9, 0.75, 0.5]` — to characterise the full degradation profile of the representation and identify, for instance, both the onset of information loss (90%) and the point of catastrophic degradation (50%).

---

## 8. The Information Bottleneck Extension: Decoder-Augmented Training

The standard NeuralMI objective trains the critic purely to maximise MI:

$$\mathcal{L}_\text{standard} = -\hat{I}(Z_X; Z_Y)$$

When `use_decoder=True`, NeuralMI appends a **reconstruction decoder** for each variable. A decoder $d_X$ maps the embedding $Z_X$ back to the input space and is trained simultaneously with the critic. The augmented **Deep Symmetric Information Bottleneck** objective is:

$$\mathcal{L}_\text{decoder} = -\hat{I}(Z_X; Z_Y) + w_X \cdot \mathcal{L}_\text{rec}(X,\hat{X}) + w_Y \cdot \mathcal{L}_\text{rec}(Y,\hat{Y})$$

where $w_X, w_Y \ge 0$ are the reconstruction weights (`decoder_weight_x`, `decoder_weight_y`) and $\hat{X} = d_X(Z_X)$, $\hat{Y} = d_Y(Z_Y)$.

### Reconstruction Loss Selection

The appropriate reconstruction loss depends on the output activation of the decoder, which is set via `decoder_output_activation_x` / `decoder_output_activation_y`:

| Output activation | Data type | Loss |
|---|---|---|
| `'linear'` (default) | Continuous (float) | Mean Squared Error (MSE) |
| `'sigmoid'` | Binary / spike presence | Mean Squared Error (MSE) |
| `'softmax'` | Categorical (one-hot over channels) | Negative Log-Likelihood (NLL), equivalent to cross-entropy |

For the softmax case, the decoder outputs a probability distribution $p_c$ over channels at each time step, and the loss is $\mathcal{L}_\text{rec} = -\sum_c y_c \log p_c$ where $y_c$ is the ground-truth one-hot label.

### Combined with Variational Training

When both `use_variational=True` and `use_decoder=True` are set, the full loss is:

$$\mathcal{L}_\text{full} = \underbrace{\overline{D}_\text{KL}(Z_X) + \overline{D}_\text{KL}(Z_Y)}_\text{IB regularisation} - \beta\,\hat{I}(Z_X; Z_Y) + w_X \cdot \mathcal{L}_\text{rec}(X,\hat{X}) + w_Y \cdot \mathcal{L}_\text{rec}(Y,\hat{Y})$$

Here the KL terms push the embeddings towards a standard Gaussian prior (information bottleneck regularisation), $\beta$ controls how strongly MI maximisation dominates, and the decoder terms enforce that each embedding retains enough information to reconstruct its own input. The combined objective therefore encourages embeddings that are: **(i)** informative about the other variable, **(ii)** compact/regular in distribution, and **(iii)** reconstructive of their own input.

> **Note on ConcatCritic + use_variational:** When `critic_type='concat'` is combined with `use_variational=True`, the variational wrapper is applied to the concatenated pair representation $[Z_X, Z_Y]$ rather than to the individual marginals. This means the KL term measures the complexity of the joint pair embedding, not the marginal IB objective. The loss is still valid but does not correspond to the classic symmetric IB. Use `critic_type='separable'` or `critic_type='hybrid'` for the theoretically clean IB formulation.

---

## 9. Primary MI Estimate: `train_mi` in `mode='estimate'`

After training, NeuralMI reports two MI values:

- **`test_mi`** — the MI estimated on the held-out test set at the best epoch. This is the metric used to select the best model checkpoint during training (via early stopping on the smoothed test MI).
- **`train_mi`** — the MI estimated on a locked-in subset of the training data using the final (best-checkpoint) model. This is reported as `result.mi_estimate` and is the **primary point estimate**.

**Why `train_mi` is preferred as the final estimate:**

1. *Larger evaluation set.* The training subset used for `train_mi` (`train_eval_size` in `result.details`) is bounded by `max_eval_samples` and the training pool, independent of the test set's size — it is not coupled to (or capped by) how large the test split happens to be, so it can be, and typically is, substantially larger than `eval_size` (the test-side count). This gives `train_mi` a higher ceiling ($\log(\text{train\_eval\_size})$ vs. $\log(\text{eval\_size})$) and lower variance from the larger sample.
2. *No selection bias.* Because `train_mi` is computed on data the model was trained on, it reflects the capacity of the learned representation rather than the noisier generalisation signal. For bias-correction purposes (`mode='rigorous'`), the same principle applies — the bias correction is performed on the set of `train_mi` values across gamma values.
3. *Consistency with the `mode='rigorous'` pipeline.* The rigorous pipeline trains models at multiple data fractions and extrapolates `train_mi` values to infinite data, so `train_mi` is the natural quantity for that extrapolation.

The `test_mi` value is still accessible via `result.details['test_mi']` and is the more conservative, lower-variance bound if generalisation to held-out data is the primary concern.

## 10. What the Reported Number Is *Per*

Every MI value NeuralMI reports is per something, and that something differs between the static and temporal paths:

- **Static data** (no windowing processor): the estimate is per joint observation — one `(x_i, y_i)` row.
- **Temporal data** (windowed via a processor's `window_size`): the estimate is **per window**, i.e. $I(X_{1:w}; Y_{1:w})$ for a window of `window_size` samples — not a rate. This quantity is **extensive**: it can only grow (or, in the limit, plateau) as `window_size` grows, because a longer window's samples are a superset of a shorter window's at the same start point, and mutual information is monotonically non-decreasing under adding more observed variables. It never legitimately *declines* as `window_size` increases.

Two consequences worth being explicit about:

- **Values at different `window_size` settings are not directly comparable** as "how much do X and Y share" — a larger window will generically report a larger raw MI regardless of whether the underlying coupling actually changed, simply because it's built from more raw data. Comparing across `window_size` meaningfully requires either a rate (MI per unit time, e.g. dividing by `window_size` and the sample rate) or looking at the marginal MI gained per additional window sample, not the raw per-window value.
- **A raw (non-rate) MI-vs-`window_size` sweep whose values rise and then *fall* is reporting an estimation artifact, not a real property of the data** — per the extensivity argument above, the true quantity cannot decline. A decline is generically explained by the evaluation set shrinking as `window_size` grows (fewer windows fit in a fixed-length recording, so `eval_size`/`train_eval_size` fall and the ceiling drops with them — §2.1) and/or a fixed embedding capacity failing to keep up with a growing `window_size`'s input dimensionality. Check `result.details['eval_size']` / `['test_ceiling_mi']` / `['train_ceiling_mi']` alongside any such sweep before reading a shape into it.

## 11. Temporal Information Quantities

Beyond a single window's MI, a family of related quantities describe how
information is organized *across time* within and between two processes $X$
and $Y$. Every one of them, with one exception (interaction information,
which combines three separate MI estimates via a formula rather than being a
single conditional MI), reduces to

$$I(A; B \mid C)$$

for a particular choice of **which time offsets go into $A$, $B$, and $C$**.
Using the convention $X_{past} = X_{t-k:t-1}$, $X_{fut} = X_{t:t+k-1}$,
$X_0 = X_t$, and $X_{all} = X_{t-k:t+k}$ (a two-sided window), the quantities
this section covers are:

| Quantity | $A$ | $B$ | $C$ |
|---|---|---|---|
| Active information storage | $X_{past}$ | $X_0$ | — |
| Excess entropy | $X_{past}$ | $X_{fut}$ | — |
| Instantaneous MI | $X_0$ | $Y_0$ | — |
| Block MI (§10) | $X_{1:w}$ | $Y_{1:w}$ | — |
| Cross-predictive information | $X_{past}$ | $Y_{fut}$ | — |
| Transfer entropy | $X_{past}$ | $Y_0$ | $Y_{past}$ |
| Conditional transfer entropy | $X_{past}$ | $Y_0$ | $Y_{past}, W_{past}$ |
| MI rate | $X_{all}$ | $Y_0$ | $Y_{past}(h)$ |
| Instantaneous exchange | $X_0$ | $Y_0$ | $X_{past}, Y_{past}$ |
| Directed information rate | $X_{past}, X_0$ | $Y_0$ | $Y_{past}$ |

The first four are unconditioned (no $C$): a single $I(A;B)$ estimate, no
subtraction, no amplification risk. `neural_mi/quantities.py` provides thin
convenience functions for these (`active_information_storage`,
`excess_entropy`, `instantaneous_mi`, `cross_predictive_information`,
`block_mi`), each building the right offset arrays and calling
`mode='estimate'` underneath, with no new estimation machinery. This table
entry is purely a data-construction problem once $I(A;B)$ itself is solved.

### Active information storage and excess entropy

**Active information storage**, $AIS_X = I(X_{past}; X_0)$, measures how much
a process's own recent history predicts its present value: how much of $X$
is "stored" rather than novel at each step.

**Excess entropy**, $E_X = I(X_{past}; X_{fut})$, is the same construction
with a multi-sample future window instead of a single present time step.
Since a longer future window can only reveal more about the past (mutual
information is monotonically non-decreasing under adding more observed
variables, the same extensivity argument as §10), $E_X \ge AIS_X$ always.

Both are built via `neural_mi.analysis.offsets.build_past_future`, a sliding
window over one signal (`torch.Tensor.unfold`, a view not a copy):
$X_{past}[i] = X_{i:i+k}$ and $X_{fut}[i] = X_{i+k:i+k+m}$, so the future
window starts exactly where the past window ends.

### Instantaneous MI and cross-predictive information

**Instantaneous MI**, $I(X_0; Y_0)$, is plain, unwindowed MI between two
processes at matching time indices, needing no offset construction at all;
it's exactly what `mode='estimate'` computes directly.

**Cross-predictive information**, $I(X_{past}; Y_{fut})$, asks how much a
window of $X$'s past tells you about a window of $Y$'s future. It's
unconditioned, so it doesn't isolate $X$'s *unique* contribution the way
transfer entropy (§6.8 in `NEURALMI_REFERENCE.md`) does by subtracting out
$Y$'s own predictability of its future; it just measures the raw shared
predictive content. One training run, no subtraction, no amplification
risk. Built via `neural_mi.analysis.offsets.build_cross_offset`, which
reuses the same sliding-window construction transfer entropy's
array-building already uses internally.

### Block MI

**Block MI**, $I(X_{1:w}; Y_{1:w})$, is already exactly what `mode='estimate'`
computes with a windowed processor. See §10 for its extensivity properties
and why raw values aren't comparable across `window_size`. `block_mi()` in
`neural_mi/quantities.py` exists for naming symmetry with the rest of this
section, not new estimation logic.

### Transfer entropy and conditional transfer entropy

**Transfer entropy**, $\text{TE}_{X\to Y} = I(X_{past}; Y_0 \mid Y_{past})$
(Schreiber, 2000), is the first quantity in this section that actually
conditions on something: conditioning on $Y_{past}$ is what makes it
*directed* rather than symmetric, since it specifically removes $Y$'s own
storage (§"Active information storage" above) from being mistakenly counted
as something $X$ transferred. Computed via the chain-rule identity
$\text{TE}_{X\to Y} = I(X_{past}, Y_{past}; Y_0) - I(Y_{past}; Y_0)$, a
difference of two separately-trained MI estimates, via `mode='transfer'`.

**Conditional transfer entropy**, $\text{TE}_{X\to Y}(W) = I(X_{past}; Y_0
\mid Y_{past}, W_{past})$, adds a third process's history to the
conditioning side, asking how much of $X$'s apparent influence on $Y$
survives once a third process $W$'s own history is also controlled for. If
$W$ explains away most of $X$'s apparent contribution, that contribution was
likely mediated through $W$ rather than a direct $X\to Y$ pathway. It's
built by folding $W_{past}$ (constructed the same way as
$X_{past}$/$Y_{past}$, same `history_window`) into both the joint and
marginal conditioning arrays, via `Transfer(w_data=...)`; see
`NEURALMI_REFERENCE.md` §6.8.

Both are, by construction, a *difference* of two independently-trained
estimates rather than a large fraction of their own components, the same
"small residual relative to its components" shape identified as fragile
elsewhere in this section's validation. Conditional transfer entropy is the
more fragile of the two: introducing a third conditioning signal that
explains away most of the plain-TE signal shrinks the residual further,
amplifying whatever noise the two component estimates carry. Report the
component estimates (`result.details['i_xypast_yfuture']`, etc.) alongside
the point estimate rather than trusting the difference in isolation, and
prefer more training data/epochs here before reading much into a small
absolute value.

### MI rate, instantaneous exchange, and directed information rate

These three share a trait the rest of this section doesn't: $A$ and $C$
genuinely differ in window length, not the one-sample rounding difference
`mode='conditional'` already tolerates when trimming to a shared start.

**MI rate**, $I(X_{all}; Y_0 \mid Y_{past}(h))$, is the two-sided,
per-sample information rate between $X$ and $Y$ as $h \to \infty$: how much
$X$'s value in a symmetric window around the present tells you about $Y$'s
present, once enough of $Y$'s own past is controlled for that little of its
remaining predictability is really $Y$'s own storage. $X_{all}$ is
two-sided (spans both $X_{past}$ and $X_{fut}$ around the same center as
$Y_0$) because a one-sided window would let $X$'s future values act as a
proxy for $Y$'s own future, per the feedback consideration behind
information rate's usual two-sided definition. In practice, `mi_rate(h=[...])`
swept over increasing $h$ should converge to a stable value once $h$ is
large enough to have absorbed $Y$'s own dependence structure; a value still
drifting upward at the largest $h$ tried means $h$ hasn't reached that
point yet, not that the estimator is unreliable.

**Instantaneous exchange**, $I(X_0; Y_0 \mid X_{past}(k), Y_{past}(k))$,
asks how much $X$ and $Y$ share at the same instant that isn't already
explained by their separate, shared pasts. Unlike transfer entropy, this
term tends to be large relative to its two components in a system with real
shared instantaneous structure (a shared upstream driver affecting both
processes at the same time step), so it doesn't inherit transfer entropy's
amplification fragility even though it's also computed as a difference of
two independently-trained estimates.

**Directed information rate**, $I(X_{past}(k), X_0; Y_0 \mid Y_{past}(k))$,
is the total directed relationship between $X$ and $Y$: everything $X$'s
past-and-present together tell you about $Y$'s present beyond $Y$'s own
past. It decomposes exactly as $\text{TE}_{X\to Y} + $ instantaneous
exchange (both sides share the same $B$ and, once $X_0$ is moved from $A$ to
being folded in, the same effective conditioning), a decomposition that's
useful as a sanity check but not as the estimation method: composing it from
its two parts would sum transfer entropy's small, high-variance residual
into an otherwise well-behaved quantity, reintroducing the fragility a
direct estimate avoids. `directed_information_rate()` always estimates it
directly, from its own $A$/$B$/$C$ arrays.

All three need `Model(embedding_model='dual_branch', ...)`
and `Conditional(align='dual_branch')` (handled automatically by
`neural_mi.quantities.mi_rate`/`instantaneous_exchange`/`directed_information_rate`,
which raise a clear error if the model isn't configured this way). See
"DualBranchEmbedding" below for why, and `NEURALMI_REFERENCE.md` §13 for the
full API.

### DualBranchEmbedding

`mode='conditional'` ordinarily concatenates $X$ and $W$ into one array
before embedding, which requires them to share a window length (a
one-sample mismatch is trimmed automatically; anything more raises an
error). MI rate, instantaneous exchange, and directed information rate all
need $A$ and $C$ at genuinely different lengths, so forcing a shared length
would mean zero-padding the shorter one out to match, letting the network
learn on its own to ignore a constant-zero region rather than that being
handled structurally.

`DualBranchEmbedding` avoids the padding: two independent sub-embedding
networks, one per input length, each processing its own array natively (an
RNN-family branch's weights depend only on channel count, not on sequence
length, so there's no length-matching constraint between the branches at
all), fused by a small MLP into one embedding vector. It ships as
`embedding_model='dual_branch'`, a real entry in the model-selection
dispatch (`build_critic`'s `_EMBEDDING_CLASSES` table) rather than only
reachable via `custom_embedding_cls` -- `branch_model` (default `'gru'`)
picks each branch's own architecture from that same table. A genuinely
custom (non-built-in) branch architecture still uses
`custom_embedding_cls=DualBranchEmbedding`-based subclass instead (see the
class docstring), which always takes priority when set.
`Conditional(align='dual_branch')`
is the corresponding data-side opt-in: it builds the "X-role" data as a
tuple `(a_data, c_data)` instead of concatenating, which
`DualBranchEmbedding` (and the rest of the training pipeline: dataset,
sweep, task, trainer, critic layers) knows how to handle. Not supported
with `use_variational=True` or `mode='sweep'`'s `max_samples_per_task`; both
raise a clear error rather than a silent wrong result if attempted.

## 12. Interaction Information

The one quantity in this taxonomy that isn't a single $I(A;B \mid C)$
estimate. Interaction information asks how much shared information between
two populations $X$ and $Y$ changes once a third population $W$ is also
observed:

$$II = I(X, W; Y) - I(X; Y) - I(W; Y)$$

Three separate MI estimates, combined by a formula, computed via
`mode='interaction'`. Two internally-related quantities emerge from the
same three terms: $I(X,W;Y) - I(X;Y)$ is $W$'s marginal contribution once
$X$ is already known, and by symmetry $I(X,W;Y) - I(W;Y)$ is $X$'s marginal
contribution once $W$ is already known; $II$ is what's left after
subtracting both marginal contributions from the joint, the part of the
relationship that only shows up when $X$ and $W$ are considered together.

Unlike every quantity in §11, $II$ is signed and can be positive or
negative, and both directions have a standard reading:

- **$II < 0$ (redundancy)**: $X$ and $W$ carry overlapping information
  about $Y$, so observing both together tells you less than the sum of
  observing each alone would suggest. The common case is a shared upstream
  cause: if $X$ and $W$ are both driven by the same underlying process that
  also drives $Y$, each is a partial proxy for that process, and knowing
  one reduces how much additional information the other can offer.
- **$II > 0$ (synergy)**: $X$ and $W$ together reveal something about $Y$
  that neither reveals alone, for example if $Y$ depends on some
  combination of $X$ and $W$ (an XOR-like relationship is the canonical
  case) that isn't visible from either variable in isolation.

Like transfer entropy, this is built from independently-trained estimates
combined by subtraction rather than being a single large-fraction quantity,
so the same fragility considerations from §11 apply: report the three
component estimates (`result.details['mi_xw_y']`, `['mi_x_y']`,
`['mi_w_y']`) alongside `interaction_info`, not in place of it, especially
when the estimate is close to zero and its sign is the main thing being
asked about. See `NEURALMI_REFERENCE.md` §6.10 for the full API.