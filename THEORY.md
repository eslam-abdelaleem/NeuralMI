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
- **Biased:** It is a lower bound on the true MI. Crucially, this bound is **theoretically limited by $\log(N)$**, where $N$ is the batch size. This means InfoNCE can never report an MI value higher than $\log(N)$. For most applications where the true MI is modest, this is not a problem and its stability is a major advantage. This is why it's the default in `NeuralMI`.

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

Use **InfoNCE** for general-purpose, stable MI estimation. Use **SMILE** for tasks like dimensionality estimation where the true MI may be very high and you need a less biased estimator.
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

Even with a perfect estimator, any analysis performed on a finite dataset of $N$ samples will be biased. The model will inevitably find spurious correlations in the random noise of the data, leading to a **systematic overestimation** of the true MI.

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

## 6. Latent Dimensionality via Spectral Metrics

Until recently, finding the dimensionality of the shared information between two datasets involved treating the bottleneck dimension ($k_z$) as a search parameter. We would artificially starve the network's capacity and sweep $k_z$ to find the exact point where Mutual Information saturated. However, this approach is computationally expensive and highly sensitive to the geometric constraints of the chosen critic (e.g., a simple dot-product in a Separable Critic might fail to capture complex dependencies in low dimensions, causing "false saturation").

Following our work in ([arxiv:2602.08105](https://arxiv.org/abs/2602.08105)), NeuralMI now treats dimensionality not as a search problem, but as an **observable property of an over-parameterized latent space**.

### The Hybrid Critic and the Saturation Hypothesis
To accurately measure dimensionality, NeuralMI uses a **Hybrid Critic**. This architecture embeds $X$ and $Y$ independently but processes their concatenation through a final Multi-Layer Perceptron (MLP) decision head. By setting a safely large bottleneck (e.g., $k_z = 64$), we ensure the network is "capacity tight." The network does not distribute a low-dimensional signal diffusely across all 64 dimensions; instead, it concentrates the shared information into a compact subspace.

### Embedding Whitening

Before computing the cross-covariance, `NeuralMI` optionally **whitens** the embeddings. The default mode (`spectral_whitening='std'`) standardises each embedding dimension by dividing it by its empirical standard deviation:

$$\tilde{Z}_{X,i} = \frac{Z_{X,i}}{\text{std}(Z_{X,i})}, \qquad \tilde{Z}_{Y,i} = \frac{Z_{Y,i}}{\text{std}(Z_{Y,i})}$$

This ensures that dimensions with accidentally large variance do not dominate the SVD spectrum, yielding a PR estimate that reflects the true geometric structure of the shared information rather than the scale of individual latent dimensions.

### Cross-Covariance and the Participation Ratio
Once the Hybrid Critic is trained at maximum capacity, we extract the learned embeddings $Z_X$ and $Z_Y$ for the test set. We compute their cross-covariance matrix $C_{XY}$:

$$C_{XY} = \frac{1}{N-1} (Z_X - \bar{Z}_X)^T (Z_Y - \bar{Z}_Y)$$

We then perform Singular Value Decomposition (SVD) on $C_{XY}$ to extract the spectrum of singular values $\sigma_i$. These singular values represent the strength of the shared variance across the orthogonal dimensions of the latent space.

From this spectrum, NeuralMI calculates two variants of the **Participation Ratio (PR)**:

$$PR_{\text{singular}} = \frac{\left(\sum_{i=1}^{k_z} \sigma_i\right)^2}{\sum_{i=1}^{k_z} \sigma_i^2}$$

$$PR_{\text{covariance}} = \frac{\left(\sum_{i=1}^{k_z} \sigma_i^2\right)^2}{\sum_{i=1}^{k_z} \sigma_i^4}$$

Both are continuous measures of the effective number of dimensions utilised by the network; a value of 5.2 indicates that the shared information effectively occupies 5.2 dimensions. The two variants differ in how they weight the spectrum:

* **$PR_{\text{singular}}$** weights each dimension proportionally to its singular value $\sigma_i$.
* **$PR_{\text{covariance}}$** weights each dimension proportionally to its eigenvalue $\lambda_i = \sigma_i^2$. Because eigenvalues amplify contrast between large and small singular values, $PR_{\text{covariance}} \le PR_{\text{singular}}$ in general and is more sensitive to the true rank of the representation — small-but-nonzero singular values contribute relatively less.

### Interaction vs. Intrinsic Dimensionality
* **Interaction Dimensionality:** When evaluating two distinct datasets ($X$ and $Y$), the PR of their cross-covariance directly yields the dimensionality of their shared information space.
* **Intrinsic Dimensionality:** When analyzing a single dataset ($X$), NeuralMI splits the data into two non-overlapping halves (e.g., randomly splitting the channels/neurons, or splitting spatially/temporally). It then computes the Interaction Dimensionality between these halves, repeating this process over multiple splits and averaging the PR to find the intrinsic dimensionality of the dataset itself.

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

1. *Larger evaluation set.* The training subset used for `train_mi` is typically larger than the test set (default `train_fraction=0.9`), so the estimate has lower variance.
2. *No selection bias.* Because `train_mi` is computed on data the model was trained on, it reflects the capacity of the learned representation rather than the noisier generalisation signal. For bias-correction purposes (`mode='rigorous'`), the same principle applies — the bias correction is performed on the set of `train_mi` values across gamma values.
3. *Consistency with the `mode='rigorous'` pipeline.* The rigorous pipeline trains models at multiple data fractions and extrapolates `train_mi` values to infinite data, so `train_mi` is the natural quantity for that extrapolation.

The `test_mi` value is still accessible via `result.details['test_mi']` and is the more conservative, lower-variance bound if generalisation to held-out data is the primary concern.