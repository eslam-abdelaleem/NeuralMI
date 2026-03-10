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

2. **Fitting:** It calculates the mean MI estimate for each data fraction size ($1/N$). Because the bias is linear in $1/N$, it fits a weighted linear regression to these points.

3. **Extrapolation:** It extrapolates this line back to the y-intercept, which corresponds to $1/N = 0$—an infinite dataset. This intercept is the final, bias-corrected MI estimate. The confidence interval of this intercept provides the error bars.

This procedure effectively subtracts the bias that is dependent on sample size, yielding a more accurate and scientifically rigorous result.

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

### Cross-Covariance and the Participation Ratio
Once the Hybrid Critic is trained at maximum capacity, we extract the learned embeddings $Z_X$ and $Z_Y$ for the test set. We compute their cross-covariance matrix $C_{XY}$:

$$C_{XY} = \frac{1}{N-1} (Z_X - \bar{Z}_X)^T (Z_Y - \bar{Z}_Y)$$

We then perform Singular Value Decomposition (SVD) on $C_{XY}$ to extract the spectrum of singular values $\sigma_i$. These singular values represent the strength of the shared variance across the orthogonal dimensions of the latent space.

From this spectrum, NeuralMI calculates the **Participation Ratio (PR)**, a continuous measure of the effective number of dimensions utilized by the network:

$$PR_{singular} = \frac{(\sum_{i=1}^{k_z} \sigma_i)^2}{\sum_{i=1}^{k_z} \sigma_i^2}$$

A $PR$ of 5.2 indicates that the shared information between $X$ and $Y$ effectively occupies 5.2 dimensions, regardless of the fact that the actual bottleneck was 64.

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
As $\tau$ increases, the severity of the corruption increases, and the Mutual Information estimated by the frozen network will inevitably drop. 

The spike timing precision of the representation is formally defined as the specific $\tau$ threshold at which the degraded Mutual Information $I(\tilde{X}; Y)$ falls below **90% of the baseline zero-noise Mutual Information**. This provides an empirical bound on the temporal resolution required to capture the majority of the available information.