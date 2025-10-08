Theoretical Foundations of NeuralMI
===================================

This document provides a concise theoretical background for the core methods used in the ``NeuralMI`` library. It is intended as a formal reference for users who wish to understand the mathematical principles behind the code.

1. The Challenge of Estimating Mutual Information
-------------------------------------------------

Mutual Information (MI) is formally defined as the Kullback-Leibler (KL) divergence between the joint distribution $p(x, y)$ and the product of the marginal distributions $p(x)p(y)$:

.. math::

   I(X; Y) = \int p(x, y) \log \frac{p(x, y)}{p(x)p(y)} dx dy

Calculating this directly requires knowing these probability distributions. For high-dimensional and continuous data, like that often found in neuroscience, these distributions are unknown and practically impossible to estimate accurately. Traditional methods like binning or kernel density estimation fail due to the "curse of dimensionality."

To overcome this, ``NeuralMI`` uses a modern approach called **neural estimation**, which reframes MI estimation as a neural network optimization problem.

2. Neural MI Estimators: A Bias-Variance Trade-off
---------------------------------------------------

Instead of estimating the probability densities, we can use a neural network, called a **critic** $f(x, y)$, to help us estimate a lower bound on the true MI. The core idea is to train the critic to distinguish between "positive" samples (pairs ``(x_i, y_i)`` that genuinely occurred together) and "negative" samples (pairs ``(x_i, y_j)`` from the same batch that did not).

Different mathematical formulations, or estimators, can be used for this task. They represent a trade-off between the **bias** of the estimate (how far it is from the true value on average) and its **variance** (how much it fluctuates on different runs). As argued in recent literature, choosing the right estimator depends on the scientific question.

    **References:**

    - "Understanding the Limitations of Variational Mutual Information Estimators" (ICLR 2020)
    - "On Variational Bounds of Mutual Information" (PMLR 2019)
    - "Accurate Estimation of Mutual Information in High Dimensional Data" (ArXiv 2025)

``NeuralMI`` focuses on two particularly effective estimators that cover the most common use cases.

2.1 InfoNCE (Low Variance, High Bias)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **InfoNCE** (Noise-Contrastive Estimation) estimator is the workhorse of ``NeuralMI``. Its formula is:

.. math::

   I(X;Y) \ge \mathbb{E}\left[ f(x,y) - \log\left(\frac{1}{N}\sum_{j=1}^N e^{f(x,y_j)}\right) \right]

**Intuition:** For each positive pair `(x, y)`, the critic `f(x,y)` tries to maximize its score relative to the scores of `N-1` negative pairs `(x, y_j)`. This is effectively a classification problem where the model tries to pick the "real" partner `y` for a given `x` out of a lineup of `N` candidates.

**Properties:**

-   **Low Variance:** InfoNCE is known to be a very stable estimator, producing consistent results across different random seeds.
-   **Biased:** It is a lower bound on the true MI. Crucially, this bound is **theoretically limited by $\log(N)$**, where $N$ is the batch size. This means InfoNCE can never report an MI value higher than $\log(N)$. For most applications where the true MI is modest, this is not a problem and its stability is a major advantage. This is why it's the default in ``NeuralMI``.

2.2 SMILE (Low Bias, Moderate Variance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **SMILE** (Smoothed Mutual Information "Lower-bound" Estimator) is designed to provide a less biased estimate, which is critical in scenarios where the true MI might be high.

.. math::

   I(X;Y) \ge \mathbb{E}\left[ f(x,y) \right] - \log \mathbb{E}\left[ e^{\text{clip}(f(x,y'), \tau)} \right]

**Intuition:** SMILE is similar to other classical estimators - like MINE *Mutual Information Neural Estimator* -, but it introduces a clipping function on the normalization factor. By clipping the scores at a value $\tau$, it prevents a few "easy" samples from dominating the loss function, which is a major source of bias.

**Properties:**

-   **Low Bias:** By mitigating the impact of easy negatives, SMILE can provide estimates that are much closer to the true MI, especially when the MI is high. It is not strictly bounded by $\log(N)$ in the same way as InfoNCE.
-   **Moderate Variance:** This reduction in bias comes at the cost of slightly higher variance compared to InfoNCE.
-   **The ``clip`` parameter (``τ``):** A clipping value of ``τ=5`` is often a robust default choice.

    **Recommendation:** Use InfoNCE for general-purpose, stable MI estimation. Use SMILE for tasks like dimensionality estimation where the true MI may be very high and you need a less biased estimator.

3. The Variational Approach
---------------------------

Standard neural estimators learn a single embedding vector, $z = g(x)$, for each input. A variational approach, in contrast, learns a posterior distribution over the embeddings, $q(z|x)$. This is typically a Gaussian distribution parameterized by a mean and a variance vector, $(\mu_x, \sigma_x) = g(x)$.

The total loss function is modified to include a KL divergence term that acts as a regularizer, encouraging the learned posterior distributions to be close to a prior (usually a standard normal distribution):

.. math::

   \mathcal{L} = (D_{KL}(p(z_x|x)||q(z_x)) + D_{KL}(p(z_y|y)||p(z_y))) - \beta  \hat{I_\text{estimator}}(Z_X;Z_Y)

This regularization can improve the quality of the learned representations and lead to more stable and robust MI estimates, particularly in complex, high-dimensional settings.

4. The Problem of Finite-Sampling Bias
--------------------------------------

Even with a perfect estimator, any analysis performed on a finite dataset of $N$ samples will be biased. The model will inevitably find spurious correlations in the random noise of the data, leading to a **systematic overestimation** of the true MI.

Theoretically, for a large number of samples $N$, this bias has a clear relationship with the sample size:

.. math::

   I_{\text{estimated}}(N) \approx I_{\text{true}} + \frac{a}{N} + O\left(\frac{1}{N^2}\right)

This means the estimated MI is approximately linear in $1/N$. This is the key insight that ``NeuralMI`` uses to correct for the bias.

5. The Solution: Rigorous Bias Correction
-----------------------------------------

The ``mode='rigorous'`` in ``NeuralMI`` automates a principled, multi-step workflow based on this theoretical relationship:

1.  **Subsampling:** The library repeatedly runs the MI estimation on different fractions of the data. For example, it might split the data into $\gamma=2$ halves, then $\gamma=3$ thirds, and so on.
2.  **Fitting:** It calculates the mean MI estimate for each data fraction size ($1/N$). Because the bias is linear in $1/N$, it fits a weighted linear regression to these points.
3.  **Extrapolation:** It extrapolates this line back to the y-intercept, which corresponds to $1/N = 0$—an infinite dataset. This intercept is the final, bias-corrected MI estimate. The confidence interval of this intercept provides the error bars.

This procedure effectively subtracts the bias that is dependent on sample size, yielding a more accurate and scientifically rigorous result.

    **References:**

    - "Estimation of mutual information for real-valued data with error bars and controlled bias" (PRE 2019)
    - "Accurate Estimation of Mutual Information in High Dimensional Data" (ArXiv 2025)

6. Estimating Latent Dimensionality
-----------------------------------

The ``mode='dimensionality'`` uses a clever trick to estimate the complexity of a single neural population ``X``. It randomly splits the channels of ``X`` into two halves, ``X_A`` and ``X_B``, and measures the "Internal Information" ``I(X_A; X_B)``.

**Intuition:**
If both ``X_A`` and ``X_B`` are just different observations of the same underlying low-dimensional latent signal ``Z``, then in theory, ``I(X_A; X_B) = I(Z; Z) = \infty``. However, the *discoverable* information is constrained by the dimensionality of ``Z``.

We can find this constraint by using a **``SeparableCritic``** and varying its ``embedding_dim``. The ``embedding_dim`` acts as a bottleneck.

-   If ``embedding_dim`` < ``dim(Z)``, the model can't capture all the shared information, and the estimated MI will be low.
-   As ``embedding_dim`` approaches ``dim(Z)``, the estimated MI will rise.
-   Once ``embedding_dim`` > ``dim(Z)``, the model has enough capacity, and the MI will **saturate**. The point of saturation is our estimate for the latent dimensionality of ``Z``.

For this specific task, **SMILE is often a better estimator than InfoNCE**. Because the true MI can be very high, InfoNCE might saturate at its theoretical limit of $\log(N)$ *before* the model's capacity (``embedding_dim``) becomes the true bottleneck. SMILE's lower bias allows the curve to rise higher, revealing the true saturation point more clearly.

Note that this process can also be done for regular ``I(X;Y)``, informing us about the intrinsic dimensionality of the *interaction* space. Here, the information won't be theoretically infinite, and InfoNCE probably will be better.