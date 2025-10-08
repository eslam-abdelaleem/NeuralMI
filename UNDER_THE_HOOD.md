# Under the Hood: How NeuralMI Works

This document is for the curious user who has completed the tutorials and wants a deeper understanding of what `NeuralMI` is doing under the hood. We won't cover all the advanced features, but we will build a simple neural MI estimator from scratch to demystify the core concepts.

The document is based heavily on [this paper](https://arxiv.org/abs/2506.00330).

Our goal is to answer three key questions:
1.  What *is* a neural MI estimator, really?
2.  How is it trained and evaluated?
3.  What is the intuition behind the "rigorous" bias correction?

Let's dive in with some simple PyTorch code.

---

## Part 1: Anatomy of a Neural MI Estimator

At its heart, one way of looking at a neural MI estimator is just a clever way of training a neural network to solve a classification problem. Instead of estimating probability densities directly, we train a **critic** network, `f(x, y)`, to distinguish between "positive" samples (pairs `(x_i, y_i)` that genuinely occurred together) and "negative" samples (pairs `(x_i, y_j)` that did not).

Let's build the three essential components from scratch.

### The Components

1.  **Embedding Networks (`g` and `h`):** These are two simple neural networks that learn to extract meaningful features from `X` and `Y`.

    ```python
    import torch
    import torch.nn as nn

    # A simple MLP to process an input vector into an embedding
    def create_embedding_net(input_dim, embedding_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    ```

2.  **The Critic `f(x, y)`:** In a `SeparableCritic`, the "critic" is just the dot product between the embeddings. It computes a similarity score for every possible pairing of samples in a batch.

    ```python
    def separable_critic(x_embedded, y_embedded):
        # x_embedded has shape (batch_size, embedding_dim)
        # y_embedded has shape (batch_size, embedding_dim)
        # The result is a (batch_size, batch_size) matrix of scores
        return torch.matmul(x_embedded, y_embedded.t())
    ```

3.  **The Estimator (Loss Function):** This is the mathematical formula that turns the score matrix from the critic into an MI estimate. Let's implement the most common one, **InfoNCE**. The formula is:

    $$ I(X;Y) \ge \mathbb{E}\left[ \frac{1}{N}\sum_{i=1}^N \left( f(x_i,y_i) - \log\left(\sum_{j=1}^N e^{f(x_i,y_j)}\right) \right) \right] + \log(N) $$

    This looks complex, but it's just a form of cross-entropy loss. For each `x_i` in the batch (each row of the score matrix), we're trying to maximize the score of its true partner `y_i` (the diagonal element) relative to all other `y_j`'s in the batch (the off-diagonal elements).

    ```python
    def infonce_estimator(scores):
        # scores is the (batch_size, batch_size) matrix from the critic
        batch_size = scores.shape[0]
        
        # The f(x_i, y_i) term is the diagonal of the score matrix
        positive_scores = torch.diag(scores)
        
        # The log-sum-exp term is calculated for each row
        log_sum_exp = torch.logsumexp(scores, dim=1)
        
        # The MI is the mean difference, plus log(batch_size)
        mi_estimate_nats = torch.mean(positive_scores - log_sum_exp) + torch.log(torch.tensor(batch_size))
        
        return mi_estimate_nats
    ```

And that's it! A neural MI estimator is just these three pieces working together.

---

## Part 2: The Training Loop Demystified

Now, how do we use these components? We train them just like any other neural network: by minimizing a loss function. For MI estimation, the loss is simply the **negative of the MI estimate**. Maximizing the MI is the same as minimizing `-MI`.

Here's a simplified training loop.

```python
# --- Setup ---
dim = 5
embedding_dim = 16
batch_size = 128
n_epochs = 10

# Create simple correlated data
x_data, y_data = torch.randn(1000, dim), torch.randn(1000, dim)

# Create our embedding networks
g_net = create_embedding_net(dim, embedding_dim)
h_net = create_embedding_net(dim, embedding_dim)

# Group parameters and create an optimizer
params = list(g_net.parameters()) + list(h_net.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# --- Training Loop ---
for epoch in range(n_epochs):
    # In a real scenario, we would use a DataLoader to get batches
    x_batch = x_data[:batch_size]
    y_batch = y_data[:batch_size]

    # 1. Get embeddings
    x_embedded = g_net(x_batch)
    y_embedded = h_net(y_batch)

    # 2. Get scores from the critic
    scores = separable_critic(x_embedded, y_embedded)

    # 3. Calculate the MI estimate
    mi_estimate = infonce_estimator(scores)

    # 4. The loss is the negative MI
    loss = -mi_estimate

    # 5. Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, MI Estimate (nats): {mi_estimate.item():.3f}")
```

### Evaluation and Early Stopping
During a real training run, we would split our data into training and validation sets. After each epoch, we'd calculate the MI on the validation set.

This produces a `test_mi_history` curve. A heuristic introduced in the paper is to stop training and use the model that achieved the highest MI on the validation set. However, this curve can be very noisy. `NeuralMI` follows the same procedure as the paper and improves on this by applying a median filter followed by a Gaussian filter to get a smoothed curve. It then stops training when this smoothed curve has stopped improving for a set number of epochs (`patience`), which is a more robust strategy.

---
## Part 3: The Intuition Behind `mode='rigorous'`
Even with a perfectly trained model, any MI estimate from a finite dataset will be biased. The model will inevitably find spurious correlations in the noise, leading to a **systematic overestimation** of the true MI.

As explained in the literature, this bias has a predictable relationship with the number of samples, `N`:
$$ I_{\text{estimated}}(N) \approx I_{\text{true}} + \frac{a}{N} $$

This means the estimated MI is approximately linear in `1/N`. The `rigorous` mode exploits this relationship to correct for the bias.

### The Extrapolation Procedure
1. **Subsample:** The library runs the MI estimation multiple times on different fractions of the data. For example, it might split the data into `γ=2` halves, then `γ=3` thirds, and so on. This gives us MI estimates for different effective sample sizes `N/γ`.
2. **Plot vs. 1/N:** The library plots the mean MI estimate for each `γ` against `1/(N/γ)`, which is proportional to `γ`. Because of the formula above, this plot should be a straight line.
3. **Extrapolate:** `NeuralMI` performs a weighted linear regression on this line and finds the y-intercept. This intercept corresponds to the point where `1/N = 0`, which represents an infinite dataset. This extrapolated value is the final, bias-corrected MI estimate. If we're not able to perform such fit --i.e., the fit is not *linear*, judging by fitting a second order WLS first, see if the ratio of the quadratic to linear contribution $\delta$ is greater than a certain threshold -set to 10%- then we reject this fitting point. If so, we drop the estimates corresponding to this $\gamma$ value, and recalculate, if $\delta \geq$ threshold, keep dropping points till we reach the $\gamma$ < `min_gamma_points`, usually 5, and if not linear yet, we deem this fit *not reliable* and shouldn't be trusted as we don't have enough data. 

The plot generated by `results.plot()` in rigorous mode is a direct visualization of this procedure. The **Corrected MI** is simply the y-intercept of the extrapolation line, giving you an estimate of what the MI would be if you could collect an infinite amount of data.



