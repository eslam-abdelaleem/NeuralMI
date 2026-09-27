# Anatomy of a neural network-based estimator

Here we build a working MI estimator from scratch in a few dozen lines of PyTorch. Read it to see what the library does when you call `run()`.

[USING.md](USING.md) documents the real API, [THEORY.md](THEORY.md) covers where the bound comes from and why finite samples bias it, and notebook 05 shows how to choose between the estimators and architectures the library ships.

The construction follows [Abdelaleem et al., 2025](https://arxiv.org/abs/2506.00330), which builds on [van den Oord et al., 2018](https://arxiv.org/abs/1807.03748), [Poole et al., 2019](https://proceedings.mlr.press/v97/poole19a.html) and [Song and Ermon, 2020](https://arxiv.org/abs/1910.06222).

---

## The three components

A neural MI estimator trains a network to solve a classification problem.
Instead of estimating densities, it trains a **critic** $f(x, y)$ to separate
positive pairs $(x_i, y_i)$ that genuinely co-occurred from negative pairs
$(x_i, y_j)$ that did not.

**1. Embedding networks $g$ and $h$**

These are two small networks that learn features of $X$ and $Y$.

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

**2. The critic $f(x, y)$**

In a `SeparableCritic` the critic is the dot product of the two embeddings, giving a similarity score for every pairing of samples in a batch.

```python
def separable_critic(x_embedded, y_embedded):
    # x_embedded has shape (batch_size, embedding_dim)
    # y_embedded has shape (batch_size, embedding_dim)
    # The result is a (batch_size, batch_size) matrix of scores
    return torch.matmul(x_embedded, y_embedded.t())
```

**3. The estimator**

The estimator turns the critic's score matrix into an MI estimate. The library's default is InfoNCE,

$$
I(X;Y) \ge \mathbb{E}\left[ \frac{1}{N}\sum_{i=1}^N \left( f(x_i,y_i) - \log\left(\sum_{j=1}^N e^{f(x_i,y_j)}\right) \right) \right] + \log(N)
$$

The formula is a cross-entropy loss over the score matrix. Each row $x_i$ has
its true partner $y_i$ on the diagonal, and the loss maximises that diagonal
score against every off-diagonal $y_j$ in the batch.

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

Those three pieces are the estimator.

---

## The training loop

The components train like standard neural networks by minimising a loss. Here the loss
is the negative of the MI estimate, since maximising MI and minimising $-\text{MI}$ are
the same problem. A simplified training loop looks like this.

```python
# --- Setup ---
dim = 5
embedding_dim = 16
batch_size = 128
n_epochs = 10

# Two views of one shared latent, so there is real MI to find
z = torch.randn(1000, dim)
x_data = z + 0.5 * torch.randn(1000, dim)
y_data = z + 0.5 * torch.randn(1000, dim)

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

    # 5. Backpropagate and optimise
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, MI Estimate (nats): {mi_estimate.item():.3f}")
```

### Evaluation and early stopping

A real run splits the data and evaluates MI on the held-out partition after every
epoch to produce a `test_mi_history` curve. We use the max-test heuristic to pick the network weights that scored highest on that curve, and evaluate the train MI with them.

The curve is noisy and its raw maximum can be unreliable, so we smooth it with a median filter followed by a Gaussian filter (one reasonable choice among several) and stop when the smoothed curve has not improved for `patience` epochs.
