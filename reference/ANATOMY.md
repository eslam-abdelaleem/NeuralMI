# Anatomy of a neural network-based estimator

Here we build a working MI estimator from scratch in a few dozen lines of PyTorch. Read it to see what the library does when you call `run()`.

[USING.md](USING.md) documents the real API, [THEORY.md](THEORY.md) covers where the bound comes from and why finite samples bias it, and notebook 05 shows how to choose between the estimators and architectures the library ships.

The construction follows [Abdelaleem et al.,
2025](https://arxiv.org/abs/2506.00330) and through it [van den Oord et al.,
2018](https://arxiv.org/abs/1807.03748), [Poole et al.,
2019](https://proceedings.mlr.press/v97/poole19a.html) and [Song and Ermon,
2020](https://arxiv.org/abs/1910.06222).

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
def create_embedding_net(input_dim, embedding_dim, hidden_dim=256):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, embedding_dim)
    )
```

**2. The critic $f(x, y)$**

In a `SeparableCritic` the critic is the dot product of the two embeddings and
gives a similarity score for every pairing of samples in a batch.

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
its true partner $y_i$ on the diagonal. The loss maximises that diagonal score
against every off-diagonal $y_j$ in the batch.

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

The components train like standard neural networks by minimising a loss. Here
the loss is the negative of the MI estimate because maximising MI and minimising
$-\text{MI}$ are the same problem.

### Data with a known answer

The data are those of notebooks 01 and 02. A latent pair is drawn jointly
Gaussian in ten dimensions with each dimension correlated only with its partner.
The correlation $\rho = \sqrt{1 - 2^{-2I/d}}$ puts exactly $I = 4$ bits between
the two halves. Each half then goes through its own random network into 500
observed channels. That map is injective and leaves the MI at 4 bits.

A run holds out part of the data to judge the training by. The samples here are
independent draws and a random tenth of them is held out.

```python
torch.manual_seed(0)

latent_dim, observed_dim, n_samples = 10, 500, 2000
true_mi_bits = 4.0

# The latent pair: each dimension correlated with its partner and nothing else
rho = (1 - 2 ** (-2 * true_mi_bits / latent_dim)) ** 0.5
z_x = torch.randn(n_samples, latent_dim)
z_y = rho * z_x + (1 - rho ** 2) ** 0.5 * torch.randn(n_samples, latent_dim)

# One fixed random network per half carries the latent into 500 channels
map_x = nn.Sequential(nn.Linear(latent_dim, 64), nn.Softplus(), nn.Linear(64, observed_dim))
map_y = nn.Sequential(nn.Linear(latent_dim, 64), nn.Softplus(), nn.Linear(64, observed_dim))
with torch.no_grad():
    x_data, y_data = map_x(z_x), map_y(z_y)

# A random tenth of the samples is held out
order = torch.randperm(n_samples)
test_idx, train_idx = order[:n_samples // 10], order[n_samples // 10:]
```

### Training while tracing both partitions

Each epoch takes minibatches of the training partition through the three
components and one optimiser step. After every epoch the MI is evaluated on
each partition with the whole partition in one score matrix. The two curves are
`train_mi_history` and `test_mi_history` in a library result.

```python
embedding_dim, batch_size, n_epochs = 32, 128, 200

g_net = create_embedding_net(observed_dim, embedding_dim)
h_net = create_embedding_net(observed_dim, embedding_dim)
params = list(g_net.parameters()) + list(h_net.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

nats_to_bits = 1 / torch.log(torch.tensor(2.0)).item()

def evaluate(idx):
    # The MI of one partition in bits, scored as a single batch
    with torch.no_grad():
        scores = separable_critic(g_net(x_data[idx]), h_net(y_data[idx]))
        return infonce_estimator(scores).item() * nats_to_bits

train_history, test_history = [], []
for epoch in range(n_epochs):
    shuffled = train_idx[torch.randperm(len(train_idx))]
    for batch in shuffled.split(batch_size):
        # 1. Embed, 2. score every pairing, 3. estimate the MI
        scores = separable_critic(g_net(x_data[batch]), h_net(y_data[batch]))
        mi_estimate = infonce_estimator(scores)

        # 4. The loss is the negative MI
        loss = -mi_estimate

        # 5. Backpropagate and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_history.append(evaluate(train_idx))
    test_history.append(evaluate(test_idx))
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  train {train_history[-1]:.2f} bits  "
              f"test {test_history[-1]:.2f} bits")
```

### Reading the number off the two curves

The held-out curve rises to a peak and then falls once the critic starts fitting
the training sample itself. The training curve keeps climbing. The max-test
heuristic reads the training curve at the epoch where the held-out curve peaks.
[THEORY.md](THEORY.md#the-reported-number) explains why the training side is the
one to read.

The raw held-out curve is noisy and its highest point can be a fluctuation. The
library smooths it with a median filter followed by a Gaussian filter (one
reasonable choice among several) and takes the peak of the smoothed curve.

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

smoothed_test = gaussian_filter1d(
    median_filter(np.array(test_history), size=5, mode='reflect'),
    sigma=1.0, mode='reflect')
best_epoch = int(np.argmax(smoothed_test))

print(f"best epoch             : {best_epoch}")
print(f"train MI at best epoch : {train_history[best_epoch]:.2f} bits  <- the estimate")
print(f"test MI at best epoch  : {test_history[best_epoch]:.2f} bits")
print(f"train MI at last epoch : {train_history[-1]:.2f} bits")
print(f"exact                  : {true_mi_bits:.2f} bits")
```

![The training and held-out MI over 200 epochs, the smoothed held-out curve and
the epoch its peak selects](../docs/source/_static/anatomy_curves.png)

The training value at the chosen epoch lands close to the exact 4 bits. Reading
the training curve at the last epoch would report whatever the critic had
memorised by then.

The library keeps the weights of the chosen epoch and reports a fresh
evaluation of the training partition with them. `Training(patience=...)` can
also stop the run once the smoothed held-out curve has not improved for that
many epochs. The default patience of 1000 epochs outlasts the default 50-epoch
run and leaves early stopping off.

### The same data through the library

```python
import neural_mi as nmi

result = nmi.run(x_data.numpy(), y_data.numpy(), mode='estimate',
                 model=nmi.Model(hidden_dim=256, embedding_dim=32),
                 training=nmi.Training(n_epochs=200, batch_size=128, learning_rate=1e-3),
                 split=nmi.Split(mode='random'), seed=0)
print(f"library estimate       : {result.mi_estimate:.2f} bits "
      f"at epoch {result.get('best_epoch')}")
```

The library draws its own split and initialisation and lands near the same
value. Everything else it adds (windowing, blocked splits, repeats, ceilings,
bias correction) wraps this loop.
