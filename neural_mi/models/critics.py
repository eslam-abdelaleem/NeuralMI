# neural_mi/models/critics.py
"""Defines the critic models for neural mutual information estimation.

This module contains various critic architectures used to compute a score
function `f(x, y)`, the core of many lower-bound estimators of
mutual information. The critics are designed to be flexible and can be
combined with different embedding models.
"""
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from neural_mi.defaults import default_of


def _batch_size_of(x) -> int:
    """Batch size of x, whether x is a plain tensor or a tuple/list of
    tensors sharing a leading (sample) dimension, BranchedEmbedding's
    compound ``(a_batch, c_batch)`` input, in particular."""
    return (x[0] if isinstance(x, (tuple, list)) else x).shape[0]


def _device_of(x) -> torch.device:
    return (x[0] if isinstance(x, (tuple, list)) else x).device


def _flat(x) -> torch.Tensor:
    """x as one ``(batch, features)`` matrix. A pair X (the joint term of a
    difference) is flattened part by part and the parts are joined."""
    if isinstance(x, (tuple, list)):
        return torch.cat([part.reshape(part.shape[0], -1) for part in x], dim=1)
    return x.reshape(x.shape[0], -1)


# The most values one block of pair activations may hold: rows of the score
# matrix times N pairs times the head's hidden width. A training batch of a few
# hundred samples fits in one block and a large evaluation set in a few dozen.
_PAIR_BLOCK_ELEMENTS = 2 ** 24


def _is_plain_mlp(net) -> bool:
    """An MLP whose first module is a linear layer applied to the whole input."""
    network = getattr(net, 'network', None)
    return (isinstance(network, nn.Sequential) and len(network) > 0
            and isinstance(network[0], nn.Linear) and hasattr(net, 'output_layer'))


def _score_pairs(mlp: nn.Module, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``mlp([a_i, b_j])`` for every pair, as an ``(len(a), len(b))`` matrix.

    The first layer is affine, so its value on a pair is the layer applied to
    ``[a_i, 0]`` plus the layer applied to ``[0, b_j]`` minus one bias. Both
    sides pass through that layer once and together, which also runs a
    spectral norm's power iteration once per call. The rest of the network runs
    on the pair sums in blocks of at most ``_PAIR_BLOCK_ELEMENTS`` values.
    """
    first, rest = mlp.network[0], mlp.network[1:]
    n_a, n_b = a.shape[0], b.shape[0]
    both = torch.cat([torch.cat([a, a.new_zeros(n_a, b.shape[1])], dim=1),
                      torch.cat([b.new_zeros(n_b, a.shape[1]), b], dim=1)], dim=0)
    hidden = first(both)
    h_a, h_b = hidden[:n_a], hidden[n_a:]
    if first.bias is not None:
        h_b = h_b - first.bias
    width = hidden.shape[1]
    rows = max(1, _PAIR_BLOCK_ELEMENTS // max(1, n_b * width))
    blocks = []
    for start in range(0, n_a, rows):
        pre = h_a[start:start + rows].unsqueeze(1) + h_b.unsqueeze(0)
        blocks.append(mlp.output_layer(rest(pre.reshape(-1, width))).view(-1, n_b))
    return torch.cat(blocks, dim=0)


def _slice_batch(x, start: int, end: int):
    """Slice the leading (batch) dimension of x, whether x is a plain
    tensor or a tuple/list of tensors sharing that dimension."""
    if isinstance(x, (tuple, list)):
        return tuple(t[start:end] for t in x)
    return x[start:end]


def _gives_own_embeddings(critic: nn.Module) -> bool:
    """Whether a critic gives its embeddings through a ``get_training_embeddings`` of its own, as a custom
    critic can, rather than through an encoder per side."""
    own = getattr(type(critic), 'get_training_embeddings', None)
    return own is not None and own is not BaseCritic.get_training_embeddings


def side_embedding_sizes(critic: nn.Module, x, y) -> Optional[Tuple[int, int]]:
    """The size of the embedding of each side a critic gives, which a decoder rebuilds that side from, or
    None when it gives none.

    The separable and hybrid critics embed each side with an encoder of its own,
    and a custom critic gives its embeddings through a ``get_training_embeddings``
    of its own. The concat critic scores the pair without embedding either side.
    The sizes are read on the rows ``x`` and ``y`` in eval mode, without gradients
    and on a forked random stream, so that asking changes nothing in the training
    that follows.
    """
    if not hasattr(critic, 'embedding_net_x') and not _gives_own_embeddings(critic):
        return None
    was_training = critic.training
    critic.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            z_x, z_y = critic.get_training_embeddings(x, y)
    finally:
        critic.train(was_training)
    return int(z_x.shape[-1]), int(z_y.shape[-1])


class BaseCritic(nn.Module):
    """Abstract base class for critic models.

    All critic models should inherit from this class. The main role of a critic
    is to produce a score matrix indicating the relationship between pairs of
    samples from two variables, X and Y.
    """
    def __init__(self):
        super().__init__()

    def _get_embeddings_and_kl(self, x_out: Any, y_out: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper to unpack embeddings and KL loss from variational models."""
        kl_loss_x = torch.tensor(0.0, device=x_out[0].device if isinstance(x_out, tuple) else x_out.device)
        kl_loss_y = torch.tensor(0.0, device=y_out[0].device if isinstance(y_out, tuple) else y_out.device)

        if isinstance(x_out, tuple):
            x_embedded, kl_loss_x = x_out
        else:
            x_embedded = x_out

        if isinstance(y_out, tuple):
            y_embedded, kl_loss_y = y_out
        else:
            y_embedded = y_out
            
        total_kl_loss = kl_loss_x + kl_loss_y
        return x_embedded, y_embedded, total_kl_loss

    def _compute_embeddings_chunked(self, x, y, net_x, net_y, max_n_batches, use_variational):
        """Computes embeddings efficiently, accumulating KL from the variational wrappers.

        ``VariationalWrapper.forward`` already returns a per-sample-mean KL (raw KL
        summed over the embedding dimensions and batch, then divided by batch size).
        We accumulate those per-sample means across chunks and average them, no
        additional division by the full batch size is applied, as that would
        double-count the normalization already performed inside the wrapper.
        """
        batch_size = _batch_size_of(x)
        n_chunks = 0

        # Fast path for small datasets, wrapper already gives per-sample mean KL.
        if batch_size <= max_n_batches:
            x_out = net_x(x)
            y_out = net_y(y)
            x_embedded, y_embedded, total_kl = self._get_embeddings_and_kl(x_out, y_out)
            return x_embedded, y_embedded, total_kl

        # Chunked processing to prevent OOM
        x_embeds, y_embeds = [], []
        total_kl_acc = torch.tensor(0.0, device=_device_of(x))

        for i in range(0, batch_size, max_n_batches):
            end_idx = min(i + max_n_batches, batch_size)
            chunk_size = end_idx - i

            x_out = net_x(_slice_batch(x, i, end_idx))
            y_out = net_y(_slice_batch(y, i, end_idx))

            x_emb, y_emb, kl = self._get_embeddings_and_kl(x_out, y_out)

            x_embeds.append(x_emb)
            y_embeds.append(y_emb)

            if use_variational and not isinstance(kl, float):
                # kl is already per-sample mean for this chunk; weight by chunk size
                # so we recover the true sum of KL values across the chunk.
                total_kl_acc = total_kl_acc + kl * chunk_size
                n_chunks += 1

        x_embedded = torch.cat(x_embeds, dim=0)
        y_embedded = torch.cat(y_embeds, dim=0)

        if use_variational and n_chunks > 0:
            # Divide by total batch_size to get the true per-sample mean over the
            # full batch (weighted average across possibly unequal chunk sizes).
            total_kl = total_kl_acc.to(x_embedded.device) / batch_size
        else:
            total_kl = torch.tensor(0.0, device=x_embedded.device)

        return x_embedded, y_embedded, total_kl

    
    def get_embeddings(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public API to extract chunked embeddings from a trained critic."""
        if hasattr(self, 'embedding_net_x'):
            net_x, net_y = self.embedding_net_x, self.embedding_net_y
        elif _gives_own_embeddings(self):
            # A critic that gives its embeddings itself is read the same way
            # wherever its embeddings are read: decoders, returned embeddings.
            return self.get_training_embeddings(x, y)
        else:
            # ConcatCritic: no separate embedding networks, return flat inputs.
            # This is semantically honest, concat critics have no separable embedding.
            return _flat(x), _flat(y)
            
        max_n = getattr(self, 'max_n_batches', 512)
        use_var = getattr(self, 'use_variational', False)
        
        zx, zy, _ = self._compute_embeddings_chunked(x, y, net_x, net_y, max_n, use_var)
        return zx, zy

    def get_training_embeddings(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Return embeddings with gradient flow (for decoder reconstruction loss).

        Unlike :meth:`get_embeddings`, this method does NOT wrap in
        ``torch.no_grad()`` so gradients can flow back through the encoder.
        Only called during training when ``use_decoder=True``.

        Parameters
        ----------
        x, y : torch.Tensor
            Input batch tensors.

        Returns
        -------
        tuple of (z_x, z_y) : torch.Tensor
            Embedding tensors, each of shape ``(batch, embedding_dim)``.
        """
        if hasattr(self, 'embedding_net_x'):
            net_x, net_y = self.embedding_net_x, self.embedding_net_y
        else:
            # ConcatCritic has no separate embedding networks
            return _flat(x), _flat(y)
        max_n = getattr(self, 'max_n_batches', 512)
        use_var = getattr(self, 'use_variational', False)
        zx, zy, _ = self._compute_embeddings_chunked(x, y, net_x, net_y, max_n, use_var)
        return zx, zy

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class SeparableCritic(BaseCritic):
    """A critic with a separable architecture based on a dot product."""
    def __init__(self, 
                 embedding_net_x: nn.Module, *, 
                 embedding_net_y: Optional[nn.Module] = None,
                 embedding_dim: int = None, 
                 max_n_batches: int = default_of('max_n_batches'), 
                 use_variational: bool = default_of('use_variational'),
                 **kwargs):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x
        self.embedding_dim = embedding_dim
        self.max_n_batches = max_n_batches
        self.use_variational = use_variational

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_embedded, y_embedded, total_kl_loss = self._compute_embeddings_chunked(
            x, y, self.embedding_net_x, self.embedding_net_y, 
            self.max_n_batches, self.use_variational
        )
        scores = torch.matmul(x_embedded, y_embedded.t())
        return scores, total_kl_loss

class HybridCritic(BaseCritic):
    """A hybrid critic that embeds inputs independently, then scores their concatenation.
    
    Combines the computational efficiency of the SeparableCritic's independent 
    embeddings with the expressive interaction of the ConcatCritic.
    """
    def __init__(self, 
                 embedding_net_x: nn.Module, *, 
                 embedding_net_y: Optional[nn.Module] = None,
                 decision_head: nn.Module,
                 embedding_dim: int = None, 
                 max_n_batches: int = default_of('max_n_batches'), 
                 use_variational: bool = default_of('use_variational'),
                 **kwargs):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x
        self.decision_head = decision_head
        self.embedding_dim = embedding_dim
        self.max_n_batches = max_n_batches
        self.use_variational = use_variational

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x may be a plain tensor or a tuple (BranchedEmbedding's compound
        # (a_batch, c_batch) input) -- _batch_size_of handles both.
        batch_size = _batch_size_of(x)

        # 1. Embed inputs (chunked over samples to bound encoder memory).
        x_embedded, y_embedded, total_kl_loss = self._compute_embeddings_chunked(
            x, y, self.embedding_net_x, self.embedding_net_y,
            self.max_n_batches, self.use_variational
        )

        # 2. Score all N² pairs. The head built by build_critic is a plain MLP
        #    and takes the fast route; any other head scores pairs in row-chunks
        #    so the full (N², 2d) pair tensor is never materialized at once.
        if _is_plain_mlp(self.decision_head):
            return _score_pairs(self.decision_head, x_embedded, y_embedded), total_kl_loss
        chunk_rows = max(1, self.max_n_batches // batch_size)
        scores = torch.zeros(batch_size, batch_size, device=x_embedded.device)
        y_exp = y_embedded.unsqueeze(0)  # (1, N, d), shared view, no copy

        for start in range(0, batch_size, chunk_rows):
            end = min(start + chunk_rows, batch_size)
            n_rows = end - start
            pairs = torch.cat(
                [x_embedded[start:end].unsqueeze(1).expand(-1, batch_size, -1),
                 y_exp.expand(n_rows, -1, -1)],
                dim=2
            ).reshape(n_rows * batch_size, -1)
            scores[start:end] = self.decision_head(pairs).view(n_rows, batch_size)

        return scores, total_kl_loss

class ConcatCritic(BaseCritic):
    """A critic that processes raw concatenated input pairs.

    The network receives ``[x_i, y_j]`` for every (i, j) pair and learns a scalar
    score directly from the concatenation.  This makes it the most expressive critic
    but also the most expensive (O(N²) forward passes per batch).

    .. note:: **Variational mode with ConcatCritic**

        With ``use_variational=True`` the variational layer sits on the output of
        the network, the score of the pair ``[x_i, y_j]``. Each score becomes a
        draw from a Gaussian whose mean and variance the network produces, and
        the KL term pulls every score toward the standard normal prior. That
        regularises the critic, but there is no embedding of X or of Y for it to
        compress, so it carries no information-bottleneck reading. The separable
        and hybrid critics give that reading.
    """
    def __init__(self,
                 embedding_net: nn.Module,
                 embedding_dim: int = None,
                 max_n_batches: int = default_of('max_n_batches'),
                 use_variational: bool = default_of('use_variational'),
                 scaling: Optional[Tuple[nn.Module, nn.Module]] = None,
                 **kwargs):
        super().__init__()
        self.embedding_net = embedding_net
        self.embedding_dim = embedding_dim
        self.max_n_batches = max_n_batches
        self.use_variational = use_variational
        # Model(input_scaling='standard'): each side's ChannelScaling, applied
        # before the pair is formed.
        self.scaling = nn.ModuleList(scaling) if scaling is not None else None

    def forward(self, x, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaling is not None:
            x, y = self.scaling[0](x), self.scaling[1](y)
        batch_size = _batch_size_of(x)
        x_flat, y_flat = _flat(x), _flat(y)
        if not self.use_variational and _is_plain_mlp(self.embedding_net):
            return (_score_pairs(self.embedding_net, x_flat, y_flat),
                    torch.tensor(0.0, device=y.device))

        # Row-wise chunking: process chunk_rows rows of x per iteration.
        # Each chunk contains chunk_rows * N pairs, bounding peak memory to
        # max_n_batches pairs, same budget as the original flat-index loop.
        chunk_rows = max(1, self.max_n_batches // batch_size)
        scores = torch.zeros(batch_size, batch_size, device=y.device)
        total_kl_acc = torch.tensor(0.0, device=y.device)
        n_pair_chunks = 0
        y_exp = y_flat.unsqueeze(0)  # (1, N, dy), shared view, no copy

        for start in range(0, batch_size, chunk_rows):
            end = min(start + chunk_rows, batch_size)
            n_rows = end - start
            n_pairs = n_rows * batch_size
            pairs = torch.cat(
                [x_flat[start:end].unsqueeze(1).expand(-1, batch_size, -1),
                 y_exp.expand(n_rows, -1, -1)],
                dim=2
            ).reshape(n_pairs, -1)
            out = self.embedding_net(pairs)

            if isinstance(out, tuple):
                chunk_scores, kl = out
                if self.use_variational and not isinstance(kl, float):
                    # kl is per-sample mean for this pair chunk; weight by n_pairs
                    # to recover the true KL sum, matching the accumulation in
                    # _compute_embeddings_chunked.
                    total_kl_acc = total_kl_acc + kl * n_pairs
                    n_pair_chunks += 1
            else:
                chunk_scores = out

            scores[start:end] = chunk_scores.squeeze(-1).view(n_rows, batch_size)

        if self.use_variational and n_pair_chunks > 0:
            kl_tensor = total_kl_acc.to(y.device) / (batch_size * batch_size)
        else:
            kl_tensor = torch.tensor(0.0, device=y.device)

        return scores, kl_tensor

def fit_input_scaling(critic: nn.Module, x, y) -> None:
    """Set every input scaling of ``critic`` (``Model(input_scaling='standard')``) from the training rows ``x``
    and ``y``. A shared encoder takes its statistics from both sides' rows. Nothing happens without one."""
    from neural_mi.models.embeddings import InputScaling
    scaling = getattr(critic, 'scaling', None)
    if scaling is not None:
        scaling[0].fit(x)
        scaling[1].fit(y)
        return

    def find(net):
        return next((m for m in net.modules() if isinstance(m, InputScaling)), None) if net is not None else None

    sx, sy = find(getattr(critic, 'embedding_net_x', None)), find(getattr(critic, 'embedding_net_y', None))
    if sx is not None and sx is sy:
        sx.fit(torch.cat([x, y], dim=0))
        return
    if sx is not None:
        sx.fit(x)
    if sy is not None:
        sy.fit(y)


def has_input_scaling(critic: nn.Module) -> bool:
    """Whether ``critic`` scales its inputs (``Model(input_scaling='standard')``)."""
    from neural_mi.models.embeddings import ChannelScaling
    return any(isinstance(m, ChannelScaling) for m in critic.modules())
