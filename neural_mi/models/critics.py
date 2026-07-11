# neural_mi/models/critics.py
"""Defines the critic models for neural mutual information estimation.

This module contains various critic architectures used to compute a score
function `f(x, y)`, which is the core of many lower-bound estimators of
mutual information. The critics are designed to be flexible and can be
combined with different embedding models.
"""
import torch
import torch.nn as nn
from typing import get_type_hints, get_origin, Optional, Tuple

class BaseCritic(nn.Module):
    """Abstract base class for critic models.

    All critic models should inherit from this class. The main role of a critic
    is to produce a score matrix indicating the relationship between pairs of
    samples from two variables, X and Y.
    """
    def __init__(self):
        super().__init__()
        
    def _get_embeddings_and_kl(self, x_out: any, y_out: any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        We accumulate those per-sample means across chunks and average them — no
        additional division by the full batch size is applied, as that would
        double-count the normalization already performed inside the wrapper.
        """
        batch_size = x.shape[0]
        n_chunks = 0

        # Fast path for small datasets — wrapper already gives per-sample mean KL.
        if batch_size <= max_n_batches:
            x_out = net_x(x)
            y_out = net_y(y)
            x_embedded, y_embedded, total_kl = self._get_embeddings_and_kl(x_out, y_out)
            return x_embedded, y_embedded, total_kl

        # Chunked processing to prevent OOM
        x_embeds, y_embeds = [], []
        total_kl_acc = torch.tensor(0.0, device=x.device)

        for i in range(0, batch_size, max_n_batches):
            end_idx = min(i + max_n_batches, batch_size)
            chunk_size = end_idx - i

            x_out = net_x(x[i:end_idx])
            y_out = net_y(y[i:end_idx])

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
        else:
            # ConcatCritic: no separate embedding networks, return flat inputs.
            # This is semantically honest — concat critics have no separable embedding.
            return x.view(x.shape[0], -1), y.view(y.shape[0], -1)
            
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
            Embedding tensors, each of shape ``(batch, embed_dim)``.
        """
        if hasattr(self, 'embedding_net_x'):
            net_x, net_y = self.embedding_net_x, self.embedding_net_y
        else:
            # ConcatCritic has no separate embedding networks
            return x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
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
                 embed_dim: int = None, 
                 max_n_batches: int = 512, 
                 use_variational: bool = False,
                 **kwargs):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x
        self.embed_dim = embed_dim
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
                 embed_dim: int = None, 
                 max_n_batches: int = 512, 
                 use_variational: bool = False,
                 **kwargs):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x
        self.decision_head = decision_head
        self.embed_dim = embed_dim
        self.max_n_batches = max_n_batches
        self.use_variational = use_variational

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        # 1. Embed inputs (chunked over samples to bound encoder memory).
        x_embedded, y_embedded, total_kl_loss = self._compute_embeddings_chunked(
            x, y, self.embedding_net_x, self.embedding_net_y,
            self.max_n_batches, self.use_variational
        )

        # 2. Score all N² pairs in row-chunks so the full (N², 2d) pair tensor
        #    is never materialized at once — same pattern as ConcatCritic.forward.
        chunk_rows = max(1, self.max_n_batches // batch_size)
        scores = torch.zeros(batch_size, batch_size, device=x_embedded.device)
        y_exp = y_embedded.unsqueeze(0)  # (1, N, d) — shared view, no copy

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

        Setting ``use_variational=True`` together with ``critic_type='concat'`` is
        supported but has a different theoretical interpretation than with separable
        or hybrid critics.  Here the variational wrapper is applied to the
        *concatenated pair* ``[x_i, y_j]``, not to individual samples, so the KL
        term measures uncertainty over the *pair* representation rather than over
        each variable's marginal distribution.  This departs from the standard
        Information Bottleneck formulation described in the docs.  The training will
        run without error, but the KL regularisation effect is weaker and harder to
        interpret.  Unless you have a specific reason to use this combination,
        prefer ``critic_type='separable'`` or ``'hybrid'`` when
        ``use_variational=True``.
    """
    def __init__(self,
                 embedding_net: nn.Module,
                 embed_dim: int = None,
                 max_n_batches: int = 512,
                 use_variational: bool = False,
                 **kwargs):
        super().__init__()
        self.embedding_net = embedding_net
        self.embed_dim = embed_dim
        self.max_n_batches = max_n_batches
        self.use_variational = use_variational

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        y_flat = y.view(batch_size, -1)

        # Row-wise chunking: process chunk_rows rows of x per iteration.
        # Each chunk contains chunk_rows * N pairs, bounding peak memory to
        # max_n_batches pairs — same budget as the original flat-index loop.
        chunk_rows = max(1, self.max_n_batches // batch_size)
        scores = torch.zeros(batch_size, batch_size, device=x.device)
        total_kl_acc = torch.tensor(0.0, device=x.device)
        n_pair_chunks = 0
        y_exp = y_flat.unsqueeze(0)  # (1, N, dy) — shared view, no copy

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
            kl_tensor = total_kl_acc.to(x.device) / (batch_size * batch_size)
        else:
            kl_tensor = torch.tensor(0.0, device=x.device)

        return scores, kl_tensor