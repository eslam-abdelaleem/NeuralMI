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
        """Computes embeddings efficiently, with a single consistent KL normalization."""
        batch_size = x.shape[0]
        
        # Fast path for small datasets
        if batch_size <= max_n_batches:
            x_out = net_x(x)
            y_out = net_y(y)
            x_embedded, y_embedded, total_kl = self._get_embeddings_and_kl(x_out, y_out)
            # Apply single consistent normalization by batch size
            if use_variational and not isinstance(total_kl, float):
                total_kl = total_kl / batch_size
            return x_embedded, y_embedded, total_kl
            
        # Chunked processing to prevent OOM
        x_embeds, y_embeds = [], []
        total_kl_raw = torch.tensor(0.0)
        
        for i in range(0, batch_size, max_n_batches):
            end_idx = min(i + max_n_batches, batch_size)
            
            x_out = net_x(x[i:end_idx])
            y_out = net_y(y[i:end_idx])
            
            x_emb, y_emb, kl = self._get_embeddings_and_kl(x_out, y_out)
            
            x_embeds.append(x_emb)
            y_embeds.append(y_emb)
            
            if use_variational and not isinstance(kl, float):
                # Accumulate raw KL — normalize once after all chunks
                total_kl_raw = total_kl_raw + kl
                
        x_embedded = torch.cat(x_embeds, dim=0)
        y_embedded = torch.cat(y_embeds, dim=0)
        
        if use_variational and not isinstance(total_kl_raw, float):
            # Single normalization by full batch size — consistent with fast path
            total_kl = total_kl_raw.to(x_embedded.device) / batch_size
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
        
        # 1. Embed inputs using the efficient chunked method
        x_embedded, y_embedded, total_kl_loss = self._compute_embeddings_chunked(
            x, y, self.embedding_net_x, self.embedding_net_y, 
            self.max_n_batches, self.use_variational
        )
        
        # 2. Compute the N^2 score matrix safely by generating pairs dynamically
        scores = torch.zeros(batch_size * batch_size, device=x_embedded.device)
        pair_batch_size = self.max_n_batches
        
        for i in range(0, batch_size * batch_size, pair_batch_size):
            end_idx = min(i + pair_batch_size, batch_size * batch_size)
            
            # Map flat index back to row (x) and column (y)
            idx = torch.arange(i, end_idx, device=x_embedded.device)
            row_idx = idx // batch_size
            col_idx = idx % batch_size
            
            pairs = torch.cat([x_embedded[row_idx], y_embedded[col_idx]], dim=1)
            chunk_scores = self.decision_head(pairs).squeeze()
            scores[i:end_idx] = chunk_scores
            
        return scores.view(batch_size, batch_size), total_kl_loss

class ConcatCritic(BaseCritic):
    """A critic that processes raw concatenated input pairs."""
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
        x_flat, y_flat = x.view(batch_size, -1), y.view(batch_size, -1)
        
        scores = torch.zeros(batch_size * batch_size, device=x.device)
        total_kl_raw = torch.tensor(0.0)
        
        pair_batch_size = self.max_n_batches
        
        for i in range(0, batch_size * batch_size, pair_batch_size):
            end_idx = min(i + pair_batch_size, batch_size * batch_size)
            idx = torch.arange(i, end_idx, device=x.device)
            row_idx = idx // batch_size
            col_idx = idx % batch_size
            pairs = torch.cat([x_flat[row_idx], y_flat[col_idx]], dim=1)
            out = self.embedding_net(pairs)
            
            if isinstance(out, tuple):
                chunk_scores, kl = out
                if self.use_variational and not isinstance(kl, float):
                    total_kl_raw = total_kl_raw + kl
            else:
                chunk_scores = out
                
            scores[i:end_idx] = chunk_scores.squeeze()

        if self.use_variational and not isinstance(total_kl_raw, float):
            kl_tensor = total_kl_raw.to(x.device) / (batch_size * batch_size)
        else:
            kl_tensor = torch.tensor(0.0, device=x.device)

        return scores.view(batch_size, batch_size), kl_tensor


