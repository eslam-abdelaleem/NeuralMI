# neural_mi/models/critics.py
"""Defines the critic models for neural mutual information estimation.

This module contains various critic architectures used to compute a score
function `f(x, y)`, which is the core of many lower-bound estimators of
mutual information. The critics are designed to be flexible and can be
combined with different embedding models.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

class BaseCritic(nn.Module):
    """Abstract base class for critic models.

    All critic models should inherit from this class. The main role of a critic
    is to produce a score matrix indicating the relationship between pairs of
    samples from two variables, X and Y.

    The `forward` method must be implemented by all subclasses.
    """
    def __init__(self):
        """Initializes the BaseCritic."""
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the score matrix for batches of samples from X and Y.

        Parameters
        ----------
        x : torch.Tensor
            A batch of samples from the first variable, with shape (batch_size, ...).
        y : torch.Tensor
            A batch of samples from the second variable, with shape (batch_size, ...).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - **scores** (*torch.Tensor*): A `(batch_size, batch_size)` tensor
              where `scores[i, j]` is the critic's output for the pair `(x[i], y[j])`.
            - **kl_loss** (*torch.Tensor*): A scalar tensor representing the sum
              of KL divergence losses from any variational embedding models used.
              Returns 0.0 if no variational models are used.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError

class SeparableCritic(BaseCritic):
    """A critic with a separable architecture based on a dot product.

    This critic computes embeddings for each sample from X and Y independently
    using two embedding networks, `g` and `h`. The score for each pair
    `(x_i, y_j)` is then calculated as the dot product of their embeddings:
    `f(x, y) = g(x)^T h(y)`.

    This is one of the most common and computationally efficient critic architectures.

    Attributes
    ----------
    embedding_net_x : nn.Module
        The network `g` used to embed samples from X.
    embedding_net_y : nn.Module
        The network `h` used to embed samples from Y. If not provided,
        `embedding_net_x` is used for both (a shared embedding network).
    """
    def __init__(self, embedding_net_x: nn.Module, embedding_net_y: Optional[nn.Module] = None):
        """
        Parameters
        ----------
        embedding_net_x : nn.Module
            The network used to embed samples from X.
        embedding_net_y : nn.Module, optional
            The network used to embed samples from Y. If None, a single network
            is shared for both variables. Defaults to None.
        """
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes scores via dot product of embeddings.

        If the embedding networks are variational (e.g., `VarMLP`), this method
        unpacks the returned tuple `(embedding, kl_loss)` and sums the KL losses.
        """
        x_out = self.embedding_net_x(x)
        y_out = self.embedding_net_y(y)

        x_embedded, y_embedded, total_kl_loss = self._get_embeddings_and_kl(x_out, y_out)

        scores = torch.matmul(x_embedded, y_embedded.t())
        return scores, total_kl_loss

class BilinearCritic(BaseCritic):
    """A critic using two embedding networks and a learnable bilinear layer.

    This architecture allows for a more powerful, learnable interaction between
    the embeddings of X and Y compared to a simple dot product. The score is
    computed as `f(x, y) = g(x)^T W h(y)`, where `g` and `h` are the
    embedding networks and `W` is a learnable matrix.

    Attributes
    ----------
    embedding_net_x : nn.Module
        The network `g` used to embed samples from X.
    embedding_net_y : nn.Module
        The network `h` used to embed samples from Y.
    similarity_layer : nn.Bilinear
        The learnable bilinear layer `W` that computes the similarity score.
    """
    def __init__(self, embedding_net_x: nn.Module, embedding_net_y: nn.Module, embed_dim: int):
        """
        Parameters
        ----------
        embedding_net_x : nn.Module
            The network used to embed samples from X.
        embedding_net_y : nn.Module
            The network used to embed samples from Y.
        embed_dim : int
            The dimension of the embedding vectors produced by the embedding networks.
        """
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y
        self.similarity_layer = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_out = self.embedding_net_x(x)
        y_out = self.embedding_net_y(y)

        x_embedded, y_embedded, total_kl_loss = self._get_embeddings_and_kl(x_out, y_out)
        
        # Create all pairs of embeddings (x_i, y_j)
        x_tiled = x_embedded.repeat_interleave(batch_size, dim=0)
        y_tiled = y_embedded.repeat(batch_size, 1)
        
        scores = self.similarity_layer(x_tiled, y_tiled)
        return scores.view(batch_size, batch_size), total_kl_loss

        
class ConcatCritic(BaseCritic):
    """A critic that processes concatenated input pairs.

    This critic flattens and concatenates each pair of samples `(x_i, y_j)`
    and feeds the resulting vector into a single, shared network to produce a
    scalar score. This allows the critic to learn a joint function over the
    raw input spaces of X and Y.

    Attributes
    ----------
    embedding_net : nn.Module
        The shared network that processes the concatenated input pairs.
    """
    def __init__(self, embedding_net: nn.Module):
        """
        Parameters
        ----------
        embedding_net : nn.Module
            The shared network that takes the concatenated `(x, y)` pair and
            outputs a scalar score. Its input dimension must match the sum of
            the flattened dimensions of x and y.
        """
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes scores by applying a network to all `(x, y)` pairs."""
        batch_size = x.size(0)
        x_flat, y_flat = x.view(batch_size, -1), y.view(batch_size, -1)
        x_tiled = x_flat.repeat_interleave(batch_size, dim=0)
        y_tiled = y_flat.repeat(batch_size, 1)
        xy_pairs = torch.cat((x_tiled, y_tiled), dim=1)
        
        out = self.embedding_net(xy_pairs)
        if isinstance(out, tuple):
            scores, kl_loss = out
        else:
            scores = out
            kl_loss = torch.tensor(0.0, device=scores.device)
            
        return scores.view(batch_size, batch_size), kl_loss

# class ConcatCriticCNN(BaseCritic):
#     """A critic for CNNs that concatenates feature maps from two towers.

#     This critic is designed for structured sequential data (e.g., time series).
#     It uses two separate CNN feature extractors (towers) to process inputs
#     from X and Y. The resulting feature maps are then concatenated and passed
#     to a final decision head to produce a score.

#     Note that this critic operates on the *feature maps* produced by the
#     convolutional layers, not on final embedding vectors. This allows the
#     decision head to learn interactions between the spatial/temporal features
#     of the two inputs.

#     Attributes
#     ----------
#     cnn_x : nn.Module
#         The CNN feature extractor for samples from X.
#     cnn_y : nn.Module
#         The CNN feature extractor for samples from Y.
#     decision_head : nn.Module
#         The final network that takes the concatenated feature maps and outputs
#         a scalar score.
#     """
#     def __init__(self, cnn_x: nn.Module, cnn_y: nn.Module, decision_head: nn.Module):
#         """
#         Parameters
#         ----------
#         cnn_x : nn.Module
#             The CNN-based embedding network for samples from X.
#         cnn_y : nn.Module
#             The CNN-based embedding network for samples from Y.
#         decision_head : nn.Module
#             The final network that takes the concatenated embeddings and outputs a score.
#         """
#         super().__init__()
#         self.cnn_x = cnn_x
#         self.cnn_y = cnn_y
#         self.decision_head = decision_head
        
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size = x.size(0)
        
#         # Create all pairs of windows (x_i, y_j)
#         x_tiled = x.repeat_interleave(batch_size, dim=0)
#         # y is tiled to match x_tiled. For each x_i, we need all y_j.
#         y_tiled = y.repeat(batch_size, 1, 1)

#         # Extract features with the CNNs
#         features_x = self.cnn_x(x_tiled)
#         features_y = self.cnn_y(y_tiled)
        
#         # Concatenate the output embedding vectors
#         combined_features = torch.cat([features_x, features_y], dim=1)
        
#         # Pass through the final decision head
#         scores = self.decision_head(combined_features)
        
#         return scores.view(batch_size, batch_size), torch.tensor(0.0, device=scores.device)

# A more efficient version
class ConcatCriticCNN(BaseCritic):
    """A critic for CNNs that concatenates embeddings from two towers.

    This critic is designed for structured sequential data (e.g., time series).
    It uses two separate CNN feature extractors (towers) to process inputs
    from X and Y, creating an embedding for each sample. The resulting
    embedding vectors are then concatenated and passed to a final decision
    head to produce a score.

    This "siamese" or "two-tower" approach is computationally efficient as it
    runs the expensive CNN operation only once per batch element.

    Attributes
    ----------
    cnn_x : nn.Module
        The CNN embedding network for samples from X.
    cnn_y : nn.Module
        The CNN embedding network for samples from Y.
    decision_head : nn.Module
        The final network that takes the concatenated embeddings and outputs
        a scalar score.
    """
    def __init__(self, cnn_x: nn.Module, cnn_y: nn.Module, decision_head: nn.Module):
        """
        Parameters
        ----------
        cnn_x : nn.Module
            The CNN-based embedding network for samples from X.
        cnn_y : nn.Module
            The CNN-based embedding network for samples from Y.
        decision_head : nn.Module
            The final network that takes the concatenated embeddings and outputs a score.
        """
        super().__init__()
        self.cnn_x = cnn_x
        self.cnn_y = cnn_y
        self.decision_head = decision_head

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes scores by applying a decision head to all pairs of embeddings."""
        batch_size = x.size(0)

        # 1. Get embeddings for each batch element (computationally efficient)
        # These calls now return (embedding, kl_loss) tuples if using VarCNN
        x_out = self.cnn_x(x)
        y_out = self.cnn_y(y)

        x_embedded, y_embedded, total_kl_loss = self._get_embeddings_and_kl(x_out, y_out)
        
        # 2. Create all pairs of embeddings for the decision head
        x_tiled = x_embedded.repeat_interleave(batch_size, dim=0)
        y_tiled = y_embedded.repeat(batch_size, 1)

        # 3. Concatenate and get scores
        combined_features = torch.cat([x_tiled, y_tiled], dim=1)
        scores = self.decision_head(combined_features)
        
        return scores.view(batch_size, batch_size), total_kl_loss