# neural_mi/models/critics.py

import torch
import torch.nn as nn
from typing import Optional

class BaseCritic(nn.Module):
    """Abstract base class for critic models.

    All critic models should inherit from this class and implement the `forward` method.
    The role of a critic is to produce a score matrix indicating the similarity or
    relationship between pairs of samples from two variables, X and Y.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the score matrix for pairs of samples.

        Parameters
        ----------
        x : torch.Tensor
            A batch of samples from the first variable, with shape (batch_size, ...).
        y : torch.Tensor
            A batch of samples from the second variable, with shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            A (batch_size, batch_size) tensor of scores, where `scores[i, j]` is the
            critic's output for the pair `(x[i], y[j])`.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError

class SeparableCritic(BaseCritic):
    """A critic with a separable architecture based on a dot product.

    This critic computes embeddings for each sample from X and Y independently
    using two embedding networks, and then calculates the score for each pair
    `(x_i, y_j)` as the dot product of their embeddings.

    Attributes
    ----------
    embedding_net_x : nn.Module
        The network used to embed samples from X.
    embedding_net_y : nn.Module
        The network used to embed samples from Y. If not provided, `embedding_net_x`
        is used for both.
    """
    def __init__(self, embedding_net_x: nn.Module, embedding_net_y: Optional[nn.Module] = None):
        """
        Parameters
        ----------
        embedding_net_x : nn.Module
            The network used to embed samples from X.
        embedding_net_y : nn.Module, optional
            The network used to embed samples from Y. If None, `embedding_net_x` is
            used for both variables (a shared embedding network). Defaults to None.
        """
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_embedded = self.embedding_net_x(x)
        y_embedded = self.embedding_net_y(y)
        return torch.matmul(x_embedded, y_embedded.t())

class BilinearCritic(BaseCritic):
    """A critic using two embedding networks and a learnable bilinear layer.

    This architecture allows for a more powerful, learnable interaction between
    the embeddings of X and Y compared to a simple dot product.

    Attributes
    ----------
    embedding_net_x : nn.Module
        The network used to embed samples from X.
    embedding_net_y : nn.Module
        The network used to embed samples from Y.
    similarity_layer : nn.Bilinear
        The bilinear layer that computes the similarity score.
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_embedded = self.embedding_net_x(x)
        y_embedded = self.embedding_net_y(y)
        
        # Create all pairs of embeddings (x_i, y_j)
        x_tiled = x_embedded.repeat_interleave(batch_size, dim=0)
        y_tiled = y_embedded.repeat(batch_size, 1)
        
        scores = self.similarity_layer(x_tiled, y_tiled)
        return scores.view(batch_size, batch_size)
        
class ConcatCritic(BaseCritic):
    """A critic that concatenates input pairs and processes them with a single network.

    This critic flattens and concatenates each pair of samples `(x_i, y_j)` and
    feeds the resulting vector into a single, shared embedding network to produce a score.

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
            outputs a scalar score.
        """
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat, y_flat = x.view(batch_size, -1), y.view(batch_size, -1)
        x_tiled = x_flat.repeat_interleave(batch_size, dim=0)
        y_tiled = y_flat.repeat(batch_size, 1)
        xy_pairs = torch.cat((x_tiled, y_tiled), dim=1)
        scores = self.embedding_net(xy_pairs)
        return scores.view(batch_size, batch_size)

class ConcatCriticCNN(BaseCritic):
    """A critic that uses two CNN towers and concatenates their output embeddings.

    This critic is designed for structured data like time series or images. It uses
    two separate CNNs to extract embedding vectors from each input, then
    concatenates these vectors, and passes them to a final decision head to
    produce a score.

    Attributes
    ----------
    cnn_x : nn.Module
        The CNN-based embedding network for samples from X.
    cnn_y : nn.Module
        The CNN-based embedding network for samples from Y.
    decision_head : nn.Module
        The final network that takes the concatenated embeddings and outputs a score.
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
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Create all pairs of windows (x_i, y_j)
        x_tiled = x.repeat_interleave(batch_size, dim=0)
        # y is tiled to match x_tiled. For each x_i, we need all y_j.
        # The original code used a confusing implicit promotion. This is equivalent and clearer.
        y_tiled = y.repeat(batch_size, 1, 1)

        # Extract features with the CNNs
        features_x = self.cnn_x(x_tiled)
        features_y = self.cnn_y(y_tiled)
        
        # Concatenate the output embedding vectors
        combined_features = torch.cat([features_x, features_y], dim=1)
        
        # Pass through the final decision head
        scores = self.decision_head(combined_features)
        
        return scores.view(batch_size, batch_size)