# neural_mi/models/critics.py

import torch
import torch.nn as nn
from typing import Optional

class BaseCritic(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SeparableCritic(BaseCritic):
    def __init__(self, embedding_net_x: nn.Module, embedding_net_y: Optional[nn.Module] = None):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y or embedding_net_x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_embedded = self.embedding_net_x(x)
        y_embedded = self.embedding_net_y(y)
        return torch.matmul(x_embedded, y_embedded.t())

class BilinearCritic(BaseCritic):
    """
    A critic that uses two embedding networks and a learnable bilinear layer
    for a more powerful comparison than a simple dot product.
    """
    def __init__(self, embedding_net_x: nn.Module, embedding_net_y: nn.Module, embed_dim: int):
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
    def __init__(self, embedding_net: nn.Module):
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
    """
    A critic that uses two CNN towers to extract features and then concatenates
    the feature maps for a final decision.
    """
    def __init__(self, cnn_x: nn.Module, cnn_y: nn.Module, decision_head: nn.Module):
        super().__init__()
        self.cnn_x = cnn_x
        self.cnn_y = cnn_y
        self.decision_head = decision_head
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Create all pairs of windows (x_i, y_j)
        x_tiled = x.repeat_interleave(batch_size, dim=0)
        y_tiled = y.repeat(batch_size, 1, 1, 1).view(batch_size * batch_size, y.shape[1], y.shape[2])

        # Extract features with the CNNs
        features_x = self.cnn_x(x_tiled)
        features_y = self.cnn_y(y_tiled)
        
        # Concatenate along the channel dimension
        combined_features = torch.cat([features_x, features_y], dim=1)
        
        # Pass through the final decision head
        scores = self.decision_head(combined_features)
        
        return scores.view(batch_size, batch_size)
