# neural_mi/models/critics.py

import torch
import torch.nn as nn

class BaseCritic(nn.Module):
    """Base class for critics."""
    def __init__(self):
        super(BaseCritic, self).__init__()

    def forward(self, x, y):
        raise NotImplementedError

class SeparableCritic(BaseCritic):
    """A separable critic that computes scores via a dot product of embeddings."""
    def __init__(self, embedding_net_x, embedding_net_y=None):
        super(SeparableCritic, self).__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y if embedding_net_y is not None else embedding_net_x

    def forward(self, x, y):
        x_embedded = self.embedding_net_x(x)
        y_embedded = self.embedding_net_y(y)
        scores = torch.matmul(x_embedded, y_embedded.t())
        return scores

class ConcatCritic(BaseCritic):
    """A concatenated critic that passes pairs of (x, y) through a single network."""
    def __init__(self, embedding_net):
        super(ConcatCritic, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x, y):
        batch_size = x.size(0)
        
        # Flatten the feature dimensions (channels, window_size)
        x_flat = x.view(batch_size, -1)
        y_flat = y.view(batch_size, -1)
        
        # Create all pairs (x_i, y_j)
        x_tiled = x_flat.repeat_interleave(batch_size, dim=0)
        # Use correct repeat dimensions for the tiled y
        y_tiled = y_flat.repeat(batch_size, 1)

        xy_pairs = torch.cat((x_tiled, y_tiled), dim=1)
        scores = self.embedding_net(xy_pairs)
        
        return scores.view(batch_size, batch_size)