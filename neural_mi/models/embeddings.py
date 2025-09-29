# neural_mi/models/embeddings.py

import torch
import torch.nn as nn

class BaseEmbedding(nn.Module):
    """
    Base class for embedding models to ensure a consistent interface.
    Custom embedding models should inherit from this class.
    """
    def __init__(self):
        super(BaseEmbedding, self).__init__()

    def forward(self, x):
        raise NotImplementedError

class MLP(BaseEmbedding):
    """
    A simple Multi-Layer Perceptron (MLP) embedding model.
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers, activation='relu'):
        super(MLP, self).__init__()
        
        activation_fn = {
            'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU,
        }[activation]

        layers = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

        self._initialize_weights()

    def forward(self, x):
        h = self.network(x.view(x.shape[0], -1)) # Flatten input
        return self.output_layer(h)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class VarMLP(BaseEmbedding):
    """
    A variational MLP that outputs samples from a learned distribution.
    Used for the DVSIB objective.
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers, activation='relu'):
        super(VarMLP, self).__init__()
        
        activation_fn = {
            'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU,
        }[activation]
        
        layers = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
            
        self.base_network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        
        self.kl_loss = 0.0
        self._initialize_weights()

    def forward(self, x):
        h = self.base_network(x.view(x.shape[0], -1)) # Flatten input
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        if not self.training:
            return mu

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Calculate KL divergence
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)