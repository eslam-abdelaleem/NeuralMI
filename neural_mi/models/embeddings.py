# neural_mi/models/embeddings.py

import torch
import torch.nn as nn

class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLP(BaseEmbedding):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, 
                 n_layers: int, activation: str = 'relu'):
        super().__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 
                       'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}
        act_fn = activations[activation]
        layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, embed_dim)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.network(x.view(x.shape[0], -1)))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

class CNN1D(BaseEmbedding):
    """
    A 1D CNN that preserves temporal feature location by dynamically creating
    its final linear layer to handle variable window sizes.
    This version correctly initializes the dynamic layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, activation: str = 'relu'):
        super().__init__()
        
        activation_fn = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}.get(activation, nn.ReLU)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=7, padding='same'),
            activation_fn(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=5, padding='same'),
            activation_fn()
        )
        
        self.final_layers = None
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self._initialize_weights() # Initialize the conv layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_layers(x)
        
        if self.final_layers is None:
            flattened_size = h.shape[1] * h.shape[2]
            self.final_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.embed_dim)
            ).to(x.device)
            # *** FIX: Initialize the new layers ***
            self._initialize_final_layers()

        return self.final_layers(h)

    def _initialize_weights(self):
        for m in self.conv_layers.modules(): # Only init conv layers here
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_final_layers(self):
        if self.final_layers is not None:
            for m in self.final_layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                    
class VarMLP(BaseEmbedding):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, 
                 n_layers: int, activation: str = 'relu'):
        super().__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
                       'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}
        act_fn = activations[activation]
        layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
        self.base_network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, embed_dim)
        self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
        self.kl_loss = 0.0
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base_network(x.view(x.shape[0], -1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        if not self.training: return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)