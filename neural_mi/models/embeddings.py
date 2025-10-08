# neural_mi/models/embeddings.py

import torch
import torch.nn as nn

class BaseEmbedding(nn.Module):
    """Abstract base class for embedding models.

    All embedding models should inherit from this class and implement the `forward` method.
    The role of an embedding model is to transform an input tensor into a
    lower-dimensional vector representation (an embedding).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the embedding for an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            A batch of samples to embed, with shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, embed_dim) representing the embeddings.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError

class _BaseMLP(BaseEmbedding):
    """Internal base class for MLP-style networks to share weight initialization."""
    def _initialize_weights(self):
        """Initializes the weights of the network using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

class MLP(_BaseMLP):
    """A standard Multi-Layer Perceptron (MLP) embedding network.

    This network flattens the input and passes it through a series of fully-connected
    layers to produce an embedding vector.

    Attributes
    ----------
    network : nn.Sequential
        The main sequence of hidden layers.
    output_layer : nn.Linear
        The final linear layer that maps to the embedding dimension.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 n_layers: int, activation: str = 'relu'):
        """
        Parameters
        ----------
        input_dim : int
            The dimensionality of the flattened input.
        hidden_dim : int
            The number of units in each hidden layer.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of hidden layers in the network.
        activation : str, optional
            The name of the activation function to use (e.g., 'relu', 'tanh').
            Defaults to 'relu'.
        """
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
        """Flattens the input and passes it through the MLP."""
        return self.output_layer(self.network(x.view(x.shape[0], -1)))

class CNN1D(BaseEmbedding):
    """A 1D CNN embedding network for sequential data.

    This network uses 1D convolutions to extract features from sequential data.
    It dynamically creates its final fully-connected layers on the first
    forward pass to adapt to variable-length input sequences.

    Attributes
    ----------
    conv_layers : nn.Sequential
        The sequence of convolutional layers.
    final_layers : nn.Sequential or None
        The dynamically created fully-connected layers. Initially None.
    embed_dim : int
        The dimensionality of the output embedding.
    hidden_dim : int
        The dimensionality of the hidden representation after the CNN layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, activation: str = 'relu'):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int
            The number of channels in the first hidden convolutional layer.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of convolutional layers (currently fixed architecture).
        activation : str, optional
            The activation function to use after convolutional layers. Defaults to 'relu'.
        """
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
        """Passes the input through CNN layers and a dynamically created MLP head."""
        h = self.conv_layers(x)

        # Lazily initialize the final linear layers based on the output size of the conv part
        if self.final_layers is None:
            flattened_size = h.shape[1] * h.shape[2]
            self.final_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_size, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.embed_dim)
            ).to(x.device)
            self._initialize_final_layers()

        return self.final_layers(h)

    def _initialize_weights(self):
        """Initializes the weights of the convolutional layers."""
        for m in self.conv_layers.modules(): # Only init conv layers here
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_final_layers(self):
        """Initializes the weights of the dynamically created final layers."""
        if self.final_layers is not None:
            for m in self.final_layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


class VarMLP(_BaseMLP):
    """A Variational MLP that produces a distribution over embeddings.

    This model learns an embedding by parameterizing a Gaussian distribution (mu and logvar).
    During training, it uses the reparameterization trick to sample from this
    distribution and returns the sample and the KL loss. During evaluation,
    it returns the mean (mu) of the distribution and a KL loss of 0.

    Attributes
    ----------
    base_network : nn.Sequential
        The shared MLP base that processes the input.
    fc_mu : nn.Linear
        The layer that produces the mean of the embedding distribution.
    fc_logvar : nn.Linear
        The layer that produces the log variance of the embedding distribution.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 n_layers: int, activation: str = 'relu'):
        """
        Parameters
        ----------
        input_dim : int
            The dimensionality of the flattened input.
        hidden_dim : int
            The number of units in each hidden layer of the base network.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of hidden layers in the base network.
        activation : str, optional
            The activation function to use in the base network. Defaults to 'relu'.
        """
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
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the embedding and KL divergence.

        In training mode, it samples from the learned distribution and returns
        the sample and the KL loss. In evaluation mode, it returns the mean of
        the distribution and a KL loss of 0.
        """
        h = self.base_network(x.view(x.shape[0], -1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        if not self.training:
            return mu, torch.tensor(0.0, device=mu.device)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Calculate KL loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # normalize by batch for stability
        return z, kl_loss

# class VarMLP(_BaseMLP):
#     """A Variational MLP that produces a distribution over embeddings.

#     This model learns an embedding by parameterizing a Gaussian distribution (mu and logvar).
#     During training, it uses the reparameterization trick to sample from this
#     distribution and returns the sample and the KL loss. During evaluation,
#     it returns the mean (mu) of the distribution and a KL loss of 0.
#     """
#     def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
#                  n_layers: int, activation: str = 'relu', logvar_clip: float = 20.0):
#         """
#         Parameters
#         ----------
#         input_dim : int
#             The dimensionality of the flattened input.
#         hidden_dim : int
#             The number of units in each hidden layer of the base network.
#         embed_dim : int
#             The dimensionality of the output embedding.
#         n_layers : int
#             The number of hidden layers in the base network.
#         activation : str, optional
#             The activation function to use in the base network. Defaults to 'relu'.
#         logvar_clip : float, optional
#             The absolute range to clip log variances. Defaults to ±20.0.
#         """
#         super().__init__()
#         self.logvar_clip = logvar_clip

#         activations = {
#             'relu': nn.ReLU,
#             'sigmoid': nn.Sigmoid,
#             'tanh': nn.Tanh,
#             'leaky_relu': nn.LeakyReLU,
#             'silu': nn.SiLU
#         }
#         act_fn = activations[activation]
#         layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
#         for _ in range(n_layers - 1):
#             layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
#         self.base_network = nn.Sequential(*layers)
#         self.fc_mu = nn.Linear(hidden_dim, embed_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, embed_dim)
#         self._initialize_weights()

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes the embedding and KL divergence.

#         In training mode, it samples from the learned distribution and returns
#         the sample and the KL loss. In evaluation mode, it returns the mean of
#         the distribution and a KL loss of 0.
#         """
#         h = self.base_network(x.view(x.shape[0], -1))
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)

#         # --- Clip log-variance to a safe range ---
#         logvar = torch.clamp(logvar, -self.logvar_clip, self.logvar_clip)

#         if not self.training:
#             return mu, torch.tensor(0.0, device=mu.device)

#         # Reparameterization trick
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps * std

#         # KL divergence (summed over dimensions, averaged per batch)
#         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         # kl_loss = kl_loss / x.size(0)  # normalize by batch for stability

#         return z, kl_loss