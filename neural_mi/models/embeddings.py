# neural_mi/models/embeddings.py

import torch
import torch.nn as nn
import math

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
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, activation: str = 'relu', kernel_size: int = 7):
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
        kernel_size : int, optional
            The size of the convolutional kernel. Must be an odd number. Defaults to 7.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number for 'same' padding.")
        
        activation_fn = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}.get(activation, nn.ReLU)

        layers = [
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same'),
            activation_fn()
        ]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding='same'),
                activation_fn()
            ])
        self.conv_layers = nn.Sequential(*layers)
        self.final_layers = None
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self._initialize_weights()

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
            self._initialize_final_layers()
        return self.final_layers(h)

    def _initialize_weights(self):
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _initialize_final_layers(self):
        if self.final_layers is not None:
            for m in self.final_layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

class GRU(BaseEmbedding):
    """A Gated Recurrent Unit (GRU) embedding network for sequential data."""
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, 
                 bidirectional: bool = False, **kwargs):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int
            The number of features in the hidden state of the GRU.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of recurrent layers.
        bidirectional : bool, optional
            If True, becomes a bidirectional GRU. Defaults to False.
        """
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, 
                          batch_first=True, bidirectional=bidirectional)
        
        num_directions = 2 if bidirectional else 1
        self.output_layer = nn.Linear(hidden_dim * num_directions, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRU expects (batch, seq, features), but our data is (batch, features, seq)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, sequence_length, input_dim)
        _, h_n = self.gru(x)
        
        # h_n is of shape (num_layers * num_directions, batch, hidden_size)
        # We take the hidden state of the last layer
        if self.gru.bidirectional:
            # Concatenate the final forward and backward hidden states
            last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            last_hidden = h_n[-1,:,:]
            
        return self.output_layer(last_hidden)

class LSTM(BaseEmbedding):
    """An LSTM (Long Short-Term Memory) embedding network for sequential data."""
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, 
                 bidirectional: bool = False, **kwargs):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int
            The number of features in the hidden state of the LSTM.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of recurrent layers.
        bidirectional : bool, optional
            If True, becomes a bidirectional LSTM. Defaults to False.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        num_directions = 2 if bidirectional else 1
        self.output_layer = nn.Linear(hidden_dim * num_directions, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        
        if self.lstm.bidirectional:
            last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            last_hidden = h_n[-1,:,:]
            
        return self.output_layer(last_hidden)

class TCN(BaseEmbedding):
    """A Temporal Convolutional Network (TCN) for sequential data."""
    # Inner class for a Chomp layer to remove padding
    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

    # Inner class for a TCN block
    class TemporalBlock(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
            super().__init__()
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.chomp1 = TCN.Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.chomp2 = TCN.Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, 
                 kernel_size: int = 3, **kwargs):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int
            The number of channels in the TCN layers.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of temporal blocks (controls depth and receptive field).
        kernel_size : int, optional
            The size of the convolutional kernel. Defaults to 3.
        """
        super().__init__()
        layers = []
        num_channels = [hidden_dim] * n_layers
        for i in range(n_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCN.TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                            padding=(kernel_size-1) * dilation_size))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # Take the last time step's output for the embedding
        return self.output_layer(out[:, :, -1])

        
class Transformer(BaseEmbedding):
    """A Transformer Encoder model for sequential data."""
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        def forward(self, x):
            return x + self.pe[:x.size(0)]

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, 
                 nhead: int = 4, **kwargs):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int
            The main model dimension (`d_model`). Must be divisible by nhead.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of stacked Transformer encoder layers.
        nhead : int, optional
            The number of multi-head attention heads. Defaults to 4.
        """
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead}).")
        
        self.model_dim = hidden_dim
        self.pos_encoder = Transformer.PositionalEncoding(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, embed_dim)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # (batch, seq, features)
        x = self.input_proj(x) * math.sqrt(self.model_dim)
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2) # Apply positional encoding
        output = self.transformer_encoder(x)
        # Use the output of the first token '[CLS]'-style for the embedding
        return self.output_layer(output[:, 0, :])

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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