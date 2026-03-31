# neural_mi/models/decoders.py
"""Decoder models for the Deep Symmetric Information Bottleneck.

Each decoder is the approximate inverse of the corresponding encoder in
``embeddings.py``.  A decoder takes a low-dimensional embedding
``z`` (shape ``(batch, embed_dim)``) and reconstructs the original input
(shape ``(batch, n_channels, window_size)``).

Decoders are used when ``use_decoder=True`` in ``base_params``.  The
reconstruction loss (MSE between input and reconstruction) is added to the
training objective, encouraging the encoder to retain information about the
input while maximising mutual information with the other variable (Deep
Symmetric IB).

Deterministic training objective (``use_variational=False``):
    L = -MI(Z_X; Z_Y) + w_x * MSE(X, X̂) + w_y * MSE(Y, Ŷ)

Variational training objective (``use_variational=True``):
    L = KL_X + KL_Y - β * MI(Z_X; Z_Y)
        + w_x * MSE(X, X̂) + w_y * MSE(Y, Ŷ)

Output activations (controlled by ``output_activation`` parameter):
    - ``'linear'``  : no activation (float / continuous data).
    - ``'sigmoid'`` : sigmoid, output in [0, 1] (spike / binary data).
    - ``'softmax'`` : softmax along the channel dimension (categorical data).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def _get_output_activation(name: str, dim: int = 1):
    """Return a callable that applies the named activation to a 3-D tensor."""
    if name == 'linear':
        return lambda x: x
    elif name == 'sigmoid':
        return torch.sigmoid
    elif name == 'softmax':
        return lambda x: F.softmax(x, dim=dim)   # softmax over channels (dim=1)
    else:
        raise ValueError(
            f"Unknown output_activation='{name}'. "
            f"Supported: 'linear', 'sigmoid', 'softmax'."
        )


class BaseDecoder(nn.Module):
    """Abstract base class for all decoder models.

    Subclasses implement ``forward(z) -> (batch, n_channels, window_size)``.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from embedding ``z``.

        Parameters
        ----------
        z : torch.Tensor
            Embedding tensor of shape ``(batch, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed input of shape ``(batch, n_channels, window_size)``.
        """
        raise NotImplementedError

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class MLPDecoder(BaseDecoder):
    """Mirror MLP decoder for the :class:`~neural_mi.models.embeddings.MLP` encoder.

    Maps ``embed_dim → hidden_dim → ... → n_channels * window_size``,
    then reshapes to ``(batch, n_channels, window_size)``.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_channels: int,
        window_size: int,
        n_layers: int = 2,
        output_activation: str = 'linear',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self._act = _get_output_activation(output_activation)
        output_dim = n_channels * window_size

        layers = [nn.Linear(embed_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, n_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.network(z)                           # (B, C*W)
        out = out.view(z.shape[0], self.n_channels, self.window_size)
        return self._act(out)


class CNN1DDecoder(BaseDecoder):
    """Approximate inverse of the :class:`~neural_mi.models.embeddings.CNN1D` encoder.

    Expands the embedding via a linear layer to a small feature map, then
    uses ``nn.Upsample`` + ``nn.Conv1d`` blocks to reach the target
    ``window_size``.  A final ``Conv1d`` maps to ``n_channels`` outputs.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_channels: int,
        window_size: int,
        n_layers: int = 2,
        kernel_size: int = 7,
        output_activation: str = 'linear',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self._act = _get_output_activation(output_activation)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1  # ensure odd for same padding

        # Start from a small spatial dimension; expand in two stages.
        self._base_len = max(4, window_size // (2 ** max(1, n_layers - 1)))
        self.expand_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * self._base_len),
            nn.ReLU(),
        )

        pad = kernel_size // 2
        conv_blocks = []
        for i in range(n_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim if i < n_layers - 1 else n_channels
            conv_blocks.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad))
            if i < n_layers - 1:
                conv_blocks.append(nn.ReLU())
        self.conv_layers = nn.ModuleList(conv_blocks)
        self._init_weights()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.expand_linear(z)                          # (B, hidden * base_len)
        h = h.view(B, -1, self._base_len)                  # (B, hidden, base_len)
        # Upsample to target window_size
        h = F.interpolate(h, size=self.window_size, mode='linear', align_corners=False)
        for layer in self.conv_layers:
            h = layer(h)
        return self._act(h)


class GRUDecoder(BaseDecoder):
    """Sequence decoder for the :class:`~neural_mi.models.embeddings.GRU` encoder.

    Projects the embedding to a hidden dimension, repeats it as an initial
    sequence, then runs a GRU to produce outputs at each time step.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_channels: int,
        window_size: int,
        n_layers: int = 1,
        output_activation: str = 'linear',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self._act = _get_output_activation(output_activation)

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, n_layers),
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, n_channels)
        self._init_weights()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.input_proj(z)                              # (B, hidden)
        h = h.unsqueeze(1).expand(-1, self.window_size, -1) # (B, W, hidden)
        out, _ = self.gru(h)                               # (B, W, hidden)
        out = self.output_proj(out)                        # (B, W, C)
        out = out.permute(0, 2, 1)                         # (B, C, W)
        return self._act(out)


class LSTMDecoder(BaseDecoder):
    """Sequence decoder for the :class:`~neural_mi.models.embeddings.LSTM` encoder."""
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_channels: int,
        window_size: int,
        n_layers: int = 1,
        output_activation: str = 'linear',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self._act = _get_output_activation(output_activation)

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(1, n_layers),
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, n_channels)
        self._init_weights()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.input_proj(z)                              # (B, hidden)
        h = h.unsqueeze(1).expand(-1, self.window_size, -1) # (B, W, hidden)
        out, _ = self.lstm(h)                              # (B, W, hidden)
        out = self.output_proj(out)                        # (B, W, C)
        out = out.permute(0, 2, 1)                         # (B, C, W)
        return self._act(out)


class TCNDecoder(BaseDecoder):
    """Approximate inverse of the :class:`~neural_mi.models.embeddings.TCN` encoder.

    Projects the embedding to a feature map (same shape as TCN output), then
    uses a sequence of dilated ``Conv1d`` blocks — mirroring the TCN encoder
    structure — to reconstruct the input.  Upsampling is handled by
    ``nn.Upsample`` before the first convolutional block.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_channels: int,
        window_size: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        output_activation: str = 'linear',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self._act = _get_output_activation(output_activation)

        self._base_len = max(4, window_size // (2 ** max(1, n_layers - 1)))
        self.expand_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * self._base_len),
            nn.ReLU(),
        )

        # Dilated conv blocks (reverse dilation order for the decoder)
        blocks = []
        for i in range(n_layers):
            dilation = 2 ** (n_layers - 1 - i)  # decreasing dilation
            padding = (kernel_size - 1) * dilation
            in_ch = hidden_dim
            out_ch = hidden_dim if i < n_layers - 1 else n_channels
            blocks.append(nn.Conv1d(
                in_ch, out_ch, kernel_size,
                dilation=dilation, padding=padding,
            ))
            if i < n_layers - 1:
                blocks.append(nn.ReLU())
        self.conv_blocks = nn.ModuleList(blocks)
        self._init_weights()

    def _chomp(self, x: torch.Tensor, padding: int) -> torch.Tensor:
        """Remove causal padding from the right."""
        if padding > 0:
            return x[:, :, :-padding].contiguous()
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.expand_linear(z)
        h = h.view(B, -1, self._base_len)
        h = F.interpolate(h, size=self.window_size, mode='linear', align_corners=False)

        # Apply dilated conv blocks (remove causal padding so output length = window_size)
        n_layers = sum(1 for m in self.conv_blocks if isinstance(m, nn.Conv1d))
        conv_idx = 0
        for layer in self.conv_blocks:
            if isinstance(layer, nn.Conv1d):
                dil = 2 ** (n_layers - 1 - conv_idx)
                pad = (layer.kernel_size[0] - 1) * dil
                h = layer(h)
                h = self._chomp(h, pad)
                conv_idx += 1
            else:
                h = layer(h)
        return self._act(h)


class TransformerDecoder(BaseDecoder):
    """Projection decoder for the :class:`~neural_mi.models.embeddings.Transformer` encoder.

    Projects the embedding to the model dimension, then uses a standard
    ``nn.TransformerDecoder`` with learned positional queries to regenerate a
    sequence of length ``window_size``.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_channels: int,
        window_size: int,
        n_layers: int = 2,
        nhead: int = 4,
        output_activation: str = 'linear',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self._act = _get_output_activation(output_activation)

        # Ensure hidden_dim is divisible by nhead
        if hidden_dim % nhead != 0:
            hidden_dim = (hidden_dim // nhead) * nhead
            if hidden_dim == 0:
                hidden_dim = nhead

        self.memory_proj = nn.Linear(embed_dim, hidden_dim)
        # Learned position queries (one per time step)
        self.pos_queries = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(1, n_layers))
        self.output_proj = nn.Linear(hidden_dim, n_channels)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        memory = self.memory_proj(z).unsqueeze(1)              # (B, 1, hidden)
        queries = self.pos_queries.expand(B, -1, -1)            # (B, W, hidden)
        out = self.transformer_decoder(queries, memory)         # (B, W, hidden)
        out = self.output_proj(out)                             # (B, W, C)
        out = out.permute(0, 2, 1)                              # (B, C, W)
        return self._act(out)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_decoder(
    embedding_model: str,
    embed_dim: int,
    hidden_dim: int,
    n_channels: int,
    window_size: int,
    n_layers: int = 2,
    output_activation: str = 'linear',
    **kwargs,
) -> BaseDecoder:
    """Build a decoder matching the given embedding model.

    Parameters
    ----------
    embedding_model : str
        Name of the encoder (e.g. ``'mlp'``, ``'cnn1d'``, ``'gru'``, …).
    embed_dim : int
        Embedding dimensionality (output size of the encoder).
    hidden_dim : int
        Hidden dimension to use in the decoder.
    n_channels : int
        Number of output channels (must match the encoder's input channels).
    window_size : int
        Sequence length of the reconstructed output.
    n_layers : int, optional
        Number of decoder layers. Defaults to 2.
    output_activation : str, optional
        Final activation: ``'linear'``, ``'sigmoid'``, or ``'softmax'``.
    **kwargs
        Forwarded to the decoder constructor (e.g. ``kernel_size``, ``nhead``).

    Returns
    -------
    BaseDecoder
        The constructed decoder module.
    """
    common = dict(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_channels=n_channels,
        window_size=window_size,
        n_layers=n_layers,
        output_activation=output_activation,
    )
    name = embedding_model.lower()
    if name in ('mlp', 'var_mlp'):
        return MLPDecoder(**common)
    elif name == 'cnn1d':
        return CNN1DDecoder(**common, kernel_size=kwargs.get('kernel_size', 7))
    elif name == 'gru':
        return GRUDecoder(**common)
    elif name == 'lstm':
        return LSTMDecoder(**common)
    elif name == 'tcn':
        return TCNDecoder(**common, kernel_size=kwargs.get('kernel_size', 3))
    elif name == 'transformer':
        return TransformerDecoder(**common, nhead=kwargs.get('nhead', 4))
    else:
        # Custom or unknown encoder — use a simple MLP decoder as fallback.
        return MLPDecoder(**common)
