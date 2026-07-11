# neural_mi/models/embeddings.py

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch.nn.utils import spectral_norm as _spectral_norm

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
    def __init__(self, input_dim: int, hidden_dim, embed_dim: int,
                 n_layers: int, activation: str = 'relu',
                 use_spectral_norm: bool = True,
                 dropout: float = 0.0,
                 norm_layer: Optional[str] = None):
        """
        Parameters
        ----------
        input_dim : int
            The dimensionality of the flattened input.
        hidden_dim : int or list of int
            The number of units in each hidden layer.  Two forms are accepted:

            - **int**: all hidden layers have the same width; ``n_layers``
              controls the total number of layers.
            - **list of int**: each element specifies the width of one hidden
              layer (e.g. ``[256, 1024, 256]`` for a bottleneck-then-expand
              architecture).  ``n_layers`` is ignored when a list is given;
              the list length determines the depth.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of hidden layers in the network.  Ignored when
            ``hidden_dim`` is a list.
        activation : str, optional
            The name of the activation function to use (e.g., 'relu', 'tanh').
            Defaults to 'relu'.
        use_spectral_norm : bool, optional
            If True, applies spectral normalisation to the hidden ``nn.Linear``
            layers.  The output layer is left unnormalised to preserve the full
            expressive range of the embedding.
            Defaults to True.
        dropout : float, optional
            Dropout probability applied after each hidden activation. A value of
            0.0 (default) disables dropout. Values in (0, 1) are useful for
            regularisation, especially with small datasets (e.g. 0.1–0.3).
        norm_layer : {None, 'batch', 'layer'}, optional
            Normalisation to apply inside each hidden block, inserted between the
            linear transformation and the activation:

            - ``None`` (default) — no normalisation.
            - ``'layer'`` — ``nn.LayerNorm``. Stable at any batch size; recommended
              for small datasets where batch statistics are unreliable.
            - ``'batch'`` — ``nn.BatchNorm1d``. Effective on large batches but can
              be unstable when batch_size is small (< ~32).
        """
        super().__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
                       'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}
        act_fn = activations[activation]
        _wrap = _spectral_norm if use_spectral_norm else (lambda x: x)

        def _make_hidden_block(in_dim: int, out_dim: int) -> list:
            block = [_wrap(nn.Linear(in_dim, out_dim))]
            if norm_layer == 'batch':
                block.append(nn.BatchNorm1d(out_dim))
            elif norm_layer == 'layer':
                block.append(nn.LayerNorm(out_dim))
            block.append(act_fn())
            if dropout > 0.0:
                block.append(nn.Dropout(p=dropout))
            return block

        dims = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * max(1, n_layers)
        layers = _make_hidden_block(input_dim, dims[0])
        for i in range(1, len(dims)):
            layers.extend(_make_hidden_block(dims[i - 1], dims[i]))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(dims[-1], embed_dim)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens the input and passes it through the MLP."""
        # Use reshape (not view) so that non-contiguous inputs (e.g. from
        # permute() in transfer-entropy history arrays) are handled correctly.
        return self.output_layer(self.network(x.reshape(x.shape[0], -1)))

class CNN1D(BaseEmbedding):
    """A 1D CNN embedding network for sequential data.

    This network uses 1D convolutions followed by global average pooling to
    produce a fixed-size embedding, regardless of input sequence length. Using
    ``AdaptiveAvgPool1d`` instead of lazy linear layers makes the model fully
    picklable at construction time (important for multiprocessing).

    When ``use_depthwise=True`` the first convolutional layer is replaced by a
    **depthwise-separable** block: a depthwise convolution (one filter per input
    channel, ``groups=input_dim``) followed immediately by a pointwise 1×1
    convolution that mixes channels. This enforces the per-channel-first
    ordering — each electrode/neuron's temporal dynamics are filtered
    independently before any cross-channel mixing occurs. Subsequent layers are
    standard convolutions operating on the ``hidden_dim`` feature space.

    Attributes
    ----------
    conv_layers : nn.Sequential
        The sequence of convolutional layers.
    final_layers : nn.Sequential
        The pooling and fully-connected layers that map to the embedding.
    embed_dim : int
        The dimensionality of the output embedding.
    hidden_dim : int
        The number of feature channels after the CNN layers.
    """
    def __init__(self, input_dim: int, hidden_dim, embed_dim: int, n_layers: int,
                 activation: str = 'relu', kernel_size: int = 7, use_depthwise: bool = False):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int or list of int
            The number of channels in the hidden convolutional layers.  When a
            list is given (e.g. ``[64, 128, 64]``), each element sets the output
            channel count of that layer and ``n_layers`` is ignored.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of convolutional layers.  Ignored when ``hidden_dim`` is
            a list.
        activation : str, optional
            The activation function to use after convolutional layers. Defaults to 'relu'.
        kernel_size : int, optional
            The size of the convolutional kernel. Must be an odd number. Defaults to 7.
        use_depthwise : bool, optional
            If True, the first convolutional layer uses a depthwise-separable
            decomposition (depthwise + pointwise 1×1) to enforce per-channel
            temporal filtering before cross-channel mixing. Subsequent layers
            remain standard convolutions. Defaults to False.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number for 'same' padding.")

        activation_fn = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}.get(activation, nn.ReLU)
        ch = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * n_layers

        # First layer: depthwise-separable or standard
        if use_depthwise:
            first_block = [
                # Depthwise: filter each channel independently in time
                nn.Conv1d(input_dim, input_dim, kernel_size, padding='same', groups=input_dim),
                # Pointwise: mix channels (no temporal extent)
                nn.Conv1d(input_dim, ch[0], kernel_size=1),
                activation_fn(),
            ]
        else:
            first_block = [
                nn.Conv1d(in_channels=input_dim, out_channels=ch[0], kernel_size=kernel_size, padding='same'),
                activation_fn(),
            ]

        layers = list(first_block)
        for i in range(1, len(ch)):
            layers.extend([
                nn.Conv1d(in_channels=ch[i - 1], out_channels=ch[i], kernel_size=kernel_size, padding='same'),
                activation_fn()
            ])
        self.conv_layers = nn.Sequential(*layers)
        # AdaptiveAvgPool1d(1) collapses any sequence length → (batch, ch[-1], 1),
        # giving a fixed-size representation that is fully picklable at construction time.
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch[-1], ch[-1]),
            nn.ReLU(),
            nn.Linear(ch[-1], embed_dim)
        )
        self.embed_dim = embed_dim
        self.hidden_dim = ch[-1]
        self._initialize_weights()
        self._initialize_final_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layers(self.conv_layers(x))

    def _initialize_weights(self):
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _initialize_final_layers(self):
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

    def __init__(self, input_dim: int, hidden_dim, embed_dim: int, n_layers: int,
                 kernel_size: int = 3, **kwargs):
        """
        Parameters
        ----------
        input_dim : int
            The number of input channels.
        hidden_dim : int or list of int
            The number of channels in each TCN temporal block.  When a list is
            given (e.g. ``[32, 64, 32]``), each element sets the channel count
            of that block and ``n_layers`` is ignored.
        embed_dim : int
            The dimensionality of the output embedding.
        n_layers : int
            The number of temporal blocks (controls depth and receptive field).
            Ignored when ``hidden_dim`` is a list.
        kernel_size : int, optional
            The size of the convolutional kernel. Defaults to 3.
        """
        super().__init__()
        layers = []
        num_channels = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * n_layers
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCN.TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                            padding=(kernel_size-1) * dilation_size))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], embed_dim)

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
            seq_len = x.size(0)
            if seq_len > self.pe.size(0):
                raise ValueError(
                    f"Input sequence length ({seq_len}) exceeds the Transformer's "
                    f"max_len ({self.pe.size(0)}). Either reduce window_size or "
                    f"increase max_len in PositionalEncoding."
                )
            return x + self.pe[:seq_len]

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


class CNN2D(BaseEmbedding):
    """A 2D CNN embedding network for image-like data with shape (N, C, H, W).

    Uses Conv2d blocks followed by ``AdaptiveAvgPool2d(1)`` to collapse any
    spatial size to a fixed-length vector, then two linear layers to produce
    the embedding.  The adaptive pooling means the network is picklable at
    construction time and handles variable spatial dimensions without any
    ``input_shape`` argument — only the number of input channels is needed.

    Attributes
    ----------
    conv_layers : nn.Sequential
        Sequence of Conv2d + activation blocks.
    final_layers : nn.Sequential
        Adaptive pooling, flatten, and two linear projection layers.
    embed_dim : int
    hidden_dim : int
    """
    def __init__(self, input_dim: int, hidden_dim, embed_dim: int, n_layers: int,
                 activation: str = 'relu', kernel_size: int = 3, **kwargs):
        """
        Parameters
        ----------
        input_dim : int
            Number of input channels (C in (N, C, H, W)).
        hidden_dim : int or list of int
            Number of feature maps in each Conv2d layer.  When a list is given
            (e.g. ``[32, 64, 32]``), each element sets the output channel count
            of that layer and ``n_layers`` is ignored.
        embed_dim : int
            Dimensionality of the output embedding.
        n_layers : int
            Number of Conv2d layers.  Ignored when ``hidden_dim`` is a list.
        activation : str, optional
            Activation function: ``'relu'`` (default) or ``'leaky_relu'``.
        kernel_size : int, optional
            Convolutional kernel size; must be odd for symmetric same-padding.
            Defaults to 3.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number for 'same' padding.")
        padding = kernel_size // 2
        activation_fn = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}.get(activation, nn.ReLU)
        ch = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * n_layers

        layers = [
            nn.Conv2d(input_dim, ch[0], kernel_size=kernel_size, padding=padding),
            activation_fn(),
        ]
        for i in range(1, len(ch)):
            layers.extend([
                nn.Conv2d(ch[i - 1], ch[i], kernel_size=kernel_size, padding=padding),
                activation_fn(),
            ])
        self.conv_layers = nn.Sequential(*layers)
        # AdaptiveAvgPool2d(1) collapses any (H, W) → (1, 1), giving a fixed-size
        # representation that is fully picklable at construction time.
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[-1], ch[-1]),
            nn.ReLU(),
            nn.Linear(ch[-1], embed_dim),
        )
        self.embed_dim = embed_dim
        self.hidden_dim = ch[-1]
        self._initialize_weights()
        self._initialize_final_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes (N, C, H, W) input through conv blocks then the projection head."""
        return self.final_layers(self.conv_layers(x))

    def _initialize_weights(self):
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_final_layers(self):
        for m in self.final_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SincEmbedding(BaseEmbedding):
    """Frequency-aware embedding for continuous neural data (EEG, LFP).

    Input shape: ``(N, n_channels, n_timepoints)`` — raw voltage or LFP
    traces produced by ``ContinuousWindowDataset``.

    **Inductive bias:** Neural activity is organized in frequency bands
    (delta, theta, alpha, beta, gamma). The first layer uses ``n_sinc_filters``
    learnable sinc bandpass filters per input channel, parameterized by a
    lower cutoff ``f_low`` and an upper cutoff ``f_high`` (in Hz). The filters
    are initialized to cover the classical neural frequency bands and are
    constrained so that ``f_low < f_high`` and both remain positive.

    This enforces per-channel temporal filtering *before* cross-channel mixing,
    following the principle that each electrode's local dynamics should be
    characterized first.

    Architecture:
    1. **Sinc filter layer**: ``n_channels → n_channels * n_sinc_filters`` feature
       maps via learnable depthwise bandpass filters.  Applied per-channel
       (``groups = n_channels``).
    2. **Convolutional body**: standard Conv1D layers operating on the filtered
       feature maps.
    3. **Global average pool + MLP head**: collapses time and projects to embed_dim.

    If ``feature_fusion='concat'``, the raw input (unfiltered, mean-pooled over
    time) is concatenated with the body output before the projection head.

    Parameters
    ----------
    input_dim : int
        Number of input channels.
    hidden_dim : int
        Number of feature maps in the convolutional body.
    embed_dim : int
        Dimensionality of the output embedding.
    n_layers : int
        Number of convolutional layers after the sinc filter stage.
    n_sinc_filters : int, optional
        Number of bandpass filters per input channel. Defaults to 8.
    sample_rate : float, optional
        Sampling rate in Hz.  Used to convert cutoff frequencies (Hz) to
        kernel coefficients (samples).  If ``None``, defaults to 1000.0 Hz
        (a typical EEG rate); a warning is emitted.
    feature_fusion : {'features', 'concat'}, optional
        Whether to concatenate mean-pooled raw input with sinc features.
    kernel_size : int, optional
        Length of the sinc FIR kernel in samples (must be odd). Defaults to 51.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int,
                 n_sinc_filters: int = 8, sample_rate: Optional[float] = None,
                 feature_fusion: str = 'features', kernel_size: int = 51, **kwargs):
        super().__init__()
        import warnings as _warnings
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.n_channels = input_dim
        self.n_sinc_filters = n_sinc_filters
        self.sample_rate = float(sample_rate) if sample_rate is not None else 1000.0
        if sample_rate is None:
            _warnings.warn(
                "SincEmbedding: sample_rate not provided; defaulting to 1000 Hz. "
                "Pass sample_rate in processor_params for accurate band initialization.",
                UserWarning, stacklevel=3,
            )
        self.feature_fusion = feature_fusion
        self.sinc_kernel_size = kernel_size

        n_filters_total = input_dim * n_sinc_filters
        # Initialise at classical EEG bands, cycling across filters
        _bands_hz = [
            (1.0, 4.0),     # delta
            (4.0, 8.0),     # theta
            (8.0, 13.0),    # alpha
            (13.0, 30.0),   # beta
            (30.0, 70.0),   # low-gamma
            (70.0, 150.0),  # high-gamma
            (0.5, 2.0),     # sub-delta
            (150.0, 200.0), # very high
        ]
        f_lows = [_bands_hz[i % len(_bands_hz)][0] for i in range(n_filters_total)]
        f_highs = [_bands_hz[i % len(_bands_hz)][1] for i in range(n_filters_total)]
        self.log_f_low = nn.Parameter(torch.log(torch.tensor(f_lows, dtype=torch.float32)))
        self.log_f_high = nn.Parameter(torch.log(torch.tensor(f_highs, dtype=torch.float32)))

        # Convolutional body operates on sinc-filtered features
        sinc_out_channels = input_dim * n_sinc_filters
        body_layers = []
        in_ch = sinc_out_channels
        for _ in range(max(1, n_layers)):
            body_layers.extend([
                nn.Conv1d(in_ch, hidden_dim, kernel_size=7, padding='same'),
                nn.ReLU(),
            ])
            in_ch = hidden_dim
        self.body = nn.Sequential(*body_layers)

        pool_in = hidden_dim + (input_dim if feature_fusion == 'concat' else 0)
        self.head = nn.Sequential(
            nn.Linear(pool_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._initialize_weights()

    def _sinc_kernel(self) -> torch.Tensor:
        """Build FIR sinc bandpass kernels from the current f_low, f_high parameters.

        Returns
        -------
        torch.Tensor
            Shape ``(n_channels * n_sinc_filters, 1, kernel_size)``, ready for a
            depthwise grouped convolution.
        """
        nyquist = self.sample_rate / 2.0
        f_low = torch.exp(self.log_f_low).clamp(0.5, nyquist - 1.0)
        # Ensure f_high > f_low by at least 0.5 Hz using elementwise maximum
        f_high_raw = torch.exp(self.log_f_high)
        f_high = torch.maximum(f_low + 0.5, f_high_raw).clamp(max=nyquist)
        fl = f_low / self.sample_rate
        fh = f_high / self.sample_rate

        half = self.sinc_kernel_size // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32, device=self.log_f_low.device)
        # sinc bandpass: 2·fh·sinc(2π·fh·t) − 2·fl·sinc(2π·fl·t)
        kernels = (2.0 * fh.unsqueeze(-1) * torch.sinc(2.0 * fh.unsqueeze(-1) * t.unsqueeze(0))
                   - 2.0 * fl.unsqueeze(-1) * torch.sinc(2.0 * fl.unsqueeze(-1) * t.unsqueeze(0)))
        hamming = torch.hamming_window(self.sinc_kernel_size, periodic=False,
                                       device=self.log_f_low.device)
        kernels = kernels * hamming.unsqueeze(0)
        return kernels.unsqueeze(1)  # (C*F, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinc filters, convolutional body, pool, and project.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, n_channels, T)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, embed_dim)``.
        """
        N, C, T = x.shape
        # Depthwise sinc filtering: each channel × each filter
        x_rep = x.repeat_interleave(self.n_sinc_filters, dim=1)  # (N, C*F, T)
        kernel = self._sinc_kernel()  # (C*F, 1, K)
        pad = self.sinc_kernel_size // 2
        sinc_out = nn.functional.conv1d(x_rep, kernel, padding=pad,
                                        groups=C * self.n_sinc_filters)  # (N, C*F, T)
        features = self.body(sinc_out)   # (N, hidden_dim, T)
        pooled = features.mean(dim=-1)   # (N, hidden_dim)
        if self.feature_fusion == 'concat':
            pooled = torch.cat([pooled, x.mean(dim=-1)], dim=-1)
        return self.head(pooled)

    def get_physics_params(self) -> dict:
        """Return current learned sinc filter cutoff frequencies in Hz.

        Returns
        -------
        dict
            Keys ``'f_low_hz'`` and ``'f_high_hz'``, each a list of floats
            with length ``n_channels * n_sinc_filters``.
        """
        with torch.no_grad():
            nyquist = self.sample_rate / 2.0
            f_low = torch.exp(self.log_f_low).clamp(0.5, nyquist - 1.0)
            f_high_raw = torch.exp(self.log_f_high)
            f_high = torch.maximum(f_low + 0.5, f_high_raw).clamp(max=nyquist)
        return {
            'f_low_hz': f_low.cpu().tolist(),
            'f_high_hz': f_high.cpu().tolist(),
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CalciumEmbedding(BaseEmbedding):
    """Embedding for calcium imaging data that accounts for indicator dynamics.

    Input shape: ``(N, n_channels, n_timepoints)`` — raw fluorescence (ΔF/F)
    traces produced by ``ContinuousWindowDataset``.

    **Inductive bias:** The observed fluorescence is the convolution of the
    true underlying spike rate with the calcium indicator's impulse response::

        f(t) ≈ (spike_rate * h)(t),   h(t) = exp(-t/τ_decay) - exp(-t/τ_rise)

    A deconvolution step in the first layer recovers an estimate of the
    underlying neural activity. The deconvolution kernel is the time-reversed,
    normalized indicator impulse response (matched-filter approximation).
    ``τ_rise`` and ``τ_decay`` can be fixed to known indicator values or learned.

    The deconvolution is applied per-channel (depthwise) to preserve the
    single-neuron / single-ROI structure before any cross-channel mixing.

    Architecture:
    1. **Deconvolution layer**: per-channel FIR deconvolution, fixed or learnable.
    2. **Convolutional body**: standard Conv1D body operating on deconvolved signals.
    3. **Global average pool + projection head** → embed_dim.

    If ``feature_fusion='concat'``, the mean-pooled raw fluorescence is
    concatenated with the deconvolved features before the projection head.

    Parameters
    ----------
    input_dim : int
        Number of input channels (ROIs / neurons).
    hidden_dim : int
        Number of feature maps in the convolutional body.
    embed_dim : int
        Dimensionality of the output embedding.
    n_layers : int
        Number of convolutional layers in the body.
    tau_rise : float, optional
        Rise time constant in seconds. Defaults to 0.05 (50 ms, ~GCaMP6f).
    tau_decay : float, optional
        Decay time constant in seconds. Defaults to 0.4 (400 ms, ~GCaMP6f).
    learn_calcium_kernel : bool, optional
        If True, ``τ_rise`` and ``τ_decay`` are made trainable. Defaults to False.
    sample_rate : float, optional
        Sampling rate in Hz. Used to convert time constants to samples.
        Defaults to 30.0 Hz if not provided.
    feature_fusion : {'features', 'concat'}, optional
        Whether to concatenate mean-pooled raw fluorescence with deconvolved output.
    kernel_size : int, optional
        Length of the deconvolution FIR kernel in samples (must be odd).
        Defaults to 31.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int,
                 tau_rise: float = 0.05, tau_decay: float = 0.4,
                 learn_calcium_kernel: bool = False,
                 sample_rate: Optional[float] = None,
                 feature_fusion: str = 'features', kernel_size: int = 31, **kwargs):
        super().__init__()
        import warnings as _warnings
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.n_channels = input_dim
        self.feature_fusion = feature_fusion
        self.deconv_kernel_size = kernel_size
        self.sample_rate = float(sample_rate) if sample_rate is not None else 30.0
        if sample_rate is None:
            _warnings.warn(
                "CalciumEmbedding: sample_rate not provided; defaulting to 30 Hz. "
                "Pass sample_rate in processor_params for correct kernel duration.",
                UserWarning, stacklevel=3,
            )

        log_tau_rise = math.log(max(tau_rise, 1e-4))
        log_tau_decay = math.log(max(tau_decay, 1e-4))
        if learn_calcium_kernel:
            self.log_tau_rise = nn.Parameter(torch.tensor(log_tau_rise))
            self.log_tau_decay = nn.Parameter(torch.tensor(log_tau_decay))
        else:
            self.register_buffer('log_tau_rise', torch.tensor(log_tau_rise))
            self.register_buffer('log_tau_decay', torch.tensor(log_tau_decay))
        self.learn_calcium_kernel = learn_calcium_kernel

        body_layers = []
        in_ch = input_dim
        for _ in range(max(1, n_layers)):
            body_layers.extend([
                nn.Conv1d(in_ch, hidden_dim, kernel_size=7, padding='same'),
                nn.ReLU(),
            ])
            in_ch = hidden_dim
        self.body = nn.Sequential(*body_layers)

        pool_in = hidden_dim + (input_dim if feature_fusion == 'concat' else 0)
        self.head = nn.Sequential(
            nn.Linear(pool_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._initialize_weights()

    def _deconv_kernel(self) -> torch.Tensor:
        """Build per-channel deconvolution kernel from current τ_rise, τ_decay.

        The indicator impulse response is::

            h(t) = exp(-t/τ_decay) - exp(-t/τ_rise)

        The deconvolution kernel is the time-reversed, normalized version of h.

        Returns
        -------
        torch.Tensor
            Shape ``(n_channels, 1, kernel_size)`` for a depthwise grouped conv.
        """
        tau_rise = torch.exp(self.log_tau_rise).clamp(1e-4, 10.0)
        tau_decay_raw = torch.exp(self.log_tau_decay).clamp(1e-3, 100.0)
        # Ensure tau_decay > tau_rise using elementwise maximum
        tau_decay = torch.maximum(tau_rise + 1e-3, tau_decay_raw)
        dt = 1.0 / self.sample_rate
        t = torch.arange(0, self.deconv_kernel_size, dtype=torch.float32,
                         device=tau_rise.device) * dt
        h = torch.exp(-t / tau_decay) - torch.exp(-t / tau_rise)
        h = h / (h.abs().max() + 1e-8)
        # Matched-filter deconvolution: time-reversed normalized impulse response
        h_inv = torch.flip(h, dims=[0])
        h_inv = h_inv / (h_inv.pow(2).sum().sqrt() + 1e-8)
        kernel = h_inv.unsqueeze(0).unsqueeze(0).expand(self.n_channels, 1, -1)
        return kernel.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deconvolve fluorescence then embed via body + head.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, n_channels, T)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, embed_dim)``.
        """
        N, C, T = x.shape
        kernel = self._deconv_kernel()   # (C, 1, K)
        pad = self.deconv_kernel_size // 2
        deconvolved = nn.functional.conv1d(x, kernel, padding=pad, groups=C)  # (N, C, T)
        features = self.body(deconvolved)  # (N, hidden_dim, T)
        pooled = features.mean(dim=-1)     # (N, hidden_dim)
        if self.feature_fusion == 'concat':
            pooled = torch.cat([pooled, x.mean(dim=-1)], dim=-1)
        return self.head(pooled)

    def get_physics_params(self) -> dict:
        """Return current learned calcium kernel time constants in seconds.

        Returns an empty dict when ``learn_calcium_kernel=False``.

        Returns
        -------
        dict
            Keys ``'tau_rise_s'`` and ``'tau_decay_s'`` (floats) when the
            kernel is learnable; empty dict otherwise.
        """
        if not self.learn_calcium_kernel:
            return {}
        with torch.no_grad():
            tau_rise = float(torch.exp(self.log_tau_rise).clamp(1e-4, 10.0).item())
            tau_decay_raw = float(torch.exp(self.log_tau_decay).clamp(1e-3, 100.0).item())
            tau_decay = max(tau_decay_raw, tau_rise + 1e-3)
        return {'tau_rise_s': tau_rise, 'tau_decay_s': tau_decay}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SpikePhysicsEmbedding(BaseEmbedding):
    """Embedding for raw spike-timestamp data with physics-informed feature extraction.

    Input shape: ``(N, n_neurons, max_spikes)`` — spike times relative to the
    window start, with invalid slots filled by ``no_spike_value`` (default -1.0).
    This is the tensor produced by ``SpikeWindowDataset``.

    **Stage 1 — per-neuron feature extraction (no learned parameters):**
    For each neuron, the following four scalar features are computed from the
    valid (non-padding) spike times:

    - *Firing rate*: ``n_valid_spikes / window_size``
    - *Mean spike time*: mean of valid spike times, normalized by ``window_size``
    - *Mean ISI*: mean inter-spike interval across consecutive valid spikes
    - *ISI variance*: variance of the inter-spike intervals

    All four are zero when a neuron fires 0 (or 1) spikes in the window.
    This produces a ``(N, n_neurons, 4)`` tensor with no trainable parameters.

    **Stage 2 — optional fusion (``feature_fusion='concat'``):**
    If enabled, the raw spike times (padding replaced with 0, divided by
    ``window_size``) are concatenated with the physics features, giving a
    ``(N, n_neurons, 4 + max_spikes)`` tensor. This preserves fine timing
    information that the summary statistics may lose.

    **Stage 3 — learned mixer:**
    The feature tensor is flattened and passed through a small MLP
    (same ``hidden_dim``, ``embed_dim``, ``n_layers`` as the standard MLP)
    to produce the final embedding of shape ``(N, embed_dim)``.

    Parameters
    ----------
    input_dim : int
        Number of neurons / channels (``n_channels_x``).
    hidden_dim : int
        Hidden size of the mixer MLP.
    embed_dim : int
        Embedding dimensionality.
    n_layers : int
        Number of hidden layers in the mixer MLP.
    max_spikes : int
        Size of the spike-slot dimension (last dim of input tensor).
    no_spike_value : float, optional
        Sentinel used to mark empty spike slots. Defaults to -1.0.
    window_size : float, optional
        Duration of each window in seconds, used to compute firing rate
        and to normalise spike times. Defaults to 1.0.
    feature_fusion : {'features', 'concat'}, optional
        - ``'features'``: use only the four physics features (default).
        - ``'concat'``: concatenate features with (normalised) raw spike times.
    """

    _K_FEATURES = 4  # firing_rate, mean_spike_time, isi_mean, isi_var

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int,
                 max_spikes: int = 50, no_spike_value: float = -1.0,
                 window_size: float = 1.0, feature_fusion: str = 'features', **kwargs):
        super().__init__()
        self.n_neurons = input_dim
        self.max_spikes = max_spikes
        self.no_spike_value = no_spike_value
        self.window_size = max(float(window_size), 1e-6)
        self.feature_fusion = feature_fusion

        if feature_fusion == 'concat':
            mlp_input_dim = input_dim * (self._K_FEATURES + max_spikes)
        else:
            mlp_input_dim = input_dim * self._K_FEATURES

        self.mixer = MLP(mlp_input_dim, hidden_dim, embed_dim, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract physics features and embed via the mixer MLP.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, n_neurons, max_spikes)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, embed_dim)``.
        """
        N = x.shape[0]
        valid_mask = (x != self.no_spike_value)  # (N, n_neurons, max_spikes)

        # --- Firing rate ---
        n_spikes = valid_mask.float().sum(dim=-1)           # (N, n_neurons)
        firing_rate = n_spikes / self.window_size

        # --- Sort spikes in time; push padding slots to the end ---
        times = x.float()
        # Replace padding with +inf so it sorts after all real spikes
        times_for_sort = torch.where(valid_mask, times, torch.full_like(times, float('inf')))
        sorted_times, sort_idx = torch.sort(times_for_sort, dim=-1)
        sorted_mask = valid_mask.gather(-1, sort_idx)       # (N, n_neurons, max_spikes)

        # --- Mean spike time (normalized to [0, 1]) ---
        valid_times = torch.where(sorted_mask, sorted_times, torch.zeros_like(sorted_times))
        n_valid_safe = n_spikes.unsqueeze(-1).clamp(min=1.0)
        mean_spike_time = valid_times.sum(dim=-1) / n_valid_safe.squeeze(-1) / self.window_size
        mean_spike_time = mean_spike_time * (n_spikes > 0).float()

        # --- ISIs ---
        if self.max_spikes > 1:
            # ISI[k] = t[k+1] - t[k] for consecutive valid spike pairs
            isi_raw = sorted_times[:, :, 1:] - sorted_times[:, :, :-1]  # (N, n_neurons, max_spikes-1)
            isi_pair_mask = sorted_mask[:, :, 1:] & sorted_mask[:, :, :-1]
            # Zero out invalid pairs (avoids inf - t = inf contaminating sums)
            isi = torch.where(isi_pair_mask, isi_raw, torch.zeros_like(isi_raw))
            n_isi = isi_pair_mask.float().sum(dim=-1).clamp(min=1.0)   # (N, n_neurons)
            isi_mean_val = isi.sum(dim=-1) / n_isi
            isi_mean_val = isi_mean_val * isi_pair_mask.any(dim=-1).float()
            isi_centered = isi - isi_mean_val.unsqueeze(-1)
            isi_centered = torch.where(isi_pair_mask, isi_centered, torch.zeros_like(isi_centered))
            isi_var_val = (isi_centered ** 2).sum(dim=-1) / n_isi
            isi_var_val = isi_var_val * isi_pair_mask.any(dim=-1).float()
        else:
            isi_mean_val = torch.zeros_like(firing_rate)
            isi_var_val = torch.zeros_like(firing_rate)

        # --- Stack physics features: (N, n_neurons, 4) ---
        features = torch.stack([firing_rate, mean_spike_time, isi_mean_val, isi_var_val], dim=-1)

        if self.feature_fusion == 'concat':
            # Normalised raw spike times (padding → 0)
            raw = torch.where(valid_mask, times / self.window_size, torch.zeros_like(times))
            combined = torch.cat([features, raw], dim=-1)  # (N, n_neurons, 4+max_spikes)
            flat = combined.reshape(N, -1)
        else:
            flat = features.reshape(N, -1)                 # (N, n_neurons * 4)

        return self.mixer(flat)


class PretrainedBackboneEmbedding(BaseEmbedding):
    """Image embedding using a frozen pretrained torchvision backbone + trainable MLP head.

    Input shape: ``(N, C, H, W)`` — image batch, identical to the input expected
    by ``CNN2D`` and ``cnn2d`` mode.

    **Inductive bias:** A pretrained CNN backbone (e.g. ResNet, VGG, EfficientNet)
    from ``torchvision.models`` encodes powerful visual representations learned from
    large image datasets.  The backbone is used as a fixed feature extractor (all
    weights frozen), and a small trainable MLP head maps the backbone's feature
    vector to the embedding dimension.  This dramatically reduces the number of
    samples needed to estimate MI because the backbone already encodes the
    perceptually relevant structure.

    The backbone's original classification head is stripped; only the convolutional
    / feature-extraction layers are retained.  A global adaptive average pool is
    applied to collapse the spatial dimensions, producing a fixed-size feature
    vector regardless of the input image size.

    The trainable MLP head uses the same ``hidden_dim``, ``embed_dim``, and
    ``n_layers`` parameters as the standard MLP embedding, so all existing
    hyperparameter sweeps work unchanged.

    Parameters
    ----------
    input_dim : int
        Number of input channels (used to verify compatibility; must match the
        backbone's expected input channels, typically 3 for RGB).
    hidden_dim : int
        Hidden size of the trainable MLP head.
    embed_dim : int
        Dimensionality of the output embedding.
    n_layers : int
        Number of hidden layers in the MLP head.
    pytorch_predefined : str, optional
        Name of a ``torchvision.models`` model (e.g. ``'resnet18'``, ``'vgg16'``,
        ``'efficientnet_b0'``). Case-insensitive. Defaults to ``'resnet18'``.
    pretrained : bool, optional
        If True, load ImageNet-pretrained weights. Defaults to False.

    Notes
    -----
    Requires ``torchvision``. Install with ``pip install torchvision`` if not
    already available.

    The backbone is always frozen (``requires_grad=False``) regardless of the
    ``pretrained`` flag.  Only the MLP head is trainable.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int,
                 pytorch_predefined: Optional[str] = None,
                 pretrained: bool = False, **kwargs):
        super().__init__()
        try:
            import torchvision.models as _tv_models
        except ImportError as exc:
            raise ImportError(
                "PretrainedBackboneEmbedding requires torchvision. "
                "Install it with:  pip install torchvision"
            ) from exc

        model_name = (pytorch_predefined or 'resnet18').lower()

        # --- Load backbone from torchvision ---
        weights_arg = 'DEFAULT' if pretrained else None
        try:
            backbone_full = _tv_models.get_model(model_name, weights=weights_arg)
        except AttributeError:
            # Older torchvision API
            ctor = getattr(_tv_models, model_name, None)
            if ctor is None:
                raise ValueError(
                    f"Unknown torchvision model: '{model_name}'. "
                    f"Check torchvision.models for available names."
                )
            backbone_full = ctor(pretrained=pretrained)

        # --- Strip classification head, keep feature extractor ---
        # torchvision models expose a consistent .features or children() hierarchy.
        # We use AdaptiveAvgPool2d to make the backbone input-size agnostic.
        if hasattr(backbone_full, 'features'):
            # VGG, AlexNet, EfficientNet, etc.
            self.backbone = backbone_full.features
        else:
            # ResNet, DenseNet, etc.: drop the final (fc / classifier) layer
            children = list(backbone_full.children())
            # The last child is typically the linear classification head
            self.backbone = nn.Sequential(*children[:-1])

        # Global pool to collapse (H, W) → (1, 1), then flatten
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # --- Detect backbone's expected input channel count from its first Conv2d ---
        _backbone_in_ch = 3  # sensible default for all standard ImageNet models
        for _m in self.backbone.modules():
            if isinstance(_m, nn.Conv2d):
                _backbone_in_ch = _m.in_channels
                break

        # --- Channel adapter: trainable 1×1 conv when input_dim != backbone_in_ch ---
        if input_dim != _backbone_in_ch:
            import warnings as _warnings
            _warnings.warn(
                f"PretrainedBackboneEmbedding: input has {input_dim} channel(s) but "
                f"backbone '{pytorch_predefined}' expects {_backbone_in_ch}. "
                f"Adding a trainable 1×1 conv channel adapter.",
                UserWarning,
                stacklevel=2,
            )
            self._channel_adapt: Optional[nn.Module] = nn.Conv2d(
                input_dim, _backbone_in_ch, kernel_size=1, bias=False
            )
        else:
            self._channel_adapt = None

        # --- Detect expected input spatial size via a dummy forward at standard resolution ---
        _probe_size = 224
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, _backbone_in_ch, _probe_size, _probe_size)
            feat = self.global_pool(self.backbone(dummy))
            backbone_out_dim = feat.shape[1]

        # --- Spatial mismatch handling ---
        # Store the expected spatial size; actual upsample layer created lazily on first forward.
        self._expected_spatial = _probe_size
        self._pretrained = pretrained
        self._upsample: Optional[nn.Module] = None  # set on first forward if needed

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- Trainable MLP head ---
        self.head = MLP(backbone_out_dim, hidden_dim, embed_dim, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen backbone features then project via the trainable head.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, embed_dim)``.
        """
        # Channel adapter: map input channels to backbone's expected channel count
        if self._channel_adapt is not None:
            x = self._channel_adapt(x)

        # Lazy upsample: detect spatial size on first forward and add bilinear resize if needed
        H, W = x.shape[-2], x.shape[-1]
        if self._upsample is None and (H != self._expected_spatial or W != self._expected_spatial):
            import warnings as _warnings
            _warnings.warn(
                f"PretrainedBackboneEmbedding: input spatial size ({H}×{W}) does not match "
                f"the expected size ({self._expected_spatial}×{self._expected_spatial}). "
                f"Adding a bilinear upsample layer. This may reduce the quality of pretrained "
                f"features at very small input sizes but is generally acceptable.",
                UserWarning,
                stacklevel=2,
            )
            self._upsample = nn.Upsample(
                size=(self._expected_spatial, self._expected_spatial),
                mode='bilinear',
                align_corners=False,
            )
        if self._upsample is not None:
            x = self._upsample(x)
        with torch.no_grad():
            feat = self.global_pool(self.backbone(x))  # (N, backbone_out_dim)
        return self.head(feat)


class VariationalWrapper(nn.Module):
    """Generic variational wrapper for any embedding model.

    Adds a reparameterized Gaussian latent variable on top of any deterministic
    base encoder.  The base encoder is treated as a feature extractor that maps
    inputs to a deterministic representation of shape ``(batch, embed_dim)``.
    Two linear heads then project that representation to the mean ``μ`` and
    log-variance ``log σ²`` of a Gaussian distribution.

    At **training time** the forward pass returns a sample
    ``z = μ + ε·σ`` (reparameterization trick) together with the
    per-sample-averaged KL divergence ``KL(N(μ, σ²) ‖ N(0, I))``.

    At **evaluation time** the forward pass returns ``μ`` directly and a KL
    contribution of exactly ``0.0``, giving a deterministic, stable embedding
    for downstream tasks.

    This class replaces the former ``VarMLP`` and generalises variational
    embeddings to *all* encoder architectures (MLP, CNN1D, GRU, LSTM, TCN,
    Transformer, and any custom encoder).  Enable it by setting
    ``use_variational=True`` in ``base_params``; ``build_critic`` wraps the
    selected encoder automatically.

    Parameters
    ----------
    base_encoder : nn.Module
        Any embedding model whose ``forward`` method returns a tensor of shape
        ``(batch, embed_dim)``.
    embed_dim : int
        The dimensionality of the embedding produced by ``base_encoder``.

    Attributes
    ----------
    base_encoder : nn.Module
        The wrapped deterministic encoder.
    mu_head : nn.Linear
        Linear projection from ``embed_dim → embed_dim`` producing the mean.
    log_var_head : nn.Linear
        Linear projection from ``embed_dim → embed_dim`` producing the
        log-variance.  Output is clamped to ``[−10, 4]`` for numerical
        stability.
    """

    def __init__(self, base_encoder: nn.Module, embed_dim: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.mu_head = nn.Linear(embed_dim, embed_dim)
        self.log_var_head = nn.Linear(embed_dim, embed_dim)
        # Xavier uniform + zero-bias for both projection heads
        for head in (self.mu_head, self.log_var_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x`` through the base encoder and sample from the posterior.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor accepted by ``base_encoder`` (any shape).

        Returns
        -------
        z : torch.Tensor
            Shape ``(batch, embed_dim)``.  Sampled embedding at training time;
            mean embedding at evaluation time.
        kl_loss : torch.Tensor
            Scalar KL divergence ``KL(N(μ, σ²) ‖ N(0, I))`` normalised by
            batch size.  Returns ``0.0`` at evaluation time.
        """
        h = self.base_encoder(x)                     # (batch, embed_dim)
        mu = self.mu_head(h)                          # (batch, embed_dim)
        log_var = self.log_var_head(h).clamp(-10.0, 4.0)

        if not self.training:
            return mu, mu.new_tensor(0.0)

        # Reparameterization trick: z = μ + ε·σ
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std

        # KL divergence normalised by batch size for training stability
        kl_loss = -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.shape[0]
        return z, kl_loss