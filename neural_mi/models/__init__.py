# neural_mi/models/__init__.py
"""This package contains the core neural network models for the library.

It is organized into two main submodules:
- `embeddings`: Contains models for embedding input data into vector representations.
- `critics`: Contains models that use embeddings to compute MI estimates.
"""
from .embeddings import (
    MLP, VariationalWrapper, BaseEmbedding, CNN1D, CNN2D,
    GRU, LSTM, TCN, Transformer,
    PretrainedBackboneEmbedding,
)
from .critics import (
    SeparableCritic, ConcatCritic, BaseCritic, HybridCritic
)

__all__ = [
    'MLP', 'VariationalWrapper', 'BaseEmbedding', 'CNN1D', 'CNN2D',
    'GRU', 'LSTM', 'TCN', 'Transformer', 'PretrainedBackboneEmbedding',
    'SeparableCritic', 'ConcatCritic', 'BaseCritic', 'HybridCritic',
]
