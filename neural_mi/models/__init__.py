# neural_mi/models/__init__.py
"""This package contains the core neural network models for the library.

It is organized into two main submodules:
- `embeddings`: Contains models for embedding input data into vector representations.
- `critics`: Contains models that use embeddings to compute MI estimates.
"""
from .embeddings import MLP, VarMLP, BaseEmbedding, CNN1D
from .critics import SeparableCritic, ConcatCritic, BaseCritic, BilinearCritic, ConcatCriticCNN