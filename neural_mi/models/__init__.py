# neural_mi/models/__init__.py
from .embeddings import MLP, VarMLP, BaseEmbedding
from .critics import SeparableCritic, ConcatCritic, BaseCritic