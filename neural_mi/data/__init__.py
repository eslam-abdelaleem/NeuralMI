# neural_mi/data/__init__.py
"""This package contains modules for data processing and handling.

It provides the `DataHandler` as a unified interface and specific processors
for different data types like `ContinuousProcessor` and `SpikeProcessor`.
"""
from .handler import DataHandler
from .processors import SpikeProcessor, ContinuousProcessor, CategoricalProcessor, BaseProcessor