# neural_mi/data/__init__.py
"""This package contains modules for data processing and handling.

It provides the function `create_dataset` as a unified interface to create
paired datasets composed of different data types `ContinuousDataset` and `SpikeDataset`.
"""
from .handler import create_dataset, PairedDataset, PairedTemporalDataset
from .views import SubsetView