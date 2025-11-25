# neural_mi/data/temporal.py
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from neural_mi.logger import logger


class StaticDataset(Dataset, ABC):
    """Base class for all temporal window datasets."""
    
    def __init__(self, device=None):
        """
        Parameters
        ----------
        device : str, optional
            Device for tensor operations
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_orig = None
        self.data = None

    @abstractmethod
    def __getitem__(self, idx):
        """Return data at index."""
        pass
    
    def __len__(self):
        return self.data.shape[0] if self.data else 0

    @abstractmethod
    def apply_noise(self, amplitude):
        """Apply noise to data."""
        pass

    @abstractmethod
    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        pass