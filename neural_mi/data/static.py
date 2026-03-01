# neural_mi/data/static.py
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from neural_mi.utils import get_device
from neural_mi.logger import logger


class BaseStaticDataset(Dataset, ABC):
    """Base class for all non-temporal, static datasets."""
    
    def __init__(self, device=None):
        """
        Parameters
        ----------
        device : str, optional
            Device for tensor operations
        """
        if device:
            self.device = device
        else:
            self.device = get_device()
        # Three versions of data are kept
        # 1. data: A numpy master matching the original but which can be modified (n_channels, n_timepoints)
        # 2. data_master: A clean tensor copy of data (n_windows, n_channels)
        # This is less memory-efficient but makes actions like applying noise repeatedly FAR faster
        self.data = None # Allocated when sub-method called
        self.data_master = None # # Allocated as needed or when sub-method called

    @abstractmethod
    def __getitem__(self, idx):
        """Return data at index."""
        pass
    
    def __len__(self):
        return self.data.shape[0] if self.data is not None else 0
    
    def _process(self):
        """Reshape input data array to 3D tensor of shape `(n_observations, n_channels, flattened_extra_dims)`"""
        pass

    @abstractmethod
    def apply_noise(self, amplitude):
        """Apply noise to data."""
        pass

    @abstractmethod
    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        pass



class StaticDataset(BaseStaticDataset):
    """Base class for all non-temporal, static datasets."""
    
    def __init__(self, data, device=None):
        """
        Parameters
        ----------
        data : array-like or torch.Tensor
            Data of shape (n_observations, n_channels, ...)
        device : str, optional
            Device for tensor operations
        """
        super().__init__(device)
        
        # Validate input type and convert to numpy if it's a list
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if not isinstance(data, np.ndarray):
            raise ValueError(f"Data must be a list, numpy array, or torch.Tensor, got {type(data)}")

        # Check for invalid values
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values")

        # Reshape input data array to tensor of shape `(n_observations, n_channels, *extra_dimensions)` 
        # Reshape depending on original shape
        if data.ndim == 1:
            reshaped = data.reshape([-1,1,1])
        elif data.ndim == 2:
            reshaped = data[:, :, None]
        else:
            # Higher-dimensional observations: 
            # Not possible to infer which dim is channel vs observation vs extra, 
            # so leaving to original shape (on user to get correct)
            reshaped = data
        self.data = torch.tensor(reshaped, device=self.device, dtype=torch.float32)

    def __getitem__(self, idx):
        """Return data at index."""
        return self.data[idx]
    
    def __len__(self):
        return self.data.shape[0]

    def apply_noise(self, amplitude):
        """Add Gaussian noise to data."""
        # Allocate data_master if not allocated yet
        if not hasattr(self, 'data_master'):
            self.data_master = self.data.detach().clone()
        noise = torch.randn_like(self.data) * amplitude
        self.data = self.data_master + noise

    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        # Allocate data_master if not allocated yet
        if not hasattr(self, 'data_master'):
            self.data_master = self.data.detach().clone()
        self.data = torch.round(self.data_master / precision_level) * precision_level