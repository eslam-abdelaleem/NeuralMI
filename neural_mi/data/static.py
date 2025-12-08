# neural_mi/data/temporal.py
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from neural_mi.utils import get_device
from neural_mi.logger import logger

# TODO: Could co-opt time-shifting interface to allow shifting rows of x relative to y


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
        # 1. data_orig: A numpy master matching the original (n_channels, n_timepoints), assigned in child classes
        # 2. data_master: A clean tensor copy of data moved to windows (n_windows, n_channels, n_timepoints)
        # 3. data: A working tensor copy of data moved to windows. 
        # This is less memory-efficient but makes actions like applying noise repeatedly FAR faster
        self.data_master = None # Will be allocated when moving to windows
        self.data = None # Will be allocated when moving to windows

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
        data : array-like
            Data of shape (n_channels, n_observations, ...)
        device : str, optional
            Device for tensor operations
        """
        super().__init__(device)
        
        # Validate input type
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Data must be a numpy array, got {type(data)}")
        # Check for invalid values
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values")
        
        self.data_orig = data
        self._process()

    def _process(self):
        """
        Reshape input data array of shape `(n_channels, n_observations, ...)` 
        to 3D tensor of shape `(n_observations, n_channels, flattened_extra_dims)`
        Extra dimensions are flattened into 3rd dimension of stored tensor
        """
        # Reshape depending on original shape
        if self.data_orig.ndim == 1:
            reshaped = self.data_orig.reshape([-1,1,1])
        if self.data_orig.ndim == 2:
            # Reshape: (n_channels, n_observations) -> (n_observations, n_channels, 1)
            reshaped = self.data_orig.T[:, :, np.newaxis]
        else:
            # Higher-dimensional observations
            n_chan = self.data_orig.shape[0]
            n_obs = self.data_orig.shape[1]
            feature_dim = np.prod(self.data_orig.shape[2:])
            # Reshape: (n_channels, n_observations, ...) -> (n_obs, n_chan, feature_dim)
            # First transpose to move observations to front: (n_obs, n_chan, ...) then flatten trailing dimensions
            reshaped = np.moveaxis(self.data_orig, 1, 0).reshape(n_obs, n_chan, feature_dim)
        self.data = torch.tensor(reshaped, device=self.device)
        self.data_master = self.data.detach().clone()

    def __getitem__(self, idx):
        """Return data at index."""
        return self.data[idx, :, :]
    
    def __len__(self):
        return self.data.shape[0]

    def apply_noise(self, amplitude):
        """Add Gaussian noise to data."""
        noise = torch.randn_like(self.data) * amplitude
        self.data = self.data_master + noise

    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        self.data = torch.round(self.data_master / precision_level) * precision_level