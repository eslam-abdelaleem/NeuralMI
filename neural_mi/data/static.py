# neural_mi/data/static.py
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
    
    def __init__(self, data, device=None, preprocessed=False):
        """
        Parameters
        ----------
        data : array-like or torch.Tensor
            Data of shape (n_channels, n_observations, ...)
            If preprocessed=True, assumes (n_observations, n_channels, ...)
        device : str, optional
            Device for tensor operations
        preprocessed : bool, optional
            If True, assumes data is already in the target shape (n_observations, n_channels, ...).
            Defaults to False.
        """
        super().__init__(device)
        
        # Validate input type and convert to numpy if it's a Tensor or list
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if not isinstance(data, np.ndarray):
            raise ValueError(f"Data must be a numpy array or torch.Tensor, got {type(data)}")

        # Check for invalid values
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values")
        
        self.data_orig = data
        self.preprocessed = preprocessed
        self._process()

    def _process(self):
        """
        Reshape input data array of shape `(n_channels, n_observations, ...)` 
        to 3D tensor of shape `(n_observations, n_channels, flattened_extra_dims)`
        Extra dimensions are flattened into 3rd dimension of stored tensor
        """
        # print(f"DEBUG: StaticDataset._process input shape={self.data_orig.shape}, preprocessed={self.preprocessed}")
        if self.preprocessed:
            # Assume data is already (n_obs, n_chan, n_feat)
            reshaped = self.data_orig
            # Ensure 3D
            if reshaped.ndim == 2:
                reshaped = reshaped[:, :, np.newaxis]
        else:
            # Reshape depending on original shape
            if self.data_orig.ndim == 1:
                reshaped = self.data_orig.reshape([-1,1,1])
            elif self.data_orig.ndim == 2:
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

        # print(f"DEBUG: StaticDataset._process output shape={reshaped.shape}")
        self.data = torch.tensor(reshaped, device=self.device, dtype=torch.float32)
        self.data_master = self.data.detach().clone()

    def __getitem__(self, idx):
        """Return data at index."""
        # Handle slice or array indexing
        try:
            return self.data[idx]
        except IndexError as e:
            print(f"DEBUG: StaticDataset.__getitem__ failed. idx={idx}")
            if isinstance(idx, tuple) and len(idx) > 0:
                 if hasattr(idx[0], 'dtype'):
                      print(f"DEBUG: idx[0].dtype={idx[0].dtype}")
            raise e
    
    def __len__(self):
        return self.data.shape[0]

    def apply_noise(self, amplitude):
        """Add Gaussian noise to data."""
        noise = torch.randn_like(self.data) * amplitude
        self.data = self.data_master + noise

    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        self.data = torch.round(self.data_master / precision_level) * precision_level