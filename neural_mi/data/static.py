# neural_mi/data/static.py
import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from neural_mi.utils import get_device


class BaseStaticDataset(Dataset, ABC):
    """Base class for all non-temporal, static datasets."""

    def __init__(self, device=None, data_device='cpu'):
        """
        Parameters
        ----------
        device : str, optional
            Compute device used by the model (CPU/CUDA/MPS).  Datasets do not
            use this for data storage; it is kept for reference only.
        data_device : str, optional
            Device on which ``self.data`` tensors are stored.  Defaults to
            ``'cpu'``, which keeps large dataset allocations in pageable system
            RAM and lets the OS reclaim memory freely between tasks.  Pass
            ``'auto'`` to co-locate data with the compute device (useful when
            the same dataset is evaluated many times without reloading, e.g.
            precision analysis).
        """
        self.device = get_device() if not device else torch.device(str(device))
        # Resolve data storage device separately from compute device.
        if data_device == 'auto':
            self.data_device = self.device
        elif data_device is None or str(data_device) == 'cpu':
            self.data_device = torch.device('cpu')
        else:
            self.data_device = torch.device(str(data_device))
        # Two versions of data are kept:
        # 1. data        : working tensor, may have noise/precision applied
        # 2. data_master : clean clone, used to restore data after modifications
        self.data = None        # Allocated when sub-method called
        self.data_master = None # Allocated as needed or when sub-method called

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
    """Dataset for pre-processed, non-temporal data."""

    def __init__(self, data, device=None, data_device='cpu'):
        """
        Parameters
        ----------
        data : array-like, torch.Tensor, or 2-tuple of either
            Data of shape (n_observations, n_channels, ...). A 2-tuple
            ``(a_data, c_data)`` is the compound "X-role" input used by
            ``mode='conditional'``'s ``align='dual_branch'`` path (paired
            with ``DualBranchEmbedding``, see ``THEORY.md``): each element is
            processed exactly like a single-array ``data`` would be, then
            stored as a matching tuple, so the two can have different
            trailing (window) shapes while sharing the leading (sample)
            dimension. A plain Python ``list`` is not treated this way, that
            still means "array-like, convert via ``np.array``", only an
            actual ``tuple`` triggers the compound path.
        device : str, optional
            Compute device (kept for reference; not used for data storage).
        data_device : str, optional
            Device for storing ``self.data``.  Defaults to ``'cpu'``.
            Pass ``'auto'`` to use the compute device instead.
        """
        super().__init__(device, data_device)

        if isinstance(data, tuple):
            if len(data) != 2:
                raise ValueError(
                    f"StaticDataset's compound (tuple) data must have exactly 2 "
                    f"elements (a_data, c_data), got {len(data)}."
                )
            self.data = tuple(self._prepare_one(d) for d in data)
            return

        self.data = self._prepare_one(data)

    def _prepare_one(self, data) -> torch.Tensor:
        """Validate and reshape a single array-like into a
        ``(n_observations, n_channels, *extra_dimensions)`` tensor. Factored
        out of ``__init__`` so the compound (tuple) path can apply the exact
        same treatment to each of its two elements."""
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
        return torch.tensor(reshaped, device=self.data_device, dtype=torch.float32)

    def __getitem__(self, idx):
        """Return data at index."""
        if isinstance(self.data, tuple):
            return tuple(d[idx] for d in self.data)
        return self.data[idx]

    def __len__(self):
        if isinstance(self.data, tuple):
            return self.data[0].shape[0]
        return self.data.shape[0]

    def apply_noise(self, amplitude):
        """Add Gaussian noise to data."""
        if isinstance(self.data, tuple):
            raise NotImplementedError(
                "apply_noise is not supported for StaticDataset's compound "
                "(tuple) data (mode='conditional'(align='dual_branch')). "
                "mode='precision' isn't used with the quantities that need "
                "this path (MI rate, instantaneous exchange, directed "
                "information rate)."
            )
        # Allocate data_master if not allocated yet
        if self.data_master is None:
            self.data_master = self.data.detach().clone()
        noise = torch.randn_like(self.data) * amplitude
        self.data = self.data_master + noise

    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        if isinstance(self.data, tuple):
            raise NotImplementedError(
                "apply_precision is not supported for StaticDataset's compound "
                "(tuple) data (mode='conditional'(align='dual_branch')). "
                "mode='precision' isn't used with the quantities that need "
                "this path (MI rate, instantaneous exchange, directed "
                "information rate)."
            )
        # Allocate data_master if not allocated yet
        if self.data_master is None:
            self.data_master = self.data.detach().clone()
        self.data = torch.round(self.data_master / precision_level) * precision_level