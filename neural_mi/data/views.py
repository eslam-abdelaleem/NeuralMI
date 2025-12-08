# neural_mi/data/views.py
import torch
import numpy as np

from .handler import PairedTemporalDataset
from neural_mi.logger import logger

class SubsetView:
    """Lightweight view into a PairedDataset or PairedTemporalDataset without copying data."""
    
    def __init__(self, dataset, indices=None, times=None, channels_x=None, channels_y=None):
        """
        Parameters
        ----------
        dataset : PairedDataset or PairedTemporalDataset
            The underlying dataset to view
        indices : np.ndarray, optional
            Indices of windows/samples to include
        times : np.ndarray, optional
            Time regions of data to include. 
            Shape (n_regions, 2). Each row is [start_time, end_time] pair
            Exclusive with indices
        channels_x : np.ndarray, optional
            Channel indices to select from X data
        channels_y : np.ndarray, optional
            Channel indices to select from Y data
        """
        self.dataset = dataset
        self.channels_x = channels_x
        self.channels_y = channels_y
        self.is_temporal = isinstance(dataset, PairedTemporalDataset)
        # Temporal and times given, use those over indices
        if times is not None and self.is_temporal:
            # Validate inputs
            if times.ndim != 2:
                raise ValueError(f"Expected 2D array, got {times.ndim}D")
            elif times.shape[1] != 2:
                raise ValueError(f"Expected shape (n_regions, 2) or (2, n_regions), got {times.shape}")
            if indices is not None:
                logger.warning("Both indices and times provided for subsetting temporal dataset, using times only")
            # Save times, get matching indices
            self.times = times
            self.indices = self._indices_from_times(self)
        # Temporal but just indices given, save to times
        elif self.is_temporal:
            self.indices = np.sort(indices)
            self.times = self._times_from_indices()
        # Not temporal
        else:
            self.times = None
            self.indices = np.sort(indices)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map through index subset if present
        actual_idx = self.indices[idx] if self.indices is not None else idx
        x, y = self.dataset[actual_idx]
        
        # Apply channel subsetting
        if self.channels_x is not None:
            x = x[:, self.channels_x, :] if x.ndim == 3 else x[self.channels_x, :]
        if self.channels_y is not None and y is not None:
            y = y[:, self.channels_y, :] if y.ndim == 3 else y[self.channels_y, :]
        
        return x, y
    
    def _indices_from_times(self):
        """Convert time ranges to indices."""
        start_end_inds = np.searchsorted(self.dataset.window_manager.window_times, self.times, side='right') - 1
        indices = np.concatenate([np.arange(start_end_inds[i,0], start_end_inds[i,1] + 1) for i in range(start_end_inds.shape[0])])
        self.indices = np.sort(np.unique(indices))

    def _times_from_indices(self):
        """Convert indices to time ranges."""
        # First get contiguous runs of indices
        breaks = np.where(np.diff(self.indices) > 1)[0]
        starts = np.concatenate([[self.indices[0]], self.indices[breaks + 1]])
        ends = np.concatenate([[self.indices[breaks]], self.indices[-1]])
        run_inds = np.column_stack([starts, ends])
        self.times = self.dataset.window_manager.window_times[run_inds]
    
    def apply_noise(self, amplitude_x=0, amplitude_y=0):
        """Forward noise operations to underlying dataset."""
        self.dataset.apply_noise(amplitude_x, amplitude_y)

    def apply_precision(self, amplitude_x=0, amplitude_y=0):
        """Forward noise operations to underlying dataset."""
        self.dataset.apply_precision(amplitude_x, amplitude_y)
    
    def time_shift(self, offset_x=0, offset_y=0):
        """Forward time shift operations to underlying dataset."""
        if not self.is_temporal:
            logger.warning("Tried to time-shift a non-windowed dataset, skipping")
            return
        self.dataset.time_shift(offset_x, offset_y)
        # Time shifting rebuilds windows, so figure out new indices that are valid from times
        self._indices_from_times()
