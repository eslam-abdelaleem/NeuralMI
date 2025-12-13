# neural_mi/data/views.py
import torch
import numpy as np

from .handler import PairedTemporalDataset
from neural_mi.logger import logger


class SubsetView:
    """
    Lightweight view into a PairedDataset or PairedTemporalDataset without copying data.
    Primarily used to keep track of indexing by time for temporal data. 
    Will automatically translate between window indices and actual times and sustain.

    Subset by indices, but then time shift data? This will automatically update 
    when temporal windows change

    Note that if a time shift is applied to temporal data, indexing times will be shifted 
    by however much x_dataset was shifted by. 
    """
    
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
        self.time_offset = 0
        
        # Register this view with the dataset so it gets notified of changes
        if not hasattr(dataset, '_subset_views'):
            dataset._subset_views = []
        dataset._subset_views.append(self)
        
        # Handle temporal subsetting
        if times is not None and self.is_temporal:
            # Validate inputs
            if times.ndim != 2:
                raise ValueError(f"Expected 2D array, got {times.ndim}D")
            elif times.shape[1] != 2:
                raise ValueError(f"Expected shape (n_regions, 2), got {times.shape}")
            if indices is not None:
                logger.warning("Both indices and times provided, using times only")
            # Store times as the source of truth for temporal views
            self.times = np.asarray(times)
            self.indices = None  # Will be computed from times
            self._update_indices_from_times()
        elif self.is_temporal and indices is not None:
            # Convert indices to times for temporal datasets
            self.indices = np.sort(np.asarray(indices))
            self.times = None  # Will be computed from indices
            self._update_times_from_indices()
        elif self.is_temporal:
            # No subsetting, use all data
            self.times = None
            self.indices = None
        else:
            # Non-temporal dataset
            self.times = None
            self.indices = np.sort(np.asarray(indices)) if indices is not None else None
        
        # Ensure indices are LongTensor
        if self.indices is not None:
            # print(f"DEBUG: SubsetView init indices type {type(self.indices)} dtype {getattr(self.indices, 'dtype', 'N/A')}")
            self.indices = torch.tensor(self.indices, device=self.dataset.x_dataset.device, dtype=torch.long)
    
    def _update_indices_from_times(self):
        """Convert time ranges to window indices. Called when windows change."""
        if self.times is None:
            self.indices = None
            return
            
        # Convert time ranges to indices using window_times
        window_times = self.dataset.window_manager.window_times
        start_end_inds = np.searchsorted(window_times, self.times, side='right') - 1
        start_end_inds = np.clip(start_end_inds, 0, len(window_times) - 1)
        
        indices = np.concatenate([np.arange(start_end_inds[i,0], start_end_inds[i,1] + 1) for i in range(start_end_inds.shape[0])])
        indices = np.sort(np.unique(indices))
        self.indices = torch.tensor(indices, device=self.dataset.x_dataset.device, dtype=torch.long)
    
    def _update_times_from_indices(self):
        """Convert indices to time ranges. Called once during initialization."""
        if self.indices is None or len(self.indices) == 0:
            self.times = None
            return
        
        # Find contiguous runs of indices
        breaks = np.where(np.diff(self.indices) > 1)[0]
        
        if len(breaks) == 0:
            # Single contiguous run
            starts = np.array([self.indices[0]])
            ends = np.array([self.indices[-1]])
        else:
            starts = np.concatenate([[self.indices[0]], self.indices[breaks + 1]])
            ends = np.concatenate([self.indices[breaks], [self.indices[-1]]])
        
        window_times = self.dataset.window_manager.window_times
        self.times = np.column_stack([window_times[starts], window_times[ends]])
    
    def _on_dataset_updated(self, time_shift=None):
        """
        Called by the dataset when windows are rebuilt (e.g., after time shift).
        Re-computes indices from stored times.
        """
        if not self.is_temporal or self.times is None:
            return
        
        # If time shift was applied, update our stored times
        if time_shift is not None:
            self.times = self.times + time_shift - self.time_offset
            self.time_offset = time_shift
        
        self._update_indices_from_times()
    
    def __len__(self):
        if self.indices is None:
            return len(self.dataset)
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
    
    def apply_noise(self, amplitude_x=0, amplitude_y=0):
        """Forward noise operations to underlying dataset."""
        self.dataset.apply_noise(amplitude_x, amplitude_y)

    def apply_precision(self, precision_x=0, precision_y=0):
        """Forward precision operations to underlying dataset."""
        self.dataset.apply_precision(precision_x, precision_y)
    
    def time_shift(self, offset_x=0.0, offset_y=0.0):
        """Forward time shift operations to underlying dataset."""
        if not self.is_temporal:
            logger.warning("Tried to time-shift a non-windowed dataset, skipping")
            return
        self.dataset.time_shift(offset_x, offset_y)
        # Note: indices will be updated automatically via _on_dataset_updated callback