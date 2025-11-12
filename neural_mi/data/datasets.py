# neural_mi/data/datasets.py
import torch
import numpy as np
from typing import List, Optional, Union
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from neural_mi.logger import logger




def max_events_in_window(event_times: np.ndarray, window_size: float) -> int:
    """
    Find the maximum number of events that can fit in a window of given size.
    
    Parameters
    ----------
    event_times : np.array
        Array of event timestamps
    window_size : float
        Length of the time window in seconds/time units
        
    Returns
    -------
    int
        Maximum number of events that fit in any window of the given size. Returns 1 if
        no spikes are found to ensure a valid tensor shape.
    """
    times = np.array(event_times)    
    max_count = 0
    left = 0
    for right in range(len(times)):
        # Slide left pointer forward until window contains times[right] - window_size
        while times[right] - times[left] > window_size:
            left += 1
        # Update max_count if current window is larger
        max_count = max(max_count, right - left + 1)
    
    return max(1, max_count)



class TemporalWindowDataset(Dataset, ABC):
    """Base class for all temporal window datasets."""
    
    def __init__(self, window_manager=None, device=None):
        """
        Parameters
        ----------
        window_manager : WindowManager, optional
            External window manager for alignment.
        device : str, optional
            Device for tensor operations
        """
        self.window_manager = window_manager
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_orig = None
        self.data = None
        
    @abstractmethod
    def _move_data_to_windows(self):
        """Process raw data into windowed format."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """Return data at index."""
        pass
    
    def __len__(self):
        if self.window_manager:
            return self.window_manager.n_windows
        else:
            return 0
    
    @abstractmethod
    def _get_temporal_extent(self):
        """Return (t_start, t_end) of the data."""
        pass
    
    @abstractmethod
    def _validate_window_coverage(self, window_times):
        """Check which windows have sufficient data coverage."""
        pass

    @abstractmethod
    def time_shift(self, time_offset):
        """Apply time shift to data"""
        pass


class PreprocessedDataset(TemporalWindowDataset):
    """Dataset for already-processed data."""
    
    def __init__(self, data):
        super().__init__(window_size=1)  # Dummy value
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.n_windows = len(self.data)
    
    def _move_data_to_windows(self):
        pass
    
    def __getitem__(self, idx):
        return self.data[idx]


class ContinuousWindowDataset(TemporalWindowDataset):
    """Dataset for continuous time series data."""
    
    def __init__(self, data, time_vector=None, window_manager=None, device=None):
        """
        Parameters
        ----------
        data : array-like
            Continuous data of shape (n_channels, n_timepoints)
        time_vector : array-like, optional
            Time stamps for each sample. 
            If None, assumes data sampled on positive integers in time [0, 1, 2, etc.]
        window_manager : WindowManager, optional
            External window manager for alignment
        """
        super().__init__(window_manager, device)

        self.data_orig = torch.as_tensor(data, dtype=torch.float32)
        self.data = None # Will be allocated when moving to windows
        self.time_offset = 0 # By default, no offset is applied
        # Time vector handling
        # Right now if no time vector given, assuming sampled on positive integers
        if time_vector is not None:
            self.time_vector = np.asarray(time_vector)
            # Determine how many samples fit inside a window. Must only be done once, will serve as max window size
            # NOTE: There is a trivial, cheap version (done below) if continuous data is sampled at a constant rate
            # This method is decently expensive. 
            # TODO: Come up with a way to determine if data is given at fixed sample rate
            self.max_samples_per_window = max(1, max_events_in_window(self.time_vector, self.window_manager.window_size))
        else:
            self.time_vector = np.arange(0, self.data.shape[1])
            self.max_samples_per_window = max(1, np.floor(self.window_manager.window_size / 1.0).astype(int))

        if len(self.time_vector) != self.data_orig.shape[1]:
            raise ValueError(
                f"time_vector must be same length as data. "
                f"Recieved {len(self.time_vector)} time points and {self.data_orig.shape[1]} data points"
            )
        # Process data if window manager is available
        if self.window_manager is not None:
            self._move_data_to_windows()
    
    def _move_data_to_windows(self):
        """
        Convert continuous data into windows. 
        Continuous data of shape (n_channels, n_timepoints) is reshaped and interpolated 
        to (n_windows, n_channels, n_timepoints_in_window)
        
        Requires an attached window manager
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        # Preallocate output
        # TODO: Deal with dtypes. Right now I use float32 for efficiency/training tradeoffs. 
        # Makes a good default, but giving users control might be nice
        self.data = np.full((self.window_manager.n_windows, self.data_orig.shape[0], self.max_samples_per_window), 0, dtype=np.float32)
        # Get indices of which window time points belong to
        window_inds = np.searchsorted(self.window_manager.window_times, self.time_vector, side='right') - 1
        # Apply mask of which windows are valid to those indices
        mask = self.window_manager.valid_windows[window_inds]
        # Use those indices + mask to assign data to windows
        # Will interpolate data so that data is evenly sampled across window
        _, index, counts = np.unique(window_inds, return_counts=True, return_index=True)
        column_inds = np.concatenate(list(map(np.arange, counts)), axis=0)[mask]
        first_inds = np.repeat(index, counts)
        time_shifts = self.time_vector[first_inds] - self.window_manager.window_times[window_inds][first_inds]
        for i in range(self.data_orig.shape[0]):
            self.data[window_inds[mask],i,column_inds] = np.interp(
                self.time_vector[mask] + time_shifts[mask], 
                self.time_vector[mask], 
                self.data_orig[i,mask]
            )
        # Trim data to only valid windows
        # window_manager.window_times will stay long to include invalid windows,
        # as time shifting variables needs to refer to those windows again
        self.data = self.data[self.window_manager.valid_windows, :, :]
        self.data = torch.tensor(self.data, device=self.device) #TODO: Set device to have a reasonable default

    def _validate_window_coverage(self):
        """Check which windows have sufficient data coverage. Assumes window manager attached"""
        # TODO: Write a faster version that doesn't retrace things we've already done
        # Technically this has already been done in _move_data_to_windows, and it can be 
        # VERY expensive for larger arrays
        window_inds = np.searchsorted(self.window_manager.window_times, self.time_vector, side='right') - 1
        valid = np.full(self.window_manager.window_times.shape, False, bool)
        valid[window_inds] = True
        return valid
    
    def time_shift(self, time_offset):
        # Undo previous offset, apply new one
        self.time_vector = self.time_vector + time_offset - self.time_offset
        # Store this time offset to undo later
        self.time_offset = time_offset
        self._move_data_to_windows()

    def _get_temporal_extent(self):
        return self.time_vector[0], self.time_vector[-1]
    
    def __getitem__(self, idx):
        return self.data[idx, :, :]
    
    def add_noise(self, amplitude):
        """Add Gaussian noise to data."""
        noise = torch.randn_like(self.data_orig) * amplitude
        self.data = self.data_orig + noise
        self._move_data_to_windows()
        # For efficiency, can make a method of move data to windows which doesn't recalculate searchsorted 
        # Only new version needed if time vector changed
    
    def reset(self):
        """Reset to original data."""
        self.data = self.data_orig.clone()
        self._move_data_to_windows()


class SpikeWindowDataset(TemporalWindowDataset):
    """Dataset for spike train data."""
    
    def __init__(self, spike_times, window_size, step_size=None, 
                 max_spikes_per_window=None,
                 no_spike_value=0.0, device=None):
        super().__init__(window_size, step_size, device)
        
        self.spike_times_orig = [np.array(st) for st in spike_times]
        self.spike_times = [np.array(st) for st in spike_times]
        self.no_spike_value = no_spike_value
        self.max_spikes = max_spikes_per_window
        # Process data if window manager is available
        if self.window_manager is not None:
            self._move_data_to_windows()

    def _get_temporal_extent(self):
        """Return temporal extent of spike data."""
        valid_trains = [st for st in self.spike_times if len(st) > 0]
        if not valid_trains:
            return 0, 0
        t_start = min(st[0] for st in valid_trains)
        t_end = max(st[-1] for st in valid_trains)
        return t_start, t_end
    
    def _move_data_to_windows(self):
        """Convert spike times into windowed format."""
        # Determine temporal extent
        t_start = min(st[0] for st in self.spike_times if len(st) > 0)
        t_end = max(st[-1] for st in self.spike_times if len(st) > 0)
        
        # Create windows
        self.window_times = np.arange(t_start, t_end - self.window_size, self.step_size)
        self.n_windows = len(self.window_times)
        
        # Determine max spikes if not provided
        if self.max_spikes is None:
            self.max_spikes = self._compute_max_spikes()
        
        # Allocate tensor
        n_channels = len(self.spike_times)
        windowed_data = np.full(
            (self.n_windows, n_channels, self.max_spikes),
            self.no_spike_value,
            dtype=np.float32
        )
        
        # Fill windows
        for ch_idx, spike_train in enumerate(self.spike_times):
            window_inds = np.searchsorted(self.window_times, spike_train) - 1
            
            for win_idx in range(self.n_windows):
                mask = window_inds == win_idx
                spikes_in_window = spike_train[mask]
                
                if len(spikes_in_window) > 0:
                    n_spikes = min(len(spikes_in_window), self.max_spikes)
                    
                    if self.use_ISI:
                        # Encode as inter-spike intervals
                        isis = np.diff(spikes_in_window)
                        windowed_data[win_idx, ch_idx, 0] = spikes_in_window[0] - self.window_times[win_idx]
                        windowed_data[win_idx, ch_idx, 1:n_spikes] = isis[:n_spikes-1]
                    else:
                        # Encode as absolute times within window
                        relative_times = spikes_in_window[:n_spikes] - self.window_times[win_idx]
                        windowed_data[win_idx, ch_idx, :n_spikes] = relative_times
        
        self.data = torch.as_tensor(windowed_data, device=self.device)
        self.data_main = self.data.clone()
        self.spike_mask = self.data != self.no_spike_value
        self.valid_windows = torch.ones(self.n_windows, dtype=torch.bool)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _compute_max_spikes(self):
        """Compute maximum spikes in any window."""
        from .processors import find_max_spikes_per_window
        return max(find_max_spikes_per_window(st, self.window_size) 
                   for st in self.spike_times)
    
    def add_noise(self, amplitude):
        """Add temporal jitter to spike times."""
        noise = torch.empty_like(self.data[self.spike_mask]).uniform_(-amplitude/2, amplitude/2)
        self.data[self.spike_mask] = self.data_main[self.spike_mask] + noise
    
    def time_shift(self, offset):
        """Shift spike times by offset."""
        self.spike_times = [st + offset for st in self.spike_times_orig]
        self._process_data()


# class CategoricalWindowDataset(TemporalWindowDataset):
#     """Dataset for categorical/discrete time series data."""
    
#     def __init__(self, data, window_size, step_size=None, device=None):
#         super().__init__(window_size, step_size, device)
#         self.data_orig = torch.as_tensor(data, dtype=torch.long)
#         self.data = self.data_orig.clone()
#         self._process_data()
    
#     def _process_data(self):
#         """Convert categorical data into windows."""
#         n_channels, n_timepoints = self.data_orig.shape
#         n_windows = (n_timepoints - self.window_size) // self.step_size + 1
        
#         indices = torch.arange(n_windows)[:, None] * self.step_size + torch.arange(self.window_size)
#         self.windowed_data = self.data[:, indices]
        
#         self.n_windows = n_windows
#         self.window_times = torch.arange(n_windows) * self.step_size
#         self.valid_windows = torch.ones(n_windows, dtype=torch.bool)
    
#     def __getitem__(self, idx):
#         return self.windowed_data[:, idx, :]