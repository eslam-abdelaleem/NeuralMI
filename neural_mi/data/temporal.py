# neural_mi/data/temporal.py
import torch
import numpy as np
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
    """
    Base class for all temporal window datasets.

    Keeps 3 versions of main data
    1. data_orig: A numpy master matching the original (n_channels, n_timepoints), assigned in child classes
    2. data_master: A clean copy of data moved to windows (n_windows, n_channels, n_timepoints)
    3. data: A working copy of data moved to windows. 
    This is less memory-efficient but makes actions like applying noise repeatedly FAR faster
    """
    
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
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.data_master = None # Will be allocated when moving to windows
        self.data = None # Will be allocated when moving to windows
        self.time_offset = 0 # By default, no offset is applied

    def set_window_manager(self, window_manager):
        """Attach window manager and compute max samples/spikes per window."""
        self.window_manager = window_manager
        self.window_manager.register_observer(self)
        self._compute_max_samples_per_window()

    def _on_window_manager_updated(self):
        """Called when window manager parameters (window size, t_start, t_end) change."""
        self._compute_max_samples_per_window()
        # self._move_data_to_windows()
    
    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        pass
    
    @abstractmethod
    def move_data_to_windows(self):
        """Process raw data into windowed format."""
        pass

    @abstractmethod
    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage."""
        pass

    def remove_invalid_windows(self):
        """Trim data to only valid windows."""
        if self.window_manager is not None:
            self.data = self.data[self.window_manager.valid_windows, :, :]
            self.data_master = self.data_master[self.window_manager.valid_windows, :, :]
    
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
    def get_temporal_extent(self):
        """Return (t_start, t_end) of the data."""
        pass

    @abstractmethod
    def time_shift(self, time_offset):
        """Apply time shift to data."""
        pass

    @abstractmethod
    def apply_noise(self, amplitude):
        """Apply noise to data."""
        pass

    @abstractmethod
    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        pass



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

        self.data_orig = data
        # Time vector handling. If no time vector given, assuming sampled on positive integers
        if time_vector is not None:
            self.time_vector = np.asarray(time_vector)
        else:
            self.time_vector = np.arange(0, self.data.shape[1])
        # Infer sample rate from time vector
        # This class assumes, outside of jumps, a constant sample rate!
        self.period = self.time_vector[1] - self.time_vector[0]

        if len(self.time_vector) != self.data_orig.shape[1]:
            raise ValueError(
                f"time_vector must be same length as data. "
                f"Recieved {len(self.time_vector)} time points and {self.data_orig.shape[1]} data points"
            )
        # Process data if window manager is available
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None

    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        # NOTE: There is a trivial, cheap version if data has fixed sample rate, and an expensive one using searchsorted
        # For now always assuming fixed sample rate (outside of large jumps)
        self.max_samples_per_window = np.ceil(self.window_manager.window_size / self.period).astype(int)
    
    def move_data_to_windows(self):
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
        data_shape = (self.window_manager.n_windows, self.data_orig.shape[0], self.max_samples_per_window)
        data = np.full(data_shape, 0.0, dtype=np.float32)
        # Don't move data that occurs after the last window
        mask = np.logical_and(
            self.time_vector <= self.window_manager.t_end, 
            self.time_vector >= self.window_manager.t_start
        )
        # Get indices of which window each time point belongs to
        # For millions of entries, this is by far the slowest part of this whole function!
        # For that reason we'll be memory-hungry and cache it
        self._cached_window_inds = np.searchsorted(self.window_manager.window_times, self.time_vector[mask], side='right') - 1
        window_inds = self._cached_window_inds
        # Use those indices to assign data to windows
        # Will interpolate data so that data is evenly sampled across window
        _, index, counts = np.unique(window_inds, return_counts=True, return_index=True)
        column_inds = np.concatenate(list(map(np.arange, counts)), axis=0)
        first_inds = np.repeat(index, counts)
        time_shifts = self.time_vector[mask][first_inds] - self.window_manager.window_times[window_inds][first_inds]
        for i in range(self.data_orig.shape[0]):
            data[window_inds,i,column_inds] = np.interp(
                self.time_vector[mask] + time_shifts, 
                self.time_vector[mask], 
                self.data_orig[i,mask]
            )
        self.data = torch.tensor(data, device=self.device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage. Assumes window manager attached"""
        window_inds = self._cached_window_inds
        valid = np.full(self.window_manager.window_times.shape, False, bool)
        valid[window_inds] = True
        return valid
    
    def time_shift(self, offset):
        # Undo previous offset, apply new one
        self.time_vector = self.time_vector + offset - self.time_offset
        # Store this time offset to undo later
        self.time_offset = offset
        # Moving data to windows will be orchestrated by paired dataset

    def get_temporal_extent(self):
        return self.time_vector[0], self.time_vector[-1]
    
    def __getitem__(self, idx):
        return self.data[idx, :, :]
    
    def reset(self):
        """Undo any added noise by resetting to original data. Does not undo time shifts."""
        self.data = self.data_master.detach().clone()

    def apply_noise(self, amplitude):
        """Add Gaussian noise to data."""
        # Reset to master copy if amplitude is zero. Useful as another interface to undo changes
        if amplitude == 0.0:
            self.reset()
            return
        # If data mask and noise buffer haven't been created, compute that now
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data, as_tuple=True)
        # Pre-allocate noise tensor to avoid repeated allocations when applying noise
        if not hasattr(self, '_noise_buffer'):
            self._noise_buffer = torch.empty(len(self._data_mask[0]), device=self.device, dtype=self.data.dtype)
        self._noise_buffer.normal_(mean=0, std=amplitude)
        self.data[self._data_mask] = self.data_master[self._data_mask] + self._noise_buffer

    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        # Reset to master copy if zero. Avoids divide by zero, useful as interface to undo changes
        if precision_level == 0.0:
            self.reset()
            return
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data, as_tuple=True)
        self.data[self._data_mask] = torch.round(self.data_master[self._data_mask] / precision_level) * precision_level



class SpikeWindowDataset(TemporalWindowDataset):
    """Dataset for spike train data."""
    
    def __init__(self, spike_times, 
                 window_manager=None,
                 no_spike_value=0.0, device=None):
        super().__init__(window_manager, device)

        self.data_orig = [np.array(st) for st in spike_times]
        self.no_spike_value = no_spike_value
        self.time_offset = 0
        # Process data if window manager is available
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None
    
    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        self.max_samples_per_window = np.max(np.array([
            max_events_in_window(x, self.window_manager.window_size) 
            for x in self.data_orig
        ]))

    def get_temporal_extent(self):
        """Return temporal extent of spike data."""
        valid_trains = [st for st in self.data_orig if len(st) > 0]
        if not valid_trains:
            return 0, 0
        t_start = min(st[0] for st in valid_trains)
        t_end = max(st[-1] for st in valid_trains)
        return t_start, t_end
    
    def move_data_to_windows(self):
        """
        Convert spike times into windowed format. 
        Spike data of form [(n_channels, n_spikes)] is reshaped to 
        (n_windows, n_channels, n_timepoints_in_window)
        
        Requires an attached window manager
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        # Preallocate
        data_shape = (self.window_manager.n_windows, len(self.data_orig), self.max_samples_per_window)
        data = np.full(data_shape, self.no_spike_value, dtype=np.float32)
        self._cached_window_inds = []
        # Loop over channels/neurons/muscles
        for i in range(len(self.data_orig)):
            # Don't move data that occurs after the last window
            mask = np.logical_and(
                self.data_orig[i] >= self.window_manager.t_start,
                self.data_orig[i] <= self.window_manager.t_end 
            )
            # Use searchsorted on each neuron/muscle to assign to chunks
            # For millions of spikes, this is by far the slowest part of this whole function!
            window_inds = np.searchsorted(self.window_manager.window_times, self.data_orig[i][mask]) - 1

            _, counts = np.unique(window_inds, return_counts=True)
            column_inds = np.concatenate(list(map(np.arange, counts)), axis=0)
            data[window_inds,i,column_inds] = self.data_orig[i][mask] - self.window_manager.window_times[window_inds]
            self._cached_window_inds.append(window_inds)
        # Convert to tensor, move to device. Make copies that noise will be applied on
        self.data = torch.tensor(data, device=self.device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage. Assumes window manager attached"""
        windows_with_spikes = np.unique(np.concatenate(self._cached_window_inds))
        valid = np.full(self.window_manager.window_times.shape, False, bool)
        valid[windows_with_spikes] = True
        return valid
    
    def __getitem__(self, idx):
        return self.data[idx,:,:]
    
    def reset(self):
        """Undo any added noise by resetting to original data. Does not undo time shifts."""
        self.data = self.data_master.detach().clone()
    
    def time_shift(self, offset):
        """Shift spike times by offset."""
        # Undo previous offset, apply new one
        self.data_orig = [st + offset - self.time_offset for st in self.data_orig]
        # Store this time offset to undo later
        self.time_offset = offset
        # Moving data to windows will be orchestrated by paired dataset class

    def apply_noise(self, amplitude):
        """Add temporal jitter to spike times."""
        # Reset to master copy if amplitude is zero. Useful as another interface to undo noise
        if amplitude == 0.0:
            self.reset()
            return
        # If spike mask hasn't been created, compute that now to more quickly modify spikes
        if not hasattr(self, '_spike_mask'):
            self._data_mask = torch.nonzero(self.data != self.no_spike_value, as_tuple=True)
        # Pre-allocate noise tensor to avoid repeated allocations when applying noise
        if not hasattr(self, '_noise_buffer'):
            self._noise_buffer = torch.empty(len(self._data_mask[0]), device=self.device, dtype=self.data.dtype)
        self._noise_buffer.uniform_(-amplitude / 2, amplitude / 2)
        self.data[self._data_mask] = self.data_master[self._data_mask] + self._noise_buffer

    def apply_precision(self, precision_level):
        """Round spike times to a specific resolution/precision level."""
        # Reset to master copy if zero. Avoids divide by zero, useful as interface to undo changes
        if precision_level == 0.0:
            self.reset()
            return
        # If spike mask hasn't been created, compute that now
        if not hasattr(self, '_spike_mask'):
            self._data_mask = torch.nonzero(self.data != self.no_spike_value, as_tuple=True)
        self.data[self._data_mask] = torch.round(self.data[self._data_mask] / precision_level) * precision_level




class CategoricalWindowDataset(TemporalWindowDataset):
    """Dataset for categorical time series data with one-hot encoding."""
    
    def __init__(self, data, time_vector=None, window_manager=None, device=None):
        """
        Parameters
        ----------
        data : array-like
            Categorical data of shape (n_channels, n_timepoints) with integer category labels
        time_vector : array-like, optional
            Time stamps for each sample. 
            If None, assumes data sampled on positive integers in time [0, 1, 2, etc.]
        window_manager : WindowManager, optional
            External window manager for alignment
        device : str, optional
            Device for tensor operations
        """
        super().__init__(window_manager, device)

        # Store original categorical data, convert to ints if not
        arr = np.array(data)
        if not np.issubdtype(arr.dtype, np.integer):
            _, indices = np.unique(arr, return_inverse=True)
            arr = indices
        self.data_orig = np.asarray(arr, dtype=np.int32)
        
        # Time vector handling
        if time_vector is not None:
            self.time_vector = np.asarray(time_vector)
        else:
            self.time_vector = np.arange(0, self.data_orig.shape[1])
        # Infer sample rate from time vector
        # This class assumes, outside of jumps, a constant sample rate!
        self.period = self.time_vector[1] - self.time_vector[0]

        if len(self.time_vector) != self.data_orig.shape[1]:
            raise ValueError(
                f"time_vector must be same length as data. "
                f"Received {len(self.time_vector)} time points and {self.data_orig.shape[1]} data points"
            )
        
        # Total one-hot encoded dimensions
        self.n_categories = self.data_orig.max() + 1
        
        # Process data if window manager is available
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None
    
    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        self.max_samples_per_window = np.ceil(self.window_manager.window_size / self.period).astype(int)
    
    def move_data_to_windows(self):
        """
        Convert categorical data into windows with one-hot encoding.
        Categorical data of shape (n_channels, n_timepoints) is reshaped and mapped to 
        (n_windows, n_channels, n_timepoints_in_window * n_categories)
        
        Requires an attached window manager.
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        # Preallocate output (one-hot encoded) (n_channels, n_windows, window_size * n_categories)
        data_shape = (self.window_manager.n_windows, self.data_orig.shape[0], self.n_categories * self.max_samples_per_window)
        data = np.zeros(data_shape, dtype=np.bool)
        
        # Don't move data that occurs after the last window
        mask = np.logical_and(
            self.time_vector <= self.window_manager.t_end, 
            self.time_vector >= self.window_manager.t_start
        )
        # Get indices of which window time points belong to
        self._cached_window_inds = np.searchsorted(self.window_manager.window_times, self.time_vector[mask], side='right') - 1
        window_inds = self._cached_window_inds
        # Compute column indices for placement
        _, counts = np.unique(window_inds, return_counts=True)
        column_inds = np.concatenate(list(map(np.arange, counts)), axis=0)
        # Expand column inds for one-hot encoding
        expanded = column_inds[:,None] * self.n_categories + np.arange(self.n_categories)
        expanded_column_inds = expanded.ravel()
        # Expand window indices for one-hot encoding
        expanded_window_inds = np.repeat(window_inds, self.n_categories)
        for i in range(self.data_orig.shape[0]):
            data[expanded_window_inds,i,expanded_column_inds] = np.eye(self.n_categories)[self.data_orig[i,mask]].flatten()
        self.data = torch.tensor(data, device=self.device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage."""
        window_inds = self._cached_window_inds
        valid = np.full(self.window_manager.window_times.shape, False, dtype=bool)
        valid[window_inds] = True
        return valid
    
    def time_shift(self, offset):
        """Apply time shift to data."""
        # Undo previous offset, apply new one
        self.time_vector = self.time_vector + offset - self.time_offset
        # Store this time offset to undo later
        self.time_offset = offset
        # Moving data to windows on new time shift will be orchestrated by paired dataset

    def get_temporal_extent(self):
        """Return temporal extent of the data."""
        return self.time_vector[0], self.time_vector[-1]
    
    def __getitem__(self, idx):
        """Return one-hot encoded windowed data at index."""
        return self.data[idx, :, :]
    
    def reset(self):
        """Undo any added noise by resetting to original data. Does not undo time shifts."""
        self.data = self.data_master.detach().clone()

    def apply_noise(self, amplitude):
        """
        Add noise to categorical data by randomly flipping categories.
        
        Parameters
        ----------
        amplitude : float
            Probability of flipping a category (between 0 and 1)
        """
        pass

    def apply_precision(self, precision_level):
        """
        Precision doesn't apply to categorical data in the same way.
        This is a no-op for categorical data since categories are already discrete.
        """
        logger.warning("Precision adjustment not applicable to categorical data. Ignoring.")
        pass