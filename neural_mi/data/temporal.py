# neural_mi/data/temporal.py
import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from neural_mi.utils import get_device
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
    1. data_orig: A numpy master matching the original (n_timepoints, n_channels), assigned in child classes
    2. data_master: A clean copy of data moved to windows (n_windows, n_timepoints_in_window, n_channels)
    3. data: A working copy of data moved to windows. 
    This is less memory-efficient but makes actions like applying noise repeatedly FAR faster
    """
    
    def __init__(self, window_manager=None, device=None, data_device='cpu'):
        """
        Parameters
        ----------
        window_manager : WindowManager, optional
            External window manager for alignment.
        device : str, optional
            Compute device used by the model.  Kept for reference; not used
            for data tensor storage.
        data_device : str, optional
            Device on which ``self.data`` tensors are stored.  Defaults to
            ``'cpu'`` so that large windowed arrays live in pageable system RAM
            and the OS can reclaim memory freely between training tasks.  Pass
            ``'auto'`` to co-locate data with the compute device.
        """
        self.window_manager = window_manager
        self.device = get_device() if not device else torch.device(str(device))
        if data_device == 'auto':
            self.data_device = self.device
        elif data_device is None or str(data_device) == 'cpu':
            self.data_device = torch.device('cpu')
        else:
            self.data_device = torch.device(str(data_device))
        self.data_master = None  # Allocated when moving to windows
        self.data = None         # Allocated when moving to windows
        self.time_offset = 0     # No offset by default

    # ------ Window handling
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
        if self.window_manager is not None and self.data is not None:
            self.data = self.data[self.window_manager.valid_windows, :, :]
            self.data_master = self.data_master[self.window_manager.valid_windows, :, :]
    
    # ------ Size and indexing
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
    
    # ------ Modifying data
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
    """
    Dataset for continuous time series data.

    Note this class allows for irregular jumps in time between blocks of data, 
    but assumes a constant sample rate.
    """
    
    def __init__(self, data, time_vector=None, window_manager=None, device=None,
                 min_coverage_fraction=0.2, data_device='cpu'):
        """
        Parameters
        ----------
        data : array-like
            Continuous data of shape (n_timepoints, n_channels).
            If more than 3D (n_timepoints, n_channels, *extra_dims), it will
            be flattened to (n_timepoints, n_channels*...).
        time_vector : array-like, optional
            Time stamps for each sample.
            If None, assumes data sampled on positive integers [0, 1, 2, ...].
        window_manager : WindowManager, optional
            External window manager for alignment.
        device : str, optional
            Compute device (kept for reference; not used for data storage).
        min_coverage_fraction : float, optional
            Minimum fraction of a window that must be covered by actual data
            for the window to be considered valid.
        data_device : str, optional
            Device for storing ``self.data``.  Defaults to ``'cpu'``.
        """
        super().__init__(window_manager, device, data_device)

        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if data.ndim == 1:
            data = np.expand_dims(data, 1)  # (n_timepoints, 1)
        elif data.ndim >= 3:
            # Flatten (n_timepoints, n_channels, *extra_dims) -> (n_timepoints, n_channels*...)
            n_timepoints = data.shape[0]
            data = data.reshape(n_timepoints, -1)
        self.data_orig = data  # Now (n_timepoints, n_channels)

        # Time vector handling. If no time vector given, assuming sampled on positive integers
        if time_vector is not None:
            self.time_vector = np.asarray(time_vector)
        else:
            self.time_vector = np.arange(0, self.data_orig.shape[0])
        # Infer sample rate from time vector
        diffs = np.diff(self.time_vector)
        self.period = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        if len(diffs) > 1:
            first_two = float(self.time_vector[1] - self.time_vector[0])
            if not np.isclose(first_two, self.period, rtol=0.05):
                from neural_mi.logger import logger as _logger
                _logger.warning(
                    f"ContinuousWindowDataset: first inter-sample interval "
                    f"({first_two:.6g}) differs from median ({self.period:.6g}) by "
                    f">5%. Using median. Check for boundary artifacts in your time vector."
                )

        self.min_coverage_fraction = min_coverage_fraction

        if len(self.time_vector) != self.data_orig.shape[0]:
            raise ValueError(
                f"time_vector must be same length as data. "
                f"Recieved {len(self.time_vector)} time points and {self.data_orig.shape[0]} data points"
            )
        # Process data if window manager is available
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None

    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        # Assuming fixed sample rate (outside of large jumps) for efficiency.
        # Adding a +1 buffer to avoid index out of bounds during interpolation at edges.
        self.max_samples_per_window = np.ceil(self.window_manager.window_size / self.period).astype(int) + 1
    
    def move_data_to_windows(self):
        """
        Convert continuous data into windows. 
        Continuous data of shape (n_timepoints, n_channels) is reshaped and interpolated 
        to (n_windows, n_channels, n_timepoints_in_window)
        
        Requires an attached window manager
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        # Preallocate output
        # max_samples_per_window is roughly window_size / period
        n_channels = self.data_orig.shape[1]
        data_shape = (self.window_manager.n_windows, n_channels, self.max_samples_per_window)
        data = np.full(data_shape, 0.0, dtype=np.float32)

        # Create target time grid for ALL windows: (n_windows, max_samples)
        # target_times[w, t] = window_start[w] + t * period
        time_offsets = np.arange(self.max_samples_per_window) * self.period

        # Use broadcasting to create (n_windows, max_samples) target times
        # Shape: (n_windows, 1) + (max_samples,) -> (n_windows, max_samples)
        target_times = self.window_manager.window_times[:, None] + time_offsets[None, :]

        # Flatten target times to 1D for efficient interpolation
        target_times_flat = target_times.ravel() # (n_windows * max_samples)

        # Mask for out-of-bounds (zero padding)
        # We check if target time is outside the range of actual data timestamps
        # Note: assuming time_vector is sorted
        t_min = self.time_vector[0]
        t_max = self.time_vector[-1]
        out_of_bounds = (target_times_flat < t_min) | (target_times_flat > t_max)
        n_oob = int(out_of_bounds.sum())
        total = len(target_times_flat)
        
        oob_frac = 0.0
        oob_windows = np.array([], dtype=int)
        if n_oob > 0:
            oob_frac = n_oob / total
            # Find which windows are affected
            oob_windows = np.unique(np.where(out_of_bounds.reshape(
                self.window_manager.n_windows, self.max_samples_per_window))[0])
            log_fn = logger.warning if oob_frac > 0.1 else logger.debug
            log_fn(
                f"ContinuousWindowDataset: {n_oob}/{total} interpolated time points "
                f"({oob_frac:.1%}) are zero-padded due to data gaps. "
                f"Affects {len(oob_windows)} window(s). "
                f"{'Consider checking your time vector for large gaps.' if oob_frac > 0.1 else ''}"
            )

        # Interpolate for each channel
        for i in range(n_channels):
            # np.interp expects 1D arrays
            # We interpolate the entire flattened grid at once
            interp_vals = np.interp(
                target_times_flat,
                self.time_vector,
                self.data_orig[:, i]
            ).astype(np.float32)
            # Apply zero padding
            interp_vals[out_of_bounds] = 0.0
            # Reshape back to (n_windows, max_samples) and assign
            data[:, i, :] = interp_vals.reshape(self.window_manager.n_windows, self.max_samples_per_window)

        self.data = torch.tensor(data, device=self.data_device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage."""
        if self.window_manager is None:
            return None

        window_starts = self.window_manager.window_times
        window_ends = window_starts + self.window_manager.window_size

        # Find index in time_vector
        # Check if window actually contains data points
        start_inds = np.searchsorted(self.time_vector, window_starts, side='left')
        end_inds = np.searchsorted(self.time_vector, window_ends, side='left')

        # Window is valid if it contains at least one data point
        # Or we can be stricter.
        # Original logic was: "windows_with_spikes".
        # For continuous, we want windows that overlap with the time series.
        # But if there are gaps (jumps), start_inds and end_inds will point to the same index (if gap is larger than window).
        actual_counts = end_inds - start_inds
        # Minimum acceptable count based on coverage fraction
        min_count = int(np.ceil(self.min_coverage_fraction * self.max_samples_per_window))
        min_count = max(min_count, 1)  # always require at least 1 point

        valid = actual_counts >= min_count

        # Also enforce strict boundary containment if desired, but "data points check" handles gaps correctly.

        return valid

    def get_temporal_extent(self):
        return self.time_vector[0], self.time_vector[-1]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def reset(self):
        self.data = self.data_master.detach().clone()
        # Invalidate mask cache so apply_noise/apply_precision recompute after reset
        self.__dict__.pop('_data_mask', None)
        self.__dict__.pop('_noise_buffer', None)

    def time_shift(self, offset):
        self.time_vector = self.time_vector + offset - self.time_offset
        self.time_offset = offset

    def apply_noise(self, amplitude):
        if amplitude == 0.0:
            self.reset()
            return
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data, as_tuple=True)
        if not hasattr(self, '_noise_buffer') or len(self._noise_buffer) != len(self._data_mask[0]):
            self._noise_buffer = torch.empty(len(self._data_mask[0]), device=self.data.device, dtype=self.data.dtype)
        self._noise_buffer.normal_(mean=0, std=amplitude)
        self.data[self._data_mask] = self.data_master[self._data_mask] + self._noise_buffer

    def apply_precision(self, precision_level):
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
                 no_spike_value=-1.0, device=None,
                 exclude_bursty_neurons: bool = False,
                 burst_threshold_multiplier: float = 5.0,
                 data_device='cpu'):

        super().__init__(window_manager, device, data_device)

        self.data_orig = [np.array(st) for st in spike_times]
        # Ensure spike times are sorted within each neuron (safe: copy already made via np.array above)
        for i, st in enumerate(self.data_orig):
            if len(st) > 1 and not np.all(st[:-1] <= st[1:]):
                logger.debug(
                    f"SpikeWindowDataset: neuron {i} spike times are not sorted. "
                    "Sorting automatically."
                )
                self.data_orig[i] = np.sort(st)
        self.no_spike_value = no_spike_value
        self.exclude_bursty_neurons = exclude_bursty_neurons
        self.burst_threshold_multiplier = burst_threshold_multiplier
        self.time_offset = 0
        self._excluded_neurons = []  # track which were excluded

        # Process data if window manager is available
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None
    
    def _compute_max_samples_per_window(self):
        per_neuron_max = np.array([
            max_events_in_window(x, self.window_manager.window_size)
            for x in self.data_orig
        ])
        median_max = np.median(per_neuron_max)

        if self.exclude_bursty_neurons and median_max > 0:
            threshold = self.burst_threshold_multiplier * median_max
            bursty = np.where(per_neuron_max > threshold)[0]
            if len(bursty) > 0:
                self._excluded_neurons = bursty.tolist()
                logger.warning(
                    f"Excluding {len(bursty)} bursty neuron(s) "
                    f"(indices: {bursty.tolist()}) whose peak spike count "
                    f"exceeds {self.burst_threshold_multiplier}x the median "
                    f"({median_max:.1f}). Remaining neurons: "
                    f"{len(self.data_orig) - len(bursty)}."
                )
                # Filter data_orig in-place for this instance
                self.data_orig = [st for i, st in enumerate(self.data_orig)
                                if i not in set(bursty)]
                per_neuron_max = np.delete(per_neuron_max, bursty)

        self.max_samples_per_window = int(np.max(per_neuron_max)) if len(per_neuron_max) > 0 else 1

        # Warning (keep existing one, but now on the post-exclusion array)
        if median_max > 0 and self.max_samples_per_window > self.burst_threshold_multiplier * median_max:
            worst_neuron = int(np.argmax(per_neuron_max))
            logger.warning(
                f"Spike tensor allocation dominated by burst: neuron {worst_neuron} "
                f"has peak {self.max_samples_per_window} spikes/window vs. median "
                f"{median_max:.1f}. Consider exclude_bursty_neurons=True."
            )

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
        # Convert to tensor on data_device. Keep a master clone for noise/precision ops.
        self.data = torch.tensor(data, device=self.data_device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage. Assumes window manager attached"""
        windows_with_spikes = np.unique(np.concatenate(self._cached_window_inds))
        valid = np.full(self.window_manager.window_times.shape, False, bool)
        valid[windows_with_spikes] = True
        return valid
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def reset(self):
        self.data = self.data_master.detach().clone()
        # Invalidate mask cache — data shape may have changed or noise was cleared
        self.__dict__.pop('_data_mask', None)
        self.__dict__.pop('_noise_buffer', None)

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
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data != self.no_spike_value, as_tuple=True)
        if not hasattr(self, '_noise_buffer') or len(self._noise_buffer) != len(self._data_mask[0]):
            self._noise_buffer = torch.empty(len(self._data_mask[0]), device=self.data.device, dtype=self.data.dtype)
        self._noise_buffer.uniform_(-amplitude / 2, amplitude / 2)
        self.data[self._data_mask] = self.data_master[self._data_mask] + self._noise_buffer

    def apply_precision(self, precision_level):
        """Round spike times to a specific resolution/precision level."""
        # Reset to master copy if zero. Avoids divide by zero, useful as interface to undo changes
        if precision_level == 0.0:
            self.reset()
            return
        # If data mask hasn't been created, compute that now
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data != self.no_spike_value, as_tuple=True)
        # Always round from data_master so repeated calls at different precision
        # levels each start from the original spike times (not re-rounded values).
        self.data[self._data_mask] = torch.round(self.data_master[self._data_mask] / precision_level) * precision_level


class BinnedSpikeDataset(TemporalWindowDataset):
    """Spike train dataset using binned firing rates instead of raw spike times.

    Parameters
    ----------
    spike_times : list of np.ndarray
        One array of spike times per neuron, in seconds relative to session start.
    bin_size : float
        Width of each firing rate bin in seconds. Smaller bins preserve more
        temporal precision at the cost of a higher-dimensional input to the critic.
    window_manager : WindowManager, optional
    device : str or torch.device, optional
    normalize : bool, optional
        If True, divide bin counts by bin_size to express as spikes/second.
        Default True. Set False to keep raw counts.
    """

    def __init__(self, spike_times, bin_size: float,
                 window_manager=None, device=None, normalize: bool = True,
                 data_device='cpu'):
        super().__init__(window_manager, device, data_device)
        self.data_orig = [np.array(st) for st in spike_times]
        self.bin_size = bin_size
        self.normalize = normalize
        self.time_offset = 0
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None

    def _compute_max_samples_per_window(self):
        """Number of bins per window = ceil(window_size / bin_size)."""
        self.max_samples_per_window = int(
            np.ceil(self.window_manager.window_size / self.bin_size)
        )

    def get_temporal_extent(self):
        valid = [st for st in self.data_orig if len(st) > 0]
        if not valid:
            return 0, 0
        return min(st[0] for st in valid), max(st[-1] for st in valid)

    def move_data_to_windows(self):
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")

        n_neurons = len(self.data_orig)
        n_windows = self.window_manager.n_windows
        n_bins = self.max_samples_per_window
        wt = self.window_manager.window_times  # (n_windows,)
        ws = self.window_manager.window_size

        data = np.zeros((n_windows, n_neurons, n_bins), dtype=np.float32)

        for i, spikes in enumerate(self.data_orig):
            if len(spikes) == 0:
                continue

            # Vectorized: find which window each spike belongs to
            # searchsorted gives index such that wt[idx-1] <= spike < wt[idx]
            # subtract 1 to get the window index (spikes before first window get -1)
            win_idx = np.searchsorted(wt, spikes, side='right') - 1

            # Keep only spikes that fall within a valid window
            in_range = (win_idx >= 0) & (win_idx < n_windows)
            spikes_in = spikes[in_range]
            win_idx_in = win_idx[in_range]

            if len(spikes_in) == 0:
                continue

            # Compute relative spike time within window, then bin index
            rel_times = spikes_in - wt[win_idx_in]
            bin_idx = np.floor(rel_times / ws * n_bins).astype(np.int32)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # guard against floating point at boundary

            # Accumulate counts using np.add.at (handles repeated indices correctly)
            np.add.at(data[:, i, :], (win_idx_in, bin_idx), 1.0)

            if self.normalize:
                data[:, i, :] /= self.bin_size

        self.data = torch.tensor(data, device=self.data_device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Mark a window valid if at least one neuron fired in it."""
        n_windows = self.window_manager.n_windows
        valid = np.zeros(n_windows, dtype=bool)
        wt = self.window_manager.window_times

        all_spikes = np.concatenate([st for st in self.data_orig if len(st) > 0]) \
            if any(len(st) > 0 for st in self.data_orig) else np.array([])

        if len(all_spikes) > 0:
            win_idx = np.searchsorted(wt, all_spikes, side='right') - 1
            in_range = (win_idx >= 0) & (win_idx < n_windows)
            valid[win_idx[in_range]] = True

        return valid
    
    def __getitem__(self, idx):
        return self.data[idx]

    def reset(self):
        self.data = self.data_master.detach().clone()

    def time_shift(self, offset):
        self.data_orig = [st + offset - self.time_offset for st in self.data_orig]
        self.time_offset = offset

    def apply_noise(self, amplitude):
        """Add Uniform noise to non-zero bins (active bins only)."""
        if amplitude == 0.0:
            self.reset()
            return
        noise = torch.empty_like(self.data).uniform_(-amplitude / 2, amplitude / 2)
        # Only perturb bins that actually had spikes
        active = self.data_master > 0
        self.data = self.data_master.clone()
        self.data[active] = (self.data_master[active] + noise[active]).clamp(min=0)
  
    def apply_precision(self, precision_level):
        """Round bin values to the nearest multiple of precision_level."""
        if precision_level == 0.0:
            self.reset()
            return
        self.data = torch.round(self.data_master / precision_level) * precision_level


class CategoricalWindowDataset(TemporalWindowDataset):
    """Dataset for categorical time series data with one-hot encoding."""
    
    def __init__(self, data, time_vector=None, window_manager=None, device=None,
                 min_coverage_fraction=0.2, encoding: str = 'majority_vote',
                 data_device='cpu'):
        """
        Parameters
        ----------
        data : array-like
            Categorical data of shape (n_timepoints, n_channels) with integer category labels.
        time_vector : array-like, optional
            Time stamps for each sample.
            If None, assumes data sampled on positive integers [0, 1, 2, ...].
        window_manager : WindowManager, optional
            External window manager for alignment.
        device : str, optional
            Compute device (kept for reference; not used for data storage).
        min_coverage_fraction : float, optional
            Minimum fraction of a window covered by actual data for it to be
            considered valid.
        encoding : str, optional
            Method for encoding categorical data into windows.
        data_device : str, optional
            Device for storing ``self.data``.  Defaults to ``'cpu'``.
        """
        super().__init__(window_manager, device, data_device)

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
            self.time_vector = np.arange(0, self.data_orig.shape[0])
        # Infer sample rate from time vector
        # This class assumes, outside of jumps, a constant sample rate!
        diffs = np.diff(self.time_vector)
        self.period = float(np.median(diffs)) if len(diffs) > 0 else 1.0

        if len(self.time_vector) != self.data_orig.shape[0]:
            raise ValueError(
                f"time_vector must be same length as data. "
                f"Received {len(self.time_vector)} time points and {self.data_orig.shape[0]} data points"
            )
        
        # Total one-hot encoded dimensions
        self.encoding = encoding
        self.n_categories = self.data_orig.max() + 1
        # Warn if full_trajectory would produce a very wide tensor
        if encoding == 'full_trajectory':
            max_samples = int(np.ceil(
                (window_manager.window_size / self.period) if window_manager else 1
            ))
            enc_dim = self.n_categories * max_samples
            if enc_dim > 500:
                logger.warning(
                    f"CategoricalWindowDataset: encoding='full_trajectory' will produce "
                    f"a {enc_dim}-dimensional input per window "
                    f"({self.n_categories} categories × ~{max_samples} timesteps). "
                    f"Consider encoding='majority_vote' for large windows."
                )
        self.min_coverage_fraction = min_coverage_fraction

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
        Categorical data of shape (n_timepoints, n_channels) is reshaped and mapped to 
        (n_windows, n_channels, n_timepoints_in_window * n_categories)
        
        Requires an attached window manager.
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        logger.debug(f"CategoricalWindowDataset: using encoding='{self.encoding}'")
        
        if self.encoding == 'majority_vote':
            self._move_majority_vote()
        elif self.encoding == 'probability':
            self._move_probability()
        elif self.encoding == 'full_trajectory':
            self._move_full_trajectory()
        else:
            raise ValueError(
                f"Unknown encoding '{self.encoding}'. "
                f"Expected 'majority_vote', 'probability', or 'full_trajectory'."
            )
        self.data_master = self.data.detach().clone()

    def _move_majority_vote(self):
        """One-hot of the most common category per window. Shape: (n_windows, n_channels, n_categories)"""
        n_channels = self.data_orig.shape[1]
        mask = (self.time_vector >= self.window_manager.t_start) & \
            (self.time_vector <= self.window_manager.t_end)
        win_idx = np.searchsorted(self.window_manager.window_times,
                                self.time_vector[mask], side='right') - 1
        data_in = self.data_orig[mask]

        n_windows = self.window_manager.n_windows
        result = np.zeros((n_windows, n_channels, self.n_categories), dtype=np.float32)

        for w in range(n_windows):
            rows = data_in[win_idx == w]
            if len(rows) == 0:
                continue
            for c in range(n_channels):
                counts = np.bincount(rows[:, c], minlength=self.n_categories)
                result[w, c, np.argmax(counts)] = 1.0

        self._cached_window_inds = win_idx
        self.data = torch.tensor(result, device=self.data_device)

    def _move_probability(self):
        """Empirical category probabilities per window. Shape: (n_windows, n_channels, n_categories)"""
        n_channels = self.data_orig.shape[1]
        mask = (self.time_vector >= self.window_manager.t_start) & \
            (self.time_vector <= self.window_manager.t_end)
        win_idx = np.searchsorted(self.window_manager.window_times,
                                self.time_vector[mask], side='right') - 1
        data_in = self.data_orig[mask]

        n_windows = self.window_manager.n_windows
        result = np.zeros((n_windows, n_channels, self.n_categories), dtype=np.float32)

        for w in range(n_windows):
            rows = data_in[win_idx == w]
            if len(rows) == 0:
                continue
            for c in range(n_channels):
                counts = np.bincount(rows[:, c], minlength=self.n_categories).astype(np.float32)
                total = counts.sum()
                if total > 0:
                    result[w, c, :] = counts / total

        self._cached_window_inds = win_idx
        self.data = torch.tensor(result, device=self.data_device)

    def _move_full_trajectory(self):
        """Original behavior: full one-hot trajectory. Shape: (n_windows, n_channels, n_categories * max_samples)"""
        # Preallocate output (one-hot encoded) (n_windows, window_size * n_categories, n_channels)
        n_channels = self.data_orig.shape[1]
        data_shape = (self.window_manager.n_windows, n_channels, self.n_categories * self.max_samples_per_window)
        data = np.zeros(data_shape, dtype=np.float32)
        
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
        for i in range(n_channels):
            data[expanded_window_inds, i, expanded_column_inds] = np.eye(self.n_categories)[self.data_orig[mask, i]].flatten()
        self.data = torch.tensor(data, device=self.data_device)
        # data_master is set by move_data_to_windows() after this returns.

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage."""
        if self.window_manager is None:
            return None
        max_samples = self.max_samples_per_window
        min_count = max(1, int(np.ceil(self.min_coverage_fraction * max_samples)))
        _, counts = np.unique(self._cached_window_inds, return_counts=True)
        # Build per-window count array (windows with no samples have count 0)
        window_counts = np.zeros(len(self.window_manager.window_times), dtype=int)
        unique_wins, win_counts = np.unique(self._cached_window_inds, return_counts=True)
        window_counts[unique_wins] = win_counts
        return window_counts >= min_count
    
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
        return self.data[idx]
    
    def reset(self):
        """Undo any added noise by resetting to original data. Does not undo time shifts."""
        self.data = self.data_master.detach().clone()

    def apply_noise(self, amplitude):
        """
        Noise corruption is not defined for categorical data.

        Raises
        ------
        NotImplementedError
            Always. Apply augmentation to the underlying signal before encoding.
        """
        raise NotImplementedError(
            "Noise corruption is not defined for categorical data. "
            "Apply augmentation to the underlying signal before encoding."
        )

    def apply_precision(self, precision_level):
        """
        Precision corruption is not defined for categorical data.

        Raises
        ------
        NotImplementedError
            Always. Apply augmentation to the underlying signal before encoding.
        """
        raise NotImplementedError(
            "Precision corruption is not defined for categorical data. "
            "Apply augmentation to the underlying signal before encoding."
        )


