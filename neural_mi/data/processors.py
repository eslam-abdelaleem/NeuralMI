# neural_mi/data/processors.py
"""Defines processors for converting raw data into trainable tensors.

This module contains classes for processing different types of raw data into
a windowed format suitable for neural network models. The standard output
format is a 3D tensor with dimensions (n_samples, n_channels, n_features).

Processors:
- `BaseProcessor`: An abstract base class for all data processors.
- `SpikeProcessor`: Processes lists of spike times into padded tensors of
  relative spike times within windows.
- `ContinuousProcessor`: Uses stride tricks to efficiently create sliding
  windows from continuous time-series data.
- `CategoricalProcessor`: One-hot encodes categorical data before creating
  sliding windows.
"""
import numpy as np
import torch
from typing import List, Optional, Union
from neural_mi.logger import logger
from neural_mi.exceptions import InsufficientDataError, DataShapeError

class BaseProcessor:
    """Abstract base class for data processors."""
    def process(self, raw_data: np.ndarray, **kwargs) -> torch.Tensor:
        """Processes raw data into a tensor.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError

def find_max_spikes_per_window(spike_times: List[np.ndarray], window_size: float) -> int:
    """Calculates the maximum number of spikes found in any window across all channels.

    This is a helper function to determine the `max_spikes_per_window` required
    by the `SpikeProcessor`.

    Parameters
    ----------
    spike_times : List[np.ndarray]
        A list of 1D numpy arrays, where each array contains the absolute
        spike times for a single channel.
    window_size : float
        The duration of the sliding window in seconds.

    Returns
    -------
    int
        The maximum number of spikes found in any single window. Returns 1 if
        no spikes are found to ensure a valid tensor shape.
    """
    if not spike_times or all(len(ch) == 0 for ch in spike_times):
        return 1
    
    max_overall = 0
    for channel_spikes in spike_times:
        if len(channel_spikes) == 0: continue
        max_in_channel = 0
        left = 0
        for right in range(len(channel_spikes)):
            while channel_spikes[right] - channel_spikes[left] > window_size:
                left += 1
            max_in_channel = max(max_in_channel, right - left + 1)
        max_overall = max(max_overall, max_in_channel)
        
    return max_overall or 1
    
class SpikeProcessor(BaseProcessor):
    """Processes spike train data into windowed, padded tensors.

    This processor takes a list of numpy arrays, where each array represents
    the spike times for a single channel. It creates sliding windows over time
    and, for each window, extracts the spike times relative to the window's start.

    Since the number of spikes can vary per window, the output tensor is padded
    up to `max_spikes_per_window`. This value can be provided or automatically
    inferred from the data on the first call to `process`.
    """
    def __init__(self, window_size: float, time_shift: float, 
                max_spikes_per_window: Optional[int] = None,
                n_seconds: Optional[float] = None):
        """
        Parameters
        ----------
        window_size : float
            The duration of each sliding window in seconds.
        step_size : float
            The step size (in seconds) to move the window forward.
        max_spikes_per_window : int
            The maximum number of spikes to store per window. This determines
            the size of the last dimension of the output tensor. If a window
            contains more spikes, they are truncated.
        n_seconds : float, optional
            The total duration of the data to process in seconds.
            ...
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if max_spikes_per_window is not None and max_spikes_per_window <= 0:
            raise ValueError("max_spikes_per_window must be positive if provided.")
        self.window_size = window_size
        self.time_shift = time_shift
        self.max_spikes = max_spikes_per_window
        self.n_seconds = n_seconds

    def process(self, raw_data: List[np.ndarray], 
                t_start: Optional[float] = None, 
                t_end: Optional[float] = None) -> torch.Tensor:
        """Converts a list of spike time arrays into a windowed tensor.

        Parameters
        ----------
        raw_data : List[np.ndarray]
            A list of 1D numpy arrays, where each array contains the absolute
            spike times for a single channel.
        t_start : float, optional
            The absolute start time for creating windows. If None, it is inferred
            from the first spike time in the data. Defaults to None.
        t_end : float, optional
            The absolute end time for creating windows. If None, it is inferred
            from the last spike time or `n_seconds`. Defaults to None.

        Returns
        -------
        torch.Tensor
            A 3D tensor of shape `(n_windows, n_channels, max_spikes_per_window)`,
            containing spike times relative to the start of each window.
        """
        if all(len(ch) == 0 for ch in raw_data):
            return torch.zeros((0, len(raw_data), self.max_spikes))
        
        # Compute max_spikes if not provided by user
        if self.max_spikes is None:
            self.max_spikes = find_max_spikes_per_window(raw_data, self.window_size)

        t_start_eff = t_start if t_start is not None else min([ch[0] for ch in raw_data if len(ch) > 0], default=0)
        
        if t_end is not None:
            t_end_eff = t_end
        elif self.n_seconds is not None:
            t_end_eff = t_start_eff + self.n_seconds
        else:
            t_end_eff = max([ch[-1] for ch in raw_data if len(ch) > 0], default=t_start_eff)
        
        if t_end_eff - t_start_eff < self.window_size:
            logger.warning(f"Recording is smaller than single window, returning zeros")
            return torch.zeros((0, len(raw_data), self.max_spikes))

        window_starts = np.arange(t_start_eff, t_end_eff, self.window_size)
        n_windows = len(window_starts) # Done this way in case we want to implement conditioning mutual information on spike activity here

        # Use searchsorted on each neuron/muscle to assign to chunks
        # This is by far the slowest part of this whole function!
        window_inds = [np.searchsorted(window_starts, x) - 1 for x in raw_data]

        # Preallocate (size is [window, neuron/muscle, spike time])
        output = np.full((n_windows, len(raw_data), self.max_spikes), self.no_spike_value, dtype=np.float32)
        for i_ch, channel_spikes in enumerate(raw_data):
            if len(channel_spikes) == 0: 
                continue
            # mask = self.valid_windows[window_inds_x[i]] # If we do "valid windows" for conditional MI
            _, _, counts = np.unique(window_inds[i_ch], return_counts=True, return_index=True)
            column_inds = np.concatenate(list(map(np.arange, counts)), axis=0)#[mask]
            # Xmain[window_inds_x[i][mask],i,column_inds] = self.Xtimes[i][mask] - self.window_times[window_inds_x[i][mask]]
            output[window_inds[i_ch],i_ch,column_inds] = output[i_ch] - window_starts[window_inds[i_ch]]
        return output


class ContinuousProcessor(BaseProcessor):
    """Processes continuous time-series data using efficient sliding windows.

    This processor converts a 2D numpy array of shape `(n_channels, n_timepoints)`
    into a 3D tensor of shape `(n_windows, n_channels, window_size)`
    Right now assumes continuous data is collected at a fixed sample rate
    """
    def __init__(self, 
            window_size: Optional[int] = 1, 
            time_shift: Optional[float] = 0.0,
            fs: Optional[float] = 1.0,
            n_seconds: Optional[float] = None
        ):
        """
        Parameters
        ----------
        window_size : int, optional
            The size (number of timepoints) of each window. Defaults to 1.
        time_shift : float, optional
            The amount of time windows are shifted by
        fs : float, optional
            Sampling frequency in Hz, used if no time vector is provided. 
            Defaults to 1.0 (1 sample per unit of time, whatever that may be) if not specified
        n_seconds : float, optional
            The total duration of the data to process in seconds.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.time_shift = time_shift
        self.fs = fs
        self.n_seconds = n_seconds

    def process(self, 
            raw_data: np.ndarray,
            time_vec: Optional[Union[np.ndarray, torch.Tensor]] = None,
            t_start: Optional[float] = None, 
            t_end: Optional[float] = None
        ) -> torch.Tensor:
        """Converts a 2D time-series array into a 3D windowed tensor.

        Parameters
        ----------
        raw_data : np.ndarray
            A 2D numpy array of shape `(n_channels, n_timepoints)`.
        time_vec : np.ndarray or torch.Tensor, optional
            A 1D array of shape (n_timepoints,) which contains the times at which raw data was acquired
        t_start : float, optional
            The absolute start time for creating windows, in whatever units of time used. If None, it is inferred
            from the first time in `time_vec`. If `time_vec` not present, defaults to 0.
        t_end : float, optional
            The absolute end time for creating windows, in whatever units of time used. If None, it is inferred
            from `n_seconds`, the last time in `time_vec`, or the number of samples in `raw_data`, 
            with precedence in that order.

        Returns
        -------
        torch.Tensor
            A 3D tensor of shape `(n_windows, n_channels, window_size)`.

        Raises
        ------
        DataShapeError
            If `raw_data` is not a 2D array, or `time_vec` and `raw_data` do not match
        InsufficientDataError
            If the total number of timepoints in `raw_data` is less than the `window_size`.
        """
        if raw_data.ndim != 2:
            raise DataShapeError("`raw_data` for ContinuousProcessor must be 2D.")
        if raw_data.shape[1] != len(time_vec):
            raise DataShapeError("`raw_data` and `time_vec` must have matching number of samples/points for ContinuousProcessor.")
        
        n_channels, n_samples = raw_data.shape
        if n_samples < self.window_size:
            raise InsufficientDataError(f"Data length ({n_samples}) < window size ({self.window_size}).")
        
        if t_start is None:
            t_start = time_vec[0] if time_vec is not None else 0.0
        if t_end is None:
            if self.n_seconds is not None:
                t_end = t_start + self.n_seconds
            elif time_vec is not None:
                t_end = time_vec[-1]
            else:
                t_end = t_start + n_samples

        # Trim data outside t_start, t_end
        if time_vec is not None:
            keep_mask = np.logical_and(time_vec >= t_start, time_vec <= t_end)
            raw_data = raw_data[:, keep_mask]
        else:
            ind_start, ind_end = t_start * self.fs, t_end * self.fs
            raw_data = raw_data[:, ind_start:ind_end]
        
        window_start_times = np.arange(t_start, t_end, self.window_size / self.fs)
        n_windows = len(window_start_times)

        # No time_vec:
        # Will treat as single continuous block of data, sampled on integers, and use fast stride method
        # No other way to know if there are bouts/trials or gaps
        if time_vec is None:
            shape = (n_windows, n_channels, self.window_size)
            strides = (self.window_size * raw_data.strides[1], raw_data.strides[0], raw_data.strides[1])
            return torch.from_numpy(np.lib.stride_tricks.as_strided(raw_data, shape=shape, strides=strides).copy()).float()

        # Otherwise, assume more complicated timing (gaps, trials, uneven time sampling, etc)
        # This method is more robust, but requires more compute time
        output = np.full((n_windows, raw_data.shape[0], ), 0, dtype=np.float32)

        # Set up bouts/trials if present (gaps in data longer than 1 window)
        bout_start_inds = np.where(np.diff(time_vec) > self.window_size)
        if bout_start_inds.size == 0:
            bout_starts = np.array([time_vec[0]])
        else:
            bout_starts = np.concatenate((time_vec[0], time_vec[bout_start_inds + 1]))

        window_inds = np.searchsorted(window_start_times, time_vec, side='right') - 1
        _, index, counts = np.unique(window_inds, return_counts=True, return_index=True)
        column_inds = np.concatenate(list(map(np.arange, counts)), axis=0)#[mask]
        first_inds = np.repeat(index, counts)
        time_shifts = time_vec[first_inds] - window_start_times[window_inds][first_inds]
        for i in range(raw_data.shape[0]):
            output[window_inds[mask],i,column_inds] = np.interp(time_vec[mask] + time_shifts[mask], time_vec[mask], raw_data[i,mask])

class CategoricalProcessor(BaseProcessor):
    """Processes categorical time-series data via one-hot encoding and windowing.

    This processor converts a 2D numpy array of integer categories of shape
    `(n_channels, n_timepoints)` into a 3D tensor. It first one-hot encodes
    the data, and then applies an efficient sliding window. The final output
    for each sample is a flattened vector of the one-hot encoded window.
    """
    def __init__(self, window_size: int = 1, step_size: int = 1):
        """
        Parameters
        ----------
        window_size : int, optional
            The size (number of timepoints) of each window. Defaults to 1.
        step_size : int, optional
            The step size (number of timepoints) to move the window forward.
            Defaults to 1.
        """
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive integers.")
        self.window_size, self.step_size = window_size, step_size

    def process(self, raw_data: np.ndarray) -> torch.Tensor:
        """Converts a 2D categorical array into a 3D windowed tensor.

        Parameters
        ----------
        raw_data : np.ndarray
            A 2D numpy array of shape `(n_channels, n_timepoints)` containing
            integer category labels.

        Returns
        -------
        torch.Tensor
            A 3D tensor of shape `(n_windows, n_channels, window_size * n_categories)`.

        Raises
        ------
        DataShapeError
            If `raw_data` is not a 2D array.
        InsufficientDataError
            If the total number of timepoints in `raw_data` is less than
            the `window_size`.
        """
        if raw_data.ndim != 2:
            raise DataShapeError("`raw_data` for CategoricalProcessor must be 2D.")
        
        n_channels, n_samples = raw_data.shape
        if n_samples < self.window_size:
            raise InsufficientDataError(f"Data length ({n_samples}) < window size ({self.window_size}).")

        # First, create sliding windows of the integer data
        starts = np.arange(0, n_samples - self.window_size + 1, self.step_size)
        shape = (n_channels, len(starts), self.window_size)
        strides = (raw_data.strides[0], self.step_size * raw_data.strides[1], raw_data.strides[1])
        windowed_integers = np.lib.stride_tricks.as_strided(raw_data, shape=shape, strides=strides)

        # Now, one-hot encode the windowed integer data
        n_categories = raw_data.max() + 1
        one_hot = np.eye(n_categories)[windowed_integers] # Shape: (n_channels, n_windows, window_size, n_categories)
        
        # Reshape to (n_windows, n_channels, window_size * n_categories)
        output = one_hot.transpose(1, 0, 2, 3).reshape(len(starts), n_channels, -1)
        
        return torch.from_numpy(output.copy()).float()