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
from typing import List, Optional
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
    def __init__(self, window_size: float, step_size: float, max_spikes_per_window: int,
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
        if window_size <= 0 or step_size <= 0 or max_spikes_per_window <= 0:
            raise ValueError("window_size, step_size, and max_spikes_per_window must be positive.")
        self.window_size = window_size
        self.step_size = step_size
        self.max_spikes = max_spikes_per_window
        self.n_seconds = n_seconds

    def process(self, raw_data: List[np.ndarray], t_start: Optional[float] = None, 
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

        t_start_eff = t_start if t_start is not None else min([ch[0] for ch in raw_data if len(ch) > 0], default=0)
        
        if t_end is not None:
            t_end_eff = t_end
        elif self.n_seconds is not None:
            t_end_eff = t_start_eff + self.n_seconds
        else:
            t_end_eff = max([ch[-1] for ch in raw_data if len(ch) > 0], default=t_start_eff)
        
        if t_end_eff - t_start_eff < self.window_size:
            return torch.zeros((0, len(raw_data), self.max_spikes))

        window_starts = np.arange(t_start_eff, t_end_eff - self.window_size + self.step_size, self.step_size)
        
        output = torch.zeros((len(window_starts), len(raw_data), self.max_spikes))
        for i_ch, channel_spikes in enumerate(raw_data):
            if len(channel_spikes) == 0: continue
            for i_win, win_start in enumerate(window_starts):
                spikes = channel_spikes[(channel_spikes >= win_start) & (channel_spikes < win_start + self.window_size)]
                if len(spikes) > 0:
                    n = min(len(spikes), self.max_spikes)
                    output[i_win, i_ch, :n] = torch.from_numpy(spikes[:n] - win_start)
        return output

class ContinuousProcessor(BaseProcessor):
    """Processes continuous time-series data using efficient sliding windows.

    This processor converts a 2D numpy array of shape `(n_channels, n_timepoints)`
    into a 3D tensor of shape `(n_windows, n_channels, window_size)` using
    `numpy.lib.stride_tricks.as_strided`. This is a highly memory-efficient
    way to create overlapping windows without duplicating data.
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
        """Converts a 2D time-series array into a 3D windowed tensor.

        Parameters
        ----------
        raw_data : np.ndarray
            A 2D numpy array of shape `(n_channels, n_timepoints)`.

        Returns
        -------
        torch.Tensor
            A 3D tensor of shape `(n_windows, n_channels, window_size)`.

        Raises
        ------
        DataShapeError
            If `raw_data` is not a 2D array.
        InsufficientDataError
            If the total number of timepoints in `raw_data` is less than
            the `window_size`.
        """
        if raw_data.ndim != 2:
            raise DataShapeError("`raw_data` for ContinuousProcessor must be 2D.")
        
        n_channels, n_samples = raw_data.shape
        if n_samples < self.window_size:
            raise InsufficientDataError(f"Data length ({n_samples}) < window size ({self.window_size}).")
            
        starts = np.arange(0, n_samples - self.window_size + 1, self.step_size)
        shape = (len(starts), n_channels, self.window_size)
        strides = (self.step_size * raw_data.strides[1], raw_data.strides[0], raw_data.strides[1])
        return torch.from_numpy(np.lib.stride_tricks.as_strided(raw_data, shape=shape, strides=strides).copy()).float()


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