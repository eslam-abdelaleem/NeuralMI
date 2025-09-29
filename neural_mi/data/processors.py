# neural_mi/data/processors.py

import numpy as np
import torch
from typing import List

class BaseProcessor:
    """Abstract base class for data processors."""
    def __init__(self, **kwargs):
        pass

    def process(self, raw_data, **kwargs):
        """Process raw data into windowed tensors."""
        raise NotImplementedError

# --- Spike Data Processor ---

def _find_max_events_in_window(event_times: np.ndarray, window_size: float) -> int:
    """Helper function to find the max number of events in any possible window."""
    if len(event_times) == 0:
        return 0
    max_count = 0; left = 0
    for right in range(len(event_times)):
        while event_times[right] - event_times[left] > window_size:
            left += 1
        max_count = max(max_count, right - left + 1)
    return max_count

class SpikeProcessor(BaseProcessor):
    """Processes raw spike time data into windowed tensors."""
    def __init__(self, window_size: float, step_size: float, max_spikes_per_window: int = None):
        super().__init__()
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive.")
        self.window_size = window_size
        self.step_size = step_size
        self.max_spikes = max_spikes_per_window
        self._is_fitted = self.max_spikes is not None

    def process(
        self,
        raw_data: List[np.ndarray],
        t_start: float = None,
        t_end: float = None,
        channels_to_group: List[int] = None
    ) -> torch.Tensor:
        """Transforms a list of spike time arrays into windowed tensors."""
        # --- 1. Determine time bounds from GLOBAL data before selection ---
        global_first_spike = min([ch[0] for ch in raw_data if len(ch) > 0], default=0)
        global_last_spike = max([ch[-1] for ch in raw_data if len(ch) > 0], default=self.window_size)
        
        t_start = t_start if t_start is not None else global_first_spike
        t_end = t_end if t_end is not None else global_last_spike
        
        # CORRECTED window start calculation to be inclusive of the last possible window
        window_starts = np.arange(t_start, t_end - self.window_size + self.step_size, self.step_size)
        num_windows = len(window_starts)

        # --- 2. Select channels AFTER defining the time grid ---
        selected_data = [raw_data[i] for i in channels_to_group] if channels_to_group else raw_data
        num_channels = len(selected_data)

        # --- 3. Fit max_spikes if not already done ---
        if not self._is_fitted:
            print("`max_spikes_per_window` not provided. Calculating from data...")
            max_spikes_list = [_find_max_events_in_window(ch, self.window_size) for ch in selected_data]
            self.max_spikes = max(max_spikes_list) if max_spikes_list else 1
            self._is_fitted = True
            print(f"Set `max_spikes_per_window` to {self.max_spikes}.")

        # --- 4. Populate the output tensor ---
        output_tensor = torch.zeros((num_windows, num_channels, self.max_spikes))
        for i_ch, channel_spikes in enumerate(selected_data):
            if len(channel_spikes) == 0: continue
            spike_indices = np.searchsorted(window_starts, channel_spikes, side='right') - 1
            for i_win in range(num_windows):
                win_spikes = channel_spikes[spike_indices == i_win]
                if len(win_spikes) > 0:
                    relative_spikes = win_spikes - window_starts[i_win]
                    n_spikes = min(len(relative_spikes), self.max_spikes)
                    output_tensor[i_win, i_ch, :n_spikes] = torch.from_numpy(relative_spikes[:n_spikes])
        return output_tensor

# --- Continuous Data Processor ---

class ContinuousProcessor(BaseProcessor):
    """Processes raw continuous time-series data into windowed tensors."""
    def __init__(self, window_size: int, step_size: int):
        super().__init__()
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive integers.")
        self.window_size = window_size
        self.step_size = step_size

    def process(
        self,
        raw_data: np.ndarray,
        channels_to_group: List[int] = None
    ) -> torch.Tensor:
        """Transforms a time-series array into windowed tensors."""
        if raw_data.ndim != 2:
            raise ValueError("`raw_data` for ContinuousProcessor must be a 2D array of shape [num_channels, num_timepoints].")

        selected_data = raw_data[channels_to_group, :] if channels_to_group else raw_data
        n_channels, n_samples = selected_data.shape
        
        if n_samples < self.window_size:
            raise ValueError("Length of data is smaller than the window size.")

        # This calculation is the standard and correct one
        start_indices = np.arange(0, n_samples - self.window_size + 1, self.step_size)
        num_windows = len(start_indices)

        shape = (num_windows, n_channels, self.window_size)
        strides = (self.step_size * selected_data.strides[1], selected_data.strides[0], selected_data.strides[1])
        windowed_data = np.lib.stride_tricks.as_strided(selected_data, shape=shape, strides=strides)
        
        return torch.from_numpy(windowed_data.copy()).float()