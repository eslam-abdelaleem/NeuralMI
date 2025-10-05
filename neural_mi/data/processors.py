# neural_mi/data/processors.py

import numpy as np
import torch
from typing import List, Optional
from neural_mi.logger import logger
from neural_mi.exceptions import InsufficientDataError, DataShapeError

class BaseProcessor:
    def process(self, raw_data: np.ndarray, **kwargs) -> torch.Tensor:
        raise NotImplementedError

def _find_max_events_in_window(event_times: np.ndarray, window_size: float) -> int:
    if len(event_times) == 0: return 0
    max_count = 0; left = 0
    for right in range(len(event_times)):
        while event_times[right] - event_times[left] > window_size:
            left += 1
        max_count = max(max_count, right - left + 1)
    return max_count

class SpikeProcessor(BaseProcessor):
    def __init__(self, window_size: float = 0.1, step_size: float = 0.01, 
                 n_seconds: Optional[float] = None, max_spikes_per_window: Optional[int] = None):
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive.")
        self.window_size, self.step_size = window_size, step_size
        self.n_seconds, self.max_spikes = n_seconds, max_spikes_per_window
        self._is_fitted = self.max_spikes is not None

    def process(self, raw_data: List[np.ndarray], t_start: Optional[float] = None, 
                t_end: Optional[float] = None) -> torch.Tensor:
        if all(len(ch) == 0 for ch in raw_data):
            # Return a tensor of the correct shape but with zero windows
            if not self._is_fitted: self.max_spikes = 1
            return torch.zeros((0, len(raw_data), self.max_spikes))

        # *** FIX: The processor should trust the t_start and t_end it is given. ***
        # The DataHandler is responsible for providing sensible, synchronized values.
        t_start_eff = t_start if t_start is not None else min([ch[0] for ch in raw_data if len(ch) > 0], default=0)
        
        if t_end is not None:
            t_end_eff = t_end
        elif self.n_seconds is not None:
            t_end_eff = t_start_eff + self.n_seconds
        else:
            t_end_eff = max([ch[-1] for ch in raw_data if len(ch) > 0], default=t_start_eff)
        
        if t_end_eff - t_start_eff < self.window_size:
            # If the range is too small, return an empty tensor of the correct shape
            if not self._is_fitted: self.max_spikes = 1
            return torch.zeros((0, len(raw_data), self.max_spikes))

        window_starts = np.arange(t_start_eff, t_end_eff - self.window_size + self.step_size, self.step_size)
        if not self._is_fitted:
            logger.debug("`max_spikes_per_window` not provided. Calculating from data...")
            self.max_spikes = max([_find_max_events_in_window(ch, self.window_size) for ch in raw_data]) or 1
            self._is_fitted = True
            logger.debug(f"Set `max_spikes_per_window` to {self.max_spikes}.")

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
    def __init__(self, window_size: int = 1, step_size: int = 1):
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive integers.")
        self.window_size, self.step_size = window_size, step_size

    def process(self, raw_data: np.ndarray) -> torch.Tensor:
        if raw_data.ndim != 2:
            raise DataShapeError("`raw_data` for ContinuousProcessor must be 2D.")
        
        n_channels, n_samples = raw_data.shape
        if n_samples < self.window_size:
            raise InsufficientDataError(f"Data length ({n_samples}) < window size ({self.window_size}).")
            
        starts = np.arange(0, n_samples - self.window_size + 1, self.step_size)
        shape = (len(starts), n_channels, self.window_size)
        strides = (self.step_size * raw_data.strides[1], raw_data.strides[0], raw_data.strides[1])
        return torch.from_numpy(np.lib.stride_tricks.as_strided(raw_data, shape=shape, strides=strides).copy()).float()