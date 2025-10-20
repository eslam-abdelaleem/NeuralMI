# neural_mi/data/handler.py
"""Manages data preprocessing by dispatching to specific processors.

This module defines the `DataHandler` class, which serves as a unified
interface for preprocessing different types of data (e.g., continuous time
series, spike trains), including combinations of different data types.
"""
from typing import Optional, Union, Dict, Any, Literal, Tuple
import torch
import numpy as np

from .processors import ContinuousProcessor, SpikeProcessor, CategoricalProcessor, find_max_spikes_per_window
from neural_mi.logger import logger
from neural_mi.exceptions import DataShapeError

ProcessorType = Literal['continuous', 'spike', 'categorical']

class DataHandler:
    """A class to manage and dispatch data to the correct processor.

    The DataHandler takes raw data for two variables, X and Y, and, based on
    the specified `processor_type`, uses the appropriate processor (`ContinuousProcessor`
    or `SpikeProcessor`) to transform the data into a format suitable for
    training a neural network (i.e., 3D torch tensors).

    Attributes
    ----------
    x_data : Union[np.ndarray, torch.Tensor, list]
        The raw input data for variable X.
    y_data : Union[np.ndarray, torch.Tensor, list]
        The raw input data for variable Y.
    processor_type_x, processor_type_y: {'continuous', 'spike', 'categorical'}, optional
        The type of processor to use for X and Y. If None, the data is assumed to be
        already preprocessed.
    processor_params_x, processor_params_y : Dict[str, Any]
        A dictionary of parameters to pass to the selected processor's
        initializer for X and Y.
    """
    def __init__(self, x_data: Any, y_data: Any,
                 processor_type_x: Optional[ProcessorType] = None,
                 processor_params_x: Optional[Dict[str, Any]] = None,
                 processor_type_y: Optional[ProcessorType] = None,
                 processor_params_y: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        x_data : Union[np.ndarray, torch.Tensor, list]
            The raw input data for variable X.
        y_data : Union[np.ndarray, torch.Tensor, list]
            The raw input data for variable Y.
        processor_type_x, processor_type_y : {'continuous', 'spike', 'categorical'}, optional
            The type of processor to use. If `None`, the data is assumed to be
            already processed and will be returned as-is after a shape check.
            Defaults to None.
        processor_params_x, processor_params_y : Dict[str, Any], optional
            A dictionary of parameters to pass to the selected processor's
            initializer. Defaults to an empty dictionary.
        """
        self.x_data, self.y_data = x_data, y_data
        self.proc_type_x = processor_type_x
        self.proc_params_x = processor_params_x if processor_params_x is not None else {}
        
        # Smartly default Y parameters to X parameters if not provided
        self.proc_type_y = processor_type_y if processor_type_y is not None else self.proc_type_x
        self.proc_params_y = processor_params_y if processor_params_y is not None else self.proc_params_x.copy()

    def _get_processor(self, proc_type: ProcessorType, proc_params: Dict[str, Any], data: Any, other_data: Any) -> Any:
        """Instantiates and returns the correct processor."""
        other_data = other_data if other_data is not None else data
        if proc_type == 'spike':
            params = proc_params.copy()
            params.setdefault('window_size', 0.1)
            params.setdefault('step_size', 0.01)
            if 'max_spikes_per_window' not in params and data is not None:
                combined_data = list(data) + (list(other_data) if other_data is not None else [])
                max_spikes = find_max_spikes_per_window(combined_data, params['window_size'])
                params['max_spikes_per_window'] = max_spikes
            return SpikeProcessor(**params)
        elif proc_type == 'continuous':
            return ContinuousProcessor(**proc_params)
        elif proc_type == 'categorical':
            return CategoricalProcessor(**proc_params)
        raise ValueError(f"Unknown processor_type: '{proc_type}'")

    def _get_num_samples(self, data: Any, proc_type: ProcessorType, proc_params: Dict[str, Any], t_start=None, t_end=None) -> int:
        """Calculates the number of samples that will be generated."""
        if data is None or proc_type is None: return -1

        if proc_type in ['continuous', 'categorical']:
            if not hasattr(data, 'shape') or len(data.shape) < 2 or data.shape[1] == 0: return 0
            n_time = data.shape[1]
            win = proc_params.get('window_size', 1)
            step = proc_params.get('step_size', 1)
            if n_time < win: return 0
            return (n_time - win) // step + 1
        elif proc_type == 'spike':
            if not any(hasattr(ch, '__len__') and len(ch) > 0 for ch in data): return 0
            t_start = t_start if t_start is not None else min([ch[0] for ch in data if hasattr(ch, '__len__') and len(ch) > 0], default=0)
            t_end = t_end if t_end is not None else max([ch[-1] for ch in data if hasattr(ch, '__len__') and len(ch) > 0], default=t_start)
            win = proc_params.get('window_size', 0.1)
            step = proc_params.get('step_size', 0.01)
            if t_end - t_start < win: return 0
            return int((t_end - t_start - win) / step + 1.00001)
        return 0

    def process(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes the raw data and returns aligned 3D torch tensors."""
        if self.proc_type_x is None:
            is_valid_x = isinstance(self.x_data, torch.Tensor) and self.x_data.ndim == 3
            is_valid_y = self.y_data is None or (isinstance(self.y_data, torch.Tensor) and self.y_data.ndim == 3)
            if not is_valid_x or not is_valid_y:
                 raise DataShapeError("If processors are not specified, data must be pre-processed 3D tensors.")
            return self.x_data.float(), self.y_data.float() if self.y_data is not None else None

        def get_np_data(data, proc_type, params):
            if data is None: return None
            if proc_type == 'spike': return list(data)
            np_data = data.numpy() if isinstance(data, torch.Tensor) else np.array(data)
            data_format = params.get('data_format', 'channels_first')
            if proc_type in ['continuous', 'categorical'] and np_data.ndim == 2 and (data_format == 'channels_last' or np_data.shape[0] > np_data.shape[1]):
                logger.debug("Transposing data to (channels, time).")
                np_data = np_data.T
            if proc_type == 'categorical': np_data = np_data.astype(int)
            return np_data

        x_np = get_np_data(self.x_data, self.proc_type_x, self.proc_params_x)
        y_np = get_np_data(self.y_data, self.proc_type_y, self.proc_params_y)
        
        proc_x = self._get_processor(self.proc_type_x, self.proc_params_x, x_np, y_np)
        proc_y = self._get_processor(self.proc_type_y, self.proc_params_y, y_np, x_np) if y_np is not None else None
        
        # --- Alignment Logic ---
        t_start, t_end = None, None
        if self.proc_type_x == 'spike' or self.proc_type_y == 'spike':
            all_spikes = (x_np if self.proc_type_x == 'spike' else []) + (y_np if self.proc_type_y == 'spike' and y_np is not None else [])
            t_start = min([ch[0] for ch in all_spikes if hasattr(ch, '__len__') and len(ch) > 0], default=0)
            t_end = max([ch[-1] for ch in all_spikes if hasattr(ch, '__len__') and len(ch) > 0], default=t_start)

        n_samples_x = self._get_num_samples(x_np, self.proc_type_x, self.proc_params_x, t_start, t_end)
        n_samples_y = self._get_num_samples(y_np, self.proc_type_y, self.proc_params_y, t_start, t_end)

        n_samples = n_samples_x if n_samples_y == -1 else min(n_samples_x, n_samples_y)

        if self.proc_type_x in ['continuous', 'categorical']:
            win_x = self.proc_params_x.get('window_size', 1)
            step_x = self.proc_params_x.get('step_size', 1)
            required_len_x = (n_samples - 1) * step_x + win_x
            x_np = x_np[:, :required_len_x]

        if y_np is not None and self.proc_type_y in ['continuous', 'categorical']:
            win_y = self.proc_params_y.get('window_size', 1)
            step_y = self.proc_params_y.get('step_size', 1)
            required_len_y = (n_samples - 1) * step_y + win_y
            y_np = y_np[:, :required_len_y]

        x_processed = proc_x.process(x_np, t_start=t_start, t_end=t_end) if self.proc_type_x == 'spike' else proc_x.process(x_np)
        y_processed = (proc_y.process(y_np, t_start=t_start, t_end=t_end) if self.proc_type_y == 'spike' else proc_y.process(y_np)) if proc_y is not None else None

        x_tensor = torch.as_tensor(x_processed, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_processed, dtype=torch.float32) if y_processed is not None else None
        
        if y_tensor is not None and x_tensor.shape[0] != y_tensor.shape[0]:
             logger.warning(f"Post-trimming mismatch: X has {x_tensor.shape[0]}, Y has {y_tensor.shape[0]}. Final truncation.")
             min_len = min(x_tensor.shape[0], y_tensor.shape[0])
             x_tensor, y_tensor = x_tensor[:min_len], y_tensor[:min_len]

        return x_tensor, y_tensor