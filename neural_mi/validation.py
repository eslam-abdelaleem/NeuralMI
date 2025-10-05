# neural_mi/validation.py

from typing import Dict, Any, Union, List
import numpy as np
import torch
from neural_mi.logger import logger
from neural_mi.exceptions import DataShapeError

class DataValidator:
    def __init__(self, x_data: Any, y_data: Any, processor_type: str):
        self.x_data, self.y_data, self.processor_type = x_data, y_data, processor_type

    def validate(self):
        self._validate_types(); self._validate_shapes(); self._validate_values(); self._validate_compatibility()

    def _validate_types(self):
        for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
            if data is None: continue
            if not isinstance(data, (np.ndarray, torch.Tensor, list)):
                raise TypeError(f"{name} must be np.ndarray, torch.Tensor, or list, got {type(data)}")
            
            if self.processor_type != 'spike':
                is_numeric = False
                if isinstance(data, np.ndarray):
                    is_numeric = np.issubdtype(data.dtype, np.number)
                elif isinstance(data, torch.Tensor):
                    is_numeric = data.is_floating_point() or data.is_complex() or \
                                 data.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
                
                if not is_numeric:
                    raise TypeError(f"{name} must contain numeric data, but found type {data.dtype}.")

    def _validate_shapes(self):
        if self.processor_type == 'continuous':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None: continue
                if not isinstance(data, (np.ndarray, torch.Tensor)): continue
                if data.ndim not in [2, 3]: raise DataShapeError(f"{name} must be 2D or 3D, got {data.ndim}D")
                if data.size == 0: raise ValueError(f"{name} is empty.")
        elif self.processor_type == 'spike':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None: continue
                if not isinstance(data, list) or len(data) == 0:
                    raise TypeError(f"{name} must be a non-empty list of arrays for spike data.")
                for i, spikes in enumerate(data):
                    if not isinstance(spikes, np.ndarray) or spikes.ndim != 1:
                        raise TypeError(f"{name}[{i}] must be a 1D np.ndarray.")

    def _validate_values(self):
        if self.processor_type == 'spike':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None: continue
                for i, spikes in enumerate(data):
                    if len(spikes) > 0 and np.any(spikes < 0):
                        raise ValueError(f"{name}[{i}] contains negative spike times.")
                    if len(spikes) > 1 and not np.all(spikes[:-1] <= spikes[1:]):
                        logger.warning(f"{name}[{i}] not sorted. Sorting automatically.")
                        data[i] = np.sort(spikes)
        else: # continuous
             for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None: continue
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    d_np = data.numpy() if isinstance(data, torch.Tensor) else data
                    if np.any(np.isnan(d_np)): raise ValueError(f"{name} contains NaN values.")
                    if np.any(np.isinf(d_np)): raise ValueError(f"{name} contains Inf values.")
    
    def _validate_compatibility(self):
        if self.y_data is None: return
        
        is_x_list = isinstance(self.x_data, list)
        is_y_list = isinstance(self.y_data, list)
        
        if self.processor_type == 'spike':
            if is_x_list and is_y_list and len(self.x_data) != len(self.y_data):
                logger.warning(f"x_data has {len(self.x_data)} channels, y_data has {len(self.y_data)}.")
        elif not is_x_list and not is_y_list: # For continuous data
            if self.x_data.ndim == 3 and self.y_data.ndim == 3 and self.x_data.shape[0] != self.y_data.shape[0]:
                raise DataShapeError(f"Pre-processed data must have same number of samples, but got {self.x_data.shape[0]} and {self.y_data.shape[0]}.")

class ParameterValidator:
    def __init__(self, params: Dict[str, Any]):
        self.params, self.mode = params, params.get("mode")
    def validate(self):
        self._validate_required(); self._validate_base(); self._validate_processor(); self._validate_sweep()
    def _validate_required(self):
        if self.params.get("base_params") is None: raise ValueError("'base_params' is required.")
    def _validate_base(self):
        bp = self.params["base_params"]
        if not isinstance(bp, dict): raise TypeError("'base_params' must be a dictionary.")
        checks = {"n_epochs": (int, 1), "learning_rate": (float, 0), "batch_size": (int, 1), 
                  "patience": (int, 0), "embedding_dim": (int, 1), "hidden_dim": (int, 1), "n_layers": (int, 0)}
        for key, (dtype, min_val) in checks.items():
            if key in bp:
                if not isinstance(bp[key], dtype): raise TypeError(f"'{key}' must be {dtype.__name__}.")
                if bp[key] < min_val: raise ValueError(f"'{key}' must be at least {min_val}.")
    def _validate_processor(self):
        if self.params.get("processor_type") and self.params.get("processor_params") is None:
            raise ValueError("'processor_params' required when 'processor_type' is specified.")
    def _validate_sweep(self):
        if self.mode in ["sweep", "dimensionality"] and self.params.get("sweep_grid") is None:
            raise ValueError(f"'sweep_grid' required for mode='{self.mode}'.")
        if self.mode == "dimensionality" and "embedding_dim" not in self.params["sweep_grid"]:
            raise ValueError("'sweep_grid' must contain 'embedding_dim' for mode='dimensionality'.")