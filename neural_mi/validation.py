# neural_mi/validation.py
"""Provides classes for validating input data and parameters.

This module contains validators that are used by the main `run` function to
ensure that the provided data and hyperparameters are valid and compatible
before starting a potentially long-running analysis.
"""
from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch
from neural_mi.logger import logger
from neural_mi.exceptions import DataShapeError

class DataValidator:
    """Validates the input data for type, shape, and content for two potentially different streams."""
    def __init__(self, x_data: Any, y_data: Any, processor_type_x: Optional[str], processor_type_y: Optional[str]):
        """
        Parameters
        ----------
        x_data : Any
            The data for variable X.
        y_data : Any
            The data for variable Y.
        processor_type_x : str, optional
            The processor type for X ('continuous', 'spike', 'categorical').
        processor_type_y : str, optional
            The processor type for Y ('continuous', 'spike', 'categorical').
        """
        self.x_data, self.y_data = x_data, y_data
        self.proc_type_x, self.proc_type_y = processor_type_x, processor_type_y

    def validate(self):
        """Runs all validation checks in sequence for both data streams."""
        self._validate_stream(self.x_data, 'x_data', self.proc_type_x)
        self._validate_stream(self.y_data, 'y_data', self.proc_type_y)
        self._validate_compatibility()

    def _validate_stream(self, data: Any, name: str, proc_type: Optional[str]):
        """Runs a full validation suite on a single data stream."""
        if data is None: return
        self._validate_type(data, name, proc_type)
        self._validate_shape(data, name, proc_type)
        self._validate_values(data, name, proc_type)

    def _validate_type(self, data: Any, name: str, proc_type: Optional[str]):
        """Validates the base type and dtype of a data stream."""
        if not isinstance(data, (np.ndarray, torch.Tensor, list)):
            raise TypeError(f"{name} must be np.ndarray, torch.Tensor, or list, got {type(data)}")

        if proc_type in ['continuous', 'categorical']:
            is_numeric = False
            if isinstance(data, np.ndarray):
                is_numeric = np.issubdtype(data.dtype, np.number)
            elif isinstance(data, torch.Tensor):
                is_numeric = data.is_floating_point() or data.is_complex() or \
                             data.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
            
            if not is_numeric:
                raise TypeError(f"{name} must contain numeric data, but found type {data.dtype}.")
            
            if proc_type == 'categorical' and isinstance(data, (np.ndarray, torch.Tensor)):
                d_np = data.numpy() if isinstance(data, torch.Tensor) else data
                if not np.issubdtype(d_np.dtype, np.integer):
                    raise TypeError(f"{name} for categorical processor must be integer type, but found {d_np.dtype}.")

    def _validate_shape(self, data: Any, name: str, proc_type: Optional[str]):
        """Validates the dimensions and size of a data stream."""
        if proc_type in ['continuous', 'categorical']:
            if not isinstance(data, (np.ndarray, torch.Tensor)): return
            if data.ndim not in [2, 3]:
                raise DataShapeError(
                    f"{name} must be a 2D array of shape (n_channels, n_timepoints) "
                    f"or a pre-processed 3D tensor, but got a {data.ndim}D array."
                )
            if data.size == 0: raise ValueError(f"{name} is empty.")
        elif proc_type == 'spike':
            if not isinstance(data, list) or len(data) == 0:
                raise TypeError(f"{name} must be a non-empty list of arrays for spike data.")
            for i, spikes in enumerate(data):
                if not isinstance(spikes, np.ndarray) or spikes.ndim != 1:
                    raise TypeError(f"{name}[{i}] must be a 1D np.ndarray.")

    def _validate_values(self, data: Any, name: str, proc_type: Optional[str]):
        """Validates the content of the data (e.g., for NaNs, sorting)."""
        if proc_type == 'spike':
            for i, spikes in enumerate(data):
                if len(spikes) > 0 and np.any(spikes < 0):
                    raise ValueError(f"{name}[{i}] contains negative spike times.")
                if len(spikes) > 1 and not np.all(spikes[:-1] <= spikes[1:]):
                    logger.warning(f"{name}[{i}] not sorted. Sorting automatically.")
                    data[i] = np.sort(spikes)
        elif proc_type in ['continuous', 'categorical']:
            if isinstance(data, (np.ndarray, torch.Tensor)):
                d_np = data.numpy() if isinstance(data, torch.Tensor) else data
                if np.any(np.isnan(d_np)): raise ValueError(f"{name} contains NaN values.")
                if np.any(np.isinf(d_np)): raise ValueError(f"{name} contains Inf values.")

    def _validate_compatibility(self):
        """Validates that the two data streams are compatible."""
        if self.y_data is None: return
        
        is_x_list = isinstance(self.x_data, list)
        is_y_list = isinstance(self.y_data, list)
        
        if self.proc_type_x == 'spike' and self.proc_type_y == 'spike':
            if is_x_list and is_y_list and len(self.x_data) != len(self.y_data):
                logger.warning(f"x_data has {len(self.x_data)} channels, y_data has {len(self.y_data)}.")
        elif not is_x_list and not is_y_list:
            if self.x_data.ndim == 3 and self.y_data.ndim == 3 and self.x_data.shape[0] != self.y_data.shape[0]:
                raise DataShapeError(f"Pre-processed data must have same number of samples, but got {self.x_data.shape[0]} and {self.y_data.shape[0]}.")

class ParameterValidator:
    """Validates the hyperparameter dictionary provided to the `run` function."""
    def __init__(self, params: Dict[str, Any]):
        """
        Parameters
        ----------
        params : Dict[str, Any]
            A dictionary of parameters, typically generated from `locals()`
            in the `run` function's scope.
        """
        self.params, self.mode = params, params.get("mode")

    def validate(self):
        """Runs all parameter validation checks in sequence."""
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
        # This check is now more complex due to the _x and _y parameters
        proc_x = self.params.get("processor_type_x")
        params_x = self.params.get("processor_params_x")
        proc_y = self.params.get("processor_type_y")
        params_y = self.params.get("processor_params_y")

        if proc_x and params_x is None:
            raise ValueError("'processor_params_x' required when 'processor_type_x' is specified.")
        if proc_y and params_y is None:
            raise ValueError("'processor_params_y' required when 'processor_type_y' is specified.")

    def _validate_sweep(self):
        if self.mode in ["sweep", "dimensionality"] and self.params.get("sweep_grid") is None:
            raise ValueError(f"'sweep_grid' required for mode='{self.mode}'.")
        if self.mode == "dimensionality" and "embedding_dim" not in self.params.get("sweep_grid", {}):
            raise ValueError("'sweep_grid' must contain 'embedding_dim' for mode='dimensionality'.")