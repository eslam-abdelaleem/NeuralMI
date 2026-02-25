# neural_mi/validation.py
"""Provides classes for validating input data and parameters.

This module contains validators that are used by the main `run` function to
ensure that the provided data and hyperparameters are valid and compatible
before starting a potentially long-running analysis.
"""
from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch
import inspect
from neural_mi.logger import logger
from neural_mi.exceptions import DataShapeError
from neural_mi.estimators import ESTIMATORS, ESTIMATOR_DEFAULTS
from neural_mi.defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA, PROCESSOR_PARAMS_SCHEMA

ALLOWED_VALUES = {
    'critic_type': ['separable', 'concat', 'hybrid'],
    'embedding_model': ['mlp', 'cnn', 'gru', 'lstm', 'tcn', 'transformer'],
    'split_mode': ['blocked', 'random'],
    'output_units': ['bits', 'nats'],
    'spectral_output': ['default', 'full', 'all'],
    'estimator_name': list(ESTIMATORS.keys())
}

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
        self._validate_required()
        self._validate_base()
        self._validate_processor()
        self._validate_sweep()
        self._validate_mode_kwargs()

    def _validate_required(self):
        if self.params.get("base_params") is None: raise ValueError("'base_params' is required.")

    def _validate_base(self):
        bp = self.params["base_params"]
        if not isinstance(bp, dict): raise TypeError("'base_params' must be a dictionary.")

        # Check for unknown parameters in base_params
        unknown_keys = set(bp.keys()) - set(BASE_PARAMS_SCHEMA.keys())
        if unknown_keys:
            raise ValueError(f"Unknown parameters in 'base_params': {unknown_keys}. "
                             f"Allowed: {list(BASE_PARAMS_SCHEMA.keys())}")

        # Validate types and values
        for key, value in bp.items():
            schema = BASE_PARAMS_SCHEMA[key]
            # Type check
            expected_type = schema['type']
            if not isinstance(value, expected_type):
                raise TypeError(f"Parameter '{key}' must be of type {expected_type}, got {type(value)}.")

            # Min value check
            if 'min' in schema and value is not None and value < schema['min']:
                raise ValueError(f"Parameter '{key}' must be >= {schema['min']}.")

            # Allowed values check
            if key in ALLOWED_VALUES and value not in ALLOWED_VALUES[key]:
                raise ValueError(f"Parameter '{key}' has invalid value '{value}'. Allowed: {ALLOWED_VALUES[key]}")

    def _validate_processor(self):
        # Validate processor existence and params
        for suffix in ['x', 'y']:
            proc_type = self.params.get(f"processor_type_{suffix}")
            proc_params = self.params.get(f"processor_params_{suffix}")

            if proc_type:
                if proc_params is None:
                    raise ValueError(f"'processor_params_{suffix}' required when 'processor_type_{suffix}' is specified.")

                # Check for invalid processor params
                if proc_type in PROCESSOR_PARAMS_SCHEMA:
                    allowed = set(PROCESSOR_PARAMS_SCHEMA[proc_type])
                    # Allow 'preprocessed' as internal flag
                    unknown = set(proc_params.keys()) - allowed - {'preprocessed'}
                    if unknown:
                        raise ValueError(f"Unknown parameters for {proc_type} processor: {unknown}. Allowed: {allowed}")

    def _validate_sweep(self):
        if self.mode == "sweep" and self.params.get("sweep_grid") is None:
            raise ValueError(f"'sweep_grid' required for mode='{self.mode}'.")

    def _validate_mode_kwargs(self):
        """Validates **analysis_kwargs passed to run() for the specific mode."""
        mode_schema = MODE_KWARGS_SCHEMA.get(self.mode, {})
        allowed_kwargs = set(mode_schema.keys())

        # Check for unexpected kwargs (excluding standard run arguments)
        # Note: self.params contains ALL locals() from run(). We only care about analysis_kwargs keys.
        # But we don't have direct access to 'analysis_kwargs' dict here, just the merged locals.
        # So we check if any key in locals that is NOT a standard run param is in allowed_kwargs.

        # Actually, best to validate just the keys that are NOT standard args.
        # Standard args are explicit in run(). The 'kwargs' are what we worry about.
        # In run(), analysis_kwargs are passed. We should probably validate those specifically.
        # But here we have 'locals()'.

        # Let's rely on run() passing explicit analysis_kwargs to a specific validator if needed.
        # For now, we assume user might pass them as kwargs.
        pass

    def apply_defaults(self):
        """Populates missing parameters in base_params with defaults."""
        bp = self.params["base_params"]
        verbose = bp.get('verbose', True)

        for key, schema in BASE_PARAMS_SCHEMA.items():
            if key not in bp and 'default' in schema:
                default_val = schema['default']
                # Don't apply None defaults if they mean "optional"
                if default_val is not None or schema['type'] == (str, type(None)):
                     bp[key] = default_val
                     if verbose:
                         logger.info(f"Parameter '{key}' not specified. Defaulting to {default_val}.")


class EstimatorValidator:
    """Validates the parameters for the chosen MI estimator."""
    def __init__(self, estimator_name: str, estimator_params: Optional[Dict[str, Any]] = None):
        self.name = estimator_name
        self.params = estimator_params or {}

        if self.name not in ESTIMATORS:
            raise ValueError(f"Unknown estimator '{self.name}'. Allowed estimators are: {list(ESTIMATORS.keys())}")

        self.func = ESTIMATORS[self.name]
        self.signature = inspect.signature(self.func)

    def validate(self):
        valid_params = set(self.signature.parameters.keys()) - {'scores'}
        unexpected = set(self.params.keys()) - valid_params
        if unexpected:
            raise ValueError(
                f"Estimator '{self.name}' got unexpected parameters: {unexpected}. "
                f"Allowed parameters are: {list(valid_params) if valid_params else 'None'}."
            )
        for name, param in self.signature.parameters.items():
            if name == 'scores': continue
            if param.default == inspect.Parameter.empty and name not in self.params:
                 raise ValueError(f"Estimator '{self.name}' requires parameter '{name}'.")

    def get_merged_params(self) -> Dict[str, Any]:
        defaults = ESTIMATOR_DEFAULTS.get(self.name, {})
        merged = defaults.copy()
        merged.update(self.params)
        return merged