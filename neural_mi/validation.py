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
    'embedding_model': ['mlp', 'cnn', 'cnn2d', 'gru', 'lstm', 'tcn', 'transformer',
                        'sinc_cnn', 'spike_physics', 'pretrained_backbone'],
    'split_mode': ['blocked', 'random'],
    'output_units': ['bits', 'nats'],
    'spectral_mode': ['none', 'summary', 'full'],
    'spectral_output': ['default', 'full', 'all'],
    'estimator_name': list(ESTIMATORS.keys()),  # 'infonce', 'smile'
    'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
    'scheduler': [None, 'cosine', 'step', 'plateau', 'cosine_warmup'],
    'norm_layer': [None, 'batch', 'layer'],
    'feature_fusion': ['features', 'concat'],
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
                    logger.warning(
                        f"{name}[{i}] spike times are not sorted; "
                        "they will be sorted automatically by SpikeWindowDataset."
                    )
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

            # Min value check (skip for non-scalar types like list/dict)
            if 'min' in schema and value is not None and not isinstance(value, (list, dict)) and value < schema['min']:
                raise ValueError(f"Parameter '{key}' must be >= {schema['min']}.")

            # Allowed values check. If a parameter accepts a class (e.g. a custom
            # optimizer type), skip the string-based lookup.
            if key in ALLOWED_VALUES and not isinstance(value, type) and value not in ALLOWED_VALUES[key]:
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

                # Validate numeric bounds for specific processor params
                ws = proc_params.get('window_size')
                if ws is not None:
                    if not isinstance(ws, (int, float)) or not np.isfinite(ws) or ws <= 0:
                        raise ValueError(
                            f"processor_params_{suffix}['window_size'] must be a positive number, "
                            f"got {ws!r}."
                        )
                sr = proc_params.get('sample_rate')
                if sr is not None:
                    if not isinstance(sr, (int, float)) or not np.isfinite(sr) or sr <= 0:
                        raise ValueError(
                            f"processor_params_{suffix}['sample_rate'] must be a positive number, "
                            f"got {sr!r}."
                        )
                ss = proc_params.get('step_size')
                if ss is not None:
                    if not isinstance(ss, (int, float)) or not np.isfinite(ss) or ss <= 0:
                        raise ValueError(
                            f"processor_params_{suffix}['step_size'] must be a positive number "
                            f"(fraction of window_size if < 1, absolute time units if >= 1), "
                            f"got {ss!r}."
                        )

    def _validate_sweep(self):
        if self.mode == "sweep" and self.params.get("sweep_grid") is None:
            raise ValueError(f"'sweep_grid' required for mode='{self.mode}'.")

    def _validate_mode_kwargs(self):
        """Validates **analysis_kwargs passed to run() for the specific mode."""
        mode_schema = MODE_KWARGS_SCHEMA.get(self.mode, {})
        if not mode_schema:
            return

        # Check required kwargs
        for key, schema in mode_schema.items():
            if schema.get('required', False) and key not in self.params:
                raise ValueError(
                    f"Mode '{self.mode}' requires keyword argument '{key}'."
                )

        # Check types of provided kwargs that match the schema
        for key, schema in mode_schema.items():
            if key in self.params and self.params[key] is not None:
                val = self.params[key]
                expected_type = schema.get('type')
                if expected_type and not isinstance(val, expected_type):
                    raise TypeError(
                        f"Keyword argument '{key}' for mode '{self.mode}' must be "
                        f"of type {expected_type}, got {type(val)}."
                    )

        # lag_range entries must be numeric (int for sample lags, float for time lags)
        if self.mode == 'lag':
            lr = self.params.get('lag_range')
            if lr is not None:
                items = list(lr)
                non_numeric = [x for x in items
                               if not isinstance(x, (int, float, np.integer, np.floating))]
                if non_numeric:
                    raise ValueError(
                        f"lag_range entries must all be numeric, but found non-numeric "
                        f"values: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}. "
                        f"Use range(-10, 11), a list of integers, or np.arange(...) for "
                        f"time-based lags (e.g. spike trains)."
                    )

        # Precision mode: validate threshold_ratio bounds
        if self.mode == 'precision':
            tr = self.params.get('threshold_ratio')
            if tr is not None:
                ratios = tr if isinstance(tr, (list, tuple)) else [tr]
                for r in ratios:
                    if not isinstance(r, (int, float)) or not (0 < r <= 1):
                        raise ValueError(
                            f"threshold_ratio must be a float in (0, 1] "
                            f"(or a list of such floats), got {r!r}."
                        )

        # Rigorous mode: validate delta_threshold and confidence_level
        if self.mode == 'rigorous':
            dt = self.params.get('delta_threshold')
            if dt is not None and (not isinstance(dt, (int, float)) or dt <= 0):
                raise ValueError(
                    f"delta_threshold must be a positive float, got {dt!r}."
                )
            cl = self.params.get('confidence_level')
            if cl is not None and (not isinstance(cl, (int, float)) or not (0 < cl < 1)):
                raise ValueError(
                    f"confidence_level must be a float in (0, 1), got {cl!r}."
                )

    def apply_defaults(self):
        """Populates missing parameters in base_params with defaults."""
        bp = self.params["base_params"]
        verbose = bp.get('verbose', True)

        for key, schema in BASE_PARAMS_SCHEMA.items():
            if key not in bp and 'default' in schema:
                default_val = schema['default']
                # Apply the default. If default is None, only apply it when the
                # schema explicitly allows None (i.e. type is a tuple containing type(None)).
                schema_type = schema['type']
                type_allows_none = isinstance(schema_type, tuple) and type(None) in schema_type
                if default_val is not None or type_allows_none:
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