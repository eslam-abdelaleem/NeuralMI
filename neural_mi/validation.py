from typing import Dict, Any
import numpy as np
import torch
import warnings
from neural_mi.exceptions import ParameterError, DataShapeError, InsufficientDataError

class ParameterValidator:
    """
    A class to validate the parameters passed to the `run` function.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the validator with all parameters from the run function.
        """
        self.params = params
        self.mode = params.get("mode")

    def validate(self):
        """
        Runs all validation checks.
        """
        self._validate_required_keys()
        self._validate_base_params()
        self._validate_processor_params()
        self._validate_sweep_params()

    def _validate_required_keys(self):
        """Checks for the presence of essential keys."""
        if self.params.get("base_params") is None:
            raise ParameterError("'base_params' dictionary is required.")

    def _validate_base_params(self):
        """Validates the contents of the 'base_params' dictionary."""
        bp = self.params["base_params"]

        if not isinstance(bp, dict):
            raise TypeError("'base_params' must be a dictionary.")

        checks = {
            "n_epochs": (int, (1, None)),
            "learning_rate": (float, (0.0, None)),
            "batch_size": (int, (1, None)),
            "patience": (int, (0, None)),
            "embedding_dim": (int, (1, None)),
            "hidden_dim": (int, (1, None)),
            "n_layers": (int, (0, None)),
        }

        for key, (dtype, value_range) in checks.items():
            if key in bp:
                if not isinstance(bp[key], dtype):
                    raise TypeError(f"'base_params[\"{key}\"]' must be of type {dtype.__name__}.")

                min_val, max_val = value_range
                if min_val is not None and bp[key] < min_val:
                    raise ParameterError(f"'base_params[\"{key}\"]' must be at least {min_val}.")
                if max_val is not None and bp[key] > max_val:
                     raise ParameterError(f"'base_params[\"{key}\"]' must be no more than {max_val}.")

    def _validate_processor_params(self):
        """Validates processor-related parameters."""
        if self.params.get("processor_type"):
            if self.params["processor_type"] not in ["continuous", "spike"]:
                raise ParameterError("'processor_type' must be 'continuous' or 'spike'.")

            if self.params.get("processor_params") is None:
                 raise ParameterError("'processor_params' are required when 'processor_type' is specified.")

            pp = self.params["processor_params"]
            if not isinstance(pp, dict):
                raise TypeError("'processor_params' must be a dictionary.")

    def _validate_sweep_params(self):
        """Validates parameters specific to sweep modes."""
        if self.mode in ["sweep", "dimensionality"]:
            if self.params.get("sweep_grid") is None:
                raise ParameterError(f"'sweep_grid' is required for mode='{self.mode}'.")

        if self.mode == "dimensionality":
            sg = self.params["sweep_grid"]
            if "embedding_dim" not in sg:
                raise ParameterError("'sweep_grid' must contain 'embedding_dim' for mode='dimensionality'.")


class DataValidator:
    """Validates input data for neural_mi."""

    def __init__(self, x_data, y_data, processor_type):
        self.x_data = x_data
        self.y_data = y_data
        self.processor_type = processor_type

    def validate(self):
        """Run all validation checks."""
        self._validate_data_types()
        self._validate_data_shapes()
        self._validate_data_values()
        self._validate_compatibility()

    def _validate_data_types(self):
        """Check data types are valid."""
        for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
            if data is None:
                continue
            if not isinstance(data, (np.ndarray, torch.Tensor, list)):
                raise TypeError(
                    f"{name} must be np.ndarray, torch.Tensor, or list, "
                    f"got {type(data)}"
                )

            if self.processor_type != 'spike':
                is_numeric = False
                if isinstance(data, torch.Tensor):
                    if data.dtype != torch.bool:
                        is_numeric = True
                elif isinstance(data, np.ndarray):
                    if np.issubdtype(data.dtype, np.number):
                        is_numeric = True
                elif isinstance(data, list):
                    if not data or all(isinstance(item, (int, float)) for item in data):
                        is_numeric = True

                if not is_numeric:
                    raise TypeError(f"{name} must contain numeric data for continuous processing.")

    def _validate_data_shapes(self):
        """Check data shapes are compatible."""
        if self.processor_type == 'continuous':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None:
                    continue
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    if data.ndim not in [2, 3]:
                        raise DataShapeError(
                            f"{name} must be 2D (for raw data) or 3D "
                            f"(for pre-processed data), got shape {data.shape}"
                        )
                    if data.ndim == 2 and data.size == 0:
                        raise InsufficientDataError(f"{name} is empty")

        elif self.processor_type == 'spike':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None:
                    continue
                if not isinstance(data, list):
                    raise TypeError(
                        f"{name} must be a list of arrays for spike data, "
                        f"got {type(data)}"
                    )
                if len(data) == 0:
                    raise InsufficientDataError(f"{name} is empty")
                for i, spikes in enumerate(data):
                    if not isinstance(spikes, np.ndarray):
                        raise TypeError(
                            f"{name}[{i}] must be np.ndarray, got {type(spikes)}"
                        )
                    if spikes.ndim != 1:
                        raise DataShapeError(
                            f"{name}[{i}] must be 1D array of spike times, "
                            f"got shape {spikes.shape}"
                        )

        if self.y_data is not None and self.x_data is not None:
            if self.processor_type == 'spike':
                if len(self.x_data) != len(self.y_data):
                    warnings.warn(
                        f"x_data has {len(self.x_data)} neurons but "
                        f"y_data has {len(self.y_data)} neurons. "
                        f"This is allowed but unusual."
                    )
            elif self.processor_type == 'continuous':
                if hasattr(self.x_data, 'ndim') and hasattr(self.y_data, 'ndim'):
                    if self.x_data.ndim == 3 and self.y_data.ndim == 3:
                        if self.x_data.shape[0] != self.y_data.shape[0]:
                            raise DataShapeError(
                                f"Pre-processed data must have same number of samples. "
                                f"x_data has {self.x_data.shape[0]} samples, "
                                f"y_data has {self.y_data.shape[0]} samples."
                            )

    def _validate_data_values(self):
        """Check data values are valid."""
        if self.processor_type == 'spike':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None:
                    continue
                for i, spikes in enumerate(data):
                    if len(spikes) > 0:
                        if np.any(spikes < 0):
                            raise ValueError(
                                f"{name}[{i}] contains negative spike times"
                            )
                        if not np.all(spikes[:-1] <= spikes[1:]):
                            warnings.warn(
                                f"{name}[{i}] spike times are not sorted. "
                                f"Sorting automatically."
                            )
                            if isinstance(data, list) and isinstance(data[i], np.ndarray):
                                data[i] = np.sort(spikes)

        elif self.processor_type == 'continuous':
            for data, name in [(self.x_data, 'x_data'), (self.y_data, 'y_data')]:
                if data is None:
                    continue
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    if isinstance(data, torch.Tensor):
                        if torch.any(torch.isnan(data)):
                            raise ValueError(f"{name} contains NaN values")
                        if torch.any(torch.isinf(data)):
                            raise ValueError(f"{name} contains Inf values")
                    else:
                        if np.any(np.isnan(data)):
                            raise ValueError(f"{name} contains NaN values")
                        if np.any(np.isinf(data)):
                            raise ValueError(f"{name} contains Inf values")

    def _validate_compatibility(self):
        """Check data is compatible with processor parameters."""
        pass