from typing import Dict, Any

class ParameterValidator:
    """
    A class to validate the parameters passed to the `run` function.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the validator with all parameters from the run function.

        Parameters
        ----------
        params : dict
            A dictionary containing all parameters passed to `nmi.run()`.
        """
        self.params = params
        self.mode = params.get("mode")

    def validate(self):
        """
        Runs all validation checks.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        self._validate_required_keys()
        self._validate_base_params()
        self._validate_processor_params()
        self._validate_sweep_params()

    def _validate_required_keys(self):
        """Checks for the presence of essential keys."""
        if self.params.get("base_params") is None:
            raise ValueError("'base_params' dictionary is required.")

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
                    raise ValueError(f"'base_params[\"{key}\"]' must be at least {min_val}.")
                if max_val is not None and bp[key] > max_val:
                     raise ValueError(f"'base_params[\"{key}\"]' must be no more than {max_val}.")

    def _validate_processor_params(self):
        """Validates processor-related parameters."""
        if self.params.get("processor_type"):
            if self.params["processor_type"] not in ["continuous", "spike"]:
                raise ValueError("'processor_type' must be 'continuous' or 'spike'.")

            if self.params.get("processor_params") is None:
                 raise ValueError("'processor_params' are required when 'processor_type' is specified.")

            pp = self.params["processor_params"]
            if not isinstance(pp, dict):
                raise TypeError("'processor_params' must be a dictionary.")

    def _validate_sweep_params(self):
        """Validates parameters specific to sweep modes."""
        if self.mode in ["sweep", "dimensionality"]:
            if self.params.get("sweep_grid") is None:
                raise ValueError(f"'sweep_grid' is required for mode='{self.mode}'.")

        if self.mode == "dimensionality":
            sg = self.params["sweep_grid"]
            if "embedding_dim" not in sg:
                raise ValueError("'sweep_grid' must contain 'embedding_dim' for mode='dimensionality'.")