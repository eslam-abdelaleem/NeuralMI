import pytest
from neural_mi.validation import ParameterValidator

# A minimal, valid set of parameters for testing
def get_valid_params():
    return {
        "x_data": None, # Not checked by validator
        "y_data": None, # Not checked by validator
        "mode": "estimate",
        "processor_type": None,
        "processor_params": None,
        "base_params": {
            "n_epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 128,
            "patience": 5,
            "embedding_dim": 16,
            "hidden_dim": 64,
            "n_layers": 2
        },
        "sweep_grid": None,
        "output_units": "bits",
        "estimator": "infonce",
        "analysis_kwargs": {}
    }

def test_validator_with_valid_params():
    """Tests that the validator passes with a correct set of parameters."""
    params = get_valid_params()
    validator = ParameterValidator(params)
    try:
        validator.validate()
    except (ValueError, TypeError) as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")

def test_validator_missing_base_params():
    """Tests that a ValueError is raised if base_params is missing."""
    params = get_valid_params()
    params["base_params"] = None
    with pytest.raises(ValueError, match="'base_params' dictionary is required"):
        ParameterValidator(params).validate()

def test_validator_wrong_base_param_type():
    """Tests that a TypeError is raised for wrong parameter types."""
    params = get_valid_params()
    params["base_params"]["n_epochs"] = "10" # Should be int
    with pytest.raises(TypeError, match="'base_params\\[\"n_epochs\"\\]' must be of type int"):
        ParameterValidator(params).validate()

def test_validator_out_of_range_base_param():
    """Tests that a ValueError is raised for out-of-range values."""
    params = get_valid_params()
    params["base_params"]["learning_rate"] = -0.1 # Should be positive
    with pytest.raises(ValueError, match="'base_params\\[\"learning_rate\"\\]' must be at least 0.0"):
        ParameterValidator(params).validate()

def test_validator_sweep_mode_missing_grid():
    """Tests that sweep mode requires a sweep_grid."""
    params = get_valid_params()
    params["mode"] = "sweep"
    params["sweep_grid"] = None
    with pytest.raises(ValueError, match="'sweep_grid' is required for mode='sweep'"):
        ParameterValidator(params).validate()

def test_validator_dimensionality_mode_missing_embedding_dim():
    """Tests that dimensionality mode requires embedding_dim in the grid."""
    params = get_valid_params()
    params["mode"] = "dimensionality"
    params["sweep_grid"] = {"learning_rate": [0.1, 0.01]} # Missing embedding_dim
    with pytest.raises(ValueError, match="'sweep_grid' must contain 'embedding_dim'"):
        ParameterValidator(params).validate()

def test_validator_processor_missing_params():
    """Tests that specifying a processor type requires processor_params."""
    params = get_valid_params()
    params["processor_type"] = "continuous"
    params["processor_params"] = None
    with pytest.raises(ValueError, match="'processor_params' are required"):
        ParameterValidator(params).validate()