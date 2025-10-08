# tests/test_validation.py
import pytest
import numpy as np
from neural_mi.validation import ParameterValidator, DataValidator
from neural_mi.exceptions import DataShapeError

def get_valid_params():
    return {"mode": "estimate", "base_params": {"n_epochs": 1}, "processor_type_x": "continuous", "processor_params_x": {}}

def test_param_validator_success():
    ParameterValidator(get_valid_params()).validate()

def test_param_validator_missing_base_params():
    with pytest.raises(ValueError, match="'base_params' is required"):
        ParameterValidator({"mode": "estimate"}).validate()

def test_data_validator_success():
    x = np.random.randn(5, 100)
    DataValidator(x, x, processor_type_x='continuous', processor_type_y='continuous').validate()

def test_data_validator_wrong_shape():
    x = np.random.randn(100) # Should be 2D or 3D
    with pytest.raises(DataShapeError):
        DataValidator(x, x, processor_type_x='continuous', processor_type_y='continuous').validate()

def test_data_validator_contains_nan():
    x = np.random.randn(5, 100)
    x[2, 5] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        DataValidator(x, x, processor_type_x='continuous', processor_type_y='continuous').validate()

def test_param_validator_invalid_types():
    with pytest.raises(TypeError):
        ParameterValidator({"mode": "estimate", "base_params": {"n_epochs": "5"}}).validate()

def test_param_validator_invalid_values():
    with pytest.raises(ValueError):
        ParameterValidator({"mode": "estimate", "base_params": {"n_epochs": 0}}).validate()

def test_param_validator_sweep_without_grid():
    with pytest.raises(ValueError, match="'sweep_grid' required"):
        ParameterValidator({"mode": "sweep", "base_params": {}}).validate()

def test_data_validator_spike_data_errors():
    with pytest.raises(TypeError, match="must be a non-empty list"):
        DataValidator([], None, processor_type_x="spike", processor_type_y="spike").validate()
    
    with pytest.raises(TypeError, match="must be a 1D np.ndarray"):
        DataValidator([np.array([[1],[2]])], None, processor_type_x="spike", processor_type_y="spike").validate()

def test_data_validator_non_numeric_continuous_data():
    x = np.array([['a'], ['b'], ['c']])
    with pytest.raises(TypeError):
        DataValidator(x, x, 'continuous', 'continuous').validate()

def test_data_validator_categorical_success():
    x = np.random.randint(0, 3, size=(2, 100))
    DataValidator(x, x, processor_type_x='categorical', processor_type_y='categorical').validate()

def test_data_validator_categorical_wrong_type():
    x = np.random.rand(2, 100) # Should be integers
    with pytest.raises(TypeError, match="must be integer type"):
        DataValidator(x, x, processor_type_x='categorical', processor_type_y='categorical').validate()