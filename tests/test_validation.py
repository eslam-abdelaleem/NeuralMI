# tests/test_validation.py
import pytest
import numpy as np
import neural_mi as nmi
from neural_mi.validation import ParameterValidator, DataValidator
from neural_mi.defaults import BASE_PARAMS_SCHEMA
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


# --- Integration-level validation tests (via nmi.run) ---

@pytest.fixture
def small_data():
    x = np.random.randn(50, 1).astype(np.float32)
    y = np.random.randn(50, 1).astype(np.float32)
    return x, y


def test_run_detects_invalid_base_params(small_data):
    x, y = small_data
    with pytest.raises(ValueError, match="Unknown parameters in 'base_params'"):
        nmi.run(x, y, base_params={'n_epochs': 1, 'typo_param': 10}, n_workers=1)


def test_run_validates_types(small_data):
    x, y = small_data
    with pytest.raises(TypeError, match="Parameter 'n_epochs' must be of type"):
        nmi.run(x, y, base_params={'n_epochs': '50'}, n_workers=1)


def test_run_validates_min_values(small_data):
    x, y = small_data
    with pytest.raises(ValueError, match="Parameter 'batch_size' must be >= 1"):
        nmi.run(x, y, base_params={'n_epochs': 1, 'batch_size': 0}, n_workers=1)


def test_run_detects_invalid_choice_values(small_data):
    x, y = small_data
    with pytest.raises(ValueError, match="Parameter 'critic_type' has invalid value 'bla'"):
        nmi.run(x, y, base_params={'critic_type': 'bla'}, n_workers=1)


def test_run_detects_invalid_processor_params(small_data):
    x, y = small_data
    with pytest.raises(ValueError, match="Unknown parameters for continuous processor"):
        nmi.run(x, y, processor_type_x='continuous',
                processor_params_x={'window_size': 1, 'invalid_param': 5},
                base_params={'n_epochs': 1}, n_workers=1)


def test_validator_apply_defaults_logic():
    params = {'mode': 'estimate', 'base_params': {'n_epochs': 10}}
    val = ParameterValidator(params)
    val.apply_defaults()
    bp = params['base_params']
    assert bp['n_layers'] == 2
    assert bp['embedding_dim'] == 64
    assert bp['hidden_dim'] == 64
    assert bp['n_epochs'] == 10  # Existing value not overwritten


def test_run_applies_defaults(small_data, caplog):
    x, y = small_data
    with caplog.at_level('INFO', logger='neural_mi'):
        result = nmi.run(x, y, base_params={'n_epochs': 2, 'batch_size': 16}, verbose=True, n_workers=1)
    assert "Parameter 'n_layers' not specified. Defaulting to 2" in caplog.text
    assert "Parameter 'embedding_dim' not specified. Defaulting to 64" in caplog.text
    assert result.mi_estimate is not None


def test_run_defaults_logging_suppressed_if_not_verbose(small_data, caplog):
    x, y = small_data
    caplog.clear()
    with caplog.at_level('INFO'):
        nmi.run(x, y, base_params={'n_epochs': 1, 'batch_size': 16}, verbose=False, n_workers=1)
    assert "Parameter 'n_layers' not specified" not in caplog.text