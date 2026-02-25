# tests/test_strict_validation.py
import pytest
import numpy as np
import neural_mi as nmi
from neural_mi.validation import ParameterValidator
from neural_mi.defaults import BASE_PARAMS_SCHEMA

class TestStrictValidation:
    @pytest.fixture
    def data(self):
        x = np.random.randn(50, 1).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)
        return x, y

    def test_run_detects_invalid_base_params(self, data):
        x, y = data
        # 'typo_param' is not in BASE_PARAMS_SCHEMA
        with pytest.raises(ValueError, match="Unknown parameters in 'base_params'"):
            nmi.run(x, y, base_params={'n_epochs': 1, 'typo_param': 10}, n_workers=1)

    def test_run_applies_defaults(self, data, caplog):
        x, y = data
        # We don't specify 'n_layers' or 'embedding_dim'
        # run() calls ParameterValidator.apply_defaults()
        # We expect it to log usage of defaults

        # We must use 'estimate' mode and capture logs
        # Note: running actual training might take time, so we set minimal epochs
        with caplog.at_level('INFO'):
            result = nmi.run(x, y, base_params={'n_epochs': 2, 'batch_size': 16}, n_workers=1)

        # Check logs for defaults
        assert "Parameter 'n_layers' not specified. Defaulting to 2" in caplog.text
        assert "Parameter 'embedding_dim' not specified. Defaulting to 64" in caplog.text

        # Check that result reflects success (defaults worked)
        assert result.mi_estimate is not None

    def test_run_defaults_logging_suppressed_if_not_verbose(self, data, caplog):
        x, y = data
        caplog.clear()
        with caplog.at_level('INFO'):
            nmi.run(x, y, base_params={'n_epochs': 1, 'batch_size': 16}, verbose=False, n_workers=1)

        # Should NOT see the logs
        assert "Parameter 'n_layers' not specified" not in caplog.text

    def test_run_validates_types(self, data):
        x, y = data
        # n_epochs should be int
        with pytest.raises(TypeError, match="Parameter 'n_epochs' must be of type"):
            nmi.run(x, y, base_params={'n_epochs': '50'}, n_workers=1)

    def test_run_validates_min_values(self, data):
        x, y = data
        # learning_rate must be >= 0 (actually > 0 usually but schema says min 0.0)
        # n_layers min 0
        # batch_size min 1
        with pytest.raises(ValueError, match="Parameter 'batch_size' must be >= 1"):
            nmi.run(x, y, base_params={'n_epochs': 1, 'batch_size': 0}, n_workers=1)

    def test_run_detects_invalid_choice_values(self, data):
        x, y = data
        # 'bla' is not a valid critic_type
        with pytest.raises(ValueError, match="Parameter 'critic_type' has invalid value 'bla'"):
            nmi.run(x, y, base_params={'critic_type': 'bla'}, n_workers=1)

    def test_run_detects_invalid_processor_params(self, data):
        x, y = data
        # 'invalid_param' for continuous processor
        with pytest.raises(ValueError, match="Unknown parameters for continuous processor"):
            nmi.run(x, y, processor_type_x='continuous',
                    processor_params_x={'window_size': 1, 'invalid_param': 5},
                    base_params={'n_epochs': 1}, n_workers=1)

    def test_validator_apply_defaults_logic(self):
        # Unit test for the method itself
        params = {
            'mode': 'estimate',
            'base_params': {'n_epochs': 10} # Missing many required fields
        }
        val = ParameterValidator(params)
        val.apply_defaults()

        bp = params['base_params']
        assert bp['n_layers'] == 2
        assert bp['embedding_dim'] == 64
        assert bp['hidden_dim'] == 64
        # Existing should not be overwritten
        assert bp['n_epochs'] == 10
