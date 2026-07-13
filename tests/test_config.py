"""Unit tests for the typed configuration objects in neural_mi.config."""
import dataclasses
import pytest

from neural_mi import config as cfg
from neural_mi.defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA


def _fill_all(cls):
    """Instantiate a dataclass with every field set to a non-None sentinel."""
    kwargs = {f.name: 1 for f in dataclasses.fields(cls)}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# as_config coercion
# ---------------------------------------------------------------------------

def test_as_config_none():
    assert cfg.as_config(None, cfg.Model) is None


def test_as_config_passthrough_instance():
    m = cfg.Model(embedding_dim=8)
    assert cfg.as_config(m, cfg.Model) is m


def test_as_config_from_dict():
    m = cfg.as_config({"embedding_dim": 8, "hidden_dim": 32}, cfg.Model)
    assert isinstance(m, cfg.Model)
    assert m.embedding_dim == 8 and m.hidden_dim == 32


def test_as_config_unknown_key_raises():
    with pytest.raises(TypeError, match="Unknown key"):
        cfg.as_config({"embedding_dim": 8, "nonsense": 1}, cfg.Model)


def test_as_config_wrong_type_raises():
    with pytest.raises(TypeError):
        cfg.as_config(42, cfg.Model)


# ---------------------------------------------------------------------------
# Lowering mechanics: drop unset (None) fields, apply renames
# ---------------------------------------------------------------------------

def test_unset_fields_dropped():
    # Only explicitly-set fields should appear; everything else defers to schema defaults.
    assert cfg.Model(embedding_dim=16).to_base_params() == {"embedding_dim": 16}
    assert cfg.Training().to_base_params() == {}


def test_split_renames():
    d = cfg.Split(mode="random", gap_fraction=0.0).to_base_params()
    assert d == {"split_mode": "random", "split_gap_fraction": 0.0}


def test_estimator_renames():
    d = cfg.Estimator(name="smile", params={"clip": 5.0}).to_base_params()
    assert d == {"estimator_name": "smile", "estimator_params": {"clip": 5.0}}


def test_output_units_and_labels_split():
    o = cfg.Output(units="nats", x_name="LFP", channel_names_x=["a", "b"])
    bp = o.to_base_params()
    assert bp == {"output_units": "nats"}          # units renamed, labels excluded
    assert o.to_labels() == {"x_name": "LFP", "channel_names_x": ["a", "b"]}


def test_processing_renames():
    d = cfg.Processing(x="continuous", x_params={"window_size": 1}).to_kwargs()
    assert d == {"processor_type_x": "continuous", "processor_params_x": {"window_size": 1}}


def test_conditional_splits_z_from_analysis():
    c = cfg.Conditional(z_data=[1, 2], rigorous=True, confidence_level=0.9)
    assert c.to_z_kwargs() == {"z_data": [1, 2]}
    assert c.to_analysis_kwargs() == {"rigorous": True, "confidence_level": 0.9}


# ---------------------------------------------------------------------------
# Consistency: every emitted base_params key must be a real schema key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [cfg.Model, cfg.Training, cfg.Split, cfg.Estimator, cfg.Output])
def test_shared_config_keys_are_valid_schema_keys(cls):
    emitted = set(_fill_all(cls).to_base_params())
    unknown = emitted - set(BASE_PARAMS_SCHEMA)
    assert not unknown, (
        f"{cls.__name__} emits base_params keys not in BASE_PARAMS_SCHEMA: {sorted(unknown)}"
    )


def test_processing_keys_are_run_processor_keys():
    emitted = set(_fill_all(cfg.Processing).to_kwargs())
    expected = {"processor_type_x", "processor_params_x",
                "processor_type_y", "processor_params_y", "x_time", "y_time"}
    assert emitted == expected


@pytest.mark.parametrize("cls,mode", [
    (cfg.Precision, "precision"),
    (cfg.Lag, "lag"),
])
def test_mode_config_keys_are_valid_mode_kwargs(cls, mode):
    # Precision and Lag map exactly onto MODE_KWARGS_SCHEMA. (Rigorous/Transfer/
    # Dimensionality/Conditional also carry keys consumed directly by their
    # analysis functions -- e.g. gamma_range, sigma_add -- so they are not
    # asserted against the schema here; their lowering mechanics are covered above.)
    emitted = set(_fill_all(cls).to_analysis_kwargs())
    allowed = set(MODE_KWARGS_SCHEMA[mode])
    unknown = emitted - allowed
    assert not unknown, f"{cls.__name__} emits unknown {mode} kwargs: {sorted(unknown)}"
