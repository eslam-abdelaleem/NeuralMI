# neural_mi/config.py
"""Typed configuration objects for :func:`neural_mi.run`.

These dataclasses group the many individual parameters of ``run()`` into a few
cohesive, discoverable objects (:class:`Model`, :class:`Training`, :class:`Split`,
:class:`Estimator`, :class:`Output`, :class:`Processing`) plus one config per
analysis mode (:class:`Rigorous`, :class:`Precision`, :class:`Lag`,
:class:`Transfer`, :class:`Dimensionality`, :class:`Conditional`).

Design notes
------------
* **Single source of defaults.** Every field defaults to ``None`` meaning
  "unset". Lowering drops unset fields so that
  :data:`neural_mi.defaults.BASE_PARAMS_SCHEMA` (via ``apply_defaults``) remains
  the *only* place default values live. The config layer never duplicates a
  default value, so the two cannot drift.
* **Pure grouping + rename layer.** Field names equal the underlying
  ``base_params`` schema keys except for a few friendlier names (``Split.mode``
  → ``split_mode``, ``Split.gap_fraction`` → ``split_gap_fraction``,
  ``Estimator.name`` → ``estimator_name``, ``Estimator.params`` →
  ``estimator_params``). Validation of the lowered values is still performed by
  the existing ``ParameterValidator``.
* **Dict-friendly.** Anywhere ``run()`` accepts a config it also accepts a plain
  ``dict`` with the same keys (coerced via :func:`as_config`), so callers are not
  forced to import these classes.
"""
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

__all__ = [
    "Model", "Training", "Split", "Estimator", "Output", "Processing",
    "Rigorous", "Precision", "Lag", "Transfer", "Dimensionality", "Conditional",
    "as_config",
]

_UNSET = None


def _non_none(obj: Any, rename: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Return the dataclass's set (non-None) fields, applying key renames."""
    rename = rename or {}
    out: Dict[str, Any] = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is _UNSET:
            continue
        out[rename.get(f.name, f.name)] = val
    return out


T = TypeVar("T")


def as_config(value: Union[None, "T", Dict[str, Any]], cls: Type[T]) -> Optional[T]:
    """Coerce ``value`` to an instance of ``cls``.

    ``None`` → ``None``; an existing instance is returned unchanged; a ``dict`` is
    expanded into ``cls(**value)`` (raising a clear error on unknown keys).
    """
    if value is None:
        return None
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        valid = {f.name for f in fields(cls)}
        unknown = set(value) - valid
        if unknown:
            raise TypeError(
                f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}."
            )
        return cls(**value)
    raise TypeError(
        f"Expected {cls.__name__}, a dict, or None; got {type(value).__name__}."
    )


# ---------------------------------------------------------------------------
# Shared configs (lower into base_params)
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Model architecture: embedding network + critic."""
    embedding_model: Optional[str] = None          # 'mlp'|'cnn'|'cnn2d'|'gru'|'lstm'|'tcn'|'transformer'|'pretrained_backbone'
    embedding_dim: Optional[int] = None
    hidden_dim: Optional[Union[int, List[int]]] = None
    n_layers: Optional[int] = None
    n_layers_head: Optional[int] = None
    hidden_dim_head: Optional[Union[int, List[int]]] = None
    critic_type: Optional[str] = None              # 'separable'|'concat'|'hybrid'
    kernel_size: Optional[int] = None              # CNN/TCN
    bidirectional: Optional[bool] = None           # RNN
    nhead: Optional[int] = None                    # Transformer
    dropout: Optional[float] = None
    norm_layer: Optional[str] = None               # 'layer'|'batch'|None
    use_spectral_norm: Optional[bool] = None
    shared_encoder: Optional[bool] = None
    max_n_batches: Optional[int] = None            # critic chunking
    custom_critic: Optional[Any] = None            # torch.nn.Module
    custom_embedding_cls: Optional[type] = None
    pytorch_predefined: Optional[str] = None       # torchvision backbone name
    pretrained: Optional[bool] = None
    use_variational: Optional[bool] = None
    beta: Optional[float] = None
    # Optional decoder / information-bottleneck head
    use_decoder: Optional[bool] = None
    decoder_weight: Optional[float] = None
    decoder_weight_x: Optional[float] = None
    decoder_weight_y: Optional[float] = None
    decoder_output_activation_x: Optional[str] = None
    decoder_output_activation_y: Optional[str] = None

    def to_base_params(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Training:
    """Optimization loop: epochs, optimizer, scheduler, evaluation, augmentation."""
    n_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    patience: Optional[int] = None
    optimizer: Optional[Union[str, type]] = None
    optimizer_params: Optional[Dict[str, Any]] = None
    scheduler: Optional[Union[str, type]] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    gradient_clip_val: Optional[float] = None
    use_amp: Optional[Union[bool, str]] = None
    eval_train: Optional[Union[bool, float, int]] = None
    peak_fraction: Optional[float] = None
    smoothing_sigma: Optional[float] = None
    median_window: Optional[int] = None
    min_improvement: Optional[float] = None
    max_eval_samples: Optional[int] = None
    train_subset_size: Optional[int] = None
    lr_head_multiplier: Optional[float] = None
    save_best_model_path: Optional[str] = None
    random_time_shifting: Optional[bool] = None
    epochs_to_max_shift: Optional[int] = None
    augmentation_params: Optional[Dict[str, Any]] = None
    augmentation_params_x: Optional[Dict[str, Any]] = None
    augmentation_params_y: Optional[Dict[str, Any]] = None
    dataset_device: Optional[str] = None
    min_reliable_samples: Optional[int] = None

    def to_base_params(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Split:
    """Train/test splitting strategy."""
    mode: Optional[str] = None                     # 'blocked'|'random'
    train_fraction: Optional[float] = None
    n_test_blocks: Optional[int] = None
    gap_fraction: Optional[float] = None
    train_indices: Optional[Any] = None
    test_indices: Optional[Any] = None

    def to_base_params(self) -> Dict[str, Any]:
        return _non_none(self, rename={"mode": "split_mode",
                                       "gap_fraction": "split_gap_fraction"})


@dataclass
class Estimator:
    """MI lower-bound estimator selection."""
    name: Optional[str] = None                     # 'infonce'|'smile'
    params: Optional[Dict[str, Any]] = None

    def to_base_params(self) -> Dict[str, Any]:
        return _non_none(self, rename={"name": "estimator_name",
                                       "params": "estimator_params"})


@dataclass
class Output:
    """Result units, spectral tracking, embedding returns, and display labels."""
    units: Optional[str] = None                    # 'bits'|'nats'
    spectral_mode: Optional[str] = None            # 'none'|'summary'|'full'
    max_index_reduction: Optional[float] = None
    return_embeddings: Optional[bool] = None
    track_embeddings: Optional[Union[bool, float, int, str]] = None
    return_rotated_embeddings: Optional[bool] = None
    rotated_embeddings_whitening: Optional[str] = None
    rotated_embeddings_per_epoch: Optional[bool] = None
    return_rotation_matrices: Optional[bool] = None
    # Display-only labels (not part of base_params; carried in result.params)
    x_name: Optional[str] = None
    y_name: Optional[str] = None
    channel_names_x: Optional[List[str]] = None
    channel_names_y: Optional[List[str]] = None

    _LABEL_FIELDS = ("x_name", "y_name", "channel_names_x", "channel_names_y")

    def to_base_params(self) -> Dict[str, Any]:
        d = _non_none(self)
        # 'units' lowers to the schema key 'output_units'
        if "units" in d:
            d["output_units"] = d.pop("units")
        for k in self._LABEL_FIELDS:
            d.pop(k, None)
        return d

    def to_labels(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._LABEL_FIELDS if getattr(self, k) is not None}


@dataclass
class Processing:
    """Raw-data processors for X and Y (and their time vectors)."""
    x: Optional[str] = None                        # processor_type_x
    x_params: Optional[Dict[str, Any]] = None
    y: Optional[str] = None                        # processor_type_y
    y_params: Optional[Dict[str, Any]] = None
    x_time: Optional[Any] = None
    y_time: Optional[Any] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return _non_none(self, rename={
            "x": "processor_type_x", "x_params": "processor_params_x",
            "y": "processor_type_y", "y_params": "processor_params_y",
        })


# ---------------------------------------------------------------------------
# Mode-specific configs (lower into analysis_kwargs / mode arguments)
# ---------------------------------------------------------------------------

@dataclass
class Rigorous:
    """Parameters for ``mode='rigorous'`` bias-corrected extrapolation."""
    gamma_range: Optional[Any] = None
    delta_threshold: Optional[float] = None
    min_gamma_points: Optional[int] = None
    confidence_level: Optional[float] = None
    residual_threshold: Optional[float] = None
    r2_threshold: Optional[float] = None
    leverage_threshold: Optional[float] = None

    def to_analysis_kwargs(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Precision:
    """Parameters for ``mode='precision'`` spike-timing precision sweep."""
    tau_grid: Optional[List[float]] = None
    corrupt_target: Optional[str] = None           # 'x'|'y'|'both'
    corruption_method: Optional[str] = None        # 'rounding'|'noise'
    n_noise_samples: Optional[int] = None
    threshold_ratio: Optional[float] = None

    def to_analysis_kwargs(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Lag:
    """Parameters for ``mode='lag'`` temporal-offset sweep."""
    lag_range: Optional[Any] = None
    equalize_n: Optional[bool] = None

    def to_analysis_kwargs(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Transfer:
    """Parameters for ``mode='transfer'`` transfer entropy."""
    history_window: Optional[int] = None
    prediction_horizon: Optional[int] = None
    bidirectional: Optional[bool] = None
    rigorous: Optional[bool] = None
    gamma_range: Optional[Any] = None
    delta_threshold: Optional[float] = None
    min_gamma_points: Optional[int] = None
    confidence_level: Optional[float] = None
    residual_threshold: Optional[float] = None
    r2_threshold: Optional[float] = None
    leverage_threshold: Optional[float] = None

    def to_analysis_kwargs(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Dimensionality:
    """Parameters for ``mode='dimensionality'`` latent-dimensionality analysis."""
    split_method: Optional[str] = None
    n_splits: Optional[int] = None
    lag: Optional[int] = None
    channel_indices_x: Optional[List[int]] = None
    sigma_add: Optional[Any] = None
    sigma_add_units: Optional[str] = None
    stabilize_counts: Optional[bool] = None

    def to_analysis_kwargs(self) -> Dict[str, Any]:
        return _non_none(self)


@dataclass
class Conditional:
    """Parameters for ``mode='conditional'`` conditional MI (the Z variable)."""
    z_data: Optional[Any] = None
    z_time: Optional[Any] = None
    z_processor_type: Optional[str] = None
    z_processor_params: Optional[Dict[str, Any]] = None
    rigorous: Optional[bool] = None
    gamma_range: Optional[Any] = None
    delta_threshold: Optional[float] = None
    min_gamma_points: Optional[int] = None
    confidence_level: Optional[float] = None
    residual_threshold: Optional[float] = None
    r2_threshold: Optional[float] = None
    leverage_threshold: Optional[float] = None

    # z_* are consumed as dedicated run arguments; the rest are analysis kwargs.
    _Z_FIELDS = ("z_data", "z_time", "z_processor_type", "z_processor_params")

    def to_z_kwargs(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._Z_FIELDS if getattr(self, k) is not None}

    def to_analysis_kwargs(self) -> Dict[str, Any]:
        d = _non_none(self)
        for k in self._Z_FIELDS:
            d.pop(k, None)
        return d
