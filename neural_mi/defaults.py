# neural_mi/defaults.py
"""
Centralized definition of default parameters and allowed keys for validation.
"""
import copy
from typing import Any, Dict, Optional, Tuple

import torch

# Allowed parameters for base_params
# Each entry is (type_check, min_value or validation_function)
BASE_PARAMS_SCHEMA = {
    # Trainer parameters
    'n_epochs': {'type': int, 'min': 1, 'default': 300},
    'learning_rate': {'type': float, 'min': 0.0, 'default': 5e-4},
    'batch_size': {'type': int, 'min': 1, 'default': 128},
    # Training stops once the smoothed held-out curve has not improved for
    # `patience` epochs, and n_epochs caps a curve that keeps improving. Shifted
    # windows keep the curve rising for longer, and a curve without shifting often
    # tops early and falls.
    'patience': {'type': int, 'min': 0, 'default': 30},
    'output_units': {'type': str, 'default': 'bits'},
    # Real-time grids for X and Y, carried so that windowing deferred to the
    # task layer sees the same grid the eager path gets via create_dataset.
    'x_time': {'type': (object, type(None))},
    'y_time': {'type': (object, type(None))},
    'verbose': {'type': bool, 'default': False},
    'show_progress': {'type': bool, 'default': True},
    'device': {'type': (str, type(None), torch.device), 'default': None},
    'split_mode': {'type': str, 'default': 'blocked'},
    # Every epoch the windows a processor cuts start one offset later, the same
    # for every stream, drawn within one step (neural_mi/data/plan.py).
    'shift_windows': {'type': bool, 'default': True},
    'smoothing_sigma': {'type': float, 'min': 0.0, 'default': 1.0},
    'median_window': {'type': int, 'min': 1, 'default': 5},
    'min_improvement': {'type': float, 'min': 0.0, 'default': 0.001},
    'max_eval_samples': {'type': int, 'min': 1, 'default': 5000},
    'train_subset_size': {'type': (int, str, type(None)), 'min': 1, 'default': None},  # count | 'full'
    'test_subset_size': {'type': (int, str, type(None)), 'min': 1, 'default': None},   # count | 'full'
    'split_gap_fraction': {'type': float, 'min': 0.0, 'default': 0.5},
    'track_spectral_history': {'type': bool, 'default': False},
    'return_embeddings': {'type': bool, 'default': False},
    # Normalization applied to the embeddings before the cross-covariance SVD
    # behind both the participation ratios and the rotated embeddings.
    'whitening': {'type': (str, type(None)), 'default': 'std'},
    'use_spectral_norm': {'type': bool, 'default': True},
    # An encoder's input with its units and offset removed by the training rows'
    # statistics, saved with the network: 'auto' scales each stream as its kind asks
    # (stream_scaling), and 'none' leaves every input as it is.
    'input_scaling': {'type': (str, type(None)), 'default': 'auto'},
    # Every score of a critic the library builds starts at exactly zero
    # (BaseCritic.start_at_zero), so the start does not depend on the data's
    # dimension or on the spread of the first scores.
    'zero_start': {'type': bool, 'default': True},
    # An MLP encoder adds a linear map from its input to its embedding, which
    # starts closed (embeddings.MLP).
    'linear_path': {'type': bool, 'default': True},
    # The nonlinearity of every rectified hidden layer the library builds
    # (embeddings.ACTIVATIONS): in the MLP, CNN, CNN2D, TCN, transformer and
    # DeepSets encoders, the MLP heads and the decoders that mirror them.
    'activation': {'type': str, 'default': 'leaky_relu'},
    # A first embedding layer with more parameters than this is noted at the info
    # level: it may overfit a small dataset.
    'large_first_layer': {'type': int, 'min': 0, 'default': 500_000},
    'use_decoder': {'type': bool, 'default': False},
    'decoder_lambda': {'type': float, 'min': 0.0, 'default': 0.001},
    'decoder_lambda_x': {'type': (float, type(None)), 'min': 0.0, 'default': None},
    'decoder_lambda_y': {'type': (float, type(None)), 'min': 0.0, 'default': None},
    'decoder_output_activation_x': {'type': str, 'default': 'linear'},
    'decoder_output_activation_y': {'type': str, 'default': 'linear'},
    'gradient_clip_val': {'type': (float, type(None)), 'default': None},
    'save_best_model_path': {'type': (str, type(None)), 'default': None},
    'estimator_name': {'type': str, 'default': 'infonce'},
    'estimator_params': {'type': dict, 'default': {}},
    # Online augmentations applied per-batch during training only.
    # augmentation_params applies to both X and Y unless overridden by _x/_y.
    # Set augmentation_params_x or augmentation_params_y to {} to disable for one variable.
    'augmentation_params':   {'type': dict, 'default': {}},
    'augmentation_params_x': {'type': (dict, type(None)), 'default': None},
    'augmentation_params_y': {'type': (dict, type(None)), 'default': None},
    'use_variational': {'type': bool, 'default': False},
    'beta': {'type': float, 'min': 0.0, 'default': 1024.0},
    'train_fraction': {'type': float, 'min': 0.0, 'default': 0.9},
    'n_test_blocks': {'type': int, 'min': 1, 'default': 5},
    'optimizer': {'type': (str, type), 'default': 'adam'},
    'optimizer_params': {'type': dict, 'default': {}},
    'lr_head_multiplier': {'type': (float, int, type(None)), 'min': 0.0, 'default': None},  # Hybrid only; None → same LR as encoders
    'scheduler': {'type': (str, type, type(None)), 'default': None},
    'scheduler_params': {'type': dict, 'default': {}},
    # Per-epoch train MI tracking, yielding 'train_mi_history'.
    #   False, no per-epoch train evaluation.
    #   True, a fixed subset of up to max_eval_samples training rows, drawn apart from train_mi's.
    #   int >= 1, exactly that many training samples.
    #   float (0, 1), that fraction of the training set.
    #   1.0 / 'full', the entire training set, uncapped.
    # An unrecognised value raises instead of silently disabling tracking.
    'eval_train': {'type': (bool, float, int, str, type(None)), 'default': False},

    # Per-epoch embedding tracking.
    # Controls whether embeddings are extracted and stored at every epoch.
    # Mirroring eval_train style:
    #   False, no tracking (the default).
    #   True, track first 512 samples.
    #   int >= 1, track exactly that many samples (first N in the dataset).
    #   float (0, 1), track that fraction of the dataset.
    #   'full', track all samples (emits a UserWarning about memory cost).
    # The tracked subset is always the *first* N samples so that user-supplied
    # labels (passed to result.animate()) align with the original data ordering.
    'track_embeddings': {'type': (bool, float, int, str, type(None)), 'default': False},
    'return_rotated_embeddings': {'type': bool, 'default': False},
    # False (default): one global rotation derived from the best epoch, applied to all
    # tracked epochs uniformly (consistent coordinate system across epochs).
    # True: each tracked epoch gets its own SVD-based rotation (shows structure emerging).
    'rotated_embeddings_per_epoch': {'type': bool, 'default': False},
    # Whether to include U and V (the rotation matrices) in the result.
    'return_rotation_matrices': {'type': bool, 'default': False},

    # Model architecture parameters
    'shared_encoder': {'type': bool, 'default': False},
    'embedding_dim': {'type': int, 'min': 1, 'default': 64},
    # hidden_dim may be an int (uniform width) or a list of ints (per-layer widths).
    # When a list is given, n_layers is inferred from the list length and any
    # explicit n_layers value is ignored for the layers that receive a list.
    'hidden_dim': {'type': (int, list), 'default': 64},
    'n_layers': {'type': int, 'min': 0, 'default': 2},
    'n_layers_head': {'type': (int, type(None)), 'min': 1, 'default': None},  # Hybrid critic head; None → max(1, n_layers-1)
    'hidden_dim_head': {'type': (int, list, type(None)), 'default': None},  # Hybrid critic head; None → min(64, hidden_dim)
    'critic_type': {'type': str, 'default': 'separable'},
    'embedding_model': {'type': str, 'default': 'mlp'},  # 'mlp'|'cnn'|'cnn2d'|'gru'|'lstm'|'tcn'|'transformer'|'pretrained_backbone'|'lru'|'deepsets'
    'kernel_size': {'type': int, 'min': 1, 'default': 3}, # CNN/TCN
    'cnn_pooling': {'type': str, 'default': 'flatten'},   # CNN: 'flatten' keeps every position, 'mean' averages
    'bidirectional': {'type': bool, 'default': False}, # RNN
    'nhead': {'type': int, 'min': 1, 'default': 4}, # Transformer
    # Per-side encoder overrides. None means "same as X" throughout.
    'embedding_model_y': {'type': str, 'default': None},
    'embedding_dim_y': {'type': int, 'default': None},
    'hidden_dim_y': {'type': (int, list), 'default': None},
    'n_layers_y': {'type': int, 'default': None},
    'custom_embedding_cls_y': {'type': type, 'default': None},
    'max_n_batches': {'type': int, 'min': 1, 'default': 512}, # Critic chunking
    'dropout': {'type': float, 'min': 0.0, 'default': 0.0},
    # 'auto' is layer normalisation for the hybrid critic in mode='dimensionality'
    # and none otherwise. Layer norm stops the fits of that mode from stalling. It
    # also divides out each sample's overall scale and the information it carries.
    'norm_layer': {'type': (str, type(None)), 'default': 'auto'},
    # Whether embedding layers carry bias terms. Without them the network is
    # positively homogeneous, so an all-zero input embeds to exactly zero.
    # A mean-centring `norm_layer` is served by an affine-free RMSNorm when
    # this is off, since centring would otherwise undo the property.
    'bias': {'type': (bool, type(None)), 'default': True},
    # PretrainedBackboneEmbedding: torchvision model name and pretrained flag.
    'pytorch_predefined': {'type': (str, type(None)), 'default': None},
    'pretrained': {'type': bool, 'default': False},

    # Internal/Inferred parameters (usually not set by user but passed down)
    'input_dim_x': {'type': int},
    'input_dim_y': {'type': int},
    'n_channels_x': {'type': int},
    'n_channels_y': {'type': int},
    'processor_type_x': {'type': (str, type(None))},
    'processor_type_y': {'type': (str, type(None))},
    'processor_params_x': {'type': (dict, type(None))},
    'processor_params_y': {'type': (dict, type(None))},
    'custom_critic': {'type': (object, type(None))}, # class or function that builds the critic
    'custom_embedding_cls': {'type': (type, type(None))},
    'train_indices': {'type': (object, type(None))}, # numpy array
    'test_indices': {'type': (object, type(None))},
    'gamma': {'type': (int, float)}, # Rigorous
    'min_reliable_samples': {'type': int, 'min': 1, 'default': None},
    'lag': {'type': int},  # Result label: injected by run_lag_analysis per task; not a user-settable parameter.
    # Reproducibility, used by run() and task.py workers
    'random_seed': {'type': (int, type(None)), 'default': None},

    # Conservative epoch selection:
    # 1.0 (default) → use the epoch where smoothed test MI is maximised.
    # < 1.0          → use the first epoch where smoothed test MI ≥ peak_fraction * max_test_mi.
    #                  This gives a more conservative train-MI estimate by avoiding the final
    #                  noisy peak.  When < 1.0, the train MI at the actual peak epoch is also
    #                  returned in details as 'train_mi_at_peak'.
    'peak_fraction': {'type': float, 'min': 0.0, 'default': 1.0},
    # The epoch of the held-out curve at which the reported values are taken: the earliest at its top
    # ('plateau_start') or its maximum ('peak'). See neural_mi/training/epochs.py.
    'epoch_rule': {'type': str, 'default': 'plateau_start'},
    # How the held-out curve is judged (neural_mi/training/epochs.py), in units of the
    # curve's epoch-to-epoch noise, of standard errors or of epochs.
    # The level of the top is the median of the smoothed values within level_noise noise
    # units of the maximum. The plateau starts at the first epoch within start_noise noise
    # units of that level.
    'level_noise': {'type': float, 'min': 0.0, 'default': 2.0},
    'start_noise': {'type': float, 'min': 0.0, 'default': 0.5},
    # The top must stand resolved_noise noise units above zero to count as resolved.
    'resolved_noise': {'type': float, 'min': 0.0, 'default': 8.0},
    # A top reached within the smoothing's reach of the start that falls by more than
    # fall_noise noise units within fast_epochs epochs came too fast.
    'fast_epochs': {'type': int, 'min': 0, 'default': 8},
    'fall_noise': {'type': float, 'min': 0.0, 'default': 3.0},
    # The band: the epochs whose smoothed held-out value lies within top_se standard
    # errors of the curve's top, which the held-out rows cannot tell apart. A band
    # whose two ends give estimates further apart than band_share of the estimate
    # is a flat top.
    'top_se': {'type': float, 'min': 0.0, 'default': 1.0},
    'band_share': {'type': float, 'min': 0.0, 'default': 0.3},
    # A held-out value more than clear_se standard errors above zero shows that the
    # network learned something.
    'clear_se': {'type': float, 'min': 0.0, 'default': 3.0},
    # A curve that rose by rising_noise noise units over its last rising_epochs epochs,
    # when training ran to its limit, was still rising.
    'rising_epochs': {'type': int, 'min': 1, 'default': 20},
    'rising_noise': {'type': float, 'min': 0.0, 'default': 3.0},
    # A value above ceiling_fraction of its InfoNCE ceiling, log of the rows per
    # evaluation, is near that ceiling.
    'ceiling_fraction': {'type': float, 'min': 0.0, 'default': 0.85},
    # The chosen epoch may be unreliable when more than saturated_epoch_fraction of the
    # smoothed held-out curve lies above saturation_fraction of its ceiling.
    'saturation_fraction': {'type': float, 'min': 0.0, 'default': 0.9},
    'saturated_epoch_fraction': {'type': float, 'min': 0.0, 'default': 0.5},
    # The held-out value of the network at the chosen epoch falls short of a positive estimate when it
    # is below gap_share of it, by more than clear_se standard errors of their difference.
    'gap_share': {'type': float, 'min': 0.0, 'default': 0.5},
    # A quantity combined from several terms warns when an error in a term reaches
    # its value amplified this many times (neural_mi/analysis/sweep.py).
    'amplification_threshold': {'type': float, 'min': 1.0, 'default': 10.0},

    # The data check (neural_mi/preprocessing.py), read from Preprocessing(...).
    # A stream's typical channel scale more than scale_band from unit scale warns.
    'scale_band': {'type': float, 'min': 1.0, 'default': 10.0},
    # For a rate quantity on a series, a channel whose own past predicts more than
    # predicted_share of its variance is reported.
    'predicted_share': {'type': float, 'min': 0.0, 'default': 0.9},
    # The check reads at most check_rows rows at a time, and fits a channel's own past on
    # the first check_rows rows.
    'check_rows': {'type': int, 'min': 1, 'default': 100_000},
    # The scale of rows a spike-time or categorical processor builds is measured on their
    # first sample_rows rows.
    'sample_rows': {'type': int, 'min': 1, 'default': 10_000},
    # Channels of a stream whose standard deviations spread over more than a factor
    # of spread_band warn: scaling channel by channel gives them equal weight.
    'spread_band': {'type': float, 'min': 1.0, 'default': 10.0},
    # A channel whose mean lies more than offset_band of its standard deviations from
    # zero warns: in single precision its variation is resolved coarsely.
    'offset_band': {'type': float, 'min': 1.0, 'default': 1e5},
    # A channel whose standard deviation exceeds tail_band interquartile ranges warns
    # that its tails are heavy.
    'tail_band': {'type': float, 'min': 0.0, 'default': 1.5},
    # A channel with more than tie_share of its values equal to one value warns, as
    # padded or sparse rows do.
    'tie_share': {'type': float, 'min': 0.0, 'default': 0.9},
    # A call with fewer than small_sample rows reaching the estimator warns that the
    # dataset is small (read from Training(...)).
    'small_sample': {'type': int, 'min': 0, 'default': 200},
    # Training stops with TrainingError after nan_streak epochs in a row whose held-out
    # evaluation is NaN.
    'nan_streak': {'type': int, 'min': 1, 'default': 3},

    # Mixed-precision (AMP) training.
    # 'auto', enable on CUDA, no-op on CPU/MPS (safe default).
    # True, explicitly enable (CUDA only; silently no-ops on other devices).
    # False, explicitly disable.
    'use_amp': {'type': (bool, str), 'default': 'auto'},

    # Memory / device layout
    # 'cpu', store dataset tensors on CPU (default; safe for long sweeps).
    # 'auto', store on the compute device (faster repeated evaluation, e.g. precision mode).
    # Any explicit device string is also accepted.
    # Precision mode overrides this to 'auto' unless the user sets it explicitly.
    'dataset_device': {'type': (str, type(None)), 'default': 'cpu'},
}

# The settings by which the trainer judges the held-out curve and warns
# (neural_mi/training/trainer.py, neural_mi/training/epochs.py). Each is a field of
# Training(...).
CURVE_SETTINGS = ('level_noise', 'start_noise', 'resolved_noise', 'fast_epochs', 'fall_noise', 'top_se',
                  'band_share', 'clear_se', 'rising_epochs', 'rising_noise', 'ceiling_fraction',
                  'saturation_fraction', 'saturated_epoch_fraction', 'gap_share', 'nan_streak')

# The settings of the data check (neural_mi/preprocessing.py). Each is a field of
# Preprocessing(...).
CHECK_SETTINGS = ('scale_band', 'predicted_share', 'check_rows', 'sample_rows', 'spread_band', 'offset_band',
                  'tail_band', 'tie_share')

# The settings that a call reads while it prepares its data, before the configurations of a grid exist.
# input_scaling decides how the encoders scale each stream, from the kind of rows the stream gives.
# shift_windows decides how the rows are built: once for the whole call, or by each task so that they can
# move. A grid over one of these runs one preparation per value, as a grid over processor parameters does,
# so that every configuration computes what a separate call with its settings computes (INTERNALS.md).
PREPARATION_SETTINGS = ('input_scaling', 'shift_windows')

# How input_scaling='auto' scales the rows of each kind of stream (stream_scaling), one scale per
# unit. The channels of a continuous stream, or of rows passed in built already, can each be a
# different quantity: 'channel' centres and scales each by its own mean and spread. Every channel of
# a spike stream shares one unit, so it takes one scale for the whole stream and quiet neurons stay
# quiet: 'counts' centres each channel of spike counts and divides by the spread of the whole
# stream, and 'times' divides spike times by their root mean square and leaves the padding where it
# is. A categorical stream's encoding is one-hot or probabilities, with no unit to remove, and rows
# that mix streams of different kinds are left as they are.
SCALING_OF_KIND = {'continuous': 'channel', 'spike_counts': 'counts', 'spike_times': 'times', 'categorical': None,
                   'mixed': None}

# The processor settings' defaults, by processor. ``None`` means unset: a
# window_size has to be given, a step_size of None is one window, and so on.
PROCESSOR_DEFAULTS = {
    'continuous': {'window_size': None, 'step_size': None, 'min_coverage_fraction': 0.2,
                   'sample_rate': None, 'period_tolerance': 0.05, 'padded_share': 0.1, 'gap_multiple': 3.0,
                   'bridged_share': 0.3},
    'spike': {'window_size': None, 'step_size': None, 'max_spikes_per_window': None, 'n_seconds': None,
              'sample_rate': None, 'no_spike_value': 0.0, 'bin_size': None, 'normalize_bins': False,
              'exclude_bursty_neurons': False, 'burst_threshold_multiplier': 5.0,
              'drop_empty_windows': True},
    'categorical': {'window_size': None, 'step_size': None, 'sample_rate': None,
                    'min_coverage_fraction': 0.2, 'encoding': 'majority_vote', 'max_encoding_width': 500},
}


# The settings each processor takes.
PROCESSOR_PARAMS_SCHEMA = {processor: list(defaults) for processor, defaults in PROCESSOR_DEFAULTS.items()}


def stream_processing(processor: Optional[str], params: Optional[Dict[str, Any]], x_processor: Optional[str],
                      x_params: Optional[Dict[str, Any]], data: Any = None
                      ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The processor and processor settings a second stream, Y or W, is read with: one rule for both.

    A stream without a processor of its own reads with X's. One without
    settings of its own reads with X's settings, kept to the keys its processor
    takes. Settings it was given are its own and are not merged with X's. A
    stream given as rows built already (``data`` with three dimensions) has no
    raw series for a processor to read and inherits nothing.
    """
    if processor is None and getattr(data, 'ndim', None) == 3:
        return None, params
    processor = processor if processor is not None else x_processor
    if params is None and processor is not None:
        accepted = PROCESSOR_DEFAULTS.get(processor)
        params = {k: v for k, v in (x_params or {}).items() if accepted is None or k in accepted}
    return processor, params


def processor_setting(processor: str, params: Dict[str, Any], name: str) -> Any:
    """Processor setting ``name`` from ``params``, or the schema's default for ``processor`` when
    ``params`` lacks it."""
    params = params or {}
    return params[name] if name in params else copy.copy(PROCESSOR_DEFAULTS[processor][name])


def default_of(name: str) -> Any:
    """The schema's default for setting ``name``: the one place a default is written. A mutable default comes
    back as a copy, so a caller cannot change it for every other caller."""
    return copy.copy(BASE_PARAMS_SCHEMA[name]['default'])


def setting(params: Dict[str, Any], name: str) -> Any:
    """Setting ``name`` from ``params``, or its schema default when ``params`` lacks it."""
    return params[name] if name in params else default_of(name)


def stream_kind_of(processor: Optional[str], params: Optional[Dict[str, Any]] = None) -> str:
    """The kind of rows a stream gives its encoder, as :data:`SCALING_OF_KIND` names them: a spike
    processor builds counts when it bins (``bin_size``) and spike times otherwise, and a stream with no
    processor, or a continuous one, gives continuous rows."""
    if processor == 'spike':
        return 'spike_counts' if (params or {}).get('bin_size') is not None else 'spike_times'
    if processor == 'categorical':
        return 'categorical'
    return 'continuous'


def stream_scaling(input_scaling: Optional[str], kind: Optional[str]) -> Optional[str]:
    """How an encoder scales the rows of a stream of ``kind`` (:func:`stream_kind_of`): under
    ``input_scaling='auto'`` as :data:`SCALING_OF_KIND` says, and under ``'none'`` not at all."""
    if input_scaling != 'auto':
        return None
    return SCALING_OF_KIND[kind or 'continuous']


def scaled_streams(params: Dict[str, Any], kinds: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """How the encoders scale each stream of ``kinds``, a stream's name to its kind, under the
    ``input_scaling`` of ``params`` (:func:`stream_scaling`)."""
    mode = setting(params, 'input_scaling')
    return {name: stream_scaling(mode, kind) for name, kind in kinds.items()}


def side_scaling(params: Dict[str, Any], x_streams=('x',), y_stream: str = 'y'
                 ) -> Tuple[Tuple[Optional[str], ...], Optional[str]]:
    """How a network scales each part of its X input, read from the streams ``x_streams``, and its Y
    input, read from ``y_stream``: as :func:`neural_mi.run.run` decided for every stream
    (``_scaled_streams``), or else as the processors in ``params`` give, Y reading with X's when it has
    none of its own."""
    decided = params.get('_scaled_streams')
    if decided is None:
        kind_x = stream_kind_of(params.get('processor_type_x'), params.get('processor_params_x'))
        kind_y = (stream_kind_of(params.get('processor_type_y'), params.get('processor_params_y'))
                  if params.get('processor_type_y') else kind_x)
        decided = scaled_streams(params, {'x': kind_x, 'w': kind_x, 'y': kind_y})

    def scaling(names) -> Optional[str]:
        # A part built from several streams (a conditioning past of Y and W) is scaled as they are when they
        # agree, and left as it is when they do not.
        modes = {decided.get(name, decided.get('x')) for name in ((names,) if isinstance(names, str) else names)}
        return modes.pop() if len(modes) == 1 else None
    return tuple(scaling(names) for names in x_streams), scaling(y_stream)


def norm_layer_of(params: Dict[str, Any]) -> Optional[str]:
    """The normalisation layer the networks build: ``norm_layer``, with 'auto' and 'none' read as none.

    A mode that wants another reading of 'auto' sets norm_layer before anything is built, as
    mode='dimensionality' does for its hybrid critic."""
    norm = setting(params, 'norm_layer')
    return None if norm in ('auto', 'none') else norm
