# neural_mi/run.py
"""Provides the main `run` function, the primary entry point for the library.

This module orchestrates the entire analysis pipeline, from data validation
and preprocessing to model training and results aggregation. The `run` function
acts as a unified interface for all supported analysis modes.
"""
import glob
import os
import time
import warnings
import numpy as np
import torch
from typing import Union, Optional, Dict, Any, List
import random

from .analysis.task import _decoder_lambda
from .analysis.assemble import build_results
from collections import OrderedDict
from dataclasses import fields
from .data.handler import create_dataset
from .data.shift_windowing import shift_family, mixed_pair_sample_rate_ok
from .results import Results
from .validation import ParameterValidator, DataValidator
from .utils import get_device
from .logger import call_verbosity, collect_repeats, logger, user_stacklevel
from .embeddings_io import model_file, resolve_model_path, warn_saving_several
from .defaults import BASE_PARAMS_SCHEMA, MODE_KWARGS_SCHEMA, PROCESSOR_PARAMS_SCHEMA
import inspect as _inspect
from .config import (
    Model, Training, Split, Estimator, Output, Processing,
    Rigorous, Precision, Lag, Transfer, Dimensionality, Conditional,
    Interaction, Pairwise, as_config,
)

# Mode name -> its dedicated config class (modes not listed take no mode config).
_MODE_CONFIG_CLASSES = {
    'rigorous': Rigorous, 'precision': Precision, 'lag': Lag,
    'transfer': Transfer, 'dimensionality': Dimensionality, 'conditional': Conditional,
    'interaction': Interaction, 'pairwise': Pairwise,
}

# Which modes window shifting reaches. Shifting needs the raw, unwindowed data
# to survive until each training task, and it must not disturb a comparison
# between runs.
#
# These modes train independent runs from raw data, or compare only a frozen,
# pre-shift view (dimensionality, precision), so both mechanisms apply.
_SHIFT_SAFE_MODES = ('estimate', 'sweep', 'pairwise', 'dimensionality', 'precision')
# shift_windows (the regular-grid family) also reaches 'rigorous', which
# translates each gamma chunk into a raw sample range before windowing it, and
# 'lag', whose lag-shifted data is windowed inside each task. 'rigorous' stays
# out of _SHIFT_SAFE_MODES because that tuple also admits spike+regular pairs
# to shift_time, and a rigorous chunk cannot be cut as a sample range on one
# side and a time range on the other.
_SHIFT_WINDOWS_SAFE_MODES = _SHIFT_SAFE_MODES + ('rigorous', 'lag')
# shift_time also reaches 'conditional' and 'interaction'. Their terms are
# separate training runs, and the Trainer draws every term's shifts from a
# generator seeded with the run's shared seed, so all terms land on the same
# offset at the same epoch and the difference stays paired. They stay out of
# _SHIFT_SAFE_MODES, which feeds _SHIFT_WINDOWS_SAFE_MODES: a regular X/Y with a
# spike W would then defer windowing without the raw-deferred path that builds W.
_SHIFT_TIME_SAFE_MODES = _SHIFT_SAFE_MODES + ('conditional', 'interaction')
# shift_time reaches 'rigorous' for a spike+spike pair, whose chunks are cut as
# time ranges on both sides.
_SHIFT_TIME_RIGOROUS_SAFE_MODES = _SHIFT_TIME_SAFE_MODES + ('rigorous',)


_MODES = ('estimate', 'sweep', 'rigorous', 'lag', 'precision', 'conditional',
          'interaction', 'transfer', 'pairwise', 'dimensionality')

# Rigorous(...) settings that run_rigorous_analysis takes as keywords, beside the
# three that _run_flat names (curvature_t_threshold, min_gamma_points,
# confidence_level).
_RIGOROUS_FIT_KEYS = ('gamma_range', 'residual_threshold', 'leverage_threshold',
                      'temporal_chunking')

# Modes that window inside each training task, so a processor parameter in their
# grid takes effect there. Every other mode prepares its data once per call and
# runs a processor grid one setting at a time (_run_processor_grid).
_WINDOWS_IN_TASK = ('sweep', 'lag')

_PROCESSOR_KEYS = frozenset().union(*PROCESSOR_PARAMS_SCHEMA.values())


def _to_2d(t):
    """``(T, C, 1)`` back to ``(T, C)``, the shape transfer entropy builds its histories from."""
    if hasattr(t, 'ndim') and t.ndim == 3 and t.shape[-1] == 1:
        return t.reshape(t.shape[0], t.shape[1]).contiguous()
    return t


def _scalar_rigorous_kwargs(analysis_kwargs: dict, curvature_t_threshold: float,
                            min_gamma_points: int, confidence_level: float) -> dict:
    """The fit settings of a difference quantity run with ``rigorous=True``."""
    return {
        'gamma_range': analysis_kwargs.get('gamma_range') or range(1, 11),
        'curvature_t_threshold': analysis_kwargs.get('curvature_t_threshold', curvature_t_threshold),
        'min_gamma_points': analysis_kwargs.get('min_gamma_points', min_gamma_points),
        'confidence_level': analysis_kwargs.get('confidence_level', confidence_level),
        'residual_threshold': analysis_kwargs.get('residual_threshold', 2.5),
        'leverage_threshold': analysis_kwargs.get('leverage_threshold', 0.20),
    }


def _stream_processors(mode, processor_type_x, processor_type_y, w_processor_type, w_data) -> dict:
    """Each stream's parameter slot, mapped to the processor that reads it."""
    streams = {'processor_params_x': processor_type_x,
               'processor_params_y': (processor_type_y if processor_type_y is not None
                                      else processor_type_x)}
    if mode in ('conditional', 'interaction', 'transfer') and w_data is not None:
        w_type = w_processor_type
        if (w_type is None and mode != 'transfer'
                and getattr(w_data, 'ndim', None) != 3):
            # A W without a processor of its own is windowed with X's.
            w_type = processor_type_x
        streams['w_processor_params'] = w_type
    return streams


# Config fields that the parameter schema, and so a sweep_grid, knows by another name.
_GRID_NAMES = {'mode': 'split_mode', 'gap_fraction': 'split_gap_fraction',
               'name': 'estimator_name', 'params': 'estimator_params'}


def _normalise_grid(sweep_grid: Optional[dict]) -> Optional[dict]:
    """Every value of `sweep_grid` as a list of the values to run.

    An iterable (list, tuple, range, array, generator) gives one configuration
    per item and is read once, here. A string, a dict or a scalar is one fixed
    value; iterating it would split 'mlp' into letters or a dict into its keys.
    """
    if sweep_grid is None:
        return None
    if not isinstance(sweep_grid, dict):
        raise TypeError(
            f"sweep_grid must be a dict of setting names to values, got "
            f"{type(sweep_grid).__name__}."
        )
    grid = {}
    for key, values in sweep_grid.items():
        if isinstance(values, (str, bytes, dict)) or not hasattr(values, '__iter__'):
            grid[key] = [values]
        elif hasattr(values, 'tolist'):
            grid[key] = list(values.tolist()) if getattr(values, 'ndim', 1) else [values.tolist()]
        else:
            grid[key] = list(values)
        if not grid[key]:
            raise ValueError(
                f"sweep_grid['{key}'] holds no values and leaves the grid no configuration "
                f"to run. Give it at least one value or remove the key."
            )
    return grid


def _check_grid_keys(mode: str, sweep_grid: Optional[dict]) -> None:
    """Refuse a sweep_grid key that would leave every configuration the same.

    A grid varies the settings of Model, Training, Split, Estimator and the
    processors. A mode's own settings are read once from its config, so a grid
    over one of them, or over a name no setting has, would run identical
    configurations and label them as different.
    """
    import dataclasses
    mode_keys = set(MODE_KWARGS_SCHEMA.get(mode, {})) - {'n_workers'}
    if mode in _MODE_CONFIG_CLASSES:
        mode_keys |= {f.name for f in dataclasses.fields(_MODE_CONFIG_CLASSES[mode])}
    for key in sweep_grid or {}:
        if key == 'run_id' or key in _PROCESSOR_KEYS:
            continue
        # A network setting wins over a mode setting of the same name, such as
        # Model(bidirectional=...) for a recurrent encoder in mode='transfer'.
        if key in BASE_PARAMS_SCHEMA and key != 'output_units':
            continue
        if key in mode_keys:
            raise ValueError(
                f"mode='{mode}' reads '{key}' once from its config. A sweep_grid over it "
                f"would run every configuration with the same value. Run one call per value "
                f"or use a named quantity whose own parameter takes a list (such as "
                f"transfer_entropy(history_window=[...]))."
            )
        if key == 'output_units':
            raise ValueError(
                "sweep_grid varies 'output_units'. That setting fixes the units of the whole "
                "result. Set Output(units=...) and leave it out of the grid."
            )
        hint = (f" That config field is swept under the name '{_GRID_NAMES[key]}'."
                if key in _GRID_NAMES else "")
        raise ValueError(
            f"NeuralMI reads no setting named '{key}'. A sweep_grid over it would run the "
            f"same call in every configuration.{hint} PARAMETERS.md lists the settings a "
            f"grid can vary."
        )


# Settings that only take effect with particular encoders, read the way
# utils.build_critic routes them to each side's model.
_ENCODER_SETTINGS = {'kernel_size': ('cnn', 'cnn2d', 'tcn'), 'bidirectional': ('gru', 'lstm'),
                     'nhead': ('transformer',), 'pytorch_predefined': ('pretrained_backbone',),
                     'pretrained': ('pretrained_backbone',), 'branch_model': ('dual_branch',)}
_HEAD_SETTINGS = ('n_layers_head', 'hidden_dim_head')
_DECODER_SETTINGS = ('decoder_lambda', 'decoder_lambda_x', 'decoder_lambda_y',
                     'decoder_output_activation_x', 'decoder_output_activation_y')


def _rotation_settings_ignored(output: Optional[Output]) -> List[str]:
    """The rotation settings in ``output`` that nothing in this call reads.

    The final embeddings are rotated when they are returned, and the per-epoch
    history when it is tracked.
    """
    if output is None:
        return []
    returned, tracked = bool(output.return_embeddings), bool(output.track_embeddings)
    rotated, per_epoch = bool(output.return_rotated_embeddings), bool(output.rotated_embeddings_per_epoch)
    ignored = []
    if rotated and not (returned or tracked):
        ignored.append("return_rotated_embeddings (needs return_embeddings or track_embeddings)")
    if per_epoch and not (rotated and tracked):
        ignored.append("rotated_embeddings_per_epoch (needs track_embeddings and "
                       "return_rotated_embeddings)")
    if output.return_rotation_matrices and not rotated:
        ignored.append("return_rotation_matrices (needs return_rotated_embeddings)")
    return ignored


def _warn_ineffective_settings(mode: str, model: Optional[Model], training: Optional[Training],
                               output: Optional[Output] = None) -> None:
    """Warn about Model, Training and Output settings that this call would ignore.

    Only values the caller set away from their default count, so a setting
    that is spelled out at its default does not warn.
    """
    def set_away(cfg, name):
        value = getattr(cfg, name, None) if cfg is not None else None
        return value is not None and value != BASE_PARAMS_SCHEMA.get(name, {}).get('default')

    chosen = {f.name for f in fields(model) if set_away(model, f.name)} if model is not None else set()
    ignored = []
    if model is not None and model.custom_critic is not None:
        others = sorted(chosen - {'custom_critic'})
        if others:
            ignored.append(f"{', '.join(others)} (custom_critic is used as given)")
    else:
        custom = model is not None and (model.custom_embedding_cls is not None
                                        or model.custom_embedding_cls_y is not None)
        x_model = (model.embedding_model if model is not None else None) or 'mlp'
        y_model = (model.embedding_model_y if model is not None else None) or x_model
        if not custom:
            for name, encoders in _ENCODER_SETTINGS.items():
                if name in chosen and not {x_model, y_model} & set(encoders):
                    names = ', '.join(f"'{e}'" for e in encoders)
                    ignored.append(f"{name} (applies to embedding_model {names})")
        critic = (model.critic_type if model is not None else None) or (
            'hybrid' if mode == 'dimensionality' else 'separable')
        head = [n for n in _HEAD_SETTINGS if n in chosen]
        if set_away(training, 'lr_head_multiplier'):
            head.append('lr_head_multiplier')
        if head and critic != 'hybrid':
            ignored.append(f"{', '.join(head)} (critic_type='hybrid' only)")
        if 'beta' in chosen and not (model.use_variational or False):
            ignored.append("beta (use_variational=True only)")
        decoder = [n for n in _DECODER_SETTINGS if n in chosen]
        if decoder and not (model.use_decoder or False):
            ignored.append(f"{', '.join(decoder)} (use_decoder=True only)")
    if mode == 'dimensionality' and 'embedding_dim' in chosen:
        ignored.append("embedding_dim (mode='dimensionality' sets it from "
                       "Dimensionality(embedding_dims=...))")
    ignored += _rotation_settings_ignored(output)
    if ignored:
        warnings.warn(
            f"These settings have no effect in this call and are ignored: {'; '.join(ignored)}.",
            UserWarning, stacklevel=user_stacklevel(),
        )


def _check_estimator_params(estimator: Optional[Estimator]) -> None:
    """Refuse estimator parameters the chosen estimator does not take, before any work."""
    from .estimators import ESTIMATORS
    if estimator is None or not estimator.params:
        return
    name = estimator.name or BASE_PARAMS_SCHEMA['estimator_name']['default']
    bound = ESTIMATORS.get(name)
    if bound is None:
        return  # an unknown name is refused with the other settings
    takes = [p for p in _inspect.signature(bound).parameters if p != 'scores']
    unknown = sorted(set(estimator.params) - set(takes))
    if unknown:
        allowed = ', '.join(repr(p) for p in takes) or 'no parameters'
        raise ValueError(
            f"Estimator(params=...) names {unknown}. The '{name}' estimator takes {allowed}."
        )


# Why a mode's networks train on rows other than the ones the caller passed,
# which custom split indices would then index wrongly.
_ROWS_DIFFER = {'rigorous': 'trains on chunks of the data',
                'lag': 'trains on copies of the data shifted by each lag',
                'transfer': 'trains on histories built from the data'}


def _check_custom_split(mode: str, train_indices, test_indices, processed: bool,
                        analysis_kwargs: dict) -> None:
    """Refuse custom split indices wherever they would index the wrong rows.

    The indices address the rows the network trains on. They mean what the
    caller intends only when those rows are the ones the caller passed.
    """
    if (train_indices is None) != (test_indices is None):
        raise ValueError(
            "Split(train_indices=...) and Split(test_indices=...) are used together. "
            "Pass both or neither."
        )
    reason = _ROWS_DIFFER.get(mode)
    if reason is None and analysis_kwargs.get('rigorous'):
        reason = 'trains on chunks of the data with rigorous=True'
    if reason is None and mode == 'dimensionality' and analysis_kwargs.get('lag'):
        reason = 'shifts one half of the data in time by lag'
    if reason is None and processed:
        reason = 'windows the data first and trains on the windows it builds'
    if reason is not None:
        raise ValueError(
            f"Split(train_indices=..., test_indices=...) indexes the rows the network "
            f"trains on. mode='{mode}' {reason}. The indices would then address rows other "
            f"than the ones you passed. Pass the prepared rows as the data or split with "
            f"Split(mode=...) and Split(train_fraction=...)."
        )


def _processor_grid(mode: str, sweep_grid: Optional[dict], streams: dict) -> dict:
    """The processor parameters of `sweep_grid` that need one data preparation each.

    Raises when the grid varies a processor parameter that no stream's
    processor reads.
    """
    if mode in ('estimate', 'precision') or not sweep_grid:
        return {}
    keys = [k for k in sweep_grid if k in _PROCESSOR_KEYS]
    for key in keys:
        if not any(key in PROCESSOR_PARAMS_SCHEMA.get(proc, ()) for proc in streams.values()):
            takers = [name for name, accepted in PROCESSOR_PARAMS_SCHEMA.items() if key in accepted]
            raise ValueError(
                f"sweep_grid varies the processor parameter '{key}'. No stream in this call "
                f"has a processor that reads it. Set Processing(x=...) to a processor that "
                f"takes '{key}' ({', '.join(repr(n) for n in takers)}) or remove '{key}' "
                f"from the grid."
            )
    if mode in _WINDOWS_IN_TASK:
        return {}
    return {k: sweep_grid[k] for k in keys}


def _trains_one_network(mode: str, n_configs: int, n_repeats: int) -> bool:
    if mode in ('estimate', 'precision'):
        return True
    if mode == 'sweep':
        return n_configs * n_repeats == 1
    return False


def _announce_call(mode: str, sweep_grid: Optional[dict], permutation_test: bool,
                   n_permutations: int, analysis_kwargs: dict, has_y: bool = True,
                   save_path: Optional[str] = None) -> None:
    """Messages about the cost of the whole call, given once per call."""
    from .analysis.assemble import split_grid
    used_grid = None if mode in ('estimate', 'precision') else sweep_grid
    configs, run_ids = split_grid(used_grid)
    n_configs = len(configs)
    if permutation_test and mode in _PERMUTABLE_MODES and (has_y or mode != 'pairwise'):
        across = (f" with all {n_configs} configurations of sweep_grid"
                  if n_configs > 1 else "")
        cost = (f"Each permutation reruns the whole call{across}. The test costs "
                f"{n_permutations} times the call itself.")
        if n_permutations < 100:
            warnings.warn(
                f"With n_permutations={n_permutations} the smallest p-value the permutation "
                f"test can report is 1/{n_permutations + 1} = {1 / (n_permutations + 1):.2g}. "
                f"A reliable p-value usually needs 100 or more permutations. {cost}",
                UserWarning, stacklevel=user_stacklevel(),
            )
        elif n_configs > 1:
            warnings.warn(cost, UserWarning, stacklevel=user_stacklevel())
        else:
            logger.info(cost)
    if save_path and not _trains_one_network(mode, n_configs, len(run_ids)):
        warn_saving_several(save_path)
    n_workers = analysis_kwargs.get('n_workers') or 1
    if (n_workers > 1 and not permutation_test
            and _trains_one_network(mode, n_configs, len(run_ids))):
        warnings.warn(
            f"n_workers={n_workers} has no effect here. This call trains one network and "
            f"has nothing to run in parallel.",
            UserWarning, stacklevel=user_stacklevel(),
        )


# Modes whose result a permutation test can be built for.
_PERMUTABLE_MODES = ('estimate', 'sweep', 'lag', 'conditional', 'interaction', 'transfer',
                     'pairwise')


def _run_processor_grid(call_args: dict, proc_grid: dict, streams: dict) -> Results:
    """Run a grid that varies processor parameters.

    The data are prepared once per processor setting, and each preparation runs
    the rest of the grid. The parts merge into one result whose configurations
    follow the full grid.
    """
    from .analysis.assemble import merge_results, split_grid
    args = dict(call_args)
    analysis_kwargs = args.pop('analysis_kwargs')
    full_grid = dict(args['sweep_grid'])
    inner = {k: v for k, v in full_grid.items() if k not in proc_grid} or None
    w_inherits = args.get('w_processor_type') is None
    parts = []
    for fixed in split_grid(proc_grid)[0]:
        part = dict(args, sweep_grid=inner, _grid_part=fixed)
        if args.get('save_best_model_path'):
            part['save_best_model_path'] = model_file(
                {'save_best_model_path': args['save_best_model_path'], '_model_labels': fixed})
        x_params = {**(args.get('processor_params_x') or {}),
                    **_accepted(fixed, streams['processor_params_x'])}
        part['processor_params_x'] = x_params
        for slot in ('processor_params_y', 'w_processor_params'):
            if slot not in streams:
                continue
            values = _accepted(fixed, streams[slot])
            own = args.get(slot)
            if own is not None:
                part[slot] = {**own, **values}
            elif slot == 'w_processor_params' and not w_inherits:
                part[slot] = values or None
            elif any(k not in x_params for k in values):
                # Reads X's parameters, and needs one X does not take.
                part[slot] = {**x_params, **values}
        parts.append((fixed, _run_flat(**part, **analysis_kwargs)))
    params = dict(parts[0][1].params)
    for key in ('processor_params_x', 'processor_params_y'):
        params[key] = args.get(key)
    params['base_params'] = {**params.get('base_params', {}), **proc_grid}
    return merge_results(parts, full_grid, params)


def _accepted(values: dict, processor: Optional[str]) -> dict:
    accepted = PROCESSOR_PARAMS_SCHEMA.get(processor, ())
    return {k: v for k, v in values.items() if k in accepted}


def _reshape_categorical_w_for_conditional(w_run_data, cat_dataset):
    """Re-lay-out a categorical-processor W tensor for ``mode='conditional'``.

    ``mode='conditional'`` builds XW by concatenating X and W along the channel axis, so both must share X's window-size axis. The
    categorical processor's encodings don't produce that layout natively:

    - ``'majority_vote'`` / ``'probability'`` collapse each window to a
      single per-category summary, shape ``(N, C, n_categories)``: W has no
      temporal extent within a window by construction. Folded here into
      ``C * n_categories`` channels with a size-1 window axis; the caller
      broadcasts that axis against X's window size.
    - ``'full_trajectory'`` keeps full per-timepoint resolution but flattens
      ``n_categories * window_size`` onto the last axis, shape
      ``(N, C, n_categories * window_size)``. Un-flattened and folded here
      into ``C * n_categories`` channels with the real window axis restored,
      preserving the per-timepoint information.

    Only reshapes the tensor handed to this specific call; the categorical
    processor's own stored data and its behavior in every other mode are
    untouched. The reshape itself lives in
    :func:`~neural_mi.data.shift_windowing.categorical_to_channel_layout`,
    shared with the shifted route.
    """
    from .data.shift_windowing import categorical_to_channel_layout
    # Shared with the shifted route, which has to do the same re-layout before
    # it can concatenate a categorical W onto X. One definition, so the two
    # routes cannot drift on the un-flatten order.
    return categorical_to_channel_layout(
        w_run_data, cat_dataset.n_categories, cat_dataset.encoding)


def run(
    x_data: Union[np.ndarray, torch.Tensor, List],
    y_data: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    *,
    mode: str = 'estimate',
    processing: Optional[Union[Processing, Dict[str, Any]]] = None,
    model: Optional[Union[Model, Dict[str, Any]]] = None,
    training: Optional[Union[Training, Dict[str, Any]]] = None,
    split: Optional[Union[Split, Dict[str, Any]]] = None,
    estimator: Optional[Union[Estimator, str, Dict[str, Any]]] = None,
    output: Optional[Union[Output, Dict[str, Any]]] = None,
    sweep_grid: Optional[Dict[str, list]] = None,
    rigorous: Optional[Union[Rigorous, Dict[str, Any]]] = None,
    precision: Optional[Union[Precision, Dict[str, Any]]] = None,
    lag: Optional[Union[Lag, Dict[str, Any]]] = None,
    transfer: Optional[Union[Transfer, Dict[str, Any]]] = None,
    dimensionality: Optional[Union[Dimensionality, Dict[str, Any]]] = None,
    conditional: Optional[Union[Conditional, Dict[str, Any]]] = None,
    interaction: Optional[Union[Interaction, Dict[str, Any]]] = None,
    pairwise: Optional[Union[Pairwise, Dict[str, Any]]] = None,
    n_workers: int = 1,
    seed: Optional[int] = None,
    verbose: Optional[bool] = None,
    show_progress: bool = True,
    device: Optional[str] = None,
    permutation_test: bool = False,
    n_permutations: int = 10,
    permutation_shuffle: str = 'circular',
    **_removed: Any,
) -> Results:
    """Unified entry point for all NeuralMI analyses (config-based API).

    Parameters are grouped into a small set of typed config objects (see
    :mod:`neural_mi.config`). Every config is optional. Omitted configs and unset fields fall back to the defaults in
    :data:`neural_mi.defaults.BASE_PARAMS_SCHEMA`. Anywhere a config is accepted
    a plain ``dict`` with the same keys works too, so importing the classes is
    optional.

    Parameters
    ----------
    x_data, y_data : array-like
        Input data for variables X and Y. ``y_data`` is required for all modes
        except ``'dimensionality'``/``'pairwise'`` (self-pairwise). With
        ``processing=Processing(x='continuous'|'categorical', ...)``, raw arrays
        are shape ``(n_timepoints, n_channels)`` (a 1-D array is treated as
        ``(n_timepoints, 1)``). With ``processing=Processing(x='spike', ...)``,
        pass a list of 1-D arrays of spike times, one per channel/neuron.
        Already-processed data (``processing=None``) is shape
        ``(n_samples, n_channels, window_size)`` (3-D) or ``(n_samples, n_channels)``
        (2-D, treated as a trailing window size of 1).
    mode : {'estimate','sweep','rigorous','dimensionality','lag','precision','conditional','interaction','transfer','pairwise'}
        The analysis to run.
    processing : Processing or dict, optional
        Raw-data processors, e.g. ``Processing(x='continuous', x_params={'window_size': 1})``.
    model : Model or dict, optional
        Architecture, e.g. ``Model(embedding_dim=16, hidden_dim=64, critic_type='separable')``.
    training : Training or dict, optional
        Optimisation loop, e.g. ``Training(n_epochs=50, learning_rate=1e-3, batch_size=128)``.
    split : Split or dict, optional
        Splitting strategy, e.g. ``Split(mode='random')``.
    estimator : Estimator, str, or dict, optional
        MI estimator. Accepts a bare name (``estimator='smile'``) or
        ``Estimator(name='smile', params={'clip': 5.0})``.
    output : Output or dict, optional
        Units, spectral tracking, embedding returns, and display labels.
    sweep_grid : dict, optional
        Setting names mapped to the values to run. A list, tuple, range or
        array gives one configuration per value and a single value fixes the
        setting. Every combination of values is one configuration and
        ``run_id`` repeats each one. Every mode except ``'estimate'`` and
        ``'precision'`` accepts it.
    rigorous, precision, lag, transfer, dimensionality, conditional, interaction, pairwise : mode config or dict, optional
        Mode-specific parameters; only the one matching ``mode`` is used. E.g.
        ``rigorous=Rigorous(confidence_level=0.68)``,
        ``precision=Precision(tau_grid=[...])``,
        ``transfer=Transfer(history_window=10)`` (or ``Transfer(history_window=10, w_data=w)`` for conditional TE),
        ``conditional=Conditional(w_data=w)``,
        ``interaction=Interaction(w_data=w)``,
        ``pairwise=Pairwise(pairs=[(0, 1), (0, 2)])``.
    n_workers : int, default=1
        Worker processes for parallelisable modes.
    seed : int, optional
        Random seed (``random``/``numpy``/``torch``). **Reproducible at any
        ``n_workers``.** Each parallel task re-seeds inside its own worker from
        ``seed`` plus a deterministic per-task key
        (``analysis/task.py::run_training_task``), so which worker runs which
        task, and in what order, does not affect the result. Verified
        bit-identical between ``n_workers=1`` and ``n_workers=3`` for the shared
        task path, ``mode='dimensionality'``'s fits over splits, sizes and restarts, and
        ``mode='pairwise'``'s per-pair dispatch.
    verbose : bool, optional
        True logs informational messages for this call and False only warnings
        and errors. None, the default, keeps the level set by
        ``nmi.set_verbosity()``.
    show_progress : bool
        Progress bars.
    device : str, optional
        Compute device ('cpu'/'cuda'/'mps'); auto-detected if None.
    permutation_test : bool, default=False
        Test every ``dataframe`` row against a null built by rerunning the call
        with X moved in time. The move breaks X's alignment with Y (and W) and
        leaves every other relation in place. Adds a ``p_value`` column and
        stores each row's null under ``details[config_id]['null_distribution']``.
    n_permutations : int, default=10
        Number of shifts when ``permutation_test=True``. Each one reruns the
        whole call. The smallest p-value reachable is ``1 / (n_permutations + 1)``,
        so a reliable p-value usually needs 100 or more.
    permutation_shuffle : {'circular', 'block'}, default='circular'
        How X is moved. ``'circular'`` shifts it by one random offset along
        its time axis, wrapping at the end, with offsets within 10% of the
        recording's length of zero excluded. ``'block'`` cuts it into
        contiguous blocks one window long and reorders them.

    Returns
    -------
    neural_mi.results.Results

    Examples
    --------
    >>> import neural_mi as nmi
    >>> from neural_mi import Model, Training, Split, Processing, Rigorous
    >>> results = nmi.run(
    ...     x_raw, y_raw, mode='rigorous',
    ...     processing=Processing(x='continuous', x_params={'window_size': 1}),
    ...     model=Model(embedding_dim=16, hidden_dim=64),
    ...     training=Training(n_epochs=50, batch_size=128),
    ...     split=Split(mode='random'),
    ...     rigorous=Rigorous(confidence_level=0.68),
    ...     n_workers=4, seed=42,
    ... )
    """
    if _removed:
        _per_mode = ', '.join(f"{name}={cls.__name__}(...)"
                              for name, cls in _MODE_CONFIG_CLASSES.items())
        raise TypeError(
            f"run() got unexpected keyword argument(s) {sorted(_removed)}. "
            f"Parameters are grouped into config objects: model=Model(...), "
            f"training=Training(...), split=Split(...), processing=Processing(...), "
            f"estimator=..., output=Output(...), and one config per mode: {_per_mode}. "
            f"See help(neural_mi.run)."
        )

    # Coerce dict/str inputs to config instances.
    model = as_config(model, Model)
    training = as_config(training, Training)
    split = as_config(split, Split)
    output = as_config(output, Output)
    processing = as_config(processing, Processing)
    if isinstance(estimator, str):
        estimator = Estimator(name=estimator)
    else:
        estimator = as_config(estimator, Estimator)
    _warn_ineffective_settings(mode, model, training, output)
    _check_estimator_params(estimator)

    # Named engine parameters (computed once at import, see _ENGINE_PARAMS) decide
    # each lowered key's bucket: a named engine kwarg vs the base_params dict /
    # analysis_kwargs.
    _named = _ENGINE_PARAMS

    base_params: Dict[str, Any] = {}
    flat: Dict[str, Any] = {}
    analysis_kwargs: Dict[str, Any] = {}

    def _route_base(d):
        for k, v in d.items():
            (flat if k in _named else base_params)[k] = v

    def _route_analysis(d):
        for k, v in d.items():
            (flat if k in _named else analysis_kwargs)[k] = v

    if model is not None:
        _route_base(model.to_base_params())
    if training is not None:
        _route_base(training.to_base_params())
    if split is not None:
        _route_base(split.to_base_params())
    if output is not None:
        _route_base(output.to_base_params())
        flat.update(output.to_labels())
    if estimator is not None:
        if estimator.name is not None:
            flat['estimator'] = estimator.name
        if estimator.params is not None:
            flat['estimator_params'] = estimator.params
    if processing is not None:
        _route_analysis(processing.to_kwargs())

    # Mode-specific config: only the one matching `mode` is consulted.
    _provided = {'rigorous': rigorous, 'precision': precision, 'lag': lag,
                 'transfer': transfer, 'dimensionality': dimensionality,
                 'conditional': conditional, 'interaction': interaction,
                 'pairwise': pairwise}
    _stray = [name for name, cfg in _provided.items() if cfg is not None and name != mode]
    if _stray:
        warnings.warn(
            f"Mode config(s) {_stray} were provided for a call with mode='{mode}' and are "
            f"ignored. Only the config of the active mode is used.",
            UserWarning, stacklevel=user_stacklevel(),
        )
    if mode in _MODE_CONFIG_CLASSES:
        mode_cfg = as_config(_provided[mode], _MODE_CONFIG_CLASSES[mode])
        if mode_cfg is not None:
            if isinstance(mode_cfg, Transfer):
                flat.update(mode_cfg.to_w_kwargs())
                ak = mode_cfg.to_analysis_kwargs()
                if 'bidirectional' in ak:
                    flat['bidirectional_te'] = ak.pop('bidirectional')
                _route_analysis(ak)
            elif isinstance(mode_cfg, Conditional):
                flat.update(mode_cfg.to_w_kwargs())
                _route_analysis(mode_cfg.to_analysis_kwargs())
            elif isinstance(mode_cfg, Interaction):
                flat.update(mode_cfg.to_w_kwargs())
                _route_analysis(mode_cfg.to_analysis_kwargs())
            else:
                _route_analysis(mode_cfg.to_analysis_kwargs())

    # Runtime / dispatch args (always forwarded).
    flat['mode'] = mode
    flat['sweep_grid'] = sweep_grid
    flat['random_seed'] = seed
    flat['verbose'] = verbose
    flat['show_progress'] = show_progress
    flat['device'] = device
    flat['permutation_test'] = permutation_test
    flat['n_permutations'] = n_permutations
    flat['permutation_shuffle'] = permutation_shuffle
    analysis_kwargs['n_workers'] = n_workers

    if base_params:
        flat['base_params'] = base_params

    save_path = resolve_model_path(flat.get('save_best_model_path'), mode)
    if save_path:
        flat['save_best_model_path'] = save_path
    started = time.time()
    with call_verbosity(verbose), collect_repeats():
        result = _run_flat(x_data, y_data, **flat, **analysis_kwargs)
    if save_path:
        _report_saved_networks(save_path, started)
    return result


def _report_saved_networks(save_path: str, started: float) -> None:
    """Say how many networks the call saved, and how much space they take."""
    root, ext = os.path.splitext(save_path)
    saved = [f for f in glob.glob(glob.escape(root) + '*' + ext)
             if os.path.getmtime(f) >= started - 1]
    if not saved:
        return
    size = sum(os.path.getsize(f) for f in saved) / 2 ** 20
    where = save_path if len(saved) == 1 else f"{root}_<labels>{ext}"
    logger.info(f"Saved {len(saved)} network(s) of {size:.1f} MB in total as {where}.")


def _run_flat(
    x_data: Union[np.ndarray, torch.Tensor, List],
    y_data: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    x_time: Optional[np.ndarray] = None,
    y_time: Optional[np.ndarray] = None,
    mode: str = 'estimate',
    processor_type_x: Optional[str] = None,
    processor_params_x: Optional[Dict[str, Any]] = None,
    processor_type_y: Optional[str] = None,
    processor_params_y: Optional[Dict[str, Any]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    sweep_grid: Optional[Dict[str, list]] = None,
    output_units: str = 'bits',
    estimator: str = 'infonce',
    estimator_params: Optional[Dict[str, Any]] = None,
    custom_critic: Optional[torch.nn.Module] = None,
    custom_embedding_cls: Optional[type] = None,
    save_best_model_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    verbose: Optional[bool] = None,
    show_progress: bool = True,
    device: Optional[str] = None,
    split_mode: str = 'blocked',
    train_fraction: float = 0.9,
    n_test_blocks: int = 5,
    split_gap_fraction: float = 0.5,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    curvature_t_threshold: float = 2.0,
    min_gamma_points: int = 5,
    confidence_level: float = 0.68,
    max_eval_samples: int = 5000,
    train_subset_size: Optional[int] = None,
    track_spectral_history: bool = False,
    max_index_reduction: float = 0.05,
    tau_grid: Optional[List[float]] = None,
    corrupt_target: str = 'x',
    corruption_method: str = 'rounding',
    n_noise_samples: int = 50,
    threshold_ratio: Union[float, List[float]] = 0.9,
    permutation_test: bool = False,
    n_permutations: int = 10,
    permutation_shuffle: str = 'circular',
    history_window: Optional[int] = None,
    prediction_horizon: int = 1,
    stride: int = 1,
    bidirectional_te: bool = False,
    w_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    w_time: Optional[np.ndarray] = None,
    w_processor_type: Optional[str] = None,
    w_processor_params: Optional[Dict[str, Any]] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    shared_encoder: Optional[bool] = None,
    return_embeddings: bool = False,
    lag_range: Optional[List] = None,
    use_spectral_norm: bool = True,
    gradient_clip_val: Optional[float] = None,
    optimizer: Union[str, type] = 'adam',
    optimizer_params: Optional[Dict[str, Any]] = None,
    scheduler: Union[str, type, None] = None,
    scheduler_params: Optional[Dict[str, Any]] = None,
    eval_train: Union[bool, float, int] = False,
    peak_fraction: float = 1.0,
    dropout: Optional[float] = None,
    norm_layer: Optional[str] = None,
    use_amp: Union[bool, str] = 'auto',
    track_embeddings: Optional[Union[bool, float, int, str]] = None,
    return_rotated_embeddings: Optional[bool] = None,
    whitening: Optional[str] = None,
    rotated_embeddings_per_epoch: Optional[bool] = None,
    return_rotation_matrices: Optional[bool] = None,
    channel_names_x: Optional[List[str]] = None,
    channel_names_y: Optional[List[str]] = None,
    _grid_part: Optional[Dict[str, Any]] = None,
    **analysis_kwargs
) -> Results:
    """Flat-kwarg engine behind :func:`run`; see that function's docstring for
    the public API and parameter semantics.

    ``_grid_part`` marks a call that runs one processor setting of a larger
    grid (see :func:`_run_processor_grid`); it skips the checks and messages
    that belong to the whole call.
    """
    sweep_grid = _normalise_grid(sweep_grid)
    _call_args = dict(locals())

    # run(verbose=True) logs INFO for this call and verbose=False only warnings
    # and errors. None keeps the level nmi.set_verbosity() chose.
    import logging as _logging
    from .data.handler import reset_retention_warnings as _reset_retention
    _reset_retention()  # dedup is per run, not per process lifetime
    _prev_level = logger.level
    _prev_handler_levels = [h.level for h in logger.handlers]
    if verbose is not None:
        target_level = _logging.INFO if verbose else _logging.WARNING
        logger.setLevel(target_level)
        for h in logger.handlers:
            h.setLevel(target_level)
    try:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if _grid_part is None:
            _check_grid_keys(mode, sweep_grid)
            _streams = _stream_processors(mode, processor_type_x, processor_type_y,
                                          w_processor_type, w_data)
            _proc_grid = _processor_grid(mode, sweep_grid, _streams)
            _announce_call(mode, sweep_grid, permutation_test, n_permutations,
                           analysis_kwargs, has_y=y_data is not None,
                           save_path=save_best_model_path)
            if _proc_grid:
                return _run_processor_grid(_call_args, _proc_grid, _streams)
        # Results do not depend on n_workers: run_training_task re-seeds
        # random/numpy/torch inside each worker from random_seed plus a
        # deterministic per-task key, so which worker runs a task, and when,
        # does not change it. Measured bit-identical at n_workers=1 and 3.

        if base_params is None: base_params = {}
        # Copy so we never mutate the caller's dict across multiple calls
        base_params = dict(base_params)

        # A Y with a processor of its own and no parameters reads with X's
        # parameters, kept to the keys its processor takes. W follows the same
        # rule further down, once its processor is resolved.
        if processor_type_y is not None and processor_params_y is None:
            _accepted_y = PROCESSOR_PARAMS_SCHEMA.get(processor_type_y, ())
            processor_params_y = {k: v for k, v in (processor_params_x or {}).items()
                                  if k in _accepted_y}

        def _inject(bp: dict, key: str, val) -> None:
            """Set bp[key] to val when val was given."""
            if val is not None:
                bp[key] = val

        # Explicit arguments go into base_params, where they are validated.
        # Time vectors travel with the params so that windowing done inside a
        # task sees the same clock create_dataset(x_time=...) sees here. Without
        # them a continuous X would be windowed in sample units while a spike Y
        # is in seconds, and no window could satisfy coverage.
        _inject(base_params, 'x_time', x_time)
        _inject(base_params, 'y_time', y_time)
        _inject(base_params, 'output_units', output_units)
        _inject(base_params, 'verbose', verbose)
        _inject(base_params, 'show_progress', show_progress)
        _inject(base_params, 'device', device)
        if 'device' not in base_params:
            base_params['device'] = get_device()
        _inject(base_params, 'estimator_name', estimator)
        # No `or {}` here: an unset argument must leave base_params alone, and
        # apply_defaults() fills the key when neither sets it.
        _inject(base_params, 'estimator_params', estimator_params)
        _inject(base_params, 'custom_critic', custom_critic)
        _inject(base_params, 'custom_embedding_cls', custom_embedding_cls)
        _inject(base_params, 'save_best_model_path', save_best_model_path)
        _inject(base_params, 'split_mode', split_mode)
        _inject(base_params, 'train_fraction', train_fraction)
        _inject(base_params, 'n_test_blocks', n_test_blocks)
        _inject(base_params, 'split_gap_fraction', split_gap_fraction)
        _inject(base_params, 'train_indices', train_indices)
        _inject(base_params, 'test_indices', test_indices)
        # Said here, where the caller's own indices arrive. The trainer also
        # receives indices the library chose itself (precision reuses one split
        # for its whole sweep, dimensionality shares one across its fits), and
        # those are no reason to warn.
        if train_indices is not None or test_indices is not None:
            _check_custom_split(mode, train_indices, test_indices,
                                processed=any(p is not None for p in
                                              (processor_type_x, processor_type_y, w_processor_type)),
                                analysis_kwargs=analysis_kwargs)
            logger.warning(
                "Custom train_indices and test_indices were provided. "
                "Split(mode, train_fraction, n_test_blocks, gap_fraction) "
                "will be ignored for this run."
            )
        # Trainer arguments
        _inject(base_params, 'max_eval_samples', max_eval_samples)
        _inject(base_params, 'train_subset_size', train_subset_size)
        _inject(base_params, 'use_spectral_norm', use_spectral_norm)
        _inject(base_params, 'gradient_clip_val', gradient_clip_val)
        _inject(base_params, 'optimizer', optimizer)
        # No `or {}`, as for estimator_params.
        _inject(base_params, 'optimizer_params', optimizer_params)
        _inject(base_params, 'scheduler', scheduler)
        _inject(base_params, 'scheduler_params', scheduler_params)
        _inject(base_params, 'eval_train', eval_train)
        _inject(base_params, 'peak_fraction', peak_fraction)
        _inject(base_params, 'dropout', dropout)
        _inject(base_params, 'norm_layer', norm_layer)
        _inject(base_params, 'use_amp', use_amp)

        _inject(base_params, 'track_spectral_history', track_spectral_history)
        _inject(base_params, 'max_index_reduction', max_index_reduction)

        _inject(base_params, 'processor_type_x', processor_type_x)
        _inject(base_params, 'processor_params_x', processor_params_x)
        _inject(base_params, 'processor_type_y', processor_type_y)
        _inject(base_params, 'processor_params_y', processor_params_y)
        if random_seed is not None:
            _inject(base_params, 'random_seed', random_seed)

        # Top-level shortcuts: inject into base_params
        _inject(base_params, 'n_epochs', n_epochs)
        _inject(base_params, 'batch_size', batch_size)
        _inject(base_params, 'shared_encoder', shared_encoder)
        if return_embeddings:
            base_params['return_embeddings'] = True
        _inject(base_params, 'track_embeddings', track_embeddings)
        _inject(base_params, 'return_rotated_embeddings', return_rotated_embeddings)
        _inject(base_params, 'whitening', whitening)
        _inject(base_params, 'rotated_embeddings_per_epoch', rotated_embeddings_per_epoch)
        _inject(base_params, 'return_rotation_matrices', return_rotation_matrices)

        if permutation_shuffle not in ('circular', 'block'):
            raise ValueError(
                f"permutation_shuffle must be 'circular' or 'block', got {permutation_shuffle!r}."
            )

        if permutation_test and mode in ('rigorous', 'precision'):
            raise ValueError(
                f"permutation_test=True is not supported for mode='{mode}'. "
                + ("The extrapolation reports its own error estimate. "
                   if mode == 'rigorous' else
                   "The mode reports how one trained network degrades, not an MI value to test. ")
                + "Permutation testing is available for mode='estimate', 'sweep', 'lag', "
                  "'conditional', 'interaction', 'transfer' and 'pairwise'."
            )
        if (permutation_test and mode in ('conditional', 'interaction', 'transfer')
                and analysis_kwargs.get('rigorous')):
            raise ValueError(
                "permutation_test=True is not supported with rigorous=True or with "
                "mode='rigorous'. The extrapolation reports its own error estimate. Test "
                "the plain estimate (rigorous=False) against its null."
            )
        if (permutation_test and mode == 'conditional'
                and analysis_kwargs.get('align') == 'dual_branch'):
            raise NotImplementedError(
                "permutation_test=True is not supported with Conditional(align='dual_branch'). "
                "Use the default align to test the conditional MI against its null."
            )

        # Verify conditional-MI / interaction-information / conditional-TE input.
        # w_data is the shared "third variable" slot for all three modes
        # (Conditional/Interaction/Transfer all name it identically, see
        # config.py) -- one variable, three possible roles, dispatched by mode.
        if w_data is not None and mode not in ('conditional', 'interaction', 'transfer'):
            logger.warning(
                f"w_data was provided but mode='{mode}' does not use it. "
                f"w_data is only consumed by mode='conditional' (conditional MI), "
                f"mode='interaction' (interaction information) and mode='transfer' "
                f"(conditional transfer entropy)."
            )

        # Validate parameters and apply defaults to base_params
        _pre_default_keys = set(base_params.keys())
        param_validator = ParameterValidator(locals())
        param_validator.validate()
        param_validator.apply_defaults()

        # Warn about n_layers/hidden_dim list mismatch only when the user explicitly
        # set n_layers (i.e., it was in base_params before defaults were applied).
        _hd = base_params.get('hidden_dim')
        if isinstance(_hd, list) and 'n_layers' in _pre_default_keys:
            _nl = base_params.get('n_layers')
            if _nl != len(_hd):
                warnings.warn(
                    f"hidden_dim is a list of length {len(_hd)} and sets the number of "
                    f"hidden layers. n_layers={_nl} is ignored.",
                    UserWarning, stacklevel=user_stacklevel(),
                )

        # Announce what beta and the lambdas mean whenever the bottleneck terms
        # are in play, since the same numbers meant something else under an
        # absolute-weight reading of the reconstruction terms.
        _variational = bool(base_params.get('use_variational', False))
        if base_params.get('use_decoder', False):
            _beta = float(base_params.get('beta', 1024.0))
            _lam_x = _decoder_lambda(base_params, 'x')
            _lam_y = _decoder_lambda(base_params, 'y')
            if _variational:
                logger.info(
                    f"With decoders and a variational encoder the loss is "
                    f"KL - beta * (MI - lambda_x * rec_x - lambda_y * rec_y). Each lambda "
                    f"is measured against the MI term. The effective weight on "
                    f"reconstruction is beta * lambda ({_beta * float(_lam_x):.4g} for X and "
                    f"{_beta * float(_lam_y):.4g} for Y at beta={_beta:g}). Lower the lambdas "
                    f"to hold reconstruction further back."
                )
            else:
                logger.info(
                    f"With decoders the loss is "
                    f"-(MI - lambda_x * rec_x - lambda_y * rec_y). Each lambda is measured "
                    f"against the MI term and carries its full weight here "
                    f"({float(_lam_x):.4g} for X and {float(_lam_y):.4g} for Y). beta "
                    f"applies only once a variational encoder supplies a KL term."
                )
            for _name, _value in (('decoder_lambda_x', _lam_x), ('decoder_lambda_y', _lam_y)):
                if float(_value) >= 1.0:
                    warnings.warn(
                        f"{_name}={float(_value):g} weighs reconstruction at least as "
                        f"heavily as the mutual information it is there to regularise. "
                        f"The lambdas are relative to the MI term. Values well below 1 are "
                        f"the usual choice.",
                        UserWarning, stacklevel=user_stacklevel(),
                    )
        elif _variational:
            logger.info(
                f"With a variational encoder the loss is KL - beta * MI. "
                f"beta={float(base_params.get('beta', 1024.0)):g} sets how far the MI term "
                f"outweighs the KL penalty on the embedding."
            )

        DataValidator(x_data, y_data, processor_type_x, processor_type_y).validate()
    
        _processor = base_params.get('processor_type_x', None)
        _embedding = base_params.get('embedding_model', 'mlp')
        # A 3-D array/tensor passed with processor_type=None is already
        # pre-windowed (N, C, W) sequential data, not a StaticDataset -- this
        # matches the same auto-detection ParameterSweep uses (`is_proc_sweep`)
        # to allow 'gru'/'lstm' on pre-processed data without re-running a
        # processor.
        _has_time_dim = hasattr(x_data, 'ndim') and x_data.ndim == 3
        # mode='transfer' builds its own (N, C, history_window) arrays from
        # raw 2-D (T, n_channels) input internally, via unfold
        # (analysis/transfer.py's _build_te_arrays) -- raw 2-D input there is
        # the intended, documented shape, not a mistake this check should
        # catch. Everywhere else (mode='estimate'/'conditional'/'interaction'/
        # etc.), the caller is expected to have already windowed the data
        # themselves before it reaches this validation.
        _mode_builds_own_windows = mode == 'transfer'
        # Checked per side, since the two sides can name different encoders and
        # are read through their own processor. A sequential encoder on Y with a
        # static Y is the same mistake as on X.
        _sides = [('embedding_model', _embedding, 'processor_type_x', x_data)]
        if base_params.get('embedding_model_y') is not None:
            _sides.append(('embedding_model_y', base_params['embedding_model_y'],
                           'processor_type_y', y_data))
        for _name, _emb, _proc_key, _side_data in _sides:
            _side_proc = base_params.get(_proc_key, None)
            if _proc_key == 'processor_type_y' and _side_proc is None:
                # processor_type_y=None inherits X's, so a static Y is only
                # static when X is too.
                _side_proc = _processor
            _side_time_dim = hasattr(_side_data, 'ndim') and _side_data.ndim == 3
            if (_side_proc is None and str(_emb).lower() in ('gru', 'lstm')
                    and not _side_time_dim and not _mode_builds_own_windows):
                _stream = _proc_key[-1]
                raise ValueError(
                    f"{_name}='{_emb}' needs input with a time axis. "
                    f"{_stream.upper()} has no processor and so no time axis. Set "
                    f"Processing({_stream}=...) to a windowed processor ('continuous', "
                    f"'spike' or 'categorical') or switch {_name} to 'mlp' or 'linear'."
                )
    
        run_params = {"mode": mode, "processor_type_x": processor_type_x, "processor_params_x": processor_params_x,
                      "processor_type_y": processor_type_y, "processor_params_y": processor_params_y,
                      "base_params": base_params, "sweep_grid": sweep_grid, "output_units": output_units,
                      "estimator": estimator, "random_seed": random_seed, "curvature_t_threshold": curvature_t_threshold,
                      "min_gamma_points": min_gamma_points, "confidence_level": confidence_level,
                      **analysis_kwargs}
        if channel_names_x is not None: run_params['channel_names_x'] = channel_names_x
        if channel_names_y is not None: run_params['channel_names_y'] = channel_names_y

        # Every processor parameter the schema knows, so a new one defers
        # windowing in a sweep without further changes.
        processor_param_keys = set().union(*PROCESSOR_PARAMS_SCHEMA.values())
        is_proc_sweep = mode == 'sweep' and any(key in (sweep_grid or {}) for key in processor_param_keys)
    
        def _to_tensor(arr):
            """Convert array-like to a float32 tensor; expand 2-D (N, C) to (N, C, 1)."""
            if torch.is_tensor(arr):
                t = arr.float()
            else:
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            if t.ndim == 2:
                t = t.unsqueeze(-1)
            return t

        # Window shifting needs the raw arrays to reach each training task, so a
        # call that shifts leaves windowing to the tasks, as a processor sweep
        # and mode='lag' do. _SHIFT_SAFE_MODES and its relatives (module scope)
        # say which modes each mechanism reaches. Transfer entropy builds its
        # histories itself and takes neither. Y without a processor reads with
        # X's.
        _effective_processor_type_y = processor_type_y if processor_type_y is not None else processor_type_x
        _shift_pair_family = shift_family(processor_type_x, _effective_processor_type_y)
        # shift_windows reslices a regular grid: continuous or categorical on
        # each side, in any combination.
        _defer_for_shift_windows = (mode in _SHIFT_WINDOWS_SAFE_MODES and base_params.get('shift_windows')
                                    and _shift_pair_family == 'regular')
        # shift_time shifts the time axis itself. It applies to spike pairs,
        # both sides in seconds, and to a spike stream paired with a regular one
        # that has 'sample_rate' set, so that a shift means the same time on
        # both sides. Regular pairs use shift_windows, which does the same job
        # at lower cost. 'rigorous' takes the spike+spike case only.
        _defer_for_shift_time = (
            base_params.get('shift_time')
            and (
                (_shift_pair_family == 'spike' and mode in _SHIFT_TIME_RIGOROUS_SAFE_MODES)
                or (_shift_pair_family == 'mixed' and mode in _SHIFT_TIME_SAFE_MODES
                    and mixed_pair_sample_rate_ok(
                        processor_type_x, processor_params_x, processor_params_y))
            )
        )
        # A W stream without a processor of its own reads with X's, and without
        # parameters of its own reads with X's parameters, kept to the keys its
        # processor takes. The same rule Processing documents for Y. A W that is
        # already windowed (3-D) is used as given.
        if (mode in ('conditional', 'interaction', 'transfer') and w_data is not None
                and getattr(w_data, 'ndim', None) != 3):
            if w_processor_type is None and processor_type_x is not None:
                w_processor_type = processor_type_x
                logger.info(
                    f"mode='{mode}': w_data has no processor of its own and reads with X's "
                    f"('{processor_type_x}'), on the same grid. Set Processing(w=...) to read "
                    f"it differently."
                )
            if w_processor_type is not None and w_processor_params is None:
                _accepted_w = PROCESSOR_PARAMS_SCHEMA.get(w_processor_type, ())
                w_processor_params = {k: v for k, v in (processor_params_x or {}).items()
                                      if k in _accepted_w}
        # conditional and interaction window W together with X inside each task
        # when X and W are both regular (continuous or categorical, in any
        # combination) under shift_windows, or both spike trains. The raw
        # arrays are concatenated channel-wise and windowed on X's grid (the
        # raw_deferred path of conditional.py and interaction.py), each block
        # encoded by its own processor. rigorous=True cuts its chunks the same
        # way. align='dual_branch' keeps W apart and is handled below.
        _cond_var_type = w_processor_type if mode in ('conditional', 'interaction') else None
        _regular_types = ('continuous', 'categorical')
        _not_dual_branch = (mode != 'conditional' or analysis_kwargs.get('align') != 'dual_branch')
        _defer_regular_conditional_interaction = (
            mode in ('conditional', 'interaction') and _not_dual_branch
            and base_params.get('shift_windows')
            and processor_type_x in _regular_types and _cond_var_type in _regular_types
        )
        # A spike X with a spike W is always windowed inside the task. That is
        # what lets shift_time reach the data, since the three-stream bundle
        # built here is already windowed when training starts.
        _defer_spike_conditional_interaction = (
            mode in ('conditional', 'interaction') and _not_dual_branch
            and processor_type_x == 'spike' and _cond_var_type == 'spike'
        )
        _defer_for_conditional_interaction = (
            _defer_regular_conditional_interaction or _defer_spike_conditional_interaction
        )
        if _defer_for_conditional_interaction:
            # The concatenated array is windowed with X's window_size and
            # step_size, so a W that sets different ones is refused.
            for _key in ('window_size', 'step_size'):
                _x_value = (processor_params_x or {}).get(_key)
                _w_value = (w_processor_params or {}).get(_key)
                if _w_value is not None and _w_value != _x_value:
                    raise ValueError(
                        f"mode='{mode}' windows W on X's window grid here. The two need the "
                        f"same {_key}. Processing(w_params=...) sets {_key}={_w_value} and "
                        f"Processing(x_params=...) sets {_x_value}. Remove {_key} from "
                        f"w_params to use X's or set them equal."
                    )
            # It is also windowed on X's clock, so a different w_time is
            # refused. Equal clocks, the usual case, pass.
            if w_time is not None and x_time is not None:
                _wt, _xt = np.asarray(w_time), np.asarray(x_time)
                if _wt.shape != _xt.shape or not np.allclose(_wt, _xt):
                    _alternative = (" or set Training(shift_windows=False), which windows W "
                                    "on its own clock" if base_params.get('shift_windows') else "")
                    raise ValueError(
                        f"mode='{mode}' windows W on X's clock here. A w_time that differs "
                        f"from x_time cannot be used. Resample W onto X's clock and drop "
                        f"Processing(w_time=...){_alternative}."
                    )
        # align='dual_branch' never concatenates X and W: W keeps its own
        # window geometry, so the checks above do not apply. shift_windows
        # reaches it through a three-way shifter that moves X, W and Y in step
        # (shift_windowing.try_build_shift_windows_dataset_dual_branch).
        _defer_for_dual_branch_shift_windows = (
            mode == 'conditional' and analysis_kwargs.get('align') == 'dual_branch'
            and base_params.get('shift_windows')
            and processor_type_x in _regular_types and _cond_var_type in _regular_types
            # The three-way shifter needs Y on a regular grid as well; without
            # it the call would fall back to create_dataset with a tuple X.
            and _effective_processor_type_y in _regular_types
        )
        # W joins X and Y on one grid whenever it needs windowing and no task
        # windows it later. Each grid starts at the latest start among its own
        # streams, so a W windowed on a grid of its own would start somewhere
        # else and share no window times with X and Y.
        _bundle_w = (
            mode in ('conditional', 'interaction')
            and w_data is not None
            and w_processor_type is not None
            and not _defer_for_conditional_interaction
            and not _defer_for_dual_branch_shift_windows
        )
        if mode == 'transfer':
            # Transfer entropy builds its histories from rows of a (T, C) series;
            # the transfer branch below puts the streams on one grid itself.
            x_run_data, y_run_data = x_data, y_data
        elif (is_proc_sweep or mode == 'lag' or _defer_for_shift_windows or _defer_for_shift_time
                or _defer_for_conditional_interaction or _defer_for_dual_branch_shift_windows):
            logger.info("Windowing inside each training task.")
            x_run_data, y_run_data = x_data, y_data
            if not is_proc_sweep:
                # The windows are built later, in each task. X's extent, window
                # and step give their count now. Each lag trains on what the
                # shift leaves, so a lag scan is counted at its largest lag.
                _lags = (lag_range if lag_range is not None
                         else analysis_kwargs.get('lag_range')) if mode == 'lag' else None
                _largest_lag = max((abs(float(v)) for v in _lags), default=0.0) if _lags is not None else 0.0
                _counted = _deferred_window_count(x_data, base_params, _largest_lag)
                if _counted is not None:
                    _warn_small_sample(*_counted, base_params=base_params, mode=mode)
        elif processor_type_x is None and processor_type_y is None:
            # Fast path: data is already pre-processed. Convert to tensors inline and skip
            # the full create_dataset / PairedDataset allocation.
            x_run_data = _to_tensor(x_data)
            y_run_data = _to_tensor(y_data) if y_data is not None else None
            if y_run_data is not None and x_run_data.shape[0] != y_run_data.shape[0]:
                _min_n = min(x_run_data.shape[0], y_run_data.shape[0])
                logger.warning(
                    f"X ({x_run_data.shape[0]}) and Y ({y_run_data.shape[0]}) differ in sample count. "
                    f"Both are truncated to {_min_n}."
                )
                x_run_data = x_run_data[:_min_n]
                y_run_data = y_run_data[:_min_n]
            base_params['processor_type_x'] = None
            base_params['processor_type_y'] = None
            if base_params.get('processor_params_x') is None:
                base_params['processor_params_x'] = {}
            if base_params.get('processor_params_y') is None:
                base_params['processor_params_y'] = {}
            base_params['processor_params_x']['preprocessed'] = True
            base_params['processor_params_y']['preprocessed'] = True
            _warn_small_sample(x_run_data.shape[0], 'samples', base_params, mode)
            if mode not in ('dimensionality', 'pairwise') and y_run_data is None:
                raise ValueError(f"y_data must be provided for mode '{mode}'.")
        else:
            # One grid, however many streams the mode needs. A conditioning
            # stream joins the same call: built on its own it would derive its
            # own grid origin and could end up sharing no window times with the
            # pair it conditions.
            streams = OrderedDict()
            streams['x'] = dict(data=x_data, time=x_time,
                                processor_type=processor_type_x,
                                processor_params=processor_params_x)
            if mode != 'dimensionality' or y_data is not None:
                # A Y without parameters of its own reads X's, as in the
                # two-argument form of create_dataset.
                streams['y'] = dict(data=y_data, time=y_time,
                                    processor_type=_effective_processor_type_y,
                                    processor_params=(processor_params_y if processor_params_y is not None
                                                      else processor_params_x))
            if _bundle_w:
                streams['w'] = dict(data=w_data, time=w_time,
                                    processor_type=w_processor_type,
                                    processor_params=w_processor_params)
            dataset = create_dataset(streams)

            base_params['processor_type_x'] = None
            base_params['processor_type_y'] = None

            if base_params.get('processor_params_x') is None: base_params['processor_params_x'] = {}
            if base_params.get('processor_params_y') is None: base_params['processor_params_y'] = {}
            base_params['processor_params_x']['preprocessed'] = True
            base_params['processor_params_y']['preprocessed'] = True

            # Windowing happens once, here -- the dataset that reaches the
            # Trainer downstream is a plain, already-windowed PairedDataset
            # with no window_manager of its own (processor_type_x/y were just
            # wiped to None above). Capture the window geometry now, while
            # dataset.window_manager is still live, so the blocked-split
            # leakage check has something to validate against.
            _wm = getattr(dataset, 'window_manager', None)
            if _wm is not None:
                base_params['leak_check_window_size'] = _wm.window_size
                base_params['leak_check_step'] = _wm.resolve_step()

            # Retention is reported per task (see analysis/task.py), since it
            # varies between tasks and one run-level scalar would misdescribe
            # every row but one on a sweep. When windowing happens here the
            # tasks receive already-windowed tensors and never see this
            # dataset, so hand the value down through base_params for them to
            # report. Tasks that window for themselves prefer their own.
            if getattr(dataset, 'window_retention', None) is not None:
                base_params['_window_retention'] = dataset.window_retention
                base_params['_n_windows_built'] = dataset.n_windows_built
                base_params['_n_windows_retained'] = dataset.n_windows_retained

            _n_built = dataset.x_data.shape[0] if getattr(dataset, 'x_data', None) is not None else 0
            _warn_small_sample(_n_built, 'windows', base_params, mode)

            if mode in ('dimensionality', 'pairwise'):
                # dimensionality and pairwise can operate on x_data alone
                x_run_data = dataset.x_data
                y_run_data = dataset.y_data if y_data is not None else None
            else:
                if y_data is None: raise ValueError(f"y_data must be provided for mode '{mode}'.")
                x_run_data = dataset.x_data
                y_run_data = dataset.y_data

        _warn_if_shift_time_dead(base_params, mode, is_proc_sweep, processor_type_x,
                                 processor_params_x, _effective_processor_type_y, processor_params_y,
                                 user_set_keys=_pre_default_keys,
                                 extra_reachable=_defer_spike_conditional_interaction)
        _warn_if_shift_windows_dead(base_params, mode, processor_type_x, _effective_processor_type_y,
                                    user_set_keys=_pre_default_keys,
                                    extra_reachable=(_defer_regular_conditional_interaction
                                                    or _defer_for_dual_branch_shift_windows))

        if mode not in _MODES:
            raise ValueError(
                f"Unknown mode: '{mode}'. Expected one of: "
                f"{', '.join(repr(m) for m in _MODES)}."
            )
        _to_bits = output_units == 'bits'
        n_workers = analysis_kwargs.get('n_workers', 1)
        grid = dict(sweep_grid or {})
        if mode in ('estimate', 'precision') and grid:
            _one_network = ("trains one model" if mode == 'estimate' else
                            "trains one baseline network and evaluates it at every tau in tau_grid")
            _instead = ("Use mode='sweep' to repeat the estimate or to sweep a setting."
                        if mode == 'estimate' else
                        "To compare settings, run mode='precision' once per setting.")
            warnings.warn(
                f"sweep_grid has no effect for mode='{mode}'. That mode {_one_network}. The "
                f"call ran once with the base configuration and the grid {sorted(grid)} was "
                f"not used. {_instead}",
                UserWarning, stacklevel=user_stacklevel(),
            )
            grid = {}
        if mode == 'dimensionality' and 'run_id' in grid:
            raise ValueError(
                "mode='dimensionality' repeats every fit through Dimensionality(n_restarts=...). "
                "Remove 'run_id' from sweep_grid and set n_restarts."
            )
        if mode == 'dimensionality' and 'embedding_dim' in grid:
            raise ValueError(
                "mode='dimensionality' varies embedding_dim itself. Remove 'embedding_dim' from "
                "sweep_grid and pass the values as Dimensionality(embedding_dims=...)."
            )

        _embedding_flags = [k for k in ('return_embeddings', 'track_embeddings',
                                        'return_rotated_embeddings', 'return_rotation_matrices')
                            if base_params.get(k)]
        _use_rigorous = bool(analysis_kwargs.get('rigorous', False))
        if _embedding_flags and (mode in ('rigorous', 'precision')
                                 or (mode in ('conditional', 'interaction', 'transfer'))):
            if mode == 'precision':
                _why = ("it trains one baseline network and reports how its MI degrades, so there "
                        "is no repeat whose embeddings the result could hold")
            elif mode == 'rigorous':
                _why = "every repeat trains a network on each chunk of the gamma ladder"
            else:
                _why = ("every repeat trains one network per term of the quantity, and no one "
                        "network's embeddings describe it")
            raise ValueError(
                f"Output({_embedding_flags[0]}=...) is not available for mode='{mode}': {_why}. "
                f"Use mode='estimate' or mode='sweep' on the pair whose embeddings you want."
            )

        # The params recorded on the result: the resolved configuration, with every
        # grid key holding its grid.
        _record = {**run_params, 'base_params': {**base_params, **grid}}
        ctx: Dict[str, Any] = {}
        x_run, y_run, w_run = x_run_data, y_run_data, None

        if mode in ('estimate', 'sweep'):
            ctx = {'is_proc_sweep': is_proc_sweep}

        elif mode == 'lag':
            # `lag_range` reaches here already unpacked from Lag(...) by run(); the
            # analysis_kwargs fallback covers direct _run_flat callers.
            lag_range_val = lag_range if lag_range is not None else analysis_kwargs.get('lag_range')
            if lag_range_val is None:
                raise ValueError(
                    "`lag_range` must be provided for mode='lag'. Pass it in the per-mode "
                    "config: nmi.run(..., mode='lag', lag=Lag(lag_range=range(-10, 11)))."
                )
            ctx = {'lag_range': lag_range_val, 'equalize_n': analysis_kwargs.get('equalize_n', False)}
            _record['lag_range'] = list(lag_range_val)

        elif mode == 'precision':
            if tau_grid is None:
                raise ValueError("`tau_grid` must be provided for mode='precision'.")
            ctx = {'precision_kwargs': dict(tau_grid=tau_grid, corrupt_target=corrupt_target,
                                            corruption_method=corruption_method,
                                            n_noise_samples=n_noise_samples,
                                            threshold_ratio=threshold_ratio)}
            _record['tau_grid'] = list(tau_grid)

        elif mode == 'rigorous':
            _rig = {k: v for k, v in analysis_kwargs.items() if k in _RIGOROUS_FIT_KEYS}
            _rig.update(curvature_t_threshold=curvature_t_threshold,
                        min_gamma_points=min_gamma_points, confidence_level=confidence_level)
            ctx = {'rigorous_kwargs': _rig}

        elif mode == 'dimensionality':
            ctx = {'dim_kwargs': {k: v for k, v in analysis_kwargs.items() if k != 'n_workers'},
                   'user_set_keys': _pre_default_keys}

        elif mode == 'pairwise':
            y_run = y_run_data if y_data is not None else None
            ctx = {'pairs': analysis_kwargs.get('pairs'), 'channel_names_x': channel_names_x,
                   'channel_names_y': channel_names_y}

        elif mode in ('conditional', 'interaction'):
            if w_data is None:
                raise ValueError(f"`w_data` must be provided for mode='{mode}'.")
            _align = analysis_kwargs.get('align') if mode == 'conditional' else None
            _deferred = (_defer_for_conditional_interaction
                         or (mode == 'conditional' and _defer_for_dual_branch_shift_windows))
            if _deferred:
                # Raw W, windowed together with X inside the task.
                w_run = w_data
            elif _bundle_w:
                # Built above as a third stream on X and Y's grid.
                _w_stream = dataset.stream('w')
                w_run = _w_stream.data
                if w_processor_type == 'categorical':
                    w_run = _reshape_categorical_w_for_conditional(w_run, _w_stream)
            else:
                w_run = w_data if torch.is_tensor(w_data) else torch.from_numpy(np.array(w_data)).float()
            if _use_rigorous and mode == 'conditional' and _defer_for_dual_branch_shift_windows:
                raise NotImplementedError(
                    "rigorous=True is not supported together with Conditional(align='dual_branch') "
                    "and shift_windows=True. The gamma ladder cuts every stream at X's window "
                    "boundaries. The dual-branch conditioning stream has a window geometry of "
                    "its own and its chunks would not line up with X's. Pass shift_windows=False "
                    "or drop rigorous=True."
                )
            ctx = {'align': _align, 'raw_deferred': _deferred, 'w_processor_type': w_processor_type,
                   'w_processor_params': w_processor_params}
            if _use_rigorous:
                from .analysis.conditional import _cmi_rigorous_scalar
                from .analysis.interaction import _ii_rigorous_scalar
                if mode == 'conditional':
                    _scalar, _extra = _cmi_rigorous_scalar, {'w_data': w_run, 'c_data': w_run}
                    _extra_kwargs = {'align': _align, 'raw_deferred': _defer_for_conditional_interaction,
                                     'w_processor_type': w_processor_type}
                else:
                    _scalar, _extra = _ii_rigorous_scalar, {'w_data': w_run}
                    _extra_kwargs = {'raw_deferred': _defer_for_conditional_interaction,
                                     'w_processor_type': w_processor_type}
                ctx.update(rigorous=True, scalar_fn=_scalar, extra_data=_extra,
                           extra_kwargs=_extra_kwargs,
                           raw_deferred=_defer_for_conditional_interaction,
                           rigorous_kwargs=_scalar_rigorous_kwargs(
                               analysis_kwargs, curvature_t_threshold, min_gamma_points,
                               confidence_level))
                _record['rigorous'] = True

        elif mode == 'transfer':
            if history_window is None:
                raise ValueError("`history_window` must be provided for mode='transfer'.")
            if y_data is None:
                raise ValueError("y_data must be provided for mode='transfer'.")
            if any(t is not None for t in (processor_type_x, processor_type_y, w_processor_type)):
                # Every stream on one grid of one-step rows, which the histories
                # index by offset.
                from .analysis.offsets import grid_rows, one_step_rows
                _specs = OrderedDict()
                _specs['x'] = dict(data=x_data, time=x_time, processor_type=processor_type_x,
                                   processor_params=processor_params_x)
                _specs['y'] = dict(data=y_data, time=y_time,
                                   processor_type=_effective_processor_type_y,
                                   processor_params=(processor_params_y if processor_params_y is not None
                                                     else processor_params_x))
                if w_data is not None:
                    _specs['w'] = dict(data=w_data, time=w_time, processor_type=w_processor_type,
                                       processor_params=w_processor_params)
                _rows = one_step_rows(grid_rows(_specs), "mode='transfer'")
                x_run, y_run, w_run = _rows['x'], _rows['y'], _rows.get('w')
            else:
                x_run, y_run = _to_2d(_to_tensor(x_data)), _to_2d(_to_tensor(y_data))
                if w_data is not None:
                    w_run = _to_2d(_to_tensor(w_data))
                for _name, _arr in (('x_data', x_run), ('y_data', y_run), ('w_data', w_run)):
                    if _arr is not None and _arr.ndim == 3:
                        raise ValueError(
                            f"mode='transfer' requires {_name} of shape (n_timepoints, n_channels) "
                            f"and received a 3-D array of shape {tuple(_arr.shape)}. Transfer "
                            f"entropy builds its histories from the raw series. Pass it unwindowed "
                            f"or set Processing(...) with one-step windows to put streams of other "
                            f"kinds on one grid."
                        )
            # The histories are built here, so the training tasks read them as given.
            base_params['processor_type_x'] = None
            base_params['processor_type_y'] = None
            for _key in ('processor_params_x', 'processor_params_y'):
                base_params[_key] = {**(base_params.get(_key) or {}), 'preprocessed': True}
            ctx = {'history_window': history_window, 'prediction_horizon': prediction_horizon,
                   'stride': stride, 'bidirectional': bidirectional_te}
            _record['history_window'] = history_window
            if _use_rigorous:
                from .analysis.transfer import _te_rigorous_scalar
                ctx.update(rigorous=True, scalar_fn=_te_rigorous_scalar,
                           extra_data={'w_data': w_run} if w_run is not None else None,
                           extra_kwargs={'history_window': history_window,
                                         'prediction_horizon': prediction_horizon,
                                         'stride': stride, 'bidirectional': bidirectional_te},
                           # Histories are time-ordered, so the ladder's chunks are contiguous.
                           temporal_kwargs={'temporal_chunking': True},
                           rigorous_kwargs=_scalar_rigorous_kwargs(
                               analysis_kwargs, curvature_t_threshold, min_gamma_points,
                               confidence_level))
                _record['rigorous'] = True

        from .analysis.modes import produce
        from .analysis.permutation import permutation_nulls, attach_nulls
        produced = produce(mode, x_run, y_run, w_run, base_params, grid, ctx=ctx,
                           to_bits=_to_bits, n_workers=n_workers)
        result = build_results(mode, _record, produced['rows'],
                               config_keys=produced['config_keys'],
                               axis_keys=produced['axis_keys'],
                               mean_cols=produced['mean_cols'], first_cols=produced['first_cols'],
                               std_cols=produced['std_cols'], details=produced['details'],
                               row_scalars=produced['row_scalars'])
        if permutation_test:
            if mode == 'dimensionality':
                logger.warning(
                    "permutation_test=True has no effect for mode='dimensionality'. The mode "
                    "reports a curve over embedding dimensions and the dimension at which it "
                    "saturates. Neither has a null distribution. No null is computed."
                )
            elif mode == 'pairwise' and y_run is None:
                logger.warning(
                    "permutation_test=True has no effect for mode='pairwise' without y_data. "
                    "Every pair is two channels of X. Moving X moves both sides of each pair "
                    "together and leaves no null to compute. Cross-pairwise MI with y_data "
                    "supports permutation testing."
                )
            else:
                trials = permutation_nulls(
                    mode, x_run, y_run, w_run, base_params, grid, ctx=ctx, to_bits=_to_bits,
                    n_permutations=n_permutations, n_workers=n_workers,
                    permutation_shuffle=permutation_shuffle)
                attach_nulls(result, trials, produced['axis_keys'])
        return result
    finally:
        logger.setLevel(_prev_level)
        for h, lv in zip(logger.handlers, _prev_handler_levels):
            h.setLevel(lv)


# Named parameters of the engine, computed once at import. run() consults this
# (via `_named`) to route each lowered config key to the correct bucket. Defined
# here because it depends on _run_flat's signature.
_ENGINE_PARAMS = frozenset(
    n for n, p in _inspect.signature(_run_flat).parameters.items()
    if p.kind in (_inspect.Parameter.POSITIONAL_OR_KEYWORD,
                  _inspect.Parameter.KEYWORD_ONLY)
) - {'x_data', 'y_data'}


_SMALL_SAMPLE = 200


def _deferred_window_count(x_data, base_params: dict, largest_lag: float = 0.0):
    """``(count, unit)`` of the rows X yields once the tasks have windowed it,
    or None when X's shape does not say.

    A spike population is counted from its time span, a regular stream from its
    length, both in the stream's own units (seconds with a clock or a
    ``sample_rate``, samples otherwise). ``largest_lag`` is taken off the extent,
    since a lag shortens the stretch where X and Y overlap.
    """
    from .analysis.permutation import _spike_population_extent
    from .data.shift_windowing import resolve_step_size, seconds_to_samples, safe_n_windows
    wp = base_params.get('processor_params_x') or {}
    window = wp.get('window_size')
    if isinstance(x_data, list):
        if window is None:
            return None
        t_start, t_end = _spike_population_extent(x_data, base_params, 'x')
        step = resolve_step_size(window, wp.get('step_size'))
        span = t_end - t_start - largest_lag
        return max(0, int((span - window) // step) + 1), 'windows'
    if getattr(x_data, 'ndim', None) is None:
        return None
    if window is None:
        return max(0, x_data.shape[0] - int(round(largest_lag))), 'samples'
    if getattr(x_data, 'ndim', None) != 2:
        return None
    period = 1.0 / wp['sample_rate'] if wp.get('sample_rate') else 1.0
    step = resolve_step_size(window, wp.get('step_size'))
    n_rows = x_data.shape[0] - int(round(largest_lag / period))
    return safe_n_windows(max(n_rows, 0), seconds_to_samples(window, period),
                          seconds_to_samples(step, period)), 'windows'


def _warn_small_sample(n_samples: int, unit: str, base_params: dict,
                       mode: str = 'estimate') -> None:
    """One warning for a small dataset, whether the caller passed arrays or the
    library windowed the data (``unit`` is 'samples' or 'windows')."""
    if not 0 < n_samples < _SMALL_SAMPLE:
        return
    tips = []
    if base_params.get('dropout', 0.0) == 0.0:
        tips.append("dropout=0.2")
    norm = base_params.get('norm_layer')
    if norm in (None, 'none') or (norm == 'auto' and mode != 'dimensionality'):
        tips.append("norm_layer='layer'")
    hidden, embed = base_params.get('hidden_dim', 64), base_params.get('embedding_dim', 64)
    if isinstance(hidden, int) and hidden > 32:
        tips.append(f"hidden_dim=32 (current: {hidden})")
    if embed > 32:
        tips.append(f"embedding_dim=32 (current: {embed})")
    tips.append("optimizer='adamw' with optimizer_params={'weight_decay': 1e-3}")
    warnings.warn(
        f"Only {n_samples} {unit} reach the estimator. The samples an estimate needs grow "
        f"roughly as N ~ d^2/I (tutorial 02, section 2). Here d is the number of latent "
        f"dimensions carrying the shared information and I is the information itself. "
        f"Neither is known before estimating. A few latent dimensions carrying several bits can settle "
        f"within a few hundred samples. Many dimensions or little information need far more. "
        f"mode='rigorous' tests whether the estimate still moves with the sample size. At "
        f"this size regularisation also helps: {'; '.join(tips)}.",
        UserWarning, stacklevel=user_stacklevel(),
    )


# Modes/paths where windowing is deferred to the worker that trains the model,
# so the dataset it builds is still a genuine PairedTemporalDataset with a
# live WindowManager -- shift_time (and anything else gated on
# is_temporal at the Trainer) is actually reachable there. Everywhere else,
# run.py windows the data once, up front, and hands the Trainer an
# already-windowed static PairedDataset that has never heard of time.
def _shift_time_is_reachable(mode: str, is_proc_sweep: bool,
                             processor_type_x: Optional[str] = None,
                             effective_processor_type_y: Optional[str] = None,
                             processor_params_x: Optional[dict] = None,
                             processor_params_y: Optional[dict] = None) -> bool:
    if is_proc_sweep or mode == 'lag':
        return True
    _family = shift_family(processor_type_x, effective_processor_type_y)
    if mode == 'rigorous':
        # Narrower than the other modes below: 'rigorous' only supports the
        # 'spike' family's chunk-to-raw-time-range translation, not 'mixed'.
        return _family == 'spike'
    if mode not in _SHIFT_TIME_SAFE_MODES:
        return False
    if _family == 'spike':
        return True
    if _family == 'mixed':
        return mixed_pair_sample_rate_ok(processor_type_x, processor_params_x,
                                         processor_params_y)
    return False


def _warn_if_shift_time_dead(base_params: dict, mode: str, is_proc_sweep: bool,
                             processor_type_x: Optional[str] = None,
                             processor_params_x: Optional[dict] = None,
                             effective_processor_type_y: Optional[str] = None,
                             processor_params_y: Optional[dict] = None,
                             user_set_keys: Optional[set] = None,
                             extra_reachable: bool = False) -> None:
    """Warn when an explicit shift_time=True cannot take effect on this call.

    This is the earliest point that knows both the mode and whether windowing
    happens inside the training tasks. Only an explicit setting warns (a key in
    `user_set_keys`, the keys present before defaults were applied); the
    schema default stays silent.

    shift_time is reachable in mode='lag', in a processor sweep, and in the
    modes of `_SHIFT_TIME_SAFE_MODES` for a spike+spike pair or a spike stream
    paired with a regular one that has 'sample_rate' set (see
    `shift_windowing.mixed_pair_sample_rate_ok`). Regular pairs use
    shift_windows, which reslices without rebuilding datasets.

    extra_reachable : bool, optional
        True when the caller knows of a path this function cannot derive from
        the mode and processors alone: a spike X with a spike W in
        conditional or interaction, which is windowed inside the task.
    """
    if user_set_keys is not None and 'shift_time' not in user_set_keys:
        return
    if base_params.get('shift_time') is not True:
        return
    if extra_reachable:
        return
    if _shift_time_is_reachable(mode, is_proc_sweep, processor_type_x,
                                effective_processor_type_y, processor_params_x, processor_params_y):
        return
    _family = shift_family(processor_type_x, effective_processor_type_y)
    _mixed_hint = ""
    if mode in _SHIFT_TIME_SAFE_MODES and _family == 'mixed':
        _mixed_hint = (
            " This pair mixes 'spike' with a continuous or categorical stream, whose "
            "shift is counted in samples unless it has a 'sample_rate'. Set "
            "'sample_rate' for that stream in Processing(x_params=...) or "
            "Processing(y_params=...) to enable shifting."
        )
    warnings.warn(
        f"shift_time=True has no effect for mode='{mode}' with this configuration. "
        f"The data are windowed once before training here. Training then receives "
        f"fixed windows with no time left to shift. shift_time takes effect where "
        f"windowing happens inside each training task: in mode='lag', in mode='sweep' "
        f"with a processor parameter such as window_size in sweep_grid, and in the "
        f"modes {list(_SHIFT_TIME_SAFE_MODES)} for a 'spike'+'spike' pair or for "
        f"a spike stream paired with a continuous or categorical one that has "
        f"'sample_rate' set. mode='rigorous' takes it for a 'spike'+'spike' pair."
        f"{_mixed_hint} For continuous and categorical data, "
        f"Training(shift_windows=True) does the same job at lower cost. Set "
        f"Training(shift_time=False) to silence this warning.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def _warn_if_shift_windows_dead(base_params: dict, mode: str, processor_type_x: Optional[str],
                                effective_processor_type_y: Optional[str],
                                user_set_keys: Optional[set] = None,
                                extra_reachable: bool = False) -> None:
    """Warn when an explicit shift_windows=True cannot take effect on this call.

    shift_windows reslices a regular grid, so it needs both X and Y read by a
    continuous or categorical processor (in any combination) and a mode in
    `_SHIFT_WINDOWS_SAFE_MODES`. Spike data and unprocessed input have no grid
    to reslice. Only an explicit setting warns, as for
    `_warn_if_shift_time_dead`.

    extra_reachable : bool, optional
        True when conditional or interaction window a regular W together with
        X inside the task, a path this function cannot derive from X and Y.
    """
    if user_set_keys is not None and 'shift_windows' not in user_set_keys:
        return
    if base_params.get('shift_windows') is not True:
        return
    if extra_reachable:
        return
    if mode in _SHIFT_WINDOWS_SAFE_MODES and shift_family(processor_type_x, effective_processor_type_y) == 'regular':
        return
    warnings.warn(
        f"shift_windows=True has no effect for mode='{mode}' with "
        f"Processing(x={processor_type_x!r}) and Y read by "
        f"{effective_processor_type_y!r}. It takes effect for the modes "
        f"{list(_SHIFT_WINDOWS_SAFE_MODES)} when X and Y are both continuous or "
        f"categorical (for example Processing(x='continuous', "
        f"x_params={{'window_size': ...}}, y='categorical')). Spike data have no regular "
        f"sampling grid to reslice. They use Training(shift_time=True). Set "
        f"Training(shift_windows=False) to silence this warning.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )
