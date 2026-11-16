"""A sweep_grid runs every configuration as a separate call with its settings would: the same estimate, the
same epoch and the same verdict on the curve.

A call prepares its data once and runs the configurations of its grid on that preparation. A setting that
the preparation reads (a processor parameter, or one of ``defaults.PREPARATION_SETTINGS``) therefore gets one
preparation per value, and every other setting is read by the task of each configuration. Each test here runs
two values of a setting once as a sweep and once as two separate calls, on a problem small enough that the
whole schema runs in seconds. A new setting joins the first test by itself; one that needs a second value or
a context of its own gets an entry in ``ALTERNATIVES`` or ``CONTEXT``.
"""
import warnings
from dataclasses import fields

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi.defaults import BASE_PARAMS_SCHEMA, CHECK_SETTINGS, PREPARATION_SETTINGS

_rng = np.random.default_rng(0)
X = _rng.normal(size=(200, 3))
Y = X + 0.7 * _rng.normal(size=(200, 3))
W = X + _rng.normal(size=(200, 3))
XS = _rng.normal(size=(1200, 2)).cumsum(0) * 0.1
YS = np.roll(XS, 1, axis=0) + 0.3 * _rng.normal(size=(1200, 2))
XL = _rng.normal(size=(3000, 2)).cumsum(0) * 0.1
YL = np.roll(XL, 1, axis=0) + 0.3 * _rng.normal(size=(3000, 2))
WINDOWED = {'processing': nmi.Processing(x='continuous', x_params={'window_size': 6, 'step_size': 3})}

BASE = {'model': {'hidden_dim': 16, 'embedding_dim': 8},
        'training': {'n_epochs': 3, 'batch_size': 64, 'patience': 100},
        'split': {'mode': 'random'}, 'estimator': {'name': 'infonce'}, 'output': {}}
# Settings that a grid knows by another name than their config field.
RENAMED = {'split_mode': ('split', 'mode'), 'split_gap_fraction': ('split', 'gap_fraction'),
           'estimator_name': ('estimator', 'name'), 'estimator_params': ('estimator', 'params')}
OWNER = {f.name: slot for slot, cls in (('model', nmi.Model), ('training', nmi.Training), ('split', nmi.Split),
                                         ('estimator', nmi.Estimator), ('output', nmi.Output))
         for f in fields(cls)}

# Two values for the settings whose schema default and type give no obvious second value.
ALTERNATIVES = {
    'split_mode': ['random', 'blocked'], 'whitening': ['std', None], 'input_scaling': ['auto', 'none'],
    'activation': ['leaky_relu', 'relu'], 'estimator_name': ['infonce', 'smile'],
    'optimizer': ['adam', 'adamw'], 'scheduler': [None, 'cosine'], 'critic_type': ['separable', 'concat'],
    'embedding_model': ['mlp', 'gru'], 'norm_layer': ['auto', 'layer'], 'epoch_rule': ['plateau_start', 'peak'],
    'use_amp': ['auto', False], 'hidden_dim': [16, 32], 'embedding_dim': [8, 4], 'n_epochs': [3, 4],
    'batch_size': [64, 32], 'train_fraction': [0.9, 0.7], 'split_gap_fraction': [0.5, 0.25],
    'peak_fraction': [1.0, 0.8], 'eval_train': [False, True], 'optimizer_params': [{}, {'weight_decay': 0.01}],
    'estimator_params': [{'clip': 5.0}, {'clip': 2.0}], 'augmentation_params': [{}, {'noise_std': 0.1}],
    'augmentation_params_x': [None, {'noise_std': 0.1}], 'augmentation_params_y': [None, {'noise_std': 0.1}],
    'embedding_model_y': [None, 'gru'], 'embedding_dim_y': [None, 4], 'hidden_dim_y': [None, 32],
    'n_layers_y': [None, 1], 'n_layers_head': [None, 1], 'hidden_dim_head': [None, 32],
    'lr_head_multiplier': [None, 2.0], 'decoder_lambda_x': [None, 0.01], 'decoder_lambda_y': [None, 0.01],
    'decoder_output_activation_x': ['linear', 'sigmoid'], 'decoder_output_activation_y': ['linear', 'sigmoid'],
    'gradient_clip_val': [None, 1.0], 'train_subset_size': [None, 100], 'test_subset_size': [None, 10],
    'max_eval_samples': [5000, 20], 'large_first_layer': [500000, 10], 'max_n_batches': [512, 2],
    'cnn_pooling': ['flatten', 'mean'], 'kernel_size': [3, 5], 'nhead': [4, 2], 'bias': [True, False],
    'dataset_device': ['cpu', None], 'scheduler_params': [{}, {'eta_min': 1e-5}],
}
# Settings that are read only in another context, and the rows a windowed processor builds.
CONTEXT = {
    'decoder_lambda': {'model': {'use_decoder': True}}, 'decoder_lambda_x': {'model': {'use_decoder': True}},
    'decoder_lambda_y': {'model': {'use_decoder': True}},
    'decoder_output_activation_x': {'model': {'use_decoder': True}},
    'decoder_output_activation_y': {'model': {'use_decoder': True}},
    'beta': {'model': {'use_variational': True}},
    'n_layers_head': {'model': {'critic_type': 'hybrid'}}, 'hidden_dim_head': {'model': {'critic_type': 'hybrid'}},
    'lr_head_multiplier': {'model': {'critic_type': 'hybrid'}},
    'embedding_dim_y': {'model': {'critic_type': 'hybrid'}},
    'bidirectional': {'model': {'embedding_model': 'gru'}, 'windowed': True},
    'kernel_size': {'model': {'embedding_model': 'cnn'}, 'windowed': True},
    'cnn_pooling': {'model': {'embedding_model': 'cnn'}, 'windowed': True},
    'nhead': {'model': {'embedding_model': 'transformer'}, 'windowed': True},
    'embedding_model': {'windowed': True}, 'embedding_model_y': {'windowed': True},
    # Rows that shift from a base that does not shift: the case a decision made once per call got wrong.
    'shift_windows': {'training': {'shift_windows': False}, 'windowed': True},
    'estimator_params': {'estimator': {'name': 'smile'}},
    'scheduler_params': {'training': {'scheduler': 'cosine'}},
    'return_rotated_embeddings': {'output': {'return_embeddings': True}},
    'rotated_embeddings_per_epoch': {'output': {'track_embeddings': True, 'return_rotated_embeddings': True}},
    'return_rotation_matrices': {'output': {'return_embeddings': True, 'return_rotated_embeddings': True}},
}
# Settings that are not a configuration of one call: the data and its shapes, the processors (swept
# through their own parameters), objects, the split indices, the call's seed and output, and the settings
# that the grid refuses because they change no estimate.
NOT_SWEPT = {'x_time', 'y_time', 'input_dim_x', 'input_dim_y', 'n_channels_x', 'n_channels_y', 'processor_type_x',
             'processor_type_y', 'processor_params_x', 'processor_params_y', 'custom_critic',
             'custom_embedding_cls', 'custom_embedding_cls_y', 'train_indices', 'test_indices', 'gamma', 'lag',
             'verbose', 'show_progress', 'device', 'save_best_model_path', 'pytorch_predefined', 'pretrained',
             'random_seed', 'output_units', 'min_reliable_samples', 'small_sample', *CHECK_SETTINGS}


def _alternatives(key):
    if key in ALTERNATIVES:
        return ALTERNATIVES[key]
    spec = BASE_PARAMS_SCHEMA[key]
    default = spec.get('default')
    if isinstance(default, bool):
        return [default, not default]
    if isinstance(default, int):
        return [default, max(spec.get('min', 0), default // 2 if default > 2 else default + 1)]
    if isinstance(default, float):
        return [default, default / 2 if default > 0 else 0.1]
    raise AssertionError(f"'{key}' needs two values in ALTERNATIVES")


def _configs(key, value, context):
    out = {slot: {**settings, **context.get(slot, {})} for slot, settings in BASE.items()}
    if value is not ...:
        slot, name = RENAMED.get(key, (OWNER.get(key), key))
        out[slot][name] = value
    return out


def _run(key, value, context, grid=None):
    data = (XS, YS, WINDOWED) if context.get('windowed') else (X, Y, {})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return nmi.run(data[0], data[1], mode='sweep' if grid else 'estimate', sweep_grid=grid, n_workers=1,
                       seed=0, show_progress=False, **_configs(key, value, context), **data[2])


def _reading(runs):
    row = runs.iloc[0]
    return float(row['mi']), int(row['best_epoch']), row['curve_diagnosis']


def _is(value):
    """Whether a cell of the results holds ``value``: a None comes back as NaN in a numeric column."""
    def match(v):
        if value is None:
            return v is None or (isinstance(v, float) and np.isnan(v))
        return v == value
    return match


def _same(alone, swept):
    return (np.isclose(alone[0], swept[0], rtol=1e-6, atol=1e-9) and alone[1:] == swept[1:])


@pytest.mark.parametrize('key', [k for k in BASE_PARAMS_SCHEMA if k not in NOT_SWEPT])
def test_every_setting_a_grid_accepts(key):
    values, context = _alternatives(key), CONTEXT.get(key, {})
    swept = _run(key, ..., context, grid={key: values}).runs
    for value in values:
        alone = _reading(_run(key, value, context).runs)
        mine = swept[swept[key].apply(_is(value))]
        assert _same(alone, _reading(mine)), (key, value, alone, _reading(mine))


SMALL = {'n_workers': 1, 'seed': 0, 'show_progress': False, 'model': {'hidden_dim': 16, 'embedding_dim': 8},
         'training': {'n_epochs': 6, 'batch_size': 32, 'patience': 100}}


def _with(base, **settings):
    """SMALL with ``settings`` added to its model or training settings."""
    out = {**base, 'model': dict(base['model']), 'training': dict(base['training'])}
    for name, value in settings.items():
        out['model' if name == 'input_scaling' else 'training'][name] = value
    return out


# The paths besides mode='sweep' that a setting of the preparation reaches: a mode that trains a difference
# of two networks, a named quantity whose rows run() builds, one that builds its rows itself, and transfer
# entropy, whose rows its own mode builds.
PATHS = {
    'conditional': lambda kw, grid=None: nmi.run(
        1000 * X, Y, mode='conditional', conditional=nmi.Conditional(w_data=W),
        sweep_grid=grid, **kw),
    'block_mi': lambda kw, grid=None: nmi.block_mi(1000 * XS, YS, window_size=3, sweep_grid=grid, **kw),
    'active_information_storage': lambda kw, grid=None: nmi.active_information_storage(
        XS, k=3, stride=3, sweep_grid=grid, **kw),
    # Shifting changes transfer entropy only once its networks train for longer than the others here.
    'transfer_entropy': lambda kw, grid=None: nmi.transfer_entropy(
        XL, YL, history_window=3, stride=3, sweep_grid=grid,
        **{**kw, 'training': {**kw['training'], 'n_epochs': 20}}),
}
CASES = [('conditional', 'input_scaling', ['auto', 'none']), ('block_mi', 'input_scaling', ['auto', 'none']),
         ('active_information_storage', 'shift_windows', [True, False]),
         ('transfer_entropy', 'shift_windows', [True, False])]


@pytest.mark.parametrize('path,key,values', CASES)
def test_preparation_settings_in_every_path(path, key, values):
    # The sweep starts from a base value that differs from the grid's first, so a value read once from the
    # base would show.
    base = _with(SMALL, **{key: values[-1]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        swept = PATHS[path](base, grid={key: values}).runs
        for value in values:
            alone = PATHS[path](_with(SMALL, **{key: value})).runs
            assert np.isclose(float(alone['mi'].iloc[0]),
                              float(swept[swept[key].apply(_is(value))]['mi'].iloc[0]), rtol=1e-6), (path, key, value)


def test_preparation_settings_are_settings_of_one_config():
    for key in PREPARATION_SETTINGS:
        assert key in BASE_PARAMS_SCHEMA
        assert OWNER.get(key) in ('model', 'training')
