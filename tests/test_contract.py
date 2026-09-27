# tests/test_contract.py
"""Every mode returns the same result shape, or refuses clearly.

Each mode is run with no grid, with repeats, with a configuration grid, with
both, with a processor parameter in the grid, and (where the mode has one)
with rigorous=True. A call either returns a Results that follows the contract
in archive/specs/results_contract.md, or raises a ValueError that says why.
Nothing is dropped or averaged silently.
"""
import itertools
import math
import warnings

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi import Model, Training, Split

N = 360
_MODEL = Model(embedding_dim=4, hidden_dim=8, n_layers=1)
_TRAINING = Training(n_epochs=1, batch_size=64, patience=1)
_COMMON = dict(model=_MODEL, training=_TRAINING, split=Split(mode='random'), n_workers=1,
               show_progress=False, seed=0)
_RIGOROUS_FIT = dict(gamma_range=range(1, 3), min_gamma_points=2)


def _data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, 2)).astype(np.float32)
    y = (0.8 * x + 0.5 * rng.standard_normal((N, 2))).astype(np.float32)
    w = rng.standard_normal((N, 2)).astype(np.float32)
    return x, y, w


def _mode_kwargs(mode, w, rigorous=False):
    fit = dict(rigorous=True, **_RIGOROUS_FIT) if rigorous else {}
    return {
        'estimate': {},
        'sweep': {},
        'rigorous': {'rigorous': nmi.Rigorous(**_RIGOROUS_FIT)},
        'lag': {'lag': nmi.Lag(lag_range=[0, 1])},
        'precision': {'precision': nmi.Precision(tau_grid=[0.0, 0.5])},
        'conditional': {'conditional': nmi.Conditional(w_data=w, **fit)},
        'interaction': {'interaction': nmi.Interaction(w_data=w, **fit)},
        'transfer': {'transfer': nmi.Transfer(history_window=2, **fit)},
        'pairwise': {},
        'dimensionality': {'dimensionality': nmi.Dimensionality(n_splits=2)},
    }[mode]


GRIDS = {
    'none': None,
    'repeats': {'run_id': [0, 1]},
    'configs': {'embedding_dim': [4, 8]},
    'both': {'embedding_dim': [4, 8], 'run_id': [0, 1]},
    'processor': {'window_size': [1, 2]},
}
MODES = ['estimate', 'sweep', 'rigorous', 'lag', 'precision', 'conditional', 'interaction',
         'transfer', 'pairwise', 'dimensionality']
# Combinations that must refuse, with the message that says why.
REFUSED = {
    ('dimensionality', 'repeats'): "n_splits",
    ('dimensionality', 'both'): "n_splits",
    ('transfer', 'processor'): "one time step wide",
}
# A mode that ignores the grid with a warning, and runs its base configuration once.
IGNORES_GRID = {'estimate', 'precision'}
AXIS = {'lag': ['lag'], 'precision': ['tau'], 'pairwise': ['ch_x', 'ch_y']}


def _expected_shape(mode, grid_name):
    grid = dict(GRIDS[grid_name] or {})
    if mode in IGNORES_GRID:
        grid = {}
    run_ids = grid.pop('run_id', [0])
    configs = list(itertools.product(*grid.values())) or [()]
    n_axis = {'lag': 2, 'precision': 2, 'pairwise': 1}.get(mode, 1)
    repeats = 2 if mode == 'dimensionality' else len(run_ids)
    return list(grid), len(configs), n_axis, repeats


def _run(mode, grid_name, rigorous=False):
    x, y, w = _data()
    grid = GRIDS[grid_name]
    kwargs = dict(_COMMON, **_mode_kwargs(mode, w, rigorous))
    if grid is not None:
        kwargs['sweep_grid'] = grid
    if grid_name == 'processor' or mode == 'transfer' and grid_name == 'processor':
        kwargs['processing'] = nmi.Processing(x='continuous', x_params={'window_size': 1},
                                              y='continuous')
    if mode == 'sweep' and grid is None:
        kwargs['sweep_grid'] = {'run_id': [0]}
    y_arg = y[:, :1] if mode == 'pairwise' else y
    x_arg = x[:, :1] if mode == 'pairwise' else x
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return nmi.run(x_arg, y_arg, mode=mode, **kwargs)


def _check(result, mode, grid_name):
    config_keys, n_configs, n_axis, n_repeats = _expected_shape(mode, grid_name)
    df, runs = result.dataframe, result.runs
    assert df is not None and runs is not None
    for col in ('config_id', 'mi_mean', 'mi_std', 'n_runs'):
        assert col in df.columns, col
    assert result.params['config_keys'] == config_keys
    assert result.params['axis_keys'] == AXIS.get(mode, [])
    assert sorted(df['config_id'].unique()) == list(range(n_configs))
    assert len(df) == n_configs * n_axis
    assert (df['n_runs'] == n_repeats).all(), df['n_runs'].tolist()
    assert len(runs) == n_configs * n_axis * n_repeats
    for key in config_keys:
        assert key in df.columns and key in runs.columns
    # mi_std is a spread of repeats: NaN with one repeat, never 0 by default.
    if n_repeats == 1:
        assert df['mi_std'].isna().all()
    if len(df) == 1:
        assert result.mi_estimate == pytest.approx(df['mi_mean'].iloc[0])
    else:
        assert result.mi_estimate is None
    assert set(result.details) <= set(range(n_configs))
    # Each configuration's mean is the mean of its own repeats.
    for _, row in df.iterrows():
        mask = runs['config_id'] == row['config_id']
        for axis in AXIS.get(mode, []):
            mask &= runs[axis] == row[axis]
        values = runs.loc[mask, 'mi'].astype(float)
        if values.notna().any():
            assert row['mi_mean'] == pytest.approx(values.mean())


@pytest.mark.parametrize('grid_name', list(GRIDS))
@pytest.mark.parametrize('mode', MODES)
def test_every_mode_follows_the_contract(mode, grid_name):
    refusal = REFUSED.get((mode, grid_name))
    if refusal is not None:
        with pytest.raises(ValueError, match=refusal):
            _run(mode, grid_name)
        return
    _check(_run(mode, grid_name), mode, grid_name)


@pytest.mark.parametrize('grid_name', ['none', 'repeats', 'configs'])
@pytest.mark.parametrize('mode', ['conditional', 'interaction', 'transfer'])
def test_rigorous_difference_quantities_follow_the_contract(mode, grid_name):
    result = _run(mode, grid_name, rigorous=True)
    _check(result, mode, grid_name)
    config_keys, n_configs, _, n_repeats = _expected_shape(mode, grid_name)
    # One extrapolation per repeat; the interval is reported only for a single repeat.
    assert 'is_reliable' in result.runs.columns
    if n_repeats == 1:
        assert result.dataframe['mi_error'].notna().all()
    else:
        assert result.dataframe['mi_error'].isna().all()
    for cid in range(n_configs):
        assert 'trainings' in result.details[cid]


def test_rigorous_repeats_are_separate_extrapolations():
    result = _run('rigorous', 'repeats')
    assert len(result.runs) == 2
    ladder = result.details[0]['trainings']
    assert sorted(ladder['run_id'].unique()) == [0, 1]
    assert math.isnan(result.dataframe['mi_error'].iloc[0])


@pytest.mark.parametrize('mode', ['rigorous', 'conditional', 'sweep', 'lag', 'pairwise',
                                  'dimensionality'])
def test_a_processor_grid_windows_each_setting(mode):
    """Each processor setting windows the data its own way: at window_size=2 the
    networks see half as many windows as at 1. A mode that windowed once and
    only relabelled the rows would report the same sizes for both."""
    result = _run(mode, 'processor')
    assert list(result.dataframe['window_size'].unique()) == [1, 2]
    if 'eval_size' in result.runs.columns:
        sizes = result.runs.groupby('window_size')['eval_size'].max()
    else:
        sizes = {w: result.details[cid]['trainings']['eval_size'].max()
                 for cid, w in enumerate((1, 2))}
    assert sizes[2] < sizes[1]
    if mode == 'lag':
        built = result.runs.groupby('window_size')['n_windows_built'].max()
        assert built[2] < built[1]


def test_a_processor_key_without_a_processor_is_refused():
    x, y, _ = _data()
    with pytest.raises(ValueError, match="no stream in this call has a processor"):
        nmi.run(x, y, mode='sweep', sweep_grid={'window_size': [1, 2]}, **_COMMON)


def test_rigorous_difference_quantities_refuse_embeddings():
    x, y, w = _data()
    for mode in ('conditional', 'rigorous'):
        with pytest.raises(ValueError, match="not available"):
            nmi.run(x, y, mode=mode, output=nmi.Output(return_embeddings=True),
                    **_COMMON, **_mode_kwargs(mode, w))


@pytest.mark.parametrize('mode', ['estimate', 'sweep', 'lag', 'dimensionality'])
def test_single_network_modes_keep_embeddings_per_repeat(mode):
    x, y, w = _data()
    kwargs = dict(_COMMON, **_mode_kwargs(mode, w))
    kwargs['sweep_grid'] = None if mode in ('estimate', 'lag', 'dimensionality') else {'run_id': [0, 1]}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x, y, mode=mode, output=nmi.Output(return_embeddings=True), **kwargs)
    embeddings = result.details[0]['embeddings']
    assert embeddings
    for entry in embeddings.values():
        assert entry['embeddings_x'].shape[0] > 0
        assert not any(k.startswith('embedding_dim') or k == 'embedding_model' for k in entry)


def test_lag_keeps_embeddings_per_lag_and_repeat():
    x, y, w = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x, y, mode='lag', output=nmi.Output(return_embeddings=True),
                         sweep_grid={'run_id': [0, 1]}, **_COMMON, **_mode_kwargs('lag', w))
    assert set(result.details[0]['embeddings']) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_n_workers_on_a_single_network_warns():
    x, y, _ = _data()
    with pytest.warns(UserWarning, match="n_workers=2 has no effect"):
        nmi.run(x, y, mode='estimate', **dict(_COMMON, n_workers=2))


@pytest.mark.parametrize('grid, mode, extra, match', [
    ({'banana': [1, 2]}, 'sweep', {}, "not a setting NeuralMI reads"),
    ({'mode': ['random']}, 'sweep', {}, "swept under the name 'split_mode'"),
    ({'history_window': [1, 3]}, 'transfer', {'transfer': nmi.Transfer(history_window=1)},
     "a setting of mode='transfer'"),
    ({'n_splits': [2, 3]}, 'dimensionality', {}, "a setting of mode='dimensionality'"),
    ({'output_units': ['nats']}, 'sweep', {}, "units of the whole result"),
])
def test_a_grid_key_that_changes_nothing_is_refused(grid, mode, extra, match):
    x, y, _ = _data()
    with pytest.raises(ValueError, match=match):
        nmi.run(x, y, mode=mode, sweep_grid=grid, **_COMMON, **extra)


def test_renamed_config_fields_are_swept_under_their_schema_names():
    x, y, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x, y, mode='sweep', sweep_grid={'split_mode': ['blocked', 'random']},
                         **{**_COMMON, 'split': None})
    assert result.dataframe['split_mode'].tolist() == ['blocked', 'random']
