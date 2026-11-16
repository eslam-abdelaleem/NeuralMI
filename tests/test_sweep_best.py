"""result.best: the configurations of a sweep ranked by held-out MI.

The best configuration has the highest held-out MI, averaged over its repeats. The configurations whose
held-out MI lies within Training(top_se=...) standard errors of it follow, highest first, as the band at the
top of one curve holds the epochs within top_se standard errors of its maximum.
"""
import warnings

import pandas as pd
import pytest

import neural_mi as nmi
from neural_mi.results import Results


def _sweep(held, top_se=1.0, units='bits', eval_size=100):
    """A sweep over hidden_dim whose configurations read the held-out values ``held``."""
    n = len(held)
    dataframe = pd.DataFrame({'config_id': range(n), 'hidden_dim': [8 * 2 ** i for i in range(n)],
                              'mi_mean': [h + 0.1 for h in held], 'test_mi_mean': held})
    runs = pd.DataFrame({'config_id': range(n), 'eval_size': [eval_size] * n})
    params = {'config_keys': ['hidden_dim'], 'output_units': units, 'top_se': top_se}
    return Results(mode='sweep', params=params, dataframe=dataframe, runs=runs)


def test_the_best_comes_first_and_those_within_top_se_follow():
    # 1 bit on 100 held-out rows has a standard error of sqrt(2 ln 2 / 100) nats, 0.17 bits.
    best = _sweep([0.5, 1.0, 0.9, 0.85]).best
    assert best['config_id'].tolist() == [1, 2, 3]
    assert list(best.columns) == ['config_id', 'hidden_dim', 'test_mi_mean', 'mi_mean']


def test_top_se_sets_how_close_a_configuration_must_be():
    assert _sweep([0.5, 1.0, 0.9, 0.85], top_se=0.0).best['config_id'].tolist() == [1]
    assert _sweep([0.5, 1.0, 0.9, 0.85], top_se=10.0).best['config_id'].tolist() == [1, 2, 3, 0]


def test_the_standard_error_is_that_of_the_value_in_nats():
    """1 nat on 100 rows has a standard error of 0.14 nats, so 0.85 nats lies outside it."""
    assert _sweep([0.5, 1.0, 0.9, 0.85], units='nats').best['config_id'].tolist() == [1, 2]


def test_the_summary_names_the_best_and_those_within_top_se(capsys):
    _sweep([0.5, 1.0, 0.9]).summary()
    out = capsys.readouterr().out
    assert 'Best by held-out MI : hidden_dim=16 (1.0000 bits held out)' in out
    assert 'Within Training(top_se=1) standard errors of it:' in out
    assert 'hidden_dim=32 (0.9000 bits held out)' in out and 'hidden_dim=8 ' not in out


def test_only_a_sweep_has_a_best():
    with pytest.raises(ValueError, match="mode='sweep'"):
        Results(mode='estimate', params={}).best


def test_a_sweep_through_run_ranks_its_configurations():
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=600, dim=3, mi=1.0, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x, y, mode='sweep', sweep_grid={'hidden_dim': [4, 32], 'run_id': range(2)},
                         split=nmi.Split(mode='random'), training=nmi.Training(n_epochs=10), seed=0,
                         show_progress=False)
    best = result.best
    assert best['test_mi_mean'].iloc[0] == result.dataframe['test_mi_mean'].max()
    assert best['config_id'].iloc[0] in result.config_ids
