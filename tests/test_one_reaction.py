# tests/test_one_reaction.py
"""One reaction to each kind of thing a call cannot do, whatever the mode or critic.

A setting the call does not read is ignored, with one warning before any work.
A request the call cannot serve, such as embeddings from a critic that gives
none, is refused before any network trains. A sweep_grid that would vary
nothing is refused before any work. Whether a critic gives an embedding of
each side is declared once, by the critic, and every check asks it.
"""
import warnings

import numpy as np
import pytest

import neural_mi as nmi
from neural_mi.models.critics import CRITICS, embeds_each_side
from neural_mi.run import _warn_ineffective_settings

RNG = np.random.default_rng(0)
X = RNG.standard_normal((400, 2)).astype(np.float32)
Y = (X + RNG.standard_normal((400, 2))).astype(np.float32)
QUICK = dict(training=nmi.Training(n_epochs=1), show_progress=False, seed=0, n_workers=1)
MODE_CONFIGS = {'sweep': {}, 'rigorous': {}, 'lag': {'lag': nmi.Lag(lag_range=[0, 1])}, 'pairwise': {}}


def _quietly(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return nmi.run(*args, **kwargs)


def test_each_critic_declares_whether_it_embeds_each_side():
    assert {name: cls.embeds_each_side for name, cls in CRITICS.items()} == {
        'separable': True, 'hybrid': True, 'concat': False}
    assert not embeds_each_side('concat') and embeds_each_side('hybrid')


def test_settings_the_concat_critic_does_not_read_are_ignored_with_one_warning():
    """It builds no encoder, so X's and Y's encoder settings and shared_encoder have no effect."""
    model = nmi.Model(critic_type='concat', embedding_dim=8, embedding_model_y='mlp', shared_encoder=True)
    with pytest.warns(UserWarning, match=r"embedding_dim, embedding_model_y, shared_encoder \(critic_type='concat'"):
        result = nmi.run(X, Y, model=model, **QUICK)
    assert result.runs is not None


@pytest.mark.parametrize('asked', [{'output': nmi.Output(return_embeddings=True)},
                                   {'output': nmi.Output(track_embeddings=8)},
                                   {'output': nmi.Output(track_spectral_history=True)},
                                   {'model': nmi.Model(critic_type='concat', use_decoder=True)}],
                         ids=['returned', 'tracked', 'spectrum', 'decoder'])
def test_a_request_for_embeddings_the_critic_cannot_give_is_refused(asked):
    kwargs = {'model': nmi.Model(critic_type='concat'), **asked}
    with pytest.raises(ValueError, match="needs the critic's embedding of each side"):
        nmi.run(X, Y, **kwargs, **QUICK)


def test_the_concat_critic_reports_no_spectrum_of_embeddings_it_does_not_have():
    concat = _quietly(X, Y, model=nmi.Model(critic_type='concat'), **QUICK)
    separable = _quietly(X, Y, **QUICK)
    assert 'pr_eig' in separable.runs.columns and 'pr_eig' not in concat.runs.columns


@pytest.mark.parametrize('mode', list(MODE_CONFIGS))
def test_a_grid_over_a_setting_nothing_reads_is_refused_in_every_mode(mode):
    """beta applies only with use_variational=True. A fixed beta is warned of, and a grid over it,
    which would train the same networks in every configuration, is refused the same way in every mode."""
    with pytest.warns(UserWarning, match=r"beta \(use_variational=True only\)"):
        _warn_ineffective_settings(mode, nmi.Model(beta=0.5), None)
    with pytest.raises(ValueError, match="sweep_grid varies 'beta', which nothing in this call reads"):
        nmi.run(X, None if mode == 'pairwise' else Y, mode=mode, sweep_grid={'beta': [0.5, 1.0]},
                **MODE_CONFIGS[mode], **QUICK)


def test_a_grid_runs_when_some_configuration_reads_the_setting():
    """The separable critic reads embedding_dim and the concat critic does not."""
    result = _quietly(X, Y, mode='sweep', sweep_grid={'critic_type': ['separable', 'concat'],
                                                      'embedding_dim': [4, 8]}, **QUICK)
    assert len(result.dataframe) == 4


@pytest.mark.parametrize('mode', ['estimate', 'precision'])
def test_a_mode_that_trains_one_network_refuses_a_grid(mode):
    kwargs = {'precision': nmi.Precision(tau_grid=[0.0, 0.01])} if mode == 'precision' else {}
    with pytest.raises(ValueError, match="runs no sweep_grid"):
        nmi.run(X, Y, mode=mode, sweep_grid={'run_id': [0, 1]}, **kwargs, **QUICK)


@pytest.mark.parametrize('mode, key', [('lag', 'lag'), ('dimensionality', 'embedding_dim'),
                                       ('dimensionality', 'run_id')])
def test_a_grid_over_a_key_the_mode_sets_itself_is_refused(mode, key):
    kwargs = {'lag': nmi.Lag(lag_range=[0, 1])} if mode == 'lag' else {}
    with pytest.raises(ValueError, match=f"sets '{key}' itself"):
        nmi.run(X, Y, mode=mode, sweep_grid={key: [1, 2]}, **kwargs, **QUICK)
