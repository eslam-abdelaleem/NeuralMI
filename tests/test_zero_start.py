"""Model(zero_start=True): every score of a critic that the library builds starts at exactly zero.

The separable critic starts half of each side's readout rows at zero, X's first half and Y's second, so the two
embeddings start in complementary halves. The hybrid and concat critics start the last layer of the network that
scores the pair at zero. A shared encoder, whose one readout could start at zero only as a whole, keeps its start,
and so does a custom critic. Every encoder gives the layer that produces its embedding as ``readout``.
"""
import numpy as np
import pytest
import torch

from neural_mi.defaults import BASE_PARAMS_SCHEMA
from neural_mi.utils import _EMBEDDING_CLASSES, build_critic

ENCODERS = ('mlp', 'cnn', 'gru', 'lstm', 'tcn', 'transformer', 'lru', 'deepsets')


def _params(**overrides):
    params = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    params.update(input_dim_x=15, input_dim_y=15, n_channels_x=3, n_channels_y=3, hidden_dim=16, embedding_dim=8,
                  input_scaling='none')
    params.update(overrides)
    return params


def _scores(critic, model='mlp', train=False):
    torch.manual_seed(1)
    x = torch.rand(16, 3, 5) + (0.5 if model == 'deepsets' else 0.0)
    critic.train(train)
    with torch.no_grad():
        scores, _ = critic(x, x)
    return scores


@pytest.mark.parametrize('model', ENCODERS)
def test_every_encoder_gives_its_readout(model):
    critic = build_critic('separable', _params(embedding_model=model))
    for net in (critic.embedding_net_x, critic.embedding_net_y):
        assert isinstance(net.readout, torch.nn.Linear) and net.readout.out_features == 8


@pytest.mark.parametrize('model', ENCODERS)
def test_the_separable_critic_starts_at_zero_with_every_encoder(model):
    critic = build_critic('separable', _params(embedding_model=model))
    assert torch.count_nonzero(_scores(critic, model)) == 0
    assert torch.count_nonzero(_scores(critic, model, train=True)) == 0
    half = critic.embedding_net_x.readout.out_features // 2
    assert torch.count_nonzero(critic.embedding_net_x.readout.weight[:half]) == 0
    assert torch.count_nonzero(critic.embedding_net_y.readout.weight[half:]) == 0
    assert torch.count_nonzero(critic.embedding_net_x.readout.weight[half:]) > 0


@pytest.mark.parametrize('critic_type', ['hybrid', 'concat'])
def test_a_network_that_scores_the_pair_starts_at_zero(critic_type):
    critic = build_critic(critic_type, _params())
    assert torch.count_nonzero(_scores(critic)) == 0


def test_scaled_and_variational_encoders_start_at_zero():
    for overrides in ({'input_scaling': 'auto', 'input_scaling_x': ('channel',), 'input_scaling_y': 'channel'},
                      {'use_variational': True}):
        critic = build_critic('separable', _params(**overrides))
        assert torch.count_nonzero(_scores(critic)) == 0, overrides


def test_a_shared_encoder_and_zero_start_false_keep_their_start():
    torch.manual_seed(0)
    assert not build_critic('separable', _params(shared_encoder=True)).start_at_zero()
    assert torch.count_nonzero(_scores(build_critic('separable', _params(shared_encoder=True)))) > 0
    assert torch.count_nonzero(_scores(build_critic('separable', _params(zero_start=False)))) > 0


def test_the_zeroed_rows_learn():
    """Each zeroed row receives a gradient from the other side's live half, so one step moves it."""
    critic = build_critic('separable', _params())
    x, y = torch.rand(32, 3, 5), torch.rand(32, 3, 5)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    critic.train()
    scores, _ = critic(x, y)
    loss = -(torch.diagonal(scores) - torch.logsumexp(scores, dim=1)).mean()
    loss.backward()
    optimizer.step()
    half = critic.embedding_net_x.readout.out_features // 2
    assert torch.count_nonzero(critic.embedding_net_x.readout.weight[:half]) > 0
    assert torch.count_nonzero(_scores(critic)) > 0


def test_a_custom_critic_starts_as_its_recipe_builds_it():
    from neural_mi.models.critics import SeparableCritic

    def recipe(input_dim_x, input_dim_y, embedding_dim):
        net = _EMBEDDING_CLASSES['mlp'](input_dim_x, 16, embedding_dim, 2)
        return SeparableCritic(embedding_net_x=net, embedding_net_y=_EMBEDDING_CLASSES['mlp'](input_dim_y, 16,
                                                                                            embedding_dim, 2))
    critic = build_critic('separable', _params(custom_critic=recipe))
    assert torch.count_nonzero(_scores(critic)) > 0
