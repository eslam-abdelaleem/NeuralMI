"""Model(linear_path=True): an MLP encoder adds a linear map from its input to its embedding.

The map has spectral normalisation and is multiplied by one trainable number that starts at zero, so it starts
closed and opens as training finds a use for it. Only MLP encoders take it: the MLP heads that score or join
embeddings take none, and a setting that no network of the call reads is reported.
"""
import warnings

import numpy as np
import torch

import neural_mi as nmi
from neural_mi.defaults import BASE_PARAMS_SCHEMA
from neural_mi.models.embeddings import MLP
from neural_mi.utils import build_critic


def _params(**overrides):
    params = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    params.update(input_dim_x=15, input_dim_y=15, n_channels_x=3, n_channels_y=3, hidden_dim=16, embedding_dim=8,
                  input_scaling='none')
    params.update(overrides)
    return params


def _paths(critic):
    return {name for name, module in critic.named_modules() if isinstance(module, MLP) and module.path is not None}


def test_mlp_encoders_take_the_path_by_default_and_the_heads_never():
    assert BASE_PARAMS_SCHEMA['linear_path']['default'] is True
    assert _paths(build_critic('separable', _params())) == {'embedding_net_x', 'embedding_net_y'}
    assert _paths(build_critic('hybrid', _params())) == {'embedding_net_x', 'embedding_net_y'}
    assert _paths(build_critic('concat', _params())) == set()
    assert _paths(build_critic('separable', _params(linear_path=False))) == set()
    branched = build_critic('separable', _params(_difference_term=True, n_channels_x=(1, 1), input_dim_x=(5, 5)))
    assert all(name.startswith('embedding_net_x.branches') or name == 'embedding_net_y'
               for name in _paths(branched))


def test_the_path_starts_closed_and_changes_no_output_at_the_start():
    torch.manual_seed(0)
    net = MLP(15, 16, 8, 2, linear_path=True).eval()          # eval: spectral norm does not update between passes
    x = torch.randn(32, 15)
    assert net.path_scale.item() == 0.0
    with torch.no_grad():
        assert torch.equal(net(x), net.output_layer(net.network(x)))


def test_the_path_is_spectrally_normalised_and_opens_with_training():
    torch.manual_seed(0)
    net = MLP(200, 16, 8, 2, linear_path=True)
    x = torch.randn(64, 200)
    net.train()
    for _ in range(20):                                  # one power iteration per pass
        net(x)
    assert float(torch.linalg.matrix_norm(net.path.weight, ord=2)) <= 1.1
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    net(x).pow(2).sum().backward()
    optimizer.step()
    assert net.path_scale.item() != 0.0


def test_an_mlp_without_hidden_layers_has_no_path():
    """An MLP with no hidden layer is linear already."""
    assert MLP(15, 16, 8, 0, linear_path=True).path is None


def test_the_concat_critics_block_scoring_keeps_an_open_path(monkeypatch):
    """A custom concat critic may give its network a path. The block route scores every pair with it, as the
    pair-by-pair route does."""
    from neural_mi.models import critics
    torch.manual_seed(0)
    net = MLP(21, 16, 1, 2, linear_path=True).eval()
    net.path_scale.data.fill_(0.7)
    critic = critics.ConcatCritic(embedding_net=net).eval()
    x, y = torch.randn(7, 12), torch.randn(7, 9)
    with torch.no_grad():
        fast, _ = critic(x, y)
        monkeypatch.setattr(critics, '_is_plain_mlp', lambda net: False)
        slow, _ = critic(x, y)
    torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)


def test_a_setting_no_network_reads_is_reported():
    """Turning the path off for an encoder that has none changes nothing, and the call says so."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2, 6)).astype(np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        nmi.run(x, x, model=nmi.Model(embedding_model='cnn', linear_path=False, hidden_dim=8, embedding_dim=4),
                training=nmi.Training(n_epochs=1), show_progress=False, seed=0)
    assert any('linear_path' in str(w.message) and "embedding_model 'mlp'" in str(w.message) for w in caught)
