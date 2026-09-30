"""The hybrid and concat critics score all pairs in blocks through the head's
first linear layer. The scores and their gradients match the pair-by-pair route."""
import copy

import pytest
import torch

from neural_mi.defaults import BASE_PARAMS_SCHEMA
from neural_mi.models import critics
from neural_mi.utils import build_critic


def _params(critic, **overrides):
    p = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    p.update(critic_type=critic, input_dim_x=12, input_dim_y=9, n_channels_x=12,
             n_channels_y=9, embedding_dim=4, hidden_dim=32, n_layers=2)
    p.update(overrides)
    return p


def _both_routes(critic, n, monkeypatch, **overrides):
    torch.manual_seed(0)
    fast = build_critic(critic, _params(critic, **overrides)).eval()
    slow = copy.deepcopy(fast)
    x, y = torch.randn(n, 12), torch.randn(n, 9)
    weights = torch.randn(n, n)
    s_fast, _ = fast(x, y)
    (s_fast * weights).sum().backward()
    with monkeypatch.context() as m:
        m.setattr(critics, '_is_plain_mlp', lambda net: False)
        s_slow, _ = slow(x, y)
    (s_slow * weights).sum().backward()
    return fast, slow, s_fast, s_slow


@pytest.mark.parametrize('critic, overrides', [
    ('hybrid', {}), ('hybrid', {'bias': False}), ('hybrid', {'n_layers': 3}), ('concat', {})],
    ids=['hybrid', 'hybrid-no-bias', 'hybrid-deeper-head', 'concat'])
@pytest.mark.parametrize('n', [5, 130])
def test_block_scoring_matches_the_pair_by_pair_route(critic, overrides, n, monkeypatch):
    fast, slow, s_fast, s_slow = _both_routes(critic, n, monkeypatch, **overrides)
    assert s_fast.shape == (n, n)
    torch.testing.assert_close(s_fast, s_slow, rtol=1e-5, atol=1e-4 * s_slow.abs().max().item())
    for p_fast, p_slow in zip(fast.parameters(), slow.parameters()):
        if p_slow.grad is not None:
            scale = p_slow.grad.abs().max().item() or 1.0
            torch.testing.assert_close(p_fast.grad, p_slow.grad, rtol=1e-4, atol=1e-4 * scale)


def test_a_large_evaluation_set_is_scored_in_several_blocks(monkeypatch):
    monkeypatch.setattr(critics, '_PAIR_BLOCK_ELEMENTS', 5_000)
    fast, slow, s_fast, s_slow = _both_routes('hybrid', 200, monkeypatch)
    torch.testing.assert_close(s_fast, s_slow, rtol=1e-5, atol=1e-4 * s_slow.abs().max().item())


def test_a_variational_concat_critic_keeps_the_pair_by_pair_route():
    critic = build_critic('concat', _params('concat', use_variational=True)).eval()
    scores, kl = critic(torch.randn(6, 12), torch.randn(6, 9))
    assert scores.shape == (6, 6)
    assert torch.isfinite(kl)


@pytest.mark.parametrize('norm, has_layer_norm', [('auto', False), ('none', False), ('layer', True)])
def test_auto_means_no_normalisation_outside_the_dimensionality_mode(norm, has_layer_norm):
    critic = build_critic('hybrid', _params('hybrid', norm_layer=norm))
    found = any(isinstance(m, torch.nn.LayerNorm) for m in critic.modules())
    assert found == has_layer_norm
