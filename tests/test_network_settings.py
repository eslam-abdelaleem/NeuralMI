"""Every Model setting reaches every network the library builds that has the part it names.

Each built-in encoder declares the settings it reads (``BaseEmbedding.settings``),
and the MLPs built around the encoders (the hybrid head, the concat network,
the branched head, the pretrained backbone's head) read the MLP settings. A
setting that no network of the call reads warns.
"""
import warnings

import numpy as np
import pytest
import torch.nn as nn

import neural_mi as nmi
from neural_mi.defaults import BASE_PARAMS_SCHEMA
from neural_mi.utils import build_critic


def _params(**overrides):
    params = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    params.update(input_dim_x=6, input_dim_y=6, n_channels_x=2, n_channels_y=2, hidden_dim=8,
                  embedding_dim=4, n_layers=2)
    params.update(overrides)
    return params


def _modules(net, kind):
    return [m for m in net.modules() if isinstance(m, kind)]


def _spectral(net):
    return [m for m in net.modules() if isinstance(m, nn.Linear) and hasattr(m, 'weight_orig')]


class TestTheHeadsReadTheMLPSettings:

    @pytest.mark.parametrize('critic', ['hybrid', 'concat'])
    def test_hybrid_head_and_concat_network(self, critic):
        on = build_critic(critic, _params(dropout=0.3, norm_layer='layer', use_spectral_norm=True))
        off = build_critic(critic, _params(use_spectral_norm=False))
        head = on.decision_head if critic == 'hybrid' else on.embedding_net
        assert _modules(head, nn.Dropout) and _modules(head, nn.LayerNorm) and _spectral(head)
        assert not _spectral(off.decision_head if critic == 'hybrid' else off.embedding_net)

    def test_branched_head_over_a_recurrent_branch(self):
        params = _params(embedding_model='gru', input_dim_x=6, n_channels_x=2, _difference_term=True,
                         dropout=0.3, norm_layer='layer', use_spectral_norm=False)
        head = build_critic('separable', params).embedding_net_x.head
        assert _modules(head, nn.Dropout) and _modules(head, nn.LayerNorm) and not _spectral(head)


class TestTheEncodersReadTheirSettings:

    @pytest.mark.parametrize('model, kind', [('tcn', nn.Dropout), ('transformer', nn.Dropout)])
    def test_dropout_reaches_tcn_and_transformer(self, model, kind):
        net = build_critic('separable', _params(embedding_model=model, dropout=0.0)).embedding_net_x
        assert all(m.p == 0.0 for m in _modules(net, kind))
        net = build_critic('separable', _params(embedding_model=model, dropout=0.25)).embedding_net_x
        assert {m.p for m in _modules(net, kind)} == {0.25}

    def test_dropout_reaches_a_stacked_recurrent_encoder(self):
        net = build_critic('separable', _params(embedding_model='gru', dropout=0.25, n_layers=2)).embedding_net_x
        assert net.gru.dropout == 0.25
        net = build_critic('separable', _params(embedding_model='gru', dropout=0.25, n_layers=1)).embedding_net_x
        assert net.gru.dropout == 0.0

    def test_a_custom_class_gets_the_settings_it_names(self):
        seen = {}

        class Custom(nn.Module):
            def __init__(self, input_dim, hidden_dim, embedding_dim, n_layers, norm_layer=None, kernel_size=3):
                super().__init__()
                seen.update(norm_layer=norm_layer, kernel_size=kernel_size)
                self.net = nn.Linear(input_dim, embedding_dim)

            def forward(self, x):
                return self.net(x.reshape(x.shape[0], -1))

        build_critic('separable', _params(custom_embedding_cls=Custom, norm_layer='batch', kernel_size=5))
        assert seen == {'norm_layer': 'batch', 'kernel_size': 5}


class TestASettingNoNetworkReadsWarns:

    @staticmethod
    def _messages(mode='estimate', **model):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((200, 2, 3)).astype(np.float32)
        y = (x + rng.standard_normal(x.shape)).astype(np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            nmi.run(x, y, model=nmi.Model(hidden_dim=8, embedding_dim=4, **model),
                    training=nmi.Training(n_epochs=1, batch_size=64), show_progress=False, seed=0)
        return [str(w.message) for w in caught if 'no effect' in str(w.message)]

    def test_spectral_norm_on_a_recurrent_separable_critic(self):
        messages = self._messages(embedding_model='gru', use_spectral_norm=False)
        assert messages and 'use_spectral_norm (applies to' in messages[0]

    def test_a_hybrid_head_reads_it(self):
        assert self._messages(embedding_model='gru', use_spectral_norm=False, critic_type='hybrid') == []

    def test_dropout_on_a_single_recurrent_layer(self):
        messages = self._messages(embedding_model='gru', dropout=0.2, n_layers=1)
        assert messages and 'n_layers=1' in messages[0]
        assert self._messages(embedding_model='gru', dropout=0.2, n_layers=2) == []

    def test_the_concat_critic_builds_no_encoder(self):
        messages = self._messages(critic_type='concat', embedding_model='cnn', kernel_size=5)
        assert messages and 'kernel_size' in messages[0] and 'embedding_model' in messages[0]
