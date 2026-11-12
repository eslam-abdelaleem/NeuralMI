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
    # These tests read the encoders themselves. The scaling in front of them has tests of its own
    # (test_input_scaling.py).
    params.update(input_dim_x=6, input_dim_y=6, n_channels_x=2, n_channels_y=2, hidden_dim=8,
                  embedding_dim=4, n_layers=2, input_scaling='none')
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

    @pytest.mark.parametrize('model', ['mlp', 'cnn', 'gru', 'lstm', 'lru', 'tcn', 'transformer', 'deepsets'])
    def test_norm_layer_reaches_every_encoder(self, model):
        """The MLP and convolutional encoders normalise each hidden layer, DeepSets the hidden layer of rho,
        and the recurrent and attention encoders the features that the readout maps to the embedding."""
        def count(kind, norm):
            return len(_modules(build_critic('separable', _params(embedding_model=model, norm_layer=norm))
                                .embedding_net_x, kind))
        assert count(nn.BatchNorm1d, 'none') == 0 < count(nn.BatchNorm1d, 'batch')
        assert count(nn.LayerNorm, 'none') < count(nn.LayerNorm, 'layer')

    def test_norm_layer_reaches_the_cnn2d_encoder(self):
        from neural_mi.models.embeddings import CNN2D
        assert not _modules(CNN2D(1, 8, 4, 2), nn.LayerNorm)
        assert len(_modules(CNN2D(1, 8, 4, 2, norm_layer='layer'), nn.LayerNorm)) == 3

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


class TestTheActivationReachesEveryRectifiedLayer:
    """Model(activation=...) sets every rectified hidden layer: in the encoders that have one, the MLP heads
    and the decoders that mirror the encoders. The GRU and LSTM cells and the LRU block keep their own."""

    ARMS = (('leaky_relu', nn.LeakyReLU), ('relu', nn.ReLU))

    @staticmethod
    def _rectifiers(net):
        return {type(m) for m in net.modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU))}

    def test_the_default_is_leaky_relu(self):
        assert BASE_PARAMS_SCHEMA['activation']['default'] == 'leaky_relu'

    @pytest.mark.parametrize('model', ['mlp', 'cnn', 'tcn', 'transformer', 'deepsets'])
    def test_every_encoder_with_rectified_layers(self, model):
        for name, kind in self.ARMS:
            critic = build_critic('separable', _params(embedding_model=model, activation=name))
            assert self._rectifiers(critic) == {kind}, (model, name)

    def test_the_cnn2d_encoder(self):
        from neural_mi.models.embeddings import CNN2D
        for name, kind in self.ARMS:
            assert self._rectifiers(CNN2D(1, 8, 4, 2, activation=name)) == {kind}

    @pytest.mark.parametrize('critic', ['hybrid', 'concat'])
    def test_the_mlp_heads(self, critic):
        for name, kind in self.ARMS:
            built = build_critic(critic, _params(embedding_model='gru', activation=name))
            assert self._rectifiers(built.decision_head if critic == 'hybrid' else built.embedding_net) == {kind}

    def test_the_branched_head(self):
        for name, kind in self.ARMS:
            params = _params(embedding_model='gru', _difference_term=True, activation=name)
            assert self._rectifiers(build_critic('separable', params).embedding_net_x.head) == {kind}

    @pytest.mark.parametrize('model', ['mlp', 'cnn', 'cnn2d', 'tcn', 'transformer'])
    def test_the_decoders(self, model):
        from neural_mi.models.decoders import build_decoder
        for name, kind in self.ARMS:
            decoder = build_decoder(model, 4, 8, 2, 16, n_layers=2, output_activation='linear', kernel_size=3,
                                    nhead=4, dropout=0.0, activation=name, height=4, width=4)
            assert self._rectifiers(decoder) == {kind}, (model, name)


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

    def test_norm_layer_on_a_recurrent_separable_critic_is_read(self):
        assert self._messages(embedding_model='gru', norm_layer='layer') == []

    def test_the_activation_on_a_recurrent_encoder(self):
        messages = self._messages(embedding_model='lstm', activation='relu')
        assert messages and 'activation (applies to' in messages[0]

    def test_the_concat_critic_builds_no_encoder(self):
        messages = self._messages(critic_type='concat', embedding_model='cnn', kernel_size=5)
        assert messages and 'kernel_size' in messages[0] and 'embedding_model' in messages[0]


class TestTheNoteOnALargeFirstLayer:
    """A first embedding layer larger than Model(large_first_layer=...) is noted at the info level."""

    def test_the_setting_sets_the_size_noted(self, caplog):
        import logging
        params = _params(input_dim_x=1000, input_dim_y=6, n_channels_x=1, hidden_dim=600)
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            build_critic('separable', params)
        assert 'Large first embedding layer detected' in caplog.text
        assert 'more than Model(large_first_layer=500000)' in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            build_critic('separable', {**params, 'large_first_layer': 10 ** 6})
        assert 'Large first embedding layer' not in caplog.text
