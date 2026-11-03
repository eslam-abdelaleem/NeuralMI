"""Model(input_scaling='standard') scales every encoder's input by the training rows, and binned spikes stay counts.

The statistics are buffers of the network: fitted from the training rows before
the first epoch, saved with the network and applied by ``extract_embeddings``.
"""
import warnings

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.defaults import BASE_PARAMS_SCHEMA
from neural_mi.models.embeddings import ChannelScaling, InputScaling
from neural_mi.utils import build_critic


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    return x, (x + 0.7 * rng.standard_normal((n, 2))).astype(np.float32)


def _estimate(x, y, **model):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return nmi.run(x, y, model=nmi.Model(hidden_dim=16, embedding_dim=4, **model),
                       training=nmi.Training(n_epochs=4, batch_size=64), show_progress=False, seed=0,
                       n_workers=1)


def test_an_estimate_no_longer_depends_on_the_units():
    x, y = _data()
    unit = _estimate(x, y, input_scaling='standard').mi_estimate
    for scale in (1e3, 1e-3):
        assert _estimate(x * scale, y * scale, input_scaling='standard').mi_estimate == pytest.approx(unit, abs=1e-3)


def test_the_statistics_are_fitted_saved_and_applied_on_rebuild(tmp_path):
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x * 1e3 + 5, y, model=nmi.Model(hidden_dim=8, embedding_dim=4, input_scaling='standard'),
                         training=nmi.Training(n_epochs=2, batch_size=64,
                                               save_best_model_path=str(tmp_path / 'net.pt')),
                         show_progress=False, seed=0, n_workers=1)
    path = result.runs['model_path'].iloc[0]
    state = torch.load(path, weights_only=False)['state_dict']
    shift, scale = state['embedding_net_x.scalers.0.shift'], state['embedding_net_x.scalers.0.scale']
    assert torch.allclose(shift, torch.full((2,), 5.0), atol=150) and bool((scale > 500).all())
    zx, _ = nmi.extract_embeddings(path, x[:8, :, None] * 1e3 + 5, y[:8, :, None])
    assert np.isfinite(zx).all() and np.abs(zx).max() < 1e3


def _params(**overrides):
    params = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    params.update(input_dim_x=6, input_dim_y=6, n_channels_x=2, n_channels_y=2, hidden_dim=8, embedding_dim=4,
                  input_scaling='standard')
    params.update(overrides)
    return params


@pytest.mark.parametrize('overrides, where', [
    ({}, 'encoders'), ({'critic_type': 'concat'}, 'concat'), ({'embedding_model': 'gru'}, 'encoders'),
    ({'_difference_term': True, 'n_channels_x': (1, 1), 'input_dim_x': (3, 3)}, 'encoders')])
def test_every_encoder_and_the_concat_critic_scale_their_input(overrides, where):
    critic = build_critic(overrides.get('critic_type', 'separable'), _params(**overrides))
    if where == 'concat':
        assert all(isinstance(s, InputScaling) for s in critic.scaling)
    else:
        assert isinstance(critic.embedding_net_x, InputScaling) and isinstance(critic.embedding_net_y, InputScaling)


def test_deepsets_is_refused():
    with pytest.raises(ValueError, match='deepsets'):
        build_critic('separable', _params(embedding_model='deepsets'))


def test_binned_spikes_enter_as_counts():
    from neural_mi.data.handler import create_dataset
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.uniform(0, 20, 200)) for _ in range(2)]
    dataset = create_dataset(spikes, processor_type_x='spike', processor_params_x={'window_size': 1.0,
                                                                                    'bin_size': 0.01})
    values = dataset.x_data if hasattr(dataset, 'x_data') else dataset.data
    values = values.numpy()
    assert np.allclose(values, np.round(values)) and values.max() < 10
