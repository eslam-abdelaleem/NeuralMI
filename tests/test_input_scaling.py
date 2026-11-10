"""Model(input_scaling=...) scales an encoder's input by the training rows, and binned spikes stay counts.

'none', the default, scales no input. 'auto' scales the input from a continuous
stream or from a stream with no processor, each stream with statistics of its
own, and leaves the rows of a spike-time or categorical processor and the input
of a DeepSets encoder as they are. 'standard' scales every input. The statistics
are buffers of the network: fitted from the training rows before the first
epoch, saved with the network and applied by ``extract_embeddings``.
"""
import warnings

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.defaults import BASE_PARAMS_SCHEMA
from neural_mi.models.embeddings import InputScaling
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


# ---------------------------------------------------------------------------
# The default and 'auto'
# ---------------------------------------------------------------------------

def _saved(tmp_path, x, y, name, processing=None, **model):
    """The state dict and build settings of the network a run saves."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x, y, model=nmi.Model(hidden_dim=8, embedding_dim=4, **model), processing=processing,
                         training=nmi.Training(n_epochs=2, batch_size=64, shift_windows=False,
                                               save_best_model_path=str(tmp_path / name)),
                         show_progress=False, seed=0, n_workers=1)
    saved = torch.load(result.runs['model_path'].iloc[0], weights_only=False)
    return saved['state_dict'], saved['build_params']


def test_which_streams_each_setting_scales():
    from neural_mi.defaults import scales_stream
    assert BASE_PARAMS_SCHEMA['input_scaling']['default'] == 'none'
    assert scales_stream('auto', None) and scales_stream('auto', 'continuous')
    assert not scales_stream('auto', 'spike') and not scales_stream('auto', 'categorical')
    assert scales_stream('standard', 'spike') and not scales_stream('none', 'continuous')
    assert not scales_stream(None, None)            # what a network saved before 'auto' records


def test_the_default_scales_nothing_and_auto_scales_a_continuous_stream(tmp_path):
    x, y = _data()
    state, built = _saved(tmp_path, x, y, 'default.pt')
    assert not any('scalers' in key for key in state) and built['input_scaling'] == 'none'
    state, built = _saved(tmp_path, x, y, 'auto.pt', input_scaling='auto')
    assert 'embedding_net_x.scalers.0.shift' in state and 'embedding_net_y.scalers.0.shift' in state
    assert built['input_scaling'] == 'auto' and built['input_scaling_x'] == (True,) and built['input_scaling_y']


def test_under_auto_an_estimate_does_not_depend_on_the_units():
    """Tiny units and large offsets as well: the variance floor is relative, and the statistics are fitted in
    double precision."""
    x, y = _data()
    model = nmi.Model(hidden_dim=16, embedding_dim=4, input_scaling='auto')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        unit = nmi.run(x, y, model=model, show_progress=False, seed=0,
                       training=nmi.Training(n_epochs=4, batch_size=64)).mi_estimate
        for f, offset in ((1e3, 0.0), (1e-12, 0.0), (1.0, 300.0)):
            moved = nmi.run(x * f + offset, y * f + offset, model=model,
                            show_progress=False, seed=0, training=nmi.Training(n_epochs=4, batch_size=64))
            assert moved.mi_estimate == pytest.approx(unit, abs=1e-3), (f, offset)


def test_spike_rows_are_left_as_they_are_unless_standard(tmp_path):
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.uniform(0, 300, 600)) for _ in range(2)]
    p = {'window_size': 1.0}
    processing = nmi.Processing(x='spike', x_params=p, y='spike', y_params=p)
    state, built = _saved(tmp_path, spikes, spikes, 'spike.pt', processing=processing, input_scaling='auto')
    assert not any('scalers' in key for key in state) and built['input_scaling_x'] == (False,)
    state, _ = _saved(tmp_path, spikes, spikes, 'spike_standard.pt', processing=processing,
                      input_scaling='standard')
    assert 'embedding_net_x.scalers.0.shift' in state


def test_each_stream_decides_for_itself():
    """The parts of a pair X decide apart: a spike W beside a continuous X leaves W's part as it is."""
    from neural_mi.defaults import side_scaling
    from neural_mi.models.embeddings import ChannelScaling
    params = {'_scaled_streams': {'x': True, 'w': False, 'y': True}}
    assert side_scaling(params, ('x', 'w'), 'y') == ((True, False), True)
    assert side_scaling(params, (('y', 'w'),), 'y') == ((False,), True)    # a part made of two streams
    critic = build_critic('separable', _params(input_scaling='auto', _difference_term=True, n_channels_x=(1, 1),
                                               input_dim_x=(3, 3), input_scaling_x=(True, False),
                                               input_scaling_y=True))
    scalers = critic.embedding_net_x.scalers
    assert isinstance(scalers[0], ChannelScaling) and isinstance(scalers[1], torch.nn.Identity)


def test_deepsets_is_left_as_it_is_under_auto():
    from neural_mi.models.embeddings import DeepSets

    class MyDeepSets(DeepSets):
        pass
    for kind in ('deepsets', None):
        params = _params(input_scaling='auto', input_scaling_x=True, input_scaling_y=True,
                         **({'embedding_model': 'deepsets'} if kind else {}))
        critic = build_critic('separable', params, None if kind else MyDeepSets)
        assert not isinstance(critic.embedding_net_x, InputScaling)
        assert not isinstance(critic.embedding_net_y, InputScaling)


def test_a_network_saved_before_auto_rebuilds_unscaled(tmp_path):
    """A network saved before 'auto' existed records input_scaling=None and no per-side decision, and
    extract_embeddings rebuilds it without scalers, so its weights load strictly."""
    old = _params(input_scaling=None)
    critic = build_critic('separable', old)
    path = tmp_path / 'old.pt'
    torch.save({'state_dict': critic.state_dict(), 'build_params': old}, path)
    x = torch.randn(5, 2, 3)
    zx, zy = nmi.extract_embeddings(str(path), x, x)
    critic.eval()
    with torch.no_grad():
        expected, _ = critic.get_embeddings(x, x)
    assert np.allclose(zx, expected.numpy(), atol=1e-5)
    bare = tmp_path / 'bare.pt'
    torch.save(critic.state_dict(), bare)
    zx_bare, _ = nmi.extract_embeddings(str(bare), x, x, base_params={k: v for k, v in old.items()
                                                                     if k != 'input_scaling'})
    assert np.allclose(zx_bare, zx, atol=1e-6)


def test_data_in_tiny_units_are_scaled_and_a_constant_channel_is_not():
    from neural_mi.models.embeddings import ChannelScaling
    rng = np.random.default_rng(0)
    x = torch.as_tensor(np.c_[rng.standard_normal(500) * 1e-12, np.full(500, 3e-12)][:, :, None])
    scaler = ChannelScaling(2)
    scaler.fit(x)
    assert scaler.scale[0] == pytest.approx(1e-12, rel=0.1) and scaler.scale[1] == 1.0
    assert scaler.shift[1] == pytest.approx(3e-12)


def test_each_side_of_a_shared_encoder_has_statistics_of_its_own():
    """The two sides share the encoder's weights, and each side's input is scaled by its own rows, even when
    their windows differ in length."""
    from neural_mi.models.critics import fit_input_scaling
    for variational in (False, True):
        critic = build_critic('separable', _params(input_scaling='auto', shared_encoder=True,
                                                   embedding_model='cnn', use_variational=variational))
        net_x, net_y = critic.embedding_net_x, critic.embedding_net_y
        if variational:
            assert net_y.mu_head is net_x.mu_head
            net_x, net_y = net_x.base_encoder, net_y.base_encoder
        assert net_x.encoder is net_y.encoder and net_x.scalers[0] is not net_y.scalers[0]
        x, y = torch.randn(64, 2, 5), 50 * torch.randn(64, 2, 7)
        fit_input_scaling(critic, x, y)
        assert bool((net_y.scalers[0].scale > 10 * net_x.scalers[0].scale).all())


def test_a_named_quantity_agrees_with_run_about_scaling():
    """By default its data check warns about rows far from unit scale, and under 'auto' it leaves the scale to
    the encoders, whose estimate then does not depend on the units, as run() does."""
    x, y = _data()
    for scaling, warns in ((None, True), ('auto', False)):
        model = nmi.Model(hidden_dim=16, embedding_dim=4, input_scaling=scaling)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            large = nmi.instantaneous_mi(x * 1e3, y * 1e3, model=model, show_progress=False, seed=0,
                                         training=nmi.Training(n_epochs=4, batch_size=64))
        assert any('typical scale' in str(w.message) for w in caught) == warns, scaling
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        unit = nmi.instantaneous_mi(x, y, model=nmi.Model(hidden_dim=16, embedding_dim=4, input_scaling='auto'),
                                    training=nmi.Training(n_epochs=4, batch_size=64), show_progress=False, seed=0)
    assert large.mi_estimate == pytest.approx(unit.mi_estimate, abs=1e-3)
