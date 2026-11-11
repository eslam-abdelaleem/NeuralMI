"""Model(input_scaling=...) removes the units and the offset of an encoder's input by the training rows.

'auto', the default, scales each stream as its kind asks (defaults.SCALING_OF_KIND), one scale per unit:
each channel of a continuous stream, or of rows passed in built already, by its own mean and spread; spike
counts centred neuron by neuron and divided by one spread for the whole stream; spike times divided by one
root mean square for the whole stream, with the padding left in place. A categorical stream is left as it is.
An encoder without biases is not centred, and one that reads a padding value (DeepSets) keeps it. 'none'
scales no input. The statistics are buffers of the network: fitted from the training rows before the first
epoch, saved with the network and applied by ``extract_embeddings``.
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


def _params(**overrides):
    params = {k: v['default'] for k, v in BASE_PARAMS_SCHEMA.items() if 'default' in v}
    params.update(input_dim_x=6, input_dim_y=6, n_channels_x=2, n_channels_y=2, hidden_dim=8, embedding_dim=4,
                  input_scaling='auto', input_scaling_x=('channel',), input_scaling_y='channel')
    params.update(overrides)
    return params


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


def _spikes(f=1.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.uniform(0, n, 2 * n)) * f for _ in range(2)]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_how_each_kind_of_stream_is_scaled():
    from neural_mi.defaults import stream_kind_of, stream_scaling
    assert BASE_PARAMS_SCHEMA['input_scaling']['default'] == 'auto'
    assert stream_kind_of(None) == stream_kind_of('continuous') == 'continuous'
    assert stream_kind_of('spike', {'bin_size': 0.01}) == 'spike_counts'
    assert stream_kind_of('spike', {'window_size': 1.0}) == 'spike_times'
    assert stream_kind_of('categorical') == 'categorical'
    assert stream_scaling('auto', 'continuous') == stream_scaling('auto', None) == 'channel'
    assert stream_scaling('auto', 'spike_counts') == 'counts' and stream_scaling('auto', 'spike_times') == 'times'
    assert stream_scaling('auto', 'categorical') is None and stream_scaling('auto', 'mixed') is None
    assert stream_scaling('none', 'continuous') is None


def test_channel_mode_centres_and_scales_each_channel():
    x = torch.randn(500, 3, 4) * torch.tensor([1.0, 10.0, 1000.0]).view(1, 3, 1) + 5
    scaler = ChannelScaling(3, 'channel')
    scaler.fit(x)
    z = scaler(x)
    assert torch.allclose(z.mean(dim=(0, 2)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(z.std(dim=(0, 2)), torch.ones(3), atol=1e-2)


def test_counts_mode_keeps_the_neurons_relative_scales():
    rates = torch.tensor([0.1, 2.0, 8.0]).view(1, 3, 1)
    x = torch.poisson(rates.expand(2000, 3, 10))
    scaler = ChannelScaling(3, 'counts')
    scaler.fit(x)
    z = scaler(x)
    assert torch.allclose(z.mean(dim=(0, 2)), torch.zeros(3), atol=1e-4)
    assert len(set(scaler.scale.tolist())) == 1                   # one scale for the whole stream
    spread = z.std(dim=(0, 2))
    assert spread[0] < 0.3 < spread[1] < spread[2]                # a quiet neuron stays quiet


def test_times_mode_keeps_the_padding_and_does_not_centre():
    t = torch.rand(1000, 4, 6) * 1000.0
    t[t < 400] = 0.0
    scaler = ChannelScaling(4, 'times', empty=0.0)
    scaler.fit(t)
    z = scaler(t)
    assert torch.equal(z == 0, t == 0) and bool((scaler.shift == 0).all())
    assert len(set(scaler.scale.tolist())) == 1
    real = t != 0
    assert float((z[real] ** 2).mean().sqrt()) == pytest.approx(1.0, rel=1e-4)


def test_data_in_tiny_units_are_scaled_and_a_constant_channel_is_not():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(np.c_[rng.standard_normal(500) * 1e-12, np.full(500, 3e-12)][:, :, None])
    scaler = ChannelScaling(2)
    scaler.fit(x)
    assert scaler.scale[0] == pytest.approx(1e-12, rel=0.1) and scaler.scale[1] == 1.0
    assert scaler.shift[1] == pytest.approx(3e-12)


# ---------------------------------------------------------------------------
# Building the networks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('overrides, where', [
    ({}, 'encoders'), ({'critic_type': 'concat'}, 'concat'), ({'embedding_model': 'gru'}, 'encoders'),
    ({'_difference_term': True, 'n_channels_x': (1, 1), 'input_dim_x': (3, 3),
      'input_scaling_x': ('channel', 'channel')}, 'encoders')])
def test_every_encoder_and_the_concat_critic_scale_their_input(overrides, where):
    critic = build_critic(overrides.get('critic_type', 'separable'), _params(**overrides))
    if where == 'concat':
        assert all(isinstance(s, InputScaling) for s in critic.scaling)
    else:
        assert isinstance(critic.embedding_net_x, InputScaling) and isinstance(critic.embedding_net_y, InputScaling)


def test_each_stream_decides_for_itself():
    """The parts of a pair X decide apart, and a part made of streams of different kinds is left as it is."""
    from neural_mi.defaults import side_scaling
    params = {'_scaled_streams': {'x': 'channel', 'w': 'times', 'y': 'channel'}}
    assert side_scaling(params, ('x', 'w'), 'y') == (('channel', 'times'), 'channel')
    assert side_scaling(params, (('y', 'w'),), 'y') == ((None,), 'channel')    # a part made of two streams
    critic = build_critic('separable', _params(_difference_term=True, n_channels_x=(1, 1), input_dim_x=(3, 3),
                                               input_scaling_x=('channel', None)))
    scalers = critic.embedding_net_x.scalers
    assert isinstance(scalers[0], ChannelScaling) and isinstance(scalers[1], torch.nn.Identity)


def test_an_encoder_without_biases_is_not_centred():
    critic = build_critic('separable', _params(bias=False))
    assert not critic.embedding_net_x.scalers[0].centre and not critic.embedding_net_y.scalers[0].centre
    assert build_critic('separable', _params()).embedding_net_x.scalers[0].centre


@pytest.mark.parametrize('custom', [False, True])
def test_deepsets_scales_spike_times_and_keeps_its_padding(custom):
    """An encoder that reads a padding value, DeepSets or a class of its own that takes no_spike_value,
    reads spike times: its input is scaled as spike times, whatever the stream recorded."""
    from neural_mi.models.embeddings import DeepSets

    class MyDeepSets(DeepSets):
        pass
    params = _params(**({} if custom else {'embedding_model': 'deepsets'}))
    critic = build_critic('separable', params, MyDeepSets if custom else None)
    for net in (critic.embedding_net_x, critic.embedding_net_y):
        scaler = net.scalers[0]
        assert isinstance(net, InputScaling) and scaler.mode == 'times' and scaler.empty == 0.0
        assert not scaler.centre


def test_each_side_of_a_shared_encoder_has_statistics_of_its_own():
    """The two sides share the encoder's weights, and each side's input is scaled by its own rows, even when
    their windows differ in length."""
    from neural_mi.models.critics import fit_input_scaling
    for variational in (False, True):
        critic = build_critic('separable', _params(shared_encoder=True, embedding_model='cnn',
                                                   use_variational=variational))
        net_x, net_y = critic.embedding_net_x, critic.embedding_net_y
        if variational:
            assert net_y.mu_head is net_x.mu_head
            net_x, net_y = net_x.base_encoder, net_y.base_encoder
        assert net_x.encoder is net_y.encoder and net_x.scalers[0] is not net_y.scalers[0]
        x, y = torch.randn(64, 2, 5), 50 * torch.randn(64, 2, 7)
        fit_input_scaling(critic, x, y)
        assert bool((net_y.scalers[0].scale > 10 * net_x.scalers[0].scale).all())


# ---------------------------------------------------------------------------
# Through run()
# ---------------------------------------------------------------------------

def test_the_default_scales_a_continuous_stream_and_none_scales_nothing(tmp_path):
    x, y = _data()
    state, built = _saved(tmp_path, x, y, 'default.pt')
    assert 'embedding_net_x.scalers.0.shift' in state and 'embedding_net_y.scalers.0.shift' in state
    assert built['input_scaling'] == 'auto' and built['input_scaling_x'] == ('channel',)
    assert built['input_scaling_y'] == 'channel'
    state, built = _saved(tmp_path, x, y, 'none.pt', input_scaling='none')
    assert not any('scalers' in key for key in state) and built['input_scaling_x'] == (None,)


def test_spike_rows_take_one_scale_for_the_stream(tmp_path):
    p = {'window_size': 1.0}
    processing = nmi.Processing(x='spike', x_params=p, y='spike', y_params=p)
    state, built = _saved(tmp_path, _spikes(), _spikes(), 'times.pt', processing=processing)
    assert built['input_scaling_x'] == ('times',)
    assert bool((state['embedding_net_x.scalers.0.shift'] == 0).all())
    assert len(set(state['embedding_net_x.scalers.0.scale'].tolist())) == 1
    counts = {'window_size': 1.0, 'bin_size': 0.1}
    processing = nmi.Processing(x='spike', x_params=counts, y='spike', y_params=counts)
    state, built = _saved(tmp_path, _spikes(), _spikes(), 'counts.pt', processing=processing)
    assert built['input_scaling_x'] == ('counts',)
    assert len(set(state['embedding_net_x.scalers.0.scale'].tolist())) == 1


def test_the_statistics_are_fitted_saved_and_applied_on_rebuild(tmp_path):
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = nmi.run(x * 1e3 + 5, y, model=nmi.Model(hidden_dim=8, embedding_dim=4),
                         training=nmi.Training(n_epochs=2, batch_size=64,
                                               save_best_model_path=str(tmp_path / 'net.pt')),
                         show_progress=False, seed=0, n_workers=1)
    path = result.runs['model_path'].iloc[0]
    state = torch.load(path, weights_only=False)['state_dict']
    shift, scale = state['embedding_net_x.scalers.0.shift'], state['embedding_net_x.scalers.0.scale']
    assert torch.allclose(shift, torch.full((2,), 5.0), atol=150) and bool((scale > 500).all())
    zx, _ = nmi.extract_embeddings(path, x[:8, :, None] * 1e3 + 5, y[:8, :, None])
    assert np.isfinite(zx).all() and np.abs(zx).max() < 1e3


def test_an_estimate_does_not_depend_on_the_units():
    """Tiny units and large offsets as well: the variance floor is relative, and the statistics are fitted in
    double precision."""
    x, y = _data()
    unit = _estimate(x, y).mi_estimate
    for f, offset in ((1e3, 0.0), (1e-12, 0.0), (1.0, 300.0)):
        assert _estimate(x * f + offset, y * f + offset).mi_estimate == pytest.approx(unit, abs=1e-3), (f, offset)


def test_spike_times_read_the_same_in_any_unit():
    """Spike times in seconds and in milliseconds give the same rows once each stream is divided by its root
    mean square, so the estimates agree."""
    readings = []
    for f in (1.0, 1000.0):
        p = {'window_size': 1.0 * f}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            readings.append(nmi.run(_spikes(f), _spikes(f), processing=nmi.Processing(x='spike', x_params=p),
                                    model=nmi.Model(hidden_dim=8, embedding_dim=4),
                                    training=nmi.Training(n_epochs=3, batch_size=64, shift_windows=False),
                                    show_progress=False, seed=0, n_workers=1).mi_estimate)
    assert readings[1] == pytest.approx(readings[0], abs=1e-3)


def test_binned_spikes_enter_as_counts():
    from neural_mi.data.handler import create_dataset
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.uniform(0, 20, 200)) for _ in range(2)]
    dataset = create_dataset(spikes, processor_type_x='spike', processor_params_x={'window_size': 1.0,
                                                                                    'bin_size': 0.01})
    values = dataset.x_data if hasattr(dataset, 'x_data') else dataset.data
    values = values.numpy()
    assert np.allclose(values, np.round(values)) and values.max() < 10


def test_a_named_quantity_agrees_with_run_about_scaling():
    """Under 'none' its data check warns about rows far from unit scale, and by default it leaves the scale to
    the encoders, whose estimate then does not depend on the units, as run() does."""
    x, y = _data()
    for scaling, warns in (('none', True), (None, False)):
        model = nmi.Model(hidden_dim=16, embedding_dim=4, input_scaling=scaling)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            large = nmi.instantaneous_mi(x * 1e3, y * 1e3, model=model, show_progress=False, seed=0,
                                         training=nmi.Training(n_epochs=4, batch_size=64))
        assert any('typical scale' in str(w.message) for w in caught) == warns, scaling
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        unit = nmi.instantaneous_mi(x, y, model=nmi.Model(hidden_dim=16, embedding_dim=4),
                                    training=nmi.Training(n_epochs=4, batch_size=64), show_progress=False, seed=0)
    assert large.mi_estimate == pytest.approx(unit.mi_estimate, abs=1e-3)
