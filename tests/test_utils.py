# tests/test_utils.py
import pytest
import torch
import numpy as np
from neural_mi.utils import (
    get_device, build_critic, build_optimizer_and_scheduler, _shift_data,
    compute_cross_covariance_spectrum, compute_spectral_metrics,
)
from neural_mi.models.critics import SeparableCritic, ConcatCritic, HybridCritic

# A list of all device types we want to test
DEVICES = ["cuda", "mps", "cpu"]

def is_available(device: str) -> bool:
    """Helper function to check if a torch device is available."""
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device == "cpu":
        return True
    return False

@pytest.mark.parametrize("target_device", DEVICES)
def test_get_device(target_device: str):
    """
    Tests that the get_device utility function correctly identifies and returns
    the specified torch.device, skipping if the hardware is unavailable.
    """
    if not is_available(target_device):
        pytest.skip(f"Device '{target_device}' not available on this system.")

    device = get_device(device_str=target_device)
    assert isinstance(device, torch.device)
    assert device.type == target_device

def test_get_device_auto_selection():
    """
    Tests the auto-selection logic of get_device when no device is specified.
    """
    device = get_device()
    assert isinstance(device, torch.device)

# A dummy set of parameters for building models
# Must include all defaults enforced by strict validation
DUMMY_EMBEDDING_PARAMS = {
    'input_dim_x': 10, 'input_dim_y': 10, 'embedding_dim': 4,
    'hidden_dim': 16, 'n_layers': 1, 'n_channels_x': 2, 'n_channels_y': 2,
    'window_size': 5,
    'use_variational': False, 'embedding_model': 'mlp', 'max_n_batches': 512,
    'kernel_size': 3, 'bidirectional': False, 'nhead': 4
}

def test_build_critic_concat():
    critic = build_critic('concat', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, ConcatCritic)


def test_build_critic_separable():
    critic = build_critic('separable', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, SeparableCritic)

def test_build_critic_hybrid():
    critic = build_critic('hybrid', DUMMY_EMBEDDING_PARAMS)
    assert isinstance(critic, HybridCritic)


def test_build_critic_custom_embedding_minimal_signature():
    """A custom embedding class following the minimal BaseEmbedding contract
    (input_dim, hidden_dim, embed_dim, n_layers) must build without receiving
    MLP-specific kwargs (use_spectral_norm/dropout/norm_layer)."""
    import torch.nn as nn
    from neural_mi.models.embeddings import BaseEmbedding

    class MinimalCustom(BaseEmbedding):
        def __init__(self, input_dim, hidden_dim, embed_dim, n_layers):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, embed_dim))

        def forward(self, x):
            return self.net(x.view(x.shape[0], -1))

    critic = build_critic('separable', DUMMY_EMBEDDING_PARAMS,
                          custom_embedding_cls=MinimalCustom)
    assert isinstance(critic, SeparableCritic)
    assert isinstance(critic.embedding_net_x, MinimalCustom)


def test_build_critic_custom_embedding_input_style_channels():
    """A custom class declaring input_style='channels' on itself must receive
    the raw channel count as input_dim, the same as any built-in sequence
    model -- the mechanism build_critic uses to decide this no longer reads
    an unrelated embedding_model= string for a custom class."""
    import torch.nn as nn
    from neural_mi.models.embeddings import BaseEmbedding

    class ChannelsStyleCustom(BaseEmbedding):
        input_style = 'channels'

        def __init__(self, input_dim, hidden_dim, embed_dim, n_layers):
            super().__init__()
            self.input_dim = input_dim
            self.net = nn.Linear(input_dim, embed_dim)

        def forward(self, x):
            return self.net(x.mean(dim=-1))

    critic = build_critic('separable', DUMMY_EMBEDDING_PARAMS,
                          custom_embedding_cls=ChannelsStyleCustom)
    # DUMMY_EMBEDDING_PARAMS: n_channels_x=2 (raw channels), input_dim_x=10
    # (flattened n_channels*window_size) -- input_style='channels' must
    # select the former, not the latter.
    assert critic.embedding_net_x.input_dim == 2


def test_build_critic_lru():
    params = {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'lru'}
    critic = build_critic('separable', params)
    assert isinstance(critic, SeparableCritic)
    from neural_mi.models.embeddings import LRUEmbedding
    assert isinstance(critic.embedding_net_x, LRUEmbedding)
    x = torch.randn(3, 2, 5)  # (batch, n_channels_x, window_size)
    out = critic.embedding_net_x(x)
    assert out.shape == (3, DUMMY_EMBEDDING_PARAMS['embedding_dim'])


def test_build_critic_dual_branch_default_branch_model():
    """embedding_model='dual_branch' with no branch_model defaults to GRU
    branches, no custom_embedding_cls needed."""
    from neural_mi.models.embeddings import DualBranchEmbedding, GRU
    params = {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'dual_branch',
              'n_channels_x': (3, 2)}
    critic = build_critic('separable', params)
    assert isinstance(critic.embedding_net_x, DualBranchEmbedding)
    assert isinstance(critic.embedding_net_x.branch_a, GRU)
    assert isinstance(critic.embedding_net_x.branch_c, GRU)
    a_batch, c_batch = torch.randn(3, 3, 7), torch.randn(3, 2, 4)
    out = critic.embedding_net_x((a_batch, c_batch))
    assert out.shape == (3, DUMMY_EMBEDDING_PARAMS['embedding_dim'])


def test_build_critic_dual_branch_custom_branch_model():
    """branch_model picks which built-in class each branch uses."""
    from neural_mi.models.embeddings import LSTM
    params = {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'dual_branch',
              'branch_model': 'lstm', 'n_channels_x': (3, 2)}
    critic = build_critic('separable', params)
    assert isinstance(critic.embedding_net_x.branch_a, LSTM)
    assert isinstance(critic.embedding_net_x.branch_c, LSTM)


def test_build_critic_dual_branch_unknown_branch_model_raises():
    params = {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'dual_branch',
              'branch_model': 'not_a_real_model', 'n_channels_x': (3, 2)}
    with pytest.raises(ValueError, match="branch_model"):
        build_critic('separable', params)


def test_build_critic_dual_branch_custom_embedding_cls_still_works():
    """Regression: the pre-existing custom_embedding_cls=DualBranchEmbedding
    form (with embedding_model set purely as a shape hint) must keep working
    unchanged -- custom_embedding_cls always takes priority over model_type."""
    from neural_mi.models.embeddings import DualBranchEmbedding, GRU
    params = {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'gru',
              'n_channels_x': (3, 2)}
    critic = build_critic('separable', params, custom_embedding_cls=DualBranchEmbedding)
    assert isinstance(critic.embedding_net_x, DualBranchEmbedding)
    assert isinstance(critic.embedding_net_x.branch_a, GRU)


def test_build_critic_unknown_embedding_model_lists_options():
    with pytest.raises(ValueError, match="lru"):
        build_critic('separable', {**DUMMY_EMBEDDING_PARAMS, 'embedding_model': 'not_a_real_model'})


# --- build_optimizer_and_scheduler (shared by task.py and precision.py) ---

class TestBuildOptimizerAndScheduler:
    def _critic(self):
        return build_critic('separable', DUMMY_EMBEDDING_PARAMS)

    def test_default_adam_optimizer(self):
        critic = self._critic()
        optimizer, scheduler = build_optimizer_and_scheduler({'learning_rate': 1e-3}, critic)
        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None

    def test_named_optimizer_string(self):
        critic = self._critic()
        optimizer, _ = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3, 'optimizer': 'sgd'}, critic)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_raises_with_helpful_message(self):
        critic = self._critic()
        with pytest.raises(ValueError, match="torch.optim.Optimizer subclass"):
            build_optimizer_and_scheduler({'learning_rate': 1e-3, 'optimizer': 'nonexistent'}, critic)

    def test_missing_learning_rate_raises_keyerror(self):
        """Strict access: must not silently substitute a different default
        than BASE_PARAMS_SCHEMA's (the drift this extraction reconciled)."""
        critic = self._critic()
        with pytest.raises(KeyError):
            build_optimizer_and_scheduler({}, critic)

    def test_cosine_scheduler(self):
        critic = self._critic()
        _, scheduler = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3, 'n_epochs': 20, 'scheduler': 'cosine'}, critic)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_unknown_scheduler_raises_with_helpful_message(self):
        critic = self._critic()
        with pytest.raises(ValueError, match="torch.optim.lr_scheduler class"):
            build_optimizer_and_scheduler(
                {'learning_rate': 1e-3, 'n_epochs': 20, 'scheduler': 'nonexistent'}, critic)

    def test_decoder_params_included_in_optimizer(self):
        """Decoder parameters (task.py's use case) must be part of the
        optimized parameter set when provided."""
        import torch.nn as nn
        critic = self._critic()
        decoder = nn.Linear(4, 4)
        optimizer, _ = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3}, critic, decoder_x=decoder)
        optimized_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert all(id(p) in optimized_ids for p in decoder.parameters())

    def test_no_decoders_by_default(self):
        """precision.py's use case: omitting decoder_x/decoder_y must work
        (they default to None) and not error."""
        critic = self._critic()
        optimizer, _ = build_optimizer_and_scheduler({'learning_rate': 1e-3}, critic)
        n_critic_params = sum(1 for _ in critic.parameters())
        n_optimized = sum(len(g['params']) for g in optimizer.param_groups)
        assert n_optimized == n_critic_params

    def test_lr_head_multiplier_splits_param_groups(self):
        """hybrid critic + lr_head_multiplier must produce two param groups
        with different learning rates."""
        params = {**DUMMY_EMBEDDING_PARAMS}
        critic = build_critic('hybrid', params)
        optimizer, _ = build_optimizer_and_scheduler(
            {'learning_rate': 1e-3, 'lr_head_multiplier': 5.0}, critic)
        assert len(optimizer.param_groups) == 2
        lrs = sorted(g['lr'] for g in optimizer.param_groups)
        assert lrs[0] == pytest.approx(1e-3)
        assert lrs[1] == pytest.approx(5e-3)


# --- Spectral Metric Tests ---
def test_cross_covariance_spectrum_shape_and_values():
    """Test that SVD extracts correct singular values from embeddings."""
    # Create two identical embeddings (perfect correlation)
    zx = torch.randn(100, 10)
    zy = zx.clone()
    
    spectrum = compute_cross_covariance_spectrum(zx, zy)
    
    # 1. Output should be a numpy array
    assert isinstance(spectrum, np.ndarray)
    
    # 2. Maximum possible singular values is the bottleneck dimension (10)
    assert len(spectrum) == 10
    
    # 3. Singular values should be non-negative and sorted descendingly
    assert np.all(spectrum >= -1e-7)  # allow tiny numerical noise
    assert np.all(np.diff(spectrum) <= 1e-7)

def test_spectral_metrics_single_dimension():
    """Test metrics when only 1 dimension is utilized (perfectly concentrated)."""
    # A spectrum where all energy is in the first dimension
    spectrum = np.array([10.0, 0.0, 0.0, 0.0])
    metrics = compute_spectral_metrics(spectrum)
    
    assert np.isclose(metrics['pr_singular'], 1.0)
    assert np.isclose(metrics['pr_eig'],1.0)
    assert np.isclose(metrics['effective_rank'], 1.0)

def test_spectral_metrics_uniform_dimensions():
    """Test metrics when all dimensions are utilized equally."""
    # A spectrum where energy is perfectly distributed across 4 dimensions
    spectrum = np.array([2.0, 2.0, 2.0, 2.0])
    metrics = compute_spectral_metrics(spectrum)

    assert np.isclose(metrics['pr_singular'], 4.0)
    assert np.isclose(metrics['pr_eig'],4.0)
    assert np.isclose(metrics['effective_rank'], 4.0)


# --- _shift_data: mixed-modality lag (spike paired with non-spike) ---

class TestShiftDataMixedModality:
    """Regression tests for _shift_data's sign convention: for lag > 0, Y is
    compared against its own future relative to X, in every
    modality-pairing branch."""

    def test_both_continuous_unaffected(self):
        """Existing continuous-continuous behavior must be unchanged."""
        x = np.arange(10.0).reshape(10, 1)
        y = np.arange(100.0, 110.0).reshape(10, 1)
        x_sh, y_sh = _shift_data(x, y, 3, 'continuous', 'continuous')
        assert x_sh.shape[0] == 7 and y_sh.shape[0] == 7
        # x's early indices paired with y's later indices (y "in the future")
        np.testing.assert_array_equal(x_sh[:, 0], np.arange(0.0, 7.0))
        np.testing.assert_array_equal(y_sh[:, 0], np.arange(103.0, 110.0))

    def test_both_spike_unaffected(self):
        """Existing spike-spike behavior (shift Y only) must be unchanged."""
        x_spikes = [np.array([1.0, 5.0])]
        y_spikes = [np.array([2.0, 6.0])]
        x_sh, y_sh = _shift_data(x_spikes, y_spikes, 1.5, 'spike', 'spike')
        assert x_sh is x_spikes  # untouched
        np.testing.assert_allclose(y_sh[0], [0.5, 4.5])

    def test_continuous_x_spike_y_shifts_y_only(self):
        """X=continuous, Y=spike: dispatch is already correct today (keyed
        off Y); must keep working identically after accepting x_processor_type."""
        x = np.arange(10.0).reshape(10, 1)
        y_spikes = [np.array([2.0, 6.0])]
        x_sh, y_sh = _shift_data(x, y_spikes, 1.5, 'continuous', 'spike')
        np.testing.assert_array_equal(x_sh, x)
        np.testing.assert_allclose(y_sh[0], [0.5, 4.5])

    def test_spike_x_continuous_y_does_not_crash(self):
        """X=spike, Y=continuous: previously crashed (dispatch was keyed off
        Y's type alone, so X's spike-time list hit the continuous branch's
        np.array/2-D-slice path)."""
        x_spikes = [np.array([1.0, 5.0])]
        y = np.arange(10.0).reshape(10, 1)
        x_sh, y_sh = _shift_data(x_spikes, y, 2.0, 'spike', 'continuous')
        np.testing.assert_array_equal(y_sh, y)  # y untouched
        np.testing.assert_allclose(x_sh[0], [3.0, 7.0])  # x shifted by +lag

    def test_spike_x_continuous_y_sign_convention_matches_true_delay(self):
        """End-to-end (through create_dataset) sign-convention check: when Y's
        real activity trails X's spikes by `true_delay` seconds, only
        lag=+true_delay must align them into the same window (matching the
        lag>0='Y in the future' convention already used for continuous-
        continuous and spike-spike)."""
        from neural_mi.data.handler import create_dataset

        true_delay = 4.0
        x_spike_times = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        x_spikes = [x_spike_times]
        t = np.arange(0, 100, 1.0)
        y_continuous = np.zeros((len(t), 1))
        for st in x_spike_times:
            idx = int(round(st + true_delay))
            if idx < len(t):
                y_continuous[idx, 0] = 1.0

        def n_co_occurring(lag):
            x_sh, y_sh = _shift_data(x_spikes, y_continuous, lag, 'spike', 'continuous')
            ds = create_dataset(
                x_sh, y_sh, x_time=None, y_time=t.copy(),
                processor_type_x='spike',
                processor_params_x={'window_size': 2.0, 'max_spikes_per_window': 5},
                processor_type_y='continuous',
                processor_params_y={'window_size': 2.0},
            )
            x_has_spike = (ds.x_data.numpy() != -1.0).any(axis=(1, 2))
            y_has_pulse = (ds.y_data.numpy() > 0.5).any(axis=(1, 2))
            return int((x_has_spike & y_has_pulse).sum())

        # 4 of the 5 X-spike windows have a valid paired Y window (the spike
        # at t=90 shifts its window past the end of the 100-sample series);
        # the key check is that only the correctly-signed lag co-occurs at all.
        assert n_co_occurring(true_delay) == 4
        assert n_co_occurring(0.0) == 0
        assert n_co_occurring(-true_delay) == 0

    def test_mixed_modality_logs_informational_not_warning(self, caplog, recwarn):
        """Per maintainer decision: mixed-modality shifting is surfaced as
        plain info, not a UserWarning."""
        import logging
        x_spikes = [np.array([1.0, 5.0])]
        y = np.arange(10.0).reshape(10, 1)
        with caplog.at_level(logging.INFO, logger='neural_mi'):
            _shift_data(x_spikes, y, 2.0, 'spike', 'continuous')
        assert any('Mixed-modality lag' in r.message for r in caplog.records)
        assert len(recwarn) == 0

# --- bias / zero-preservation -------------------------------------------

def _n_bias(module, prefix=''):
    return sum(1 for k, _ in module.named_parameters()
               if 'bias' in k and k.startswith(prefix))


class TestBias:
    """`bias` controls whether embedding layers carry bias terms.

    With none, an all-zero input embeds to exactly zero, so a spike window
    consisting entirely of padding contributes nothing.
    """

    @pytest.mark.parametrize('proc_type', ['continuous', 'spike'])
    def test_default_is_biased_for_every_data_type(self, proc_type):
        params = dict(DUMMY_EMBEDDING_PARAMS, bias=None,
                      processor_type_x=proc_type, processor_type_y=proc_type)
        critic = build_critic('separable', params)
        assert _n_bias(critic) > 0

    @pytest.mark.parametrize('explicit', [True, False])
    def test_explicit_value_overrides_the_data_type(self, explicit):
        params = dict(DUMMY_EMBEDDING_PARAMS, bias=explicit,
                      processor_type_x='spike', processor_type_y='spike')
        critic = build_critic('separable', params)
        assert (_n_bias(critic) > 0) is explicit

    @pytest.mark.parametrize('model,kw', [
        ('mlp', {}), ('cnn', {}), ('gru', {}), ('lstm', {}), ('tcn', {}), ('lru', {}),
    ])
    def test_zero_input_embeds_to_zero(self, model, kw):
        """The property that makes bias=False worth having.

        Every parameter is randomised first, so this cannot pass merely because
        biases initialise to zero.
        """
        params = dict(DUMMY_EMBEDDING_PARAMS, bias=False, embedding_model=model,
                      processor_type_x='spike', processor_type_y='spike', **kw)
        critic = build_critic('separable', params)
        net = critic.embedding_net_x
        torch.manual_seed(0)
        with torch.no_grad():
            for prm in net.parameters():
                prm.normal_(0, 0.5)
        net.eval()
        style = getattr(type(net), 'input_style', 'flattened')
        x = (torch.zeros(1, 2, 5) if style == 'channels'
             else torch.zeros(1, DUMMY_EMBEDDING_PARAMS['input_dim_x']))
        assert net(x).abs().max().item() == 0.0

    def test_custom_class_without_bias_still_builds(self):
        """A custom class on the minimal contract must not be handed `bias`."""
        import torch.nn as nn
        from neural_mi.models.embeddings import BaseEmbedding

        class Minimal(BaseEmbedding):
            def __init__(self, input_dim, hidden_dim, embed_dim, n_layers):
                super().__init__()
                self.net = nn.Linear(input_dim, embed_dim)

            def forward(self, x):
                return self.net(x.view(x.shape[0], -1))

        params = dict(DUMMY_EMBEDDING_PARAMS, bias=False,
                      processor_type_x='spike', processor_type_y='spike')
        with pytest.warns(UserWarning, match='does not accept a `bias` argument'):
            critic = build_critic('separable', params, custom_embedding_cls=Minimal)
        assert isinstance(critic, SeparableCritic)

    def test_architecture_that_cannot_preserve_zero_warns(self):
        """Transformer adds a positional encoding, so it cannot honour bias=False."""
        params = dict(DUMMY_EMBEDDING_PARAMS, bias=False, embedding_model='transformer',
                      processor_type_x='spike', processor_type_y='spike')
        with pytest.warns(UserWarning, match='input-independent additive term'):
            build_critic('separable', params)



class TestDeepSets:
    """`embedding_model='deepsets'` aggregates over spikes, not slots."""

    def _net(self, **overrides):
        params = dict(DUMMY_EMBEDDING_PARAMS, embedding_model='deepsets',
                      n_channels_x=3, n_channels_y=3,
                      processor_type_x='spike', processor_type_y='spike',
                      processor_params_x={'window_size': 1.0},
                      processor_params_y={'window_size': 1.0})
        params.update(overrides)
        return build_critic('separable', params).embedding_net_x

    def test_builds_and_forwards(self):
        net = self._net()
        out = net(torch.rand(4, 3, 16))
        assert out.shape == (4, DUMMY_EMBEDDING_PARAMS['embedding_dim'])

    def test_is_invariant_to_spike_order(self):
        """A window is a set of times, so permuting the slots must not matter."""
        net = self._net()
        x = torch.rand(2, 3, 16)
        permuted = x[:, :, torch.randperm(16)]
        assert torch.allclose(net(x), net(permuted), atol=1e-5)

    def test_padded_slots_are_masked_out(self):
        """Changing how many slots are padding must not move the embedding,
        because padded slots are excluded by the mask rather than by relying on
        the sentinel being zero."""
        net = self._net()
        with torch.no_grad():
            for p in net.parameters():
                p.normal_(0, 0.5)          # biases non-zero, so masking is doing the work
        net.eval()
        a = torch.zeros(1, 3, 16); a[:, :, :4] = torch.tensor([0.2, 0.4, 0.6, 0.8])
        b = a.clone(); b[:, :, 4:] = 0.0    # identical real spikes, all else padding
        assert torch.allclose(net(a), net(b), atol=1e-6)

    def test_sentinel_comes_from_the_processor_params(self):
        net = self._net(processor_params_x={'window_size': 1.0, 'no_spike_value': -1.0})
        assert net.no_spike_value == -1.0

    def test_sentinel_defaults_to_zero(self):
        assert self._net().no_spike_value == 0.0

    def test_a_nonzero_sentinel_is_actually_excluded(self):
        """With no_spike_value=-1.0 the padding is not zero, so only an explicit
        mask can keep it out of the sum."""
        net = self._net(processor_params_x={'window_size': 1.0, 'no_spike_value': -1.0})
        with torch.no_grad():
            for p in net.parameters():
                p.normal_(0, 0.5)
        net.eval()
        a = torch.full((1, 3, 16), -1.0); a[:, :, :3] = torch.tensor([0.2, 0.5, 0.9])
        b = a.clone()                      # same 3 spikes; the rest is sentinel either way
        b[:, :, 3:] = -1.0
        assert torch.allclose(net(a), net(b), atol=1e-6)
