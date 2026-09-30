"""Tests for the models in neural_mi."""

import numpy as np
import torch
import torch.nn as nn
import pytest
import neural_mi as nmi
from neural_mi.models.critics import (
    SeparableCritic,
    ConcatCritic,
    HybridCritic,
)
from neural_mi.models.embeddings import (
    MLP,
    CNN1D,
    VariationalWrapper,
    GRU,
    LSTM,
    TCN,
    Transformer,
    LRUEmbedding,
)

# --- Fixtures ---

@pytest.fixture
def x_data():
    """Fixture for sample MLP input data X."""
    return torch.randn(10, 32)

@pytest.fixture
def y_data():
    """Fixture for sample MLP input data Y."""
    return torch.randn(10, 32)

@pytest.fixture
def x_data_cnn():
    """Fixture for sample CNN input data X."""
    return torch.randn(10, 1, 128)

@pytest.fixture
def y_data_cnn():
    """Fixture for sample CNN input data Y."""
    return torch.randn(10, 1, 128)

@pytest.fixture
def mlp_embedding():
    return MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)

@pytest.fixture
def cnn1d_embedding():
    return CNN1D(input_dim=1, hidden_dim=16, embedding_dim=8, n_layers=2)

# --- Embedding Model Tests ---

def test_mlp_embedding(x_data, mlp_embedding):
    """Test the MLP embedding network."""
    output = mlp_embedding(x_data)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (10, 16)

def test_cnn1d_embedding(x_data_cnn, cnn1d_embedding):
    """Test the CNN1D embedding network."""
    output = cnn1d_embedding(x_data_cnn)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (10, 8)


class TestUnifiedActivationHandling:
    """MLP, CNN1D and CNN2D share one activation resolver: the same set of
        names, and a clean ValueError on an unknown name, with no silent
        substitution.
    """

    SHARED_ACTIVATIONS = ['relu', 'gelu', 'tanh', 'elu', 'leaky_relu', 'sigmoid', 'silu']

    @pytest.mark.parametrize("activation", SHARED_ACTIVATIONS)
    def test_mlp_accepts_all_shared_activations(self, activation, x_data):
        model = MLP(input_dim=32, hidden_dim=8, embedding_dim=4, n_layers=1, activation=activation)
        out = model(x_data)
        assert out.shape == (10, 4)

    @pytest.mark.parametrize("activation", SHARED_ACTIVATIONS)
    def test_cnn1d_accepts_all_shared_activations(self, activation, x_data_cnn):
        model = CNN1D(input_dim=1, hidden_dim=8, embedding_dim=4, n_layers=1, activation=activation)
        out = model(x_data_cnn)
        assert out.shape == (10, 4)

    @pytest.mark.parametrize("activation", SHARED_ACTIVATIONS)
    def test_cnn2d_accepts_all_shared_activations(self, activation):
        from neural_mi.models.embeddings import CNN2D
        model = CNN2D(input_dim=1, hidden_dim=8, embedding_dim=4, n_layers=1, activation=activation)
        out = model(torch.randn(4, 1, 8, 8))
        assert out.shape == (4, 4)

    def test_mlp_unknown_activation_raises_clean_value_error(self):
        with pytest.raises(ValueError, match="Unknown activation.*Supported"):
            MLP(input_dim=32, hidden_dim=8, embedding_dim=4, n_layers=1, activation='not_a_real_activation')

    def test_cnn1d_unknown_activation_raises_clean_value_error(self):
        """An unknown activation raises a ValueError that lists the supported ones."""
        with pytest.raises(ValueError, match="Unknown activation.*Supported"):
            CNN1D(input_dim=1, hidden_dim=8, embedding_dim=4, n_layers=1, activation='not_a_real_activation')

    def test_cnn2d_unknown_activation_raises_clean_value_error(self):
        from neural_mi.models.embeddings import CNN2D
        with pytest.raises(ValueError, match="Unknown activation.*Supported"):
            CNN2D(input_dim=1, hidden_dim=8, embedding_dim=4, n_layers=1, activation='not_a_real_activation')

    def test_cnn1d_gelu_now_reachable(self, x_data_cnn):
        """'gelu' builds a GELU layer in CNN1D's conv blocks. (CNN1D's final
                projection head has its own fixed ReLU whatever `activation` is; that
                is a separate design choice.)
        """
        model = CNN1D(input_dim=1, hidden_dim=8, embedding_dim=4, n_layers=1, activation='gelu')
        has_gelu_in_conv = any(isinstance(m, nn.GELU) for m in model.conv_layers.modules())
        assert has_gelu_in_conv


# --- VariationalWrapper Tests ---

def test_variational_wrapper_embedding(x_data):
    """VariationalWrapper wrapping MLP returns a tuple (z, kl) with correct shapes."""
    base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embedding_dim=16)
    wrapper.train()
    output = wrapper(x_data)
    assert isinstance(output, tuple), "VariationalWrapper should return a (z, kl) tuple"
    embedding, kl_loss = output
    assert embedding.shape == (10, 16), f"Expected (10, 16), got {embedding.shape}"
    assert kl_loss.shape == (), "KL loss should be a scalar"

def test_variational_wrapper_kl_loss(x_data):
    """KL loss is positive at training time and exactly 0 at eval time."""
    base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embedding_dim=16)
    wrapper.train()
    _, kl_loss_train = wrapper(x_data)
    assert kl_loss_train > 0.0, "KL loss should be positive during training"
    wrapper.eval()
    _, kl_loss_eval = wrapper(x_data)
    assert kl_loss_eval == 0.0, "KL loss should be 0.0 during evaluation"

def test_variational_wrapper_eval_returns_mu(x_data):
    """At eval time the wrapper returns the deterministic mean (no sampling noise)."""
    base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embedding_dim=16)
    wrapper.eval()
    with torch.no_grad():
        z1, _ = wrapper(x_data)
        z2, _ = wrapper(x_data)
    # Deterministic at eval time: two calls produce identical results
    assert torch.allclose(z1, z2), "eval-mode forward should be deterministic"

def test_variational_wrapper_gradients_flow(x_data):
    """Gradients must flow through both the base encoder and the mu/log_var heads."""
    base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embedding_dim=16)
    wrapper.train()
    z, kl = wrapper(x_data)
    loss = z.mean() + kl
    loss.backward()
    # Check that mu_head and log_var_head received gradients
    assert wrapper.mu_head.weight.grad is not None
    assert wrapper.log_var_head.weight.grad is not None
    # Check that base encoder received gradients
    for p in wrapper.base_encoder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


# --- VariationalWrapper with all encoder types ---

class TestVariationalWrapperAllEncoders:
    """Ensure VariationalWrapper produces correct (z, kl) for every encoder type."""

    EMBED_DIM = 8
    HIDDEN_DIM = 16

    @pytest.fixture
    def mlp_input(self):
        return torch.randn(10, 32)   # (batch, flat_input)

    @pytest.fixture
    def seq_input(self):
        return torch.randn(10, 4, 20)  # (batch, channels, seq_len)

    def _check_variational_output(self, wrapper, x, embedding_dim):
        wrapper.train()
        z, kl = wrapper(x)
        assert z.shape == (x.shape[0], embedding_dim), \
            f"Expected z shape ({x.shape[0]}, {embedding_dim}), got {z.shape}"
        assert kl.shape == (), "KL loss must be a scalar"
        assert kl > 0.0, "KL loss must be positive during training"
        wrapper.eval()
        z_eval, kl_eval = wrapper(x)
        assert z_eval.shape == (x.shape[0], embedding_dim)
        assert kl_eval == 0.0, "KL must be 0.0 in eval mode"

    def test_mlp_variational(self, mlp_input):
        base = MLP(input_dim=32, hidden_dim=self.HIDDEN_DIM, embedding_dim=self.EMBED_DIM, n_layers=1)
        wrapper = VariationalWrapper(base, embedding_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, mlp_input, self.EMBED_DIM)

    def test_cnn1d_variational(self, seq_input):
        base = CNN1D(input_dim=4, hidden_dim=self.HIDDEN_DIM, embedding_dim=self.EMBED_DIM, n_layers=2, kernel_size=3)
        wrapper = VariationalWrapper(base, embedding_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_gru_variational(self, seq_input):
        base = GRU(input_dim=4, hidden_dim=self.HIDDEN_DIM, embedding_dim=self.EMBED_DIM, n_layers=1)
        wrapper = VariationalWrapper(base, embedding_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_lstm_variational(self, seq_input):
        base = LSTM(input_dim=4, hidden_dim=self.HIDDEN_DIM, embedding_dim=self.EMBED_DIM, n_layers=1)
        wrapper = VariationalWrapper(base, embedding_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_tcn_variational(self, seq_input):
        base = TCN(input_dim=4, hidden_dim=self.HIDDEN_DIM, embedding_dim=self.EMBED_DIM, n_layers=2, kernel_size=3)
        wrapper = VariationalWrapper(base, embedding_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_transformer_variational(self, seq_input):
        base = Transformer(input_dim=4, hidden_dim=self.HIDDEN_DIM, embedding_dim=self.EMBED_DIM, n_layers=2, nhead=4)
        wrapper = VariationalWrapper(base, embedding_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)


# --- Critic Model Tests ---

def test_separable_critic(x_data, y_data, mlp_embedding):
    """Test the SeparableCritic returns a tuple (scores, kl_loss=0)."""
    critic = SeparableCritic(embedding_net_x=mlp_embedding)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_separable_critic_with_variational_wrapper(x_data, y_data):
    """SeparableCritic wrapping MLP with VariationalWrapper returns positive KL."""
    base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
    wrapped = VariationalWrapper(base, embedding_dim=16)
    critic = SeparableCritic(embedding_net_x=wrapped, use_variational=True)
    critic.train()
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss > 0.0

def test_hybrid_critic(x_data, y_data, mlp_embedding):
    """Test the HybridCritic returns a tuple (scores, kl_loss=0)."""
    decision_head = MLP(input_dim=32, hidden_dim=16, embedding_dim=1, n_layers=1) # 16 from X + 16 from Y
    critic = HybridCritic(embedding_net_x=mlp_embedding, decision_head=decision_head)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_concat_critic(x_data, y_data):
    """Test the ConcatCritic returns a tuple (scores, kl_loss=0)."""
    embedding_net = MLP(input_dim=64, hidden_dim=128, embedding_dim=1, n_layers=2) # 32 from X + 32 from Y
    critic = ConcatCritic(embedding_net=embedding_net)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_concat_critic_with_variational_wrapper(x_data, y_data):
    """ConcatCritic wrapping MLP with VariationalWrapper returns positive KL."""
    base = MLP(input_dim=64, hidden_dim=128, embedding_dim=1, n_layers=2)
    wrapped = VariationalWrapper(base, embedding_dim=1)
    critic = ConcatCritic(embedding_net=wrapped, use_variational=True)
    critic.train()
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss > 0.0

def test_build_critic_makes_the_concat_score_variational():
    """use_variational=True reaches the concat critic, which has no encoders of its own."""
    from neural_mi.utils import build_critic
    params = dict(embedding_dim=8, hidden_dim=16, n_layers=1, input_dim_x=4, input_dim_y=4,
                  use_variational=True, embedding_model='mlp', max_n_batches=512)
    critic = build_critic('concat', params)
    assert isinstance(critic.embedding_net, VariationalWrapper)
    critic.train()
    _, kl_loss = critic(torch.randn(6, 4), torch.randn(6, 4))
    assert kl_loss > 0.0


def test_concat_with_variational_runs_through_run():
    import numpy as np
    import neural_mi as nmi
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 2))
    result = nmi.run(x, x + 0.5 * rng.standard_normal((400, 2)), mode='estimate',
                     model=nmi.Model(critic_type='concat', hidden_dim=16, n_layers=1, use_variational=True),
                     training=nmi.Training(n_epochs=2, batch_size=64),
                     split=nmi.Split(mode='random'), show_progress=False, seed=0, device='cpu')
    assert np.isfinite(result.mi_estimate)


# --- Chunking Equivalency Tests ---

@pytest.mark.parametrize("critic_type", ["Separable", "Hybrid"])
def test_critic_chunking_equivalency(critic_type):
    """Proves that chunked processing yields the EXACT same math as full-batch processing."""
    # Fix seed before EVERYTHING so the test is deterministic regardless of test ordering.
    torch.manual_seed(42)
    x_data = torch.randn(10, 32)
    y_data = torch.randn(10, 32)

    base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
    net_x = VariationalWrapper(base, embedding_dim=16)
    net_x.eval()

    kwargs = {'embedding_net_x': net_x, 'use_variational': True}

    if critic_type == "Hybrid":
        decision_head = MLP(input_dim=32, hidden_dim=16, embedding_dim=1, n_layers=1)
        decision_head.eval()
        kwargs['decision_head'] = decision_head
        critic_class = HybridCritic
    else:
        critic_class = SeparableCritic

    # 1. Run without chunking (max_n_batches > batch_size)
    critic_full = critic_class(**kwargs, max_n_batches=100)
    scores_full, kl_full = critic_full(x_data, y_data)

    # 2. Run with aggressive chunking (max_n_batches < batch_size)
    critic_chunked = critic_class(**kwargs, max_n_batches=3)
    scores_chunked, kl_chunked = critic_chunked(x_data, y_data)

    # 3. Assert mathematical equivalence.
    # In eval mode VariationalWrapper returns mu (no sampling), so embeddings are
    # purely deterministic linear ops that are batch-size independent.  The scores
    # (bilinear dot products) should therefore be bit-exact after chunking.
    # We use rtol=1e-4 as a tiny guard against float32 matmul reorder artefacts.
    assert torch.allclose(scores_full, scores_chunked, rtol=1e-4, atol=1e-4), \
        f"{critic_type} chunking altered the score matrix!"
    assert torch.allclose(kl_full, kl_chunked, rtol=1e-4, atol=1e-4), \
        f"{critic_type} chunking altered the variational KL loss!"

# --- Gradient Tests for Critics ---

@pytest.fixture
def critic_and_data(request, x_data, y_data):
    """Fixture to provide different critics and their corresponding data."""
    if request.param == "Separable":
        embedding_net = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
        critic = SeparableCritic(embedding_net_x=embedding_net)
        return critic, x_data, y_data
    elif request.param == "SeparableVariational":
        base = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
        wrapped = VariationalWrapper(base, embedding_dim=16)
        critic = SeparableCritic(embedding_net_x=wrapped, use_variational=True)
        return critic, x_data, y_data
    elif request.param == "Hybrid":
        embedding_net = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=2)
        decision_head = MLP(input_dim=32, hidden_dim=16, embedding_dim=1, n_layers=1)
        critic = HybridCritic(embedding_net_x=embedding_net, decision_head=decision_head)
        return critic, x_data, y_data
    elif request.param == "Concat":
        embedding_net = MLP(input_dim=64, hidden_dim=128, embedding_dim=1, n_layers=2)
        critic = ConcatCritic(embedding_net=embedding_net)
        return critic, x_data, y_data
    return None

@pytest.mark.parametrize(
    "critic_and_data",
    ["Separable", "Hybrid", "Concat", "SeparableVariational"],
    indirect=True
)
def test_critic_gradients(critic_and_data):
    """Test that gradients are computed for all critic parameters."""
    critic, x, y = critic_and_data
    if critic is None:
        pytest.skip("Invalid critic specified.")

    critic.train()

    scores, kl_loss = critic(x, y)
    loss = scores.mean() + kl_loss

    loss.backward()

    for param in critic.parameters():
        assert param.grad is not None
        assert torch.sum(torch.abs(param.grad)) > 0

def test_critic_get_embeddings(x_data, y_data):
    """Test that critics can expose their embeddings for spectral analysis."""
    # Test Separable
    net_x = MLP(input_dim=32, hidden_dim=64, embedding_dim=16, n_layers=1)
    critic_sep = SeparableCritic(embedding_net_x=net_x, max_n_batches=5)
    zx, zy = critic_sep.get_embeddings(x_data, y_data)
    assert zx.shape == (10, 16)
    assert zy.shape == (10, 16)

    # Test Hybrid
    decision_head = MLP(input_dim=32, hidden_dim=16, embedding_dim=1, n_layers=1)
    critic_hybrid = HybridCritic(embedding_net_x=net_x, decision_head=decision_head, max_n_batches=5)
    zx_h, zy_h = critic_hybrid.get_embeddings(x_data, y_data)
    assert zx_h.shape == (10, 16)
    assert zy_h.shape == (10, 16)


# --- Sequential Embedding Model Tests ---


torchvision_available = pytest.mark.skipif(
    not __import__('importlib').util.find_spec('torchvision'),
    reason="torchvision not installed",
)


class TestPretrainedBackboneEmbedding:
    """Tests for PretrainedBackboneEmbedding (frozen torchvision backbone + MLP head)."""

    @torchvision_available
    def test_output_shape_no_pretrained(self):
        from neural_mi.models.embeddings import PretrainedBackboneEmbedding
        model = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embedding_dim=8, n_layers=2,
            pytorch_predefined='resnet18', pretrained=False,
        )
        x = torch.randn(4, 3, 64, 64)
        with pytest.warns(UserWarning, match="spatial size") as caught:
            out = model(x)
        assert out.shape == (4, 8)
        # Raised in forward, which torch calls, and still named at this line.
        assert caught[0].filename == __file__

    @torchvision_available
    def test_backbone_frozen(self):
        from neural_mi.models.embeddings import PretrainedBackboneEmbedding
        model = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embedding_dim=8, n_layers=1,
            pytorch_predefined='resnet18', pretrained=False,
        )
        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone parameters should be frozen"

    @torchvision_available
    def test_head_trainable_grads_flow(self):
        from neural_mi.models.embeddings import PretrainedBackboneEmbedding
        model = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embedding_dim=8, n_layers=1,
            pytorch_predefined='resnet18', pretrained=False,
        )
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        out.sum().backward()
        # MLP head parameters should have gradients
        for p in model.head.parameters():
            assert p.grad is not None

    @torchvision_available
    def test_no_nan_inf(self):
        from neural_mi.models.embeddings import PretrainedBackboneEmbedding
        model = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embedding_dim=8, n_layers=1,
            pytorch_predefined='resnet18', pretrained=False,
        )
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestSequentialEmbeddings:
    """Tests for RNN, TCN, and Transformer embedding architectures."""

    @pytest.fixture
    def seq_input(self):
        # (batch, channels, seq_len)
        return torch.randn(32, 5, 100)

    def test_gru(self, seq_input):
        model = GRU(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=1)
        out = model(seq_input)
        assert out.shape == (32, 8)

        model_bi = GRU(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=1, bidirectional=True)
        out_bi = model_bi(seq_input)
        assert out_bi.shape == (32, 8)

    def test_lstm(self, seq_input):
        model = LSTM(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=1)
        out = model(seq_input)
        assert out.shape == (32, 8)

        model_bi = LSTM(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=1, bidirectional=True)
        out_bi = model_bi(seq_input)
        assert out_bi.shape == (32, 8)

    def test_tcn(self, seq_input):
        model = TCN(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=2, kernel_size=3)
        out = model(seq_input)
        assert out.shape == (32, 8)

    def test_transformer(self, seq_input):
        model = Transformer(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=2, nhead=4)
        out = model(seq_input)
        assert out.shape == (32, 8)

    def test_lru(self, seq_input):
        model = LRUEmbedding(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=2, dropout=0.1)
        out = model(seq_input)
        assert out.shape == (32, 8)

    def test_lru_gradients_flow(self, seq_input):
        model = LRUEmbedding(input_dim=5, hidden_dim=16, embedding_dim=8, n_layers=1)
        out = model(seq_input)
        out.sum().backward()
        assert all(p.grad is not None for p in model.parameters())


class TestCustomEmbeddingUsesTheConfigSpelling:
    """A custom class takes `embedding_dim`, the same name the caller writes
        in `Model(embedding_dim=...)`, so a class declaring that name is
        constructible through `custom_embedding_cls`.
    """

    def test_a_custom_class_builds_with_the_documented_name(self):
        import neural_mi as nmi
        import torch.nn as nn
        from neural_mi.models import BaseEmbedding

        class Custom(BaseEmbedding):
            def __init__(self, input_dim, hidden_dim=64, embedding_dim=32, n_layers=2):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, embedding_dim))

            def forward(self, x):
                return self.net(x.reshape(x.shape[0], -1))

        x, y = nmi.generators.generate_correlated_gaussians(
            n_samples=2000, dim=4, mi=2.0, seed=0)
        result = nmi.run(
            x, y, mode='estimate',
            model=nmi.Model(custom_embedding_cls=Custom, embedding_dim=16, hidden_dim=64),
            training=nmi.Training(n_epochs=30, batch_size=128),
            split=nmi.Split(mode='random'), show_progress=False, seed=0, n_workers=1)
        assert abs(result.mi_estimate - 2.0) < 0.5

    def test_the_built_ins_take_the_same_name(self):
        import inspect
        from neural_mi.models.embeddings import CNN1D, GRU, TCN
        for cls in (CNN1D, GRU, TCN):
            params = inspect.signature(cls.__init__).parameters
            assert 'embedding_dim' in params, cls.__name__
            assert 'embed' + '_dim' not in params, cls.__name__


class TestPerSideEncoders:
    """X and Y can carry different encoders, and Y follows X when unset.

    Spikes on one side and a continuous behavioural variable on the other want
    different architectures, and before this the only route was one class that
    branched internally on the channel count it was handed.
    """

    @staticmethod
    def _pair():
        rng = np.random.default_rng(0)
        latent = rng.standard_normal((500, 2))
        x = (latent @ rng.standard_normal((2, 6))
             + 0.4 * rng.standard_normal((500, 6))).astype(np.float32)
        y = (latent @ rng.standard_normal((2, 3))
             + 0.4 * rng.standard_normal((500, 3))).astype(np.float32)
        return x, y

    @staticmethod
    def _windowed():
        from neural_mi import Processing
        return Processing(x='continuous', y='continuous',
                          x_params={'window_size': 10, 'step_size': 10},
                          y_params={'window_size': 10, 'step_size': 10})

    def _go(self, model):
        from neural_mi import Training
        x, y = self._pair()
        return nmi.run(x_data=x, y_data=y, mode='estimate', model=model,
                       processing=self._windowed(), training=Training(n_epochs=2),
                       n_workers=1, show_progress=False)

    def test_the_two_sides_take_different_architectures(self):
        from neural_mi import Model
        r = self._go(Model(embedding_model='gru', embedding_model_y='mlp',
                           embedding_dim=8, hidden_dim=16, n_layers=1))
        assert np.isfinite(r.mi_estimate)

    def test_each_side_takes_its_own_width_and_depth(self):
        from neural_mi import Model
        r = self._go(Model(embedding_model='gru', embedding_model_y='mlp',
                           embedding_dim=8, hidden_dim=16, hidden_dim_y=32,
                           n_layers=1, n_layers_y=2))
        assert np.isfinite(r.mi_estimate)

    def test_naming_a_builtin_for_y_drops_x_custom_class_on_that_side(self):
        """The two settings are alternatives, so the explicit one wins."""
        from neural_mi.utils import build_critic
        from neural_mi.models.embeddings import BaseEmbedding
        import torch.nn as nn

        class Marker(BaseEmbedding):
            def __init__(self, input_dim, hidden_dim=8, embedding_dim=4, n_layers=1, **kw):
                super().__init__()
                self.net = nn.Linear(input_dim, embedding_dim)
            def forward(self, x):
                return self.net(x.flatten(start_dim=1))

        params = dict(use_variational=False, embedding_model='mlp', hidden_dim=8,
                      n_layers=1, embedding_dim=4, max_n_batches=64,
                      input_dim_x=12, input_dim_y=6, n_channels_x=6, n_channels_y=3,
                      custom_embedding_cls=Marker, embedding_model_y='mlp')
        critic = build_critic('separable', params, Marker)
        assert isinstance(critic.embedding_net_x, Marker)
        assert not isinstance(critic.embedding_net_y, Marker)

    def test_embedding_dim_y_needs_hybrid(self):
        from neural_mi import Model
        with pytest.raises(ValueError, match="critic_type='separable'"):
            self._go(Model(embedding_model='mlp', embedding_dim=8, embedding_dim_y=4,
                           hidden_dim=16, n_layers=1))
        r = self._go(Model(embedding_model='mlp', embedding_dim=8, embedding_dim_y=4,
                           hidden_dim=16, n_layers=1, critic_type='hybrid'))
        assert np.isfinite(r.mi_estimate)

    @pytest.mark.parametrize("kwargs,match", [
        (dict(critic_type='concat'), "no separate embedding"),
        (dict(shared_encoder=True), "cannot also be a second"),
        (dict(embedding_model_y='dual_branch'), "compound"),
    ])
    def test_the_combinations_that_cannot_work_are_refused(self, kwargs, match):
        from neural_mi import Model
        base = dict(embedding_model='mlp', embedding_dim=8, hidden_dim=16, n_layers=1)
        if 'embedding_model_y' not in kwargs:
            base['embedding_model_y'] = 'cnn'
        with pytest.raises(ValueError, match=match):
            self._go(Model(**base, **kwargs))

    def test_a_sequential_encoder_on_a_static_y_is_caught(self):
        from neural_mi import Model, Training
        x, y = self._pair()
        with pytest.raises(ValueError, match="embedding_model_y='gru'"):
            nmi.run(x_data=x, y_data=y, mode='estimate',
                    model=Model(embedding_model='mlp', embedding_model_y='gru',
                                embedding_dim=8, hidden_dim=16, n_layers=1),
                    training=Training(n_epochs=1), n_workers=1, show_progress=False)
