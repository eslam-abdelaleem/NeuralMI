"""Tests for the models in neural_mi."""

import torch
import torch.nn as nn
import pytest
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
    SincEmbedding,
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
    return MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)

@pytest.fixture
def cnn1d_embedding():
    return CNN1D(input_dim=1, hidden_dim=16, embed_dim=8, n_layers=2)

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


# --- VariationalWrapper Tests ---

def test_variational_wrapper_embedding(x_data):
    """VariationalWrapper wrapping MLP returns a tuple (z, kl) with correct shapes."""
    base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embed_dim=16)
    wrapper.train()
    output = wrapper(x_data)
    assert isinstance(output, tuple), "VariationalWrapper should return a (z, kl) tuple"
    embedding, kl_loss = output
    assert embedding.shape == (10, 16), f"Expected (10, 16), got {embedding.shape}"
    assert kl_loss.shape == (), "KL loss should be a scalar"

def test_variational_wrapper_kl_loss(x_data):
    """KL loss is positive at training time and exactly 0 at eval time."""
    base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embed_dim=16)
    wrapper.train()
    _, kl_loss_train = wrapper(x_data)
    assert kl_loss_train > 0.0, "KL loss should be positive during training"
    wrapper.eval()
    _, kl_loss_eval = wrapper(x_data)
    assert kl_loss_eval == 0.0, "KL loss should be 0.0 during evaluation"

def test_variational_wrapper_eval_returns_mu(x_data):
    """At eval time the wrapper returns the deterministic mean (no sampling noise)."""
    base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embed_dim=16)
    wrapper.eval()
    with torch.no_grad():
        z1, _ = wrapper(x_data)
        z2, _ = wrapper(x_data)
    # Deterministic at eval time: two calls produce identical results
    assert torch.allclose(z1, z2), "eval-mode forward should be deterministic"

def test_variational_wrapper_gradients_flow(x_data):
    """Gradients must flow through both the base encoder and the mu/log_var heads."""
    base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    wrapper = VariationalWrapper(base, embed_dim=16)
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

    def _check_variational_output(self, wrapper, x, embed_dim):
        wrapper.train()
        z, kl = wrapper(x)
        assert z.shape == (x.shape[0], embed_dim), \
            f"Expected z shape ({x.shape[0]}, {embed_dim}), got {z.shape}"
        assert kl.shape == (), "KL loss must be a scalar"
        assert kl > 0.0, "KL loss must be positive during training"
        wrapper.eval()
        z_eval, kl_eval = wrapper(x)
        assert z_eval.shape == (x.shape[0], embed_dim)
        assert kl_eval == 0.0, "KL must be 0.0 in eval mode"

    def test_mlp_variational(self, mlp_input):
        base = MLP(input_dim=32, hidden_dim=self.HIDDEN_DIM, embed_dim=self.EMBED_DIM, n_layers=1)
        wrapper = VariationalWrapper(base, embed_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, mlp_input, self.EMBED_DIM)

    def test_cnn1d_variational(self, seq_input):
        base = CNN1D(input_dim=4, hidden_dim=self.HIDDEN_DIM, embed_dim=self.EMBED_DIM, n_layers=2, kernel_size=3)
        wrapper = VariationalWrapper(base, embed_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_gru_variational(self, seq_input):
        base = GRU(input_dim=4, hidden_dim=self.HIDDEN_DIM, embed_dim=self.EMBED_DIM, n_layers=1)
        wrapper = VariationalWrapper(base, embed_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_lstm_variational(self, seq_input):
        base = LSTM(input_dim=4, hidden_dim=self.HIDDEN_DIM, embed_dim=self.EMBED_DIM, n_layers=1)
        wrapper = VariationalWrapper(base, embed_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_tcn_variational(self, seq_input):
        base = TCN(input_dim=4, hidden_dim=self.HIDDEN_DIM, embed_dim=self.EMBED_DIM, n_layers=2, kernel_size=3)
        wrapper = VariationalWrapper(base, embed_dim=self.EMBED_DIM)
        self._check_variational_output(wrapper, seq_input, self.EMBED_DIM)

    def test_transformer_variational(self, seq_input):
        base = Transformer(input_dim=4, hidden_dim=self.HIDDEN_DIM, embed_dim=self.EMBED_DIM, n_layers=2, nhead=4)
        wrapper = VariationalWrapper(base, embed_dim=self.EMBED_DIM)
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
    base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    wrapped = VariationalWrapper(base, embed_dim=16)
    critic = SeparableCritic(embedding_net_x=wrapped, use_variational=True)
    critic.train()
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss > 0.0

def test_hybrid_critic(x_data, y_data, mlp_embedding):
    """Test the HybridCritic returns a tuple (scores, kl_loss=0)."""
    decision_head = MLP(input_dim=32, hidden_dim=16, embed_dim=1, n_layers=1) # 16 from X + 16 from Y
    critic = HybridCritic(embedding_net_x=mlp_embedding, decision_head=decision_head)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_concat_critic(x_data, y_data):
    """Test the ConcatCritic returns a tuple (scores, kl_loss=0)."""
    embedding_net = MLP(input_dim=64, hidden_dim=128, embed_dim=1, n_layers=2) # 32 from X + 32 from Y
    critic = ConcatCritic(embedding_net=embedding_net)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_concat_critic_with_variational_wrapper(x_data, y_data):
    """ConcatCritic wrapping MLP with VariationalWrapper returns positive KL."""
    base = MLP(input_dim=64, hidden_dim=128, embed_dim=1, n_layers=2)
    wrapped = VariationalWrapper(base, embed_dim=1)
    critic = ConcatCritic(embedding_net=wrapped, use_variational=True)
    critic.train()
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss > 0.0

# --- Chunking Equivalency Tests ---

@pytest.mark.parametrize("critic_type", ["Separable", "Hybrid"])
def test_critic_chunking_equivalency(critic_type):
    """Proves that chunked processing yields the EXACT same math as full-batch processing."""
    # Fix seed before EVERYTHING so the test is deterministic regardless of test ordering.
    torch.manual_seed(42)
    x_data = torch.randn(10, 32)
    y_data = torch.randn(10, 32)

    base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    net_x = VariationalWrapper(base, embed_dim=16)
    net_x.eval()

    kwargs = {'embedding_net_x': net_x, 'use_variational': True}

    if critic_type == "Hybrid":
        decision_head = MLP(input_dim=32, hidden_dim=16, embed_dim=1, n_layers=1)
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
        embedding_net = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        critic = SeparableCritic(embedding_net_x=embedding_net)
        return critic, x_data, y_data
    elif request.param == "SeparableVariational":
        base = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        wrapped = VariationalWrapper(base, embed_dim=16)
        critic = SeparableCritic(embedding_net_x=wrapped, use_variational=True)
        return critic, x_data, y_data
    elif request.param == "Hybrid":
        embedding_net = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        decision_head = MLP(input_dim=32, hidden_dim=16, embed_dim=1, n_layers=1)
        critic = HybridCritic(embedding_net_x=embedding_net, decision_head=decision_head)
        return critic, x_data, y_data
    elif request.param == "Concat":
        embedding_net = MLP(input_dim=64, hidden_dim=128, embed_dim=1, n_layers=2)
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
    net_x = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=1)
    critic_sep = SeparableCritic(embedding_net_x=net_x, max_n_batches=5)
    zx, zy = critic_sep.get_embeddings(x_data, y_data)
    assert zx.shape == (10, 16)
    assert zy.shape == (10, 16)

    # Test Hybrid
    decision_head = MLP(input_dim=32, hidden_dim=16, embed_dim=1, n_layers=1)
    critic_hybrid = HybridCritic(embedding_net_x=net_x, decision_head=decision_head, max_n_batches=5)
    zx_h, zy_h = critic_hybrid.get_embeddings(x_data, y_data)
    assert zx_h.shape == (10, 16)
    assert zy_h.shape == (10, 16)


# --- Sequential Embedding Model Tests ---


class TestSincEmbedding:
    """Tests for SincEmbedding (learnable sinc bandpass filters for EEG/LFP)."""

    N_CH = 4
    T = 200
    BATCH = 16
    SR = 500.0  # Hz

    @pytest.fixture
    def lfp_input(self):
        torch.manual_seed(7)
        return torch.randn(self.BATCH, self.N_CH, self.T)

    def test_output_shape_features(self, lfp_input):
        model = SincEmbedding(input_dim=self.N_CH, hidden_dim=16, embed_dim=8,
                              n_layers=2, n_sinc_filters=4, sample_rate=self.SR,
                              feature_fusion='features')
        out = model(lfp_input)
        assert out.shape == (self.BATCH, 8)

    def test_output_shape_concat(self, lfp_input):
        model = SincEmbedding(input_dim=self.N_CH, hidden_dim=16, embed_dim=8,
                              n_layers=2, n_sinc_filters=4, sample_rate=self.SR,
                              feature_fusion='concat')
        out = model(lfp_input)
        assert out.shape == (self.BATCH, 8)

    def test_gradients_flow_through_sinc_params(self, lfp_input):
        model = SincEmbedding(input_dim=self.N_CH, hidden_dim=16, embed_dim=8,
                              n_layers=2, n_sinc_filters=4, sample_rate=self.SR)
        out = model(lfp_input)
        out.sum().backward()
        assert model.log_f_low.grad is not None
        assert model.log_f_high.grad is not None

    def test_no_nan_inf(self, lfp_input):
        model = SincEmbedding(input_dim=self.N_CH, hidden_dim=16, embed_dim=8,
                              n_layers=1, n_sinc_filters=8, sample_rate=self.SR)
        out = model(lfp_input)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_single_channel(self):
        x = torch.randn(8, 1, 100)
        model = SincEmbedding(input_dim=1, hidden_dim=8, embed_dim=4, n_layers=1,
                              n_sinc_filters=4, sample_rate=250.0)
        out = model(x)
        assert out.shape == (8, 4)


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
            input_dim=3, hidden_dim=16, embed_dim=8, n_layers=2,
            pytorch_predefined='resnet18', pretrained=False,
        )
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 8)

    @torchvision_available
    def test_backbone_frozen(self):
        from neural_mi.models.embeddings import PretrainedBackboneEmbedding
        model = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embed_dim=8, n_layers=1,
            pytorch_predefined='resnet18', pretrained=False,
        )
        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone parameters should be frozen"

    @torchvision_available
    def test_head_trainable_grads_flow(self):
        from neural_mi.models.embeddings import PretrainedBackboneEmbedding
        model = PretrainedBackboneEmbedding(
            input_dim=3, hidden_dim=16, embed_dim=8, n_layers=1,
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
            input_dim=3, hidden_dim=16, embed_dim=8, n_layers=1,
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
        model = GRU(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1)
        out = model(seq_input)
        assert out.shape == (32, 8)

        model_bi = GRU(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1, bidirectional=True)
        out_bi = model_bi(seq_input)
        assert out_bi.shape == (32, 8)

    def test_lstm(self, seq_input):
        model = LSTM(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1)
        out = model(seq_input)
        assert out.shape == (32, 8)

        model_bi = LSTM(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1, bidirectional=True)
        out_bi = model_bi(seq_input)
        assert out_bi.shape == (32, 8)

    def test_tcn(self, seq_input):
        model = TCN(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=2, kernel_size=3)
        out = model(seq_input)
        assert out.shape == (32, 8)

    def test_transformer(self, seq_input):
        model = Transformer(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=2, nhead=4)
        out = model(seq_input)
        assert out.shape == (32, 8)
