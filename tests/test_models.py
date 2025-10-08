"""Tests for the models in neural_mi."""

import torch
import pytest
from neural_mi.models.critics import (
    SeparableCritic,
    BilinearCritic,
    ConcatCritic,
    ConcatCriticCNN,
)
from neural_mi.models.embeddings import (
    MLP,
    CNN1D,
    VarMLP,
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

def test_varmlp_embedding(x_data):
    """Test the VarMLP embedding network returns a tuple with correct shapes."""
    var_mlp = VarMLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    var_mlp.train()
    output = var_mlp(x_data)
    assert isinstance(output, tuple)
    embedding, kl_loss = output
    assert embedding.shape == (10, 16)
    assert kl_loss.shape == ()

def test_varmlp_kl_loss(x_data):
    """Test the KL loss calculation in VarMLP."""
    var_mlp = VarMLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    # Test in training mode
    var_mlp.train()
    _, kl_loss_train = var_mlp(x_data)
    assert kl_loss_train > 0.0
    # Test in evaluation mode
    var_mlp.eval()
    _, kl_loss_eval = var_mlp(x_data)
    assert kl_loss_eval == 0.0

# --- Critic Model Tests ---

def test_separable_critic(x_data, y_data, mlp_embedding):
    """Test the SeparableCritic returns a tuple (scores, kl_loss=0)."""
    critic = SeparableCritic(embedding_net_x=mlp_embedding)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_separable_critic_with_varmlp(x_data, y_data):
    """Test SeparableCritic with VarMLP returns a positive KL loss."""
    var_mlp = VarMLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    critic = SeparableCritic(embedding_net_x=var_mlp)
    critic.train()
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss > 0.0

def test_bilinear_critic(x_data, y_data, mlp_embedding):
    """Test the BilinearCritic returns a tuple (scores, kl_loss=0)."""
    embedding_net_y = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
    critic = BilinearCritic(embedding_net_x=mlp_embedding, embedding_net_y=embedding_net_y, embed_dim=16)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_concat_critic(x_data, y_data):
    """Test the ConcatCritic returns a tuple (scores, kl_loss=0)."""
    embedding_net = MLP(input_dim=64, hidden_dim=128, embed_dim=1, n_layers=2)
    critic = ConcatCritic(embedding_net=embedding_net)
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

def test_concat_critic_with_varmlp(x_data, y_data):
    """Test ConcatCritic with VarMLP returns a positive KL loss."""
    embedding_net = VarMLP(input_dim=64, hidden_dim=128, embed_dim=1, n_layers=2)
    critic = ConcatCritic(embedding_net=embedding_net)
    critic.train()
    scores, kl_loss = critic(x_data, y_data)
    assert scores.shape == (10, 10)
    assert kl_loss > 0.0

def test_concat_critic_cnn(x_data_cnn, y_data_cnn, cnn1d_embedding):
    """Test the ConcatCriticCNN returns a tuple (scores, kl_loss=0)."""
    cnn_y = CNN1D(input_dim=1, hidden_dim=16, embed_dim=8, n_layers=2)
    decision_head_input_dim = cnn1d_embedding.embed_dim + cnn_y.embed_dim
    decision_head = MLP(
        input_dim=decision_head_input_dim, hidden_dim=32, embed_dim=1, n_layers=1
    )
    critic = ConcatCriticCNN(cnn_x=cnn1d_embedding, cnn_y=cnn_y, decision_head=decision_head)
    scores, kl_loss = critic(x_data_cnn, y_data_cnn)
    assert scores.shape == (10, 10)
    assert kl_loss == 0.0

# --- Gradient Tests for Critics ---

@pytest.fixture
def critic_and_data(request, x_data, y_data, x_data_cnn, y_data_cnn):
    """Fixture to provide different critics and their corresponding data."""
    if request.param == "Separable":
        embedding_net = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        critic = SeparableCritic(embedding_net_x=embedding_net)
        return critic, x_data, y_data
    if request.param == "SeparableVarMLP":
        embedding_net = VarMLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        critic = SeparableCritic(embedding_net_x=embedding_net)
        return critic, x_data, y_data
    elif request.param == "Bilinear":
        embedding_net_x = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        embedding_net_y = MLP(input_dim=32, hidden_dim=64, embed_dim=16, n_layers=2)
        critic = BilinearCritic(embedding_net_x=embedding_net_x, embedding_net_y=embedding_net_y, embed_dim=16)
        return critic, x_data, y_data
    elif request.param == "Concat":
        embedding_net = MLP(input_dim=64, hidden_dim=128, embed_dim=1, n_layers=2)
        critic = ConcatCritic(embedding_net=embedding_net)
        return critic, x_data, y_data
    elif request.param == "ConcatCNN":
        cnn_x = CNN1D(input_dim=1, hidden_dim=16, embed_dim=8, n_layers=2)
        cnn_y = CNN1D(input_dim=1, hidden_dim=16, embed_dim=8, n_layers=2)
        decision_head_input_dim = cnn_x.embed_dim + cnn_y.embed_dim
        decision_head = MLP(input_dim=decision_head_input_dim, hidden_dim=32, embed_dim=1, n_layers=1)
        critic = ConcatCriticCNN(cnn_x=cnn_x, cnn_y=cnn_y, decision_head=decision_head)
        return critic, x_data_cnn, y_data_cnn
    return None

@pytest.mark.parametrize(
    "critic_and_data",
    ["Separable", "Bilinear", "Concat", "SeparableVarMLP", "ConcatCNN"],
    indirect=True
)
def test_critic_gradients(critic_and_data):
    """Test that gradients are computed for all critic parameters."""
    critic, x, y = critic_and_data
    critic.train()

    scores, kl_loss = critic(x, y)
    loss = scores.mean() + kl_loss

    loss.backward()

    for param in critic.parameters():
        assert param.grad is not None
        assert torch.sum(torch.abs(param.grad)) > 0