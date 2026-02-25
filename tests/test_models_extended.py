# tests/test_models_extended.py
import pytest
import torch
from neural_mi.models.embeddings import MLP, CNN1D, GRU, LSTM, TCN, Transformer, VarMLP

class TestEmbeddings:
    @pytest.fixture
    def input_tensor(self):
        # (batch, channels, seq_len) - assuming 3D input for sequential models
        return torch.randn(32, 5, 100)

    def test_cnn1d(self, input_tensor):
        model = CNN1D(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=2)
        out = model(input_tensor)
        assert out.shape == (32, 8)

    def test_gru(self, input_tensor):
        model = GRU(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1)
        out = model(input_tensor)
        assert out.shape == (32, 8)

        model_bi = GRU(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1, bidirectional=True)
        out_bi = model_bi(input_tensor)
        assert out_bi.shape == (32, 8)

    def test_lstm(self, input_tensor):
        model = LSTM(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1)
        out = model(input_tensor)
        assert out.shape == (32, 8)

        model_bi = LSTM(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=1, bidirectional=True)
        out_bi = model_bi(input_tensor)
        assert out_bi.shape == (32, 8)

    def test_tcn(self, input_tensor):
        model = TCN(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=2, kernel_size=3)
        out = model(input_tensor)
        assert out.shape == (32, 8)

    def test_transformer(self, input_tensor):
        model = Transformer(input_dim=5, hidden_dim=16, embed_dim=8, n_layers=2, nhead=4)
        out = model(input_tensor)
        assert out.shape == (32, 8)

    def test_varmlp(self):
        x = torch.randn(32, 10)
        model = VarMLP(input_dim=10, hidden_dim=16, embed_dim=8, n_layers=2)

        # Training mode
        model.train()
        z, kl = model(x)
        assert z.shape == (32, 8)
        assert kl.dim() == 0 # scalar

        # Eval mode
        model.eval()
        z, kl = model(x)
        assert z.shape == (32, 8)
        assert kl == 0.0

    def test_mlp(self):
        x = torch.randn(32, 10)
        model = MLP(input_dim=10, hidden_dim=16, embed_dim=8, n_layers=2)
        out = model(x)
        assert out.shape == (32, 8)
