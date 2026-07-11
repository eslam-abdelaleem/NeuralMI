"""Tests for the Trainer class in neural_mi."""

import pytest
import torch
import numpy as np
import torch.nn as nn

from neural_mi.training.trainer import Trainer
from neural_mi.data import PairedDataset
from neural_mi.models.critics import SeparableCritic
from neural_mi.models.embeddings import MLP

# --- Fixtures ---

@pytest.fixture
def dummy_data():
    """Provides a tiny paired dataset."""
    x = torch.randn(100, 5)
    y = torch.randn(100, 5)
    return PairedDataset(x, y)

@pytest.fixture
def dummy_model():
    """Provides a simple separable critic."""
    net_x = MLP(input_dim=5, hidden_dim=8, embed_dim=4, n_layers=1)
    return SeparableCritic(embedding_net_x=net_x)

def dummy_estimator(scores, **kwargs):
    """A mock estimator that just returns the mean of the scores."""
    return scores.mean()

# --- Tests ---

def test_trainer_basic_execution(dummy_data, dummy_model):
    """Tests that the trainer runs end-to-end and returns expected keys."""
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    trainer = Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'))
    
    results = trainer.train(
        dummy_data, 
        n_epochs=2, 
        batch_size=20, 
        split_mode='random', 
        train_fraction=0.8,
        verbose=False
    )
    
    assert 'train_mi' in results
    assert 'test_mi' in results
    assert 'test_mi_history' in results
    assert not np.isnan(results['test_mi'])
    assert len(results['test_mi_history']) == 2

def test_trainer_safe_eval_chunking(dummy_data, dummy_model):
    """Forces the trainer to chunk the evaluation to prove it prevents OOM."""
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    trainer = Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'))
    
    results = trainer.train(
        dummy_data, 
        n_epochs=1, 
        batch_size=20, 
        split_mode='random',
        max_eval_samples=5, # Forces dataset into mini-batches during eval
        verbose=False
    )
    
    assert not np.isnan(results['test_mi'])

def test_trainer_spectral_metrics(dummy_data, dummy_model):
    """Tests that dimensionality metrics are extracted when requested."""
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    trainer = Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'))
    
    results = trainer.train(
        dummy_data, 
        n_epochs=1, 
        batch_size=20, 
        track_spectral_metrics=True,
        spectral_output='all',
        verbose=False
    )
    
    # Check that spectral metrics were successfully injected into results
    assert 'participation_ratio' in results or 'pr_singular' in results
    assert 'effective_rank' in results

def test_trainer_custom_smoothing(dummy_data, dummy_model):
    """Tests the custom smoothing hook for early stopping."""
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)

    # A custom smoothing function that does absolutely nothing (identity)
    identity_smooth = lambda arr: np.array(arr)

    trainer = Trainer(
        dummy_model,
        dummy_estimator,
        optimizer,
        torch.device('cpu'),
        custom_smoothing_fn=identity_smooth
    )

    results = trainer.train(dummy_data, n_epochs=2, batch_size=20, verbose=False)
    assert 'test_mi' in results


class TestDecoderLoss:
    """Unit tests for Trainer._decoder_loss."""

    def test_linear_activation_returns_mse(self):
        """'linear' activation uses MSE loss."""
        B, C, W = 4, 3, 8
        recon = torch.randn(B, C, W)
        target = torch.randn(B, C, W)
        loss = Trainer._decoder_loss(recon, target, 'linear')
        expected = torch.nn.functional.mse_loss(recon, target)
        assert torch.isclose(loss, expected)

    def test_sigmoid_activation_returns_mse(self):
        """'sigmoid' activation uses MSE loss."""
        B, C, W = 4, 3, 8
        recon = torch.sigmoid(torch.randn(B, C, W))
        target = torch.sigmoid(torch.randn(B, C, W))
        loss = Trainer._decoder_loss(recon, target, 'sigmoid')
        expected = torch.nn.functional.mse_loss(recon, target)
        assert torch.isclose(loss, expected)

    def test_softmax_activation_returns_nll(self):
        """'softmax' activation uses NLL loss; scalar, finite, non-negative."""
        B, C, W = 4, 5, 8
        # recon: probability distribution over C channels per time step
        recon = torch.softmax(torch.randn(B, C, W), dim=1)
        # target: one-hot over channels
        class_idx = torch.randint(0, C, (B, W))
        target = torch.zeros(B, C, W)
        target.scatter_(1, class_idx.unsqueeze(1), 1.0)
        loss = Trainer._decoder_loss(recon, target, 'softmax')
        assert loss.ndim == 0           # scalar
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

class TestPeakFraction:
    """Tests for the improvement-checkpoint-based peak_fraction feature."""

    def _dataset(self, n=200):
        x = torch.randn(n, 4)
        y = torch.randn(n, 4)
        return PairedDataset(x, y)

    def _trainer(self):
        net_x = MLP(input_dim=4, hidden_dim=8, embed_dim=4, n_layers=1)
        model = SeparableCritic(embedding_net_x=net_x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return Trainer(model, dummy_estimator, optimizer, torch.device('cpu'))

    def test_conservative_keys_present_when_peak_fraction_lt_1(self):
        """peak_fraction < 1.0 must add conservative_epoch and train_mi_at_peak to results."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.5, verbose=False,
        )
        assert 'conservative_epoch' in results, "conservative_epoch missing"
        assert 'train_mi_at_peak' in results, "train_mi_at_peak missing"

    def test_both_estimates_are_finite_floats(self):
        """train_mi and train_mi_at_peak are both proper fresh evaluations (finite floats)."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.5, verbose=False,
        )
        assert isinstance(results['train_mi'], float)
        assert isinstance(results['train_mi_at_peak'], float)
        assert np.isfinite(results['train_mi'])
        assert np.isfinite(results['train_mi_at_peak'])

    def test_conservative_epoch_le_best_epoch(self):
        """The conservative epoch must not be later than the best epoch."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.5, verbose=False,
        )
        if 'conservative_epoch' in results:
            assert results['conservative_epoch'] <= results['best_epoch']

    def test_no_conservative_keys_when_peak_fraction_is_1(self):
        """peak_fraction=1.0 (default) must NOT add conservative_epoch to results."""
        results = self._trainer().train(
            self._dataset(), n_epochs=3, batch_size=20,
            peak_fraction=1.0, verbose=False,
        )
        assert 'conservative_epoch' not in results

    def test_eval_train_not_required_for_peak_fraction(self):
        """peak_fraction < 1.0 must not need eval_train; no per-epoch train history produced."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.7, eval_train=False, verbose=False,
        )
        # Proper train_mi must be present
        assert isinstance(results['train_mi'], float)
        # Per-epoch train history must NOT be present (eval_train=False)
        assert 'train_mi_history' not in results


    def test_softmax_loss_lower_for_perfect_prediction(self):
        """NLL loss is lower when predictions match targets perfectly."""
        B, C, W = 4, 5, 8
        # Perfect prediction: prob≈1 on the correct class
        class_idx = torch.randint(0, C, (B, W))
        target = torch.zeros(B, C, W)
        target.scatter_(1, class_idx.unsqueeze(1), 1.0)
        # Nearly-perfect recon: high probability on true class
        perfect_logits = target * 10.0
        recon_good = torch.softmax(perfect_logits, dim=1)
        # Random (bad) prediction
        recon_bad = torch.softmax(torch.randn(B, C, W), dim=1)
        loss_good = Trainer._decoder_loss(recon_good, target, 'softmax')
        loss_bad = Trainer._decoder_loss(recon_bad, target, 'softmax')
        assert loss_good.item() < loss_bad.item()