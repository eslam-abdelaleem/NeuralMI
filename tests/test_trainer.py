"""Tests for the Trainer class in neural_mi."""

import warnings

import pytest
import torch
import numpy as np

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

def test_trainer_final_spectral_metrics_always_present(dummy_data, dummy_model):
    """pr_eig/pr_singular/spectrum at the best epoch are unconditional -- present
    even with track_spectral_history left at its default (False)."""
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    trainer = Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'))

    results = trainer.train(
        dummy_data,
        n_epochs=1,
        batch_size=20,
        verbose=False
    )

    assert 'pr_eig' in results
    assert 'pr_singular' in results
    assert 'spectrum' in results
    assert 'spectral_metrics_history' not in results
    # effective_rank/spectral_entropy were dropped from the returned metrics --
    # the raw spectrum is kept instead, from which they're cheaply derivable.
    assert 'effective_rank' not in results

def test_trainer_spectral_history_per_epoch(dummy_data, dummy_model):
    """track_spectral_history=True records pr_eig/pr_singular/spectrum at every
    epoch, and nothing else (no effective_rank/spectral_entropy)."""
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
    trainer = Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'))

    results = trainer.train(
        dummy_data,
        n_epochs=2,
        batch_size=20,
        track_spectral_history=True,
        verbose=False
    )

    history = results['spectral_metrics_history']
    assert len(history) == 2
    for epoch_metrics in history:
        assert set(epoch_metrics.keys()) == {'spectral_whitening', 'pr_eig', 'pr_singular', 'spectrum'}

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


class TestUnderTrainingWarning:
    """Warn when training exhausts all epochs while test MI is still
    climbing (patience defaults high enough that early stopping rarely fires,
    so nothing else signals this -- and since these are lower-bound
    estimators, under-training silently biases the estimate downward)."""

    def _trainer(self, dummy_model, custom_smoothing_fn):
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
        return Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'),
                       custom_smoothing_fn=custom_smoothing_fn)

    def test_fires_when_best_epoch_is_the_last_one(self, dummy_data, dummy_model):
        """Monotonically 'improving' smoothed MI with no early stopping -> warn."""
        increasing = lambda history: np.arange(1, len(history) + 1, dtype=float)
        trainer = self._trainer(dummy_model, increasing)
        with pytest.warns(UserWarning, match="under-trained lower bound"):
            trainer.train(dummy_data, n_epochs=3, batch_size=20, patience=10, verbose=False)

    def test_absent_when_peak_is_not_the_last_epoch(self, dummy_data, dummy_model):
        """Smoothed MI peaks mid-training then declines -> no warning."""
        sequence = [0.1, 0.9, 0.5, 0.3]
        peaked = lambda history: np.array(sequence[:len(history)])
        trainer = self._trainer(dummy_model, peaked)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train(dummy_data, n_epochs=4, batch_size=20, patience=10, verbose=False)
        assert not any("under-trained lower bound" in str(w.message) for w in caught)

    def test_absent_when_early_stopping_engages(self, dummy_data, dummy_model):
        """A plateaued metric with tight patience triggers real early stopping;
        the run never reaches n_epochs, so under-training does not apply."""
        flat = lambda history: np.full(len(history), 0.5)
        trainer = self._trainer(dummy_model, flat)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.train(dummy_data, n_epochs=10, batch_size=20, patience=1, verbose=False)
        assert not any("under-trained lower bound" in str(w.message) for w in caught)


class TestSelectTrainEvalIndices:
    """Regression tests for the SubsetView eval-subset fix: a temporal eval
    subset must be built from contiguous sub-chunks of train_idx's own
    contiguous segments, not a scattered np.random.choice sample -- the
    latter degenerates into thousands of zero-width time ranges that
    collide/drop en masse on the very next shift_time rebuild
    (see CHANGELOG / NEURALMI_REFERENCE.md's shift-mechanisms section)."""

    def _select(self, train_idx, target_size, is_temporal):
        return Trainer.__new__(Trainer)._select_train_eval_indices(train_idx, target_size, is_temporal)

    def test_static_data_uses_plain_random_choice(self):
        np.random.seed(0)
        train_idx = np.arange(100)
        result = self._select(train_idx, 20, is_temporal=False)
        assert len(result) == 20
        assert len(np.unique(result)) == 20
        assert set(result.tolist()) <= set(train_idx.tolist())

    def test_target_size_at_or_above_full_set_returns_everything(self):
        train_idx = np.array([1, 2, 3, 10, 11, 12])
        result = self._select(train_idx, 100, is_temporal=True)
        assert sorted(result.tolist()) == sorted(train_idx.tolist())

    def test_temporal_selection_is_contiguous_per_segment(self):
        """train_idx is two contiguous segments (a real blocked split leaves
        several such segments after carving out test blocks) -- the eval
        subset must be built from contiguous sub-chunks of each, not a
        scatter, so it degenerates into only a handful of SubsetView time
        ranges (matching train_view/test_view's own, already-correct,
        rebuild-safe representation)."""
        seg_a = np.arange(0, 500)      # segment 1: 500 samples
        seg_b = np.arange(600, 700)    # segment 2: 100 samples (gap 500-600 excluded)
        train_idx = np.concatenate([seg_a, seg_b])
        np.random.seed(0)
        result = np.sort(self._select(train_idx, 120, is_temporal=True))

        breaks = np.where(np.diff(result) > 1)[0]
        n_segments = len(breaks) + 1
        # Proportional to segment sizes (500:100 = 5:1), so roughly 100/20 split
        # -- the key property is "a handful of contiguous chunks", not scattered.
        assert n_segments <= 2, f"Expected at most 2 contiguous chunks, found {n_segments} runs"
        assert len(result) == 120
        assert set(result.tolist()) <= set(train_idx.tolist())

    def test_temporal_selection_reduces_subsetview_loss_after_shift(self):
        """End-to-end confirmation against a real PairedTemporalDataset +
        SubsetView + a real blocked split: the contiguous-chunk selection
        must lose dramatically less of the eval subset after a shift than
        the scattered approach it replaces."""
        from neural_mi.data.handler import PairedTemporalDataset, WindowManager
        from neural_mi.data.temporal import SpikeWindowDataset
        from neural_mi.data.views import SubsetView

        rng = np.random.default_rng(0)
        n_neurons, n_seconds, rate = 5, 2000.0, 5.0
        spikes_x = [np.sort(rng.uniform(0, n_seconds, int(n_seconds * rate))) for _ in range(n_neurons)]
        spikes_y = [np.sort(rng.uniform(0, n_seconds, int(n_seconds * rate))) for _ in range(n_neurons)]
        x_ds = SpikeWindowDataset(spikes_x)
        y_ds = SpikeWindowDataset(spikes_y)
        ptd = PairedTemporalDataset(x_ds, y_ds, window_size=2.0, step_size=2.0)
        n = ptd.window_manager.n_windows

        t = Trainer.__new__(Trainer)
        train_idx, _ = t._create_blocked_split(n, 0.9, 5, gap_fraction=0.5)

        target = 200
        np.random.seed(0)
        scattered = np.random.choice(train_idx, target, replace=False)
        chunked = t._select_train_eval_indices(train_idx, target, is_temporal=True)

        view_old = SubsetView(ptd, indices=scattered)
        view_new = SubsetView(ptd, indices=chunked)
        old_before, new_before = len(view_old.indices), len(view_new.indices)
        ptd.time_shift(offset_x=0.5, offset_y=0.5)
        old_after, new_after = len(view_old.indices), len(view_new.indices)

        old_loss = (old_before - old_after) / old_before
        new_loss = (new_before - new_after) / new_before
        assert new_loss < 0.10, f"Contiguous-chunk selection lost {new_loss:.1%} after a shift"
        assert new_loss < old_loss, (
            f"Expected the fix to lose less than the scattered baseline "
            f"(old={old_loss:.1%}, new={new_loss:.1%})"
        )


class TestShiftEvaluationConsistency:
    """shift_time/shift_windows are meant to affect training
    dynamics only. Regression test for a real gap: the reported test_mi/
    train_mi used to be evaluated against whichever shift happened to be
    left over from the last epoch trained, decoupled from which epoch's
    weights were actually being scored. Fixed by freezing both the
    evaluation *content* (a snapshot taken before any shift) and *which
    indices* count as the test/train-eval set (the original arrays, not
    SubsetView's live-updating `.indices`, which drifts for
    shift_time's real PairedTemporalDataset)."""

    def test_shift_windows_final_mi_matches_canonical_reeval(self):
        from neural_mi.data.static import StaticDataset
        from neural_mi.data.handler import PairedDataset as PD
        from neural_mi.data.shift_windowing import PairedWindowShifter
        from neural_mi.estimators import infonce_lower_bound

        np.random.seed(0)
        torch.manual_seed(0)
        T, C, window_size, step_size = 3000, 2, 20, 20
        raw_x, raw_y = torch.randn(T, C), torch.randn(T, C)
        shifter = PairedWindowShifter(raw_x, raw_y, window_size, step_size)
        x0, y0 = shifter.windows_at(0)
        dataset = PD(StaticDataset(x0), StaticDataset(y0))
        dataset._window_shifter = shifter

        n = len(dataset)
        train_idx = np.arange(int(n * 0.8))
        test_idx = np.arange(int(n * 0.8), n)

        net_x = MLP(input_dim=C * window_size, hidden_dim=16, embed_dim=8, n_layers=1)
        net_y = MLP(input_dim=C * window_size, hidden_dim=16, embed_dim=8, n_layers=1)
        critic = SeparableCritic(embedding_net_x=net_x, embedding_net_y=net_y)
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        trainer = Trainer(critic, infonce_lower_bound, optimizer, torch.device('cpu'))

        results = trainer.train(dataset, n_epochs=8, batch_size=32, patience=3,
                                train_indices=train_idx, test_indices=test_idx,
                                shift_windows=True, show_progress=False, verbose=False,
                                output_units='nats')

        x0b, y0b = shifter.windows_at(0)  # re-derive the canonical (shift=0) view independently
        with torch.no_grad():
            manual_mi = trainer._safe_eval_mi(x0b[test_idx], y0b[test_idx], 5000)
        assert abs(results['test_mi'] - manual_mi) < 1e-9, (
            f"reported test_mi ({results['test_mi']}) should exactly match a fresh eval of the "
            f"trained model against the canonical (shift=0) view ({manual_mi})"
        )

    def test_shift_time_final_mi_matches_canonical_reeval(self):
        from neural_mi.data.handler import PairedTemporalDataset, ContinuousWindowDataset
        from neural_mi.estimators import infonce_lower_bound

        np.random.seed(0)
        torch.manual_seed(0)
        T, C = 3000, 2
        raw_x = np.random.randn(T, C).astype('float32')
        raw_y = np.random.randn(T, C).astype('float32')
        x_ds = ContinuousWindowDataset(raw_x)
        y_ds = ContinuousWindowDataset(raw_y)
        dataset = PairedTemporalDataset(x_ds, y_ds, window_size=20)

        # Prime the shift grid to its final, margin-reserved size before
        # deriving train/test indices -- Trainer.train() does this
        # internally too, but a caller supplying custom train_indices/
        # test_indices (as this test does) must size them against the same
        # post-priming count Trainer will use.
        dataset.time_shift(offset_x=0.0, offset_y=0.0)

        n = len(dataset)
        train_idx = np.arange(int(n * 0.8))
        test_idx = np.arange(int(n * 0.8), n)
        window_size = x_ds.max_samples_per_window

        # Snapshot BEFORE training mutates dataset.x_dataset.data in place.
        x0_canonical = dataset.x_dataset.data.clone()
        y0_canonical = dataset.y_dataset.data.clone()

        net_x = MLP(input_dim=C * window_size, hidden_dim=16, embed_dim=8, n_layers=1)
        net_y = MLP(input_dim=C * window_size, hidden_dim=16, embed_dim=8, n_layers=1)
        critic = SeparableCritic(embedding_net_x=net_x, embedding_net_y=net_y)
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        trainer = Trainer(critic, infonce_lower_bound, optimizer, torch.device('cpu'))

        results = trainer.train(dataset, n_epochs=6, batch_size=32, patience=3,
                                train_indices=train_idx, test_indices=test_idx,
                                shift_time=True, show_progress=False, verbose=False,
                                output_units='nats')

        with torch.no_grad():
            manual_mi = trainer._safe_eval_mi(x0_canonical[test_idx], y0_canonical[test_idx], 5000)
        assert abs(results['test_mi'] - manual_mi) < 1e-6, (
            f"reported test_mi ({results['test_mi']}) should exactly match a fresh eval of the "
            f"trained model against the canonical (shift=0) view ({manual_mi}) -- if this fails, "
            f"either the content snapshot or the frozen test-index handling has regressed"
        )