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
    net_x = MLP(input_dim=5, hidden_dim=8, embedding_dim=4, n_layers=1)
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
        assert set(epoch_metrics.keys()) == {'whitening', 'pr_eig', 'pr_singular', 'spectrum'}

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

class TestEvalTrain:
    """Tests for eval_train's accepted values and the sizes they select."""

    def _dataset(self, n=200):
        x = torch.randn(n, 4)
        y = torch.randn(n, 4)
        return PairedDataset(x, y)

    def _trainer(self):
        net_x = MLP(input_dim=4, hidden_dim=8, embedding_dim=4, n_layers=1)
        model = SeparableCritic(embedding_net_x=net_x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return Trainer(model, dummy_estimator, optimizer, torch.device('cpu'))

    @pytest.mark.parametrize('value', ['full', 1.0, True, 32, 0.5])
    def test_recognised_values_produce_history(self, value):
        """Every documented eval_train value yields a per-epoch train history."""
        results = self._trainer().train(
            self._dataset(), n_epochs=4, batch_size=20,
            eval_train=value, verbose=False,
        )
        assert 'train_mi_history' in results, f"eval_train={value!r} produced no history"
        assert len(results['train_mi_history']) == 4

    @pytest.mark.parametrize('value', ['fulll', 'all', 'true', -1, [1]])
    def test_unrecognised_values_raise(self, value):
        """An unrecognised eval_train raises rather than silently disabling tracking."""
        with pytest.raises(ValueError, match='not a recognised value'):
            self._trainer().train(
                self._dataset(), n_epochs=2, batch_size=20,
                eval_train=value, verbose=False,
            )

    def test_full_is_uncapped_by_max_eval_samples(self):
        """'full' evaluates the whole training set even when max_eval_samples is smaller.

        A capped evaluation subset carries a lower InfoNCE ceiling, so the train
        curve would sit below the reported estimate for reasons unrelated to
        training. 'full' is what makes the two comparable.
        """
        ds = self._dataset(n=200)
        capped = self._trainer().train(
            ds, n_epochs=2, batch_size=20, eval_train=True,
            max_eval_samples=16, verbose=False,
        )
        full = self._trainer().train(
            ds, n_epochs=2, batch_size=20, eval_train='full',
            max_eval_samples=16, verbose=False,
        )
        # Both produce history; they must not be the same evaluation, since one
        # is capped at 16 samples and the other spans the whole training split.
        assert len(capped['train_mi_history']) == len(full['train_mi_history']) == 2
        assert capped['train_mi_history'] != full['train_mi_history']

    @pytest.mark.parametrize('value,expected_frac', [
        ('full', 1.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.25),
    ])
    def test_selected_evaluation_size(self, monkeypatch, value, expected_frac):
        """'full' and 1.0 select the whole training split; a fraction selects that share.

        Asserted on the requested subset size rather than on the resulting MI,
        since two training runs are not bit-identical.
        """
        trainer = self._trainer()
        seen = []
        original = trainer._select_eval_indices

        def spy(train_idx, n, is_temporal):
            seen.append((len(train_idx), n))
            return original(train_idx, n, is_temporal)

        monkeypatch.setattr(trainer, '_select_eval_indices', spy)
        trainer.train(self._dataset(), n_epochs=2, batch_size=20,
                      eval_train=value, verbose=False)

        # The last call is the per-epoch tracking subset.
        n_train, n_selected = seen[-1]
        assert n_selected == pytest.approx(n_train * expected_frac, abs=2)


class TestTrackEmbeddings:
    """track_embeddings accepts the same value forms as eval_train."""

    def _dataset(self, n=200):
        return PairedDataset(torch.randn(n, 4), torch.randn(n, 4))

    def _trainer(self):
        net_x = MLP(input_dim=4, hidden_dim=8, embedding_dim=4, n_layers=1)
        model = SeparableCritic(embedding_net_x=net_x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return Trainer(model, dummy_estimator, optimizer, torch.device('cpu'))

    @pytest.mark.parametrize('value,expected', [
        ('full', 200), (1.0, 200), (32, 32), (0.25, 50),
    ])
    def test_recognised_values_select_expected_count(self, value, expected):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = self._trainer().train(
                self._dataset(n=200), n_epochs=2, batch_size=20,
                track_embeddings=value, verbose=False,
            )
        history = results['embedding_history_x']
        assert history[0].shape[0] == expected

    @pytest.mark.parametrize('value', ['fulll', 'all', -1])
    def test_unrecognised_values_raise(self, value):
        with pytest.raises(ValueError, match='not a recognised value'):
            self._trainer().train(
                self._dataset(), n_epochs=2, batch_size=20,
                track_embeddings=value, verbose=False,
            )


class TestPeakFraction:
    """Tests for peak_fraction, which takes a fraction of the peak and applies with epoch_rule='peak'."""

    def _dataset(self, n=200):
        x = torch.randn(n, 4)
        y = torch.randn(n, 4)
        return PairedDataset(x, y)

    def _trainer(self):
        net_x = MLP(input_dim=4, hidden_dim=8, embedding_dim=4, n_layers=1)
        model = SeparableCritic(embedding_net_x=net_x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return Trainer(model, dummy_estimator, optimizer, torch.device('cpu'))

    def test_conservative_keys_present_when_peak_fraction_lt_1(self):
        """peak_fraction < 1.0 must add conservative_epoch and train_mi_at_peak to results."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.5, epoch_rule='peak', verbose=False,
        )
        assert 'conservative_epoch' in results, "conservative_epoch missing"
        assert 'train_mi_at_peak' in results, "train_mi_at_peak missing"

    def test_both_estimates_are_finite_floats(self):
        """train_mi and train_mi_at_peak are both proper fresh evaluations (finite floats)."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.5, epoch_rule='peak', verbose=False,
        )
        assert isinstance(results['train_mi'], float)
        assert isinstance(results['train_mi_at_peak'], float)
        assert np.isfinite(results['train_mi'])
        assert np.isfinite(results['train_mi_at_peak'])

    def test_conservative_epoch_le_best_epoch(self):
        """The conservative epoch must not be later than the best epoch."""
        results = self._trainer().train(
            self._dataset(), n_epochs=6, batch_size=20,
            peak_fraction=0.5, epoch_rule='peak', verbose=False,
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
            peak_fraction=0.7, epoch_rule='peak', eval_train=False, verbose=False,
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
    climbing. Patience ends training only once the curve stops improving, so
    nothing else signals a run cut off on the rise, and since these are
    lower-bound estimators, under-training biases the estimate downward."""

    def _trainer(self, dummy_model, custom_smoothing_fn):
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
        return Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'),
                       custom_smoothing_fn=custom_smoothing_fn)

    def test_fires_when_the_curve_is_still_rising_at_the_epoch_limit(self, dummy_data, dummy_model, monkeypatch):
        """A held-out curve still rising when training reaches n_epochs -> warn. It ends at 3 nats, clear of
        3 standard errors of the 10 held-out rows, sqrt(2 * 3 / 10), so the curve is resolved."""
        rising = iter(0.1 * np.arange(1, 31) + 0.001 * np.random.default_rng(0).standard_normal(30))
        evaluate = Trainer._safe_eval_mi
        monkeypatch.setattr(Trainer, '_safe_eval_mi', lambda self, *a, **k: next(rising, None) or evaluate(self, *a, **k))
        trainer = self._trainer(dummy_model, None)
        with pytest.warns(UserWarning, match="under-trained lower bound"):
            trainer.train(dummy_data, n_epochs=30, batch_size=20, patience=1000, verbose=False)

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


class TestTheDefaultStoppingRule:
    """Training stops once the smoothed held-out curve has not improved for `patience` epochs, and `n_epochs`
    caps a curve that keeps improving. Shifted windows keep the curve rising for longer, and a curve without
    them often tops early and falls, so no one fixed budget serves both (ROADMAP C18)."""

    def _epochs_trained(self, curve, dummy_data, dummy_model, monkeypatch):
        from neural_mi.defaults import default_of
        values = iter(curve)
        evaluate = Trainer._safe_eval_mi
        monkeypatch.setattr(Trainer, '_safe_eval_mi', lambda self, *a, **k: next(values, None) or evaluate(self, *a, **k))
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.01)
        trainer = Trainer(dummy_model, dummy_estimator, optimizer, torch.device('cpu'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = trainer.train(dummy_data, n_epochs=default_of('n_epochs'), batch_size=20, verbose=False)
        return len(result['test_mi_history']), [str(w.message) for w in caught]

    def test_a_curve_that_tops_early_stops_patience_epochs_later(self, dummy_data, dummy_model, monkeypatch):
        from neural_mi.defaults import default_of
        top = 10
        curve = [min(e, top) / top for e in range(1, default_of('n_epochs') + 1)]
        trained, _ = self._epochs_trained(curve, dummy_data, dummy_model, monkeypatch)
        assert top + default_of('patience') <= trained < top + default_of('patience') + 10

    def test_a_curve_that_keeps_improving_trains_every_epoch_and_says_so(self, dummy_data, dummy_model,
                                                                        monkeypatch):
        from neural_mi.defaults import default_of
        n = default_of('n_epochs')
        rising = 0.01 * np.arange(1, n + 1) + 0.001 * np.random.default_rng(0).standard_normal(n)
        trained, messages = self._epochs_trained(rising, dummy_data, dummy_model, monkeypatch)
        assert trained == n
        assert any('still rising' in m for m in messages)


class TestSelectTrainEvalIndices:
    """An eval subset of rows in time order is built from contiguous sub-chunks
        of train_idx's own contiguous segments, in proportion to their lengths, so
        it covers every part of the recording.
    """

    def _select(self, train_idx, target_size, is_temporal):
        return Trainer.__new__(Trainer)._select_eval_indices(train_idx, target_size, is_temporal)

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
        scatter."""
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


class TestShiftEvaluationConsistency:
    """Shifting affects training dynamics only. The reported test_mi and
        train_mi are evaluated on content frozen before any shift and on the
        original test and train-eval indices, so they do not depend on whichever
        shift the last epoch left behind.
    """

    @staticmethod
    def _check(plan):
        from neural_mi.data.plan import PAIR
        from neural_mi.estimators import infonce_lower_bound
        dataset = plan.build(PAIR)
        assert dataset._window_shifter is not None
        x0, y0 = dataset.x_data.clone(), dataset.y_data.clone()
        n = len(dataset)
        train_idx, test_idx = np.arange(int(n * 0.8)), np.arange(int(n * 0.8), n)
        net_x = MLP(input_dim=x0.shape[1] * x0.shape[2], hidden_dim=16, embedding_dim=8, n_layers=1)
        net_y = MLP(input_dim=y0.shape[1] * y0.shape[2], hidden_dim=16, embedding_dim=8, n_layers=1)
        critic = SeparableCritic(embedding_net_x=net_x, embedding_net_y=net_y)
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        trainer = Trainer(critic, infonce_lower_bound, optimizer, torch.device('cpu'))
        results = trainer.train(dataset, n_epochs=6, batch_size=32, patience=3,
                                train_indices=train_idx, test_indices=test_idx,
                                shift_windows=True, show_progress=False, verbose=False,
                                output_units='nats')
        with torch.no_grad():
            manual_mi = trainer._safe_eval_mi(x0[test_idx], y0[test_idx], 5000)
        assert abs(results['test_mi'] - manual_mi) < 1e-6, (
            f"reported test_mi ({results['test_mi']}) should exactly match a fresh eval of the "
            f"trained model against the canonical (shift=0) view ({manual_mi})"
        )

    def test_rows_on_a_regular_clock(self):
        from collections import OrderedDict
        from neural_mi.data.plan import RowPlan, resolve_stream
        np.random.seed(0)
        torch.manual_seed(0)
        raw_x, raw_y = np.random.randn(3000, 2), np.random.randn(3000, 2)
        self._check(RowPlan(OrderedDict(x=resolve_stream(raw_x, None, 'continuous', {'window_size': 20}),
                                        y=resolve_stream(raw_y, None, 'continuous', {})), shift=True))

    def test_rows_on_a_grid_in_time(self):
        from collections import OrderedDict
        from neural_mi.data.plan import RowPlan, resolve_stream
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        spikes = [np.sort(rng.uniform(0, 300, 1500)) for _ in range(3)]
        self._check(RowPlan(OrderedDict(x=resolve_stream(spikes, None, 'spike', {'window_size': 0.5}),
                                        y=resolve_stream([s + 0.02 for s in spikes], None, 'spike', {})),
                            shift=True))


def test_a_reported_mi_is_never_negative_and_keeps_the_measured_value():
    """A network's reported MI is 0 when its test MI never rose above zero, and
    also when the training-side value at the reported epoch came out below zero.
    Either way raw_train_mi keeps the measured value, and the second case warns.

    Small, briefly trained runs produce both cases. The scan requires at least
    one run of the second kind, so the rule is exercised and not assumed.
    """
    import neural_mi as nmi
    rng = np.random.default_rng(0)
    partly_negative = 0
    for seed in range(12):
        x = rng.standard_normal((60, 2)).astype(np.float32)
        y = (x + rng.standard_normal((60, 2))).astype(np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = nmi.run(x, y, model=nmi.Model(embedding_dim=4, hidden_dim=8, n_layers=1),
                             training=nmi.Training(n_epochs=2, learning_rate=1e-4, batch_size=8,
                                                   patience=1),
                             split=nmi.Split(mode='random'), seed=seed, show_progress=False)
        row = result.runs.iloc[0]
        said = any('reported epoch is negative' in str(w.message) for w in caught)
        assert result.mi_estimate >= 0 and row['train_mi'] >= 0
        if row['raw_train_mi'] < 0 and not row['all_mi_negative']:
            partly_negative += 1
            assert row['train_mi'] == 0 and said
        else:
            assert not said
    assert partly_negative >= 1, "no run came out partly negative; widen the scan"
