# tests/test_best_epoch.py
"""The network evaluated is the one at the reported best epoch.

The best epoch is read from the held-out curve smoothed after training: its
peak (epoch_rule='peak'), or the earliest epoch at its top ('plateau_start', the
default). The smoothed value at an epoch depends on the epochs after it, so the
trainer keeps copies of the network for every epoch that can still turn out to
be the one read (for peak_fraction < 1, the first epoch to reach that fraction
of the peak), and evaluates that epoch's network. These tests script the held-out curve, record
the network at every epoch, and check which network the reported value came from.
"""
import numpy as np
import pytest
import torch

from neural_mi.data import PairedDataset
from neural_mi.estimators import infonce_lower_bound
from neural_mi.models.critics import SeparableCritic
from neural_mi.models.embeddings import MLP
from neural_mi.training.epochs import choose_epoch
from neural_mi.training.trainer import Trainer

N, N_TEST, N_EPOCHS = 400, 40, 40          # a random split holds out 40 of 400 rows


def _curve(seed, nan_at=None):
    """A noisy held-out curve that rises to a peak and falls, in nats."""
    e = np.arange(N_EPOCHS)
    curve = 0.3 * np.exp(-0.5 * ((e - 12) / 4) ** 2) + 0.05 * np.random.default_rng(seed).standard_normal(N_EPOCHS)
    if nan_at is not None:
        curve[nan_at] = np.nan
    return curve


def _fingerprint(model):
    return float(sum(p.detach().double().sum() for p in model.parameters()))


def _train(monkeypatch, curve, custom_smoothing_fn=None, **kwargs):
    """Train with the held-out values replaced by `curve`. Returns the results, the network's
    fingerprint at every epoch, and the fingerprints behind every train-side evaluation."""
    at_epoch, train_side = [], []
    evaluate = Trainer._safe_eval_mi

    def spy(trainer, x, y, max_samples, all_rows=False):
        if len(y) == N_TEST and len(at_epoch) < len(curve):
            at_epoch.append(_fingerprint(trainer.model))
            return float(curve[len(at_epoch) - 1])
        if len(y) != N_TEST:
            train_side.append(_fingerprint(trainer.model))
        return evaluate(trainer, x, y, max_samples, all_rows)

    monkeypatch.setattr(Trainer, '_safe_eval_mi', spy)
    torch.manual_seed(0)
    net = MLP(input_dim=3, hidden_dim=8, embedding_dim=4, n_layers=1)
    trainer = Trainer(SeparableCritic(embedding_net_x=net), infonce_lower_bound,
                      torch.optim.Adam(net.parameters(), lr=1e-2), torch.device('cpu'),
                      custom_smoothing_fn=custom_smoothing_fn)
    x = torch.randn(N, 3)
    results = trainer.train(PairedDataset(x, x + torch.randn(N, 3)), n_epochs=len(curve), batch_size=32,
                            patience=1000, split_mode='random', split_seed=0, verbose=False,
                            show_progress=False, **kwargs)
    return results, at_epoch, train_side


@pytest.mark.parametrize('seed', range(5))
def test_evaluated_network_is_the_one_at_best_epoch(monkeypatch, seed):
    curve = _curve(seed)
    results, at_epoch, train_side = _train(monkeypatch, curve, epoch_rule='peak')
    assert results['best_epoch'] == int(np.argmax(Trainer._smooth(None, list(curve), 1.0, 5)))
    assert train_side[-1] == at_epoch[results['best_epoch']]


def test_conservative_epoch_evaluates_its_own_network(monkeypatch):
    curve = _curve(0)
    results, at_epoch, train_side = _train(monkeypatch, curve, peak_fraction=0.5, epoch_rule='peak')
    smoothed = Trainer._smooth(None, list(curve), 1.0, 5)
    expected = int(np.flatnonzero(smoothed >= 0.5 * smoothed.max())[0])
    assert results['conservative_epoch'] == expected < results['best_epoch']
    assert at_epoch[expected] in train_side and at_epoch[results['best_epoch']] in train_side


def test_a_nan_epoch_is_never_chosen(monkeypatch):
    curve = _curve(1, nan_at=12)
    results, at_epoch, train_side = _train(monkeypatch, curve, epoch_rule='peak')
    smoothed = Trainer._smooth(None, list(curve), 1.0, 5)
    assert results['best_epoch'] == int(np.nanargmax(smoothed)) != 12
    assert train_side[-1] == at_epoch[results['best_epoch']]


def test_custom_smoothing_keeps_every_epoch(monkeypatch):
    curve = _curve(2)
    results, at_epoch, train_side = _train(monkeypatch, curve, custom_smoothing_fn=np.asarray, epoch_rule='peak')
    assert results['best_epoch'] == int(np.argmax(curve))
    assert train_side[-1] == at_epoch[results['best_epoch']]


def test_plateau_start_reads_and_evaluates_the_start_of_the_top(monkeypatch):
    """The default rule reads the earliest epoch at the top of a plateau, and that epoch's network is evaluated."""
    e = np.arange(N_EPOCHS)
    curve = 0.3 * (1 - np.exp(-(e + 1) / 2)) + 0.005 * np.random.default_rng(4).standard_normal(N_EPOCHS)
    results, at_epoch, train_side = _train(monkeypatch, curve)
    smoothed = Trainer._smooth(None, list(curve), 1.0, 5)
    expected, peak, diagnosis = choose_epoch(curve, smoothed, 'plateau_start', Trainer._smoothing_reach(1.0, 5),
                                             ran_to_limit=True)
    assert results['best_epoch'] == expected < results['peak_epoch'] == peak
    assert results['curve_diagnosis'] == diagnosis
    assert train_side[-1] == at_epoch[expected]


def test_reach_covers_the_smoothing():
    """Changing epochs beyond the reach leaves an epoch's smoothed value unchanged."""
    rng = np.random.default_rng(0)
    for sigma, window in ((1.0, 5), (2.0, 3), (0.0, 5), (1.0, 1), (0.5, 4)):
        reach = Trainer._smoothing_reach(sigma, window)
        a = rng.standard_normal(60)
        b = a.copy()
        b[30 + reach + 1:] += 10 * rng.standard_normal(60 - 31 - reach)
        assert Trainer._smooth(None, list(a), sigma, window)[30] == Trainer._smooth(None, list(b), sigma, window)[30]
