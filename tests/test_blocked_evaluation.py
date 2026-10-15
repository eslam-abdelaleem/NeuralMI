# tests/test_blocked_evaluation.py
"""A train-side evaluation larger than max_eval_samples runs in blocks and averages them.

One evaluation scores every pair of its rows, so its memory grows with the
square of the rows. Up to max_eval_samples rows nothing changes. Above it, the
train side evaluates every row in blocks of at most max_eval_samples and
averages the blocks, which keeps the bound and ceiling of one block and the
precision of all the rows. The held-out side keeps one random subset.
"""
import math

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.data import PairedDataset
from neural_mi.estimators import infonce_lower_bound
from neural_mi.models.critics import SeparableCritic
from neural_mi.models.embeddings import MLP
from neural_mi.training.trainer import Trainer


def _trainer():
    torch.manual_seed(0)
    net = MLP(input_dim=3, hidden_dim=8, embedding_dim=4, n_layers=1)
    trainer = Trainer(SeparableCritic(embedding_net_x=net), infonce_lower_bound,
                      torch.optim.Adam(net.parameters(), lr=1e-3), torch.device('cpu'))
    trainer.model.eval()          # as during evaluation: spectral norm stops updating, so scores are repeatable
    return trainer


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    return x, x + 0.5 * torch.randn(n, 3, generator=g)


class TestSafeEvalMi:

    def test_small_sets_are_one_evaluation(self):
        trainer = _trainer()
        x, y = _rows(40)
        assert trainer._safe_eval_mi(x, y, 40, all_rows=True) == pytest.approx(trainer._eval_mi(x, y))

    def test_large_train_side_sets_average_interleaved_blocks(self):
        trainer = _trainer()
        x, y = _rows(10)
        blocks = [torch.arange(b, 10, 3) for b in range(3)]           # sizes 4, 3, 3
        expected = np.average([trainer._eval_mi(x[i], y[i]) for i in blocks], weights=[len(i) for i in blocks])
        assert trainer._safe_eval_mi(x, y, 4, all_rows=True) == pytest.approx(expected)
        assert trainer._blocked_eval == (10, 3, 4)

    def test_held_out_side_keeps_one_random_subset(self, monkeypatch):
        trainer = _trainer()
        sizes = []
        evaluate = trainer._eval_mi
        monkeypatch.setattr(trainer, '_eval_mi', lambda x, y: sizes.append(len(x)) or evaluate(x, y))
        trainer._safe_eval_mi(*_rows(10), 4)
        assert sizes == [4]


class TestTrainSubsetSize:

    def _train(self, **kwargs):
        x, y = _rows(400, seed=1)
        return _trainer().train(PairedDataset(x, y), n_epochs=2, batch_size=32, split_mode='random',
                                verbose=False, show_progress=False, **kwargs)

    def test_default_is_unchanged(self):
        results = self._train(max_eval_samples=50)
        assert results['train_eval_size'] == 50 and results['train_eval_blocks'] == 1
        assert results['train_ceiling_mi'] == pytest.approx(math.log(50))

    def test_full_evaluates_every_training_row_in_blocks(self, caplog):
        with caplog.at_level('WARNING', logger='neural_mi'):
            results = self._train(max_eval_samples=50, train_subset_size='full')
        n_train = 360                                                   # train_fraction 0.9 of 400
        assert results['train_eval_size'] == n_train
        assert results['train_eval_blocks'] == math.ceil(n_train / 50)
        per_block = math.ceil(n_train / results['train_eval_blocks'])
        assert results['train_ceiling_mi'] == pytest.approx(math.log(per_block))
        assert 'evaluated on all 360 rows' in caplog.text

    def test_unrecognised_string_raises(self):
        with pytest.raises(ValueError, match='not a recognised value'):
            self._train(train_subset_size='all')


def test_full_reaches_run_and_is_reported_per_repeat():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 2))
    y = x + rng.standard_normal((600, 2))
    result = nmi.run(x, y, model=nmi.Model(embedding_dim=4, hidden_dim=8),
                     training=nmi.Training(n_epochs=1, batch_size=64, max_eval_samples=100, train_subset_size='full'),
                     seed=0, show_progress=False, verbose=False)
    run = result.runs.iloc[0]
    assert run['train_eval_size'] > 100 and run['train_eval_blocks'] == math.ceil(run['train_eval_size'] / 100)
