# tests/test_paired_terms.py
"""The terms of a difference quantity are trained and evaluated on the same rows.

Conditional MI, transfer entropy, interaction information and the quantities
built on them are differences of separately trained networks. The difference
is the information of one dataset only when every term splits and evaluates
the same rows. Before this was enforced, the terms shared their rows only when
the caller passed a seed: without one, each term drew its own split.
"""
import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.training.trainer import Trainer
from neural_mi.data import PairedDataset
from neural_mi.models.critics import SeparableCritic
from neural_mi.models.embeddings import MLP

_FAST = dict(model=nmi.Model(embedding_dim=4, hidden_dim=8),
             training=nmi.Training(n_epochs=1, batch_size=64, max_eval_samples=200),
             show_progress=False, verbose=False)


def _data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = 0.5 * np.roll(x, 1, axis=0) + rng.standard_normal((n, 2))
    w = rng.standard_normal((n, 2))
    return x, y, w


@pytest.fixture
def rows(monkeypatch):
    """Every split and every set of evaluation rows the trainer draws, in order."""
    seen = {'split': [], 'eval': []}
    split, select = Trainer._create_blocked_split, Trainer._select_eval_indices

    def spy_split(self, *args, **kwargs):
        out = split(self, *args, **kwargs)
        seen['split'].append((out[0].copy(), out[1].copy()))
        return out

    def spy_select(self, *args, **kwargs):
        out = select(self, *args, **kwargs)
        seen['eval'].append(out.copy())
        return out

    monkeypatch.setattr(Trainer, '_create_blocked_split', spy_split)
    monkeypatch.setattr(Trainer, '_select_eval_indices', spy_select)
    return seen


CALLS = {
    'transfer': (lambda x, y, w, **kw: nmi.transfer_entropy(x, y, history_window=2, **_FAST, **kw), 2),
    'conditional': (lambda x, y, w, **kw: nmi.run(x, y, mode='conditional',
                                                  conditional=nmi.Conditional(w_data=w), **_FAST, **kw), 2),
    'interaction': (lambda x, y, w, **kw: nmi.run(x, y, mode='interaction',
                                                  interaction=nmi.Interaction(w_data=w), **_FAST, **kw), 3),
}


@pytest.mark.parametrize('name', CALLS)
def test_terms_share_their_rows_without_a_seed(rows, name):
    call, n_terms = CALLS[name]
    call(*_data())
    assert len(rows['split']) == n_terms and len(rows['eval']) == 2 * n_terms   # held-out rows, then training rows
    train, test = rows['split'][0]
    for other_train, other_test in rows['split'][1:]:
        assert np.array_equal(train, other_train) and np.array_equal(test, other_test)
    for i, other in enumerate(rows['eval'][2:]):
        assert np.array_equal(rows['eval'][i % 2], other)


def test_repeats_differ_and_pair_term_by_term(rows):
    """Each repeat gets a split of its own, shared by the two terms of that repeat."""
    x, y, w = _data()
    nmi.transfer_entropy(x, y, history_window=2, sweep_grid={'run_id': range(2)}, **_FAST)
    (joint_0, joint_1, cond_0, cond_1) = [test for _, test in rows['split']]
    assert np.array_equal(joint_0, cond_0) and np.array_equal(joint_1, cond_1)
    assert not np.array_equal(joint_0, joint_1)


def test_unseeded_call_records_a_seed_that_reproduces_it():
    x, y, w = _data()
    first = nmi.transfer_entropy(x, y, history_window=2, **_FAST)
    seed = first.params['random_seed']
    assert isinstance(seed, int)
    again = nmi.transfer_entropy(x, y, history_window=2, seed=seed, **_FAST)
    for column in ('i_xypast_yfuture_mean', 'i_ypast_yfuture_mean'):
        assert again.dataframe[column].iloc[0] == first.dataframe[column].iloc[0]


def test_split_seed_isolates_the_split_from_other_draws():
    """Two runs with the same split_seed split alike even when other code draws from NumPy in between."""
    def split_of(extra_draws):
        torch.manual_seed(0)
        np.random.seed(0)
        np.random.standard_normal(extra_draws)
        net = MLP(input_dim=3, hidden_dim=8, embedding_dim=4, n_layers=1)
        trainer = Trainer(SeparableCritic(embedding_net_x=net), lambda s, **k: s.mean(),
                          torch.optim.Adam(net.parameters(), lr=1e-3), torch.device('cpu'))
        data = PairedDataset(torch.randn(300, 3), torch.randn(300, 3))
        trainer.train(data, n_epochs=1, batch_size=32, split_seed=11, verbose=False, show_progress=False)
        return trainer.last_train_indices

    assert np.array_equal(split_of(0), split_of(17))
