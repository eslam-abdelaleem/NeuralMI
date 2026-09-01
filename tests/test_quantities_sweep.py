"""Model-hyperparameter sweeps through the quantities API."""
import numpy as np
import pandas as pd
import pytest

import neural_mi as nmi
from neural_mi import quantities as q
from neural_mi.results import Results


def _ar1(T=4000, n_ch=3, phi=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T + 100, n_ch))
    for t in range(1, T + 100):
        x[t] = phi * x[t - 1] + rng.normal(0, 1, n_ch)
    return x[100:].astype('float32')


_FAST = dict(training=nmi.Training(n_epochs=6, patience=3, batch_size=128),
             model=nmi.Model(embedding_model='mlp', embedding_dim=8, hidden_dim=32),
             n_workers=1, show_progress=False)


class TestQuantitySweep:
    def test_sweep_grid_aggregates_over_a_group_variable(self):
        x = _ar1()
        r = q.active_information_storage(
            x, k=2, sweep_grid={'hidden_dim': [16, 32], 'run_id': [0, 1]},
            **{k: v for k, v in _FAST.items() if k != 'model'},
            model=nmi.Model(embedding_model='mlp', embedding_dim=8))
        assert isinstance(r, Results)
        df = r.dataframe
        assert 'mi_mean' in df.columns and 'mi_std' in df.columns
        assert sorted(df['hidden_dim']) == [16, 32]

    def test_without_a_grid_it_is_still_a_single_estimate(self):
        x = _ar1()
        r = q.active_information_storage(x, k=2, **_FAST)
        assert isinstance(r, Results)
        assert np.isfinite(r.mi_estimate)

    def test_parameter_sweep_over_k_is_unchanged(self):
        x = _ar1()
        r = q.active_information_storage(x, k=[1, 2], **_FAST)
        assert isinstance(r, nmi.Results)
        assert sorted(r.dataframe['k']) == [1, 2]

    @pytest.mark.parametrize('fn,kwargs', [
        (q.excess_entropy, dict(k=2, future_k=2)),
        (q.cross_predictive_information, dict(past_k=2, future_k=1)),
    ])
    def test_other_single_mi_quantities_accept_a_grid(self, fn, kwargs):
        x = _ar1()
        args = (x,) if fn is q.excess_entropy else (x, _ar1(seed=1))
        r = fn(*args, **kwargs,
               sweep_grid={'hidden_dim': [16, 32], 'run_id': [0]},
               **{k: v for k, v in _FAST.items() if k != 'model'},
               model=nmi.Model(embedding_model='mlp', embedding_dim=8))
        assert isinstance(r, Results)
        assert 'mi_mean' in r.dataframe.columns
