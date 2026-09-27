"""Model-hyperparameter sweeps through the quantities API."""
import numpy as np
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
        (q.predictive_information, dict(k=2)),
        (q.cross_predictive_information, dict(k=2)),
    ])
    def test_other_single_mi_quantities_accept_a_grid(self, fn, kwargs):
        x = _ar1()
        args = (x,) if fn is q.predictive_information else (x, _ar1(seed=1))
        r = fn(*args, **kwargs,
               sweep_grid={'hidden_dim': [16, 32], 'run_id': [0]},
               **{k: v for k, v in _FAST.items() if k != 'model'},
               model=nmi.Model(embedding_model='mlp', embedding_dim=8))
        assert isinstance(r, Results)
        assert 'mi_mean' in r.dataframe.columns


class TestMiRateSweepsEitherWindow:
    """`mi_rate` carries two windows and both change the answer.

    They bias in opposite directions, so a curve that has flattened along one
    of them proves nothing on its own: too little conditioning history reads
    high, too narrow a window on X reads low. Sweeping either one has to be
    possible for that check to be doable at all.
    """

    @staticmethod
    def _cfg():
        from neural_mi import Model, Training
        return dict(model=Model(embedding_model='dual_branch', embedding_dim=4,
                                hidden_dim=16, n_layers=1),
                    training=Training(n_epochs=1), n_workers=1, show_progress=False)

    @staticmethod
    def _pair():
        rng = np.random.default_rng(0)
        latent = rng.standard_normal((400, 2))
        mk = lambda: (latent @ rng.standard_normal((2, 2))
                      + 0.4 * rng.standard_normal((400, 2))).astype(np.float32)
        return mk(), mk()

    def test_sweeping_h_records_the_fixed_half_width(self):
        x, y = self._pair()
        r = q.mi_rate(x, y, h=[1, 2], half_width=5, **self._cfg())
        assert list(r.dataframe['h']) == [1, 2]
        assert r.params['config_keys'] == ['h']
        assert r.params['half_width'] == 5

    def test_sweeping_half_width_records_the_fixed_h(self):
        x, y = self._pair()
        r = q.mi_rate(x, y, h=2, half_width=[3, 5], **self._cfg())
        assert r.dataframe['half_width'].tolist() == [3, 5]
        assert r.params['config_keys'] == ['half_width']
        assert r.params['h'] == 2

    def test_both_at_once_is_refused(self):
        x, y = self._pair()
        with pytest.raises(ValueError, match="iterable for h or for half_width"):
            q.mi_rate(x, y, h=[1, 2], half_width=[3, 5], **self._cfg())
