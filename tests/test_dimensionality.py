"""mode='dimensionality': the MI against embedding dimension and where it saturates.

The procedure is tested with a stand-in for the fitting step that returns known
curves, so the grid, the early stop, the extension, the readings and the warnings
are checked exactly and fast. One small real run checks the wiring end to end.
"""
import logging
import math
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi.analysis import dimensionality as dim
from neural_mi.analysis.dimensionality import run_dimensionality_analysis

BITS = math.log(2)          # one bit in nats
TOTAL = 4 * BITS            # the fake curves carry 4 bits


class FakeFits:
    """Stands in for ``_fit``: every network reads ``curve(k, split, restart)``."""

    def __init__(self, curve, pr=3.0, test_ratio=0.97, eval_size=1000):
        self.curve, self.pr, self.test_ratio, self.eval_size = curve, pr, test_ratio, eval_size
        self.calls, self.params = [], []

    def __call__(self, views, ks, n_restarts, n_workers):
        self.calls.append(list(ks))
        out = []
        for split_id, _x, _y, params in views:
            self.params.append(params)
            for k in ks:
                for r in range(n_restarts):
                    mi = self.curve(k, split_id, r)
                    out.append(dict(split_id=split_id, embedding_dim=k, run_id=r, train_mi=mi,
                                    test_mi=self.test_ratio * mi, eval_size=self.eval_size,
                                    pr_singular=self.pr, pr_eig=self.pr, best_epoch=5,
                                    test_mi_history=[0.0] * 20))
        return out


def saturating_at(d):
    return lambda k, s, r: TOTAL * min(k, d) / d


@pytest.fixture
def xy():
    return torch.randn(200, 6), torch.randn(200, 6)


def analyse(monkeypatch, fake, x, y=None, **kwargs):
    monkeypatch.setattr(dim, '_fit', fake)
    kwargs.setdefault('n_restarts', 2)
    return run_dimensionality_analysis(x, {'split_mode': 'random'}, y_data=y, **kwargs)


# ---------------------------------------------------------------------------
# The grid, the curve and the reading
# ---------------------------------------------------------------------------

class TestGrid:
    def test_a_small_participation_ratio_runs_one_to_ten(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            assert dim._default_grid(3.0) == list(range(1, 11))
        assert not caplog.records

    def test_a_middle_ratio_runs_one_to_twenty_and_says_so(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            assert dim._default_grid(7.0) == list(range(1, 21))
        assert '1 to 20' in caplog.text

    def test_a_large_ratio_runs_a_log_grid_to_twice_the_ratio(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            grid = dim._default_grid(15.0)
        assert grid[0] == 1 and grid[-1] == 30 and len(grid) <= 10
        assert 'log scale' in caplog.text and 'embedding_dims' in caplog.text

    def test_the_log_grid_is_increasing_and_spans_its_range(self):
        grid = dim._log_grid(11, 63)
        assert grid[0] == 11 and grid[-1] == 63 and grid == sorted(set(grid))


class TestReading:
    def test_the_curve_is_the_running_maximum(self):
        assert dim._curve({1: 1.0, 2: 3.0, 3: 2.5, 4: 3.2}) == {1: 1.0, 2: 3.0, 3: 3.0, 4: 3.2}

    def test_the_reading_is_the_first_k_at_the_threshold(self):
        best = {1: 1.0, 2: 2.0, 3: 3.9, 4: 4.0, 64: 4.0}
        assert dim._reading(best, 0.95) == (3, 4.0)

    def test_the_best_restart_counts(self):
        fits = [dict(split_id=0, embedding_dim=2, train_mi=v) for v in (1.0, 3.0, 2.0)]
        assert dim._best(fits, 0) == {2: 3.0}

    def test_three_consecutive_values_confirm(self):
        best = {1: 1.0, 2: 4.0, 3: 4.0, 4: 4.0}
        assert dim._confirmed(best, [1, 2, 3], 3.8) is None
        assert dim._confirmed(best, [1, 2, 3, 4], 3.8) == 2


# ---------------------------------------------------------------------------
# The procedure
# ---------------------------------------------------------------------------

class TestProcedure:
    def test_the_reading_matches_a_known_dimension(self, monkeypatch, xy):
        fits, info = analyse(monkeypatch, FakeFits(saturating_at(3)), *xy)
        assert info['dimension_at_most'] == 3
        assert info['dimension_at_most_per_split'] == {0: 3}
        assert info['dimension_at_most_std'] is None

    def test_the_reference_fit_runs_first(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(3))
        _, info = analyse(monkeypatch, fake, *xy)
        assert fake.calls[0] == [64] and info['reference_dim'] == 64

    def test_the_default_grid_stops_after_three_values_at_the_threshold(self, monkeypatch, xy, caplog):
        fake = FakeFits(saturating_at(3))
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            _, info = analyse(monkeypatch, fake, *xy)
        assert fake.calls[1:] == [[1, 2, 3], [4, 5, 6]]
        assert info['stopped_early'] and info['embedding_dims'] == [1, 2, 3, 4, 5, 6, 64]
        assert any('the grid stopped at embedding_dim=6' in r.getMessage() for r in caplog.records)

    def test_explicit_embedding_dims_are_all_fitted(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(3))
        _, info = analyse(monkeypatch, fake, *xy, embedding_dims=range(1, 9))
        assert fake.calls[1:] == [list(range(1, 9))]
        assert not info['stopped_early'] and info['dimension_at_most'] == 3

    def test_a_reference_close_to_its_ratio_is_refitted_larger(self, monkeypatch, xy, caplog):
        fake = FakeFits(saturating_at(3), pr=40.0)
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            _, info = analyse(monkeypatch, fake, *xy, embedding_dims=[1, 2, 3, 4])
        assert fake.calls[:2] == [[64], [160]] and info['reference_dim'] == 160
        assert any('refitted at embedding_dim=160' in r.getMessage() for r in caplog.records)

    def test_a_reference_you_set_is_kept(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(3), pr=40.0)
        _, info = analyse(monkeypatch, fake, *xy, reference_dim=16, embedding_dims=[1, 2, 3, 4])
        assert fake.calls[0] == [16] and info['reference_dim'] == 16

    def test_a_curve_still_climbing_past_the_grid_extends_it(self, monkeypatch, xy, caplog):
        fake = FakeFits(saturating_at(30), pr=3.0)
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            _, info = analyse(monkeypatch, fake, *xy)
        assert any('The grid is extended' in r.getMessage() for r in caplog.records)
        assert 28 <= info['dimension_at_most'] < 64

    def test_a_curve_saturating_only_at_the_reference_warns(self, monkeypatch, xy):
        fake = FakeFits(lambda k, s, r: TOTAL * k / 64)
        with pytest.warns(UserWarning, match='only at the reference'):
            _, info = analyse(monkeypatch, fake, *xy, embedding_dims=[1, 2, 4])
        assert info['dimension_at_most_per_split'][0] == 64

    def test_several_random_splits_give_a_median_and_a_spread(self, monkeypatch, xy):
        fake = FakeFits(lambda k, s, r: TOTAL * min(k, 2 + s) / (2 + s))
        _, info = analyse(monkeypatch, fake, xy[0], n_splits=3, embedding_dims=range(1, 8))
        assert info['dimension_at_most_per_split'] == {0: 2, 1: 3, 2: 4}
        assert info['dimension_at_most'] == 3
        assert info['dimension_at_most_std'] == pytest.approx(1.0)

    def test_x_alone_is_split_five_times_by_default(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        _, info = analyse(monkeypatch, fake, xy[0], embedding_dims=[1, 2, 3])
        assert len(info['dimension_at_most_per_split']) == 5

    def test_the_restarts_share_one_held_out_set(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        analyse(monkeypatch, fake, xy[0], n_splits=2, embedding_dims=[1, 2, 3])
        tests = [tuple(np.asarray(p['test_indices'])) for p in fake.params]
        assert len(set(tests)) == 1


class TestWarnings:
    def test_restarts_that_disagree_below_the_reading_warn(self, monkeypatch, xy):
        fake = FakeFits(lambda k, s, r: TOTAL * min(k, 3) / 3 * (0.5 if r == 0 and k < 3 else 1.0))
        with pytest.warns(UserWarning, match='below the reading differ by more than'):
            analyse(monkeypatch, fake, *xy, embedding_dims=range(1, 6))

    def test_a_held_out_plateau_far_below_the_training_side_warns(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(3), test_ratio=0.7)
        with pytest.warns(UserWarning, match='overstate the dimension'):
            analyse(monkeypatch, fake, *xy, embedding_dims=range(1, 6))

    def test_a_plateau_near_the_ceiling_warns(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(3), eval_size=20)
        with pytest.warns(UserWarning, match='near its evaluation ceiling'):
            analyse(monkeypatch, fake, *xy, embedding_dims=range(1, 6))

    def test_views_that_share_nothing_have_no_reading(self, monkeypatch, xy):
        with pytest.warns(UserWarning, match='share no information'):
            _, info = analyse(monkeypatch, FakeFits(lambda k, s, r: 0.0), *xy, embedding_dims=[1, 2])
        assert info['dimension_at_most'] is None

    def test_a_clean_curve_raises_no_warning(self, monkeypatch, xy):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            analyse(monkeypatch, FakeFits(saturating_at(3)), *xy, embedding_dims=range(1, 6))


class TestSettings:
    def test_the_mode_trains_a_hybrid_critic_to_convergence(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        analyse(monkeypatch, fake, *xy, embedding_dims=[1, 2, 3])
        p = fake.params[0]
        assert p['critic_type'] == 'hybrid' and p['n_epochs'] == 500 and p['patience'] == 50

    def test_the_hybrid_critic_is_layer_normalised_here_only(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        analyse(monkeypatch, fake, *xy, embedding_dims=[1, 2, 3])
        assert fake.params[0]['norm_layer'] == 'layer'
        fake = FakeFits(saturating_at(2))
        monkeypatch.setattr(dim, '_fit', fake)
        with pytest.warns(UserWarning, match="critic_type='separable'"):
            run_dimensionality_analysis(xy[0], {'critic_type': 'separable', 'norm_layer': 'auto'},
                                        y_data=xy[1], embedding_dims=[1, 2, 3])
        assert fake.params[0]['norm_layer'] == 'auto'

    def test_a_norm_you_chose_is_kept(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        monkeypatch.setattr(dim, '_fit', fake)
        run_dimensionality_analysis(xy[0], {'norm_layer': 'none'}, y_data=xy[1],
                                    embedding_dims=[1, 2, 3], user_set_keys={'norm_layer'})
        assert fake.params[0]['norm_layer'] == 'none'

    def test_settings_you_chose_are_kept(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        monkeypatch.setattr(dim, '_fit', fake)
        run_dimensionality_analysis(xy[0], {'n_epochs': 7, 'patience': 3}, y_data=xy[1],
                                    embedding_dims=[1, 2, 3], user_set_keys={'n_epochs', 'patience'})
        assert fake.params[0]['n_epochs'] == 7 and fake.params[0]['patience'] == 3

    def test_halves_of_x_share_an_encoder_and_x_and_y_do_not(self, monkeypatch, xy):
        fake = FakeFits(saturating_at(2))
        analyse(monkeypatch, fake, xy[0], split_method='spatial', embedding_dims=[1, 2, 3])
        assert fake.params[0]['shared_encoder'] is True
        fake = FakeFits(saturating_at(2))
        analyse(monkeypatch, fake, *xy, embedding_dims=[1, 2, 3])
        assert fake.params[0]['shared_encoder'] is False

    def test_a_concat_critic_is_refused(self, xy):
        with pytest.raises(ValueError, match="cannot use critic_type='concat'"):
            run_dimensionality_analysis(xy[0], {'critic_type': 'concat'}, y_data=xy[1])

    def test_a_separable_critic_warns(self, monkeypatch, xy):
        monkeypatch.setattr(dim, '_fit', FakeFits(saturating_at(2)))
        with pytest.warns(UserWarning, match="critic_type='separable'"):
            run_dimensionality_analysis(xy[0], {'critic_type': 'separable'}, y_data=xy[1],
                                        embedding_dims=[1, 2, 3])

    def test_n_splits_with_y_is_refused(self, xy):
        with pytest.raises(ValueError, match='applies without y_data'):
            run_dimensionality_analysis(xy[0], {}, y_data=xy[1], n_splits=3)

    def test_n_splits_with_a_fixed_split_is_refused(self, xy):
        with pytest.raises(ValueError, match='would repeat one split'):
            run_dimensionality_analysis(xy[0], {}, split_method='spatial', n_splits=3)

    def test_an_unknown_split_method_is_refused(self, xy):
        with pytest.raises(ValueError, match='Unknown split_method'):
            run_dimensionality_analysis(xy[0], {}, split_method='zigzag')

    @pytest.mark.parametrize('kwargs', [dict(embedding_dims=[2, 64]),
                                        dict(embedding_dims=[2, 8], reference_dim=8)])
    def test_a_grid_that_reaches_the_reference_is_refused(self, xy, kwargs):
        with pytest.raises(ValueError, match='must be smaller than the reference'):
            run_dimensionality_analysis(xy[0], {}, y_data=xy[1], **kwargs)


# ---------------------------------------------------------------------------
# Splitting X
# ---------------------------------------------------------------------------

class TestHalves:
    def test_random_draws_one_assignment_per_split(self):
        halves = dim._halves(torch.randn(50, 8), {}, 'random', 4, {})
        assert len(halves) == 4 and all(a.shape == (50, 4) and b.shape == (50, 4) for a, b, _ in halves)

    def test_spatial_splits_at_the_midpoint_once(self):
        x = torch.arange(24.).reshape(3, 8)
        [(a, b, _)] = dim._halves(x, {}, 'spatial', 5, {})
        assert torch.equal(a, x[:, :4]) and torch.equal(b, x[:, 4:])

    def test_temporal_pairs_x_with_itself_later(self):
        x = torch.arange(10.).reshape(10, 1).repeat(1, 2)
        [(a, b, _)] = dim._halves(x, {}, 'temporal', 1, {'lag': 2})
        assert torch.equal(a, x[:-2]) and torch.equal(b, x[2:])

    @pytest.mark.parametrize('shape', [(20, 6), (20, 6, 3)])
    def test_index_takes_the_named_channels_and_the_rest(self, shape):
        x = torch.randn(*shape)
        [(a, b, _)] = dim._halves(x, {}, 'index', 1, {'channel_indices_x': [0, 2, 4]})
        assert torch.equal(a, x[:, [0, 2, 4], ...]) and torch.equal(b, x[:, [1, 3, 5], ...])

    def test_unequal_index_halves_turn_off_a_shared_encoder(self, caplog):
        with caplog.at_level(logging.WARNING, logger='neural_mi'):
            [(_, _, params)] = dim._halves(torch.randn(20, 5), {'shared_encoder': True}, 'index', 1,
                                           {'channel_indices_x': [0, 1]})
        assert params['shared_encoder'] is False
        assert 'unequal channel counts' in caplog.text

    @pytest.mark.parametrize('indices, match', [
        (None, "requires a 'channel_indices_x'"), ([0, 9], 'must be integers in'),
        ([0, 1, 2, 3], 'covers all channels'), ([], 'is empty')])
    def test_bad_index_splits_are_refused(self, indices, match):
        with pytest.raises(ValueError, match=match):
            dim._halves(torch.randn(20, 4), {}, 'index', 1, {'channel_indices_x': indices})

    def test_image_splits_need_four_dimensional_input(self):
        with pytest.raises(ValueError, match='requires 4-D input'):
            dim._halves(torch.randn(20, 4), {}, 'horizontal', 1, {})

    @pytest.mark.parametrize('method, a_shape, b_shape', [
        ('horizontal', (5, 1, 2, 4), (5, 1, 2, 4)), ('vertical', (5, 1, 4, 2), (5, 1, 4, 2)),
        ('row_interleaved', (5, 1, 2, 4), (5, 1, 2, 4)), ('col_interleaved', (5, 1, 4, 2), (5, 1, 4, 2)),
        ('diagonal', (5, 1, 10), (5, 1, 6)), ('antidiagonal', (5, 1, 10), (5, 1, 6))])
    def test_image_splits_cut_the_picture(self, method, a_shape, b_shape):
        [(a, b, _)] = dim._halves(torch.randn(5, 1, 4, 4), {}, method, 1, {})
        assert tuple(a.shape) == a_shape and tuple(b.shape) == b_shape

    def test_triangular_splits_refuse_a_convolutional_encoder(self):
        with pytest.raises(ValueError, match='triangular'):
            dim._halves(torch.randn(5, 1, 4, 4), {'embedding_model': 'cnn2d'}, 'diagonal', 1, {})


# ---------------------------------------------------------------------------
# Through run()
# ---------------------------------------------------------------------------

SMALL = dict(dimensionality=nmi.Dimensionality(n_restarts=1, reference_dim=4, embedding_dims=[1, 2]),
             training=nmi.Training(n_epochs=3, batch_size=64), model=nmi.Model(hidden_dim=16),
             show_progress=False)


def _data():
    return nmi.generators.generate_nonlinear_from_latent(400, 2, 6, 1.0, seed=0, use_torch=False)


def test_run_returns_the_curve_and_the_reading():
    x, y = _data()
    r = nmi.run(x, y, mode='dimensionality', output=nmi.Output(return_embeddings=True), seed=0, **SMALL)
    df = r.dataframe
    assert list(df['embedding_dim']) == [1, 2, 4]
    assert {'mi_best', 'mi_curve', 'split_id', 'n_runs'} <= set(df.columns)
    assert (np.diff(df['mi_curve']) >= 0).all()
    assert r.get('dimension_at_most') in (1, 2, 4)
    entry = r.details[0]
    assert set(entry['embeddings']) == {(0, 1, 0), (0, 2, 0), (0, 4, 0)}
    assert 0 in entry['embeddings_at_bound']
    r.summary()
    r.plot(show=False)


def test_an_embedding_dim_set_on_the_model_is_reported_as_ignored():
    x, y = _data()
    settings = dict(SMALL, model=nmi.Model(hidden_dim=16, embedding_dim=8))
    with pytest.warns(UserWarning, match=r"embedding_dim \(mode='dimensionality' sets it"):
        nmi.run(x, y, mode='dimensionality', seed=0, **settings)


@pytest.mark.parametrize('settings, match', [
    (dict(n_restarts=0), 'n_restarts must be a whole number'),
    (dict(n_splits=0), 'n_splits must be a whole number'),
    (dict(saturation_ratio=1.5), r'saturation_ratio must lie in \(0, 1\]'),
    (dict(reference_dim=1), 'reference_dim must be a whole number of 2'),
    (dict(embedding_dims=[0, 2]), 'embedding_dims must hold whole numbers'),
    (dict(embedding_dims=[]), 'embedding_dims must hold whole numbers')])
def test_bad_settings_are_refused_before_any_fit(settings, match):
    x, y = _data()
    with pytest.raises(ValueError, match=match):
        nmi.run(x, None, mode='dimensionality', dimensionality=nmi.Dimensionality(**settings),
                show_progress=False)


@pytest.mark.parametrize('grid, match', [({'embedding_dim': [2, 4]}, 'embedding_dims'),
                                         ({'run_id': [0, 1]}, 'n_restarts')])
def test_the_grid_cannot_vary_what_the_mode_varies(grid, match):
    x, y = _data()
    with pytest.raises(ValueError, match=match):
        nmi.run(x, y, mode='dimensionality', sweep_grid=grid, **SMALL)
