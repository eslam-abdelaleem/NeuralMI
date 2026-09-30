# tests/test_results.py
import json
import math
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

from neural_mi.results import Results
from tests import results_factory as rf


class TestResults:
    def test_repr(self):
        r = rf.estimate(mi=1.2345)
        text = repr(r)
        assert "mode='estimate'" in text
        assert "mi_estimate=1.2345" in text
        assert "dataframe_rows=1" in text
        assert "runs=1" in text

        r = rf.sweep()
        assert "mi_estimate" not in repr(r)
        assert "dataframe_rows=4" in repr(r)

    def test_config_ids_follow_the_dataframe(self):
        assert rf.sweep().config_ids == [0, 1, 2, 3]
        assert rf.estimate().config_ids == [0]

    @patch('matplotlib.pyplot.show')
    @patch('neural_mi.visualize.plot.plot_sweep_curve')
    def test_plot_draws_a_sweep_against_its_key(self, mock_plot_sweep, mock_show):
        rf.sweep().plot(show=False)
        mock_plot_sweep.assert_called_once()
        assert mock_plot_sweep.call_args.kwargs['param_col'] == 'embedding_dim'

    @patch('matplotlib.pyplot.show')
    @patch('neural_mi.visualize.plot.plot_sweep_heatmap')
    def test_plot_draws_two_keys_as_a_heatmap(self, mock_heatmap, mock_show):
        rows = [{'config_id': 2 * i + j, 'a': a, 'b': b, 'run_id': 0, 'mi': a + b}
                for i, a in enumerate((1, 2)) for j, b in enumerate((10, 20))]
        rf.make_results('sweep', rows, config_keys=('a', 'b')).plot(show=False)
        mock_heatmap.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_raises_not_implemented(self, mock_show):
        r = rf.make_results('unknown_mode', [{'config_id': 0, 'run_id': 0, 'mi': 0.1}])
        with pytest.raises(NotImplementedError):
            r.plot()

    @patch('matplotlib.pyplot.show')
    def test_plot_rigorous_without_a_ladder_raises(self, mock_show):
        r = rf.make_results('rigorous', [{'config_id': 0, 'run_id': 0, 'mi': 0.5}])
        with pytest.raises(ValueError, match="no extrapolation to draw"):
            r.plot()

    @patch('matplotlib.pyplot.show')
    def test_plot_needs_a_config_id_for_a_per_configuration_figure(self, mock_show):
        rows = [{'config_id': c, 'k': k, 'run_id': 0, 'mi': 0.1, 'test_mi_history': [0.1]}
                for c, k in enumerate((1, 2))]
        r = rf.make_results('estimate', rows, config_keys=('k',))
        with pytest.raises(ValueError, match="config_id"):
            r.plot(kind='line', config_id=5)

    # ------------------------------------------------------------------ #
    # summary()                                                           #
    # ------------------------------------------------------------------ #

    def test_summary_estimate_mode(self, capsys):
        rf.estimate(mi=1.2345).summary()
        captured = capsys.readouterr().out
        assert "mode = 'estimate'" in captured
        assert "1.2345" in captured
        assert "bits" in captured

    def test_summary_reports_the_spread_of_repeats(self, capsys):
        rows = [{'config_id': 0, 'run_id': r, 'mi': v} for r, v in enumerate((1.0, 1.2, 1.4))]
        rf.make_results('sweep', rows).summary()
        captured = capsys.readouterr().out
        assert "1.2000" in captured
        assert "spread over 3 repeats" in captured

    def test_summary_of_several_rows_points_to_the_dataframe(self, capsys):
        rf.sweep().summary()
        captured = capsys.readouterr().out
        assert "mode = 'sweep'" in captured
        assert "4 rows over 4 configuration(s)" in captured

    def test_summary_rigorous_reliable(self, capsys):
        r = rf.rigorous(fits=[{'mi': 0.8, 'mi_error': 0.05, 'mi_error_pred': 0.09,
                               'is_reliable': True}], units='nats')
        r.summary()
        captured = capsys.readouterr().out
        assert "rigorous" in captured
        assert "0.8000" in captured
        assert "0.0500" in captured   # CI half-width
        assert "0.0900" in captured   # PI half-width
        assert "is_reliable = True" in captured

    def test_summary_rigorous_unreliable(self, capsys):
        rf.rigorous(fits=[{'mi': 0.3, 'mi_error': 0.2, 'is_reliable': False}]).summary()
        assert "is_reliable = False" in capsys.readouterr().out

    def test_summary_rigorous_reliable_shows_r_squared(self, capsys):
        rf.rigorous(fits=[{'mi': 0.8, 'mi_error': 0.05, 'is_reliable': True,
                           'r_squared': 0.987}]).summary()
        assert "R² = 0.987" in capsys.readouterr().out

    def test_summary_rigorous_reliable_hides_nan_r_squared(self, capsys):
        rf.rigorous(fits=[{'mi': 0.8, 'mi_error': 0.05, 'is_reliable': True,
                           'r_squared': float('nan')}]).summary()
        assert "R²" not in capsys.readouterr().out

    def test_summary_rigorous_unreliable_names_the_deciding_checks(self, capsys):
        # fit_quality_warning is reported beside the fit and decides nothing,
        # so it is never given as the reason.
        rf.rigorous(fits=[{'mi': 0.3, 'mi_error': 0.2, 'is_reliable': False,
                           'fit_quality_warning': True, 'leverage_warning': True,
                           'loo_intercept_shift': 0.31, 'saturated_gammas': [1, 2]}]).summary()
        captured = capsys.readouterr().out
        assert "LOO shift=0.310" in captured
        assert "ceiling-saturated gammas in the fit ([1, 2])" in captured
        assert "fit_quality_warning" not in captured

    def test_summary_rigorous_repeats_count_reliable_fits(self, capsys):
        r = rf.rigorous(fits=[{'mi': 0.5, 'mi_error': 0.1, 'is_reliable': True},
                              {'mi': 0.6, 'mi_error': 0.1, 'is_reliable': False}])
        r.summary()
        captured = capsys.readouterr().out
        assert "Reliable fits : 1 of 2" in captured
        # The repeats share data, so their intervals are not combined into one.
        assert "CI half-width" not in captured
        assert math.isnan(r.dataframe['mi_error'].iloc[0])

    # ------------------------------------------------------------------ #
    # plot() for one configuration                                        #
    # ------------------------------------------------------------------ #

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_returns_axes(self, mock_show):
        ax = rf.estimate().plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_no_best_epoch(self, mock_show):
        ax = rf.estimate(history=(0.1, 0.2, 0.3), best_epoch=None).plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_missing_history_raises(self, mock_show):
        r = rf.estimate(history=None)
        with pytest.raises(ValueError, match="test_mi_history"):
            r.plot()

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_uses_units_from_params(self, mock_show):
        ax = rf.estimate(history=(0.1, 0.2), best_epoch=1, units='nats').plot(show=False)
        assert 'nats' in ax.get_ylabel()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_rigorous_draws_every_repeat(self, mock_show):
        r = rf.rigorous(fits=[{'mi': 0.5, 'mi_error': 0.1, 'slope': -0.1, 'is_reliable': True},
                              {'mi': 0.6, 'mi_error': 0.1, 'slope': -0.1, 'is_reliable': True}])
        ax = r.plot(show=False)
        assert {'run 0', 'run 1'} <= {t.get_text() for t in ax.get_legend().get_texts()}
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_pairwise_returns_axes(self, mock_show):
        n = 4
        matrix = np.abs(np.random.default_rng(0).standard_normal((n, n)))
        np.fill_diagonal(matrix, 0.0)
        ax = rf.pairwise((matrix + matrix.T) / 2).plot(show=False)
        assert ax is not None
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_pairwise_missing_matrix_raises(self, mock_show):
        r = rf.make_results('pairwise', [{'config_id': 0, 'ch_x': 0, 'ch_y': 1, 'run_id': 0,
                                          'mi': 0.1}], axis_keys=('ch_x', 'ch_y'))
        with pytest.raises(ValueError, match="mi_matrix"):
            r.plot()

    # ------------------------------------------------------------------ #
    # save() / load() / to_json()                                         #
    # ------------------------------------------------------------------ #

    def test_save_creates_pkl_file(self, tmp_path):
        filepath = rf.estimate(mi=1.5).save(str(tmp_path))
        assert os.path.exists(filepath)
        assert filepath.endswith('.pkl')
        assert 'estimate' in filepath

    def test_save_no_overwrite(self, tmp_path):
        r = rf.estimate()
        path1 = r.save(str(tmp_path))
        path2 = r.save(path1)
        assert path2 != path1
        assert '_1' in path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)

    def test_load_roundtrip(self, tmp_path):
        r = rf.sweep(units='nats')
        r2 = Results.load(r.save(str(tmp_path)))
        assert r2.mode == r.mode
        assert r2.mi_estimate == r.mi_estimate
        assert list(r2.dataframe.columns) == list(r.dataframe.columns)
        assert len(r2.runs) == len(r.runs)

    def test_load_wrong_type_raises(self, tmp_path):
        import pickle
        bad_path = str(tmp_path / 'bad.pkl')
        with open(bad_path, 'wb') as f:
            pickle.dump({'not': 'a results'}, f)
        with pytest.raises(TypeError, match="Results"):
            Results.load(bad_path)

    def test_to_json_creates_json_file(self, tmp_path):
        filepath = rf.estimate(mi=1.23).to_json(str(tmp_path))
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        with open(filepath) as f:
            assert isinstance(json.load(f), dict)

    def test_to_json_contents(self, tmp_path):
        rows = [{'config_id': 0, 'lag': lag, 'run_id': 0, 'mi': v}
                for lag, v in ((0, 0.5), (1, 1.0))]
        r = rf.make_results('lag', rows, axis_keys=('lag',))
        with open(r.to_json(str(tmp_path))) as f:
            data = json.load(f)
        assert data['mode'] == 'lag'
        assert data['mi_estimate'] is None
        assert len(data['dataframe']) == 2
        assert len(data['runs']) == 2

    # ------------------------------------------------------------------ #
    # summary() for mode-specific content                                 #
    # ------------------------------------------------------------------ #

    def test_summary_precision_shows_baseline_mi(self, capsys):
        r = rf.precision(mis=(1.2, 1.1, 0.9), precision_tau=0.005, threshold_value=1.08)
        r.summary()
        captured = capsys.readouterr().out
        assert "precision" in captured.lower()
        assert "Baseline MI" in captured
        assert "1.2" in captured
        assert "Precision" in captured
        assert "0.005" in captured

    def test_precision_has_no_single_estimate(self):
        """One row per tau: the baseline is read from details, and mi_estimate is None."""
        r = rf.precision(mis=(1.234, 1.0, 0.8), precision_tau=0.007)
        assert r.mi_estimate is None
        assert r.get('baseline_mi') == 1.234
        assert r.get('precision_tau') == 0.007

    def test_summary_conditional_shows_components(self, capsys):
        rf.difference('conditional', {'mi_xw_y': 1.25, 'mi_w_y': 0.43}, 0.82).summary()
        captured = capsys.readouterr().out
        assert "conditional" in captured.lower()
        assert "0.8200" in captured
        assert "I(X,W;Y)" in captured
        assert "I(W;Y)" in captured

    def test_summary_transfer_shows_te(self, capsys):
        r = rf.difference('transfer', {'i_xypast_yfuture': 0.9, 'i_ypast_yfuture': 0.34,
                                       'te_yx': 0.12, 'directionality_index': 0.65}, 0.56)
        r.summary()
        captured = capsys.readouterr().out
        assert "transfer" in captured.lower()
        assert "TE(Y→X)" in captured
        assert "Directionality" in captured


class TestToDict:
    """Results.to_dict() and Results.to_json()."""

    def test_to_dict_returns_dict(self):
        assert isinstance(rf.estimate().to_dict(), dict)

    def test_to_dict_keys(self):
        assert set(rf.estimate().to_dict().keys()) == {
            'mode', 'mi_estimate', 'params', 'details', 'dataframe', 'runs'}

    def test_to_dict_arrays_as_nested_lists(self):
        r = rf.estimate(embeddings={'embeddings_x': np.array([[0.1, 0.2], [0.3, 0.4]])})
        d = r.to_dict()
        emb = d['details']['0']['embeddings']['0']['embeddings_x']
        assert isinstance(emb, list) and isinstance(emb[0], list)
        assert abs(emb[0][0] - 0.1) < 1e-6

    def test_to_dict_2d_array_as_nested_lists(self):
        d = rf.pairwise(np.eye(3)).to_dict()
        matrix = d['details']['0']['mi_matrix']
        assert isinstance(matrix, list)
        assert isinstance(matrix[0], list)

    def test_to_dict_training_history_included(self):
        d = rf.estimate(history=(0.1, 0.2, 0.3, 0.25)).to_dict()
        assert d['runs'][0]['test_mi_history'] == pytest.approx([0.1, 0.2, 0.3, 0.25])

    def test_to_dict_dataframe_as_records(self):
        d = rf.estimate(mi=0.5).to_dict()
        assert d['dataframe'][0]['mi_mean'] == 0.5
        assert d['dataframe'][0]['n_runs'] == 1

    def test_to_json_history_roundtrip(self, tmp_path):
        history = [0.1, 0.2, 0.35, 0.3]
        fp = rf.estimate(mi=0.35, history=history).to_json(str(tmp_path))
        with open(fp) as f:
            data = json.load(f)
        assert data['runs'][0]['test_mi_history'] == pytest.approx(history)


class TestRepeatsThatProducedNothing:
    """A repeat reported as 0 produced nothing. Averages use the other repeats."""

    @staticmethod
    def _aggregate(values):
        import warnings
        import pandas as pd
        from neural_mi.analysis.assemble import aggregate
        runs = pd.DataFrame({'config_id': [0] * len(values), 'run_id': range(len(values)),
                             'mi': values, 'test_mi': [v / 2 for v in values]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            frame = aggregate(runs, ['config_id'], mean_cols=['test_mi'])
        return frame.iloc[0], [str(w.message) for w in caught]

    def test_zeros_are_left_out_of_the_mean_and_spread(self):
        row, messages = self._aggregate([2.0, 2.2, 0.0, 1.8])
        assert row['mi_mean'] == pytest.approx(2.0) and row['n_zero'] == 1 and row['n_runs'] == 4
        assert row['mi_std'] == pytest.approx(0.2)
        assert row['test_mi_mean'] == pytest.approx(1.0)       # the same repeats
        assert len(messages) == 1 and messages[0].startswith("1 of 4 repeats produced nothing")

    def test_all_zeros_report_zero(self):
        row, messages = self._aggregate([0.0, 0.0, 0.0])
        assert row['mi_mean'] == 0.0 and row['n_zero'] == 3
        assert "no repeat that produced a value" in messages[0]

    def test_one_repeat_is_reported_as_it_is(self):
        row, messages = self._aggregate([0.0])
        assert row['mi_mean'] == 0.0 and messages == []

    def test_the_null_is_averaged_the_same_way(self):
        from neural_mi.analysis.permutation import row_values
        produced = {'rows': [{'config_id': 0, 'run_id': r, 'mi': v, 'raw_train_mi': v - 0.1}
                             for r, v in enumerate([0.3, 0.0, 0.5])],
                    'details': {}, 'axis_keys': []}
        mi, raw = row_values('estimate', produced)[(0,)]
        assert mi == pytest.approx(0.4) and raw == pytest.approx(0.1666667)
