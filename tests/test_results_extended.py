# tests/test_results_extended.py
import json
import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from neural_mi.results import Results

class TestResults:
    def test_repr(self):
        r = Results(mode='estimate', mi_estimate=1.2345)
        assert "mi_estimate=1.2345" in repr(r)

        r = Results(mode='sweep', dataframe=pd.DataFrame({'a': [1, 2]}))
        assert "dataframe_shape=(2, 1)" in repr(r)

        r = Results(mode='rigorous', details={'key': 'val'})
        assert "details_keys=['key']" in repr(r)

    @patch('matplotlib.pyplot.show')
    def test_plot_raises_value_error_no_dataframe(self, mock_show):
        r = Results(mode='sweep')
        with pytest.raises(ValueError, match="Cannot plot: results do not contain a DataFrame"):
            r.plot()

    @patch('matplotlib.pyplot.show')
    def test_plot_raises_value_error_no_sweep_var(self, mock_show):
        # Dataframe has multiple potential sweep variables
        df = pd.DataFrame({'a': [1], 'b': [2], 'mi_mean': [0], 'mi_std': [0]})
        r = Results(mode='sweep', dataframe=df)
        with pytest.raises(ValueError, match="Cannot determine sweep variable"):
            r.plot()

    @patch('matplotlib.pyplot.show')
    @patch('neural_mi.visualize.plot.plot_sweep_curve')
    def test_plot_infers_sweep_var(self, mock_plot_sweep, mock_show):
        df = pd.DataFrame({'param': [1], 'mi_mean': [0], 'mi_std': [0]})
        r = Results(mode='sweep', dataframe=df)
        r.plot(show=False)
        mock_plot_sweep.assert_called_once()
        args, kwargs = mock_plot_sweep.call_args
        assert kwargs['param_col'] == 'param'

    @patch('matplotlib.pyplot.show')
    def test_plot_raises_not_implemented(self, mock_show):
        r = Results(mode='unknown_mode')
        with pytest.raises(NotImplementedError):
            r.plot()

    @patch('matplotlib.pyplot.show')
    def test_plot_rigorous_incomplete(self, mock_show):
        r = Results(mode='rigorous', dataframe=pd.DataFrame())
        # details missing
        with pytest.raises(ValueError, match="Rigorous results are incomplete"):
            r.plot()

    # ------------------------------------------------------------------ #
    # summary()                                                           #
    # ------------------------------------------------------------------ #

    def test_summary_estimate_mode(self, capsys):
        """summary() prints MI estimate and mode for 'estimate' mode."""
        r = Results(
            mode='estimate',
            mi_estimate=1.2345,
            params={'output_units': 'bits'},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "mode = 'estimate'" in captured
        assert "1.2345" in captured
        assert "bits" in captured

    def test_summary_no_mi_estimate(self, capsys):
        """summary() falls back gracefully when mi_estimate is None."""
        r = Results(mode='sweep')
        r.summary()
        captured = capsys.readouterr().out
        assert "mode = 'sweep'" in captured
        assert "none" in captured.lower()

    def test_summary_rigorous_reliable(self, capsys):
        """summary() shows CI half-width, PI half-width, and reliability for rigorous mode."""
        r = Results(
            mode='rigorous',
            mi_estimate=0.8,
            params={'output_units': 'nats'},
            details={'mi_error': 0.05, 'mi_error_pred': 0.09, 'is_reliable': True},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "rigorous" in captured
        assert "0.8000" in captured
        assert "0.0500" in captured   # CI half-width
        assert "0.0900" in captured   # PI half-width
        assert "is_reliable = True" in captured

    def test_summary_rigorous_unreliable(self, capsys):
        """summary() warns when is_reliable is False."""
        r = Results(
            mode='rigorous',
            mi_estimate=0.3,
            details={'mi_error': 0.2, 'is_reliable': False},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "is_reliable = False" in captured

    def test_summary_rigorous_reliable_shows_r_squared(self, capsys):
        """summary() shows R² when is_reliable=True and r_squared is a real float."""
        r = Results(
            mode='rigorous',
            mi_estimate=0.8,
            details={'mi_error': 0.05, 'is_reliable': True, 'r_squared': 0.987},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "R² = 0.987" in captured

    def test_summary_rigorous_reliable_hides_nan_r_squared(self, capsys):
        """summary() must not print a NaN r_squared value."""
        r = Results(
            mode='rigorous',
            mi_estimate=0.8,
            details={'mi_error': 0.05, 'is_reliable': True, 'r_squared': float('nan')},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "R²" not in captured

    def test_summary_rigorous_unreliable_shows_fit_quality_diagnostics(self, capsys):
        """The unreliable reason string surfaces r_squared but hides a NaN max_abs_residual."""
        r = Results(
            mode='rigorous',
            mi_estimate=0.3,
            details={
                'mi_error': 0.2,
                'is_reliable': False,
                'fit_quality_warning': True,
                'r_squared': 0.42,
                'max_abs_residual': float('nan'),
            },
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "R²=0.420" in captured
        assert "max|residual|" not in captured

    def test_summary_with_dataframe(self, capsys):
        """summary() prints DataFrame shape and column names."""
        df = pd.DataFrame({'tau': [1, 2, 3], 'mi_mean': [0.1, 0.2, 0.3]})
        r = Results(mode='sweep', dataframe=df)
        r.summary()
        captured = capsys.readouterr().out
        assert "3 rows" in captured
        assert "tau" in captured
        assert "mi_mean" in captured

    # ------------------------------------------------------------------ #
    # plot() — mode='estimate'                                            #
    # ------------------------------------------------------------------ #

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_returns_axes(self, mock_show):
        """plot() for mode='estimate' returns an Axes object without raising."""
        history = [0.1, 0.3, 0.5, 0.4, 0.45]
        r = Results(
            mode='estimate',
            mi_estimate=0.45,
            details={'test_mi_history': history, 'best_epoch': 2},
        )
        ax = r.plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_no_best_epoch(self, mock_show):
        """plot() for mode='estimate' handles missing best_epoch gracefully."""
        history = [0.1, 0.2, 0.3]
        r = Results(
            mode='estimate',
            mi_estimate=0.3,
            details={'test_mi_history': history},  # no best_epoch key
        )
        ax = r.plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_missing_history_raises(self, mock_show):
        """plot() for mode='estimate' raises ValueError when test_mi_history absent."""
        r = Results(
            mode='estimate',
            mi_estimate=0.5,
            details={'some_other_key': 42},
        )
        with pytest.raises(ValueError, match="test_mi_history"):
            r.plot()

    @patch('matplotlib.pyplot.show')
    def test_plot_estimate_uses_units_from_params(self, mock_show):
        """plot() for mode='estimate' labels y-axis with output_units from params."""
        history = [0.1, 0.2]
        r = Results(
            mode='estimate',
            mi_estimate=0.2,
            params={'output_units': 'nats'},
            details={'test_mi_history': history, 'best_epoch': 1},
        )
        ax = r.plot(show=False)
        assert 'nats' in ax.get_ylabel()
        plt.close('all')

    # ------------------------------------------------------------------ #
    # plot() — mode='pairwise'                                           #
    # ------------------------------------------------------------------ #

    @patch('matplotlib.pyplot.show')
    def test_plot_pairwise_returns_axes(self, mock_show):
        """plot() for mode='pairwise' renders a heatmap without raising."""
        import numpy as np
        n = 4
        mi_matrix = np.abs(np.random.randn(n, n))
        np.fill_diagonal(mi_matrix, 0.0)
        mi_matrix = (mi_matrix + mi_matrix.T) / 2  # symmetric
        df = pd.DataFrame({'ch_x': [0], 'ch_y': [1], 'mi_estimate': [0.5]})
        r = Results(
            mode='pairwise',
            params={'output_units': 'bits'},
            dataframe=df,
            details={'mi_matrix': mi_matrix},
        )
        ax = r.plot(show=False)
        assert ax is not None
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_pairwise_missing_matrix_raises(self, mock_show):
        """plot() for mode='pairwise' raises ValueError when mi_matrix is missing."""
        r = Results(mode='pairwise', details={})
        with pytest.raises(ValueError, match="mi_matrix"):
            r.plot()
    # save() / load() / to_json()                                         #
    # ------------------------------------------------------------------ #

    def test_save_creates_pkl_file(self, tmp_path):
        """save() creates a .pkl file in the specified directory."""
        r = Results(mode='estimate', mi_estimate=1.5, params={'output_units': 'bits'})
        filepath = r.save(str(tmp_path))
        assert os.path.exists(filepath)
        assert filepath.endswith('.pkl')
        assert 'estimate' in filepath

    def test_save_no_overwrite(self, tmp_path):
        """save() appends a numeric suffix rather than overwriting an existing file."""
        r = Results(mode='estimate', mi_estimate=0.5)
        path1 = r.save(str(tmp_path))
        # Force the second save to the same exact filename (simulate clash)
        path2 = r.save(path1)
        assert path2 != path1
        assert '_1' in path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)

    def test_load_roundtrip(self, tmp_path):
        """load() restores a Results object saved with save()."""
        r = Results(
            mode='sweep',
            mi_estimate=2.0,
            params={'output_units': 'nats'},
            dataframe=pd.DataFrame({'mi_mean': [1.0, 2.0]}),
        )
        filepath = r.save(str(tmp_path))
        r2 = Results.load(filepath)
        assert r2.mode == r.mode
        assert r2.mi_estimate == r.mi_estimate
        assert list(r2.dataframe.columns) == list(r.dataframe.columns)

    def test_load_wrong_type_raises(self, tmp_path):
        """load() raises TypeError when the file does not contain a Results object."""
        import pickle
        bad_path = str(tmp_path / 'bad.pkl')
        with open(bad_path, 'wb') as f:
            pickle.dump({'not': 'a results'}, f)
        with pytest.raises(TypeError, match="Results"):
            Results.load(bad_path)

    def test_to_json_creates_json_file(self, tmp_path):
        """to_json() creates a valid .json file."""
        r = Results(mode='estimate', mi_estimate=1.23, params={'output_units': 'bits'})
        filepath = r.to_json(str(tmp_path))
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_to_json_contents(self, tmp_path):
        """to_json() output contains mode, mi_estimate, and dataframe keys."""
        df = pd.DataFrame({'mi_mean': [0.5, 1.0], 'mi_std': [0.1, 0.2]})
        r = Results(mode='lag', mi_estimate=0.75, dataframe=df,
                    params={'output_units': 'bits'})
        filepath = r.to_json(str(tmp_path))
        with open(filepath) as f:
            data = json.load(f)
        assert data['mode'] == 'lag'
        assert abs(data['mi_estimate'] - 0.75) < 1e-9
        assert data['dataframe'] is not None
        assert len(data['dataframe']) == 2

    # ------------------------------------------------------------------ #
    # summary() — precision and mode-specific sections                    #
    # ------------------------------------------------------------------ #

    def test_summary_precision_shows_baseline_mi(self, capsys):
        """summary() for precision mode prints Baseline MI and Precision tau."""
        r = Results(
            mode='precision',
            mi_estimate=1.20,
            params={'output_units': 'bits'},
            details={
                'baseline_mi': 1.20,
                'precision_tau': 0.005,
                'threshold_value': 1.08,
            },
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "precision" in captured.lower()
        assert "Baseline MI" in captured
        assert "1.2000" in captured
        assert "Precision" in captured
        assert "0.005" in captured

    def test_precision_mi_estimate_is_baseline(self):
        """mi_estimate for precision results should equal baseline_mi, not precision_tau."""
        baseline = 1.234
        tau = 0.007
        r = Results(
            mode='precision',
            mi_estimate=baseline,
            details={'baseline_mi': baseline, 'precision_tau': tau},
        )
        assert r.mi_estimate == baseline
        assert r.mi_estimate != tau

    def test_summary_conditional_shows_components(self, capsys):
        """summary() for conditional mode shows CMI and component MI values."""
        r = Results(
            mode='conditional',
            mi_estimate=0.82,
            params={'output_units': 'bits'},
            details={'cmi_estimate': 0.82, 'mi_xz_y': 1.25, 'mi_z_y': 0.43},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "conditional" in captured.lower()
        assert "CMI" in captured
        assert "I(XZ;Y)" in captured
        assert "I(Z;Y)" in captured

    def test_summary_transfer_shows_te(self, capsys):
        """summary() for transfer mode shows TE(X→Y) and directionality when present."""
        r = Results(
            mode='transfer',
            mi_estimate=0.56,
            params={'output_units': 'bits'},
            details={
                'te_xy': 0.56, 'te_yx': 0.12,
                'directionality_index': 0.65,
            },
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "transfer" in captured.lower()
        assert "TE(X" in captured
        assert "Directionality" in captured


class TestToDict:
    """Tests for Results.to_dict() and the updated Results.to_json()."""

    def test_to_dict_returns_dict(self):
        """to_dict() always returns a plain dict."""
        r = Results(mode='estimate', mi_estimate=1.0)
        assert isinstance(r.to_dict(), dict)

    def test_to_dict_keys(self):
        """to_dict() has exactly the expected top-level keys."""
        r = Results(mode='estimate', mi_estimate=1.0)
        assert set(r.to_dict().keys()) == {'mode', 'mi_estimate', 'params', 'details', 'dataframe'}

    def test_to_dict_arrays_as_nested_lists(self):
        """numpy arrays must be nested lists, not shape-summary strings."""
        import numpy as np
        arr = np.array([0.1, 0.2, 0.3])
        r = Results(mode='estimate', mi_estimate=0.5, details={'history': arr})
        d = r.to_dict()
        assert isinstance(d['details']['history'], list)
        assert len(d['details']['history']) == 3
        assert abs(d['details']['history'][0] - 0.1) < 1e-6

    def test_to_dict_2d_array_as_nested_lists(self):
        """2-D numpy arrays become nested lists, not shape strings."""
        import numpy as np
        mat = np.eye(3)
        r = Results(mode='pairwise', details={'mi_matrix': mat})
        d = r.to_dict()
        assert isinstance(d['details']['mi_matrix'], list)
        assert isinstance(d['details']['mi_matrix'][0], list)

    def test_to_dict_training_history_included(self):
        """test_mi_history and train_mi_history appear in full in to_dict output."""
        r = Results(
            mode='estimate', mi_estimate=0.5,
            details={
                'test_mi_history': [0.1, 0.2, 0.3, 0.25],
                'train_mi_history': [0.15, 0.25, 0.35, 0.30],
            }
        )
        d = r.to_dict()
        assert 'test_mi_history' in d['details']
        assert len(d['details']['test_mi_history']) == 4
        assert 'train_mi_history' in d['details']
        assert len(d['details']['train_mi_history']) == 4

    def test_to_dict_dataframe_as_records(self):
        """DataFrames in details are converted to list-of-dicts."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        r = Results(mode='sweep', dataframe=df)
        d = r.to_dict()
        assert d['dataframe'] == [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]

    def test_to_dict_none_dataframe(self):
        """to_dict() returns None for dataframe when not set."""
        r = Results(mode='estimate', mi_estimate=0.5)
        assert r.to_dict()['dataframe'] is None

    def test_to_json_arrays_are_lists(self, tmp_path):
        """to_json() must serialise arrays as lists, not shape-summary strings."""
        import numpy as np
        r = Results(mode='estimate', mi_estimate=0.5,
                    details={'arr': np.array([1.0, 2.0, 3.0])})
        fp = r.to_json(str(tmp_path))
        with open(fp) as f:
            data = json.load(f)
        assert isinstance(data['details']['arr'], list)
        assert data['details']['arr'] == pytest.approx([1.0, 2.0, 3.0])

    def test_to_json_history_roundtrip(self, tmp_path):
        """test_mi_history survives a to_json → json.load roundtrip intact."""
        history = [0.1, 0.2, 0.35, 0.3]
        r = Results(mode='estimate', mi_estimate=0.35,
                    details={'test_mi_history': history})
        fp = r.to_json(str(tmp_path))
        with open(fp) as f:
            data = json.load(f)
        assert data['details']['test_mi_history'] == pytest.approx(history)
