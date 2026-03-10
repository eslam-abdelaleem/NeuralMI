# tests/test_results_extended.py
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
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
        """summary() shows half-CI and reliability flag for rigorous mode."""
        r = Results(
            mode='rigorous',
            mi_estimate=0.8,
            params={'output_units': 'nats'},
            details={'mi_error': 0.05, 'is_reliable': True},
        )
        r.summary()
        captured = capsys.readouterr().out
        assert "rigorous" in captured
        assert "0.8000" in captured
        assert "0.0500" in captured
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
