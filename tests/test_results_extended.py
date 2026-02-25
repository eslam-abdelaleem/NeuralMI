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
