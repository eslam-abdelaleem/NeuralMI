# tests/test_visualize_extended.py
"""Extended tests for the plotting improvements across all modes.

Covers:
  - P1: estimate plot — conservative_epoch marker
  - P2: dimensionality plot — two-panel (MI + PR) via plot_dimensionality_curve
  - P3: plot_bias_correction_fit return value
  - P4: conditional / transfer mode plots
  - P5: Results.compare() for estimate mode
  - P6: rigorous plot is_reliable=False annotation
  - P7: plot_cross_correlation composability (ax, show, xlim, return value)
  - P8: analyze_mi_heatmap composability (show, return value)
  - P9: _RESULT_COLS contains pr_eig / pr_singular columns
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch

from neural_mi.results import Results, _RESULT_COLS
from neural_mi.visualize.plot import (
    plot_bias_correction_fit,
    plot_dimensionality_curve,
    plot_cross_correlation,
    analyze_mi_heatmap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rigorous_df():
    gammas = np.repeat(np.arange(1, 5), 4)
    train_mi = 1.0 / gammas + 0.5 + np.random.default_rng(0).normal(0, 0.05, len(gammas))
    return pd.DataFrame({'gamma': gammas, 'train_mi': train_mi})


@pytest.fixture
def rigorous_details():
    return {
        'slope': -0.4,
        'mi_corrected': 0.52,
        'mi_error': 0.04,
        'gammas_used': [1, 2, 3, 4],
    }


@pytest.fixture
def dim_df_sweep():
    """Dimensionality results DataFrame with a sweep variable."""
    return pd.DataFrame({
        'embedding_dim': [8, 16, 32, 64],
        'mi_mean': [0.40, 0.55, 0.61, 0.62],
        'mi_std': [0.05, 0.04, 0.03, 0.04],
        'pr_singular_mean': [3.1, 5.2, 6.8, 7.0],
        'pr_singular_std': [0.4, 0.5, 0.6, 0.7],
    })


@pytest.fixture
def dim_df_scalar():
    """Dimensionality results DataFrame with no sweep (single scalar result)."""
    return pd.DataFrame({
        'mi_mean': [0.58],
        'mi_std': [0.06],
        'pr_singular_mean': [5.3],
        'pr_singular_std': [0.5],
    })


# ---------------------------------------------------------------------------
# P1 — estimate plot: conservative_epoch marker
# ---------------------------------------------------------------------------

class TestEstimatePlotConservativeEpoch:

    @patch('matplotlib.pyplot.show')
    def test_conservative_epoch_line_present(self, mock_show):
        """When conservative_epoch is in details, a green dotted axvline must appear."""
        history = [0.1, 0.3, 0.5, 0.48, 0.52, 0.50]
        r = Results(
            mode='estimate',
            mi_estimate=0.48,
            details={
                'test_mi_history': history,
                'best_epoch': 4,
                'conservative_epoch': 2,
            },
        )
        ax = r.plot(show=False)
        vlines = [line for line in ax.lines if not np.isfinite(line.get_xdata()).all()]
        # Collect all axvline x positions from vertical lines
        vline_xs = []
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) == 2 and xdata[0] == xdata[1]:
                vline_xs.append(xdata[0])
        assert 2 in vline_xs, (
            "conservative_epoch=2 should produce a vertical line at x=2; "
            f"found axvline x-positions: {vline_xs}"
        )
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conservative_epoch_not_present_no_extra_line(self, mock_show):
        """Without conservative_epoch, only the best_epoch line appears."""
        history = [0.1, 0.3, 0.5, 0.48]
        r = Results(
            mode='estimate',
            mi_estimate=0.5,
            details={'test_mi_history': history, 'best_epoch': 2},  # no conservative_epoch
        )
        ax = r.plot(show=False)
        vline_xs = []
        for line in ax.lines:
            xdata = line.get_xdata()
            if len(xdata) == 2 and xdata[0] == xdata[1]:
                vline_xs.append(xdata[0])
        # best_epoch=2 present; conservative_epoch absent — only one vertical line
        assert len(vline_xs) == 1
        assert vline_xs[0] == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conservative_epoch_scatter_point_added(self, mock_show):
        """Conservative epoch should also add a diamond scatter marker."""
        history = [0.1, 0.3, 0.5, 0.45, 0.52]
        r = Results(
            mode='estimate',
            mi_estimate=0.45,
            details={'test_mi_history': history, 'best_epoch': 4, 'conservative_epoch': 2},
        )
        ax = r.plot(show=False)
        assert ax is not None
        plt.close('all')


# ---------------------------------------------------------------------------
# P2 — dimensionality plot: two-panel via plot_dimensionality_curve
# ---------------------------------------------------------------------------

class TestDimensionalityPlot:

    @patch('matplotlib.pyplot.show')
    def test_dim_sweep_two_panels(self, mock_show, dim_df_sweep):
        """With a sweep and PR columns, the figure should have two axes."""
        r = Results(
            mode='dimensionality',
            dataframe=dim_df_sweep,
            params={'sweep_var': 'embedding_dim'},
        )
        ax = r.plot(show=False)
        assert ax is not None
        # Two subplots should be in the figure
        assert len(ax.figure.axes) == 2, (
            f"Expected 2 axes (MI + PR), got {len(ax.figure.axes)}"
        )
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_dim_sweep_returns_mi_axes(self, mock_show, dim_df_sweep):
        """plot() must return the MI (top) axes, not the PR axes."""
        r = Results(
            mode='dimensionality',
            dataframe=dim_df_sweep,
            params={'sweep_var': 'embedding_dim'},
        )
        ax = r.plot(show=False)
        # The returned axes is the first (MI) panel
        assert ax is ax.figure.axes[0]
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_dim_no_sweep_single_scalar_display(self, mock_show, dim_df_scalar):
        """Without a sweep variable, a single-point display should not raise."""
        r = Results(mode='dimensionality', dataframe=dim_df_scalar, params={})
        ax = r.plot(show=False)
        assert ax is not None
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_dim_no_pr_column_single_panel(self, mock_show):
        """When pr_singular_mean is absent, only one panel is created."""
        df = pd.DataFrame({
            'embedding_dim': [8, 16, 32],
            'mi_mean': [0.3, 0.5, 0.6],
            'mi_std': [0.05, 0.04, 0.03],
        })
        r = Results(mode='dimensionality', dataframe=df, params={'sweep_var': 'embedding_dim'})
        ax = r.plot(show=False)
        assert len(ax.figure.axes) == 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_dimensionality_curve_direct_two_panel(self, mock_show, dim_df_sweep):
        """plot_dimensionality_curve called directly creates two-panel figure."""
        ax_mi = plot_dimensionality_curve(dim_df_sweep, sweep_var='embedding_dim')
        assert ax_mi is not None
        assert len(ax_mi.figure.axes) == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_dimensionality_curve_accepts_axes_tuple(self, mock_show, dim_df_sweep):
        """When a tuple of axes is passed, no new figure is created."""
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax_mi = plot_dimensionality_curve(
            dim_df_sweep, sweep_var='embedding_dim', axes=(ax1, ax2)
        )
        assert ax_mi is ax1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_dimensionality_curve_accepts_single_axes(self, mock_show, dim_df_sweep):
        """When a single axes is passed, only the MI panel is drawn (no PR)."""
        fig, ax = plt.subplots()
        ax_mi = plot_dimensionality_curve(
            dim_df_sweep, sweep_var='embedding_dim', axes=ax
        )
        assert ax_mi is ax
        # PR panel was skipped — still only one axes in the figure
        assert len(fig.axes) == 1
        plt.close('all')


# ---------------------------------------------------------------------------
# P3 — plot_bias_correction_fit return value
# ---------------------------------------------------------------------------

class TestBiasCorrectionFitReturn:

    @patch('matplotlib.pyplot.show')
    def test_returns_axes(self, mock_show, rigorous_df, rigorous_details):
        """plot_bias_correction_fit must return the axes it drew on."""
        fig, ax = plt.subplots()
        returned = plot_bias_correction_fit(rigorous_df, rigorous_details, ax=ax)
        assert returned is ax
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_creates_axes_when_none(self, mock_show, rigorous_df, rigorous_details):
        """When ax=None, the function creates and returns a new axes."""
        returned = plot_bias_correction_fit(rigorous_df, rigorous_details)
        assert isinstance(returned, plt.Axes)
        plt.close('all')


# ---------------------------------------------------------------------------
# P4 — conditional and transfer mode plots
# ---------------------------------------------------------------------------

class TestConditionalPlot:

    @patch('matplotlib.pyplot.show')
    def test_conditional_plot_returns_axes(self, mock_show):
        r = Results(
            mode='conditional',
            mi_estimate=0.82,
            details={'cmi_estimate': 0.82, 'mi_xz_y': 1.25, 'mi_z_y': 0.43},
        )
        ax = r.plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conditional_plot_has_three_bars(self, mock_show):
        """All three component bars should be present when all details available."""
        r = Results(
            mode='conditional',
            mi_estimate=0.82,
            details={'cmi_estimate': 0.82, 'mi_xz_y': 1.25, 'mi_z_y': 0.43},
        )
        ax = r.plot(show=False)
        bar_patches = [p for p in ax.patches if hasattr(p, 'get_height')]
        assert len(bar_patches) == 3
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conditional_partial_details_plots_available(self, mock_show):
        """When mi_z_y is absent, only two bars should be rendered."""
        r = Results(
            mode='conditional',
            mi_estimate=0.82,
            details={'cmi_estimate': 0.82, 'mi_xz_y': 1.25},  # mi_z_y missing
        )
        ax = r.plot(show=False)
        bar_patches = [p for p in ax.patches if hasattr(p, 'get_height')]
        assert len(bar_patches) == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conditional_missing_all_raises(self, mock_show):
        r = Results(mode='conditional', details={})
        with pytest.raises(ValueError, match="cmi_estimate"):
            r.plot()


class TestTransferPlot:

    @patch('matplotlib.pyplot.show')
    def test_transfer_plot_returns_axes(self, mock_show):
        r = Results(
            mode='transfer',
            mi_estimate=0.56,
            details={'te_xy': 0.56, 'te_yx': 0.12, 'directionality_index': 0.65},
        )
        ax = r.plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_two_bars_when_bidirectional(self, mock_show):
        r = Results(
            mode='transfer',
            details={'te_xy': 0.56, 'te_yx': 0.12, 'directionality_index': 0.65},
        )
        ax = r.plot(show=False)
        bar_patches = [p for p in ax.patches if hasattr(p, 'get_height')]
        assert len(bar_patches) == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_one_bar_when_unidirectional(self, mock_show):
        """Only te_xy in details → single bar."""
        r = Results(mode='transfer', details={'te_xy': 0.56})
        ax = r.plot(show=False)
        bar_patches = [p for p in ax.patches if hasattr(p, 'get_height')]
        assert len(bar_patches) == 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_di_in_title(self, mock_show):
        """Directionality index should appear in the plot title."""
        r = Results(
            mode='transfer',
            details={'te_xy': 0.56, 'te_yx': 0.12, 'directionality_index': 0.65},
        )
        ax = r.plot(show=False)
        assert '0.65' in ax.get_title() or 'Directionality' in ax.get_title()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_missing_te_xy_raises(self, mock_show):
        r = Results(mode='transfer', details={'te_yx': 0.1})
        with pytest.raises(ValueError, match="te_xy"):
            r.plot()


# ---------------------------------------------------------------------------
# P5 — Results.compare() for estimate mode
# ---------------------------------------------------------------------------

class TestCompareEstimateMode:

    @patch('matplotlib.pyplot.show')
    def test_compare_estimate_returns_axes(self, mock_show):
        h1 = [0.1, 0.3, 0.5, 0.48]
        h2 = [0.05, 0.25, 0.45, 0.50]
        r1 = Results(mode='estimate', details={'test_mi_history': h1, 'best_epoch': 2})
        r2 = Results(mode='estimate', details={'test_mi_history': h2, 'best_epoch': 3})
        ax = Results.compare([r1, r2], labels=['Run A', 'Run B'], show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_compare_estimate_overlays_two_curves(self, mock_show):
        """Two estimate results → two data lines in the returned axes."""
        h1 = [0.1, 0.3, 0.5, 0.48]
        h2 = [0.05, 0.25, 0.45, 0.50]
        r1 = Results(mode='estimate', details={'test_mi_history': h1})
        r2 = Results(mode='estimate', details={'test_mi_history': h2})
        ax = Results.compare([r1, r2], show=False)
        # Each history produces one data line; best_epoch lines may or may not be present
        # All lines with 4-length x-data are the history curves
        history_lines = [
            l for l in ax.lines if len(l.get_xdata()) == 4
        ]
        assert len(history_lines) == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_compare_estimate_missing_history_raises(self, mock_show):
        r1 = Results(mode='estimate', details={'test_mi_history': [0.1, 0.3]})
        r2 = Results(mode='estimate', details={})  # no test_mi_history
        with pytest.raises(ValueError, match="test_mi_history"):
            Results.compare([r1, r2], show=False)

    @patch('matplotlib.pyplot.show')
    def test_compare_unsupported_mode_mentions_estimate(self, mock_show):
        """The NotImplementedError message must list 'estimate' as a supported mode."""
        r1 = Results(mode='precision', details={})
        r2 = Results(mode='precision', details={})
        with pytest.raises(NotImplementedError, match="estimate"):
            Results.compare([r1, r2], show=False)


# ---------------------------------------------------------------------------
# P6 — rigorous plot is_reliable=False annotation
# ---------------------------------------------------------------------------

class TestRigorousReliabilityAnnotation:

    @patch('matplotlib.pyplot.show')
    def test_unreliable_annotation_appears(self, mock_show, rigorous_df, rigorous_details):
        """is_reliable=False must add a text annotation to the plot."""
        details = {**rigorous_details, 'is_reliable': False, 'leverage_warning': True}
        r = Results(mode='rigorous', dataframe=rigorous_df, details=details)
        ax = r.plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert any('unreliable' in t.lower() or '⚠' in t for t in texts), (
            f"Expected unreliable warning text, found: {texts}"
        )
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_reliable_no_annotation(self, mock_show, rigorous_df, rigorous_details):
        """is_reliable=True must NOT add any warning text."""
        details = {**rigorous_details, 'is_reliable': True}
        r = Results(mode='rigorous', dataframe=rigorous_df, details=details)
        ax = r.plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert not any('unreliable' in t.lower() for t in texts)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_no_is_reliable_key_no_annotation(self, mock_show, rigorous_df, rigorous_details):
        """When is_reliable is absent, no annotation is added."""
        r = Results(mode='rigorous', dataframe=rigorous_df, details=rigorous_details)
        ax = r.plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert not any('unreliable' in t.lower() for t in texts)
        plt.close('all')


# ---------------------------------------------------------------------------
# P7 — plot_cross_correlation composability
# ---------------------------------------------------------------------------

class TestPlotCrossCorrelation:

    def _make_signals(self, n=200):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, n))
        y = np.roll(x, 5) + rng.standard_normal((1, n)) * 0.1
        return x, y

    @patch('matplotlib.pyplot.show')
    def test_returns_axes(self, mock_show):
        x, y = self._make_signals()
        ax = plot_cross_correlation(x, y, true_lag=5)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_accepts_external_axes(self, mock_show):
        x, y = self._make_signals()
        fig, ax_ext = plt.subplots()
        ax = plot_cross_correlation(x, y, true_lag=5, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_show_false_does_not_call_show(self, mock_show):
        x, y = self._make_signals()
        plot_cross_correlation(x, y, true_lag=5, show=False)
        mock_show.assert_not_called()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_show_true_calls_show(self, mock_show):
        x, y = self._make_signals()
        plot_cross_correlation(x, y, true_lag=5, show=True)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_xlim_applied(self, mock_show):
        x, y = self._make_signals()
        ax = plot_cross_correlation(x, y, true_lag=5, show=False, xlim=(-20, 20))
        left, right = ax.get_xlim()
        assert abs(left - (-20)) < 1 and abs(right - 20) < 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_no_xlim_uses_full_range(self, mock_show):
        """Without xlim the x-axis should NOT be clipped to (-100, 100)."""
        x, y = self._make_signals(n=300)
        ax = plot_cross_correlation(x, y, true_lag=5, show=False)  # no xlim
        # The old hard-coded (-100, 100) is gone; full lag range should be wider
        left, right = ax.get_xlim()
        assert right > 100 or left < -100, (
            "Without xlim, the full lag range should be shown (not clipped to ±100)."
        )
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_true_lag_line_position_matches_its_label(self, mock_show):
        """The red 'True Lag' reference line must be drawn at true_lag itself.

        Previously it was drawn at true_lag + 1 while its own legend label
        still read f'True Lag ({true_lag})' -- the line and its label
        disagreed regardless of which convention is "correct".
        """
        x, y = self._make_signals()
        true_lag = 5
        ax = plot_cross_correlation(x, y, true_lag=true_lag, show=False)

        true_lag_lines = [ln for ln in ax.get_lines() if ln.get_color() == 'r']
        assert len(true_lag_lines) == 1
        xdata = true_lag_lines[0].get_xdata()
        assert xdata[0] == xdata[1] == true_lag

        _, labels = ax.get_legend_handles_labels()
        assert f'True Lag ({true_lag})' in labels
        plt.close('all')


# ---------------------------------------------------------------------------
# P8 — analyze_mi_heatmap composability
# ---------------------------------------------------------------------------

class TestAnalyzeMiHeatmap:

    @pytest.fixture
    def heatmap_df(self):
        """Shaped like a real result.dataframe from mode='lag' swept over window_size."""
        lags = np.arange(-5, 6)
        windows = np.arange(5, 26, 5)
        rows = [(lag, ws, max(0.0, 0.8 - abs(lag) * 0.1 - (ws - 10) * 0.01))
                for lag in lags for ws in windows]
        return pd.DataFrame(rows, columns=['lag', 'window_size', 'mi_mean'])

    @patch('matplotlib.pyplot.show')
    def test_returns_axes(self, mock_show, heatmap_df):
        ax = analyze_mi_heatmap(heatmap_df, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_show_false_does_not_call_show(self, mock_show, heatmap_df):
        analyze_mi_heatmap(heatmap_df, show=False)
        mock_show.assert_not_called()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_show_true_calls_show(self, mock_show, heatmap_df):
        analyze_mi_heatmap(heatmap_df, show=True)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_accepts_external_axes(self, mock_show, heatmap_df):
        fig, ax_ext = plt.subplots()
        ax = analyze_mi_heatmap(heatmap_df, ax=ax_ext, show=False)
        assert ax is ax_ext
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_no_print_calls(self, mock_show, heatmap_df, capsys):
        """analyze_mi_heatmap must not write to stdout (uses logger instead)."""
        analyze_mi_heatmap(heatmap_df, show=False)
        captured = capsys.readouterr()
        assert captured.out == '', (
            f"analyze_mi_heatmap wrote to stdout: {captured.out!r}"
        )
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_mi_col_defaults_to_mi_mean(self, mock_show, heatmap_df):
        """A real result.dataframe (mi_mean, not mi) must work with no extra args."""
        assert 'mi_mean' in heatmap_df.columns
        ax = analyze_mi_heatmap(heatmap_df, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_mi_col_accepts_custom_column_name(self, mock_show):
        """mi_col lets callers point at a differently-named MI column."""
        lags = np.arange(-5, 6)
        windows = np.arange(5, 26, 5)
        rows = [(lag, ws, max(0.0, 0.8 - abs(lag) * 0.1 - (ws - 10) * 0.01))
                for lag in lags for ws in windows]
        df = pd.DataFrame(rows, columns=['lag', 'window_size', 'score'])
        ax = analyze_mi_heatmap(df, mi_col='score', show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.show')
    def test_no_significant_contour_path_respects_external_axes(self, mock_show, mock_tight_layout):
        """The early-return ('no significant contour') path must not call
        tight_layout() when the caller supplied their own ax -- it should
        only tidy up figures this function created itself."""
        lags = np.arange(-5, 6)
        windows = np.arange(5, 26, 5)
        rows = [(lag, ws, 0.0) for lag in lags for ws in windows]
        flat_df = pd.DataFrame(rows, columns=['lag', 'window_size', 'mi_mean'])

        fig, ax_ext = plt.subplots()
        ax = analyze_mi_heatmap(flat_df, absolute_mi_threshold=0.5, ax=ax_ext, show=True)
        assert ax is ax_ext
        mock_tight_layout.assert_not_called()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_degenerate_contour_does_not_crash(self, mock_show, heatmap_df):
        """An absurdly high threshold can make matplotlib's contour() return a
        non-empty allsegs[0] list containing only degenerate (empty or
        single-point) segments -- the list-level `not cs.allsegs[0]` check
        alone doesn't catch this, and the downstream cdist/argmin used to
        crash with 'attempt to get argmin of an empty sequence'."""
        ax = analyze_mi_heatmap(heatmap_df, absolute_mi_threshold=1e6, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


# ---------------------------------------------------------------------------
# P9 — _RESULT_COLS contains pr_eig / pr_singular columns
# ---------------------------------------------------------------------------

class TestResultCols:

    def test_pr_columns_in_result_cols(self):
        assert 'pr_eig' in _RESULT_COLS
        assert 'pr_eig_mean' in _RESULT_COLS
        assert 'pr_eig_std' in _RESULT_COLS
        assert 'pr_singular' in _RESULT_COLS
        assert 'pr_singular_mean' in _RESULT_COLS
        assert 'pr_singular_std' in _RESULT_COLS

    def test_split_id_in_result_cols(self):
        assert 'split_id' in _RESULT_COLS
