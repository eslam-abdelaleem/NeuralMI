# tests/test_visualize_extended.py
"""Extended tests for the plotting improvements across all modes.

Covers:
  - estimate plot — conservative_epoch marker
  - dimensionality plot — per-rank stable/degenerate/below-floor chart via plot_dimensionality_curve
  - plot_bias_correction_fit return value
  - conditional / transfer mode plots
  - Results.compare() for estimate mode
  - rigorous plot is_reliable=False annotation
  - plot_cross_correlation composability (ax, show, xlim, return value)
  - analyze_mi_heatmap composability (show, return value)
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch

from neural_mi.results import Results
from tests import results_factory as rf
from neural_mi.visualize.plot import (
    plot_bias_correction_fit,
    plot_dimensionality_curve,
    plot_cross_correlation,
    analyze_mi_heatmap,
    plot_sweep_heatmap,
    plot_sweep_bar,
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


def _rigorous_result(fit=None, n_repeats=1):
    """A rigorous result with one ladder and fit per repeat."""
    base = {'mi': 0.52, 'slope': -0.4, 'mi_error': 0.04, 'gammas_used': [1, 2, 3, 4]}
    return rf.rigorous(fits=[{**base, **(fit or {})} for _ in range(n_repeats)],
                       gammas=range(1, 5))


@pytest.fixture
def dim_details():
    """result.details from a mode='dimensionality' run: 2 individually-stable
    ranks, one degenerate pair, one below the noise floor."""
    return {
        'stability_per_rank': {
            1: {'mean_strength': 3.0, 'min_abs_corr': 0.95, 'stable': True, 'below_noise_floor': False},
            2: {'mean_strength': 1.2, 'min_abs_corr': 0.90, 'stable': True, 'below_noise_floor': False},
            3: {'mean_strength': 1.1, 'min_abs_corr': 0.88, 'stable': True, 'below_noise_floor': False},
            4: {'mean_strength': 0.001, 'min_abs_corr': 0.20, 'stable': False, 'below_noise_floor': True},
        },
        'stable_directions': [1],
        'stable_but_degenerate_groups': [[2, 3]],
        'n_stable_total': 3,
        'converged': True,
    }


# ---------------------------------------------------------------------------
# estimate plot: conservative_epoch marker
# ---------------------------------------------------------------------------

class TestEstimatePlotConservativeEpoch:

    @staticmethod
    def _vline_xs(ax):
        return [line.get_xdata()[0] for line in ax.lines
                if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]

    @patch('matplotlib.pyplot.show')
    def test_conservative_epoch_line_present(self, mock_show):
        """When the repeat records a conservative_epoch, a vertical line marks it."""
        r = rf.estimate(mi=0.48, history=(0.1, 0.3, 0.5, 0.48, 0.52, 0.50), best_epoch=4,
                        conservative_epoch=2)
        ax = r.plot(show=False)
        assert 2 in self._vline_xs(ax), self._vline_xs(ax)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conservative_epoch_not_present_no_extra_line(self, mock_show):
        """Without conservative_epoch, only the best_epoch line appears."""
        ax = rf.estimate(mi=0.5, history=(0.1, 0.3, 0.5, 0.48), best_epoch=2).plot(show=False)
        assert self._vline_xs(ax) == [2]
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_several_repeats_draw_one_curve_each(self, mock_show):
        rows = [{'config_id': 0, 'run_id': rid, 'mi': 0.4, 'test_mi_history': [0.1, 0.3, 0.4]}
                for rid in range(3)]
        ax = rf.make_results('sweep', rows).plot(show=False)
        labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert {'run 0', 'run 1', 'run 2'} <= labels
        plt.close('all')


# ---------------------------------------------------------------------------
# dimensionality plot: per-rank stable/degenerate/below-floor chart via
# plot_dimensionality_curve
# ---------------------------------------------------------------------------

class TestDimensionalityPlot:

    @patch('matplotlib.pyplot.show')
    def test_dispatch_via_results_plot(self, mock_show, dim_details):
        """Results.plot() for mode='dimensionality' draws the per-rank chart
        from the configuration's details."""
        rows = [{'config_id': 0, 'split_id': s, 'mi': 0.3, 'pr_eig': 1.0} for s in range(2)]
        r = rf.make_results('dimensionality', rows, details={0: dim_details})
        ax = r.plot(show=False)
        assert ax is not None
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_missing_stability_per_rank_raises_clear_error(self, mock_show):
        """Calling the plot before at least 2 splits produced usable stability
        data should raise a clear error, not a KeyError deep inside plotting."""
        with pytest.raises(ValueError, match="stability_per_rank"):
            plot_dimensionality_curve({})

    @patch('matplotlib.pyplot.show')
    def test_bars_colored_by_status(self, mock_show, dim_details):
        """One bar per rank; individually-stable, degenerate-group, and
        below-floor ranks get visually distinct colors."""
        ax = plot_dimensionality_curve(dim_details)
        bars = [p for p in ax.patches]
        assert len(bars) == 4  # one per rank in stability_per_rank
        colors = {tuple(b.get_facecolor()) for b in bars}
        # Stable, degenerate-group, and below-floor should not all share one color.
        assert len(colors) >= 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_accepts_external_axes(self, mock_show, dim_details):
        """When an axes is passed, no new figure is created."""
        fig, ax_in = plt.subplots()
        ax_out = plot_dimensionality_curve(dim_details, ax=ax_in)
        assert ax_out is ax_in
        plt.close('all')


# ---------------------------------------------------------------------------
# plot_bias_correction_fit return value
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

    @patch('matplotlib.pyplot.show')
    def test_default_appearance_unchanged_without_label_or_color(
        self, mock_show, rigorous_df, rigorous_details
    ):
        """label=None, color=None give the single-result look: black mean line,
                red fit and marker, three descriptive legend entries.
        """
        ax = plot_bias_correction_fit(rigorous_df, rigorous_details)
        colors = {l.get_color() for l in ax.lines}
        assert colors == {'black', 'red'}
        legend_labels = {l.get_label() for l in ax.lines if not l.get_label().startswith('_')}
        assert legend_labels == {'Mean MI per Gamma', 'WLS Extrapolation'}
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_color_kwarg_applied_to_all_elements(self, mock_show, rigorous_df, rigorous_details):
        """color= reaches every drawn element: the points, the mean line and the fit."""
        ax = plot_bias_correction_fit(rigorous_df, rigorous_details, color='blue')
        colors = {l.get_color() for l in ax.lines}
        assert colors == {'blue'}, f"Expected all elements in 'blue', got {colors}"
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_label_kwarg_collapses_to_one_legend_entry(self, mock_show, rigorous_df, rigorous_details):
        """label= gives one legend entry per result, so the overlays in
        Results.compare() carry the caller's labels."""
        ax = plot_bias_correction_fit(rigorous_df, rigorous_details, label='Condition A')
        legend_labels = [l.get_label() for l in ax.lines if not l.get_label().startswith('_')]
        assert legend_labels == ['Condition A'], f"Expected one clean entry, got {legend_labels}"
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_two_overlaid_calls_stay_visually_distinct(self, mock_show, rigorous_df, rigorous_details):
        """The actual compare() scenario: two calls on one shared ax must not
        collide into identical-looking, identically-labeled series."""
        fig, ax = plt.subplots()
        plot_bias_correction_fit(rigorous_df, rigorous_details, ax=ax,
                                  label='A', color='blue', show=False)
        plot_bias_correction_fit(rigorous_df, rigorous_details, ax=ax,
                                  label='B', color='green', show=False)
        legend_labels = [l.get_label() for l in ax.lines if not l.get_label().startswith('_')]
        assert legend_labels == ['A', 'B']
        colors = [l.get_color() for l in ax.lines if not l.get_label().startswith('_')]
        assert colors == ['blue', 'green']
        plt.close('all')


# ---------------------------------------------------------------------------
# conditional and transfer mode plots
# ---------------------------------------------------------------------------

class TestConditionalPlot:

    @staticmethod
    def _result():
        return rf.difference('conditional', {'mi_xw_y': 1.25, 'mi_w_y': 0.43}, 0.82)

    @patch('matplotlib.pyplot.show')
    def test_conditional_plot_returns_axes(self, mock_show):
        ax = self._result().plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_conditional_plot_has_three_bars(self, mock_show):
        """Both components and their difference are drawn."""
        ax = self._result().plot(show=False)
        assert len(ax.patches) == 3
        assert [t.get_text() for t in ax.get_xticklabels()] == ['I(X,W;Y)', 'I(W;Y)', 'I(X;Y|W)']
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_interaction_plot_has_four_bars(self, mock_show):
        r = rf.difference('interaction', {'mi_xw_y': 1.0, 'mi_x_y': 0.5, 'mi_w_y': 0.4}, 0.1)
        ax = r.plot(show=False)
        assert len(ax.patches) == 4
        plt.close('all')


class TestTransferPlot:

    @staticmethod
    def _result(bidirectional=True):
        comps = {'i_xypast_yfuture': 0.9, 'i_ypast_yfuture': 0.34}
        if bidirectional:
            comps.update(te_yx=0.12, directionality_index=0.65)
        return rf.difference('transfer', comps, 0.56)

    @patch('matplotlib.pyplot.show')
    def test_transfer_plot_returns_axes(self, mock_show):
        ax = self._result().plot(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_two_bars_when_bidirectional(self, mock_show):
        ax = self._result().plot(show=False)
        assert len(ax.patches) == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_one_bar_when_unidirectional(self, mock_show):
        ax = self._result(bidirectional=False).plot(show=False)
        assert len(ax.patches) == 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_transfer_di_in_title(self, mock_show):
        ax = self._result().plot(show=False)
        assert '0.65' in ax.get_title() or 'Directionality' in ax.get_title()
        plt.close('all')


# ---------------------------------------------------------------------------
# Results.compare() for estimate mode
# ---------------------------------------------------------------------------

class TestCompareEstimateMode:

    @patch('matplotlib.pyplot.show')
    def test_compare_estimate_returns_axes(self, mock_show):
        r1 = rf.estimate(history=(0.1, 0.3, 0.5, 0.48), best_epoch=2)
        r2 = rf.estimate(history=(0.05, 0.25, 0.45, 0.50), best_epoch=3)
        ax = Results.compare([r1, r2], labels=['Run A', 'Run B'], show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_compare_estimate_overlays_two_curves(self, mock_show):
        """Two estimate results give two history curves."""
        r1 = rf.estimate(history=(0.1, 0.3, 0.5, 0.48), best_epoch=None)
        r2 = rf.estimate(history=(0.05, 0.25, 0.45, 0.50), best_epoch=None)
        ax = Results.compare([r1, r2], show=False)
        assert len([l for l in ax.lines if len(l.get_xdata()) == 4]) == 2
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_compare_estimate_missing_history_raises(self, mock_show):
        r1 = rf.estimate(history=(0.1, 0.3))
        r2 = rf.estimate(history=None)
        with pytest.raises(ValueError, match="test_mi_history"):
            Results.compare([r1, r2], show=False)

    @patch('matplotlib.pyplot.show')
    def test_compare_refuses_a_result_with_several_repeats(self, mock_show):
        """compare() overlays one curve per result; drawing only the first repeat
        of a result would drop the others without saying so."""
        rows = [{'config_id': 0, 'run_id': rid, 'mi': 0.4, 'test_mi_history': [0.1, 0.4]}
                for rid in range(2)]
        repeated = rf.make_results('estimate', rows)
        with pytest.raises(ValueError, match="holds 2 repeats"):
            Results.compare([rf.estimate(), repeated], show=False)

    @patch('matplotlib.pyplot.show')
    def test_compare_needs_one_swept_key_for_other_modes(self, mock_show):
        r1, r2 = rf.precision(), rf.precision()
        with pytest.raises(ValueError, match="exactly one swept key"):
            Results.compare([r1, r2], show=False)


# ---------------------------------------------------------------------------
# rigorous plot is_reliable=False annotation
# ---------------------------------------------------------------------------

class TestRigorousReliabilityAnnotation:

    @patch('matplotlib.pyplot.show')
    def test_unreliable_annotation_appears(self, mock_show):
        ax = _rigorous_result({'is_reliable': False, 'leverage_warning': True}).plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert any('unreliable' in t.lower() or '⚠' in t for t in texts), texts
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_reliable_annotation_appears(self, mock_show):
        ax = _rigorous_result({'is_reliable': True}).plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert any('reliable' in t.lower() and 'unreliable' not in t.lower() for t in texts), texts
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_no_is_reliable_key_no_annotation(self, mock_show):
        ax = _rigorous_result().plot(show=False)
        assert not any('reliable' in t.get_text().lower() for t in ax.texts)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_unreliable_reason_reflects_actual_flags(self, mock_show):
        """The reason shown is the check that decided it, and only that one."""
        ax = _rigorous_result({'is_reliable': False, 'fit_quality_warning': True,
                               'leverage_warning': True}).plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert any('gamma=1 leverage' in t for t in texts), texts
        assert not any('fit_quality' in t for t in texts), texts
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_unreliable_no_false_reason_when_neither_flag_set(self, mock_show):
        """is_reliable can be False from too few surviving gamma points alone."""
        ax = _rigorous_result({'is_reliable': False, 'fit_quality_warning': False,
                               'leverage_warning': False}).plot(show=False)
        texts = [t.get_text() for t in ax.texts]
        assert any('unreliable' in t.lower() for t in texts)
        assert not any('warning=True' in t for t in texts), texts
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_every_repeat_gets_its_own_annotation(self, mock_show):
        r = rf.rigorous(fits=[{'mi': 0.5, 'is_reliable': True},
                              {'mi': 0.6, 'is_reliable': False, 'leverage_warning': True}],
                        gammas=range(1, 5))
        ax = r.plot(show=False)
        joined = '\n'.join(t.get_text() for t in ax.texts)
        assert 'run 0: extrapolation reliable' in joined
        assert 'run 1: extrapolation unreliable' in joined
        plt.close('all')

    def test_show_false_is_forwarded_to_the_bias_correction_plotter(self):
        """Every per-repeat call draws with show=False, so the figure is shown at
        most once, at the end, and only when the caller asked for it."""
        r = _rigorous_result()
        with patch('neural_mi.visualize.plot.plot_bias_correction_fit') as mock_fn:
            mock_fn.return_value = plt.subplots()[1]
            r.plot(show=False)
            assert mock_fn.call_args.kwargs.get('show') is False
        plt.close('all')

    def test_show_true_shows_once_at_the_end(self):
        r = _rigorous_result()
        with patch('neural_mi.visualize.plot.plot_bias_correction_fit') as mock_fn:
            mock_fn.return_value = plt.subplots()[1]
            with patch('matplotlib.pyplot.show') as mock_show:
                r.plot(show=True)
            assert mock_fn.call_args.kwargs.get('show') is False
            mock_show.assert_called_once()
        plt.close('all')


class TestRigorousCompareReliability:

    def test_per_result_reliability_lines(self):
        """compare() labels each overlaid result's reliability."""
        r1 = _rigorous_result({'is_reliable': True})
        r2 = _rigorous_result({'is_reliable': False, 'leverage_warning': True})
        ax = Results.compare([r1, r2], labels=['Cond A', 'Cond B'], show=False)
        joined = '\n'.join(t.get_text() for t in ax.texts)
        assert 'Cond A' in joined and 'reliable' in joined.lower()
        assert 'Cond B' in joined and 'unreliable' in joined.lower()
        plt.close('all')

    def test_loop_calls_never_show_even_when_outer_show_true(self):
        """Each per-result call inside the loop is show=False, so showing mid-loop
        cannot truncate the overlay."""
        r1, r2 = _rigorous_result(), _rigorous_result()
        with patch('neural_mi.visualize.plot.plot_bias_correction_fit') as mock_fn:
            mock_fn.return_value = plt.subplots()[1]
            with patch('matplotlib.pyplot.show'):
                Results.compare([r1, r2], show=True)
            assert all(c.kwargs.get('show') is False for c in mock_fn.call_args_list)
        plt.close('all')

    def test_overlay_includes_all_results_not_just_first(self):
        """With the real plotter, both results' series land on the axes."""
        ax = Results.compare([_rigorous_result(), _rigorous_result()], labels=['A', 'B'],
                             show=False)
        # Each call draws 6 Line2D artists when a label is passed; two results give 12.
        assert len(ax.lines) == 12, f"Expected lines from both results, got {len(ax.lines)}"
        plt.close('all')

    def test_compare_refuses_several_repeats(self):
        with pytest.raises(ValueError, match="holds 2 repeats"):
            Results.compare([_rigorous_result(), _rigorous_result(n_repeats=2)], show=False)


# ---------------------------------------------------------------------------
# plot_cross_correlation composability
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
        """The red 'True Lag' reference line is drawn at true_lag itself,
        where its legend label says it is."""
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
# analyze_mi_heatmap composability
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
                non-empty allsegs[0] containing only degenerate (empty or single-point)
                segments. The heatmap skips them instead of taking the argmin of an
                empty sequence.
        """
        ax = analyze_mi_heatmap(heatmap_df, absolute_mi_threshold=1e6, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')


# ---------------------------------------------------------------------------
# pairwise plot: channel-count-aware figure sizing
# ---------------------------------------------------------------------------

class TestPairwisePlotFigsize:
    """Results.plot() for mode='pairwise' must use its own channel-count-aware
    default figure size, not the generic (10, 6) every other mode gets --
    the generic top-level axes creation in plot() was pre-empting pairwise's
    own sizing logic before it ever ran (figsize was already consumed and ax
    was already non-None by the time the pairwise branch checked them)."""

    def _make_pairwise_result(self, n_channels):
        mi_matrix = np.random.rand(n_channels, n_channels)
        np.fill_diagonal(mi_matrix, 0)
        return rf.pairwise(mi_matrix)

    def test_large_matrix_uses_channel_count_sizing_not_generic_default(self):
        result = self._make_pairwise_result(20)
        ax = result.plot(show=False)
        expected = (max(4, 20 * 0.65 + 1.2), max(3, 20 * 0.65 + 1.0))
        assert tuple(ax.figure.get_size_inches()) == pytest.approx(expected)
        plt.close('all')

    def test_explicit_figsize_still_overrides(self):
        result = self._make_pairwise_result(20)
        ax = result.plot(show=False, figsize=(5, 5))
        assert tuple(ax.figure.get_size_inches()) == pytest.approx((5.0, 5.0))
        plt.close('all')

    def test_existing_ax_is_not_replaced(self):
        result = self._make_pairwise_result(20)
        fig, ax = plt.subplots(figsize=(3, 3))
        result.plot(ax=ax, show=False)
        assert tuple(fig.get_size_inches()) == pytest.approx((3.0, 3.0))
        plt.close('all')


# ---------------------------------------------------------------------------
# Multi-parameter sweep plotting: heatmap (2 params) / bar (3+ params)
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_df_2param():
    """Sweep results DataFrame over two swept parameters (int-valued)."""
    return pd.DataFrame({
        'embedding_dim': [4, 4, 8, 8],
        'hidden_dim': [16, 32, 16, 32],
        'mi_mean': [0.10, 0.20, 0.30, 0.40],
        'mi_std': [0.01, 0.02, 0.03, 0.04],
    })


@pytest.fixture
def sweep_df_3param():
    """Sweep results DataFrame over three swept parameters (mixed int/float)."""
    return pd.DataFrame({
        'embedding_dim': [4, 4, 8, 8],
        'hidden_dim': [16, 16, 32, 32],
        'dropout': [0.0, 0.1, 0.0, 0.1],
        'mi_mean': [0.10, 0.20, 0.30, 0.40],
        'mi_std': [0.01, 0.02, 0.03, 0.04],
    })


class TestPlotSweepHeatmap:

    def test_returns_axes_with_heatmap_mesh(self, sweep_df_2param):
        ax = plot_sweep_heatmap(sweep_df_2param, param_x='embedding_dim',
                                param_y='hidden_dim', show=False)
        assert isinstance(ax, plt.Axes)
        assert len(ax.collections) > 0  # seaborn heatmap draws a QuadMesh
        plt.close('all')

    def test_axis_labels_from_param_names(self, sweep_df_2param):
        ax = plot_sweep_heatmap(sweep_df_2param, param_x='embedding_dim',
                                param_y='hidden_dim', show=False)
        assert ax.get_xlabel() == 'Embedding Dim'
        assert ax.get_ylabel() == 'Hidden Dim'
        plt.close('all')


class TestPlotSweepBar:

    def test_returns_axes_with_one_bar_per_row(self, sweep_df_3param):
        ax = plot_sweep_bar(sweep_df_3param,
                            param_cols=['embedding_dim', 'hidden_dim', 'dropout'],
                            show=False)
        assert isinstance(ax, plt.Axes)
        assert len(ax.patches) == len(sweep_df_3param)
        plt.close('all')

    def test_labels_preserve_int_dtype_alongside_float_column(self, sweep_df_3param):
        """A mixed int/float row (embedding_dim int, dropout float) must not
        upcast the int column to '4.0' in the label -- see plot_sweep_bar's
        column-wise (not row-wise .apply) label construction."""
        ax = plot_sweep_bar(sweep_df_3param,
                            param_cols=['embedding_dim', 'hidden_dim', 'dropout'],
                            show=False)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any('embedding_dim=4,' in label for label in labels)
        assert not any('embedding_dim=4.0' in label for label in labels)
        plt.close('all')


class TestResultsPlotSweepKindDispatch:
    """Results.plot() on several configurations picks line, heatmap or bar from
    how many keys vary (result.params['config_keys'])."""

    @staticmethod
    def _result(frame, keys):
        rows = [{'config_id': i, **{k: rec[k] for k in keys}, 'run_id': 0, 'mi': rec['mi_mean']}
                for i, rec in enumerate(frame.to_dict(orient='records'))]
        return rf.make_results('sweep', rows, config_keys=keys)

    def test_single_param_defaults_to_line(self, sweep_df_2param):
        df = sweep_df_2param[sweep_df_2param['hidden_dim'] == 16]
        ax = self._result(df, ['embedding_dim']).plot(show=False)
        assert len(ax.lines) > 0
        plt.close('all')

    def test_two_params_defaults_to_heatmap(self, sweep_df_2param):
        ax = self._result(sweep_df_2param, ['embedding_dim', 'hidden_dim']).plot(show=False)
        assert len(ax.collections) > 0
        plt.close('all')

    def test_three_params_defaults_to_bar(self, sweep_df_3param):
        result = self._result(sweep_df_3param, ['embedding_dim', 'hidden_dim', 'dropout'])
        ax = result.plot(show=False)
        assert len(ax.patches) == len(sweep_df_3param)
        plt.close('all')

    def test_kind_override_forces_bar_for_two_params(self, sweep_df_2param):
        ax = self._result(sweep_df_2param, ['embedding_dim', 'hidden_dim']).plot(show=False, kind='bar')
        assert len(ax.patches) == len(sweep_df_2param)
        plt.close('all')

    def test_heatmap_kind_rejects_three_params(self, sweep_df_3param):
        result = self._result(sweep_df_3param, ['embedding_dim', 'hidden_dim', 'dropout'])
        with pytest.raises(ValueError, match="needs exactly 2 swept keys"):
            result.plot(show=False, kind='heatmap')
        plt.close('all')

    def test_compare_rejects_multi_param_sweep_results(self, sweep_df_2param):
        a = self._result(sweep_df_2param, ['embedding_dim', 'hidden_dim'])
        b = self._result(sweep_df_2param, ['embedding_dim', 'hidden_dim'])
        with pytest.raises(ValueError, match="exactly one swept key"):
            Results.compare([a, b], show=False)
        plt.close('all')
