# tests/test_animate.py
"""Tests for animate_training() and result.animate()."""
import warnings
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from unittest.mock import patch

from neural_mi.results import Results
from neural_mi.visualize.animate import (
    animate_training,
    _auto_panels,
    _fit_reducer,
    _resolve_scatter_color,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    n_epochs=5,
    n_tracked=20,
    embed_dim=4,
    include_spectral=False,
    include_spectrum=False,
    include_embeddings=True,
    include_train=False,
):
    """Build a minimal Results object with synthetic training history."""
    test_history = list(np.linspace(0.1, 1.0, n_epochs))
    details = {
        'test_mi_history': test_history,
        'best_epoch': n_epochs - 1,
    }
    if include_train:
        details['train_mi_history'] = list(np.linspace(0.2, 1.1, n_epochs))
    if include_spectral or include_spectrum:
        spectral = []
        for e in range(n_epochs):
            entry = {'pr_singular': 1.0 + e * 0.1}
            if include_spectrum:
                entry['spectrum'] = list(np.linspace(1.0, 0.1, 8))
            spectral.append(entry)
        details['spectral_metrics_history'] = spectral
    if include_embeddings:
        details['embedding_history_x'] = [
            np.random.randn(n_tracked, embed_dim).astype(np.float32)
            for _ in range(n_epochs)
        ]
        details['embedding_history_y'] = [
            np.random.randn(n_tracked, embed_dim).astype(np.float32)
            for _ in range(n_epochs)
        ]
    return Results(
        mode='dimensionality',
        params={'output_units': 'bits'},
        details=details,
    )


# ---------------------------------------------------------------------------
# _auto_panels
# ---------------------------------------------------------------------------

class TestAutoPanels:
    def test_mi_always_present(self):
        panels = _auto_panels({'test_mi_history': [1.0]})
        assert 'mi' in panels

    def test_spectral_metrics_detected(self):
        details = {
            'test_mi_history': [1.0],
            'spectral_metrics_history': [{'pr_singular': 2.0}],
        }
        panels = _auto_panels(details)
        assert 'spectral_metrics' in panels

    def test_spectrum_detected(self):
        details = {
            'test_mi_history': [1.0],
            'spectral_metrics_history': [{'pr_singular': 2.0, 'spectrum': [1, 2, 3]}],
        }
        panels = _auto_panels(details)
        assert 'spectrum' in panels

    def test_embeddings_detected(self):
        details = {
            'test_mi_history': [1.0],
            'embedding_history_x': [np.zeros((5, 4))],
        }
        panels = _auto_panels(details)
        assert 'embeddings' in panels

    def test_no_embeddings_when_absent(self):
        panels = _auto_panels({'test_mi_history': [1.0]})
        assert 'embeddings' not in panels


# ---------------------------------------------------------------------------
# _fit_reducer
# ---------------------------------------------------------------------------

class TestFitReducer:
    def test_no_reduction_when_embed_dim_le_n_components(self):
        # embed_dim=2, n_components=2 — no reduction needed
        history = [np.random.randn(10, 2) for _ in range(3)]
        reducer, reduced = _fit_reducer(history, n_components=2, reduction='pca')
        assert reducer is None
        assert len(reduced) == 3
        assert reduced[0].shape == (10, 2)

    def test_reduction_none_returns_first_n_components(self):
        history = [np.random.randn(10, 8) for _ in range(3)]
        reducer, reduced = _fit_reducer(history, n_components=2, reduction='none')
        assert reducer is None
        assert reduced[0].shape == (10, 2)

    def test_pca_reduction_shape(self):
        pytest.importorskip('sklearn')
        history = [np.random.randn(30, 8).astype(np.float32) for _ in range(4)]
        reducer, reduced = _fit_reducer(history, n_components=2, reduction='pca')
        assert reducer is not None
        assert len(reduced) == 4
        assert all(r.shape == (30, 2) for r in reduced)

    def test_empty_history_returns_empty(self):
        reducer, reduced = _fit_reducer([], n_components=2, reduction='pca')
        assert reducer is None
        assert reduced == []

    def test_unknown_reduction_raises(self):
        history = [np.random.randn(10, 8) for _ in range(2)]
        with pytest.raises(ValueError, match="not recognised"):
            _fit_reducer(history, n_components=2, reduction='tsne')


# ---------------------------------------------------------------------------
# _resolve_scatter_color
# ---------------------------------------------------------------------------

class TestResolveScatterColor:
    def test_none_returns_viridis(self):
        c, cmap, vmin, vmax = _resolve_scatter_color(None)
        assert c is None
        assert cmap == 'viridis'
        assert vmin is None

    def test_continuous_float_array(self):
        arr = np.array([0.1, 0.5, 0.9], dtype=float)
        c, cmap, vmin, vmax = _resolve_scatter_color(arr)
        assert cmap == 'viridis'
        np.testing.assert_array_almost_equal(c, arr)
        assert vmin is None

    def test_categorical_int_array(self):
        arr = np.array([0, 1, 2, 0, 1], dtype=int)
        c, cmap, vmin, vmax = _resolve_scatter_color(arr)
        # vmin/vmax should be set for categorical
        assert vmin is not None
        assert vmax is not None
        # c should be float (mapped indices)
        assert c.dtype == float

    def test_categorical_string_array(self):
        arr = np.array(['a', 'b', 'a', 'c'])
        c, cmap, vmin, vmax = _resolve_scatter_color(arr)
        assert vmin is not None
        assert len(c) == 4


# ---------------------------------------------------------------------------
# animate_training — basic smoke tests
# ---------------------------------------------------------------------------

class TestAnimateTraining:

    @patch('matplotlib.pyplot.show')
    def test_mi_only_returns_funcanimation(self, mock_show):
        result = _make_result(include_embeddings=False)
        anim = animate_training(result, panels=['mi'], show=False)
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_auto_panels_runs(self, mock_show):
        result = _make_result(include_embeddings=False)
        anim = animate_training(result, show=False)
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_spectral_metrics_panel(self, mock_show):
        result = _make_result(include_spectral=True, include_embeddings=False)
        anim = animate_training(result, panels=['mi', 'spectral_metrics'], show=False)
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_spectrum_panel(self, mock_show):
        result = _make_result(include_spectrum=True, include_embeddings=False)
        anim = animate_training(result, panels=['mi', 'spectrum'], show=False)
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_train_mi_overlay(self, mock_show):
        result = _make_result(include_train=True, include_embeddings=False)
        anim = animate_training(result, panels=['mi'], show=False)
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    def test_missing_test_mi_history_raises(self):
        result = Results(
            mode='estimate',
            params={},
            details={'train_mi_history': [0.5, 0.6]},
        )
        with pytest.raises(ValueError, match="test_mi_history"):
            animate_training(result, show=False)

    def test_no_panels_raises(self):
        """If panels list leads to empty col_spec, should raise."""
        result = _make_result(include_embeddings=False)
        # 'embeddings' panel with no embedding data → warning then removed → empty
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            with pytest.raises(ValueError, match="No panels"):
                animate_training(result, panels=['embeddings'], show=False)
        plt.close('all')


# ---------------------------------------------------------------------------
# animate_training — embedding panels
# ---------------------------------------------------------------------------

class TestAnimateEmbeddings:

    @patch('matplotlib.pyplot.show')
    def test_single_embedding_label_array(self, mock_show):
        pytest.importorskip('sklearn')
        result = _make_result(n_tracked=20, embed_dim=8)
        labels = np.random.randint(0, 3, size=20)
        anim = animate_training(
            result,
            panels=['mi', 'embeddings'],
            embedding_labels=labels,
            show=False,
            reduction='pca',
        )
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_dict_embedding_labels_multiple_subplots(self, mock_show):
        pytest.importorskip('sklearn')
        result = _make_result(n_tracked=20, embed_dim=8)
        labels = {
            'category': np.random.randint(0, 3, size=20),
            'position': np.random.randn(20).astype(float),
        }
        anim = animate_training(
            result,
            panels=['mi', 'embeddings'],
            embedding_labels=labels,
            show=False,
            reduction='pca',
        )
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_embeddings_without_history_warns_and_removes_panel(self, mock_show):
        """When embedding panel is requested but no history, emit a warning."""
        result = _make_result(include_embeddings=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            anim = animate_training(
                result,
                panels=['mi', 'embeddings'],
                show=False,
            )
        messages = [str(w.message) for w in caught]
        assert any('embedding_history_x' in m or 'track_embeddings' in m
                   for m in messages), f"Expected embedding warning, got: {messages}"
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_3d_embeddings(self, mock_show):
        pytest.importorskip('sklearn')
        result = _make_result(n_tracked=20, embed_dim=8)
        anim = animate_training(
            result,
            panels=['embeddings'],
            n_components=3,
            show=False,
            reduction='pca',
        )
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_reduction_none(self, mock_show):
        """reduction='none' should use first n_components dimensions directly."""
        result = _make_result(n_tracked=20, embed_dim=4)
        anim = animate_training(
            result,
            panels=['mi', 'embeddings'],
            n_components=2,
            reduction='none',
            show=False,
        )
        assert isinstance(anim, manim.FuncAnimation)
        plt.close('all')


# ---------------------------------------------------------------------------
# result.animate() — thin wrapper
# ---------------------------------------------------------------------------

class TestResultAnimate:
    @patch('neural_mi.visualize.animate.animate_training')
    def test_result_animate_delegates_to_animate_training(self, mock_fn):
        mock_fn.return_value = MagicMock(spec=manim.FuncAnimation)
        result = _make_result(include_embeddings=False)
        result.animate(show=False, fps=5)
        mock_fn.assert_called_once_with(result, show=False, fps=5)


# Avoid matplotlib object import error in mock
from unittest.mock import MagicMock  # noqa: E402 (already imported via top-level, harmless)
