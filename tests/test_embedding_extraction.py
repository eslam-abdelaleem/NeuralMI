# tests/test_embedding_extraction.py
"""Tests for Phase G: embedding extraction (G1/G2) and plot_embeddings (G3)."""
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt

import numpy as np
import pytest
import torch

import neural_mi as nmi
from neural_mi import Model, Training, Output
from neural_mi.visualize import plot_embeddings

# _BASE kept for assertions that reference embedding_dim etc.
_BASE = {
    'n_epochs': 2, 'learning_rate': 1e-4, 'batch_size': 32,
    'patience': 1, 'embedding_dim': 8, 'hidden_dim': 16, 'n_layers': 1,
}
_MODEL = Model(embedding_dim=8, hidden_dim=16, n_layers=1)
_TRAINING = Training(n_epochs=2, learning_rate=1e-4, batch_size=32, patience=1)


@pytest.fixture(scope='module')
def simple_data():
    x, y = nmi.generators.generate_correlated_gaussians(n_samples=500, dim=5, mi=1.0)
    return x, y


# ---------------------------------------------------------------------------
# G1 — return_embeddings during training (mode='estimate')
# ---------------------------------------------------------------------------

def test_g1_return_embeddings_keys_present(simple_data):
    """return_embeddings=True must add embeddings_x/y to results.details."""
    x, y = simple_data
    results = nmi.run(
        x, y, mode='estimate',
        model=_MODEL, training=_TRAINING,
        output=Output(return_embeddings=True),
        n_workers=1,
    )
    assert 'embeddings_x' in results.details, "embeddings_x missing from details."
    assert 'embeddings_y' in results.details, "embeddings_y missing from details."


def test_g1_return_embeddings_shapes(simple_data):
    """Extracted embeddings should be 2-D numpy arrays."""
    x, y = simple_data
    results = nmi.run(
        x, y, mode='estimate',
        model=_MODEL, training=_TRAINING,
        output=Output(return_embeddings=True),
        n_workers=1,
    )
    zx = results.details['embeddings_x']
    zy = results.details['embeddings_y']
    assert isinstance(zx, np.ndarray) and zx.ndim == 2
    assert isinstance(zy, np.ndarray) and zy.ndim == 2
    assert zx.shape[1] == _BASE['embedding_dim']
    assert zy.shape[1] == _BASE['embedding_dim']


def test_g1_return_embeddings_false_no_keys(simple_data):
    """return_embeddings=False (default) must NOT add embeddings to details."""
    x, y = simple_data
    results = nmi.run(
        x, y, mode='estimate',
        model=_MODEL, training=_TRAINING,
        n_workers=1,   # return_embeddings defaults to False
    )
    assert 'embeddings_x' not in results.details
    assert 'embeddings_y' not in results.details


def test_g1_embeddings_full_dataset_not_capped_by_max_eval_samples(simple_data):
    """max_eval_samples must not cap embedding extraction (it only controls eval MI).

    Previously, the embedding extraction block reused max_eval_samples (default
    5000) and randomly subsampled the dataset when n > threshold.  Setting a
    tiny max_eval_samples should no longer affect the embedding count.
    """
    x, y = simple_data  # 500 samples
    results = nmi.run(
        x, y, mode='estimate',
        model=_MODEL,
        training=Training(n_epochs=2, learning_rate=1e-4, batch_size=32,
                          patience=1, max_eval_samples=10),
        output=Output(return_embeddings=True),
        n_workers=1,
    )
    n_emb = results.details['embeddings_x'].shape[0]
    # With max_eval_samples=10 the old code would have returned ≤10 rows.
    # Now all ~500 windows must be embedded.
    assert n_emb > 10, (
        f"Expected all dataset windows embedded, got {n_emb}. "
        "max_eval_samples must not gate embedding extraction."
    )


# ---------------------------------------------------------------------------
# G2 — extract_embeddings() from a saved model file
# ---------------------------------------------------------------------------

def test_g2_model_saved_in_new_format(simple_data, tmp_path):
    """Saved model must use the extended format {state_dict, build_params}."""
    x, y = simple_data
    model_path = str(tmp_path / 'model.pt')
    nmi.run(
        x, y, mode='estimate',
        model=_MODEL,
        training=Training(n_epochs=2, learning_rate=1e-4, batch_size=32,
                          patience=1, save_best_model_path=model_path),
        n_workers=1,
    )
    assert os.path.exists(model_path), "Model file was not created."
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)
    assert isinstance(loaded, dict), "Saved model is not a dict."
    assert 'state_dict' in loaded, "Missing 'state_dict' key."
    assert 'build_params' in loaded, "Missing 'build_params' key (new format)."


def test_g2_extract_embeddings_returns_arrays(simple_data, tmp_path):
    """extract_embeddings() must return two 2-D numpy arrays."""
    x, y = simple_data
    model_path = str(tmp_path / 'model.pt')
    nmi.run(
        x, y, mode='estimate',
        model=_MODEL,
        training=Training(n_epochs=2, learning_rate=1e-4, batch_size=32,
                          patience=1, save_best_model_path=model_path),
        n_workers=1,
    )
    zx, zy = nmi.extract_embeddings(model_path, x, y)
    assert isinstance(zx, np.ndarray) and zx.ndim == 2
    assert isinstance(zy, np.ndarray) and zy.ndim == 2


def test_g2_extract_embeddings_old_format_without_params_raises(tmp_path):
    """Calling extract_embeddings on old-format file without base_params must raise."""
    # Simulate an old-format save (raw state dict)
    dummy_state = {'weight': torch.ones(4, 4)}
    model_path = str(tmp_path / 'old_model.pt')
    torch.save(dummy_state, model_path)
    with pytest.raises(ValueError, match="base_params"):
        nmi.extract_embeddings(model_path, np.zeros((10, 4)), np.zeros((10, 4)))


def test_g2_extract_embeddings_no_max_samples_param(simple_data, tmp_path):
    """extract_embeddings no longer accepts max_samples — it must raise TypeError."""
    x, y = simple_data
    model_path = str(tmp_path / 'model_ms.pt')
    nmi.run(
        x, y, mode='estimate',
        model=_MODEL,
        training=Training(n_epochs=2, learning_rate=1e-4, batch_size=32,
                          patience=1, save_best_model_path=model_path),
        n_workers=1,
    )
    with pytest.raises(TypeError, match="max_samples"):
        nmi.extract_embeddings(model_path, x, y, max_samples=100)


# ---------------------------------------------------------------------------
# G3 — plot_embeddings() visualization helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ['pca', 'none'])
def test_g3_plot_embeddings_basic_methods(method):
    """plot_embeddings() must return Axes for 'pca' and 'none' methods."""
    z = np.random.randn(100, 4 if method == 'pca' else 2)
    ax = plot_embeddings(z, method=method, dim=2)
    assert ax is not None
    plt.close('all')


def test_g3_plot_embeddings_with_continuous_color():
    """plot_embeddings() with continuous color should not raise."""
    z = np.random.randn(80, 4)
    color = np.random.randn(80)
    ax = plot_embeddings(z, color=color, method='pca', dim=2)
    assert ax is not None
    plt.close('all')


def test_g3_plot_embeddings_with_categorical_color():
    """plot_embeddings() with integer labels should produce a legend."""
    z = np.random.randn(80, 4)
    labels = np.array([0, 1, 2] * 26 + [0, 1])
    ax = plot_embeddings(z, color=labels, method='pca', dim=2)
    assert ax is not None
    assert ax.get_legend() is not None
    plt.close('all')


def test_g3_plot_embeddings_auto_method_pca_fallback():
    """method='auto' should fall back to pca when embed_dim > dim (and umap missing)."""
    z = np.random.randn(100, 16)
    # With embed_dim=16 > dim=2, auto should apply reduction
    ax = plot_embeddings(z, method='auto', dim=2)
    assert ax is not None
    plt.close('all')


def test_g3_plot_embeddings_none_method_requires_enough_dims():
    """method='none' with embed_dim < dim must raise ValueError."""
    z = np.random.randn(50, 1)
    with pytest.raises(ValueError, match="embed_dim"):
        plot_embeddings(z, method='none', dim=2)


def test_g3_plot_embeddings_invalid_method_raises():
    """Unknown method must raise ValueError."""
    z = np.random.randn(50, 4)
    with pytest.raises(ValueError, match="not recognised"):
        plot_embeddings(z, method='xyz')


def test_g3_plot_embeddings_3d():
    """plot_embeddings() with dim=3 should create a 3-D axes."""
    z = np.random.randn(60, 8)
    ax = plot_embeddings(z, method='pca', dim=3)
    assert ax is not None
    # 3-D axes have a set_zlabel method
    assert hasattr(ax, 'set_zlabel')
    plt.close('all')


# ---------------------------------------------------------------------------
# Build params keys — norm_layer / dropout / use_spectral_norm
# ---------------------------------------------------------------------------

def test_build_params_keys_include_norm_and_dropout():
    """_BUILD_PARAMS_KEYS must include dropout, norm_layer, and use_spectral_norm.

    Without these keys, models saved with BatchNorm/LayerNorm/dropout cannot
    be correctly reconstructed by extract_embeddings().
    """
    from neural_mi.analysis.task import _BUILD_PARAMS_KEYS
    for key in ('dropout', 'norm_layer', 'use_spectral_norm'):
        assert key in _BUILD_PARAMS_KEYS, (
            f"'{key}' is missing from _BUILD_PARAMS_KEYS. "
            f"Models using this setting cannot be reloaded via extract_embeddings()."
        )
