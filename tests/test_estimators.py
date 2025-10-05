import torch
import pytest
from neural_mi.estimators.bounds import (
    tuba_lower_bound,
    nwj_lower_bound,
    infonce_lower_bound,
    js_fgan_lower_bound,
    smile_lower_bound,
    logmeanexp_nodiag,
)

@pytest.fixture
def scores():
    """A fixture to create a sample scores tensor."""
    return torch.randn(10, 10)

@pytest.fixture
def scores_independent():
    """Scores for independent variables (MI should be ~0)."""
    return torch.randn(100, 100) * 0.1 # Smaller variance for stability

@pytest.fixture
def scores_identical():
    """Scores for identical variables (MI should be high)."""
    return torch.eye(100) * 10 + torch.randn(100, 100) * 0.1

def test_tuba_lower_bound(scores):
    """Test the TUBA lower bound estimator."""
    mi = tuba_lower_bound(scores)
    assert isinstance(mi, torch.Tensor)
    assert mi.shape == ()

def test_nwj_lower_bound(scores):
    """Test the NWJ lower bound estimator."""
    mi = nwj_lower_bound(scores)
    assert isinstance(mi, torch.Tensor)
    assert mi.shape == ()

def test_infonce_lower_bound(scores):
    """Test the InfoNCE lower bound estimator."""
    mi = infonce_lower_bound(scores)
    assert isinstance(mi, torch.Tensor)
    assert mi.shape == ()

def test_js_fgan_lower_bound(scores):
    """Test the JS-GAN lower bound estimator."""
    mi = js_fgan_lower_bound(scores)
    assert isinstance(mi, torch.Tensor)
    assert mi.shape == ()

def test_smile_lower_bound(scores):
    """Test the SMILE lower bound estimator."""
    mi = smile_lower_bound(scores)
    assert isinstance(mi, torch.Tensor)
    assert mi.shape == ()

def test_smile_lower_bound_with_clip(scores):
    """Test the SMILE lower bound estimator with clipping."""
    mi = smile_lower_bound(scores, clip=5.0)
    assert isinstance(mi, torch.Tensor)
    assert mi.shape == ()

def test_logmeanexp_nodiag(scores):
    """Test the logmeanexp_nodiag helper function."""
    result = logmeanexp_nodiag(scores)
    assert isinstance(result, torch.Tensor)
    assert result.shape == ()

@pytest.mark.parametrize("estimator", [
    tuba_lower_bound,
    nwj_lower_bound,
    infonce_lower_bound,
    js_fgan_lower_bound,
    smile_lower_bound,
])
def test_bounds_on_independent_data(estimator, scores_independent):
    """Test that MI bounds are not large and positive for independent data."""
    mi = estimator(scores_independent)
    # For independent data, the lower bound can be negative, but it shouldn't be a large positive number.
    # We check that it's less than a small positive value.
    assert mi < 1.0

@pytest.mark.parametrize("estimator", [
    tuba_lower_bound,
    nwj_lower_bound,
    infonce_lower_bound,
    # js_fgan_lower_bound is excluded as it's unstable with this test setup
    smile_lower_bound,
])
def test_bounds_on_correlated_data(estimator, scores_identical):
    """Test that MI bounds are positive for correlated data."""
    mi = estimator(scores_identical)
    assert mi > 0.0

def test_js_bound_on_correlated_data(scores_identical):
    """Test JS bound separately as it's unstable."""
    mi = js_fgan_lower_bound(scores_identical)
    # For JS, the bound can be negative even for correlated data without a trained critic.
    # We just check that it runs without error. A more robust test would require training.
    assert isinstance(mi, torch.Tensor)