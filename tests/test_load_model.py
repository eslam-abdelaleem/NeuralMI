import pytest
import torch
import os
import tempfile
import neural_mi as nmi
from neural_mi.utils import load_model, build_critic
import numpy as np

def test_load_model_successfully():
    x = np.random.randn(100, 2)
    y = np.random.randn(100, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        nmi.run(x, y, save_best_model_path=path, base_params={'n_epochs': 2})

        assert os.path.exists(path)
        critic = load_model(path)

        assert isinstance(critic, torch.nn.Module)
        assert hasattr(critic, 'forward')

def test_load_model_old_format_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_old.pt")
        # Simulate old format
        torch.save({'state_dict': {}}, path)

        with pytest.raises(ValueError, match="does not contain 'build_params'"):
            load_model(path)
