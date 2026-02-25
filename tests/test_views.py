# tests/test_views.py
import pytest
import torch
import numpy as np
from neural_mi.data.views import SubsetView
from neural_mi.data.static import StaticDataset
from neural_mi.data.handler import PairedDataset, PairedTemporalDataset, ContinuousWindowDataset, WindowManager

class TestSubsetView:
    @pytest.fixture
    def static_dataset(self):
        # (samples, channels, features)
        x = np.random.randn(10, 2, 1)
        y = np.random.randn(10, 2, 1)
        x_ds = StaticDataset(x)
        y_ds = StaticDataset(y)
        return PairedDataset(x_ds, y_ds)

    @pytest.fixture
    def temporal_dataset(self):
        # Continuous data: (timepoints, channels)
        x = np.random.randn(100, 1)
        y = np.random.randn(100, 1)
        # Window size 10 -> 10 windows (no overlap logic in this simple mock, wait, ContinuousWindowDataset overlaps?)
        # ContinuousWindowDataset moves data to windows based on window_manager.
        # If t_start=0, t_end=100, window_size=10.
        # Window starts: 0, 10, ... 90. 10 windows.

        wm = WindowManager(window_size=10, t_start=0, t_end=100)
        x_ds = ContinuousWindowDataset(x, window_manager=wm)
        y_ds = ContinuousWindowDataset(y, window_manager=wm)
        return PairedTemporalDataset(x_ds, y_ds, window_size=10, t_start=0, t_end=100)

    def test_subset_view_indices_static(self, static_dataset):
        indices = [0, 2, 4]
        view = SubsetView(static_dataset, indices=indices)
        assert len(view) == 3
        x, y = view[0] # Should be index 0 of original
        assert torch.allclose(x, static_dataset[0][0])

        x, y = view[1] # Should be index 2 of original
        assert torch.allclose(x, static_dataset[2][0])

    def test_subset_view_channels_static(self, static_dataset):
        # Select channel 0 only
        view = SubsetView(static_dataset, channels_x=[0], channels_y=[1])
        x, y = view[0]
        # Original x is (2, 1) (channels, features) after squeeze?
        # StaticDataset returns (channels, features).
        # x shape should be (1, 1).
        assert x.shape == (1, 1)
        assert y.shape == (1, 1)

    def test_subset_view_times_temporal(self, temporal_dataset):
        # Windows are [0, 10), [10, 20), ... [90, 100)
        # Select times [0, 20) -> windows 0, 1
        # Times must be (n_regions, 2)
        times = np.array([[0, 20]])
        view = SubsetView(temporal_dataset, times=times)
        # With the fix (side='left' for end), end=20 should align with start of window 2 [20, 30).
        # 20 <= 20. index 2. Range 0 to 2 (exclusive). indices 0, 1.
        # So len should be 2.
        assert len(view) == 2

        # Test time shift
        temporal_dataset.time_shift(offset_x=10)
        # _on_dataset_updated is called.
        # self.times = self.times + 10 - 0 = [10, 30].
        assert view.times[0, 0] == 10
        assert view.times[0, 1] == 30

        assert len(view) > 0

    def test_subset_view_indices_temporal_convert_to_times(self, temporal_dataset):
        indices = [0, 1] # First two windows: [0, 10), [10, 20)
        view = SubsetView(temporal_dataset, indices=indices)
        # Should populate self.times
        assert view.times is not None
        assert view.times.shape == (1, 2)
