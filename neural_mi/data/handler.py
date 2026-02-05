# neural_mi/data/handler.py
import torch
import numpy as np
import inspect

from .temporal import *
from .static import *
from torch.utils.data import Dataset
from neural_mi.logger import logger


class WindowManager:
    """Centralized manager for creating and aligning temporal windows."""
    
    def __init__(self, window_size, t_start=None, t_end=None):
        self.window_size = window_size
        self.t_start = t_start
        self.t_end = t_end
        self.window_times = None
        self.valid_windows = None
        self.n_windows = 0
        self._observers = []  # Datasets that need to be notified
        if t_start is not None and t_end is not None:
            self.create_windows()
        
    def register_observer(self, observer):
        """Register a dataset to be notified of changes."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def _notify_observers(self):
        """Notify all registered datasets of window changes (lightweight updates only)."""
        for observer in self._observers:
            observer._on_window_manager_updated()

    def update_parameters(self, window_size=None, t_start=None, t_end=None):
        """Update window parameters and regenerate windows."""
        if window_size is not None:
            self.window_size = window_size
        if t_start is not None:
            self.t_start = t_start
        if t_end is not None:
            self.t_end = t_end
        
        if self.t_start is not None and self.t_end is not None:
            self.create_windows()
            self._notify_observers()
    
    def create_windows(self):
        """Create window times for a given temporal range."""
        if self.t_start is None or self.t_end is None:
            raise RuntimeError("t_start and t_end parameters need to be set to create windows")
        self.window_times = np.arange(self.t_start, self.t_end, self.window_size)
        self.n_windows = len(self.window_times)
        # Initialize all windows as valid - will be updated by datasets
        self.valid_windows = np.full(self.window_times.size, True, dtype=bool)
    
    def __len__(self):
        """len() will return however many valid windows there are"""
        return self.n_windows


class PairedTemporalDataset(Dataset):
    """Wrapper for paired X and Y datasets with temporal alignment."""
    
    def __init__(self, x_dataset, y_dataset=None,
                 window_size=None,
                 t_start=None, t_end=None,
                 validate_windows=True):
        """
        Parameters
        ----------
        x_dataset : TemporalWindowDataset
            Dataset for X variable (not yet initialized with windows)
        y_dataset : TemporalWindowDataset, optional
            Dataset for Y variable (not yet initialized with windows)
        window_size : float
            Window size in time units
        t_start : float, optional
            Start time for windows
        t_end : float, optional
            End time for windows
        validate_windows : bool, optional
            Whether to return/use only "valid" windows where data is present
            Akin to a conditional MI on the presence of data
        """
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.validate_windows = validate_windows

        # Create window manager shared between X and Y
        if window_size is None:
            raise ValueError("window_size must be provided")
        # Determine temporal extent and create windows
        self.window_manager = WindowManager(window_size)
        self._initialize_windows(t_start, t_end)
        # Attach window manager to datasets
        self.x_dataset.set_window_manager(self.window_manager)
        if self.y_dataset is not None:
            self.y_dataset.set_window_manager(self.window_manager)
        # Build windows with full orchestration
        self._build_windows()
        
        logger.info(f"Created {self.window_manager.n_windows} aligned windows")

    # Small properties for convenient access to main data
    @property
    def x_data(self):
        return self.x_dataset.data
    
    @property
    def y_data(self):
        return self.y_dataset.data
    
    def _initialize_windows(self, t_start, t_end):
        """Create windows based on data extent."""
        # Get temporal extent from both datasets
        x_start, x_end = self.x_dataset.get_temporal_extent()
        if self.y_dataset is not None:
            y_start, y_end = self.y_dataset.get_temporal_extent()
            data_start, data_end = max(x_start, y_start), min(x_end, y_end)
        else:
            data_start, data_end = x_start, x_end
        # Store data time extents to be used when time shifting, but only once!
        if not hasattr(self, 'original_data_start'):
            self.original_data_start = data_start
            self.original_data_end = data_end
        # Bound data time extents by original
        data_start = min(data_start, self.original_data_start)
        data_end = max(data_end, self.original_data_end)
        # Apply user-specified bounds if provided
        final_start = t_start if t_start is not None else data_start
        final_end = t_end if t_end is not None else data_end
        if final_start >= final_end:
            raise ValueError(
                f"Invalid temporal range: t_start={final_start}, t_end={final_end}"
            )
        # Create windows
        self.window_manager.update_parameters(t_start=final_start, t_end=final_end)
        
        if self.window_manager.window_times is None or len(self.window_manager.window_times) == 0:
            raise ValueError(
                f"No windows could be created in range [{final_start}, {final_end}] "
                f"with window_size={self.window_manager.window_size}"
            )
    
    def _build_windows(self, time_shift=None):
        """
        Full sequence of moving data to windows and checking for presence of data.
        
        Parameters
        ----------
        time_shift : offset_x, optional
            If windows are being rebuilt due to time shift, contains the offset applied to x
        """
        # Step 1: Move data to ALL windows for both X and Y
        self.x_dataset.move_data_to_windows()
        if self.y_dataset is not None:
            self.y_dataset.move_data_to_windows()
        # Step 2: Validate and filter if requested
        if self.validate_windows:
            x_valid = self.x_dataset.validate_window_coverage()
            if self.y_dataset is not None:
                y_valid = self.y_dataset.validate_window_coverage()
                # Only keep windows valid for BOTH datasets
                combined_valid = np.logical_and(x_valid, y_valid)
            else:
                combined_valid = x_valid
            # Step 3: Update WindowManager's tracking
            self.window_manager.valid_windows = combined_valid
            self.window_manager.n_windows = int(combined_valid.sum())
            # Step 4: Apply the same mask to both datasets
            self.x_dataset.remove_invalid_windows()
            if self.y_dataset is not None:
                self.y_dataset.remove_invalid_windows()
            if self.window_manager.n_windows == 0:
                print(f"DEBUG: No valid windows. x_valid sum: {x_valid.sum()}, total: {len(x_valid)}")
                raise ValueError("No valid windows after checking data coverage")
            logger.info(
                f"Window coverage: {self.window_manager.n_windows}/{len(self.window_manager.window_times)} "
                f"windows have sufficient data"
            )
        else:
            # No validation - all windows are valid
            self.window_manager.n_windows = len(self.window_manager.window_times)
        
        # Notify any views that windows have been rebuilt
        self._notify_subset_views(time_shift=time_shift)

    def _notify_subset_views(self, time_shift=None):
        """
        Notify all registered subset views that windows have been rebuilt.

        Parameters
        ----------
        time_shift : offset_x, optional
            If windows were rebuilt due to time shift, contains the offset applied to x
        """
        if hasattr(self, '_subset_views'):
            for view in self._subset_views:
                view._on_dataset_updated(time_shift=time_shift)
    
    def __len__(self):
        return len(self.x_dataset)
    
    def __getitem__(self, idx):
        x_data = self.x_dataset[idx]
        y_data = self.y_dataset[idx] if self.y_dataset else None
        return x_data, y_data
    
    def apply_noise(self, amplitude_x=0, amplitude_y=0):
        """Apply noise to both datasets."""
        if amplitude_x > 0:
            self.x_dataset.apply_noise(amplitude_x)
        if amplitude_y > 0 and self.y_dataset:
            self.y_dataset.apply_noise(amplitude_y)

    def apply_precision(self, precision_x=0, precision_y=0):
        if precision_x > 0:
            self.x_dataset.apply_precision(precision_x)
        if precision_y > 0 and self.y_dataset:
            self.y_dataset.apply_precision(precision_y)
    
    def time_shift(self, offset_x=0, offset_y=0):
        """Apply time shifts."""
        if hasattr(self.x_dataset, 'time_shift'):
            self.x_dataset.time_shift(offset_x)
        if self.y_dataset and hasattr(self.y_dataset, 'time_shift'):
            self.y_dataset.time_shift(offset_y)
        self._initialize_windows(None, None)
        self._build_windows(time_shift=offset_x)

    def set_window_size(self, window_size):
        """Change window size and rebuild windows."""
        self.window_manager.update_parameters(window_size=window_size)
        self._build_windows()


class PairedDataset(Dataset):
    """
    Dataset object for when both X/Y are given processor type of None. 
    Assumes user already preprocessed data as much as they want, so avoids windowing or any temporal features.
    """
    def __init__(self, x_dataset, y_dataset=None):
        """
        Parameters
        ----------
        x_dataset : StaticDataset
            Dataset for X variable
        y_dataset : StaticDataset, optional
            Dataset for Y variable
        """
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        # Ensure sizes match
        if y_dataset is not None:
            self._align_datasets()
        logger.info(f"Created PairedDataset")

    # Small properties for convenient access to main data
    @property
    def x_data(self):
        return self.x_dataset.data
    
    @property
    def y_data(self):
        return self.y_dataset.data
    
    def _align_datasets(self):
        """Ensure X and Y have matching number of windows."""
        n_x = len(self.x_dataset)
        n_y = len(self.y_dataset)
        if n_x != n_y:
            min_len = min(n_x, n_y)
            self.x_dataset.n_windows = min_len
            self.y_dataset.n_windows = min_len
            logger.warning(f"Truncated datasets to {min_len} windows for alignment")
    
    def __len__(self):
        return len(self.x_dataset)
    
    def __getitem__(self, idx):
        x_data = self.x_dataset[idx]
        y_data = self.y_dataset[idx] if self.y_dataset else None
        return x_data, y_data
    
    def apply_noise(self, amplitude_x=0, amplitude_y=0):
        """Apply noise to both datasets."""
        if amplitude_x > 0:
            self.x_dataset.apply_noise(amplitude_x)
        if amplitude_y > 0 and self.y_dataset:
            self.y_dataset.apply_noise(amplitude_y)

    def apply_precision(self, precision_x=0, precision_y=0):
        if precision_x > 0:
            self.x_dataset.apply_precision(precision_x)
        if precision_y > 0 and self.y_dataset:
            self.y_dataset.apply_precision(precision_y)



def create_single_dataset(data, time, proc_type, proc_params, device=None):
    """Create dataset for single variable."""
    if proc_type is None:
        return StaticDataset(data, device=device)
    if proc_type == 'continuous':
        return ContinuousWindowDataset(data, time, device=device)
    elif proc_type == 'spike':
        return SpikeWindowDataset(data, time, device=device)
    elif proc_type == 'categorical':
        return CategoricalWindowDataset(data, time, device=device)
    else:
        raise ValueError(f"Unknown processor type: {proc_type}")

def create_dataset(
        x_data, y_data=None,
        x_time=None, y_time=None,
        processor_type_x=None, processor_params_x=None,
        processor_type_y=None, processor_params_y=None,
        device=None
    ):
    """
    Factory function for creating appropriate dataset objects.
    """

    # Set up inputs
    proc_type_x = processor_type_x
    # Copy to avoid side-effects on original dict passed by reference
    proc_params_x = (processor_params_x or {}).copy()
    proc_type_y = processor_type_y if processor_type_y else processor_type_x
    proc_params_y = (processor_params_y or {}).copy() if processor_params_y else proc_params_x.copy()

    # Validation should be run before this function. Just quick check for pre-processed data
    if proc_type_x is None or proc_type_y is None:
        logger.warning(
            "Pre-processed data detected. Skipping compatibility check. "
            "Ensure X and Y have compatible representations."
        )
    
    # Initialize datasets
    x_dataset = create_single_dataset(x_data, x_time, proc_type_x, proc_params_x, device=device)
    y_dataset = create_single_dataset(y_data, y_time, proc_type_y, proc_params_y, device=device) if y_data is not None else None

    # Create and return paired dataset
    # If both types of data are temporal, create a paired temporal dataset
    if proc_type_x is not None or proc_type_y is not None:
        window_size = proc_params_x.pop('window_size', None)
        # Filter kwargs to only valid ones for PairedTemporalDataset
        valid_kwargs = inspect.signature(PairedTemporalDataset).parameters
        filtered_kwargs = {k: v for k, v in proc_params_x.items() if k in valid_kwargs}
        return PairedTemporalDataset(x_dataset, y_dataset, window_size=window_size, **filtered_kwargs)
    else:
        # Filter kwargs to only valid ones for PairedDataset
        valid_kwargs = inspect.signature(PairedDataset).parameters
        filtered_kwargs = {k: v for k, v in proc_params_x.items() if k in valid_kwargs}
        return PairedDataset(x_dataset, y_dataset, **filtered_kwargs)