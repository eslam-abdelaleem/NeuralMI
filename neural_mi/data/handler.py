# neural_mi/data/handler.py
import torch
import numpy as np

from .datasets import *
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
        """Notify all registered datasets of window changes."""
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
        self.valid_windows = np.full(self.window_times.size, True, dtype=bool)
    
    def invalidate_windows(self, mask):
        """Mark certain windows as invalid."""
        self.valid_windows = np.logical_and(self.valid_windows, mask)
        self.n_windows = int(self.valid_windows.sum())


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
        # Using separate object allows X and Y to always be synchronized
        if window_size is None:
            raise ValueError("window_size must be provided")
        # Determine temporal extent and create windows
        self.window_manager = WindowManager(window_size)
        self._initialize_windows(t_start, t_end)
        # Attach window manager to datasets
        self.x_dataset.set_window_manager(self.window_manager)
        if self.y_dataset is not None:
            self.y_dataset.set_window_manager(self.window_manager)
        # Ensure alignment
        if y_dataset is not None:
            self._align_datasets()
        # Process datasets now that they have window managers
        if hasattr(self.x_dataset, '_move_data_to_windows'):
            self.x_dataset._move_data_to_windows()
        if self.y_dataset is not None:
            if hasattr(self.y_dataset, '_move_data_to_windows'):
                self.y_dataset._move_data_to_windows()
        
        logger.info(f"Created {self.window_manager.n_windows} aligned windows")
    
    def _align_datasets(self):
        """Ensure X and Y have matching number of windows."""
        n_x = len(self.x_dataset)
        n_y = len(self.y_dataset)
        if n_x != n_y:
            min_len = min(n_x, n_y)
            self.x_dataset.n_windows = min_len
            self.y_dataset.n_windows = min_len
            logger.warning(f"Truncated datasets to {min_len} windows for alignment")
            # TODO: Actually make this truncation/handling happen
    
    def _initialize_windows(self, t_start, t_end):
        """Create windows based on data extent and validate coverage."""
        # Get temporal extent from both datasets
        x_start, x_end = self.x_dataset._get_temporal_extent()
        if self.y_dataset is not None:
            y_start, y_end = self.y_dataset._get_temporal_extent()
            data_start, data_end = max(x_start, y_start), min(x_end, y_end)
        else:
            data_start, data_end = x_start, x_end
        # Apply user-specified bounds if provided
        final_start = t_start if t_start is not None else data_start
        final_end = t_end if t_end is not None else data_end
        if final_start >= final_end:
            raise ValueError(
                f"Invalid temporal range: t_start={final_start}, t_end={final_end}"
            )
        # Create windows
        self.window_manager.update_parameters(final_start, final_end)
        if len(self.window_manager.window_times) == 0:
            raise ValueError(
                f"No windows could be created in range [{final_start}, {final_end}] "
                f"with window_size={self.window_manager.window_size}"
            )
        # Validate window coverage for each dataset (optional)
        if self.validate_windows:
            x_valid = self.x_dataset._validate_window_coverage()
            if self.y_dataset is not None:
                y_valid = self.y_dataset._validate_window_coverage()
                # Only keep windows valid for BOTH datasets
                self.window_manager.valid_windows = np.logical_and(x_valid, y_valid)
            else:
                self.window_manager.valid_windows = x_valid
            self.window_manager.n_windows = int(self.window_manager.valid_windows.sum())
            if self.window_manager.n_windows == 0:
                raise ValueError("No valid windows after checking data coverage")
            logger.info(
                f"Window coverage: {self.window_manager.n_windows}/{len(self.window_manager.window_times)} "
                f"windows have sufficient data"
            )
    
    def __len__(self):
        return len(self.x_dataset)
    
    def __getitem__(self, idx):
        x_data = self.x_dataset[idx]
        y_data = self.y_dataset[idx] if self.y_dataset else None
        return x_data, y_data
    
    def add_noise(self, amplitude_x=0, amplitude_y=0):
        """Add noise to both datasets."""
        if amplitude_x > 0:
            self.x_dataset.add_noise(amplitude_x)
        if amplitude_y > 0 and self.y_dataset:
            self.y_dataset.add_noise(amplitude_y)
    
    def time_shift(self, offset_x=0, offset_y=0):
        """Apply time shifts."""
        if offset_x != 0 and hasattr(self.x_dataset, 'time_shift'):
            self.x_dataset.time_shift(offset_x)
        if offset_y != 0 and self.y_dataset and hasattr(self.y_dataset, 'time_shift'):
            self.y_dataset.time_shift(offset_y)

    def set_window_size(self, window_size):
        """Change window size, update windows of data."""
        self.window_manager.update_parameters(window_size=window_size)
        # Datasets are automatically notified and updated via observer pattern



class PairedDataset(Dataset):
    """
    Dataset object for when both X/Y are given processor type of None. 
    Assumes user already preprocessed data as much as they want, so avoids windowing or any temporal features."""
    def __init__(self, x_dataset, y_dataset=None):
        """
        Parameters
        ----------
        x_dataset : PreprocessedDataset
            Dataset for X variable
        y_dataset : PreprocessedDataset, optional
            Dataset for Y variable
        """
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        # Ensure sizes match
        if y_dataset is not None:
            self._align_datasets()
        logger.info(f"Created PairedDataset")
    
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
    
    def add_noise(self, amplitude_x=0, amplitude_y=0):
        """Add noise to both datasets."""
        if amplitude_x > 0:
            self.x_dataset.add_noise(amplitude_x)
        if amplitude_y > 0 and self.y_dataset:
            self.y_dataset.add_noise(amplitude_y)
    
    def time_shift(self, offset_x=0, offset_y=0):
        """Apply time shifts."""
        if offset_x != 0 and hasattr(self.x_dataset, 'time_shift'):
            self.x_dataset.time_shift(offset_x)
        if offset_y != 0 and self.y_dataset and hasattr(self.y_dataset, 'time_shift'):
            self.y_dataset.time_shift(offset_y)




class DataHandler:
    """Factory for creating appropriate dataset objects."""

    INCOMPATIBLE_PAIRS = [
        {} # Right now no incompatible pairs
    ]
    
    def __init__(self, 
                 x_data, y_data=None,
                 x_time=None, y_time=None,
                 processor_type_x=None, processor_params_x=None,
                 processor_type_y=None, processor_params_y=None):
        """
        Parameters
        ----------
        x_data : Union[np.ndarray, torch.Tensor, list]
            The raw input data for variable X.
        y_data : Union[np.ndarray, torch.Tensor, list]
            The raw input data for variable Y.
        x_time, y_time : Union[np.ndarray, torch.Tensor], optional
            Vector of times variable X/Y was sampled at if processor type is continuous, must be the same length as X/Y
        processor_type_x, processor_type_y : {'continuous', 'spike', 'categorical'}, optional
            The type of processor to use. If `None`, the data is assumed to be
            already processed and will be returned as-is after a shape check.
            Defaults to None.
        processor_params_x, processor_params_y : Dict[str, Any], optional
            A dictionary of parameters to pass to the selected processor's
            initializer. Defaults to None.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.x_time = x_time
        self.y_time = y_time
        self.proc_type_x = processor_type_x
        self.proc_params_x = processor_params_x or {}
        self.proc_type_y = processor_type_y if processor_type_y else processor_type_x
        self.proc_params_y = processor_params_y if processor_params_y else self.proc_params_x.copy()
        # Validate combination
        self._validate_combination()
        # Initialize datasets
        x_dataset = self._create_single_dataset(self.x_data, self.x_time, self.proc_type_x, self.proc_params_x)
        y_dataset = None
        if self.y_data is not None:
            y_dataset = self._create_single_dataset(self.y_data, self.y_time, self.proc_type_y, self.proc_params_y)
        # If both types of data are temporal, create a paired temporal dataset
        if processor_type_x is not None and processor_type_y is not None:
            return PairedTemporalDataset(x_dataset, y_dataset)
        else:
            return PairedDataset(x_dataset, y_dataset)

    def _validate_combination(self):
        """Check if X and Y data types are compatible"""
        if self.y_data is None:
            return # Single variable is always OK
        if self.proc_type_x is None or self.proc_type_y is None:
            # If either is pre-processed, assume user knows what they're doing
            logger.warning(
                "Pre-processed data detected. Skipping compatibility check. "
                "Ensure X and Y have compatible temporal representations."
            )
            return
        pair = {self.proc_type_x, self.proc_type_y}
        if any([pair == x for x in self.INCOMPATIBLE_PAIRS]):
            raise ValueError(
                f"Incompatible data type combination: X is '{self.proc_type_x}' "
                f"and Y is '{self.proc_type_y}'. Categorical data cannot be paired "
                f"with spike or continuous data because they use different temporal "
                f"representations.\n\n"
                f"Currently all combinations of 'spike', 'continuous', or 'categorical' are compatible. "
            )
    
    def _create_single_dataset(self, data, time, proc_type):
        """Create dataset for single variable."""
        if proc_type is None:
            # Already processed
            return PreprocessedDataset(data)
        if proc_type == 'continuous':
            return ContinuousWindowDataset(data, time)
        elif proc_type == 'spike':
            return SpikeWindowDataset(data, time)
        elif proc_type == 'categorical':
            return CategoricalWindowDataset(data)
        else:
            raise ValueError(f"Unknown processor type: {proc_type}")