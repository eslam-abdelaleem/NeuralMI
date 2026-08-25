# neural_mi/data/handler.py
import numpy as np
import inspect

from .temporal import (
    ContinuousWindowDataset, SpikeWindowDataset, BinnedSpikeDataset, CategoricalWindowDataset,
)
from .static import StaticDataset
from .shift_windowing import shift_family, mixed_pair_sample_rate_ok
from torch.utils.data import Dataset
from neural_mi.logger import logger


def _truncate_leading(data, min_len: int):
    """Truncate the leading (sample) dimension to ``min_len``, whether
    ``data`` is a plain tensor or a tuple of tensors sharing that dimension
    (``StaticDataset``'s compound "X-role" data for
    ``mode='conditional'(align='dual_branch')``). A naive ``data[:min_len]``
    on a raw tuple would slice the *outer* 2-element tuple instead of
    truncating each inner tensor -- this is the fix for that.
    """
    if isinstance(data, tuple):
        return tuple(d[:min_len] for d in data)
    return data[:min_len]


class WindowManager:
    """Centralized manager for creating and aligning temporal windows."""

    def __init__(self, window_size, t_start=None, t_end=None, step_size=None):
        self.window_size = window_size
        self.step_size = step_size  # None = non-overlapping (step = window_size)
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

    def update_parameters(self, window_size=None, t_start=None, t_end=None, step_size=None):
        """Update window parameters and regenerate windows."""
        if window_size is not None:
            self.window_size = window_size
        if step_size is not None:
            self.step_size = step_size
        if t_start is not None:
            self.t_start = t_start
        if t_end is not None:
            self.t_end = t_end

        if self.t_start is not None and self.t_end is not None:
            self.create_windows()
            self._notify_observers()

    def resolve_step(self):
        """Return the actual step size in time units.

        - None or step >= 1 : absolute step (None defaults to window_size).
        - 0 < step < 1      : fraction of window_size used as step.
        """
        if self.step_size is None:
            return self.window_size
        if 0 < self.step_size < 1:
            return self.step_size * self.window_size
        if self.step_size >= 1:
            return float(self.step_size)
        raise ValueError(f"step_size must be > 0, got {self.step_size}")

    def create_windows(self):
        """Create window times for a given temporal range."""
        if self.t_start is None or self.t_end is None:
            raise RuntimeError("t_start and t_end parameters need to be set to create windows")
        step = self.resolve_step()
        self.window_times = np.arange(self.t_start, self.t_end, step)
        self.n_windows = len(self.window_times)
        # Initialize all windows as valid - will be updated by datasets
        self.valid_windows = np.full(self.window_times.size, True, dtype=bool)

    def set_fixed_grid(self, t_start, n_windows):
        """Deterministically build exactly ``n_windows`` window starts
        beginning at ``t_start``, spaced by the resolved step.

        Used by the time-shift path instead of ``create_windows()``'s
        ``np.arange(t_start, t_end, step)`` so the window count stays
        exactly fixed across repeated shifts -- ``np.arange`` can drop or
        add a trailing window across float-step boundaries as ``t_start``
        drifts, which ``create_windows()`` never has to contend with since
        it's only ever called once per (fixed) t_start/t_end pair.
        """
        step = self.resolve_step()
        self.t_start = t_start
        self.window_times = t_start + np.arange(n_windows) * step
        self.n_windows = n_windows
        self.valid_windows = np.full(n_windows, True, dtype=bool)

    def __len__(self):
        """len() will return however many valid windows there are"""
        return self.n_windows


class PairedTemporalDataset(Dataset):
    """Wrapper for paired X and Y datasets with temporal alignment."""
    
    def __init__(self, x_dataset, y_dataset=None,
                 window_size=None,
                 t_start=None, t_end=None,
                 validate_windows=True,
                 step_size=None):
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
        step_size : float or int, optional
            Step between consecutive window starts.  ``None`` (default) gives
            non-overlapping windows (step = window_size).  Values in ``(0, 1)``
            are treated as a fraction of ``window_size`` (e.g. 0.5 → 50%
            overlap).  Values ≥ 1 are used as an absolute step in the same
            time units as ``window_size``.
        """
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.validate_windows = validate_windows

        # Create window manager shared between X and Y
        if window_size is None:
            raise ValueError("window_size must be provided")
        # Determine temporal extent and create windows
        self.window_manager = WindowManager(window_size, step_size=step_size)
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
        # Record the original recording bounds on the first call so they can
        # be used to clamp the range after time-shifts.
        if not hasattr(self, 'original_data_start'):
            self.original_data_start = data_start
            self.original_data_end = data_end
        # Clamp the shifted extent so it cannot exceed the original recording.
        data_start = max(data_start, self.original_data_start)
        data_end   = min(data_end,   self.original_data_end)
        # Apply user-specified bounds if provided
        final_start = t_start if t_start is not None else data_start
        final_end = t_end if t_end is not None else data_end
        if final_start >= final_end:
            raise ValueError(
                f"Invalid temporal range: t_start={final_start}, t_end={final_end}"
            )
        # Record the effective (user-bound-aware) range once, for time_shift's
        # fixed-grid margin reservation -- distinct from original_data_start/
        # _end above, which is the natural full extent and can be wider than
        # this when the caller passed an explicit t_start/t_end.
        if not hasattr(self, '_base_t_start'):
            self._base_t_start = final_start
            self._base_t_end = final_end
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
                raise ValueError("No valid windows after checking data coverage")
            self.window_manager.window_times = self.window_manager.window_times[combined_valid]
            self.window_manager.valid_windows = np.ones(self.window_manager.n_windows, dtype=bool)

            logger.info(
                f"Window coverage: {self.window_manager.n_windows}/{len(combined_valid)} "
                f"windows have sufficient data"
            )
        else:
            self.window_manager.n_windows = len(self.window_manager.window_times)
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
    
    def _reserve_shift_margin(self):
        """Compute (once, cached) the window count that stays fixed for
        every shift offset in ``[0, window_size)``.

        The window grid slides forward by up to a full ``window_size`` over
        fixed, unmutated raw data. Reserving ``2*window_size`` off the
        effective span (one ``window_size`` so the grid can start that far
        forward without any window running past ``_base_t_end``, and
        another so ``set_fixed_grid`` never needs to look earlier than
        ``_base_t_start``) keeps the same number of windows valid across the
        whole shift range -- the fixed-grid analogue of
        ``shift_windowing.safe_n_windows``'s margin for the raw-array-slicing
        case.
        """
        if not hasattr(self, '_shift_n_windows'):
            step = self.window_manager.resolve_step()
            window_size = self.window_manager.window_size
            usable_span = self._base_t_end - self._base_t_start
            margin = 2 * window_size
            n_safe = int((usable_span - margin) // step) + 1
            if n_safe < 1:
                raise ValueError(
                    f"Recording span ({usable_span:.6g}) is too short to reserve a "
                    f"safe time-shift margin ({margin:.6g}, i.e. 2*window_size) with "
                    f"window_size={window_size}. Reduce window_size, increase the "
                    f"recording length, or disable shift_time."
                )
            self._shift_n_windows = n_safe
        return self._shift_n_windows

    def time_shift(self, offset_x=0, offset_y=0):
        """Slide the (fixed-size) window grid forward by ``offset_x`` over
        the fixed, unmutated raw data.

        Does NOT rewrite any raw data -- unlike the child datasets' own
        ``time_shift()`` (still reachable directly for unit-level callers),
        which mutate their stored data and derive the grid's live extent
        from it, causing the offset to cancel out of every window-membership
        test. ``offset_y`` is accepted for backward compatibility with
        callers that pass both, but only ``offset_x`` drives the single grid
        shared by X and Y; production call sites always pass equal values.
        """
        if offset_y != offset_x:
            logger.warning(
                f"PairedTemporalDataset.time_shift got offset_x={offset_x} != "
                f"offset_y={offset_y}; only offset_x is used to shift the "
                f"single window grid shared by X and Y."
            )
        n_windows = self._reserve_shift_margin()
        self.window_manager.set_fixed_grid(self._base_t_start + offset_x, n_windows)
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
        logger.info("Created PairedDataset")

    # Small properties for convenient access to main data
    @property
    def x_data(self):
        return self.x_dataset.data
    
    @property
    def y_data(self):
        return self.y_dataset.data
    
    def _align_datasets(self):
        """Ensure X and Y have matching number of samples.

        Truncates the longer dataset's ``self.data`` tensor via a leading-
        dimension slice so that ``len(x_dataset) == len(y_dataset)``.
        ``StaticDataset.__len__`` reads ``self.data.shape[0]``, so the slice
        is required for the length change to take effect.
        """
        n_x = len(self.x_dataset)
        n_y = len(self.y_dataset)
        if n_x != n_y:
            min_len = min(n_x, n_y)
            max_len = max(n_x, n_y)
            lost = max_len - min_len
            pct = 100.0 * lost / max_len
            logger.warning(
                f"X and Y have different numbers of samples (X: {n_x}, Y: {n_y}). "
                f"Truncating both to {min_len} samples "
                f"({lost} samples discarded, {pct:.1f}% of the larger dataset lost)."
            )
            self.x_dataset.data = _truncate_leading(self.x_dataset.data, min_len)
            self.y_dataset.data = _truncate_leading(self.y_dataset.data, min_len)
            # Invalidate lazily-allocated data_master so it is re-cloned from
            # the truncated data the next time apply_noise/apply_precision runs.
            self.x_dataset.data_master = None
            self.y_dataset.data_master = None

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


def create_single_dataset(data, time, proc_type, proc_params, device=None, data_device='cpu'):
    """Create a dataset for a single variable.

    Parameters
    ----------
    data : array-like
        Raw data for this variable.
    time : array-like or None
        Time vector; required for temporal processors.
    proc_type : str or None
        Processor type: ``'continuous'``, ``'spike'``, ``'categorical'``, or
        ``None`` for pre-processed static data.
    proc_params : dict or None
        Processor-specific parameters extracted from ``processor_params_x/y``.
    device : str, optional
        Compute device (reference only).
    data_device : str, optional
        Device for storing dataset tensors.  Defaults to ``'cpu'``.  Pass
        ``'auto'`` to co-locate data with the compute device (useful for
        precision analysis where the same dataset is evaluated many times).
    """
    if proc_type is None:
        return StaticDataset(data, device=device, data_device=data_device)

    if proc_type == 'continuous':
        min_cov    = (proc_params or {}).get('min_coverage_fraction', 0.2)
        sample_rate = (proc_params or {}).get('sample_rate', None)
        return ContinuousWindowDataset(data, time, device=device,
                                       min_coverage_fraction=min_cov,
                                       data_device=data_device,
                                       sample_rate=sample_rate)

    elif proc_type == 'spike':
        bin_size = (proc_params or {}).get('bin_size', None)
        if bin_size is not None:
            normalize = (proc_params or {}).get('normalize_bins', True)
            return BinnedSpikeDataset(data, bin_size=bin_size, device=device,
                                      normalize=normalize, data_device=data_device)
        no_spike_val        = (proc_params or {}).get('no_spike_value', -1.0)
        excl_bursty         = (proc_params or {}).get('exclude_bursty_neurons', False)
        burst_mult          = (proc_params or {}).get('burst_threshold_multiplier', 5.0)
        max_spikes_per_win  = (proc_params or {}).get('max_spikes_per_window', None)
        n_seconds           = (proc_params or {}).get('n_seconds', None)
        return SpikeWindowDataset(data, time, device=device,
                                  no_spike_value=no_spike_val,
                                  exclude_bursty_neurons=excl_bursty,
                                  burst_threshold_multiplier=burst_mult,
                                  data_device=data_device,
                                  max_spikes_per_window=max_spikes_per_win,
                                  n_seconds=n_seconds)

    elif proc_type == 'categorical':
        min_cov     = (proc_params or {}).get('min_coverage_fraction', 0.2)
        encoding    = (proc_params or {}).get('encoding', 'majority_vote')
        sample_rate = (proc_params or {}).get('sample_rate', None)
        return CategoricalWindowDataset(data, time, device=device,
                                        min_coverage_fraction=min_cov,
                                        encoding=encoding,
                                        data_device=data_device,
                                        sample_rate=sample_rate)

    else:
        raise ValueError(f"Unknown processor type: '{proc_type}'.")

def create_dataset(
        x_data, y_data=None,
        x_time=None, y_time=None,
        processor_type_x=None, processor_params_x=None,
        processor_type_y=None, processor_params_y=None,
        device=None,
        data_device='cpu',
    ):
    """Factory function for creating appropriate dataset objects.

    Parameters
    ----------
    data_device : str, optional
        Device on which dataset tensors are stored.  ``'cpu'`` (default) keeps
        large arrays in pageable system RAM; ``'auto'`` co-locates data with the
        compute device.  See :func:`create_single_dataset` for details.
    """
    proc_type_x = processor_type_x
    proc_params_x = (processor_params_x or {}).copy()
    proc_type_y = processor_type_y if processor_type_y is not None else processor_type_x
    proc_params_y = (processor_params_y or {}).copy() if processor_params_y else proc_params_x.copy()

    if proc_type_x is None or proc_type_y is None:
        logger.debug(
            "Pre-processed data detected. Skipping compatibility check. "
            "Ensure X and Y have compatible representations."
        )

    x_dataset = create_single_dataset(x_data, x_time, proc_type_x, proc_params_x,
                                      device=device, data_device=data_device)
    y_dataset = create_single_dataset(y_data, y_time, proc_type_y, proc_params_y,
                                      device=device, data_device=data_device) if y_data is not None else None

    if proc_type_x is not None or proc_type_y is not None:
        # PairedTemporalDataset requires both sides to expose get_temporal_extent()
        # (i.e. both must actually be windowed via a real temporal processor); a
        # StaticDataset (proc_type=None, already pre-processed) has no notion of
        # time to align windows against, so pairing one with a windowed processor
        # on the other side can never work. Catch it here with an actionable
        # message instead of the get_temporal_extent AttributeError it would
        # otherwise raise deep inside PairedTemporalDataset._initialize_windows.
        if isinstance(x_dataset, StaticDataset) != isinstance(y_dataset, StaticDataset):
            _static_name, _windowed_name = (
                ('x_data', 'y_data') if isinstance(x_dataset, StaticDataset) else ('y_data', 'x_data')
            )
            raise ValueError(
                f"processor_type_x/y=None (pre-processed {_static_name}) cannot be paired "
                f"with a windowed processor_type on {_windowed_name} -- a pre-processed "
                f"variable has no time axis to align windows against. Either give "
                f"{_static_name} a real processor_type too (e.g. 'continuous'), or "
                f"pre-process both variables yourself and pass processor_type_x=None and "
                f"processor_type_y=None for both."
            )

        # 'spike' timestamps are always in seconds; 'continuous'/'categorical'
        # default to raw sample-index units unless 'sample_rate' is given.
        # X and Y share a single WindowManager (below), which combines their
        # get_temporal_extent() values via plain numeric min/max -- if the two
        # sides are in different units, that combination is meaningless (e.g.
        # a 20000-sample continuous recording with no sample_rate paired with
        # a 40-second spike recording silently windows only "sample 0 to 40"
        # of the continuous side, discarding 99.8% of it, with no error).
        # This is unrelated to shifting -- it affects any use of a mismatched
        # pair, so it's checked here rather than only when a shift is
        # requested (see shift_windowing.mixed_pair_sample_rate_ok, reused
        # here so "which pairs need a shared unit" is defined in one place).
        if (shift_family(proc_type_x, proc_type_y) == 'mixed'
                and not mixed_pair_sample_rate_ok(proc_type_x, proc_params_x, proc_type_y, proc_params_y)):
            _regular_side, _regular_type = (
                ('x', proc_type_x) if proc_type_x != 'spike' else ('y', proc_type_y)
            )
            logger.warning(
                f"processor_type mixes 'spike' with '{_regular_type}' on side "
                f"{_regular_side}, with no 'sample_rate' set on that side. 'spike' "
                f"timestamps are always in seconds; '{_regular_type}' defaults to raw "
                f"sample-index units without a sample_rate -- X and Y's window "
                f"boundaries may not correspond to the same real time, so window "
                f"alignment (and therefore this pairing's validity) may be "
                f"meaningless. Pass processor_params_{_regular_side}="
                f"{{'sample_rate': ...}} to put both sides on a shared time unit."
            )

        # X and Y share a single WindowManager, so they must use the same window_size.
        # Extract it from both param dicts; warn if the user supplied conflicting values.
        window_size   = proc_params_x.pop('window_size', None)
        y_window_size = proc_params_y.pop('window_size', None)
        if window_size is None:
            # X has no processor of its own (proc_type_x is None) -- nothing to
            # conflict with, so fall back to Y's window_size rather than always
            # preferring X's (which would stay None here and make windowing
            # impossible even though Y specified a valid window_size).
            window_size = y_window_size
        elif y_window_size is not None and y_window_size != window_size:
            logger.warning(
                f"processor_params_y specifies window_size={y_window_size}, but X and Y "
                f"share a single WindowManager and must use the same window size. "
                f"Using window_size={window_size} from processor_params_x."
            )
        step_size   = proc_params_x.pop('step_size', None)
        proc_params_y.pop('step_size', None)  # discard y-side; x-side controls step
        valid_kwargs = inspect.signature(PairedTemporalDataset).parameters
        filtered_kwargs = {k: v for k, v in proc_params_x.items() if k in valid_kwargs}
        return PairedTemporalDataset(x_dataset, y_dataset, window_size=window_size,
                                     step_size=step_size, **filtered_kwargs)
    else:
        valid_kwargs = inspect.signature(PairedDataset).parameters
        filtered_kwargs = {k: v for k, v in proc_params_x.items() if k in valid_kwargs}
        return PairedDataset(x_dataset, y_dataset, **filtered_kwargs)