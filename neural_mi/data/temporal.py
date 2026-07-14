# neural_mi/data/temporal.py
import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from neural_mi.utils import get_device
from neural_mi.logger import logger


def max_events_in_window(event_times: np.ndarray, window_size: float) -> int:
    """
    Find the maximum number of events that can fit in a window of given size.
    
    Parameters
    ----------
    event_times : np.array
        Array of event timestamps
    window_size : float
        Length of the time window in seconds/time units
        
    Returns
    -------
    int
        Maximum number of events that fit in any window of the given size. Returns 1 if
        no spikes are found to ensure a valid tensor shape.
    """
    times = np.array(event_times)    
    max_count = 0
    left = 0
    for right in range(len(times)):
        # Slide left pointer forward until window contains times[right] - window_size
        while times[right] - times[left] > window_size:
            left += 1
        # Update max_count if current window is larger
        max_count = max(max_count, right - left + 1)
    
    return max(1, max_count)


class TemporalWindowDataset(Dataset, ABC):
    """
    Base class for all temporal window datasets.

    Keeps 3 versions of main data
    1. data_orig: A numpy master matching the original (n_timepoints, n_channels), assigned in child classes
    2. data_master: A clean copy of data moved to windows (n_windows, n_timepoints_in_window, n_channels)
    3. data: A working copy of data moved to windows. 
    This is less memory-efficient but makes actions like applying noise repeatedly FAR faster
    """
    
    def __init__(self, window_manager=None, device=None, data_device='cpu'):
        """
        Parameters
        ----------
        window_manager : WindowManager, optional
            External window manager for alignment.
        device : str, optional
            Compute device used by the model.  Kept for reference; not used
            for data tensor storage.
        data_device : str, optional
            Device on which ``self.data`` tensors are stored.  Defaults to
            ``'cpu'`` so that large windowed arrays live in pageable system RAM
            and the OS can reclaim memory freely between training tasks.  Pass
            ``'auto'`` to co-locate data with the compute device.
        """
        self.window_manager = window_manager
        self.device = get_device() if not device else torch.device(str(device))
        if data_device == 'auto':
            self.data_device = self.device
        elif data_device is None or str(data_device) == 'cpu':
            self.data_device = torch.device('cpu')
        else:
            self.data_device = torch.device(str(data_device))
        self.data_master = None  # Allocated when moving to windows
        self.data = None         # Allocated when moving to windows
        self.time_offset = 0     # No offset by default

    # ------ Window handling
    def set_window_manager(self, window_manager):
        """Attach window manager and compute max samples/spikes per window."""
        self.window_manager = window_manager
        self.window_manager.register_observer(self)
        self._compute_max_samples_per_window()

    def _on_window_manager_updated(self):
        """Called when window manager parameters (window size, t_start, t_end) change.

        Only recomputes sizing here. Actually moving data to windows is left
        to the caller: PairedTemporalDataset._build_windows() always calls
        move_data_to_windows() itself right after update_parameters() (in
        __init__, time_shift(), and set_window_size()), and does so in the
        correct order relative to window-coverage validation. Calling it here
        too would just redo that work a second time.
        """
        self._compute_max_samples_per_window()
    
    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        pass
    
    @abstractmethod
    def move_data_to_windows(self):
        """Process raw data into windowed format."""
        pass

    @abstractmethod
    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage."""
        pass

    def remove_invalid_windows(self):
        """Trim data to only valid windows."""
        if self.window_manager is not None and self.data is not None:
            self.data = self.data[self.window_manager.valid_windows, :, :]
            self.data_master = self.data_master[self.window_manager.valid_windows, :, :]
    
    # ------ Size and indexing
    @abstractmethod
    def __getitem__(self, idx):
        """Return data at index."""
        pass
    
    def __len__(self):
        if self.window_manager:
            return self.window_manager.n_windows
        else:
            return 0
    
    @abstractmethod
    def get_temporal_extent(self):
        """Return (t_start, t_end) of the data."""
        pass
    
    # ------ Modifying data
    @abstractmethod
    def time_shift(self, time_offset):
        """Apply time shift to data."""
        pass

    @abstractmethod
    def apply_noise(self, amplitude):
        """Apply noise to data."""
        pass

    @abstractmethod
    def apply_precision(self, precision_level):
        """Round data to a specific resolution/precision level."""
        pass


class ContinuousWindowDataset(TemporalWindowDataset):
    """
    Dataset for continuous time series data.

    Note this class allows for irregular jumps in time between blocks of data,
    but assumes a constant sample rate.

    Notes
    -----
    **What ``min_coverage_fraction`` actually gates.** Coverage is measured purely
    by counting how many *source timestamps* (``self.time_vector`` entries) fall
    inside a window (see :meth:`validate_window_coverage`) — it does not inspect
    whether the corresponding data *values* are finite. A window whose timestamps
    are all present but whose values are ``NaN`` is **not** flagged invalid by this
    check; ``np.interp`` has no NaN-awareness and will happily interpolate through
    NaN source values, producing NaN output. If your data has present-but-invalid
    stretches, filter/impute them before construction.

    **What gets zero-padded vs. interpolated.** Within a window, every target time
    is filled by linear interpolation (:func:`numpy.interp`) against the *entire*
    ``time_vector``, regardless of how large any internal gap is — the coverage
    fraction only decides whether the resulting window is later flagged valid or
    invalid; it does not bound how much of the window content is
    interpolation-bridged. Only target times that fall entirely before the first
    or after the last timestamp in ``time_vector`` are zero-padded (see
    :meth:`move_data_to_windows`). A window that straddles a large mid-recording
    gap can therefore still be marked "valid" (if enough raw timestamps happen to
    fall within its span) while most of its content is interpolated across the
    gap. If your data has large mid-recording gaps, consider pre-collapsing them
    (build a reindexed timeline with the gap removed) or raising
    ``min_coverage_fraction`` so gappy windows are excluded outright.
    """
    
    def __init__(self, data, time_vector=None, window_manager=None, device=None,
                 min_coverage_fraction=0.2, data_device='cpu', sample_rate=None):
        """
        Parameters
        ----------
        data : array-like
            Continuous data of shape ``(n_timepoints, n_channels)``.  Arrays
            with additional trailing dimensions are flattened to
            ``(n_timepoints, n_channels * ...)``.
        time_vector : array-like, optional
            Timestamps for each sample.  If None, integer indices
            ``[0, 1, 2, ...]`` are used.
        window_manager : WindowManager, optional
            External window manager for temporal alignment with another dataset.
        device : str, optional
            Compute device (kept for reference; not used for data storage).
        min_coverage_fraction : float, optional
            Minimum fraction of a window that must be covered by actual data
            for the window to be considered valid.  Defaults to 0.2.  "Covered"
            means a count of source *timestamps* falling inside the window
            (see :meth:`validate_window_coverage`), not a check that the
            corresponding values are non-NaN, and it does not bound how much
            of a retained window's content is interpolation-bridged across
            internal gaps (see the class-level Notes and
            :meth:`move_data_to_windows`).
        data_device : str, optional
            Device for storing ``self.data``.  Defaults to ``'cpu'``.
        sample_rate : float, optional
            Explicit sample rate in Hz.  When provided, this overrides the
            inter-sample period inferred from ``time_vector``, which is useful
            when the time vector has floating-point rounding noise or when no
            time vector is supplied.
        """
        super().__init__(window_manager, device, data_device)

        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if data.ndim == 1:
            data = np.expand_dims(data, 1)  # (n_timepoints, 1)
        elif data.ndim >= 3:
            n_timepoints = data.shape[0]
            data = data.reshape(n_timepoints, -1)
        self.data_orig = data  # (n_timepoints, n_channels)

        if time_vector is not None:
            self.time_vector = np.asarray(time_vector)
        elif sample_rate is not None:
            # When sample_rate is given without a time vector, construct a
            # seconds-based time vector so that window_size is also in seconds.
            self.time_vector = np.arange(0, self.data_orig.shape[0]) / float(sample_rate)
        else:
            self.time_vector = np.arange(0, self.data_orig.shape[0])

        # Derive the inter-sample period from the time vector.
        diffs = np.diff(self.time_vector)
        self.period = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        if len(diffs) > 1:
            first_two = float(self.time_vector[1] - self.time_vector[0])
            if not np.isclose(first_two, self.period, rtol=0.05):
                from neural_mi.logger import logger as _logger
                _logger.warning(
                    f"ContinuousWindowDataset: first inter-sample interval "
                    f"({first_two:.6g}) differs from the median ({self.period:.6g}) by "
                    f">5%. Using the median. Check your time vector for boundary artifacts."
                )

        # An explicit sample_rate overrides the inferred period.
        if sample_rate is not None:
            self.period = 1.0 / sample_rate

        self.min_coverage_fraction = min_coverage_fraction

        if len(self.time_vector) != self.data_orig.shape[0]:
            raise ValueError(
                f"time_vector length ({len(self.time_vector)}) must match the number of "
                f"data time points ({self.data_orig.shape[0]})."
            )

        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None

    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        # Assuming fixed sample rate (outside of large jumps) for efficiency.
        # Adding a +1 buffer to avoid index out of bounds during interpolation at edges.
        self.max_samples_per_window = np.ceil(self.window_manager.window_size / self.period).astype(int) + 1
    
    def move_data_to_windows(self):
        """
        Convert continuous data into windows.
        Continuous data of shape (n_timepoints, n_channels) is reshaped and interpolated
        to (n_windows, n_channels, n_timepoints_in_window)

        Requires an attached window manager

        Notes
        -----
        Every target sample time within a window is filled via
        :func:`numpy.interp` against the full ``time_vector`` — interior gaps
        (target times that fall between two real timestamps, however far
        apart) are linearly interpolated across regardless of gap size. Only
        target times that fall entirely *before the first* or *after the
        last* timestamp in ``time_vector`` are zero-padded. Whether a given
        window is later kept or discarded is decided separately by
        :meth:`validate_window_coverage` and ``min_coverage_fraction``, which
        counts source timestamps per window and does not distinguish
        interpolated content from directly-sampled content.
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        # Preallocate output
        # max_samples_per_window is roughly window_size / period
        n_channels = self.data_orig.shape[1]
        data_shape = (self.window_manager.n_windows, n_channels, self.max_samples_per_window)
        data = np.full(data_shape, 0.0, dtype=np.float32)

        # Create target time grid for ALL windows: (n_windows, max_samples)
        # target_times[w, t] = window_start[w] + t * period
        time_offsets = np.arange(self.max_samples_per_window) * self.period

        # Use broadcasting to create (n_windows, max_samples) target times
        # Shape: (n_windows, 1) + (max_samples,) -> (n_windows, max_samples)
        target_times = self.window_manager.window_times[:, None] + time_offsets[None, :]

        # Flatten target times to 1D for efficient interpolation
        target_times_flat = target_times.ravel() # (n_windows * max_samples)

        # Mask for out-of-bounds (zero padding)
        # We check if target time is outside the range of actual data timestamps
        # Note: assuming time_vector is sorted
        t_min = self.time_vector[0]
        t_max = self.time_vector[-1]
        out_of_bounds = (target_times_flat < t_min) | (target_times_flat > t_max)
        n_oob = int(out_of_bounds.sum())
        total = len(target_times_flat)
        
        oob_frac = 0.0
        oob_windows = np.array([], dtype=int)
        if n_oob > 0:
            oob_frac = n_oob / total
            # Find which windows are affected
            oob_windows = np.unique(np.where(out_of_bounds.reshape(
                self.window_manager.n_windows, self.max_samples_per_window))[0])
            log_fn = logger.warning if oob_frac > 0.1 else logger.debug
            log_fn(
                f"ContinuousWindowDataset: {n_oob}/{total} interpolated time points "
                f"({oob_frac:.1%}) are zero-padded due to data gaps. "
                f"Affects {len(oob_windows)} window(s). "
                f"{'Consider checking your time vector for large gaps.' if oob_frac > 0.1 else ''}"
            )

        # U2: min_coverage_fraction only counts source *timestamps* per window
        # (see validate_window_coverage) -- it says nothing about whether an
        # in-bounds target time is close to a real sample or is being bridged
        # across a large internal gap by np.interp. Track that separately per
        # window here; remove_invalid_windows() below warns about *retained*
        # windows once the final valid set is known.
        if len(self.time_vector) >= 2:
            _LARGE_GAP_MULTIPLE = 3.0
            left_idx = np.clip(
                np.searchsorted(self.time_vector, target_times_flat, side='right') - 1,
                0, len(self.time_vector) - 2,
            )
            local_gap = self.time_vector[left_idx + 1] - self.time_vector[left_idx]
            bridges_large_gap = (~out_of_bounds) & (local_gap > _LARGE_GAP_MULTIPLE * self.period)
            self._interp_gap_fraction = bridges_large_gap.reshape(
                self.window_manager.n_windows, self.max_samples_per_window
            ).mean(axis=1)
        else:
            self._interp_gap_fraction = np.zeros(self.window_manager.n_windows)

        # Interpolate for each channel
        for i in range(n_channels):
            # np.interp expects 1D arrays
            # We interpolate the entire flattened grid at once
            interp_vals = np.interp(
                target_times_flat,
                self.time_vector,
                self.data_orig[:, i]
            ).astype(np.float32)
            # Apply zero padding
            interp_vals[out_of_bounds] = 0.0
            # Reshape back to (n_windows, max_samples) and assign
            data[:, i, :] = interp_vals.reshape(self.window_manager.n_windows, self.max_samples_per_window)

        self.data = torch.tensor(data, device=self.data_device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage.

        Notes
        -----
        "Coverage" here is a count of source **timestamps** from
        ``self.time_vector`` that fall within each window's
        ``[window_start, window_end)`` span (via :func:`numpy.searchsorted`),
        compared against ``min_coverage_fraction * max_samples_per_window``.
        It does *not* inspect the corresponding data values — a window whose
        timestamps are all present but whose values are ``NaN`` passes this
        check. It also does not measure how much of the window's
        *interpolated* content (see :meth:`move_data_to_windows`) spans an
        internal gap; a window can have enough raw timestamps to be marked
        valid while still being mostly gap-bridged by interpolation.
        """
        if self.window_manager is None:
            return None

        window_starts = self.window_manager.window_times
        window_ends = window_starts + self.window_manager.window_size

        # Find index in time_vector
        # Check if window actually contains data points
        start_inds = np.searchsorted(self.time_vector, window_starts, side='left')
        end_inds = np.searchsorted(self.time_vector, window_ends, side='left')

        # Window is valid if it contains at least one data point
        # Or we can be stricter.
        # Original logic was: "windows_with_spikes".
        # For continuous, we want windows that overlap with the time series.
        # But if there are gaps (jumps), start_inds and end_inds will point to the same index (if gap is larger than window).
        actual_counts = end_inds - start_inds
        # Minimum acceptable count based on coverage fraction
        min_count = int(np.ceil(self.min_coverage_fraction * self.max_samples_per_window))
        min_count = max(min_count, 1)  # always require at least 1 point

        valid = actual_counts >= min_count

        # Also enforce strict boundary containment if desired, but "data points check" handles gaps correctly.

        return valid

    def remove_invalid_windows(self):
        """Trim to valid windows, then warn if any *retained* window is mostly
        interpolated across an internal time_vector gap (see the gap-fraction
        computed in move_data_to_windows). min_coverage_fraction alone can't
        catch this -- it only counts raw timestamps per window, not how far
        those timestamps are from the target sample times."""
        _gap_frac = getattr(self, '_interp_gap_fraction', None)
        _valid_mask = self.window_manager.valid_windows if self.window_manager is not None else None
        super().remove_invalid_windows()
        if _gap_frac is not None and _valid_mask is not None and len(_gap_frac) == len(_valid_mask):
            _retained_frac = _gap_frac[_valid_mask]
            _flagged = _retained_frac > 0.3
            _n_flagged = int(_flagged.sum())
            if _n_flagged > 0:
                logger.warning(
                    f"ContinuousWindowDataset: {_n_flagged}/{len(_retained_frac)} retained "
                    f"window(s) have over 30% of their samples bridged by interpolation "
                    f"across a gap more than 3x the typical inter-sample period. These "
                    f"windows passed min_coverage_fraction (enough raw timestamps are "
                    f"nearby) but much of their content may be fabricated by np.interp. "
                    f"Consider raising min_coverage_fraction or pre-collapsing large gaps "
                    f"in your data."
                )

    def get_temporal_extent(self):
        return self.time_vector[0], self.time_vector[-1]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def reset(self):
        self.data = self.data_master.detach().clone()
        # Invalidate mask cache so apply_noise/apply_precision recompute after reset
        self.__dict__.pop('_data_mask', None)
        self.__dict__.pop('_noise_buffer', None)

    def time_shift(self, offset):
        self.time_vector = self.time_vector + offset - self.time_offset
        self.time_offset = offset

    def apply_noise(self, amplitude):
        if amplitude == 0.0:
            self.reset()
            return
        # Derive the mask from the master copy so repeated noise applications
        # at different amplitudes always start from the original non-zero positions.
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data_master, as_tuple=True)
        if not hasattr(self, '_noise_buffer') or len(self._noise_buffer) != len(self._data_mask[0]):
            self._noise_buffer = torch.empty(len(self._data_mask[0]), device=self.data.device, dtype=self.data.dtype)
        self._noise_buffer.normal_(mean=0, std=amplitude)
        self.data[self._data_mask] = self.data_master[self._data_mask] + self._noise_buffer

    def apply_precision(self, precision_level):
        if precision_level == 0.0:
            self.reset()
            return
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data, as_tuple=True)
        self.data[self._data_mask] = torch.round(self.data_master[self._data_mask] / precision_level) * precision_level



class SpikeWindowDataset(TemporalWindowDataset):
    """Dataset for spike train data."""
    
    def __init__(self, spike_times,
                 window_manager=None,
                 no_spike_value=-1.0, device=None,
                 exclude_bursty_neurons: bool = False,
                 burst_threshold_multiplier: float = 5.0,
                 data_device='cpu',
                 max_spikes_per_window=None,
                 n_seconds=None):
        """
        Parameters
        ----------
        spike_times : list of np.ndarray
            One array of spike timestamps per neuron/channel.
        window_manager : WindowManager, optional
            External window manager for temporal alignment.
        no_spike_value : float, optional
            Placeholder value for empty spike-time slots.  Defaults to -1.0.
        device : str, optional
            Compute device (kept for reference; not used for data storage).
        exclude_bursty_neurons : bool, optional
            If True, neurons with a peak spike count exceeding
            ``burst_threshold_multiplier`` times the median are excluded.
        burst_threshold_multiplier : float, optional
            Threshold multiplier for bursty-neuron exclusion.  Defaults to 5.0.
        data_device : str, optional
            Device for storing ``self.data``.  Defaults to ``'cpu'``.
        max_spikes_per_window : int, optional
            Hard cap on the number of spike-time slots allocated per window.
            Reduces memory use when a small number of neurons burst
            occasionally.  The data tensor is always large enough to hold at
            least one spike; any spikes beyond the cap within a window are
            silently dropped.
        n_seconds : float, optional
            Explicit recording duration in seconds.  When provided, this is
            used as the temporal end-point by ``get_temporal_extent()``
            instead of the last observed spike time.  Useful when the
            recording extends beyond the final spike.
        """
        super().__init__(window_manager, device, data_device)

        self.data_orig = [np.array(st) for st in spike_times]
        for i, st in enumerate(self.data_orig):
            if len(st) > 1 and not np.all(st[:-1] <= st[1:]):
                logger.debug(
                    f"SpikeWindowDataset: neuron {i} spike times are not sorted. "
                    "Sorting automatically."
                )
                self.data_orig[i] = np.sort(st)
        self.no_spike_value = no_spike_value
        self.exclude_bursty_neurons = exclude_bursty_neurons
        self.burst_threshold_multiplier = burst_threshold_multiplier
        self.time_offset = 0
        self._excluded_neurons = []
        self._max_spikes_cap = max_spikes_per_window
        self._n_seconds = n_seconds

        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None
    
    def _compute_max_samples_per_window(self):
        per_neuron_max = np.array([
            max_events_in_window(x, self.window_manager.window_size)
            for x in self.data_orig
        ])
        median_max = np.median(per_neuron_max)

        if self.exclude_bursty_neurons and median_max > 0:
            threshold = self.burst_threshold_multiplier * median_max
            bursty = np.where(per_neuron_max > threshold)[0]
            if len(bursty) > 0:
                self._excluded_neurons = bursty.tolist()
                logger.warning(
                    f"Excluding {len(bursty)} bursty neuron(s) "
                    f"(indices: {bursty.tolist()}) whose peak spike count "
                    f"exceeds {self.burst_threshold_multiplier}x the median "
                    f"({median_max:.1f}). Remaining neurons: "
                    f"{len(self.data_orig) - len(bursty)}."
                )
                # Filter data_orig in-place for this instance
                self.data_orig = [st for i, st in enumerate(self.data_orig)
                                if i not in set(bursty)]
                per_neuron_max = np.delete(per_neuron_max, bursty)

        self.max_samples_per_window = int(np.max(per_neuron_max)) if len(per_neuron_max) > 0 else 1

        # Apply user-specified cap on allocated spike slots.
        if self._max_spikes_cap is not None:
            capped = max(1, int(self._max_spikes_cap))
            if self.max_samples_per_window > capped:
                logger.warning(
                    f"SpikeWindowDataset: max_spikes_per_window cap applied "
                    f"({self.max_samples_per_window} → {capped} slots). "
                    "Spikes beyond the cap within any window will be dropped."
                )
            self.max_samples_per_window = min(self.max_samples_per_window, capped)

        if median_max > 0 and self.max_samples_per_window > self.burst_threshold_multiplier * median_max:
            worst_neuron = int(np.argmax(per_neuron_max))
            logger.warning(
                f"Spike tensor allocation dominated by burst: neuron {worst_neuron} "
                f"has peak {self.max_samples_per_window} spikes/window vs. median "
                f"{median_max:.1f}. Consider exclude_bursty_neurons=True."
            )

    def get_temporal_extent(self):
        """Return (t_start, t_end) of the spike data.

        When ``n_seconds`` was specified at construction, it is used as
        ``t_end`` instead of the time of the last observed spike, allowing
        the window manager to cover the full recording even if the recording
        extends beyond the final spike.
        """
        valid_trains = [st for st in self.data_orig if len(st) > 0]
        if not valid_trains:
            t_start = 0.0
        else:
            t_start = min(st[0] for st in valid_trains)

        if self._n_seconds is not None:
            t_end = float(self._n_seconds)
        elif valid_trains:
            t_end = max(st[-1] for st in valid_trains)
        else:
            t_end = 0.0

        return t_start, t_end
    
    def move_data_to_windows(self):
        """
        Convert spike times into windowed format.
        Spike data of form [(n_channels, n_spikes)] is reshaped to
        (n_windows, n_channels, n_timepoints_in_window).

        Supports both non-overlapping and overlapping windows (step_size < window_size).
        Each spike is assigned to every window whose time range contains it.

        Requires an attached window manager.
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")

        wt = self.window_manager.window_times   # sorted window start times
        ws = self.window_manager.window_size
        n_windows = self.window_manager.n_windows

        # Preallocate
        data_shape = (n_windows, len(self.data_orig), self.max_samples_per_window)
        data = np.full(data_shape, self.no_spike_value, dtype=np.float32)
        self._cached_window_inds = []

        # Two-pointer loop: O(n_spikes + n_windows) per neuron.
        # Works correctly for both overlapping and non-overlapping windows:
        # a spike belongs to window w iff  wt[w] <= spike_time < wt[w] + ws.
        for i in range(len(self.data_orig)):
            spikes = self.data_orig[i]  # pre-sorted

            # Restrict to spikes that could fall in any window
            if len(spikes) == 0 or n_windows == 0:
                self._cached_window_inds.append(np.array([], dtype=np.intp))
                continue

            spikes = spikes[(spikes >= wt[0]) & (spikes < wt[-1] + ws)]
            has_data = np.zeros(n_windows, dtype=bool)

            L = R = 0
            for w in range(n_windows):
                w_start = wt[w]
                w_end = w_start + ws
                # Advance L past spikes that ended before this window
                while L < len(spikes) and spikes[L] < w_start:
                    L += 1
                # Advance R to include all spikes in this window
                while R < len(spikes) and spikes[R] < w_end:
                    R += 1
                # spikes[L:R] all fall in [w_start, w_end)
                n_sp = min(R - L, self.max_samples_per_window)
                if n_sp > 0:
                    data[w, i, :n_sp] = spikes[L:L + n_sp] - w_start
                    has_data[w] = True

            self._cached_window_inds.append(np.where(has_data)[0])

        # Convert to tensor on data_device. Keep a master clone for noise/precision ops.
        self.data = torch.tensor(data, device=self.data_device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage. Assumes window manager attached."""
        windows_with_spikes = np.unique(np.concatenate(self._cached_window_inds)) \
            if any(len(w) > 0 for w in self._cached_window_inds) else np.array([], dtype=np.intp)
        valid = np.full(self.window_manager.window_times.shape, False, bool)
        valid[windows_with_spikes] = True
        return valid
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def reset(self):
        self.data = self.data_master.detach().clone()
        # Invalidate mask cache — data shape may have changed or noise was cleared
        self.__dict__.pop('_data_mask', None)
        self.__dict__.pop('_noise_buffer', None)

    def time_shift(self, offset):
        """Shift spike times by offset."""
        # Undo previous offset, apply new one
        self.data_orig = [st + offset - self.time_offset for st in self.data_orig]
        # Store this time offset to undo later
        self.time_offset = offset
        # Moving data to windows will be orchestrated by paired dataset class

    def apply_noise(self, amplitude):
        """Add uniform temporal jitter to spike times."""
        if amplitude == 0.0:
            self.reset()
            return
        # Derive the mask from the master copy so that repeated calls at different
        # amplitudes always start from the original (un-jittered) spike positions.
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data_master != self.no_spike_value, as_tuple=True)
        if not hasattr(self, '_noise_buffer') or len(self._noise_buffer) != len(self._data_mask[0]):
            self._noise_buffer = torch.empty(len(self._data_mask[0]), device=self.data.device, dtype=self.data.dtype)
        self._noise_buffer.uniform_(-amplitude / 2, amplitude / 2)
        self.data[self._data_mask] = self.data_master[self._data_mask] + self._noise_buffer

    def apply_precision(self, precision_level):
        """Round spike times to a specific resolution/precision level."""
        # Reset to master copy if zero. Avoids divide by zero, useful as interface to undo changes
        if precision_level == 0.0:
            self.reset()
            return
        # If data mask hasn't been created, compute that now
        if not hasattr(self, '_data_mask'):
            self._data_mask = torch.nonzero(self.data != self.no_spike_value, as_tuple=True)
        # Always round from data_master so repeated calls at different precision
        # levels each start from the original spike times (not re-rounded values).
        self.data[self._data_mask] = torch.round(self.data_master[self._data_mask] / precision_level) * precision_level


class BinnedSpikeDataset(TemporalWindowDataset):
    """Spike train dataset using binned firing rates instead of raw spike times.

    Parameters
    ----------
    spike_times : list of np.ndarray
        One array of spike times per neuron, in seconds relative to session start.
    bin_size : float
        Width of each firing rate bin in seconds. Smaller bins preserve more
        temporal precision at the cost of a higher-dimensional input to the critic.
    window_manager : WindowManager, optional
    device : str or torch.device, optional
    normalize : bool, optional
        If True, divide bin counts by bin_size to express as spikes/second.
        Default True. Set False to keep raw counts.
    """

    def __init__(self, spike_times, bin_size: float,
                 window_manager=None, device=None, normalize: bool = True,
                 data_device='cpu'):
        super().__init__(window_manager, device, data_device)
        self.data_orig = [np.array(st) for st in spike_times]
        self.bin_size = bin_size
        self.normalize = normalize
        self.time_offset = 0
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None

    def _compute_max_samples_per_window(self):
        """Number of bins per window = ceil(window_size / bin_size)."""
        self.max_samples_per_window = int(
            np.ceil(self.window_manager.window_size / self.bin_size)
        )

    def get_temporal_extent(self):
        valid = [st for st in self.data_orig if len(st) > 0]
        if not valid:
            return 0, 0
        return min(st[0] for st in valid), max(st[-1] for st in valid)

    def move_data_to_windows(self):
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")

        n_neurons = len(self.data_orig)
        n_windows = self.window_manager.n_windows
        n_bins = self.max_samples_per_window
        wt = self.window_manager.window_times  # (n_windows,)
        ws = self.window_manager.window_size

        data = np.zeros((n_windows, n_neurons, n_bins), dtype=np.float32)
        # Bin indices are assigned via rel_time / ws * n_bins, i.e. bins of
        # width ws/n_bins -- not necessarily self.bin_size, since n_bins =
        # ceil(ws/bin_size) rounds up whenever ws isn't an exact multiple of
        # bin_size. Normalizing by the requested bin_size in that case would
        # misreport spikes/second; normalize by the actual bin width used.
        actual_bin_width = ws / n_bins

        # Two-pointer loop: correctly handles overlapping windows.
        # Each spike contributes to every window whose range contains it.
        for i, spikes in enumerate(self.data_orig):
            if len(spikes) == 0 or n_windows == 0:
                continue

            spikes = spikes[(spikes >= wt[0]) & (spikes < wt[-1] + ws)]
            if len(spikes) == 0:
                continue

            L = R = 0
            for w in range(n_windows):
                w_start = wt[w]
                w_end = w_start + ws
                while L < len(spikes) and spikes[L] < w_start:
                    L += 1
                while R < len(spikes) and spikes[R] < w_end:
                    R += 1
                if R > L:
                    rel_times = spikes[L:R] - w_start
                    bin_idx = np.floor(rel_times / ws * n_bins).astype(np.int32)
                    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                    np.add.at(data[w, i, :], bin_idx, 1.0)

            if self.normalize:
                data[:, i, :] /= actual_bin_width

        self.data = torch.tensor(data, device=self.data_device)
        self.data_master = self.data.detach().clone()

    def validate_window_coverage(self):
        """Mark a window valid if at least one neuron fired in it."""
        # Use the filled data tensor: any window with a non-zero bin has data.
        # .cpu() first: self.data may live on an accelerator (dataset_device),
        # and .numpy() requires a CPU tensor.
        return (self.data.sum(dim=(1, 2)).cpu().numpy() > 0)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def reset(self):
        self.data = self.data_master.detach().clone()

    def time_shift(self, offset):
        self.data_orig = [st + offset - self.time_offset for st in self.data_orig]
        self.time_offset = offset

    def apply_noise(self, amplitude):
        """Add Uniform noise to non-zero bins (active bins only)."""
        if amplitude == 0.0:
            self.reset()
            return
        noise = torch.empty_like(self.data).uniform_(-amplitude / 2, amplitude / 2)
        # Only perturb bins that actually had spikes
        active = self.data_master > 0
        self.data = self.data_master.clone()
        self.data[active] = (self.data_master[active] + noise[active]).clamp(min=0)
  
    def apply_precision(self, precision_level):
        """Round bin values to the nearest multiple of precision_level."""
        if precision_level == 0.0:
            self.reset()
            return
        self.data = torch.round(self.data_master / precision_level) * precision_level


class CategoricalWindowDataset(TemporalWindowDataset):
    """Dataset for categorical time series data with one-hot encoding."""
    
    def __init__(self, data, time_vector=None, window_manager=None, device=None,
                 min_coverage_fraction=0.2, encoding: str = 'majority_vote',
                 data_device='cpu', sample_rate=None):
        """
        Parameters
        ----------
        data : array-like
            Categorical data of shape ``(n_timepoints, n_channels)`` with
            integer category labels.  Non-integer arrays are mapped to
            consecutive integers automatically.
        time_vector : array-like, optional
            Timestamps for each sample.  If None, integer indices are used.
        window_manager : WindowManager, optional
            External window manager for temporal alignment.
        device : str, optional
            Compute device (kept for reference; not used for data storage).
        min_coverage_fraction : float, optional
            Minimum fraction of a window covered by actual data for it to be
            considered valid.  Defaults to 0.2.
        encoding : {'majority_vote', 'probability', 'full_trajectory'}, optional
            Method for encoding categorical data within each window.
        data_device : str, optional
            Device for storing ``self.data``.  Defaults to ``'cpu'``.
        sample_rate : float, optional
            Explicit sample rate in Hz.  Overrides the inter-sample period
            inferred from ``time_vector`` when provided.
        """
        super().__init__(window_manager, device, data_device)

        arr = np.array(data)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 1)  # (n_timepoints, 1), mirrors ContinuousWindowDataset
        if not np.issubdtype(arr.dtype, np.integer):
            # return_inverse always returns a flattened array regardless of
            # input shape (both NumPy <2.0 and current behavior) -- reshape
            # back or multi-channel labels get silently collapsed to 1-D.
            _, indices = np.unique(arr, return_inverse=True)
            arr = indices.reshape(arr.shape)
        elif arr.size > 0 and arr.min() < 0:
            # Integer-typed input skips the relabeling above, so a negative
            # label would otherwise reach np.bincount downstream (via
            # n_categories = data.max() + 1) and raise an opaque error there.
            raise ValueError(
                f"CategoricalWindowDataset: integer-typed labels must be "
                f"non-negative (they are used directly as category indices "
                f"for np.bincount); got a minimum value of {arr.min()}. "
                f"Non-integer labels (e.g. strings or floats) are relabeled "
                f"to consecutive non-negative integers automatically -- if "
                f"these values are meant to be category codes, remap them "
                f"to [0, n_categories) first."
            )
        self.data_orig = np.asarray(arr, dtype=np.int32)

        if time_vector is not None:
            self.time_vector = np.asarray(time_vector)
        elif sample_rate is not None:
            # When sample_rate is given without a time vector, construct a
            # seconds-based time vector so that window_size is also in seconds.
            self.time_vector = np.arange(0, self.data_orig.shape[0]) / float(sample_rate)
        else:
            self.time_vector = np.arange(0, self.data_orig.shape[0])

        # Derive the inter-sample period from the time vector.
        diffs = np.diff(self.time_vector)
        self.period = float(np.median(diffs)) if len(diffs) > 0 else 1.0

        # An explicit sample_rate overrides the inferred period.
        if sample_rate is not None:
            self.period = 1.0 / sample_rate

        if len(self.time_vector) != self.data_orig.shape[0]:
            raise ValueError(
                f"time_vector must be same length as data. "
                f"Received {len(self.time_vector)} time points and {self.data_orig.shape[0]} data points"
            )
        
        # Total one-hot encoded dimensions
        self.encoding = encoding
        self.n_categories = self.data_orig.max() + 1
        # Warn if full_trajectory would produce a very wide tensor
        if encoding == 'full_trajectory':
            max_samples = int(np.ceil(
                (window_manager.window_size / self.period) if window_manager else 1
            ))
            enc_dim = self.n_categories * max_samples
            if enc_dim > 500:
                logger.warning(
                    f"CategoricalWindowDataset: encoding='full_trajectory' will produce "
                    f"a {enc_dim}-dimensional input per window "
                    f"({self.n_categories} categories × ~{max_samples} timesteps). "
                    f"Consider encoding='majority_vote' for large windows."
                )
        self.min_coverage_fraction = min_coverage_fraction

        # Process data if window manager is available
        if window_manager is not None:
            self.set_window_manager(window_manager)
            self.move_data_to_windows()
        else:
            self.window_manager = None
    
    def _compute_max_samples_per_window(self):
        """Compute maximum samples that fit in a window."""
        self.max_samples_per_window = np.ceil(self.window_manager.window_size / self.period).astype(int)
    
    def move_data_to_windows(self):
        """
        Convert categorical data into windows with one-hot encoding.
        Categorical data of shape (n_timepoints, n_channels) is reshaped and mapped to 
        (n_windows, n_channels, n_timepoints_in_window * n_categories)
        
        Requires an attached window manager.
        """
        if self.window_manager is None:
            raise RuntimeError("Cannot move data to windows: Window manager not initialized")
        
        logger.debug(f"CategoricalWindowDataset: using encoding='{self.encoding}'")
        
        if self.encoding == 'majority_vote':
            self._move_majority_vote()
        elif self.encoding == 'probability':
            self._move_probability()
        elif self.encoding == 'full_trajectory':
            self._move_full_trajectory()
        else:
            raise ValueError(
                f"Unknown encoding '{self.encoding}'. "
                f"Expected 'majority_vote', 'probability', or 'full_trajectory'."
            )
        self.data_master = self.data.detach().clone()

    def _move_majority_vote(self):
        """One-hot of the most common category per window. Shape: (n_windows, n_channels, n_categories)

        Uses a two-pointer sliding window so each time point is included in
        every window whose range [wt[w], wt[w]+window_size) it falls in.
        Supports overlapping windows (step_size < window_size).
        """
        n_channels = self.data_orig.shape[1]
        wt = self.window_manager.window_times
        ws = self.window_manager.window_size
        n_windows = self.window_manager.n_windows

        mask = (self.time_vector >= self.window_manager.t_start) & \
            (self.time_vector < self.window_manager.t_end + ws)
        times_in = self.time_vector[mask]
        data_in = self.data_orig[mask]

        result = np.zeros((n_windows, n_channels, self.n_categories), dtype=np.float32)
        window_counts = np.zeros(n_windows, dtype=int)

        L = R = 0
        for w in range(n_windows):
            w_start = wt[w]
            w_end = w_start + ws
            while L < len(times_in) and times_in[L] < w_start:
                L += 1
            while R < len(times_in) and times_in[R] < w_end:
                R += 1
            if R <= L:
                continue
            rows = data_in[L:R]
            window_counts[w] = R - L
            for c in range(n_channels):
                counts = np.bincount(rows[:, c], minlength=self.n_categories)
                result[w, c, np.argmax(counts)] = 1.0

        self._window_counts = window_counts
        self.data = torch.tensor(result, device=self.data_device)

    def _move_probability(self):
        """Empirical category probabilities per window. Shape: (n_windows, n_channels, n_categories)

        Uses a two-pointer sliding window so each time point is included in
        every window whose range [wt[w], wt[w]+window_size) it falls in.
        Supports overlapping windows (step_size < window_size).
        """
        n_channels = self.data_orig.shape[1]
        wt = self.window_manager.window_times
        ws = self.window_manager.window_size
        n_windows = self.window_manager.n_windows

        mask = (self.time_vector >= self.window_manager.t_start) & \
            (self.time_vector < self.window_manager.t_end + ws)
        times_in = self.time_vector[mask]
        data_in = self.data_orig[mask]

        result = np.zeros((n_windows, n_channels, self.n_categories), dtype=np.float32)
        window_counts = np.zeros(n_windows, dtype=int)

        L = R = 0
        for w in range(n_windows):
            w_start = wt[w]
            w_end = w_start + ws
            while L < len(times_in) and times_in[L] < w_start:
                L += 1
            while R < len(times_in) and times_in[R] < w_end:
                R += 1
            if R <= L:
                continue
            rows = data_in[L:R]
            window_counts[w] = R - L
            for c in range(n_channels):
                counts = np.bincount(rows[:, c], minlength=self.n_categories).astype(np.float32)
                total = counts.sum()
                if total > 0:
                    result[w, c, :] = counts / total

        self._window_counts = window_counts
        self.data = torch.tensor(result, device=self.data_device)

    def _move_full_trajectory(self):
        """Full one-hot trajectory per window. Shape: (n_windows, n_channels, n_categories * max_samples)

        Uses a two-pointer sliding window so each time point is included in
        every window whose range [wt[w], wt[w]+window_size) it falls in.
        Supports overlapping windows (step_size < window_size).
        """
        n_channels = self.data_orig.shape[1]
        wt = self.window_manager.window_times
        ws = self.window_manager.window_size
        n_windows = self.window_manager.n_windows
        data_shape = (n_windows, n_channels, self.n_categories * self.max_samples_per_window)
        data = np.zeros(data_shape, dtype=np.float32)

        mask = (self.time_vector >= self.window_manager.t_start) & \
            (self.time_vector < self.window_manager.t_end + ws)
        times_in = self.time_vector[mask]
        data_in = self.data_orig[mask]

        window_counts = np.zeros(n_windows, dtype=int)
        L = R = 0
        for w in range(n_windows):
            w_start = wt[w]
            w_end = w_start + ws
            while L < len(times_in) and times_in[L] < w_start:
                L += 1
            while R < len(times_in) and times_in[R] < w_end:
                R += 1
            n_pts = min(R - L, self.max_samples_per_window)
            if n_pts == 0:
                continue
            window_counts[w] = R - L
            # Vectorised one-hot placement: position p → columns [p*K .. p*K+K)
            slice_data = data_in[L:L + n_pts, :]        # (n_pts, n_channels)
            pos_offsets = np.arange(n_pts) * self.n_categories  # (n_pts,)
            col_indices = pos_offsets[:, None] + slice_data      # (n_pts, n_channels)
            for c in range(n_channels):
                data[w, c, col_indices[:, c]] = 1.0

        self._window_counts = window_counts
        self.data = torch.tensor(data, device=self.data_device)
        # data_master is set by move_data_to_windows() after this returns.

    def validate_window_coverage(self):
        """Check which windows have sufficient data coverage."""
        if self.window_manager is None:
            return None
        max_samples = self.max_samples_per_window
        min_count = max(1, int(np.ceil(self.min_coverage_fraction * max_samples)))
        return self._window_counts >= min_count
    
    def time_shift(self, offset):
        """Apply time shift to data."""
        # Undo previous offset, apply new one
        self.time_vector = self.time_vector + offset - self.time_offset
        # Store this time offset to undo later
        self.time_offset = offset
        # Moving data to windows on new time shift will be orchestrated by paired dataset

    def get_temporal_extent(self):
        """Return temporal extent of the data."""
        return self.time_vector[0], self.time_vector[-1]
    
    def __getitem__(self, idx):
        """Return one-hot encoded windowed data at index."""
        return self.data[idx]
    
    def reset(self):
        """Undo any added noise by resetting to original data. Does not undo time shifts."""
        self.data = self.data_master.detach().clone()

    def apply_noise(self, amplitude):
        """
        Noise corruption is not defined for categorical data.

        Raises
        ------
        NotImplementedError
            Always. Apply augmentation to the underlying signal before encoding.
        """
        raise NotImplementedError(
            "Noise corruption is not defined for categorical data. "
            "Apply augmentation to the underlying signal before encoding."
        )

    def apply_precision(self, precision_level):
        """
        Precision corruption is not defined for categorical data.

        Raises
        ------
        NotImplementedError
            Always. Apply augmentation to the underlying signal before encoding.
        """
        raise NotImplementedError(
            "Precision corruption is not defined for categorical data. "
            "Apply augmentation to the underlying signal before encoding."
        )


