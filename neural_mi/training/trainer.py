# neural_mi/training/trainer.py
"""Handles the training and evaluation of critic models for MI estimation.

This module provides the `Trainer` class, a comprehensive utility for training
a critic model, monitoring its performance, implementing early stopping, and
saving the best-performing model state.
"""
import warnings
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, Optional, List, Callable, Union
import torch.nn as nn
import copy
import contextlib

from neural_mi.data import (PairedDataset, PairedTemporalDataset, SubsetView,
                            AlignedStreams)
from neural_mi.logger import logger, user_stacklevel
from neural_mi.exceptions import TrainingError
from neural_mi.utils import (compute_cross_covariance_spectrum, compute_spectral_metrics,
                             compute_cross_covariance_rotation, warn_if_blocked_split_leaks)
from neural_mi.augmentations import apply_augmentations

# Fraction of the (smoothed) test-MI history that must sit above
# TEST_TRACE_SATURATION_THRESHOLD * ceiling before peak-epoch selection is
# flagged as unreliable. A judgement call, kept as a module constant so it can
# be tuned in one place.
TEST_TRACE_SATURATED_EPOCH_FRACTION_THRESHOLD = 0.5
TEST_TRACE_SATURATION_THRESHOLD = 0.9
CEILING_PROXIMITY_WARNING_THRESHOLD = 0.85

def _ranks(sample: np.ndarray) -> List[int]:
    """Return each element's rank (position in sorted order) within `sample`."""
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def _sample_with_minimum_distance(n: int, k: int, d: int, rng=np.random) -> np.ndarray:
    """Sample k block start positions in [0, n) with pairwise distance >= d, drawn from rng.

    Draws k values from a compressed range of size n - (k-1)*(d-1), then
    spreads them out by each value's rank so consecutive picks are always
    at least d apart. This is what keeps the blocked test-split's blocks
    from overlapping or sitting adjacent to each other.
    """
    sample = rng.choice(n - (k - 1) * (d - 1), k, replace=False)
    return np.array([s + (d - 1) * r for s, r in zip(sample, _ranks(sample))])


def _blocks_of(n_rows: int, max_rows: int) -> Tuple[int, int]:
    """Blocks of at most max_rows that n_rows are evaluated in, and the rows of the largest."""
    n_blocks = -(-n_rows // max_rows)
    return n_blocks, -(-n_rows // n_blocks)


def _rng_of(trainer):
    """The generator a trainer draws its split and evaluation rows from.

    Some callers use the split helpers on a bare instance or as plain functions
    (``Trainer._create_blocked_split(None, ...)``); they get NumPy's global state.
    """
    return getattr(trainer, '_rng', np.random)


def _batch_size_of(x) -> int:
    """Leading (sample) dim of x, whether x is a plain tensor or a tuple
    (DualBranchEmbedding's compound "X-role" data, mode='conditional'
    (align='dual_branch')) sharing that dimension."""
    return (x[0] if isinstance(x, tuple) else x).shape[0]


def _to_device(x, device):
    """``.to(device)``, mapped over each element if x is a tuple."""
    if isinstance(x, tuple):
        return tuple(t.to(device) for t in x)
    return x.to(device)


def _index_batch(x, idx):
    """Index the leading (sample) dim of x by idx (an int array/tensor or a
    slice), mapped over each element if x is a tuple."""
    if isinstance(x, tuple):
        return tuple(t[idx] for t in x)
    return x[idx]


def _detach_clone(x):
    """``.detach().clone()``, mapped over each element if x is a tuple
    (dual_branch's compound "X-role" data)."""
    if isinstance(x, tuple):
        return tuple(t.detach().clone() for t in x)
    return x.detach().clone()


class Trainer:
    """Manages the training loop for a critic model.

    The Trainer class encapsulates the logic for training a critic model to
    estimate mutual information. It handles data splitting, batching, epoch
    iteration, loss calculation, backpropagation, and model evaluation.

    It also includes features like early stopping based on a smoothed validation
    MI score, and the ability to save the best model checkpoint.
    """
    def __init__(self, model: nn.Module, estimator_fn: Callable, optimizer: torch.optim.Optimizer,
                 device: torch.device, use_variational: bool = False, beta: float = 1024,
                 estimator_params: Optional[Dict[str, Any]] = None,
                 custom_smoothing_fn: Optional[Callable] = None,
                 whitening: Optional[str] = 'std',
                 gradient_clip_val: Optional[float] = None,
                 decoder_x: Optional[nn.Module] = None,
                 decoder_y: Optional[nn.Module] = None,
                 decoder_lambda_x: float = 0.001,
                 decoder_lambda_y: float = 0.001,
                 decoder_output_activation_x: str = 'linear',
                 decoder_output_activation_y: str = 'linear',
                 use_amp: Union[bool, str] = 'auto',
                 augmentation_params_x: Optional[Dict[str, Any]] = None,
                 augmentation_params_y: Optional[Dict[str, Any]] = None):

        # The split, the evaluation rows and their subsampling are drawn from
        # self._rng: NumPy's global state, or a generator of its own once
        # train() is given a split_seed.
        self._rng = np.random
        """
        Parameters
        ----------
        model : nn.Module
            The critic model to be trained.
        estimator_fn : Callable
            A function that takes the critic's score matrix and returns a scalar
            MI estimate.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training (e.g., Adam).
        device : torch.device
            The device (CPU or GPU) on which to perform training.
        use_variational : bool, optional
            If True, the trainer will expect the model to return a KL divergence
            loss term, which will be incorporated into the total loss.
            Defaults to False.
        beta : float, optional
            Weight applied to the MI term in the variational loss:
            ``L = KL - beta * MI``.  A large ``beta`` (default 1024.0) causes
            MI maximisation to dominate while the KL term acts as a mild
            regulariser on the embedding distributions.  Decrease ``beta``
            to increase the influence of the KL prior relative to MI.
            Only used when ``use_variational=True``.  Defaults to 1024.0.
        estimator_params : dict, optional
            Additional keyword arguments for the estimator function.
        custom_smoothing_fn : Callable, optional
            A custom function for smoothing the validation MI history, which takes
            a list of MI values and returns a smoothed array. If not provided, a default Gaussian + median filter will be used.
        whitening : {'std', 'zca', None}, optional
            Normalization of the embeddings before the cross-covariance SVD behind
            both the spectral metrics and the rotated embeddings (see
            :func:`neural_mi.utils.cross_covariance_svd`). Defaults to 'std'.
        gradient_clip_val : float, optional
            If set, applies ``torch.nn.utils.clip_grad_norm_`` with this value as
            the maximum gradient norm after each backward pass, before the
            optimiser step.  Helps prevent gradient explosions with high learning
            rates or difficult distributions.  ``None`` disables clipping.
        decoder_x : nn.Module, optional
            Decoder module that reconstructs X from the X-embedding Z_X.
            When provided, a reconstruction term enters the objective inside the
            bottleneck bracket. Defaults to ``None`` (no decoder).
        decoder_y : nn.Module, optional
            Decoder module for Y. Defaults to ``None``.
        decoder_lambda_x : float, optional
            What reconstructing X is worth *relative to the MI term*, so the
            coefficient it actually contributes with is ``beta * decoder_lambda_x``
            when ``use_variational`` and ``decoder_lambda_x`` otherwise. Defaults
            to 0.001, so at the default ``beta`` of 1024 a reconstruction enters
            the loss with an absolute coefficient near 1.
        decoder_lambda_y : float, optional
            The same for Y. Defaults to 0.001.
        decoder_output_activation_x : str, optional
            Output activation of decoder_x: ``'linear'``, ``'sigmoid'``, or ``'softmax'``.
            When ``'softmax'``, NLL loss (equivalent to cross-entropy) is used instead of MSE.
            Defaults to ``'linear'``.
        decoder_output_activation_y : str, optional
            Output activation of decoder_y.  Same options as above.  Defaults to ``'linear'``.
        use_amp : bool or str, optional
            Mixed-precision (AMP) training.  ``'auto'`` (default) enables AMP when
            ``device.type == 'cuda'`` and is a no-op on CPU/MPS.  ``True`` enables
            explicitly (CUDA only; silently ignored on other devices).  ``False``
            disables entirely.
       """
        self.device, self.model = device, model.to(device)
        self.estimator_fn, self.optimizer = estimator_fn, optimizer
        self.use_variational, self.beta = use_variational, beta
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.custom_smoothing_fn = custom_smoothing_fn
        self.whitening = whitening
        self.gradient_clip_val = gradient_clip_val
        self.decoder_x = decoder_x.to(device) if decoder_x is not None else None
        self.decoder_y = decoder_y.to(device) if decoder_y is not None else None
        self.decoder_lambda_x = decoder_lambda_x if decoder_lambda_x is not None else 0.001
        self.decoder_lambda_y = decoder_lambda_y if decoder_lambda_y is not None else 0.001
        self.decoder_output_activation_x = decoder_output_activation_x or 'linear'
        self.decoder_output_activation_y = decoder_output_activation_y or 'linear'
        self.use_amp = use_amp
        self.aug_params_x = augmentation_params_x or {}
        self.aug_params_y = augmentation_params_y or {}

    def train(self, dataset: Union[PairedDataset, PairedTemporalDataset], n_epochs: int, batch_size: int,
              train_fraction: float = 0.9, n_test_blocks: int = 5,
              shift_time: bool = False,
              shift_windows: bool = False,
              shift_seed: Optional[int] = None,
              split_seed: Optional[int] = None,
              patience: int = 10, smoothing_sigma: float = 1.0, median_window: int = 5,
              min_improvement: float = 0.001,
              save_best_model_path: Optional[str] = None,
              output_units: str = 'nats', verbose: bool = True, show_progress: bool = True,
              split_mode: str = 'blocked',
              train_indices: Optional[np.ndarray] = None,
              test_indices: Optional[np.ndarray] = None,
              max_eval_samples: int = 5000,
              train_subset_size: Optional[Union[int, str]] = None,
              test_subset_size: Optional[Union[int, str]] = None,
              split_gap_fraction: float = 0.5,
              track_spectral_history: bool = False,
              max_index_reduction: float = 0.05,
              eval_train: Union[bool, float, int] = False,
              peak_fraction: float = 1.0,
              scheduler: Optional[Any] = None,
              track_embeddings: Union[bool, float, int, str] = False,
              return_rotated_embeddings: bool = False,
              rotated_embeddings_per_epoch: bool = False,
              return_rotation_matrices: bool = False,
              leak_check_window_size: Optional[float] = None,
              leak_check_step: Optional[float] = None) -> Dict[str, Any]:

        """Trains the critic model and returns performance metrics.

        This method implements the main training loop, including data splitting,
        training, evaluation, and early stopping.

        Parameters
        ----------
        n_epochs : int
            The maximum number of epochs to train for.
        batch_size : int
            The number of samples per batch.
        train_fraction : float, optional
            The fraction of the data to use for training. Defaults to 0.9.
        n_test_blocks : int, optional
            For 'blocked' split_mode, the number of contiguous blocks for the test set.
            Defaults to 5.
        shift_time : bool, optional
            For temporal, windowed data: re-tiles the windows from a fresh
            random time offset every epoch (via `PairedTemporalDataset.time_shift`),
            so training doesn't see one fixed set of window boundaries.
            Shifts at full magnitude every epoch, up to `window_manager.window_size`.
            A no-op on a non-temporal dataset.
        shift_windows : bool, optional
            Cheaper, narrower alternative to `shift_time` for regularly-sampled
            data windowed via a fixed `window_size`/`step_size`: re-slices
            (not interpolates) a fresh window tiling from a random integer
            sample offset every epoch, at negligible per-epoch cost. Shifts
            at full magnitude every epoch, up to `window_size` samples. Only
            takes effect when `dataset` was built with an attached
            `_window_shifter` (`neural_mi/data/shift_windowing.py`); a
            no-op otherwise. Mutually exclusive in practice with
            `shift_time` (they gate on different dataset types).
        shift_seed : int, optional
            Seeds the generator the shifts are drawn from.
        split_seed : int, optional
            Seeds a generator of the run's own for the train/test split, the
            evaluation rows and their subsampling. Runs given the same
            split_seed share all three, whatever else draws random numbers
            before or during training, which is what pairs the terms of a
            difference quantity. Without it they come from NumPy's global state.
        patience : int, optional
            Epochs to wait for improvement before early stopping. Defaults to 10.
            Early stopping decides when training ends. The network evaluated is
            the one at the peak of the held-out curve smoothed after training
            (``best_epoch``), whichever epoch that is: copies of the network are
            kept for the epochs that can still turn out to be that peak.
        smoothing_sigma : float, optional
            Standard deviation for the Gaussian smoothing kernel on validation MI.
            Defaults to 1.0.
        median_window : int, optional
            Window size for the median filter on validation MI. Defaults to 5.
        min_improvement : float, optional
            Minimum relative improvement to reset the patience counter. Defaults to 0.001.
        save_best_model_path : str, optional
            If provided, saves the best model's state dictionary to this path.
        output_units : {'nats', 'bits'}, optional
            The units for displaying the MI estimate. Defaults to 'nats'.
        verbose : bool, optional
            If True, details and defaults will be displayed. Defaults to False.
        show_progress : bool, optional
            If True, progress bar will be shown. Defaults to True.
        split_mode : {'blocked', 'random'}, optional
            The method for splitting data into training and validation sets.
            - 'blocked': Samples contiguous blocks, useful for time-series data.
            - 'random': Performs a simple random shuffle, treating samples as IID.
            This parameter is ignored if `train_indices` and `test_indices` are provided.
            Defaults to 'blocked'.
        train_indices : np.ndarray, optional
            An array of specific indices to use for the training set. If provided,
            `split_mode` and `train_fraction` are ignored.
        test_indices : np.ndarray, optional
            An array of specific indices to use for the test set. If provided,
            `split_mode` and `train_fraction` are ignored.
        max_eval_samples : int, optional
            The most rows a single evaluation uses, which bounds its memory.
            A larger set is evaluated in blocks of at most this many rows whose
            values are averaged. Defaults to 5000.
        train_subset_size : int or 'full', optional
            Size of the fixed subset of training samples the reported,
            training-side MI is evaluated on, or 'full' for all of them.
            Training itself uses every training sample. Defaults to None,
            which takes ``min(n_train, max_eval_samples)``.
        test_subset_size : int or 'full', optional
            Size of the fixed subset of held-out samples evaluated at every
            epoch and at the end, or 'full' for all of them. It is drawn once,
            right after the split, so every epoch scores the same rows and runs
            given the same ``split_seed`` score the same rows. A blocked split
            takes one contiguous stretch per held-out block, a random split
            random rows. Defaults to None, which takes
            ``min(n_test, max_eval_samples)``.
        split_gap_fraction : float, optional
            When using 'blocked' split_mode, this fraction of the data will be left as a gap between training and test blocks to reduce leakage.
        leak_check_window_size, leak_check_step : float, optional
            Window geometry for the blocked-split leakage check, in time units.
            ``gap_fraction`` above buys a gap of *windows*, not time, the
            actual time-domain buffer is ``gap_size * step``, and if that's
            shorter than the window, train and test windows can share raw
            samples even though their indices don't overlap. Most callers
            don't need to pass these explicitly: if omitted, the check falls
            back to ``dataset.window_manager`` when one is live (e.g.
            deferred-processing paths like ``mode='lag'``). Callers whose
            windowing already happened upstream of this Trainer call (the
            common case for most modes. See ``run.py``, which builds the
            windowed dataset once and passes only the resulting tensor down)
            must pass these explicitly, since ``dataset`` here is a plain,
            already-windowed dataset with no window manager of its own.
        track_spectral_history : bool, optional
            If True, records ``pr_eig``, ``pr_singular``, and the raw cross-covariance
            spectrum (singular values) at every epoch in the returned
            ``spectral_metrics_history``, can be expensive, since it evaluates on
            ``train_eval_view`` each epoch. Defaults to False. Independent of this
            flag, the same three values are always computed once at the best epoch
            (see ``pr_eig``, ``pr_singular``, ``spectrum`` in the returned dict).
        max_index_reduction : float, optional
            When using temporal datasets with windowing, random time shifting can reduce the number of valid windows
            due to edge effects. This parameter sets a threshold for acceptable reduction in valid windows after shifting.
            If the reduction exceeds this threshold, a warning is logged. Defaults to 0.05 (5%).
        eval_train : bool, float, or int, optional
            Controls whether train-set MI is evaluated at every epoch alongside test-set MI,
            yielding a ``'train_mi_history'`` in the returned results.

            - ``False`` (default), no per-epoch train evaluation.
            - ``True``: evaluate on the same locked-in training evaluation subset
              used for the final ``train_mi`` (size capped by ``max_eval_samples``).
            - ``'full'`` or ``1.0``: evaluate on the entire training set, not
              capped by ``max_eval_samples``. Use this when the train curve needs
              to be read against the reported estimate: a smaller evaluation
              subset carries a lower InfoNCE ceiling (``log2(n_eval)``) and so
              sits below the estimate for reasons that have nothing to do with
              training.
            - ``float`` in ``(0, 1)``: use that fraction of training samples.
            - ``int >= 1``: use exactly that many training samples (capped at
              the available training set size).

            An unrecognised value raises ``ValueError``.
        peak_fraction : float, optional
            Controls how the best epoch is selected for reporting train MI.

            - ``1.0`` (default), use the epoch where smoothed test MI is maximised.
            - ``< 1.0``: use the first epoch whose smoothed test MI reaches
              ``peak_fraction`` of its peak.  This gives a
              conservative estimate that avoids the noisiest tail of training.
              Both the conservative and best-epoch estimates are obtained via
              fresh full evaluations at the end of training (no per-epoch train
              tracking is needed).  The results dict will contain
              ``'conservative_epoch'`` (the epoch used) and
              ``'train_mi_at_peak'`` (train MI at the actual peak epoch, for
              comparison).
        scheduler : torch.optim.lr_scheduler instance, optional
            A PyTorch learning-rate scheduler to step at the end of each epoch.
            ``ReduceLROnPlateau`` is stepped with the current test MI as the metric
            (maximisation mode); all other schedulers are stepped unconditionally.
            Build the scheduler via ``task.py`` using the ``scheduler`` /
            ``scheduler_params`` keys in ``base_params``. Defaults to ``None``.
        track_embeddings : bool, float, int, or str, optional
            Controls whether embeddings are extracted and stored at every epoch for
            post-hoc animation.  Mirrors the ``eval_train`` style:

            - ``False`` (default), no embedding tracking.
            - ``True``: track the first 512 samples.
            - ``int >= 1``: track exactly that many samples (first N in dataset).
            - ``float`` in ``(0, 1)``: track that fraction of the total dataset.
            - ``'full'`` or ``1.0``: track all samples (emits a ``UserWarning``
              about cost).

            An unrecognised value raises ``ValueError``.

            The tracked subset is always the **first** N samples so that
            user-supplied labels align with the original data ordering.
            Results are stored as ``'embedding_history_x'`` and
            ``'embedding_history_y'`` in the returned dict (each a list of
            ``(n_tracked, embedding_dim)`` arrays, one per epoch).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of the training run.
        """
        nats_to_bits = 1 / np.log(2) if output_units == 'bits' else 1.0
        self._rng = np.random.RandomState(int(split_seed) % (2 ** 31)) if split_seed is not None else np.random
        # Any bundle carrying a window grid is temporal, not only the
        # two-stream subclass: a three-stream conditional bundle has the same
        # grid and the same time_shift, and testing the subclass would leave
        # it silently unshiftable.
        is_temporal = isinstance(dataset, AlignedStreams)

        # Prime the time-shift grid once, up front -- before the train/test
        # split or the frozen eval snapshot below are computed -- so
        # len(dataset) and every index array derived from it are sized
        # against the final, margin-reserved window count from the start
        # (see PairedTemporalDataset._reserve_shift_margin). A recording too
        # short for a safe shift fails here, not partway through training.
        if is_temporal and shift_time:
            dataset.time_shift(offset_x=0.0, offset_y=0.0)

        # Move decoders to device and set to training mode
        if self.decoder_x is not None:
            self.decoder_x = self.decoder_x.to(self.device)
        if self.decoder_y is not None:
            self.decoder_y = self.decoder_y.to(self.device)

        # 1. Split Data
        if train_indices is not None and test_indices is not None:
            train_idx, test_idx = train_indices, test_indices
        elif split_mode == 'random':
            train_idx, test_idx = self._create_random_split(len(dataset), train_fraction)
        else:
            # Window geometry for the leakage check below can come from two
            # places, depending on how this dataset was built:
            #  - dataset.window_manager is live when windowing was deferred to
            #    this worker (e.g. mode='lag', a processor-swept mode='sweep')
            #    and `dataset` really is the PairedTemporalDataset that did it.
            #  - leak_check_window_size/step (explicit args, set by callers like
            #    run.py's single-preprocess path for mode='estimate' and most
            #    other modes) cover the common case where windowing already
            #    happened once upstream and `dataset` here is a plain,
            #    already-windowed PairedDataset with no window_manager at all.
            _wm = getattr(dataset, 'window_manager', None)
            if leak_check_window_size is not None and leak_check_step is not None:
                _leak_kwargs = {'window_size': leak_check_window_size, 'step': leak_check_step}
            elif _wm is not None:
                _leak_kwargs = {'window_size': _wm.window_size, 'step': _wm.resolve_step()}
            else:
                _leak_kwargs = {}
            if is_temporal and shift_time and 'window_size' in _leak_kwargs:
                # Under a time shift a training window's content can drift by
                # up to a full window_size while the frozen eval window stays
                # at offset 0, so the safe leak-check margin is
                # 2 * window_size, not window_size. The check only warns; it
                # does not change the split.
                _leak_kwargs = {**_leak_kwargs, 'window_size': 2 * _leak_kwargs['window_size']}
            train_idx, test_idx = self._create_blocked_split(len(dataset), train_fraction, n_test_blocks,
                                                             gap_fraction=split_gap_fraction, **_leak_kwargs)
        # Kept for callers that evaluate the trained network again afterwards,
        # as mode='precision' does on corrupted copies of the training rows.
        self.last_train_indices = train_idx

        n_train = len(train_idx)
        if batch_size > n_train > 0:
            # Say so instead of capping silently. Batch size sets how many
            # contrastive samples each estimate is scored against, so raising it
            # is the standard response to an estimate that looks bounded -- and
            # above n_train that response does nothing at all. Naming the real
            # constraint here is what stops a user tuning a knob that cannot
            # move.
            warnings.warn(
                f"batch_size={batch_size} exceeds the {n_train} training samples "
                f"available and is capped to {n_train}. Batch size sets how many "
                f"negatives each training step sees. A batch cannot hold more samples "
                f"than the training set. A longer recording, a smaller window_size or "
                f"a smaller step_size gives more training samples and allows larger "
                f"batches.",
                UserWarning, stacklevel=user_stacklevel(),
            )
            batch_size = n_train
        if batch_size < 2 and n_train > 1:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}.")

        train_view = SubsetView(dataset, indices=train_idx, max_index_reduction=max_index_reduction)

        # Lock in the held-out rows every evaluation scores. Drawn once, before
        # anything else draws from the run's generator, so every epoch scores the
        # same rows, and the terms of a difference (same split_seed) score the
        # same rows whatever their other settings. A blocked split takes one
        # contiguous stretch per held-out block, a random split random rows,
        # like the split itself.
        if isinstance(test_subset_size, str):
            if test_subset_size != 'full':
                raise ValueError(
                    f"test_subset_size={test_subset_size!r} is not a recognised value. Use an int >= 1, "
                    f"'full' for the whole held-out set, or None for min(n_test, max_eval_samples)."
                )
            test_subset_size = len(test_idx)
        if test_subset_size is not None and test_subset_size > len(test_idx):
            warnings.warn(
                f"test_subset_size={test_subset_size} exceeds the number of held-out samples "
                f"({len(test_idx)}). Clamping to {len(test_idx)}.",
                UserWarning,
                stacklevel=user_stacklevel(),
            )
        actual_test_subset_size = min(test_subset_size or max_eval_samples, len(test_idx))
        test_eval_idx = self._select_eval_indices(test_idx, actual_test_subset_size,
                                                  is_temporal or split_mode != 'random')
        test_eval_view = SubsetView(dataset, indices=test_eval_idx, max_index_reduction=max_index_reduction)
        if actual_test_subset_size > max_eval_samples:
            _blocks, _per_block = _blocks_of(actual_test_subset_size, max_eval_samples)
            logger.warning(
                f"Held-out MI evaluated on all {actual_test_subset_size} rows in {_blocks} blocks of at most "
                f"{_per_block} rows at every epoch, averaged. A single evaluation is limited to "
                f"max_eval_samples={max_eval_samples}. Each block uses its own rows as negatives, so the "
                f"ceiling is log({_per_block}) and the precision is that of all {actual_test_subset_size} rows. "
                f"Every epoch costs {_blocks} evaluations. Raise max_eval_samples to evaluate in fewer, "
                f"larger blocks if memory allows."
            )

        # 2. Lock in a Train Evaluation Subset (to prevent OOM/slowdown during train_mi tracking)
        # Deliberately NOT coupled to len(test_idx): the reported train-side MI's
        # ceiling is log(actual_train_subset_size), and there is no statistical
        # reason to let a small test split (e.g. 10-20% of windows by default)
        # needlessly shrink that ceiling. Only max_eval_samples and the available
        # training pool bound it.
        if isinstance(train_subset_size, str):
            if train_subset_size != 'full':
                raise ValueError(
                    f"train_subset_size={train_subset_size!r} is not a recognised value. Use an int >= 1, "
                    f"'full' for the whole training set, or None for min(n_train, max_eval_samples)."
                )
            train_subset_size = len(train_idx)
        actual_train_subset_size = train_subset_size or min(len(train_idx), max_eval_samples)
        if train_subset_size is not None and train_subset_size > len(train_idx):
            warnings.warn(
                f"train_subset_size={train_subset_size} exceeds the number of available "
                f"training samples ({len(train_idx)}). Clamping to {len(train_idx)}. "
                f"Evaluation metrics may be less stable than expected.",
                UserWarning,
                stacklevel=user_stacklevel(),
            )
        # Clamp to available training samples to avoid ValueError from np.random.choice
        actual_train_subset_size = min(actual_train_subset_size, len(train_idx))
        train_eval_idx = self._select_eval_indices(train_idx, actual_train_subset_size, is_temporal)
        train_eval_view = SubsetView(dataset, indices=train_eval_idx, max_index_reduction=max_index_reduction)

        # Determine the subset for per-epoch train MI tracking (eval_train parameter)
        _do_epoch_train_eval = bool(eval_train is not False and eval_train is not None and eval_train != 0)
        if _do_epoch_train_eval and len(train_idx) > 0:
            # `eval_train is True` must be tested before the 1.0 branch: in Python
            # True == 1.0, so a value check would swallow it.
            if eval_train is True:
                epoch_train_n = min(len(train_idx), max_eval_samples)
            elif eval_train == 'full' or (isinstance(eval_train, float) and eval_train == 1.0):
                # The whole training set, deliberately not capped by max_eval_samples.
                # This is what makes the per-epoch train curve comparable to the
                # reported estimate instead of sitting below it on a smaller
                # evaluation subset (and so under a lower InfoNCE ceiling).
                epoch_train_n = len(train_idx)
            elif isinstance(eval_train, float) and 0.0 < eval_train < 1.0:
                epoch_train_n = max(2, min(int(len(train_idx) * eval_train), len(train_idx)))
            elif isinstance(eval_train, int) and eval_train >= 1:
                epoch_train_n = min(eval_train, len(train_idx))
            else:
                raise ValueError(
                    f"eval_train={eval_train!r} is not a recognised value. Use False to "
                    f"disable, True for the locked-in evaluation subset (capped by "
                    f"max_eval_samples={max_eval_samples}), 'full' or 1.0 for the entire "
                    f"training set, a float in (0, 1) for a fraction of it, or an int >= 1 "
                    f"for that many samples."
                )
            if _do_epoch_train_eval:
                epoch_train_eval_idx = self._select_eval_indices(train_idx, epoch_train_n, is_temporal)
                # Wrapped in a SubsetView (like train_view/test_eval_view/train_eval_view)
                # instead of indexed as a plain fixed array -- for temporal data a
                # shift_time rebuild can shrink window_manager.n_windows (windows
                # failing min_coverage_fraction after the shift get dropped),
                # which would otherwise leave this array's indices pointing past
                # the end of the rebuilt dataset.
                epoch_train_eval_view = SubsetView(dataset, indices=epoch_train_eval_idx,
                                                   max_index_reduction=max_index_reduction)
        else:
            _do_epoch_train_eval = False

        # Determine the subset for per-epoch embedding tracking (track_embeddings param)
        _N_total = len(dataset)
        _DEFAULT_EMBED_N = 512
        _do_embed_tracking = not (
            track_embeddings is False or track_embeddings is None or track_embeddings == 0
        )
        if _do_embed_tracking:
            # `is True` must precede the numeric branches: True == 1.0 in Python,
            # and isinstance(True, int) is also True.
            if track_embeddings is True:
                embed_track_n = min(_DEFAULT_EMBED_N, _N_total)
            elif track_embeddings == 'full' or (isinstance(track_embeddings, float)
                                                and track_embeddings == 1.0):
                warnings.warn(
                    f"track_embeddings={track_embeddings!r}: storing embeddings for all "
                    f"{_N_total} samples at every epoch can be very memory-intensive "
                    f"({_N_total} × embedding_dim × n_epochs × 4 bytes). "
                    f"Pass an integer (e.g. track_embeddings=512) to limit tracking to the "
                    f"first N samples.",
                    UserWarning,
                    stacklevel=user_stacklevel(),
                )
                embed_track_n = _N_total
            elif isinstance(track_embeddings, int) and track_embeddings >= 1:
                embed_track_n = min(track_embeddings, _N_total)
            elif isinstance(track_embeddings, float) and 0.0 < track_embeddings < 1.0:
                embed_track_n = max(2, min(int(_N_total * track_embeddings), _N_total))
            else:
                raise ValueError(
                    f"track_embeddings={track_embeddings!r} is not a recognised value. Use "
                    f"False to disable, True for the first {_DEFAULT_EMBED_N} samples, "
                    f"'full' or 1.0 for every sample, a float in (0, 1) for a fraction of "
                    f"them, or an int >= 1 for that many."
                )
            if _do_embed_tracking:
                # Always the FIRST N samples so user labels align with original data order
                embed_track_idx = np.arange(embed_track_n)
        else:
            embed_track_n = 0

        # Rotation is only meaningful for critics with separate embedding networks.
        # The trainer rotates the per-epoch history, so it has work only when
        # embeddings are tracked. The final embeddings are rotated after
        # training (analysis/task.py) whether or not they were tracked, and
        # run() warns up front about a rotation setting that nothing uses.
        _has_embed_nets = hasattr(self.model, 'embedding_net_x')
        _do_rotation = False
        if return_rotated_embeddings:
            if not _has_embed_nets:
                warnings.warn(
                    "return_rotated_embeddings=True was requested and this trainer's model "
                    "exposes no embedding_net_x. It has no separate embedding networks to "
                    "rotate and rotation is skipped.",
                    UserWarning, stacklevel=user_stacklevel(),
                )
            elif _do_embed_tracking:
                _do_rotation = True

        # AMP: only active on CUDA; silently no-ops on CPU/MPS.
        _amp_active = (
            self.device.type == 'cuda'
            and (self.use_amp is True or self.use_amp == 'auto')
        )
        _scaler = torch.amp.GradScaler('cuda') if _amp_active else None

        # shift_time/shift_windows are training-time augmentations only:
        # evaluation always reads a frozen, pre-shift snapshot (both the
        # data and the index arrays, not SubsetView's live-updating
        # `.indices`, which can drift for `shift_time`'s real
        # PairedTemporalDataset across rebuilds) instead of whatever the
        # dataset currently holds. Training batches still read the live,
        # currently-shifted `dataset.x_dataset`/`.y_dataset` below.
        # Per-epoch shifts are drawn from their own generator, seeded from the
        # run's base seed instead of the per-task one. Quantities built as a
        # difference of estimates (conditional MI, interaction information) train
        # one leg per term, and those legs carry different task seeds by design,
        # so drawing shifts from the global stream had every leg looking at a
        # different stretch of the recording each epoch. Subtracting estimates
        # taken on different data adds between-chunk variance to a residual that
        # is already a small difference of large numbers. A shared seed puts
        # every leg on the same shift at the same epoch, which costs nothing:
        # no coordination between runs, just a generator they can each rebuild.
        _shift_generator = None
        if shift_seed is not None:
            _shift_generator = torch.Generator()
            _shift_generator.manual_seed(int(shift_seed) % (2 ** 31))
        _window_shifter = getattr(dataset, '_window_shifter', None)
        _shifting_active = (is_temporal and shift_time) or (shift_windows and _window_shifter is not None)
        if _shifting_active:
            _eval_x_source = _detach_clone(dataset.x_dataset.data)
            _eval_y_source = dataset.y_dataset.data.detach().clone()
            _eval_test_idx = torch.as_tensor(test_eval_idx, dtype=torch.long)
            _eval_train_eval_idx = torch.as_tensor(train_eval_idx, dtype=torch.long)
            if _do_epoch_train_eval:
                _eval_epoch_train_eval_idx = torch.as_tensor(epoch_train_eval_idx, dtype=torch.long)
        else:
            _eval_x_source = dataset.x_dataset
            _eval_y_source = dataset.y_dataset
            _eval_test_idx = test_eval_view.indices
            _eval_train_eval_idx = train_eval_view.indices
            if _do_epoch_train_eval:
                _eval_epoch_train_eval_idx = epoch_train_eval_view.indices

        history, train_history, metrics_tracked, best_mi, no_improve = [], [], [], -float('inf'), 0
        embedding_history_x: list = []
        embedding_history_y: list = []
        embedding_history_x_rotated: list = []
        embedding_history_y_rotated: list = []
        _rotation_singular_values_history: list = []
        _rotation_history_x: list = []
        _rotation_history_y: list = []
        # Copies of the network at every epoch that can still turn out to be the
        # one evaluated: the peak of the held-out curve smoothed after training,
        # or for peak_fraction < 1 the first epoch to reach that fraction of it.
        # An epoch's smoothed value depends on up to _reach epochs after it. Once
        # it is final, the epoch is kept only if it beats every earlier final
        # epoch (the first epoch to reach any fraction of the peak always does),
        # and with peak_fraction = 1 it replaces the one it beats. A custom
        # smoothing function has no known reach, so every epoch is kept.
        _states: Dict[int, Any] = {}
        _reach = None if self.custom_smoothing_fn else self._smoothing_reach(smoothing_sigma, median_window)
        _record, _record_ep = -np.inf, None
        nan_streak = 0
        
        display_progress = show_progress if show_progress is not None else verbose
        epoch_iterator = tqdm(range(n_epochs), desc="Training", leave=False,
                              disable=not display_progress)
        
        # 3. Epoch Loop
        for epoch in epoch_iterator:
            self.model.train()
            if self.decoder_x is not None:
                self.decoder_x.train()
            if self.decoder_y is not None:
                self.decoder_y.train()
            
            # Manual batching for efficiency and temporal shifting support
            current_train_idx = train_view.indices
            shuffled_train_idx = current_train_idx[torch.randperm(current_train_idx.nelement())]
            
            for batch_idx in shuffled_train_idx.split(batch_size):
                self.optimizer.zero_grad()
                x_batch = _to_device(dataset.x_dataset[batch_idx, ...], self.device)
                y_batch = dataset.y_dataset[batch_idx, ...].to(self.device)
                if self.aug_params_x:
                    x_batch = apply_augmentations(x_batch, self.aug_params_x)
                if self.aug_params_y:
                    y_batch = apply_augmentations(y_batch, self.aug_params_y)
                _fwd_ctx = (torch.autocast(device_type='cuda')
                            if _amp_active else contextlib.nullcontext())
                with _fwd_ctx:
                    scores, kl_loss = self.model(x_batch, y_batch)
                    mi_estimate = self.estimator_fn(scores, **self.estimator_params)
                    # In the bottleneck Lagrangian beta scales everything on the
                    # relevance side, the MI term and both reconstructions alike.
                    # Keeping the reconstructions inside it is what lets the
                    # lambdas stay relative to the MI term: without it, changing
                    # beta would silently re-weight reconstruction against MI.
                    mi_coeff = self.beta if self.use_variational else 1.0
                    if self.use_variational:
                        loss = kl_loss - self.beta * mi_estimate
                    else:
                        loss = -mi_estimate
                    # Optional decoder reconstruction loss
                    if self.decoder_x is not None or self.decoder_y is not None:
                        z_x, z_y = self.model.get_training_embeddings(x_batch, y_batch)
                        if self.decoder_x is not None:
                            recon_x = self.decoder_x(z_x)
                            loss = loss + mi_coeff * self.decoder_lambda_x * self._decoder_loss(
                                recon_x, x_batch, self.decoder_output_activation_x)
                        if self.decoder_y is not None:
                            recon_y = self.decoder_y(z_y)
                            loss = loss + mi_coeff * self.decoder_lambda_y * self._decoder_loss(
                                recon_y, y_batch, self.decoder_output_activation_y)
                if _amp_active:
                    _scaler.scale(loss).backward()
                    if self.gradient_clip_val is not None:
                        all_params = list(self.model.parameters())
                        if self.decoder_x is not None:
                            all_params.extend(self.decoder_x.parameters())
                        if self.decoder_y is not None:
                            all_params.extend(self.decoder_y.parameters())
                        _scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(all_params, self.gradient_clip_val)
                    _scaler.step(self.optimizer)
                    _scaler.update()
                else:
                    loss.backward()
                    if self.gradient_clip_val is not None:
                        # Clip gradients across model + decoder parameters
                        all_params = list(self.model.parameters())
                        if self.decoder_x is not None:
                            all_params.extend(self.decoder_x.parameters())
                        if self.decoder_y is not None:
                            all_params.extend(self.decoder_y.parameters())
                        nn.utils.clip_grad_norm_(all_params, self.gradient_clip_val)
                    self.optimizer.step()

            # Fast evaluation using safety chunking
            self.model.eval()
            if self.decoder_x is not None:
                self.decoder_x.eval()
            if self.decoder_y is not None:
                self.decoder_y.eval()
            with torch.no_grad():
                x_test = _index_batch(_eval_x_source, _eval_test_idx)
                y_test = _eval_y_source[_eval_test_idx, ...]
                mi_nats = self._safe_eval_mi(x_test, y_test, max_eval_samples, all_rows=True)

                if _do_epoch_train_eval:
                    x_etrain = _index_batch(_eval_x_source, _eval_epoch_train_eval_idx)
                    y_etrain = _eval_y_source[_eval_epoch_train_eval_idx, ...]
                    train_mi_nats = self._safe_eval_mi(x_etrain, y_etrain, max_eval_samples, all_rows=True)
                    train_history.append(train_mi_nats)

                # Per-epoch spectral history if requested (can be expensive, so optional)
                if track_spectral_history:
                    metrics_during = self._extract_spectral_metrics(
                        _index_batch(_eval_x_source, _eval_train_eval_idx),
                        _eval_y_source[_eval_train_eval_idx, ...],
                    )
                    metrics_tracked.append(metrics_during)

                # Per-epoch embedding tracking
                if _do_embed_tracking:
                    _zx_list, _zy_list = [], []
                    for _b_start in range(0, embed_track_n, batch_size):
                        _b_idx = embed_track_idx[_b_start:_b_start + batch_size]
                        _xb = _to_device(_index_batch(_eval_x_source, _b_idx), self.device)
                        _yb = _eval_y_source[_b_idx, ...].to(self.device)
                        _zx_b, _zy_b = self.model.get_embeddings(_xb, _yb)
                        _zx_list.append(_zx_b.cpu().numpy())
                        _zy_list.append(_zy_b.cpu().numpy())
                    embedding_history_x.append(np.concatenate(_zx_list, axis=0))
                    embedding_history_y.append(np.concatenate(_zy_list, axis=0))
                    if _do_rotation and rotated_embeddings_per_epoch:
                        _rot = compute_cross_covariance_rotation(
                            embedding_history_x[-1], embedding_history_y[-1],
                            whitening=self.whitening,
                        )
                        embedding_history_x_rotated.append(_rot['zx_rotated'])
                        embedding_history_y_rotated.append(_rot['zy_rotated'])
                        _rotation_singular_values_history.append(_rot['singular_values'])
                        if return_rotation_matrices:
                            _rotation_history_x.append(_rot['rotation_x'])
                            _rotation_history_y.append(_rot['rotation_y'])

            if np.isnan(mi_nats):
                nan_streak += 1
                if nan_streak >= 3:
                    raise TrainingError(
                        f"Training aborted: {nan_streak} consecutive NaN MI values "
                        f"(epochs {epoch + 2 - nan_streak} to {epoch + 1}). "
                        f"Check your learning_rate, batch_size and input data for "
                        f"numerical instability (for example exploding gradients or "
                        f"zero-variance channels)."
                    )
                logger.warning(
                    f"NaN MI detected at epoch {epoch + 1} (consecutive NaN streak: "
                    f"{nan_streak}/3). This step will be skipped for early stopping. "
                    f"If this persists, check your data and hyperparameters."
                )
            else:
                nan_streak = 0
            history.append(mi_nats)
            _states[epoch] = copy.deepcopy(self.model.state_dict())
            _final_ep = epoch - _reach if _reach is not None else -1
            if (_final_ep >= 0 and len(history) >= median_window
                    and not np.isnan(history[max(0, _final_ep - _reach):]).any()):
                _value = self._smooth(history, smoothing_sigma, median_window)[_final_ep]
                if _value > _record:
                    if peak_fraction >= 1.0 and _record_ep is not None:
                        del _states[_record_ep]
                    _record, _record_ep = _value, _final_ep
                else:
                    del _states[_final_ep]

            # Smoothing (Custom or Default)
            if self.custom_smoothing_fn:
                smoothed_nats = self.custom_smoothing_fn(history)[-1]
            else:
                smoothed_nats = self._smooth(history, smoothing_sigma, median_window)[-1]
            
            has_valid_baseline = not np.isinf(best_mi)
            improvement = (smoothed_nats - best_mi) / (abs(best_mi) + 1e-8) if has_valid_baseline else float('inf')

            # Data Augmentation: Temporal Shifting
            if is_temporal and shift_time:
                # Same reasoning as the window shift below: drawn from the
                # shared generator so every leg of a difference lands on the
                # same stretch at the same epoch.
                if _shift_generator is not None:
                    time_shift = float(torch.rand(1, generator=_shift_generator).item()
                                       * dataset.window_manager.window_size)
                else:
                    time_shift = np.random.uniform(high=dataset.window_manager.window_size)
                dataset.time_shift(offset_x=time_shift, offset_y=time_shift)

            # Data Augmentation: cheap reslice-based window shift (see
            # neural_mi/data/shift_windowing.py). No-op unless `dataset` was
            # built with an attached shifter.
            if shift_windows and _window_shifter is not None:
                _shift = _window_shifter.random_shift(_shift_generator)
                _x_shifted, _y_shifted = _window_shifter.windows_at(_shift)
                dataset.x_dataset.data = _x_shifted
                dataset.y_dataset.data = _y_shifted
                dataset.x_dataset.data_master = None
                dataset.y_dataset.data_master = None

            if display_progress:
                epoch_iterator.set_description(f"Training | MI: {mi_nats * nats_to_bits:.3f}")

            # LR scheduler step
            if scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau as _ROP
                if isinstance(scheduler, _ROP):
                    if not np.isnan(mi_nats):
                        scheduler.step(mi_nats)
                else:
                    scheduler.step()

            # Early stopping on the running smoothed curve. It decides when to
            # stop, not which network is evaluated.
            if not np.isnan(smoothed_nats) and (improvement > min_improvement or np.isinf(best_mi)):
                best_mi, no_improve = smoothed_nats, 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.debug(f"Early stopping at epoch {epoch+1}.")
                break
        
        # 4. Finalization: the network at the peak of the smoothed held-out curve
        _smoothed_history = np.asarray(
            self.custom_smoothing_fn(history) if self.custom_smoothing_fn
            else self._smooth(history, smoothing_sigma, median_window)
        )
        if np.isnan(_smoothed_history).all():
            raise TrainingError("Training failed to produce a valid model checkpoint.")
        best_ep = int(np.nanargmax(_smoothed_history))
        best_model_state = _states[best_ep]
        self.model.load_state_dict(best_model_state)
        if save_best_model_path:
            torch.save(best_model_state, save_best_model_path)
            
        with torch.no_grad():
            final_test_mi = self._safe_eval_mi(
                _index_batch(_eval_x_source, _eval_test_idx),
                _eval_y_source[_eval_test_idx, ...], max_eval_samples, all_rows=True)
            final_train_mi = self._safe_eval_mi(
                _index_batch(_eval_x_source, _eval_train_eval_idx),
                _eval_y_source[_eval_train_eval_idx, ...], max_eval_samples, all_rows=True)
        
        from neural_mi.estimators import infonce_lower_bound
        _scale = nats_to_bits  # 1/ln(2) for bits, 1.0 for nats -- same scale as every other reported MI
        _units = output_units

        # Ceiling diagnostics, for every estimator: even one without a hard
        # ceiling is informative about its distance from log(n_eval).
        # `eval_size` is the held-out sample count, and `train_eval_size` the
        # count behind the reported estimate (`train_mi`). They can differ, so
        # each has its own name and ceiling. Each side evaluates every row of its
        # subset, in blocks of at most max_eval_samples rows when the subset is
        # larger. Each block's rows are its own negatives, so the rows of one
        # block set the ceiling.
        _eval_size = actual_test_subset_size if actual_test_subset_size > 0 else None
        _test_blocks, _test_block_size = _blocks_of(_eval_size, max_eval_samples) if _eval_size else (1, None)
        _train_eval_size = actual_train_subset_size if actual_train_subset_size > 0 else None
        _train_blocks, _train_block_size = (_blocks_of(_train_eval_size, max_eval_samples)
                                            if _train_eval_size else (1, None))
        _test_ceiling_mi = (np.log(_test_block_size) * _scale) if _test_block_size and _test_block_size >= 2 else None
        _train_ceiling_mi = (np.log(_train_block_size) * _scale) if _train_block_size and _train_block_size >= 2 else None
        _test_saturation = (final_test_mi * _scale / _test_ceiling_mi) if _test_ceiling_mi else None
        _train_saturation = (final_train_mi * _scale / _train_ceiling_mi) if _train_ceiling_mi else None

        _rows = max(actual_train_subset_size, epoch_train_n if _do_epoch_train_eval else 0)
        if _rows > max_eval_samples:
            _blocks, _per_block = _blocks_of(_rows, max_eval_samples)
            logger.warning(
                f"Train-side MI evaluated on all {_rows} rows in {_blocks} blocks of at most {_per_block} "
                f"rows, averaged. A single evaluation is limited to max_eval_samples={max_eval_samples}. "
                f"Each block uses its own rows as negatives, so the ceiling is log({_per_block}) "
                f"and the precision is that of all {_rows} rows. Raise max_eval_samples to evaluate "
                f"in fewer, larger blocks if memory allows."
            )
        if self.estimator_fn is infonce_lower_bound:
            _tripped = []
            if _test_ceiling_mi is not None and final_test_mi * _scale > CEILING_PROXIMITY_WARNING_THRESHOLD * _test_ceiling_mi:
                _tripped.append(f"test MI ({final_test_mi * _scale:.3f} {_units}) vs. test ceiling "
                                f"log(rows per evaluation={_test_block_size})={_test_ceiling_mi:.3f} {_units}")
            if _train_ceiling_mi is not None and final_train_mi * _scale > CEILING_PROXIMITY_WARNING_THRESHOLD * _train_ceiling_mi:
                _tripped.append(f"reported train MI ({final_train_mi * _scale:.3f} {_units}) vs. train-eval ceiling "
                                f"log(rows per evaluation={_train_block_size})={_train_ceiling_mi:.3f} {_units}")
            if _tripped:
                logger.warning(
                    f"InfoNCE estimate is near its ceiling: {'; '.join(_tripped)}. "
                    f"The true MI may be higher. Consider increasing max_eval_samples "
                    f"(and train_subset_size, if set) or switching to the 'smile' "
                    f"estimator for high-MI scenarios."
                )

        # Does peak-epoch selection distinguish epochs at all? When the
        # smoothed test trace rides near its ceiling for most of training, the
        # argmax is close to a coin flip over flat noise, and the reported
        # best_epoch is arbitrary.
        _test_trace_saturated_fraction = None
        if _test_ceiling_mi is not None and _smoothed_history.size > 0:
            _valid = _smoothed_history[~np.isnan(_smoothed_history)]
            if _valid.size > 0:
                _test_trace_saturated_fraction = float(
                    np.mean((_valid * _scale) > TEST_TRACE_SATURATION_THRESHOLD * _test_ceiling_mi)
                )
                if _test_trace_saturated_fraction > TEST_TRACE_SATURATED_EPOCH_FRACTION_THRESHOLD:
                    logger.warning(
                        f"Peak-epoch selection may be unreliable: "
                        f"{_test_trace_saturated_fraction:.0%} of the smoothed test-MI "
                        f"history sits above {TEST_TRACE_SATURATION_THRESHOLD:.0%} of the "
                        f"test ceiling ({_test_ceiling_mi:.3f} {_units}). With the trace "
                        f"riding near its ceiling, the epoch with the (noisy) maximum is "
                        f"close to arbitrary. Consider increasing max_eval_samples."
                    )

        # Early stopping is effectively off by default (patience defaults to
        # 1000), so nothing else signals "training simply ran out of epochs while
        # test MI was still climbing." Since these are lower-bound estimators,
        # under-training always biases the reported value downward -- the
        # dangerous direction -- silently.
        if no_improve < patience and len(history) > 1 and best_ep >= len(history) - 1:
            warnings.warn(
                f"Training completed all {len(history)} epoch(s) without early "
                f"stopping and the best (smoothed) test MI occurred at the final "
                f"epoch. MI may still have been rising when training stopped. The "
                f"reported estimate may then be an under-trained lower bound. Raise "
                f"n_epochs or lower patience to let early stopping act.",
                UserWarning,
                stacklevel=user_stacklevel(),
            )

        # All-negative flag (from test MI history; warning deferred until _raw_train_mi is set)
        valid_history = [v for v in history if not np.isnan(v)]
        _all_mi_negative = bool(valid_history and max(valid_history) <= 0)

        # At this point the model is loaded at best_ep; final_train_mi is the fresh eval there.
        _best_ep_train_mi = final_train_mi  # save for train_mi_at_peak reporting

        # Conservative epoch: the first epoch whose smoothed test MI reaches
        # peak_fraction of the peak. Its network is evaluated afresh, then the
        # best-epoch network is restored.
        _conservative_ep = None
        _conservative_train_mi = None
        if peak_fraction < 1.0:
            _peak = _smoothed_history[best_ep]
            _conservative_ep = (int(np.flatnonzero(_smoothed_history >= peak_fraction * _peak)[0])
                                if _peak > 0 else best_ep)
            if _conservative_ep != best_ep:
                self.model.load_state_dict(_states[_conservative_ep])
                with torch.no_grad():
                    _conservative_train_mi = self._safe_eval_mi(
                        _index_batch(_eval_x_source, _eval_train_eval_idx),
                        _eval_y_source[_eval_train_eval_idx, ...],
                        max_eval_samples, all_rows=True,
                    )
                self.model.load_state_dict(best_model_state)
            else:
                _conservative_train_mi = _best_ep_train_mi

        # Assign final_train_mi and _raw_train_mi
        if _conservative_train_mi is not None:
            final_train_mi = _conservative_train_mi
            _raw_train_mi = _conservative_train_mi
        else:
            _raw_train_mi = _best_ep_train_mi  # peak_fraction == 1.0 path

        # All-negative warning and zeroing (now that _raw_train_mi is correctly set)
        if _all_mi_negative:
            warnings.warn(
                f"All test MI values in the training history are non-positive "
                f"(max test MI = {max(valid_history) * _scale:.4f} {_units} at epoch {best_ep}). "
                f"The model learned no representation that generalises. This usually "
                f"means too few epochs, too high a learning rate or degenerate data. "
                f"Train MI is reported as 0 {_units}. The raw train MI was "
                f"{_raw_train_mi * _scale:.4f} {_units} and most likely reflects overfitting. "
                f"Raise n_epochs, lower learning_rate or inspect the data.",
                UserWarning,
                stacklevel=user_stacklevel(),
            )
            final_train_mi = 0.0
        elif final_train_mi is not None and final_train_mi < 0:
            # The test MI rose above zero, so the network learned something,
            # but the training-side value at the reported epoch came out below
            # zero. MI cannot be negative, so it is reported as 0 like the case
            # above, with the measured value kept as raw_train_mi.
            warnings.warn(
                f"The train MI at the reported epoch is negative "
                f"({final_train_mi * _scale:.4f} {_units}) while the test MI peaked at "
                f"{max(valid_history) * _scale:.4f} {_units} at epoch {best_ep}. MI cannot be "
                f"negative. Train MI is reported as 0 {_units} and the measured value is "
                f"kept as raw_train_mi. This usually means too few epochs or too few "
                f"samples for the network to learn the dependence.",
                UserWarning,
                stacklevel=user_stacklevel(),
            )
            final_train_mi = 0.0

        results = {
            'train_mi': final_train_mi,
            'raw_train_mi': _raw_train_mi,
            'test_mi': final_test_mi,
            'best_epoch': best_ep,
            'test_mi_history': history,
            'all_mi_negative': _all_mi_negative,
        }
        if _shifting_active:
            # Any post-training read of the dataset (e.g. task.py's
            # return_embeddings extraction) must use this frozen, pre-shift
            # snapshot instead of dataset.x_data/y_data directly -- those
            # properties reflect whatever shift was last applied during
            # training, not the canonical view best_model_state was scored
            # against.
            results['_frozen_eval_x'] = _eval_x_source
            results['_frozen_eval_y'] = _eval_y_source
        if _eval_size is not None:
            # eval_size, the held-out rows every evaluation scores. Its ceiling
            # is log(rows per block), not log(batch_size).
            results['eval_size'] = _eval_size
            results['test_eval_blocks'] = _test_blocks
            results['test_ceiling_mi'] = _test_ceiling_mi
            results['test_saturation'] = _test_saturation
        if _train_eval_size is not None:
            # train_eval_size, the sample count behind the reported estimate
            # (train_mi). It can exceed eval_size, and it is the ceiling that
            # applies to the number the caller is shown.
            results['train_eval_size'] = _train_eval_size
            results['train_eval_blocks'] = _train_blocks
            results['train_ceiling_mi'] = _train_ceiling_mi
            results['train_saturation'] = _train_saturation
        if _test_trace_saturated_fraction is not None:
            results['test_trace_saturated_fraction'] = _test_trace_saturated_fraction
        if _conservative_ep is not None:
            results['conservative_epoch'] = _conservative_ep
            results['train_mi_at_peak'] = _best_ep_train_mi
        if _do_epoch_train_eval:
            results['train_mi_history'] = train_history

        if track_spectral_history:
            results['spectral_metrics_history'] = metrics_tracked

        if _do_embed_tracking:
            results['embedding_history_x'] = embedding_history_x
            results['embedding_history_y'] = embedding_history_y
            results['embedding_track_n'] = embed_track_n

        if _do_rotation:
            if rotated_embeddings_per_epoch:
                # Per-epoch mode: rotation already computed inside the loop.
                results['embedding_history_x_rotated'] = embedding_history_x_rotated
                results['embedding_history_y_rotated'] = embedding_history_y_rotated
                results['embedding_rotation_singular_values'] = _rotation_singular_values_history
                if return_rotation_matrices:
                    results['embedding_rotation_history_x'] = _rotation_history_x
                    results['embedding_rotation_history_y'] = _rotation_history_y
            else:
                # Global mode: derive one rotation from the best epoch's stored embeddings
                # (best_ep indexes into embedding_history_x since both are populated every epoch),
                # then apply it uniformly to every epoch's history so the coordinate system is
                # consistent across epochs.
                _ref_ep = min(best_ep, len(embedding_history_x) - 1)
                _rot = compute_cross_covariance_rotation(
                    embedding_history_x[_ref_ep], embedding_history_y[_ref_ep],
                    whitening=self.whitening,
                )
                U, V = _rot['rotation_x'], _rot['rotation_y']
                for _zx_ep, _zy_ep in zip(embedding_history_x, embedding_history_y):
                    _zx_c = _zx_ep - _zx_ep.mean(axis=0, keepdims=True)
                    _zy_c = _zy_ep - _zy_ep.mean(axis=0, keepdims=True)
                    embedding_history_x_rotated.append(_zx_c @ U)
                    embedding_history_y_rotated.append(_zy_c @ V)
                results['embedding_history_x_rotated'] = embedding_history_x_rotated
                results['embedding_history_y_rotated'] = embedding_history_y_rotated
                results['embedding_rotation_singular_values'] = _rot['singular_values']
                if return_rotation_matrices:
                    results['embedding_rotation_x'] = U
                    results['embedding_rotation_y'] = V

        # Optionally evaluate reconstruction loss for decoder-augmented training
        if self.decoder_x is not None or self.decoder_y is not None:
            with torch.no_grad():
                # Evaluate on train eval subset
                _tx = _to_device(_index_batch(_eval_x_source, _eval_train_eval_idx), self.device)
                _ty = _eval_y_source[_eval_train_eval_idx, ...].to(self.device)
                _zx, _zy = self.model.get_embeddings(_tx, _ty)  # uses existing no_grad method
                _mi_coeff = self.beta if self.use_variational else 1.0
                _recon_loss = 0.0
                if self.decoder_x is not None:
                    _recon_x = self.decoder_x(_zx)
                    _recon_loss += _mi_coeff * self.decoder_lambda_x * float(
                        self._decoder_loss(_recon_x, _tx, self.decoder_output_activation_x).item())
                if self.decoder_y is not None:
                    _recon_y = self.decoder_y(_zy)
                    _recon_loss += _mi_coeff * self.decoder_lambda_y * float(
                        self._decoder_loss(_recon_y, _ty, self.decoder_output_activation_y).item())
            results['decoder_recon_loss'] = _recon_loss

        # 5. Final spectral metrics (pr_eig, pr_singular, spectrum) at the best epoch --
        # always computed, independent of track_spectral_history above.
        metrics_final = self._extract_spectral_metrics(
            _index_batch(_eval_x_source, _eval_train_eval_idx),
            _eval_y_source[_eval_train_eval_idx, ...],
        )
        results.update(metrics_final)

        return results

    @staticmethod
    def _decoder_loss(recon: torch.Tensor, target: torch.Tensor, activation: str) -> torch.Tensor:
        """Compute decoder reconstruction loss appropriate for the output activation.

        Parameters
        ----------
        recon : torch.Tensor
            Decoder output, shape ``(batch, n_channels, window_size)``.
            For ``activation='softmax'`` this is a probability distribution over
            channels (already post-softmax).
        target : torch.Tensor
            Ground-truth input, same shape as *recon*.
        activation : str
            Output activation used by the decoder: ``'linear'``, ``'sigmoid'``,
            or ``'softmax'``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if activation == 'softmax':
            # recon: (B, C, W), probability over C channels for each time step.
            # target: (B, C, W), ground-truth (one-hot or soft target over channels).
            # Use distributional cross-entropy:
            #   L = -E_{b,w} [ sum_c target_{b,c,w} * log(recon_{b,c,w}) ].
            log_probs = torch.log(recon.clamp(min=1e-8))  # (B, C, W)
            loss_per_timestep = -(target * log_probs).sum(dim=1)  # (B, W)
            return loss_per_timestep.mean()
        else:
            # 'linear' or 'sigmoid': MSE is appropriate.
            return nn.functional.mse_loss(recon, target)

    def _safe_eval_mi(self, x: torch.Tensor, y: torch.Tensor, max_samples: int, all_rows: bool = False) -> float:
        """Evaluates the MI on the given rows, never on more than max_samples at once.

        Up to max_samples rows are evaluated in one go. Above that, the held-out
        side (``all_rows=False``) draws one random subset of max_samples rows, and
        the train side (``all_rows=True``) evaluates every row in blocks of at most
        max_samples and averages the blocks, weighted by their sizes. Row r goes to
        block r % n_blocks, so every block spans the whole set. Each block is an
        ordinary evaluation whose own rows are its negatives: the average keeps
        the bound and the ceiling of one block, log(rows per block), and the
        precision of all the rows.

        Parameters
        ----------
        x, y : torch.Tensor
            The rows to evaluate.
        max_samples : int
            The most rows a single evaluation uses, which bounds its memory.
        all_rows : bool, optional
            Evaluate every row, in blocks if needed, instead of a random subset.
        """
        n_samples = _batch_size_of(x)
        if n_samples < 2:
            return float('nan')

        if n_samples <= max_samples:
            result = self._eval_mi(x, y)
        elif not all_rows:
            idx_t = torch.from_numpy(_rng_of(self).choice(n_samples, max_samples, replace=False))
            result = self._eval_mi(_index_batch(x, idx_t), y[idx_t])
        else:
            n_blocks = -(-n_samples // max_samples)
            values, sizes = [], []
            for b in range(n_blocks):
                idx_t = torch.arange(b, n_samples, n_blocks)
                values.append(self._eval_mi(_index_batch(x, idx_t), y[idx_t]))
                sizes.append(len(idx_t))
            result = float(np.average(values, weights=sizes))

        if np.isnan(result):
            logger.warning(
                "MI evaluation returned NaN. This may indicate numerical instability, "
                "a degenerate batch or exploding gradients. Check your learning_rate, "
                "batch_size and input data for anomalies."
            )
        return result

    def _eval_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        scores, _ = self.model(_to_device(x, self.device), y.to(self.device))
        if torch.isnan(scores).any():
            logger.warning(
                "Score matrix contains NaN values during evaluation. "
                "Returning NaN for this step. Check for exploding gradients or "
                "degenerate embeddings."
            )
            return float('nan')
        if torch.isinf(scores).any():
            dtype = scores.dtype
            safe_max = torch.finfo(dtype).max / 2
            scores = torch.clamp(scores, min=-safe_max, max=safe_max)
            logger.warning(
                f"Score matrix contains Inf values. Clamping to ±{safe_max:.3e} "
                f"(dtype={dtype}, machine-epsilon-aware bound)."
            )
        return self.estimator_fn(scores, **self.estimator_params).item()

    def _extract_spectral_metrics(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Extracts embeddings and computes cross-covariance spectral metrics.

        Always returns both participation-ratio variants (``pr_eig``,
        ``pr_singular``) plus the raw spectrum (singular values) they were
        computed from, ``effective_rank``/``spectral_entropy`` are omitted
        since they're cheaply derivable from the spectrum if ever needed.
        """
        self.model.eval()
        with torch.no_grad():
            zx, zy = self.model.get_embeddings(_to_device(x, self.device), y.to(self.device))

        spectrum = compute_cross_covariance_spectrum(zx, zy, whitening=self.whitening)
        metrics = compute_spectral_metrics(spectrum)

        results = {
            'whitening': self.whitening,
            'pr_eig': metrics['pr_eig'],
            'pr_singular': metrics['pr_singular'],
            'spectrum': spectrum,
        }
            
        return results

    @staticmethod
    def _smoothing_reach(sigma: float, med_win: int) -> int:
        """How many later epochs the smoothed value at an epoch depends on (see _smooth):
        half the median window plus the Gaussian's radius, which scipy cuts at 4 sigma."""
        return (med_win // 2 if med_win > 1 else 0) + (int(4.0 * sigma + 0.5) if sigma > 0 else 0)

    def _smooth(self, arr: List[float], sigma: float, med_win: int) -> np.ndarray:
        hist = np.array(arr)
        if len(hist) < 2: return hist
        nan_mask = np.isnan(hist)
        valid_hist = hist[~nan_mask]
        if len(valid_hist) == 0: return hist
        hist[nan_mask] = valid_hist[-1]
        if med_win > 1 and len(hist) >= med_win:
            hist = median_filter(hist, size=med_win, mode='reflect')
        if sigma > 0: 
            hist = gaussian_filter1d(hist, sigma=sigma, mode='reflect')
        hist[nan_mask] = np.nan
        return hist
        
    def _create_random_split(self, n: int, frac: float) -> Tuple[np.ndarray, np.ndarray]:
        indices = _rng_of(self).permutation(n)
        n_train = int(n * frac)
        return indices[:n_train], indices[n_train:]

    def _create_blocked_split(self, n: int, frac: float, k: int, gap_fraction: float = 0.0,
                              window_size: Optional[float] = None, step: Optional[float] = None,
                              leak_check_label: str = "") -> Tuple[np.ndarray, np.ndarray]:
        n_test = int(n * (1 - frac))
        if n_test == 0:
            return np.arange(n), np.array([])
        if n_test < k:
            k = n_test
        block, rem = divmod(n_test, k) if k > 0 else (0, 0)
        if n - block < k:
            logger.warning(
                "Blocked split parameters produced an invalid configuration. "
                "Falling back to random split. Consider reducing n_test_blocks or "
                "increasing your dataset size."
            )
            indices = _rng_of(self).permutation(n)
            return indices[n_test:], indices[:n_test]
        starts = _sample_with_minimum_distance(n - block, k, block + 1, _rng_of(self))
        test_idx = np.concatenate([
            np.arange(s, s + block + (1 if i < rem else 0))
            for i, s in enumerate(starts)
        ])

        gap_size = int(round(gap_fraction * block)) if block > 0 else 0
        if window_size is not None and step is not None:
            warn_if_blocked_split_leaks(gap_size, block, step, window_size, gap_fraction,
                                        path_label=leak_check_label)
        if gap_size > 0:
            gap_idx = set()
            for i, s in enumerate(starts):
                blk_len = block + (1 if i < rem else 0)
                # Buffer before and after each test block
                for g in range(1, gap_size + 1):
                    if s - g >= 0:
                        gap_idx.add(s - g)
                    if s + blk_len + g - 1 < n:
                        gap_idx.add(s + blk_len + g - 1)
            excluded = np.array(sorted(gap_idx), dtype=int)
            train_idx = np.setdiff1d(np.arange(n), np.union1d(test_idx, excluded))
            logger.debug(
                f"Blocked split gap: excluded {len(excluded)} samples "
                f"({gap_size} samples per block boundary) from training set."
            )
        else:
            train_idx = np.setdiff1d(np.arange(n), test_idx)

        return train_idx, test_idx

    def _select_eval_indices(self, idx: np.ndarray, target_size: int, contiguous: bool) -> np.ndarray:
        """Select a fixed, representative subset of ``idx`` for evaluation:
        the training rows behind the reported estimate and the per-epoch
        training curve (``eval_train``), or the held-out rows scored every epoch.

        ``contiguous`` takes one contiguous stretch per segment of ``idx``
        instead of random rows: always for temporal data, for the reason
        below, and on the held-out side for a blocked split, like the split.
        For static training rows a plain random subsample is fine. There's no
        temporal structure whose contiguity needs to survive anything.

        For temporal data, a *scattered* random subsample is wrong even
        when ``shift_time``/``shift_windows`` never fires
        for this run: ``SubsetView`` tracks a temporal subset by converting
        it to time ranges once, then re-deriving indices from those ranges
        whenever the dataset's windows are rebuilt (`views.py`). A random
        scatter of individual window indices has almost no contiguous
        runs, so it degenerates into one zero-width time range per index,
        thousands of them for a large subset. Re-quantizing that many
        degenerate ranges against a shifted window grid collides many of
        them onto the same window (deduplicated away) or drops them into a
        gap between windows entirely, silently discarding a large fraction
        of the subset on the very next shift. This is a lossy
        representation problem, not a "large shift" problem. It fires
        regardless of shift magnitude once the subset is scattered enough.

        The fix is to never *construct* a scattered temporal subset in the
        first place: ``idx`` for temporal data is itself a union of a
        handful of contiguous segments (whatever's left after
        ``_create_blocked_split`` carves out test blocks + gap buffers).
        Pick one contiguous sub-chunk per segment, sized proportionally to
        that segment's length, so the eval subset stays representable as a
        handful of wide time ranges, exactly what ``SubsetView`` already
        handles correctly and cheaply across rebuilds (`train_view`/
        `test_view` are also wide contiguous ranges), while still
        covering every part of the recording instead of sampling
        disproportionately from just one segment (a non-stationary
        recording would otherwise bias the eval MI toward whichever
        segment happened to be sampled, the same "don't let one chunk
        dominate" reasoning behind the contiguous-chunking fix in
        `analysis/rigorous.py`).
        """
        if target_size >= len(idx):
            return idx.copy()
        if not contiguous:
            return _rng_of(self).choice(idx, target_size, replace=False)

        sorted_idx = np.sort(idx)
        breaks = np.where(np.diff(sorted_idx) > 1)[0]
        seg_starts = np.concatenate([[0], breaks + 1])
        seg_ends = np.concatenate([breaks + 1, [len(sorted_idx)]])  # exclusive
        seg_lens = seg_ends - seg_starts

        raw_alloc = seg_lens / seg_lens.sum() * target_size
        alloc = np.floor(raw_alloc).astype(int)
        remainder = target_size - int(alloc.sum())
        if remainder > 0:
            # Give the extra samples to the segments with the largest
            # fractional remainder first, so the total matches target_size
            # exactly without a per-segment allocation bias.
            order = np.argsort(-(raw_alloc - alloc))
            for i in order[:remainder]:
                alloc[i] += 1
        alloc = np.minimum(alloc, seg_lens)

        chosen = []
        for start, end, n_take in zip(seg_starts, seg_ends, alloc):
            if n_take <= 0:
                continue
            seg = sorted_idx[start:end]
            if n_take >= len(seg):
                chosen.append(seg)
                continue
            # A random contiguous window within the segment (not always its
            # own start), so repeated runs don't all sample the same edge.
            offset = _rng_of(self).randint(0, len(seg) - n_take + 1)
            chosen.append(seg[offset:offset + n_take])
        return np.concatenate(chosen) if chosen else np.array([], dtype=idx.dtype)


