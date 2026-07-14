# neural_mi/training/trainer.py
"""Handles the training and evaluation of critic models for MI estimation.

This module provides the `Trainer` class, a comprehensive utility for training
a critic model, monitoring its performance, implementing early stopping, and
saving the best-performing model state.
"""
import warnings
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.ndimage import gaussian_filter1d, median_filter
import os
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, Optional, List, Callable, Union
import torch.nn as nn
import copy
import contextlib

from neural_mi.data import PairedDataset, PairedTemporalDataset, SubsetView
from neural_mi.logger import logger
from neural_mi.exceptions import TrainingError
from neural_mi.utils import compute_cross_covariance_spectrum, compute_spectral_metrics, compute_cross_covariance_rotation
from neural_mi.augmentations import apply_augmentations

def _ranks(sample: np.ndarray) -> List[int]:
    """Return each element's rank (position in sorted order) within `sample`."""
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def _sample_with_minimum_distance(n: int, k: int, d: int) -> np.ndarray:
    """Sample k block start positions in [0, n) with pairwise distance >= d.

    Draws k values from a compressed range of size n - (k-1)*(d-1), then
    spreads them out by each value's rank so consecutive picks are always
    at least d apart -- this is what keeps the blocked test-split's blocks
    from overlapping or sitting adjacent to each other.
    """
    sample = np.random.choice(n - (k - 1) * (d - 1), k, replace=False)
    return np.array([s + (d - 1) * r for s, r in zip(sample, _ranks(sample))])

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
                 spectral_whitening: str = 'std',
                 gradient_clip_val: Optional[float] = None,
                 decoder_x: Optional[nn.Module] = None,
                 decoder_y: Optional[nn.Module] = None,
                 decoder_weight_x: float = 1.0,
                 decoder_weight_y: float = 1.0,
                 decoder_output_activation_x: str = 'linear',
                 decoder_output_activation_y: str = 'linear',
                 use_amp: Union[bool, str] = 'auto',
                 augmentation_params_x: Optional[Dict[str, Any]] = None,
                 augmentation_params_y: Optional[Dict[str, Any]] = None):
        
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
         spectral_whitening : str, optional
            Method for spectral whitening when computing spectral metrics. Options are 'std' for standard whitening
            and 'zca' for ZCA whitening and None. Defaults to 'std'.
        gradient_clip_val : float, optional
            If set, applies ``torch.nn.utils.clip_grad_norm_`` with this value as
            the maximum gradient norm after each backward pass, before the
            optimiser step.  Helps prevent gradient explosions with high learning
            rates or difficult distributions.  ``None`` disables clipping.
        decoder_x : nn.Module, optional
            Decoder module that reconstructs X from the X-embedding Z_X.
            When provided, a reconstruction loss ``decoder_weight_x * MSE(X, decoder_x(Z_X))``
            is added to the training objective. Defaults to ``None`` (no decoder).
        decoder_y : nn.Module, optional
            Decoder module for Y. Defaults to ``None``.
        decoder_weight_x : float, optional
            Weight for the X reconstruction loss. Defaults to 1.0.
        decoder_weight_y : float, optional
            Weight for the Y reconstruction loss. Defaults to 1.0.
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
        self.spectral_whitening = spectral_whitening
        self.gradient_clip_val = gradient_clip_val
        self.decoder_x = decoder_x.to(device) if decoder_x is not None else None
        self.decoder_y = decoder_y.to(device) if decoder_y is not None else None
        self.decoder_weight_x = decoder_weight_x if decoder_weight_x is not None else 1.0
        self.decoder_weight_y = decoder_weight_y if decoder_weight_y is not None else 1.0
        self.decoder_output_activation_x = decoder_output_activation_x or 'linear'
        self.decoder_output_activation_y = decoder_output_activation_y or 'linear'
        self.use_amp = use_amp
        self.aug_params_x = augmentation_params_x or {}
        self.aug_params_y = augmentation_params_y or {}

    def train(self, dataset: Union[PairedDataset, PairedTemporalDataset], n_epochs: int, batch_size: int,
              train_fraction: float = 0.9, n_test_blocks: int = 5,
              random_time_shifting: bool = True, epochs_to_max_shift: int = 5,
              patience: int = 10, smoothing_sigma: float = 1.0, median_window: int = 5,
              min_improvement: float = 0.001,
              save_best_model_path: Optional[str] = None, run_id: Optional[str] = None,
              output_units: str = 'nats', verbose: bool = True, show_progress: bool = True,
              split_mode: str = 'blocked',
              train_indices: Optional[np.ndarray] = None,
              test_indices: Optional[np.ndarray] = None,
              max_eval_samples: int = 5000,
              train_subset_size: Optional[int] = None,
              split_gap_fraction: float = 0.5,
              track_spectral_metrics: bool = False,
              spectral_output: str = 'default',
              return_spectrum: bool = False,
              max_index_reduction: float = 0.05,
              eval_train: Union[bool, float, int] = False,
              peak_fraction: float = 1.0,
              scheduler: Optional[Any] = None,
              track_embeddings: Union[bool, float, int, str] = False,
              return_rotated_embeddings: bool = False,
              rotated_embeddings_whitening: Optional[str] = 'std',
              rotated_embeddings_per_epoch: bool = False,
              return_rotation_matrices: bool = False) -> Dict[str, Any]:
        
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
        random_time_shifting : bool, optional
            If data is temporal and windowed, will randomly shift in time to encourage learning 
            a robust representation of data
        epochs_to_max_shift : int, optional
            Number of epochs at start to wait before time shifting. 
            Can be useful to burn-in a working model before time shifting full amounts
        patience : int, optional
            Epochs to wait for improvement before early stopping. Defaults to 10.
        smoothing_sigma : float, optional
            Standard deviation for the Gaussian smoothing kernel on validation MI.
            Defaults to 1.0.
        median_window : int, optional
            Window size for the median filter on validation MI. Defaults to 5.
        min_improvement : float, optional
            Minimum relative improvement to reset the patience counter. Defaults to 0.001.
        save_best_model_path : str, optional
            If provided, saves the best model's state dictionary to this path.
        run_id : str, optional
            An identifier for the training run, used for display purposes.
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
            Maximum number of samples to use when evaluating MI on the validation set.
            If the test set is larger than this, a random subset will be used for evaluation.
            Defaults to 5000.
        train_subset_size : int, optional
            If provided, limits the number of training samples used in each epoch to this number.
            If the training set is larger than this, a random subset will be selected each epoch.
            Defaults to None (use all training samples).
        split_gap_fraction : float, optional
            When using 'blocked' split_mode, this fraction of the data will be left as a gap between training and test blocks to reduce leakage.
        track_spectral_metrics : bool, optional
            If True, computes and tracks spectral metrics of the learned representations at each epoch.
            Defaults to False.
        spectral_output : str, optional
            Determines which spectral metrics are returned. ``'default'`` returns both
            participation-ratio variants (``pr_eig``, ``pr_singular``). ``'all'`` additionally
            returns ``effective_rank`` and ``spectral_entropy``. Defaults to ``'default'``.
        return_spectrum : bool, optional
            If True, includes the full spectrum in the returned results when `track_spectral_metrics` is True. Defaults to False.
        max_index_reduction : float, optional
            When using temporal datasets with windowing, random time shifting can reduce the number of valid windows
            due to edge effects. This parameter sets a threshold for acceptable reduction in valid windows after shifting.
            If the reduction exceeds this threshold, a warning is logged. Defaults to 0.05 (5%).
        eval_train : bool, float, or int, optional
            Controls whether train-set MI is evaluated at every epoch alongside test-set MI,
            yielding a ``'train_mi_history'`` in the returned results.

            - ``False`` (default) — no per-epoch train evaluation.
            - ``True`` — evaluate on the same locked-in training evaluation subset
              used for the final ``train_mi`` (size capped by ``max_eval_samples``).
            - ``float`` in ``(0, 1)`` — use that fraction of training samples.
            - ``int >= 1`` — use exactly that many training samples (capped at
              the available training set size).
        peak_fraction : float, optional
            Controls how the best epoch is selected for reporting train MI.

            - ``1.0`` (default) — use the epoch where smoothed test MI is maximised.
            - ``< 1.0`` — use the *first improvement checkpoint* where smoothed test
              MI reaches ``peak_fraction × max_test_mi``.  This gives a
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

            - ``False`` (default) — no embedding tracking.
            - ``True`` — track the first 512 samples.
            - ``int >= 1`` — track exactly that many samples (first N in dataset).
            - ``float`` in ``(0, 1)`` — track that fraction of the total dataset.
            - ``'full'`` — track all samples (emits a ``UserWarning`` about cost).

            The tracked subset is always the **first** N samples so that
            user-supplied labels align with the original data ordering.
            Results are stored as ``'embedding_history_x'`` and
            ``'embedding_history_y'`` in the returned dict (each a list of
            ``(n_tracked, embed_dim)`` arrays, one per epoch).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of the training run.
        """
        nats_to_bits = 1 / np.log(2) if output_units == 'bits' else 1.0
        is_temporal = isinstance(dataset, PairedTemporalDataset)
        if is_temporal and random_time_shifting and patience < epochs_to_max_shift:
            logger.warning(
                f"patience={patience} < epochs_to_max_shift={epochs_to_max_shift}: "
                f"early stopping may fire before random_time_shifting has reached its "
                f"full range. The model may train only on unshifted windows. Consider "
                f"increasing patience or decreasing epochs_to_max_shift."
            )

        # Move decoders to device and set to training mode
        if self.decoder_x is not None:
            self.decoder_x = self.decoder_x.to(self.device)
        if self.decoder_y is not None:
            self.decoder_y = self.decoder_y.to(self.device)

        # 1. Split Data
        if train_indices is not None and test_indices is not None:
            logger.warning(
                "Custom train_indices and test_indices were provided. "
                "The split_mode, train_fraction, n_test_blocks, and split_gap_fraction "
                "parameters will be ignored for this run."
            )
            train_idx, test_idx = train_indices, test_indices
        elif split_mode == 'random':
            train_idx, test_idx = self._create_random_split(len(dataset), train_fraction)
        else:
            train_idx, test_idx = self._create_blocked_split(len(dataset), train_fraction, n_test_blocks,
                                                             gap_fraction=split_gap_fraction)
        
        n_train = len(train_idx)
        if batch_size > n_train > 0:
            batch_size = n_train
        if batch_size < 2 and n_train > 1: 
            raise ValueError(f"batch_size must be >= 2, got {batch_size}.")

        train_view = SubsetView(dataset, indices=train_idx, max_index_reduction=max_index_reduction)
        test_view = SubsetView(dataset, indices=test_idx, max_index_reduction=max_index_reduction)

        # 2. Lock in a Train Evaluation Subset (to prevent OOM/slowdown during train_mi tracking)
        actual_train_subset_size = train_subset_size or min(len(train_idx), len(test_idx), max_eval_samples)
        if train_subset_size is not None and train_subset_size > len(train_idx):
            warnings.warn(
                f"train_subset_size={train_subset_size} exceeds the number of available "
                f"training samples ({len(train_idx)}). Clamping to {len(train_idx)}. "
                f"Evaluation metrics may be less stable than expected.",
                UserWarning,
                stacklevel=2,
            )
        # Clamp to available training samples to avoid ValueError from np.random.choice
        actual_train_subset_size = min(actual_train_subset_size, len(train_idx))
        train_eval_idx = np.random.choice(train_idx, actual_train_subset_size, replace=False)
        train_eval_view = SubsetView(dataset, indices=train_eval_idx, max_index_reduction=max_index_reduction)

        # Determine the subset for per-epoch train MI tracking (eval_train parameter)
        _do_epoch_train_eval = bool(eval_train is not False and eval_train is not None and eval_train != 0)
        if _do_epoch_train_eval and len(train_idx) > 0:
            if eval_train is True:
                epoch_train_n = min(len(train_idx), max_eval_samples)
            elif isinstance(eval_train, float) and 0.0 < eval_train < 1.0:
                epoch_train_n = max(2, min(int(len(train_idx) * eval_train), len(train_idx)))
            elif isinstance(eval_train, int) and eval_train >= 1:
                epoch_train_n = min(eval_train, len(train_idx))
            else:
                _do_epoch_train_eval = False
                epoch_train_n = 0
            if _do_epoch_train_eval:
                epoch_train_eval_idx = np.random.choice(train_idx, epoch_train_n, replace=False)
        else:
            _do_epoch_train_eval = False

        # Determine the subset for per-epoch embedding tracking (track_embeddings param)
        _N_total = len(dataset)
        _DEFAULT_EMBED_N = 512
        _do_embed_tracking = not (
            track_embeddings is False or track_embeddings is None or track_embeddings == 0
        )
        if _do_embed_tracking:
            if track_embeddings == 'full':
                warnings.warn(
                    f"track_embeddings='full': storing embeddings for all {_N_total} samples "
                    f"at every epoch can be very memory-intensive "
                    f"({_N_total} × embed_dim × n_epochs × 4 bytes). "
                    f"Pass an integer (e.g. track_embeddings=512) to limit tracking to the "
                    f"first N samples.",
                    UserWarning,
                    stacklevel=2,
                )
                embed_track_n = _N_total
            elif track_embeddings is True:
                embed_track_n = min(_DEFAULT_EMBED_N, _N_total)
            elif isinstance(track_embeddings, int) and track_embeddings >= 1:
                embed_track_n = min(track_embeddings, _N_total)
            elif isinstance(track_embeddings, float) and 0.0 < track_embeddings < 1.0:
                embed_track_n = max(2, min(int(_N_total * track_embeddings), _N_total))
            else:
                _do_embed_tracking = False
                embed_track_n = 0
            if _do_embed_tracking:
                # Always the FIRST N samples so user labels align with original data order
                embed_track_idx = np.arange(embed_track_n)
        else:
            embed_track_n = 0

        # Rotation is only meaningful for critics with separate embedding networks.
        _has_embed_nets = hasattr(self.model, 'embedding_net_x')
        _do_rotation = False
        if return_rotated_embeddings:
            if not _has_embed_nets:
                warnings.warn(
                    "return_rotated_embeddings=True has no effect for ConcatCritic, which "
                    "has no separate embedding networks. Skipping rotation.",
                    UserWarning, stacklevel=2,
                )
            elif not _do_embed_tracking:
                warnings.warn(
                    "return_rotated_embeddings=True requires track_embeddings to be enabled. "
                    "No per-epoch embeddings are being tracked, so rotation will be skipped. "
                    "Set track_embeddings=True (or an integer/fraction) to enable rotation.",
                    UserWarning, stacklevel=2,
                )
            else:
                _do_rotation = True

        # AMP: only active on CUDA; silently no-ops on CPU/MPS.
        _amp_active = (
            self.device.type == 'cuda'
            and (self.use_amp is True or self.use_amp == 'auto')
        )
        _scaler = torch.amp.GradScaler('cuda') if _amp_active else None

        history, train_history, metrics_tracked, best_mi, no_improve = [], [], [], -float('inf'), 0
        embedding_history_x: list = []
        embedding_history_y: list = []
        embedding_history_x_rotated: list = []
        embedding_history_y_rotated: list = []
        _rotation_singular_values_history: list = []
        _rotation_history_x: list = []
        _rotation_history_y: list = []
        best_model_state = None
        # Improvement checkpoints: saved only when peak_fraction < 1.0 to avoid
        # unnecessary memory use.  Each entry is (epoch, smoothed_mi, state_dict).
        # The list is monotonically increasing in smoothed_mi by construction.
        _improvement_checkpoints = []
        nan_streak = 0
        
        display_progress = show_progress if show_progress is not None else verbose
        epoch_iterator = tqdm(range(n_epochs), desc=f"Run {run_id or ''}", leave=False,
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
                x_batch = dataset.x_dataset[batch_idx, ...].to(self.device)
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
                    if self.use_variational:
                        loss = kl_loss - self.beta * mi_estimate
                    else:
                        loss = -mi_estimate
                    # Optional decoder reconstruction loss
                    if self.decoder_x is not None or self.decoder_y is not None:
                        z_x, z_y = self.model.get_training_embeddings(x_batch, y_batch)
                        if self.decoder_x is not None:
                            recon_x = self.decoder_x(z_x)
                            loss = loss + self.decoder_weight_x * self._decoder_loss(
                                recon_x, x_batch, self.decoder_output_activation_x)
                        if self.decoder_y is not None:
                            recon_y = self.decoder_y(z_y)
                            loss = loss + self.decoder_weight_y * self._decoder_loss(
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
                x_test = dataset.x_dataset[test_view.indices, ...]
                y_test = dataset.y_dataset[test_view.indices, ...]
                mi_nats = self._safe_eval_mi(x_test, y_test, max_eval_samples)

                if _do_epoch_train_eval:
                    x_etrain = dataset.x_dataset[epoch_train_eval_idx, ...]
                    y_etrain = dataset.y_dataset[epoch_train_eval_idx, ...]
                    train_mi_nats = self._safe_eval_mi(x_etrain, y_etrain, max_eval_samples)
                    train_history.append(train_mi_nats)

                # Track spectral metrics if requested (can be expensive, so optional)
                if track_spectral_metrics:
                    metrics_during = self._extract_spectral_metrics(
                    dataset.x_dataset[train_eval_view.indices, ...],
                    dataset.y_dataset[train_eval_view.indices, ...],
                    spectral_output, return_spectrum)
                    metrics_tracked.append(metrics_during)

                # Per-epoch embedding tracking
                if _do_embed_tracking:
                    _zx_list, _zy_list = [], []
                    for _b_start in range(0, embed_track_n, batch_size):
                        _b_idx = embed_track_idx[_b_start:_b_start + batch_size]
                        _xb = dataset.x_dataset[_b_idx, ...].to(self.device)
                        _yb = dataset.y_dataset[_b_idx, ...].to(self.device)
                        _zx_b, _zy_b = self.model.get_embeddings(_xb, _yb)
                        _zx_list.append(_zx_b.cpu().numpy())
                        _zy_list.append(_zy_b.cpu().numpy())
                    embedding_history_x.append(np.concatenate(_zx_list, axis=0))
                    embedding_history_y.append(np.concatenate(_zy_list, axis=0))
                    if _do_rotation and rotated_embeddings_per_epoch:
                        _rot = compute_cross_covariance_rotation(
                            embedding_history_x[-1], embedding_history_y[-1],
                            whitening=rotated_embeddings_whitening,
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
                        f"(epochs {epoch + 2 - nan_streak}–{epoch + 1}). "
                        f"Check your learning_rate, batch_size, and input data for "
                        f"numerical instability (e.g. exploding gradients, zero-variance channels)."
                    )
                logger.warning(
                    f"NaN MI detected at epoch {epoch + 1} (consecutive NaN streak: "
                    f"{nan_streak}/3). This step will be skipped for early stopping. "
                    f"If this persists, check your data and hyperparameters."
                )
            else:
                nan_streak = 0
            history.append(mi_nats)

            # Smoothing (Custom or Default)
            if self.custom_smoothing_fn:
                smoothed_nats = self.custom_smoothing_fn(history)[-1]
            else:
                smoothed_nats = self._smooth(history, smoothing_sigma, median_window)[-1]
            
            has_valid_baseline = not np.isinf(best_mi)
            improvement = (smoothed_nats - best_mi) / (abs(best_mi) + 1e-8) if has_valid_baseline else float('inf')

            # Data Augmentation: Temporal Shifting
            if is_temporal and random_time_shifting:
                max_shift = np.clip(epoch / epochs_to_max_shift, 0, 1) * dataset.window_manager.window_size
                time_shift = np.random.uniform(high=max_shift)
                dataset.time_shift(offset_x=time_shift, offset_y=time_shift)

            if display_progress:
                epoch_iterator.set_description(f"Run {run_id or ''} | MI: {mi_nats * nats_to_bits:.3f}")

            # LR scheduler step
            if scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau as _ROP
                if isinstance(scheduler, _ROP):
                    if not np.isnan(mi_nats):
                        scheduler.step(mi_nats)
                else:
                    scheduler.step()

            # In-Memory Early Stopping
            if not np.isnan(smoothed_nats) and (improvement > min_improvement or np.isinf(best_mi) or best_model_state is None):
                best_mi, no_improve = smoothed_nats, 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                if peak_fraction < 1.0:
                    # best_model_state is a fresh deepcopy; appending it is safe —
                    # the next improvement will create a new deepcopy, not mutate this one.
                    _improvement_checkpoints.append((epoch, smoothed_nats, best_model_state))
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.debug(f"Early stopping at epoch {epoch+1}.")
                break
        
        if best_model_state is None:
            raise TrainingError("Training failed to produce a valid model checkpoint.")

        # 4. Finalization
        self.model.load_state_dict(best_model_state)
        if save_best_model_path:
            torch.save(best_model_state, save_best_model_path)
            
        with torch.no_grad():
            final_test_mi = self._safe_eval_mi(
                dataset.x_dataset[test_view.indices, ...], 
                dataset.y_dataset[test_view.indices, ...], max_eval_samples)
            final_train_mi = self._safe_eval_mi(
                dataset.x_dataset[train_eval_view.indices, ...], 
                dataset.y_dataset[train_eval_view.indices, ...], max_eval_samples)
        
        from neural_mi.estimators import infonce_lower_bound
        _eval_size = None
        if self.estimator_fn is infonce_lower_bound:
            n_eval = min(len(test_idx), max_eval_samples)
            _eval_size = n_eval
            eval_ceiling_nats = np.log(n_eval)
            if final_test_mi > 0.85 * eval_ceiling_nats:
                _scale = nats_to_bits  # 1/ln(2) for bits, 1.0 for nats
                _units = output_units
                _est_disp = final_test_mi * _scale
                _ceil_disp = eval_ceiling_nats * _scale
                logger.warning(
                    f"InfoNCE estimate ({_est_disp:.3f} {_units}) is near the ceiling "
                    f"for evaluation batch size log(n_eval={n_eval})={_ceil_disp:.3f} {_units}. "
                    f"The true MI may be higher. Consider increasing max_eval_samples or "
                    f"switching to the 'smile' estimator for high-MI scenarios."
                )

        best_ep = np.argmax(self.custom_smoothing_fn(history) if self.custom_smoothing_fn else self._smooth(history, smoothing_sigma, median_window))

        # U5: early stopping is effectively off by default (patience defaults to
        # 1000), so nothing else signals "training simply ran out of epochs while
        # test MI was still climbing." Since these are lower-bound estimators,
        # under-training always biases the reported value downward -- the
        # dangerous direction -- silently.
        if no_improve < patience and len(history) > 1 and best_ep >= len(history) - 1:
            warnings.warn(
                f"Training completed all {len(history)} epoch(s) without early "
                f"stopping, and the best (smoothed) test MI occurred at the final "
                f"epoch. MI may still have been increasing when training stopped, "
                f"so the reported estimate could be an under-trained lower bound. "
                f"Consider increasing n_epochs (or lowering patience to enable "
                f"early stopping).",
                UserWarning,
                stacklevel=2,
            )

        # All-negative flag (from test MI history; warning deferred until _raw_train_mi is set)
        valid_history = [v for v in history if not np.isnan(v)]
        _all_mi_negative = bool(valid_history and max(valid_history) <= 0)

        # At this point the model is loaded at best_ep; final_train_mi is the fresh eval there.
        _best_ep_train_mi = final_train_mi  # save for train_mi_at_peak reporting

        # Conservative epoch selection via improvement checkpoints.
        # Finds the first checkpoint where smoothed test MI >= peak_fraction * final_max,
        # loads that state, does a fresh full evaluation, then restores the best-epoch model.
        _conservative_ep = None
        _conservative_train_mi = None
        if peak_fraction < 1.0 and _improvement_checkpoints:
            _cons_state = None
            _max_smoothed = _improvement_checkpoints[-1][1]  # monotonically last = highest
            if _max_smoothed > 0:
                _threshold = peak_fraction * _max_smoothed
                for _ckpt_ep, _ckpt_sm, _ckpt_state in _improvement_checkpoints:
                    if _ckpt_sm >= _threshold:
                        _conservative_ep = _ckpt_ep
                        _cons_state = _ckpt_state
                        break
                if _cons_state is None:
                    # All checkpoints below threshold (shouldn't happen, but be safe)
                    _conservative_ep = best_ep
            else:
                _conservative_ep = best_ep  # no positive peak; conservative == best

            if _conservative_ep != best_ep and _cons_state is not None:
                self.model.load_state_dict(_cons_state)
                with torch.no_grad():
                    _conservative_train_mi = self._safe_eval_mi(
                        dataset.x_dataset[train_eval_view.indices, ...],
                        dataset.y_dataset[train_eval_view.indices, ...],
                        max_eval_samples,
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
                f"(max test MI = {max(valid_history):.4f} nats at epoch {best_ep}). "
                f"The model failed to learn a generalising representation — this typically "
                f"indicates too few epochs, too high a learning rate, or degenerate data. "
                f"Reporting train MI = 0 nats. The raw train MI was "
                f"{_raw_train_mi:.4f} nats (likely reflecting overfitting, not true MI). "
                f"Consider increasing n_epochs, reducing learning_rate, or inspecting data quality.",
                UserWarning,
                stacklevel=2,
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
        if _eval_size is not None:
            # eval_size = min(len(test_idx), max_eval_samples): the InfoNCE evaluation
            # denominator. The ceiling is log(eval_size), NOT log(batch_size) — exposed
            # here so callers (e.g. the dimensionality noise-injection ladder) can key
            # ceiling comparisons on it without recomputing the train/test split.
            results['eval_size'] = _eval_size
        if _conservative_ep is not None:
            results['conservative_epoch'] = _conservative_ep
            results['train_mi_at_peak'] = _best_ep_train_mi
        if _do_epoch_train_eval:
            results['train_mi_history'] = train_history

        if track_spectral_metrics:
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
                    whitening=rotated_embeddings_whitening,
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
                _tx = dataset.x_dataset[train_eval_view.indices, ...].to(self.device)
                _ty = dataset.y_dataset[train_eval_view.indices, ...].to(self.device)
                _zx, _zy = self.model.get_embeddings(_tx, _ty)  # uses existing no_grad method
                _recon_loss = 0.0
                if self.decoder_x is not None:
                    _recon_x = self.decoder_x(_zx)
                    _recon_loss += self.decoder_weight_x * float(
                        self._decoder_loss(_recon_x, _tx, self.decoder_output_activation_x).item())
                if self.decoder_y is not None:
                    _recon_y = self.decoder_y(_zy)
                    _recon_loss += self.decoder_weight_y * float(
                        self._decoder_loss(_recon_y, _ty, self.decoder_output_activation_y).item())
            results['decoder_recon_loss'] = _recon_loss

        # 5. Final Spectral Metrics (Dimensionality)
        metrics_final = self._extract_spectral_metrics(
            dataset.x_dataset[train_eval_view.indices, ...], 
            dataset.y_dataset[train_eval_view.indices, ...], 
            spectral_output, return_spectrum
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
            # recon: (B, C, W) — probability over C channels for each time step.
            # target: (B, C, W) — ground-truth (one-hot or soft target over channels).
            # Use distributional cross-entropy:
            #   L = -E_{b,w} [ sum_c target_{b,c,w} * log(recon_{b,c,w}) ].
            log_probs = torch.log(recon.clamp(min=1e-8))  # (B, C, W)
            loss_per_timestep = -(target * log_probs).sum(dim=1)  # (B, W)
            return loss_per_timestep.mean()
        else:
            # 'linear' or 'sigmoid': MSE is appropriate.
            return nn.functional.mse_loss(recon, target)

    def _safe_eval_mi(self, x: torch.Tensor, y: torch.Tensor, max_samples: int) -> float:
        """Evaluates MI on at most max_samples samples, drawn as a single random subset.
        If the dataset exceeds max_samples, draw ONE random
        subset of size max_samples and evaluate MI on that single set. This gives a
        valid (if higher-variance) unbiased estimate.

        Parameters
        ----------
        x : torch.Tensor
            Test set X data.
        y : torch.Tensor
            Test set Y data.
        max_samples : int
            Maximum number of samples for a single evaluation call. If the dataset
            is larger, a random subset of this size is drawn once.
        """
        n_samples = x.shape[0]
        if n_samples < 2:
            return float('nan')

        if n_samples > max_samples:
            # Sample once 
            idx = np.random.choice(n_samples, max_samples, replace=False)
            idx_t = torch.from_numpy(idx)
            x = x[idx_t]
            y = y[idx_t]

        result = self._eval_mi(x, y)
        if np.isnan(result):
            logger.warning(
                "MI evaluation returned NaN. This may indicate numerical instability, "
                "a degenerate batch, or exploding gradients. Check your learning_rate, "
                "batch_size, and input data for anomalies."
            )
        return result

    def _eval_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        scores, _ = self.model(x.to(self.device), y.to(self.device))
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

    def _extract_spectral_metrics(self, x: torch.Tensor, y: torch.Tensor, 
                                  spectral_output: str, return_spectrum: bool) -> Dict[str, Any]:
        """Extracts embeddings and computes cross-covariance spectral metrics."""
        self.model.eval()
        with torch.no_grad():
            zx, zy = self.model.get_embeddings(x.to(self.device), y.to(self.device))

        spectrum = compute_cross_covariance_spectrum(zx, zy, whitening=self.spectral_whitening)
        metrics = compute_spectral_metrics(spectrum)
        
        results = {}
        results['spectral_whitening'] = self.spectral_whitening
        if spectral_output == 'all':
            # Return all metrics: pr_eig, pr_singular, effective_rank, spectral_entropy
            results.update(metrics)
        else:
            # Default: both participation-ratio variants, without effective_rank/spectral_entropy
            results['pr_eig'] = metrics['pr_eig']
            results['pr_singular'] = metrics['pr_singular']

        if return_spectrum:
            results['spectrum'] = spectrum
            
        return results

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
        indices = np.random.permutation(n)
        n_train = int(n * frac)
        return indices[:n_train], indices[n_train:]

    def _create_blocked_split(self, n: int, frac: float, k: int, gap_fraction: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        n_test = int(n * (1 - frac))
        if n_test == 0:
            return np.arange(n), np.array([])
        if n_test < k:
            k = n_test
        block, rem = divmod(n_test, k) if k > 0 else (0, 0)
        if n - block < k or block + 1 <= 0:
            logger.warning(
                "Blocked split parameters produced an invalid configuration. "
                "Falling back to random split. Consider reducing n_test_blocks or "
                "increasing your dataset size."
            )
            indices = np.random.permutation(n)
            return indices[n_test:], indices[:n_test]
        starts = _sample_with_minimum_distance(n - block, k, block + 1)
        test_idx = np.concatenate([
            np.arange(s, s + block + (1 if i < rem else 0))
            for i, s in enumerate(starts)
        ])

        gap_size = int(round(gap_fraction * block)) if block > 0 else 0
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


