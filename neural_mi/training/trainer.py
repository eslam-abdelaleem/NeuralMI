# neural_mi/training/trainer.py
"""Handles the training and evaluation of critic models for MI estimation.

This module provides the `Trainer` class, a comprehensive utility for training
a critic model, monitoring its performance, implementing early stopping, and
saving the best-performing model state.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.ndimage import gaussian_filter1d, median_filter
import os
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, Optional, List, Callable, Union
import torch.nn as nn
import copy

from neural_mi.data import PairedDataset, PairedTemporalDataset, SubsetView
from neural_mi.logger import logger
from neural_mi.exceptions import TrainingError
from neural_mi.utils import compute_cross_covariance_spectrum, compute_spectral_metrics

def _ranks(sample: np.ndarray) -> List[int]:
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def _sample_with_minimum_distance(n: int, k: int, d: int) -> np.ndarray:
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
                 device: torch.device, use_variational: bool = False, beta: float = 512,
                 estimator_params: Optional[Dict[str, Any]] = None,
                 custom_smoothing_fn: Optional[Callable] = None,
                 spectral_whitening: str = 'std'):
        
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
            The weight to apply to the KL divergence term in the loss function,
            if `use_variational` is True. Defaults to 512.0.
        estimator_params : dict, optional
            Additional keyword arguments for the estimator function.
        custom_smoothing_fn : Callable, optional
            A custom function for smoothing the validation MI history, which takes
            a list of MI values and returns a smoothed array. If not provided, a default Gaussian + median filter will be used.
         spectral_whitening : str, optional
            Method for spectral whitening when computing spectral metrics. Options are 'std' for standard whitening
            and 'zca' for ZCA whitening and None. Defaults to 'std'.
       """
        self.device, self.model = device, model.to(device)
        self.estimator_fn, self.optimizer = estimator_fn, optimizer
        self.use_variational, self.beta = use_variational, beta
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.custom_smoothing_fn = custom_smoothing_fn
        self.spectral_whitening = spectral_whitening

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
              split_gap_fraction: float = 0.0,
              track_spectral_metrics: bool = False,
              spectral_output: str = 'default',
              return_spectrum: bool = False,
              max_index_reduction: float = 0.05) -> Dict[str, Any]:
        
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
            Defaults to 2.0.
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
            Determines the output of spectral metrics. If 'default', uses a predefined set of metrics.
            If 'full', returns the full spectrum. Defaults to 'default'.
        return_spectrum : bool, optional
            If True, includes the full spectrum in the returned results when `track_spectral_metrics` is True. Defaults to False.
        max_index_reduction : float, optional
            When using temporal datasets with windowing, random time shifting can reduce the number of valid windows
            due to edge effects. This parameter sets a threshold for acceptable reduction in valid windows after shifting.
            If the reduction exceeds this threshold, a warning is logged. Defaults to 0.05 (5%).

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

        # 1. Split Data
        if train_indices is not None and test_indices is not None:
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
        train_eval_idx = np.random.choice(train_idx, actual_train_subset_size, replace=False)
        train_eval_view = SubsetView(dataset, indices=train_eval_idx, max_index_reduction=max_index_reduction)
        
        history, metrics_tracked, best_mi, no_improve = [], [], -float('inf'), 0
        best_model_state = None
        
        display_progress = show_progress if show_progress is not None else verbose
        epoch_iterator = tqdm(range(n_epochs), desc=f"Run {run_id or ''}", leave=False,
                              disable=not display_progress)
        
        # 3. Epoch Loop
        for epoch in epoch_iterator:
            self.model.train()
            
            # Manual batching for efficiency and temporal shifting support
            current_train_idx = train_view.indices
            shuffled_train_idx = current_train_idx[torch.randperm(current_train_idx.nelement())]
            
            for batch_idx in shuffled_train_idx.split(batch_size):
                self.optimizer.zero_grad()
                scores, kl_loss = self.model(dataset.x_dataset[batch_idx, ...].to(self.device), dataset.y_dataset[batch_idx, ...].to(self.device))
                loss = -self.estimator_fn(scores, **self.estimator_params)
                if self.use_variational:
                    loss += self.beta * kl_loss
                loss.backward()
                self.optimizer.step()

            # Fast evaluation using safety chunking
            self.model.eval()
            with torch.no_grad():
                x_test = dataset.x_dataset[test_view.indices, ...]
                y_test = dataset.y_dataset[test_view.indices, ...]
                mi_nats = self._safe_eval_mi(x_test, y_test, max_eval_samples)

                # Track spectral metrics if requested (can be expensive, so optional)
                if track_spectral_metrics:
                    metrics_during = self._extract_spectral_metrics(
                    dataset.x_dataset[train_eval_view.indices, ...], 
                    dataset.y_dataset[train_eval_view.indices, ...], 
                    spectral_output, return_spectrum)            
                    metrics_tracked.append(metrics_during)
            
            history.append(mi_nats)
            
            # Smoothing (Custom or Default)
            if self.custom_smoothing_fn:
                smoothed_nats = self.custom_smoothing_fn(history)[-1]
            else:
                smoothed_nats = self._smooth(history, smoothing_sigma, median_window)[-1]
            
            is_first_valid_epoch = not np.isinf(best_mi)
            improvement = (smoothed_nats - best_mi) / (abs(best_mi) + 1e-8) if is_first_valid_epoch else float('inf')

            # Data Augmentation: Temporal Shifting
            if is_temporal and random_time_shifting:
                max_shift = np.clip(epoch / epochs_to_max_shift, 0, 1) * dataset.window_manager.window_size
                time_shift = np.random.uniform(high=max_shift)
                dataset.time_shift(offset_x=time_shift, offset_y=time_shift)

            if display_progress:
                epoch_iterator.set_description(f"Run {run_id or ''} | MI: {mi_nats * nats_to_bits:.3f}")
            
            # In-Memory Early Stopping
            if not np.isnan(smoothed_nats) and (improvement > min_improvement or np.isinf(best_mi) or best_model_state is None):
                best_mi, no_improve = smoothed_nats, 0
                best_model_state = copy.deepcopy(self.model.state_dict())
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
        if self.estimator_fn is infonce_lower_bound:
            n_eval = min(len(test_idx), max_eval_samples)
            eval_ceiling_nats = np.log(n_eval)
            if final_test_mi > 0.85 * eval_ceiling_nats:
                logger.warning(
                    f"InfoNCE estimate ({final_test_mi:.3f} nats) is near the ceiling "
                    f"for evaluation batch size log(n_eval={n_eval})={eval_ceiling_nats:.3f} nats. "
                    f"The true MI may be higher. Consider increasing max_eval_samples or "
                    f"switching to the 'smile' estimator for high-MI scenarios."
                )

        best_ep = np.argmax(self.custom_smoothing_fn(history) if self.custom_smoothing_fn else self._smooth(history, smoothing_sigma, median_window))
        
        results = {
            'train_mi': final_train_mi, 
            'test_mi': final_test_mi,
            'best_epoch': best_ep,
            'test_mi_history': history
        }

        if track_spectral_metrics:
             results['spectral_metrics_history'] = metrics_tracked

        # 5. Final Spectral Metrics (Dimensionality)
        metrics_final = self._extract_spectral_metrics(
            dataset.x_dataset[train_eval_view.indices, ...], 
            dataset.y_dataset[train_eval_view.indices, ...], 
            spectral_output, return_spectrum
        )
        results.update(metrics_final)

        return results

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

        return self._eval_mi(x, y)

    def _eval_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        scores, _ = self.model(x.to(self.device), y.to(self.device))
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
            results = metrics
        else:
            results['participation_ratio'] = metrics['pr_singular']
            
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
            if gap_size > 0:
                logger.debug(
                    f"Blocked split gap: excluded {len(excluded)} samples "
                    f"({gap_size} samples per block boundary) from training set."
                )
        else:
            train_idx = np.setdiff1d(np.arange(n), test_idx)

        return train_idx, test_idx


