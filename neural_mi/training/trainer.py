# neural_mi/training/trainer.py
"""Handles the training and evaluation of critic models for MI estimation.

This module provides the `Trainer` class, a comprehensive utility for training
a critic model, monitoring its performance, implementing early stopping, and
saving the best-performing model state.
"""
import torch
import tempfile
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.ndimage import gaussian_filter1d, median_filter
import os
import shutil
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, Optional, List, Callable, Union
import torch.nn as nn

from neural_mi.data import PairedDataset, PairedTemporalDataset, SubsetView
from neural_mi.logger import logger
from neural_mi.exceptions import TrainingError

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
                 estimator_params: Optional[Dict[str, Any]] = None):
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
        """
        self.device, self.model = device, model.to(device)
        self.estimator_fn, self.optimizer = estimator_fn, optimizer
        self.use_variational, self.beta = use_variational, beta
        self.estimator_params = estimator_params if estimator_params is not None else {}

    def train(self, dataset : Union[PairedDataset, PairedTemporalDataset], n_epochs: int, batch_size: int,
              train_fraction: float = 0.9, n_test_blocks: int = 5, 
              random_time_shifting : bool = True, epochs_to_max_shift : int = 5,
              patience: int = 10, smoothing_sigma: float = 2.0, median_window: int = 5, min_improvement: float = 0.001,
              save_best_model_path: Optional[str] = None, run_id: Optional[str] = None,
              output_units: str = 'nats', verbose: bool = True,
              split_mode: str = 'blocked',
              train_indices: Optional[np.ndarray] = None,
              test_indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Trains the critic model and returns performance metrics.

        This method implements the main training loop, including data splitting,
        training, evaluation, and early stopping.

        Parameters
        ----------
        x_data : torch.Tensor
            The complete dataset for the first variable, X.
        y_data : torch.Tensor
            The complete dataset for the second variable, Y.
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
            If True, a progress bar will be displayed. Defaults to True.
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

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of the training run.
        """
        nats_to_bits = 1 / np.log(2) if output_units == 'bits' else 1.0

        # x_data = x_data.to(self.device)
        # y_data = y_data.to(self.device)
        is_temporal = True if isinstance(dataset, PairedTemporalDataset) else False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "best_model.pt")

            if train_indices is not None and test_indices is not None:
                logger.debug("Using user-provided train and test indices.")
                train_idx, test_idx = train_indices, test_indices
            elif split_mode == 'random':
                logger.debug("Using random train/test split.")
                train_idx, test_idx = self._create_random_split(len(dataset), train_fraction)
            else: # 'blocked' is the default
                logger.debug("Using blocked train/test split for temporal data.")
                train_idx, test_idx = self._create_blocked_split(len(dataset), train_fraction, n_test_blocks)
            
            n_train = len(train_idx)
            if batch_size > n_train > 0:
                logger.warning(f"batch_size ({batch_size}) > n_train_samples ({n_train}). Reducing to {n_train}.")
                batch_size = n_train
            if batch_size < 2 and n_train > 1: raise ValueError(f"batch_size must be >= 2, got {batch_size}.")

            # Subset dataset into train and test views
            train_view = SubsetView(dataset, indices=train_idx)
            test_view = SubsetView(dataset, indices=test_idx)
            
            history, best_mi, no_improve = [], -float('inf'), 0
            best_model_saved = False
            
            epoch_iterator = tqdm(range(n_epochs), desc=f"Run {run_id or ''}", leave=False, disable=not verbose)

            for epoch in epoch_iterator:
                self.model.train()
                # Shuffle trainig indices each epoch
                train_idx = train_view.indices
                shuffled_train_idx = train_idx[torch.randperm(train_idx.nelement())]
                for batch_idx in shuffled_train_idx.split(batch_size):
                    
                    self.optimizer.zero_grad()
                    scores, kl_loss = self.model(dataset.x_dataset[batch_idx, ...], dataset.y_dataset[batch_idx, ...])
                    loss = -self.estimator_fn(scores, **self.estimator_params)
                    if self.use_variational:
                        loss += self.beta * kl_loss
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    x_test = dataset.x_dataset[test_view.indices, ...]
                    y_test = dataset.y_dataset[test_view.indices, ...]
                    mi_nats = self._eval_mi(x_test, y_test)
                history.append(mi_nats)
                
                smoothed_nats = self._smooth(history, smoothing_sigma, median_window)[-1]
                
                is_first_valid_epoch = not np.isinf(best_mi)
                improvement = (smoothed_nats - best_mi) / (abs(best_mi) + 1e-8) if is_first_valid_epoch else float('inf')

                # Random time shift if data is temporal
                if is_temporal and random_time_shifting:
                    max_shift = np.clip(epoch / epochs_to_max_shift, 0, 1) * dataset.window_manager.window_size
                    time_shift = np.random.uniform(high=max_shift)
                    dataset.time_shift(offset_x=time_shift, offset_y=time_shift)

                
                if verbose:
                    epoch_iterator.set_description(f"Run {run_id or ''} | MI: {mi_nats * nats_to_bits:.3f}")
                
                # Save if improved OR if it's the first valid epoch (to ensure we have at least one checkpoint)
                if not np.isnan(smoothed_nats) and (improvement > min_improvement or np.isinf(best_mi) or not best_model_saved):
                    best_mi, no_improve = smoothed_nats, 0
                    torch.save(self.model.state_dict(), tmp_path)
                    best_model_saved = True
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.debug(f"Early stopping at epoch {epoch+1}.")
                    break
            
            if not best_model_saved:
                raise TrainingError("Training failed to produce a valid model checkpoint.")

            self.model.load_state_dict(torch.load(tmp_path, map_location=self.device, weights_only=True))
            with torch.no_grad():
                train_mi = self._eval_mi(
                    dataset.x_dataset[train_view.indices, ...], 
                    dataset.y_dataset[train_view.indices, ...]
                )
                test_mi = self._eval_mi(
                    dataset.x_dataset[test_view.indices, ...], 
                    dataset.y_dataset[test_view.indices, ...]
                )
            
            if save_best_model_path:
                shutil.copy(tmp_path, save_best_model_path)
            
            return {'train_mi': train_mi, 'test_mi': test_mi,
                    'best_epoch': np.argmax(self._smooth(history, smoothing_sigma, median_window)),
                    'test_mi_history': history}


    def _eval_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        if x.shape[0] < 2:
            return float('nan')
        scores, _ = self.model(x, y)
        return self.estimator_fn(scores, **self.estimator_params).item()

    def _smooth(self, arr: List[float], sigma: float, med_win: int) -> np.ndarray:
        hist = np.array(arr)
        if len(hist) < 2: return hist
        nan_mask = np.isnan(hist)
        valid_hist = hist[~nan_mask]
        if len(valid_hist) == 0: return hist
        hist[nan_mask] = valid_hist[-1]
        if med_win > 1 and len(hist) >= med_win:
            hist = median_filter(hist, size=med_win, mode='reflect')
        if sigma > 0: hist = gaussian_filter1d(hist, sigma=sigma, mode='reflect')
        hist[nan_mask] = np.nan
        return hist
        
    def _create_random_split(self, n: int, frac: float) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a simple random split of indices."""
        indices = np.random.permutation(n)
        n_train = int(n * frac)
        return indices[:n_train], indices[n_train:]

    def _create_blocked_split(self, n: int, frac: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
        n_test = int(n * (1 - frac))
        if n_test == 0: return np.arange(n), np.array([])
        if n_test < k: k = n_test
        block, rem = divmod(n_test, k) if k > 0 else (0, 0)
        if n - block < k or block + 1 <= 0:
                indices = np.random.permutation(n)
                return indices[n_test:], indices[:n_test]
        starts = _sample_with_minimum_distance(n - block, k, block + 1)
        test_idx = np.concatenate([np.arange(s, s + block + (1 if i<rem else 0)) for i, s in enumerate(starts)])
        return np.setdiff1d(np.arange(n), test_idx), test_idx