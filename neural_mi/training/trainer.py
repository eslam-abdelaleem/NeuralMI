# neural_mi/training/trainer.py

import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.ndimage import gaussian_filter1d, median_filter
import os
import tempfile
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, Optional, List
import torch.nn as nn

from neural_mi.logger import logger
from neural_mi.exceptions import TrainingError

def _ranks(sample: np.ndarray) -> List[int]:
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def _sample_with_minimum_distance(n: int, k: int, d: int) -> np.ndarray:
    sample = np.random.choice(n - (k - 1) * (d - 1), k, replace=False)
    return np.array([s + (d - 1) * r for s, r in zip(sample, _ranks(sample))])

class Trainer:
    def __init__(self, model: nn.Module, estimator_fn, optimizer, device: torch.device,
                 use_variational: bool = False, beta: float = 1.0):
        self.device, self.model = device, model.to(device)
        self.estimator_fn, self.optimizer = estimator_fn, optimizer
        self.use_variational, self.beta = use_variational, beta

    def train(self, x_data: torch.Tensor, y_data: torch.Tensor, n_epochs: int, batch_size: int,
              train_fraction: float = 0.9, n_test_blocks: int = 5, patience: int = 10,
              smoothing_sigma: float = 2.0, median_window: int = 5, min_improvement: float = 0.001,
              save_best_model_path: Optional[str] = None, run_id: Optional[str] = None,
              output_units: str = 'nats', verbose: bool = True) -> Dict[str, Any]:
        
        nats_to_bits = 1 / np.log(2) if output_units == 'bits' else 1.0

        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            train_idx, test_idx = self._create_blocked_split(x_data.shape[0], train_fraction, n_test_blocks)
            
            n_train = len(train_idx)
            if batch_size > n_train > 0:
                logger.warning(f"batch_size ({batch_size}) > n_train_samples ({n_train}). Reducing to {n_train}.")
                batch_size = n_train
            if batch_size < 2 and n_train > 1: raise ValueError(f"batch_size must be >= 2, got {batch_size}.")

            loader = DataLoader(torch.utils.data.TensorDataset(x_data, y_data), batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
            x_train, y_train = x_data[train_idx], y_data[train_idx]
            x_test, y_test = x_data[test_idx], y_data[test_idx]
            
            history, best_mi, no_improve = [], -float('inf'), 0
            
            epoch_iterator = tqdm(range(n_epochs), desc=f"Run {run_id or ''}", leave=False, disable=not verbose)

            for epoch in epoch_iterator:
                self.model.train()
                for x_batch, y_batch in loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    loss = -self.estimator_fn(self.model(x_batch, y_batch))
                    if self.use_variational:
                        kl_loss = getattr(self.model.embedding_net_x, 'kl_loss', 0) + \
                                  getattr(self.model.embedding_net_y, 'kl_loss', 0)
                        loss -= self.beta * kl_loss
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    mi_nats = float('nan') if x_test.shape[0] < 2 else self.estimator_fn(self.model(x_test, y_test)).item()
                history.append(mi_nats)
                
                smoothed_nats = self._smooth(history, smoothing_sigma, median_window)[-1]
                
                is_first_valid_epoch = not np.isinf(best_mi)
                improvement = (smoothed_nats - best_mi) / (abs(best_mi) + 1e-8) if is_first_valid_epoch else float('inf')
                
                if verbose:
                    epoch_iterator.set_description(f"Run {run_id or ''} | MI: {mi_nats * nats_to_bits:.3f}")
                
                if not np.isnan(smoothed_nats) and (improvement > min_improvement or np.isinf(best_mi)):
                    best_mi, no_improve = smoothed_nats, 0
                    torch.save(self.model.state_dict(), tmp_path)
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.debug(f"Early stopping at epoch {epoch+1}.")
                    break
            
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise TrainingError("Training failed to produce a valid model checkpoint.")

            self.model.load_state_dict(torch.load(tmp_path, map_location=self.device, weights_only=True))
            with torch.no_grad():
                train_mi = self._eval_mi(x_train, y_train)
                test_mi = self._eval_mi(x_test, y_test)
            
            if save_best_model_path: os.rename(tmp_path, save_best_model_path)
            
            return {'train_mi': train_mi, 'test_mi': test_mi,
                    'best_epoch': np.argmax(self._smooth(history, smoothing_sigma, median_window)),
                    'test_mi_history': history}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _eval_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return float('nan') if x.shape[0] < 2 else self.estimator_fn(self.model(x, y)).item()

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