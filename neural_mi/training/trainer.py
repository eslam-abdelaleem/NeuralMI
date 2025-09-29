# neural_mi/training/trainer.py

import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from scipy.ndimage import gaussian_filter1d, median_filter
import os
import uuid
import warnings
import tempfile
from tqdm.auto import tqdm

def _ranks(sample):
    """Helper function to return the ranks of each element in a sample."""
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def _sample_with_minimum_distance(n, k, d):
    """Samples k elements from range(n) with a minimum distance d."""
    sample = np.random.choice(n - (k - 1) * (d - 1), k, replace=False)
    return np.array([s + (d - 1) * r for s, r in zip(sample, _ranks(sample))])

class Trainer:
    """
    An advanced trainer for MI estimation with early stopping, model checkpointing,
    and robust smoothing for noisy evaluation curves.
    """
    def __init__(self, model, estimator_fn, optimizer, device, use_variational=False, beta=1.0):
        self.device = device
        self.model = model.to(self.device)
        self.estimator_fn = estimator_fn
        self.optimizer = optimizer
        self.use_variational = use_variational
        self.beta = beta

    def train(self, x_data, y_data, n_epochs, batch_size,
              train_fraction=0.9, n_test_blocks=5, patience=10,
              sigma=1.0, median_window=3,
              save_best_model_path=None, run_id=None):
        
        x_data = x_data.to(self.device)
        y_data = y_data.to(self.device)
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as temp_model_file:
            temp_model_path = temp_model_file.name

            train_indices, test_indices = self._create_blocked_split(
                n_samples=x_data.shape[0], train_fraction=train_fraction, n_test_blocks=n_test_blocks)

            train_loader = DataLoader(torch.utils.data.TensorDataset(x_data, y_data),
                                      batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
            x_train, y_train = x_data[train_indices], y_data[train_indices]
            x_test, y_test = x_data[test_indices], y_data[test_indices]
            
            test_mi_history = []
            best_smoothed_mi = -float('inf')
            epochs_no_improve = 0

            epoch_pbar = tqdm(range(n_epochs), desc="Training Progress")

            for epoch in epoch_pbar:
                self.model.train()

                for x_batch, y_batch in train_loader:
                    self.optimizer.zero_grad()
                    scores = self.model(x_batch, y_batch)
                    mi_estimate = self.estimator_fn(scores)
                    if self.use_variational:
                        kl_loss = self.model.embedding_net_x.kl_loss + self.model.embedding_net_y.kl_loss
                        loss = kl_loss - self.beta * mi_estimate
                    else:
                        loss = -mi_estimate
                    loss.backward()
                    self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    test_mi = float('nan') if x_test.shape[0] < 2 else self.estimator_fn(self.model(x_test, y_test)).item()
                test_mi_history.append(test_mi)

                epoch_pbar.set_description(f"Epoch {epoch+1}/{n_epochs} | Test MI: {test_mi:.4f}")

                current_smoothed_mi = self._smooth(test_mi_history, sigma, median_window)[-1]
                if not np.isnan(current_smoothed_mi) and current_smoothed_mi > best_smoothed_mi:
                    best_smoothed_mi = current_smoothed_mi
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), temp_model_path)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                    break

            if not test_mi_history or not os.path.exists(temp_model_path):
                 return {'train_mi': float('nan'), 'test_mi': float('nan'), 'best_epoch': -1, 'test_mi_history': []}

            smoothed_history = self._smooth(test_mi_history, sigma, median_window)
            best_epoch = np.argmax(smoothed_history)

            self.model.load_state_dict(torch.load(temp_model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            with torch.no_grad():
                final_train_mi = self._evaluate_final_mi(x_train, y_train)
                final_test_mi = self._evaluate_final_mi(x_test, y_test)
            
            print(f"Best epoch identified (via smoothed curve): {best_epoch + 1} (Final MI: {final_train_mi:.4f})")

            if save_best_model_path:
                # If the temporary file still exists, move it.
                if os.path.exists(temp_model_path):
                    os.rename(temp_model_path, save_best_model_path)
            
        return {
            'train_mi': final_train_mi, 'test_mi': final_test_mi,
            'best_epoch': best_epoch, 'test_mi_history': test_mi_history
        }

    def _evaluate_final_mi(self, x, y):
        if x.shape[0] < 2:
            return float('nan')
        scores = self.model(x, y)
        return self.estimator_fn(scores).item()

    def _smooth(self, mi_array, sigma, median_window):
        history = np.array(mi_array)
        if np.all(np.isnan(history)): return history
        nan_mask = np.isnan(history)
        if np.any(nan_mask):
            last_valid = history[~nan_mask][-1] if np.any(~nan_mask) else 0
            history[nan_mask] = last_valid
        if median_window > 1 and len(history) >= median_window:
            history = median_filter(history, size=median_window, mode='reflect')
        if sigma > 0 and len(history) > 1:
            history = gaussian_filter1d(history, sigma=sigma, mode='reflect')
        return history

    def _create_blocked_split(self, n_samples, train_fraction, n_test_blocks):
        n_test = int(n_samples * (1 - train_fraction))
        if n_test < n_test_blocks:
            warnings.warn(f"Reducing n_test_blocks to {n_test}.")
            n_test_blocks = n_test
        if n_test == 0: return np.arange(n_samples), np.array([])
        block_size, remainder = divmod(n_test, n_test_blocks)
        min_dist = block_size + 1
        max_start_pos = n_samples - block_size
        if max_start_pos < n_test_blocks:
            warnings.warn("Not enough space for blocked split, falling back to random split.")
            indices = np.random.permutation(n_samples)
            return indices[n_test:], indices[:n_test]
        start_inds = _sample_with_minimum_distance(n=max_start_pos, k=n_test_blocks, d=min_dist)
        test_indices = []
        for i in range(n_test_blocks):
            size = block_size + (1 if i < remainder else 0)
            test_indices.extend(np.arange(start_inds[i], start_inds[i] + size))
        return np.setdiff1d(np.arange(n_samples), test_indices), np.array(test_indices)