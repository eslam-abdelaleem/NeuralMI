# neural_mi/analysis/precision.py
"""Estimates spike-timing precision of a representation relative to a target.

This module trains a baseline mutual information estimator, then freezes the
network and repeatedly evaluates the test set across a grid of precision levels 
(tau). It corrupts the data using deterministic rounding or additive noise to 
find the precision threshold where mutual information degrades.
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from neural_mi.training.trainer import Trainer
from neural_mi.estimators import ESTIMATORS
from neural_mi.utils import build_critic, get_device
from neural_mi.data.handler import create_dataset
from neural_mi.logger import logger

def apply_corruption(data: torch.Tensor, tau: float, method: str) -> torch.Tensor:
    """Applies precision corruption to a tensor."""
    if tau == 0.0:
        return data
        
    if method == 'rounding':
        # Deterministic rounding to the nearest integer grid of tau
        return tau * torch.round(data / tau)
    elif method == 'noise':
        # Additive uniform noise centered around 0 (interval [-tau/2, tau/2])
        noise = (torch.rand_like(data) - 0.5) * tau
        return data + noise
    else:
        raise ValueError(f"Unknown corruption method: {method}")

def run_precision_analysis(
    x_data: Any, y_data: Any, base_params: Dict[str, Any], 
    tau_grid: List[float], corrupt_target: str = 'x', 
    corruption_method: str = 'rounding', n_noise_samples: int = 50,
    threshold_ratio: float = 0.9, **kwargs
) -> Dict[str, Any]:
    
    logger.info("Initializing Precision Analysis...")
    
    # 1. Prepare Data & Model
    dataset = create_dataset(
        x_data, y_data,
        processor_type_x=base_params.get('processor_type_x'),
        processor_type_y=base_params.get('processor_type_y'),
        processor_params_x=base_params.get('processor_params_x'),
        processor_params_y=base_params.get('processor_params_y')
    )
    
    # Update dimensions in base_params based on the dataset
    if dataset.x_data is not None and hasattr(dataset.x_data, 'shape'):
        base_params['input_dim_x'] = dataset.x_data.shape[1] * dataset.x_data.shape[2]
        base_params['n_channels_x'] = dataset.x_data.shape[1]
    if dataset.y_data is not None and hasattr(dataset.y_data, 'shape'):
        base_params['input_dim_y'] = dataset.y_data.shape[1] * dataset.y_data.shape[2]
        base_params['n_channels_y'] = dataset.y_data.shape[1]

    device = get_device(base_params.get('device'))
    
    if base_params.get('custom_critic') is not None:
        critic = base_params['custom_critic']
        logger.debug("Using provided custom critic.")
    else:
        critic = build_critic(base_params.get('critic_type', 'separable'), base_params, base_params.get('custom_embedding_cls'))
        
    optimizer = torch.optim.Adam(critic.parameters(), lr=base_params.get('learning_rate', 0.001))
    
    trainer = Trainer(
        model=critic.to(device),
        estimator_fn=ESTIMATORS[base_params.get('estimator_name', 'infonce')],
        optimizer=optimizer,
        device=device,
        use_variational=base_params.get('use_variational', False),
        beta=base_params.get('beta', 512.0),
        estimator_params=base_params.get('estimator_params')
    )
    
    # Determine train/test splits explicitly so we can reuse the exact test set for evaluation
    n_samples = len(dataset)
    train_frac = base_params.get('train_fraction', 0.9)
    split_mode = base_params.get('split_mode', 'blocked')
    if split_mode == 'random':
        train_idx, test_idx = trainer._create_random_split(n_samples, train_frac)
    else:
        train_idx, test_idx = trainer._create_blocked_split(n_samples, train_frac, kwargs.get('n_test_blocks', 5))
        
    # 2. Train the Baseline Model (Zero-Noise)
    logger.info("Training baseline model at maximum precision...")
    baseline_results = trainer.train(
        dataset, 
        n_epochs=base_params.get('n_epochs', 50),
        batch_size=base_params.get('batch_size', 256),
        patience=base_params.get('patience', 10),
        train_indices=train_idx,
        test_indices=test_idx,
        verbose=base_params.get('verbose', False),
        show_progress=base_params.get('show_progress', True),
        max_eval_samples=base_params.get('max_eval_samples', 5000),
        track_spectral_metrics=False # Skip dimensionality math to save time
    )
    
    baseline_mi = baseline_results['test_mi']
    logger.info(f"Baseline MI established: {baseline_mi:.3f} nats")
    
    # 3. The Precision Sweep (Inference Only)
    logger.info(f"Starting precision sweep using '{corruption_method}' on target '{corrupt_target}'...")
    trainer.model.eval()
    x_test_raw = dataset.x_dataset[test_idx, ...]
    y_test_raw = dataset.y_dataset[test_idx, ...]
    max_eval = base_params.get('max_eval_samples', 5000)
    
    results_list = []
    
    # Force 0.0 into the grid to log the exact baseline
    sorted_tau = sorted(list(set([0.0] + tau_grid)))
    
    with torch.no_grad():
        for tau in sorted_tau:
            if corruption_method == 'rounding':
                x_c = apply_corruption(x_test_raw, tau, 'rounding') if corrupt_target in ['x', 'both'] else x_test_raw
                y_c = apply_corruption(y_test_raw, tau, 'rounding') if corrupt_target in ['y', 'both'] else y_test_raw
                mi = trainer._safe_eval_mi(x_c.to(device), y_c.to(device), max_eval)
                results_list.append({'tau': tau, 'test_mi': mi, 'test_mi_std': 0.0})
                
            elif corruption_method == 'noise':
                # Average over multiple forward passes to stabilize stochastic noise bounds
                mis = []
                for _ in range(n_noise_samples if tau > 0 else 1):
                    x_c = apply_corruption(x_test_raw, tau, 'noise') if corrupt_target in ['x', 'both'] else x_test_raw
                    y_c = apply_corruption(y_test_raw, tau, 'noise') if corrupt_target in ['y', 'both'] else y_test_raw
                    mis.append(trainer._safe_eval_mi(x_c.to(device), y_c.to(device), max_eval))
                results_list.append({'tau': tau, 'test_mi': np.mean(mis), 'test_mi_std': np.std(mis)})

    df = pd.DataFrame(results_list)
    
    # 4. Find the Precision Threshold
    threshold_value = baseline_mi * threshold_ratio
    precision_tau = None
    
    # Iterate through ascending tau to find the exact point it drops below threshold
    for _, row in df.iterrows():
        if row['tau'] > 0 and row['test_mi'] < threshold_value:
            precision_tau = row['tau']
            break
            
    if precision_tau is None:
        logger.warning(f"MI never dropped below {threshold_ratio*100}% of baseline. Consider increasing the maximum tau.")
        
    logger.info(f"Precision Threshold estimated at tau = {precision_tau}")
    
    return {
        'dataframe': df,
        'details': {
            'baseline_mi': baseline_mi,
            'precision_tau': precision_tau,
            'threshold_ratio': threshold_ratio,
            'threshold_value': threshold_value,
            'corruption_method': corruption_method,
            'corrupt_target': corrupt_target
        }
    }