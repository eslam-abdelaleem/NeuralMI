# neural_mi/analysis/workflow.py

import torch
import numpy as np
import pandas as pd
import itertools
import uuid
import warnings
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from collections import Counter

from neural_mi.estimators import bounds
from neural_mi.utils import run_training_task


def __find_linear_region(group, delta_threshold, min_gamma_points, verbose):
    """
    Iteratively prunes gamma values to find the linear region of the bias curve.
    """
    gammas_to_fit = sorted(group['gamma'].unique())
    final_delta = float('inf')

    while len(gammas_to_fit) >= min_gamma_points:
        subset = group[group['gamma'].isin(gammas_to_fit)]
        if len(subset) < 3:
            break  # Not enough points for a quadratic fit

        gamma_counts = subset['gamma'].value_counts()
        weights = subset['gamma'].map(lambda g: 1 / gamma_counts[g])
        if verbose:
            print(f'Total number of successful used runs: {len(weights)}')
            print(f'Quadratic fit weights: {weights}')

        X_quad = sm.add_constant(np.vstack([subset['gamma'], subset['gamma']**2]).T)

        model_quad = sm.WLS(subset['test_mi'], X_quad, weights=weights).fit()
        _, a1, a2 = model_quad.params
        final_delta = abs(a2 / a1) if a1 != 0 else float('inf')

        if verbose:
            print(f"  Fitting gammas {gammas_to_fit}: delta = {final_delta:.4f}")

        if final_delta < delta_threshold:
            if verbose:
                print(f"  Delta ({final_delta:.4f}) < threshold ({delta_threshold}). Stopping pruning.")
            break
        else:
            pruned_gamma = gammas_to_fit.pop(-1)
            if verbose:
                print(f"  Delta > threshold. Pruning gamma = {pruned_gamma}.")

    return gammas_to_fit, final_delta


def __extrapolate_mi(group, gammas_to_fit, confidence_level, verbose):
    """
    Performs the final WLS linear fit to extrapolate the MI estimate.
    """
    final_subset = group[group['gamma'].isin(gammas_to_fit)]

    if len(final_subset) < 2:
        warnings.warn("Not enough points for even an unreliable linear fit. Returning NaN.")
        return float('nan'), float('nan'), float('nan')

    final_gamma_counts = final_subset['gamma'].value_counts()
    final_weights = final_subset['gamma'].map(lambda g: 1 / final_gamma_counts[g])
    if verbose:
        print(f'Total number of successful used runs: {len(final_weights)}')
        print(f'Linear fit weights: {final_weights}')

    X_linear = sm.add_constant(final_subset['gamma'])
    fit_linear = sm.WLS(final_subset['test_mi'], X_linear, weights=final_weights).fit()

    intercept, slope = fit_linear.params
    alpha = 1 - confidence_level
    conf_interval = fit_linear.get_prediction(exog=[1, 0]).conf_int(obs=True, alpha=alpha)[0]
    mi_error = (conf_interval[1] - conf_interval[0]) / 2.0

    return intercept, mi_error, slope


def _post_process_and_correct(df, delta_threshold, min_gamma_points, confidence_level, verbose):
    """Performs iterative WLS fitting on all individual points."""
    df = df.dropna(subset=['gamma', 'test_mi'])

    param_cols = [c for c in df.columns if c not in ['gamma', 'train_mi', 'test_mi', 'best_epoch', 'test_mi_history']]

    corrected_results = []
    group_keys = [col for col in param_cols if col in df.columns]
    if not group_keys:
        df['dummy_group'] = 0
        group_keys = ['dummy_group']

    for params, group in df.groupby(group_keys):
        param_dict = dict(zip(group_keys, [params])) if len(group_keys) == 1 and not isinstance(params, tuple) else dict(zip(group_keys, params))

        if verbose:
            print(f"\n--- Correcting for params: {param_dict} ---")

        gammas_used, final_delta = __find_linear_region(group, delta_threshold, min_gamma_points, verbose)

        is_reliable = True
        if len(gammas_used) < min_gamma_points:
            is_reliable = False
            warnings.warn(f"Fit for {param_dict} is unreliable (final gamma points < {min_gamma_points}).")

        mi_corrected, mi_error, slope = __extrapolate_mi(group, gammas_used, confidence_level, verbose)

        # The R-squared warning was removed to align with the user's preference for the
        # quadratic pruning (delta-check) method to determine fit reliability.
        # The `is_reliable` flag, based on having enough points after pruning, is now the primary indicator.

        param_dict.update({
            'mi_corrected': mi_corrected, 'mi_error': mi_error, 'slope': slope,
            'is_reliable': is_reliable, 'gammas_used': gammas_used, 'final_delta': final_delta
        })
        corrected_results.append(param_dict)

    return corrected_results


class AnalysisWorkflow:
    """
    Orchestrates the full, rigorous MI analysis including iterative bias correction.
    """
    def __init__(self, x_data, y_data, base_params, critic_type='separable',
                 estimator_fn=bounds.infonce_lower_bound, use_variational=False):
        self.x_data = x_data; self.y_data = y_data; self.base_params = base_params
        self.base_params.update({
            'critic_type': critic_type, 'estimator_fn': estimator_fn,
            'use_variational': use_variational,
            'input_dim_x': x_data.shape[1] * x_data.shape[2],
            'input_dim_y': y_data.shape[1] * y_data.shape[2]
        })

    def run(self, param_grid, gamma_range=range(1, 11), n_workers=None,
            delta_threshold=0.1, min_gamma_points=5, confidence_level=0.68, verbose=False):
        if n_workers is None: n_workers = cpu_count()
        print(f"Starting rigorous analysis with {n_workers} workers...")
        tasks = self._prepare_tasks(param_grid, gamma_range)
        if not tasks: print("No tasks to run."); return []

        with Pool(processes=n_workers) as pool:
            raw_results = list(pool.map(run_training_task, tasks))
        
        print("All training tasks finished. Performing bias correction...")
        
        raw_results_df = pd.DataFrame(raw_results)
        corrected_results = _post_process_and_correct(raw_results_df, delta_threshold, min_gamma_points, confidence_level, verbose)
        
        return {
            "corrected_results": corrected_results,
            "raw_results_df": raw_results_df
        }

    def _prepare_tasks(self, param_grid, gamma_range):
        tasks = []; run_id_base = str(uuid.uuid4())
        if self.base_params['critic_type'] == 'concat' and 'embedding_dim' in param_grid:
            param_grid.pop('embedding_dim')
            if not param_grid: param_grid = {'_dummy': [None]}
        keys, values = zip(*param_grid.items()) if param_grid else ([], [])
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if param_grid else [{}]
        for i_combo, params in enumerate(param_combinations):
            current_params = self.base_params.copy(); current_params.update(params)
            for gamma in gamma_range:
                current_params['gamma'] = gamma
                indices = np.random.permutation(self.x_data.shape[0])
                subset_indices_list = np.array_split(indices, gamma)
                for i_subset, subset_indices in enumerate(subset_indices_list):
                    x_subset = self.x_data[subset_indices]; y_subset = self.y_data[subset_indices]
                    task_run_id = f"{run_id_base}_c{i_combo}_g{gamma}_s{i_subset}"
                    tasks.append((x_subset, y_subset, current_params.copy(), task_run_id))
        print(f"Created {len(tasks)} tasks to run...")
        return tasks