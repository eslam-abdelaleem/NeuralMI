# neural_mi/datasets/generators.py
"""Generates synthetic datasets for testing and validating MI estimators.

This module provides functions to create various kinds of synthetic datasets
with known or expected properties. These are useful for tutorials, debugging,
and validating the behavior of different MI estimators and models.
"""

import numpy as np
import torch
from typing import Tuple, List

def mi_to_rho(dim: int, mi: float) -> float:
    """Calculates the correlation coefficient `rho` for a given MI and dimension.

    This function is used for generating correlated Gaussian variables with a
    pre-defined mutual information. The formula is derived from the analytical
    expression for MI between two multivariate Gaussian variables.

    Parameters
    ----------
    dim : int
        The dimension of the Gaussian variables.
    mi : float
        The desired mutual information in bits.

    Returns
    -------
    float
        The corresponding correlation coefficient `rho`.
    """
    # Convert MI from bits to nats for the formula
    mi_nats = mi * np.log(2)
    return np.sqrt(1 - np.exp(-2.0 / dim * mi_nats))

def generate_correlated_gaussians(
    n_samples: int, dim: int, mi: float, use_torch: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two correlated multivariate Gaussian datasets.

    The ground truth mutual information between these two variables can be
    calculated analytically.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    dim : int
        The number of dimensions for each variable.
    mi : float
        The ground truth mutual information in bits.
    use_torch : bool, optional
        If True, returns torch.Tensors; otherwise, returns NumPy arrays.
        Defaults to True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - **x** (*np.ndarray* or *torch.Tensor*): The first dataset, of shape `(n_samples, dim)`.
        - **y** (*np.ndarray* or *torch.Tensor*): The second dataset, of shape `(n_samples, dim)`.
    """
    rho = mi_to_rho(dim, mi)
    mean = np.zeros(2 * dim)
    cov = np.eye(2 * dim)
    cov[dim:, :dim] = np.eye(dim) * rho
    cov[:dim, dim:] = np.eye(dim) * rho
    
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    x = data[:, :dim]
    y = data[:, dim:]
    
    if use_torch:
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y

def generate_nonlinear_from_latent(
    n_samples: int, latent_dim: int, observed_dim: int, mi: float, 
    hidden_dim: int = 64, use_torch: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two nonlinearly related datasets from a shared latent variable.

    A low-dimensional latent variable `z` is first generated. Two observed
    variables, `x` and `y`, are then created as nonlinear projections of `z`
    with added noise.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    latent_dim : int
        The dimensionality of the shared latent variable `z`.
    observed_dim : int
        The dimensionality of the observed variables `x` and `y`.
    mi : float
        The ground truth MI between the latent variables Z_x and Z_y in bits.
    hidden_dim : int, optional
        The hidden dimension of the transforming MLPs. Defaults to 64.
    use_torch : bool, optional
        If True, returns torch.Tensors. Defaults to True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - **x** (*np.ndarray* or *torch.Tensor*): The first dataset, of shape `(n_samples, observed_dim)`.
        - **y** (*np.ndarray* or *torch.Tensor*): The second dataset, of shape `(n_samples, observed_dim)`.
    """
    z_x, z_y = generate_correlated_gaussians(n_samples, latent_dim, mi, use_torch=True)
    
    mlp_x = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, hidden_dim), torch.nn.Softplus(),
        torch.nn.Linear(hidden_dim, observed_dim)
    )
    mlp_y = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, hidden_dim), torch.nn.Softplus(),
        torch.nn.Linear(hidden_dim, observed_dim)
    )
    
    with torch.no_grad():
        x = mlp_x(z_x)
        y = mlp_y(z_y)
        
    if not use_torch:
        return x.numpy(), y.numpy()
    return x, y

def generate_temporally_convolved_data(n_samples, lag=30, noise=0.1, use_torch=True):
    """
    Generates data where Y is a simple time-delayed version of X.

    This creates a clean, unambiguous temporal relationship ideal for testing
    the windowing functionality of the MI estimator.

    Args:
        n_samples (int): The number of time points to generate.
        lag (int): The number of timepoints to delay Y relative to X.
        noise (float): The amount of Gaussian noise to add to Y.
        use_torch (bool): If True, returns torch.Tensors.

    Returns:
        tuple: A tuple (x, y) of the generated temporal data, each of shape
               [1, n_samples] for compatibility with our processors.
    """
    full_signal = np.cumsum(np.random.randn(n_samples + lag))
    full_signal = (full_signal - full_signal.mean()) / full_signal.std()
    
    x = full_signal[:-lag]
    y = full_signal[lag:] + np.random.randn(n_samples) * noise
    
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)

    if use_torch:
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y

def generate_xor_data(
    n_samples: int, noise: float = 0.1, use_torch: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates data for the XOR task, a classic test for synergy.

    The XOR problem is a classic example where the mutual information between
    the joint variable `(x1, x2)` and `y` is high, but the MI between either
    `x1` and `y` or `x2` and `y` individually is zero.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    noise : float, optional
        Noise to add to the continuous Y variable. Defaults to 0.1.
    use_torch : bool, optional
        If True, returns torch.Tensors. Defaults to True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple (x, y) where x is `(n_samples, 2)` and y is `(n_samples, 1)`.
    """
    x1 = np.random.randint(0, 2, size=n_samples)
    x2 = np.random.randint(0, 2, size=n_samples)
    y = np.bitwise_xor(x1, x2).astype(float) + np.random.randn(n_samples) * noise
    
    x = np.vstack([x1, x2]).T
    y = y.reshape(-1, 1)

    if use_torch:
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y

def generate_correlated_spike_trains(
    n_neurons: int = 10, duration: float = 100.0, firing_rate: float = 5.0, 
    delay: float = 0.02, jitter: float = 0.005
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generates two populations of spike trains with a time-lagged correlation.

    Population X is a homogeneous Poisson process. Population Y fires with a
    specified delay and jitter after the corresponding neuron in population X.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons in each population. Defaults to 10.
    duration : float, optional
        Duration of the recording in seconds. Defaults to 100.0.
    firing_rate : float, optional
        Firing rate in Hz for the source population (X). Defaults to 5.0.
    delay : float, optional
        The time lag in seconds for the target population's (Y) response.
        Defaults to 0.02.
    jitter : float, optional
        Standard deviation of the Gaussian noise added to the delay.
        Defaults to 0.005.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        A tuple (pop_x, pop_y) of two lists of numpy arrays, where each
        array contains the spike times for a single neuron.
    """
    pop_x = []
    for _ in range(n_neurons):
        n_spikes = np.random.poisson(duration * firing_rate)
        spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
        pop_x.append(spike_times)
    
    pop_y = []
    for i in range(n_neurons):
        delayed_spikes = pop_x[i] + delay + np.random.normal(0, jitter, len(pop_x[i]))
        delayed_spikes = np.sort(delayed_spikes[(delayed_spikes > 0) & (delayed_spikes < duration)])
        pop_y.append(delayed_spikes)
        
    return pop_x, pop_y


def generate_correlated_categorical_series(
    n_samples: int, n_channels: int = 1, n_categories: int = 3, 
    transition_probability: float = 0.9, use_torch: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two correlated categorical time series.

    A base series `x` is generated where the state at time `t` has a high
    probability of being the same as at `t-1`. A second series `y` is generated
    where the state at time `t` is highly likely to be the same as in `x` at `t`.

    Parameters
    ----------
    n_samples : int
        The number of timepoints to generate.
    n_channels : int, optional
        The number of channels for the data. Defaults to 1.
    n_categories : int, optional
        The number of distinct categories (e.g., 3 for states 0, 1, 2). Defaults to 3.
    transition_probability : float, optional
        The probability of Y's state matching X's state at each timepoint. Defaults to 0.9.
    use_torch : bool, optional
        If True, returns torch.Tensors. Defaults to False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple (x, y) of the generated categorical data, each of shape
        `(n_channels, n_samples)`.
    """
    x = np.zeros((n_channels, n_samples), dtype=int)
    y = np.zeros((n_channels, n_samples), dtype=int)

    for ch in range(n_channels):
        # Generate the first series with some temporal smoothness
        x[ch, 0] = np.random.randint(n_categories)
        for t in range(1, n_samples):
            if np.random.rand() < 0.95:  # High prob of staying in the same state
                x[ch, t] = x[ch, t - 1]
            else:
                x[ch, t] = np.random.randint(n_categories)

        # Generate the second series correlated with the first
        for t in range(n_samples):
            if np.random.rand() < transition_probability:
                y[ch, t] = x[ch, t]
            else:
                y[ch, t] = np.random.randint(n_categories)
                
    if use_torch:
        return torch.from_numpy(x).int(), torch.from_numpy(y).int()
    return x, y