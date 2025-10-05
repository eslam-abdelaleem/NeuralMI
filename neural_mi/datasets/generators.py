# neural_mi/datasets/generators.py

import numpy as np
import torch
from scipy.signal import convolve

def mi_to_rho(dim, mi):
    """Calculates the correlation coefficient `rho` for a given MI and dimension."""
    # Convert MI from bits to nats for the formula
    mi_nats = mi * np.log(2)
    return np.sqrt(1 - np.exp(-2.0 / dim * mi_nats))

def generate_correlated_gaussians(n_samples, dim, mi, use_torch=True):
    """
    Generates two correlated Gaussian variables with a known mutual information.

    Args:
        n_samples (int): The number of samples to generate.
        dim (int): The dimension of each variable.
        mi (float): The ground truth mutual information in bits.
        use_torch (bool): If True, returns torch.Tensors, otherwise NumPy arrays.

    Returns:
        tuple: A tuple (x, y) of the generated data.
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

def generate_nonlinear_from_latent(n_samples, latent_dim, observed_dim, mi, hidden_dim=64, use_torch=True):
    """
    Generates nonlinearly transformed data from a low-dimensional latent space.

    This is a biologically-plausible model where a high-dimensional observation (e.g.,
    neural activity) is a nonlinear projection of a low-dimensional latent signal.

    Args:
        n_samples (int): Number of samples.
        latent_dim (int): The ground truth dimensionality of the latent variables.
        observed_dim (int): The dimensionality of the final observed variables (e.g., num_neurons).
        mi (float): The ground truth MI between the latent variables Z_x and Z_y.
        hidden_dim (int): The hidden dimension of the transforming MLPs.
        use_torch (bool): If True, returns torch.Tensors.

    Returns:
        tuple: A tuple (x, y) of the generated nonlinear data.
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


def generate_xor_data(n_samples, noise=0.1, use_torch=True):
    """
    Generates data for the XOR task, a classic test for synergy.

    Args:
        n_samples (int): Number of samples.
        noise (float): Noise to add to the continuous Y variable.
        use_torch (bool): If True, returns torch.Tensors.

    Returns:
        tuple: A tuple (x, y) where x is [n_samples, 2] and y is [n_samples, 1].
    """
    x1 = np.random.randint(0, 2, size=n_samples)
    x2 = np.random.randint(0, 2, size=n_samples)
    y = np.bitwise_xor(x1, x2).astype(float) + np.random.randn(n_samples) * noise
    
    x = np.vstack([x1, x2]).T
    y = y.reshape(-1, 1)

    if use_torch:
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    return x, y

def generate_correlated_spike_trains(n_neurons=10, duration=100.0, firing_rate=5.0, delay=0.02, jitter=0.005):
    """
    Generates two populations of spike trains with a time-lagged correlation.

    Args:
        n_neurons (int): Number of neurons in each population.
        duration (float): Duration of the recording in seconds.
        firing_rate (float): Firing rate in Hz for the source population.
        delay (float): The time lag in seconds for the target population's response.
        jitter (float): Standard deviation of the noise added to the delay.

    Returns:
        tuple: A tuple (pop_x, pop_y) of two lists of numpy arrays, where each
               array contains the spike times for a single neuron.
    """
    # Population X: Homogeneous Poisson process
    pop_x = []
    for _ in range(n_neurons):
        n_spikes = np.random.poisson(duration * firing_rate)
        spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
        pop_x.append(spike_times)
    
    # Population Y: Fires with a delay and jitter after X
    pop_y = []
    for i in range(n_neurons):
        delayed_spikes = pop_x[i] + delay + np.random.normal(0, jitter, len(pop_x[i]))
        delayed_spikes = np.sort(delayed_spikes[(delayed_spikes > 0) & (delayed_spikes < duration)])
        pop_y.append(delayed_spikes)
        
    return pop_x, pop_y