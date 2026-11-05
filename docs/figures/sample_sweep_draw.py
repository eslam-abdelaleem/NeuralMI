"""One dataset of the README's sample sweep (readme_figures.ipynb, section 2).

The notebook estimates its datasets in parallel through the library's worker
pool, which can only hand its workers a function they can import. A function
defined in the notebook cannot be imported, so this one lives next to it.
"""
import warnings

import numpy as np

import neural_mi as nmi


def one_dataset(task):
    """NeuralMI on its defaults and KSG averaged over k = 5 and 10, in bits, on one dataset."""
    n_samples, seed, latent_dim, observed_dim, mi = task
    warnings.filterwarnings('ignore')
    x, y = nmi.generators.generate_nonlinear_from_latent(
        n_samples=n_samples, latent_dim=latent_dim, observed_dim=observed_dim, mi=mi, seed=seed)
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    neural = nmi.run(x, y, mode='estimate', split=nmi.Split(mode='random'),
                     seed=seed, show_progress=False).mi_estimate
    ksg = float(np.mean([nmi.estimators.ksg_mi(x, y, k=k) for k in (5, 10)]) / np.log(2))
    return float('nan') if neural is None else float(neural), ksg
