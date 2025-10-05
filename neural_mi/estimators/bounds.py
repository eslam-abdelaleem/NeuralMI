# neural_mi/estimators/bounds.py

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple

def tuba_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the TUBA (Tilted U-statistic of the B-spline approximation) lower bound for mutual information.

    This estimator is based on the Donsker-Varadhan representation of the KL-divergence.
    It is known for its low bias but can have high variance.

    Parameters
    ----------
    scores : torch.Tensor
        A (batch_size, batch_size) tensor of scores, where `scores[i, j]` is the
        critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated lower bound on mutual information.
    """
    joint = scores.diag().mean()
    marg = torch.exp(logmeanexp_nodiag(scores))
    return 1. + joint - marg

def nwj_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the NWJ (Nguyen-Wainwright-Jordan) lower bound for mutual information.

    This estimator is also based on the Donsker-Varadhan representation and is closely
    related to the TUBA estimator. It often exhibits lower variance than TUBA.

    Parameters
    ----------
    scores : torch.Tensor
        A (batch_size, batch_size) tensor of scores, where `scores[i, j]` is the
        critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated lower bound on mutual information.
    """
    return tuba_lower_bound(scores - 1.)

def infonce_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the InfoNCE (Noise-Contrastive Estimation) lower bound.

    Also known as the CPC (Contrastive Predictive Coding) loss. This estimator is
    based on Noise-Contrastive Estimation and is widely used in self-supervised learning.
    It is a lower bound on the mutual information.

    Parameters
    ----------
    scores : torch.Tensor
        A (batch_size, batch_size) tensor of scores, where `scores[i, j]` is the
        critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated lower bound on mutual information.
    """
    nll = scores.diag() - torch.logsumexp(scores, dim=1)
    mi = torch.log(torch.tensor(scores.size(0), device=scores.device)) + nll.mean()
    return mi

def js_fgan_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the Jensen-Shannon (JS) f-GAN lower bound.

    This estimator is based on the f-divergence representation of mutual information
    and uses the Jensen-Shannon divergence. It is known to be less biased than
    InfoNCE but can be more unstable during training.

    Parameters
    ----------
    scores : torch.Tensor
        A (batch_size, batch_size) tensor of scores, where `scores[i, j]` is the
        critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated lower bound on mutual information.
    """
    f_diag = scores.diag()
    n = scores.size(0)
    first_term = -F.softplus(-f_diag).mean()
    second_term = (torch.sum(F.softplus(scores)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term

def smile_lower_bound(scores: torch.Tensor, clip: float = None) -> torch.Tensor:
    """Computes the SMILE (State-wise Mutual Information Lower-bound Estimator).

    SMILE is a technique that aims to reduce the bias of MI estimators by clipping
    the critic's scores. It combines the JS estimator with a bias correction term.

    Parameters
    ----------
    scores : torch.Tensor
        A (batch_size, batch_size) tensor of scores, where `scores[i, j]` is the
        critic's output for the pair `(x[i], y[j])`.
    clip : float, optional
        The value to which the scores will be clipped. If None, no clipping is applied.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated lower bound on mutual information.
    """
    scores_ = torch.clamp(scores, -clip, clip) if clip is not None else scores
    z = logmeanexp_nodiag(scores_, dim=(0, 1))
    dv = scores.diag().mean() - z
    js = js_fgan_lower_bound(scores)
    with torch.no_grad():
        dv_js = dv - js
    return js + dv_js

def logmeanexp_nodiag(x: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    """Computes the log of the mean of exponentials of off-diagonal elements.

    This is a helper function used in several MI estimators to compute the
    log-partition function, excluding the diagonal terms which correspond to
    joint distribution samples.

    Parameters
    ----------
    x : torch.Tensor
        A square 2D tensor.
    dim : int or tuple of ints, optional
        The dimension or dimensions to reduce. If None, the reduction is performed
        over all off-diagonal elements. Defaults to None.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the log-mean-exp of the off-diagonal elements.
    """
    batch_size = x.size(0)
    inf_diag = torch.diag(float('inf') * torch.ones(batch_size, device=x.device))
    logsumexp = torch.logsumexp(x - inf_diag, dim=dim or (0,1))
    
    num_elem = batch_size * (batch_size - 1.) if dim is None or isinstance(dim, tuple) else batch_size - 1.
    return logsumexp - torch.log(torch.tensor(num_elem, device=x.device))