# neural_mi/estimators/bounds.py
"""Implementations of various mutual information lower bounds.

This module provides functions for several popular variational lower bounds on
mutual information, as described in the following papers:
- "Understanding the Limitations of Variational Mutual Information Estimators" (ICLR 2020)
- "On Variational Bounds of Mutual Information" (PMLR 2019)
- "Accurate Estimation of Mutual Information in High Dimensional Data" (ArXiv 2025)

Each function takes a `(batch_size, batch_size)` score matrix from a critic
model as input and returns a scalar tensor representing the MI estimate.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple

def tuba_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the TUBA lower bound on mutual information.

    TUBA stands for Tractable Unnormalized version of the Barber and Agakov bound.

    References
    ----------
    Poole, B., et al. (2019). On Variational Bounds of Mutual Information. PMLR.

    Parameters
    ----------
    scores : torch.Tensor
        A `(batch_size, batch_size)` tensor of scores, where `scores[i, j]`
        is the critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated MI lower bound.
    """
    joint = scores.diag().mean()
    marg = torch.exp(logmeanexp_nodiag(scores))
    return 1. + joint - marg

def nwj_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the NWJ lower bound on mutual information.

    NWJ stands for Nguyen-Wainwright-Jordan, and it is closely related to the
    Donsker-Varadhan representation of KL-divergence.

    References
    ----------
    Poole, B., et al. (2019). On Variational Bounds of Mutual Information. PMLR.

    Parameters
    ----------
    scores : torch.Tensor
        A `(batch_size, batch_size)` tensor of scores, where `scores[i, j]`
        is the critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated MI lower bound.
    """
    return tuba_lower_bound(scores - 1.)

def infonce_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the InfoNCE lower bound on mutual information.

    InfoNCE (Noise-Contrastive Estimation) is also known as the CPC loss from
    the Contrastive Predictive Coding paper. It is a lower bound on MI and is
    widely used in self-supervised learning.

    References
    ----------
    Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation Learning with
    Contrastive Predictive Coding. ArXiv.

    Parameters
    ----------
    scores : torch.Tensor
        A `(batch_size, batch_size)` tensor of scores, where `scores[i, j]`
        is the critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated MI lower bound.
    """
    nll = scores.diag() - torch.logsumexp(scores, dim=1)
    mi = torch.log(torch.tensor(scores.size(0), device=scores.device)) + nll.mean()
    return mi

def js_fgan_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    """Computes the Jensen-Shannon f-GAN lower bound on mutual information.

    This bound is based on the f-GAN framework applied to the Jensen-Shannon
    divergence.

    References
    ----------
    Poole, B., et al. (2019). On Variational Bounds of Mutual Information. PMLR.

    Parameters
    ----------
    scores : torch.Tensor
        A `(batch_size, batch_size)` tensor of scores, where `scores[i, j]`
        is the critic's output for the pair `(x[i], y[j])`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated MI lower bound.
    """
    f_diag = scores.diag()
    n = scores.size(0)
    first_term = -F.softplus(-f_diag).mean()
    second_term = (torch.sum(F.softplus(scores)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term

def smile_lower_bound(scores: torch.Tensor, clip: float = None) -> torch.Tensor:
    """Computes the SMILE lower bound on mutual information.

    SMILE (Smoothed Mutual Information "Lower-bound" Estimator) is designed to
    reduce the bias of other estimators by applying a clipping transformation.

    References
    ----------
    Song, J., & Ermon, S. (2020). Understanding the Limitations of
    Variational Mutual Information Estimators. ICLR.

    Parameters
    ----------
    scores : torch.Tensor
        A `(batch_size, batch_size)` tensor of scores, where `scores[i, j]`
        is the critic's output for the pair `(x[i], y[j])`.
    clip : float, optional
        The value to which the scores will be clipped. If None, no clipping
        is applied. Defaults to None.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the estimated MI lower bound.
    """
    scores_ = torch.clamp(scores, -clip, clip) if clip is not None else scores
    z = logmeanexp_nodiag(scores_, dim=(0, 1))
    dv = scores.diag().mean() - z
    js = js_fgan_lower_bound(scores)
    with torch.no_grad():
        dv_js = dv - js
    return js + dv_js

def logmeanexp_nodiag(x: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    """Computes log-mean-exp of off-diagonal elements of a square matrix.

    This is a helper function used in several MI estimators to compute the
    log-partition function over the marginal distribution samples, which are
    represented by the off-diagonal elements of the score matrix.

    Parameters
    ----------
    x : torch.Tensor
        A square 2D tensor of shape `(batch_size, batch_size)`.
    dim : int or tuple of ints, optional
        The dimension or dimensions to reduce. If None, the reduction is
        performed over all off-diagonal elements. Defaults to None.

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
