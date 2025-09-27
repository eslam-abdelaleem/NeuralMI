# neural_mi/estimators/bounds.py

import torch
import torch.nn.functional as F
import numpy as np
# We no longer need to import the global device variable

def tuba_lower_bound(scores):
    """TUBA lower bound on mutual information."""
    joint_term = scores.diag().mean()
    # Pass the scores tensor to the helper to infer the device
    marg_term = torch.exp(logmeanexp_nodiag(scores))
    return 1. + joint_term - marg_term

def nwj_lower_bound(scores):
    """NWJ lower bound on mutual information, a biased version of TUBA."""
    return tuba_lower_bound(scores - 1.)

def infonce_lower_bound(scores):
    """InfoNCE lower bound on mutual information."""
    nll = scores.diag() - torch.logsumexp(scores, dim=1)
    # Create the new tensor on the same device as the scores matrix
    mi = torch.log(torch.tensor(scores.size(0), dtype=torch.float32, device=scores.device)) + nll.mean()
    return mi

def js_fgan_lower_bound(scores):
    """Jensen-Shannon f-GAN lower bound on mutual information."""
    f_diag = scores.diag()
    n = scores.size(0)
    first_term = -F.softplus(-f_diag).mean()
    second_term = (torch.sum(F.softplus(scores)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term

def smile_lower_bound(scores, clip=None):
    """SMILE lower bound on mutual information."""
    if clip is not None:
        scores_ = torch.clamp(scores, -clip, clip)
    else:
        scores_ = scores
        
    # Pass the scores tensor to the helper to infer the device
    z = logmeanexp_nodiag(scores_, dim=(0, 1))
    dv = scores.diag().mean() - z
    js = js_fgan_lower_bound(scores)

    with torch.no_grad():
        dv_js = dv - js
    
    return js + dv_js

# --- Helper functions ---

def logmeanexp_nodiag(x, dim=None):
    """
    Compute log(mean(exp(x))) for off-diagonal elements.
    Device is inferred from the input tensor 'x'.
    """
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    # Create the diagonal mask on the same device as x
    inf_diag = torch.diag(float('inf') * torch.ones(batch_size, device=x.device))
    logsumexp = torch.logsumexp(x - inf_diag, dim=dim)
    
    if isinstance(dim, int):
        num_elem = batch_size - 1.
    else:
        num_elem = batch_size * (batch_size - 1.)
        
    # Create the num_elem tensor on the same device as x
    return logsumexp - torch.log(torch.tensor(num_elem, device=x.device))