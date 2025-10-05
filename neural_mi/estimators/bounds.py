# neural_mi/estimators/bounds.py

import torch
import torch.nn.functional as F

def tuba_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    joint = scores.diag().mean()
    marg = torch.exp(logmeanexp_nodiag(scores))
    return 1. + joint - marg

def nwj_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    return tuba_lower_bound(scores - 1.)

def infonce_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    nll = scores.diag() - torch.logsumexp(scores, dim=1)
    mi = torch.log(torch.tensor(scores.size(0), device=scores.device)) + nll.mean()
    return mi

def js_fgan_lower_bound(scores: torch.Tensor) -> torch.Tensor:
    f_diag = scores.diag()
    n = scores.size(0)
    first_term = -F.softplus(-f_diag).mean()
    second_term = (torch.sum(F.softplus(scores)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term

def smile_lower_bound(scores: torch.Tensor, clip: float = None) -> torch.Tensor:
    scores_ = torch.clamp(scores, -clip, clip) if clip is not None else scores
    z = logmeanexp_nodiag(scores_, dim=(0, 1))
    dv = scores.diag().mean() - z
    js = js_fgan_lower_bound(scores)
    with torch.no_grad():
        dv_js = dv - js
    return js + dv_js

def logmeanexp_nodiag(x: torch.Tensor, dim=None) -> torch.Tensor:
    batch_size = x.size(0)
    inf_diag = torch.diag(float('inf') * torch.ones(batch_size, device=x.device))
    logsumexp = torch.logsumexp(x - inf_diag, dim=dim or (0,1))
    
    num_elem = batch_size * (batch_size - 1.) if dim is None or isinstance(dim, tuple) else batch_size - 1.
    return logsumexp - torch.log(torch.tensor(num_elem, device=x.device))