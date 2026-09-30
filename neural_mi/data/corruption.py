# neural_mi/data/corruption.py
"""Degrading data at a resolution, for ``mode='precision'`` and the datasets.

One function serves every caller, so rounding and noise mean the same thing
wherever they are applied.
"""
import torch

METHODS = ('rounding', 'noise')


def corrupt(data: torch.Tensor, tau: float, method: str, empty_value: float = 0.0) -> torch.Tensor:
    """Degrade the entries of `data` that hold a value, at resolution `tau`.

    ``'rounding'`` moves every held value to the centre of its bin of width
    `tau`, :math:`\\tau(\\lfloor v/\\tau \\rfloor + 1/2)`. ``'noise'`` adds an
    independent draw from :math:`U(-\\tau/2, \\tau/2)` to every held value.

    An entry equal to `empty_value` holds no measurement: an unused spike-time
    slot, an empty bin, or a gap the continuous processor padded with zeros. It
    is left as it is, so corruption never creates activity where none was
    recorded. Bin centres lie :math:`\\tau/2` away from every multiple of
    :math:`\\tau`, so rounding never moves a held value onto an empty value of
    zero either, and the number of spikes stays the same.

    Parameters
    ----------
    data : torch.Tensor
        The values to degrade. Not modified.
    tau : float
        The resolution. ``0`` returns `data` unchanged.
    method : {'rounding', 'noise'}
        How to degrade.
    empty_value : float, optional
        The value of an entry that holds no measurement. Defaults to 0.

    Returns
    -------
    torch.Tensor
        A new tensor, or `data` itself when `tau` is 0.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown corruption method: {method!r}. Use one of {list(METHODS)}.")
    if tau == 0.0:
        return data
    if method == 'rounding':
        degraded = tau * (torch.floor(data / tau) + 0.5)
    else:
        degraded = data + (torch.rand_like(data) - 0.5) * tau
    return torch.where(data != empty_value, degraded, data)
