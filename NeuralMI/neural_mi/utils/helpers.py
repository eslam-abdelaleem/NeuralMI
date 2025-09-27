# neural_mi/utils/helpers.py

import torch

def get_device():
    """
    Determines the appropriate device for PyTorch computations (CUDA, MPS, or CPU).

    Returns:
        str: The device string ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Instantiate the device for consistent use across the library
device = get_device()