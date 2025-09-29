import multiprocessing
import platform

# Set the multiprocessing start method to 'spawn' on macOS systems.
# This is necessary to prevent potential deadlocks when using PyTorch
# with multiprocessing, ensuring stability in modes like 'sweep' and 'rigorous'.
# This is placed in the top-level __init__.py to ensure it runs on import.
try:
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # The start method can only be set once, so we pass if it's already been set.
    pass

# Expose the main run function and other key components at the top level
from .run import run
from . import data
from . import datasets
from . import estimators
from . import models
from . import results
from . import utils
from . import validation
from . import visualize