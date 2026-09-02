# Expose the main run function and other key components at the top level
__version__ = "1.0.0"

# Suppress the tqdm "IProgress not found" warning that fires when ipywidgets is
# absent (e.g. plain terminal use).  tqdm.auto still falls back to the standard
# text bar; this just silences the noisy one-time warning.
import warnings as _warnings
_warnings.filterwarnings("ignore", message="IProgress not found", category=UserWarning)

from .run import run
from .config import (
    Model, Training, Split, Estimator, Output, Processing,
    Rigorous, Precision, Lag, Transfer, Dimensionality, Conditional,
    Interaction, Pairwise, Sweep,
)
from .logger import logger, set_verbosity, set_verbose
from .results import Results
from .exceptions import NeuralMIError, DataShapeError, InsufficientDataError, TrainingError
from .embeddings_io import extract_embeddings
from .quantities import (
    active_information_storage, excess_entropy, instantaneous_mi,
    cross_predictive_information, block_mi, transfer_entropy,
    conditional_transfer_entropy,
    interaction_information, mi_rate, instantaneous_exchange,
    directed_information_rate,
)
from . import data
from . import generators
from . import estimators
from . import models
from . import results
from . import utils
from . import validation
from . import visualize

__all__ = [
    'run',
    'Model', 'Training', 'Split', 'Estimator', 'Output', 'Processing',
    'Rigorous', 'Precision', 'Lag', 'Transfer', 'Dimensionality', 'Conditional',
    'Interaction', 'Pairwise', 'Sweep',
    'Results',
    'logger',
    'set_verbosity',
    'set_verbose',
    'NeuralMIError',
    'DataShapeError',
    'InsufficientDataError',
    'TrainingError',
    'extract_embeddings',
    'active_information_storage', 'excess_entropy', 'instantaneous_mi',
    'cross_predictive_information', 'block_mi', 'transfer_entropy',
    'conditional_transfer_entropy',
    'interaction_information', 'mi_rate', 'instantaneous_exchange',
    'directed_information_rate',
    'data',
    'generators',
    'estimators',
    'models',
    'results',
    'utils',
    'validation',
    'visualize',
]