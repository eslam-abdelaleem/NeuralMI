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
    Interaction, Pairwise,
)
from .logger import logger, set_verbosity, set_verbose, grouped_warnings
from .logger import _group_repeats_by_cell
from .results import Results
from .exceptions import (NeuralMIError, DataShapeError, InsufficientDataError, TrainingError,
                         CombinationWarning)
from .embeddings_io import extract_embeddings
from .quantities import (
    active_information_storage, predictive_information, instantaneous_mi,
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

# In IPython and Jupyter a warning that several NeuralMI calls in one cell raise
# is shown once, and the cell ends with one line giving the count. Outside a
# notebook, nmi.grouped_warnings() does the same over a block of calls.
_group_repeats_by_cell()

__all__ = [
    'run',
    'Model', 'Training', 'Split', 'Estimator', 'Output', 'Processing',
    'Rigorous', 'Precision', 'Lag', 'Transfer', 'Dimensionality', 'Conditional',
    'Interaction', 'Pairwise',
    'Results',
    'logger',
    'set_verbosity',
    'set_verbose',
    'grouped_warnings',
    'NeuralMIError',
    'DataShapeError',
    'InsufficientDataError',
    'TrainingError',
    'CombinationWarning',
    'extract_embeddings',
    'active_information_storage', 'predictive_information', 'instantaneous_mi',
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