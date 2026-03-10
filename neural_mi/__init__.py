# Expose the main run function and other key components at the top level
__version__ = "2.0.0"

from .run import run
from .logger import logger, set_verbosity, set_verbose
from .results import Results
from .exceptions import NeuralMIError, DataShapeError, InsufficientDataError, TrainingError
from .embeddings_io import extract_embeddings
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
    'Results',
    'logger',
    'set_verbosity',
    'set_verbose',
    'NeuralMIError',
    'DataShapeError',
    'InsufficientDataError',
    'TrainingError',
    'extract_embeddings',
    'data',
    'generators',
    'estimators',
    'models',
    'results',
    'utils',
    'validation',
    'visualize',
]