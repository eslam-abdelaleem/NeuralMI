# neural_mi/exceptions.py
"""The library's own exceptions and its warning category."""

class NeuralMIError(Exception):
    """Base class for all custom exceptions in the neural_mi library."""
    pass

class DataShapeError(NeuralMIError, ValueError):
    """Exception raised for errors related to the shape of input data.

    This is typically raised when an input tensor or array does not have the
    expected number of dimensions or when dimensions have an incorrect size.
    """
    pass

class InsufficientDataError(NeuralMIError):
    """Exception raised when not enough data is provided for an operation.

    This is a subclass of `NeuralMIError` and is used, for example, when
    the length of a time series is smaller than the required window size for
    processing, or when insufficient data points remain after subsampling.
    """
    pass

class TrainingError(NeuralMIError):
    """Exception raised for critical errors that occur during model training.

    This exception is used to signal that the training process has failed
    and cannot continue, for example, if no valid model checkpoint could be
    created.
    """
    pass


class CombinationWarning(UserWarning):
    """A quantity combined from several MI estimates cannot be read as it stands.

    Raised for a conditional MI or transfer entropy that came out negative, for
    interaction information whose components are in an impossible order, and for
    a combined value with a large error-amplification factor. Filter it with
    ``warnings.filterwarnings('ignore', category=nmi.CombinationWarning)``.
    """
