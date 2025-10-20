# neural_mi/exceptions.py
"""Defines custom exceptions for the neural_mi library.

Using custom exceptions allows for more specific error handling and clearer
error messages, making the library easier to debug and use.
"""

class NeuralMIError(Exception):
    """Base class for all custom exceptions in the neural_mi library."""
    pass

class DataShapeError(NeuralMIError, ValueError):
    """Exception raised for errors related to the shape of input data.

    This is typically raised when an input tensor or array does not have the
    expected number of dimensions or when dimensions have an incorrect size.
    """
    pass

class InsufficientDataError(DataShapeError):
    """Exception raised when not enough data is provided for an operation.

    This is a subclass of `DataShapeError` and is used, for example, when
    the length of a time series is smaller than the required window size for
    processing.
    """
    pass

class TrainingError(NeuralMIError):
    """Exception raised for critical errors that occur during model training.

    This exception is used to signal that the training process has failed
    and cannot continue, for example, if no valid model checkpoint could be
    created.
    """
    pass