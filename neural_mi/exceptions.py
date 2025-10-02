class NeuralMIError(Exception):
    """Base exception for all errors raised by the neural_mi library."""
    pass

class DataShapeError(NeuralMIError, ValueError):
    """Raised when data has an incorrect or incompatible shape."""
    pass

class InsufficientDataError(NeuralMIError, ValueError):
    """Raised when not enough data is provided for a given analysis."""
    pass

class ParameterError(NeuralMIError, ValueError):
    """Raised when a parameter is invalid or missing."""
    pass