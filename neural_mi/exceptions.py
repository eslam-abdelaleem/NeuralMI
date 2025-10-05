# neural_mi/exceptions.py

class NeuralMIError(Exception):
    """Base exception for all errors raised by the neural_mi library."""
    pass

class DataShapeError(NeuralMIError):
    """Raised when input data has an incorrect or incompatible shape."""
    pass

class InsufficientDataError(NeuralMIError):
    """Raised when there is not enough data to perform the requested analysis."""
    pass

class TrainingError(NeuralMIError):
    """Raised when a model fails to train successfully."""
    pass