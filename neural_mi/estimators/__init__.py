# neural_mi/estimators/__init__.py
"""Initializes the estimators module.

This file makes the core mutual information estimator functions from the
`bounds` module directly accessible under the `neural_mi.estimators` namespace.

It also defines the `ESTIMATORS` dictionary, which provides a convenient,
string-based way to access the different estimator functions. This is useful
for dynamically selecting an estimator based on configuration parameters.
"""
from .bounds import infonce_lower_bound, smile_lower_bound
from .ksg import ksg_cmi, ksg_mi

# A dictionary to easily access estimators by name
ESTIMATORS = {
    'infonce': infonce_lower_bound,
    'smile': smile_lower_bound,
}

# Estimators that read the information from the rows directly and train no network.
NONPARAMETRIC = {
    'ksg': ksg_mi,
}
# Their conditional forms, which the joint term of a difference uses.
NONPARAMETRIC_CONDITIONAL = {
    'ksg': ksg_cmi,
}

# Default keyword arguments for estimators.  Override via ``estimator_params`` in base_params.
ESTIMATOR_DEFAULTS = {
    'smile': {'clip': 5.0},
}
