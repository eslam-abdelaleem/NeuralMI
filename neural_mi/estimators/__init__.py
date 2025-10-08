# neural_mi/estimators/__init__.py
"""Initializes the estimators module.

This file makes the core mutual information estimator functions from the
`bounds` module directly accessible under the `neural_mi.estimators` namespace.

It also defines the `ESTIMATORS` dictionary, which provides a convenient,
string-based way to access the different estimator functions. This is useful
for dynamically selecting an estimator based on configuration parameters.
"""
from .bounds import infonce_lower_bound, nwj_lower_bound, tuba_lower_bound, smile_lower_bound

# A dictionary to easily access estimators by name
ESTIMATORS = {
    'infonce': infonce_lower_bound,
    'nwj': nwj_lower_bound,
    'tuba': tuba_lower_bound,
    'smile': smile_lower_bound,
}