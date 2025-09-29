# neural_mi/estimators/__init__.py
from .bounds import infonce_lower_bound, nwj_lower_bound, tuba_lower_bound, smile_lower_bound

# A dictionary to easily access estimators by name
ESTIMATORS = {
    'infonce': infonce_lower_bound,
    'nwj': nwj_lower_bound,
    'tuba': tuba_lower_bound,
    'smile': smile_lower_bound
}