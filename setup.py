from setuptools import setup, find_packages

setup(
    name="neural_mi",
    version="1.1.0",
    description="A toolbox for Mutual Information estimation in neural data.",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "statsmodels",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "seaborn"
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)