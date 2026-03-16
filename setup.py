from setuptools import setup, find_packages

setup(
    name="neural_mi",
    version="2.1.0",
    description="A toolbox for rigorous mutual information estimation in neuroscience.",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "ipykernel>=6.0.0",
        ],
    },
)
