# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuralMI'
copyright = '2025, Eslam Abdelaleem'
author = 'Eslam Abdelaleem'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'myst_parser',  # Added for Markdown support
]

# MyST-Parser configuration for advanced Markdown features
myst_enable_extensions = [
    "amsmath",      # For LaTeX math
    "colon_fence",  # For ::: admonitions
    "deflist",      # For definition lists
    "dollarmath",   # For $$ math blocks
    "fieldlist",    # For field lists
    "html_admonition",
    "html_image",
    "linkify",      # Auto-link URLs
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# This is the updated setting
html_baseurl = 'https://eslam-abdelaleem.github.io/NeuralMI/'

html_theme_options = {
    'canonical_url': 'https://eslam-abdelaleem.github.io/NeuralMI/',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = 'bysource'

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}