# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuralMI'
copyright = '2026, Eslam Abdelaleem'
author = 'Eslam Abdelaleem'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'nbsphinx_link',   # render notebooks from a single source via .nblink stubs
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'myst_parser',  
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

suppress_warnings = ['misc.highlighting_failure', 'ref.ref']

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
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Do not show the "View page source" link. By default Sphinx copies each page's
# raw source into _sources/ and links to it, so on a notebook page the link opens
# the raw .ipynb JSON (and on an .rst page, the reStructuredText) rather than
# anything readable. Hide the link and skip copying the sources entirely.
html_show_sourcelink = False
html_copy_source = False

# -- nbsphinx settings -------------------------------------------------------
# Never execute notebooks during the docs build. Tutorials ship with their
# outputs pre-computed; executing them on CI would require example datasets and
# long model-training runs. With the default 'auto', nbsphinx executes any
# notebook that has no stored outputs (e.g. tutorial 08), which fails the build.
# Re-run and save a notebook locally to embed/update its outputs.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = 'bysource'

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}