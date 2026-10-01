# docs/source/conf.py
"""Sphinx configuration for the NeuralMI documentation.

Options: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
import os
import sys

# The package is importable from the repository root, for autodoc and for the
# version below.
sys.path.insert(0, os.path.abspath('../../'))
import neural_mi  # noqa: E402

# -- Project -----------------------------------------------------------------

project = 'NeuralMI'
author = 'Eslam Abdelaleem'
copyright = f'2026, {author}'
release = neural_mi.__version__
version = '.'.join(release.split('.')[:2])

# -- General -----------------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'nbsphinx_link',   # notebooks are rendered from tutorials/ through .nblink stubs
    'myst_parser',
    'sphinx_copybutton',   # a copy button on every code block
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
templates_path = ['_templates']
exclude_patterns = []

# The reference pages are Markdown included from reference/*.md. Quotes and
# dashes are left as written, so the rendered text matches the source.
myst_enable_extensions = [
    'amsmath',          # LaTeX math environments
    'dollarmath',       # $...$ and $$...$$
    'colon_fence',      # ::: admonitions
    'deflist',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',          # bare URLs become links
    'strikethrough',
    'substitution',
    'tasklist',
]
# GitHub-style anchors down to level 3, so the in-page contents tables of
# reference/*.md resolve both here and on GitHub.
myst_heading_anchors = 3

suppress_warnings = ['misc.highlighting_failure', 'ref.ref']

# -- HTML --------------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = '_static/logo/favicon.ico'
# The logo carries the name. Its colours read on a white header.
html_logo = '_static/logo/neuralmi_logo_horizontal.png'
html_css_files = ['header.css']
html_baseurl = 'https://eslam-abdelaleem.github.io/NeuralMI/'
html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': 'white',
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}
# No "View page source" link: for a notebook it would open the raw .ipynb JSON,
# so the sources are neither linked nor copied.
html_show_sourcelink = False
html_copy_source = False

# -- Notebooks ---------------------------------------------------------------

# The tutorials ship with their outputs, and the build never executes them:
# running them would need the example data and long training runs.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- Autodoc and intersphinx -------------------------------------------------

autodoc_member_order = 'bysource'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
