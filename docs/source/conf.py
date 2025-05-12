# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Include your package directory

project = 'MPhil Research Project - Jacob Tutt'
copyright = '2025, Jacob Tutt'
author = 'Jacob Tutt'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax', 
]

autodoc_default_options = {
    'members': True,             # Include all public members
    'undoc-members': True,       # Include members without docstrings
    'private-members': True,     # Include members starting with _
    'special-members': '__init__',  # Include special methods (like __init__)
    'show-inheritance': True,    # Show class inheritance
    'alphabetical': False,       # To maintain source order (optional)
    'member-order': 'bysource',  # To maintain source order (optional)
}
# Dependencies to mock during the documentation build
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "astropy",
    "astroquery",
    "jupyter",
    "notebook",
    "ipykernel",
    "tqdm",
    "tabulate",
    "umap",
    "hdbscan",
    "extreme_deconvolution", 
    "mpl_toolkits"
]
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']