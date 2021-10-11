# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "glum"
copyright = "(C) 2020-2021 QuantCo Inc."
author = "QuantCo Inc."

extensions = [
    "sphinx.ext.napoleon",
    "altair.sphinxext.altairplot",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = [
    "modules.rst",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "lasso.ipynb",
    "spatial-smoothing.ipynb",
    "benchmarks/*",
    "tutorials/rossman/explore_data.ipynb",
]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    # this puts the entire table of contents structure in the sidebar.
    # unfortunately, it's not possible yet to have it expanded by default.
    # see: https://github.com/readthedocs/sphinx_rtd_theme/issues/455
    "collapse_navigation": False,
    "navigation_depth": 2,
}

# For the altairplot extension
altairplot_links = {"editor": True, "source": True, "export": True}
