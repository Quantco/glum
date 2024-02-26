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
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "glum"
copyright = f"2020–{datetime.datetime.now().year}, QuantCo Inc."
author = "QuantCo Inc."

extensions = [
    "sphinx.ext.napoleon",
    "sphinxext_altair.altairplot",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

autodoc_default_options = {
    "exclude-members": (
        "set_fit_request, "
        "set_fit_predict_request, "
        "set_fit_transform_request, "
        "set_partial_fit_request, "
        "set_predict_request, "
        "set_predict_proba_request, "
        "set_predict_log_proba_request, "
        "set_decision_function_request, "
        "set_score_request, "
        "set_split_request, "
        "set_transform_request, "
        "set_inverse_transform_request"
    )
}

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
