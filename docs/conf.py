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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

import sys

sys.path.append("../src/quantcore/")

project = "quantcore.glm"
copyright = "(C) 2020-2021 QuantCo Inc."
author = "QuantCo Inc."

extensions = [
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
    "altair.sphinxext.altairplot",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

apidoc_module_dir = "../src/quantcore"
apidoc_excluded_paths = ["glm_benchmarks"]
apidoc_output_dir = "./api"
apidoc_separate_modules = False

templates_path = ["_templates"]
exclude_patterns = [
    "modules.rst",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
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
