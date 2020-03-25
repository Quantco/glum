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

project = "glm_benchmarks"
copyright = "2019, QuantCo, Inc."
author = "QuantCo, Inc."

extensions = ["numpydoc", "sphinxcontrib.apidoc"]

apidoc_module_dir = "../src/glm_benchmarks"
apidoc_output_dir = "api"
apidoc_separate_modules = True
apidoc_extra_args = ["--implicit-namespaces"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
