Welcome to quantcore.glm's documentation!
=========================================

``quantcore.glm`` is a fast, Python-first GLM library. In addition to fitting basic GLMs, it supports a wide range of features. These include:

* Built-in cross validation for optimal regularization, efficiently exploiting a “regularization path”
* L1 regularization, which produces sparse and easily interpretable solutions
* L2 regularization, including variable matrix-valued (Tikhonov) penalties, which are useful in modeling correlated effects
* Elastic net regularization
* Normal, Poisson, logistic, gamma, and Tweedie distributions, plus varied and customizable link functions
* Box constraints, linear inequality constraints, sample weights, offsets



We suggest visiting the :doc:`Installation<install>` and :doc:`Getting Started<getting_started/getting_started>` sections first.

.. toctree::
   :maxdepth: 1

   Installation <install.rst>
   Getting Started <getting_started/getting_started.ipynb>
   Motivation <motivation.rst>

.. toctree::
   :maxdepth: 2

   Tutorials <tutorials/tutorials.rst>

.. toctree::
   :maxdepth: 1

   Contributing/Development <contributing.rst>
   Design and background <background/background.rst>
   API Reference <glm>
   GitHub <https://github.com/Quantco/quantcore.glm>
   Changelog <changelog>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
