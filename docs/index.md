# Welcome to glum's documentation!

`glum` is a fast, modern, Python-first GLM estimation library. Generalized linear modeling (GLM) is a core statistical tool that includes many common methods like least-squares regression, Poisson regression and logistic regression as special cases. In addition to fitting basic GLMs, `glum` supports a wide range of features. These include:

* Built-in cross validation for optimal regularization, efficiently exploiting a “regularization path”
* L1 and elastic net regularization, which produce sparse and easily interpretable solutions
* L2 regularization, including variable matrix-valued (Tikhonov) penalties, which are useful in modeling correlated effects
* Normal, Poisson, logistic, gamma, and Tweedie distributions, plus varied and customizable link functions
* Built-in formula-based model specification using `formulaic`
* Classical statistical inference for unregularized models using dispersion and standard errors
* Box and linear inequality constraints, sample weights, offsets
* A scikit-learn-like API to fit smoothly into existing workflows

`glum` was also built with performance in mind. The following figure shows the runtime of a realistic example using an insurance dataset. For more details and other benchmarks, see the [Benchmarks](benchmarks.md) section.

<!-- BENCHMARK_FIGURES_START -->
![Benchmark results](_static/wide-insurance-gamma-normalized.png#only-light){ width="600" }
![Benchmark results](_static/wide-insurance-gamma-normalized_dark.png#only-dark){ width="600" }
<!-- BENCHMARK_FIGURES_END -->

We suggest visiting the [Installation](install.md) and [Getting Started](getting_started/getting_started.md) sections first.
