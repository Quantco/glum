Yet another GLM package?
========================

I'd like to migrate a bunch of info from https://docs.google.com/document/d/1GT402wGCHSrsfx_Vio-E3HjK1582Xr_QNxmvnQ8rlnk/edit# to this page.



The GLM Project

The “GLM Project” was inspired by the repeated failure of common open-source GLM packages to live up to the requirements of QC projects in areas like e-commerce, health, and P&C insurance. As the initial user, Jan Tilly wrote up a summary of GLMs and what features he was looking for. The initial goal of the project is to provide an upgrade for the commercial GLM tools that the actuaries at DIL use for claims modeling. GLMs are used to model the number of claims and the monetary size of the claims. A particularly important requested upgrade was to provide regularization capabilities, something that the current tools do not do.

The R package “glmnet” is the gold standard, but it’s missing some useful features like support for Tweedie and Gamma distributions and clean, understandable code (the Fortran source for glmnet is terrifying). Although Python-to-glmnet interfaces exist, none is complete and well maintained. We also looked into the h2o implementation. It’s more feature-complete than glmnet, but there are serious integration issues with Python. There were many other contenders, but none met the requirements. 
We wrote up a Google Sheet that compares the existing implementations.
So we decided to improve an existing package. Which one? To be a bit more precise, the question we wanted to answer was “Which library will be the least work to make feature-complete, high performance and correct? To decide, we began by building a suite of benchmarks to compare the different libraries, and compared the libraries in terms of speed, the number of benchmarks that ran successfully, and code quality. ” We put together a final horse race between the contenders in this google doc. In the end, we went with the code from an sklearn pull request. We called it “sklearn-fork” and actually gave our code that name for quite a while too. sklearn-fork had decent, understandable code, converged in most situations, and included almost all the features that we wanted. But it was slow. We figured it would be easier to speed up a functioning library than fix a broken but fast library. So we decided to start improving sklearn-fork. 

Ultimately, improving sklearn-fork seems to have been the right choice. However, over time, we uncovered more flaws in the optimizer than expected and, like most projects, building sklearn-fork into a feature-complete, fast, GLM library (see our issue tracker history) was a harder task than we predicted. When we started, sklearn-fork successfully converged for most problems. But, it was slow, taking hundreds or thousands of iteratively reweighted least squares (IRLS) iterations, many more than other similar libraries. Overall, the improvements we’ve made separate into three categories: algorithmic improvements, detailed software optimizations, and new features. 



Features
---------

For our use case in insurance, the critial features for GLMs to be useful for us are:

- Need to support Gamma, Poisson (pyglmnet, sklearn-fork, sklearn-master, h20, tfp, statsmodels)
- Need to support sample weights (glmnet, sklearn-fork, sklearn-master, h2ostatsmodels)
- Need to support offsets (an offset is a variable for which we force the parameter to equal 1). (glmnet, h2o, statsmodels)
- Need to work on large-ish data sets (25 million rows, up to several hundred columns)
- Can deal with high-dimensional categorical variables (either through sparse matrices or a dedicated solution) (sparse support: glmnet, h2o, tfp)
- Need to be otherwise sensible (e.g. do not regularize the intercept, etc)
- Decent performance in terms of CPU and memory (commercial competitor without regularization needs a couple of minutes on laptop)
- Reliably convergent algorithms (_not_ tfp or sklearn-fork)
- Support L1 regularization

Nice to have features
----------------------

- Support Tweedie
- Ability to specify custom weights for parameters for the L1 penalty
- Ability to specify custom weight matrix for parameters for the L2 penalty (generalized Tikhonov regularization)

Which packages/implementations exist?
-------------------------------------

- glmnet (R-Package, Fortran implementation, GPL-2, no Tweedie, no gamma)
https://glmnet.stanford.edu/

- pyglmnet (python only, no support of sample weights, no Tweedie, otherwise looks fairly feature-rich, under active development) https://github.com/glm-tools/pyglmnet

- python-glmnet (python bindings to glmnet's Fortran code, not actively developed) https://github.com/civisanalytics/python-glmnet

- glmnet_python (used at wayfair, GPL licensed) https://github.com/bbalasub1/glmnet_python

- scikit-learn fork (python only, has all the features we want, poor performance with sparse matrices, convergence trouble with lasso) https://github.com/scikit-learn/scikit-learn/pull/9405

- scikit-learn master (python only, only ridge, poor performnce with sparse matrices) https://github.com/scikit-learn/scikit-learn/pull/14300

- h2o (Java, python bindings, data needs to be copied from Python to Java all the time, convergence problems) http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html

- Tensorflow Probability (C++, python bindings, no sample weights) https://www.tensorflow.org/probability/api_docs/python/tfp/glm

- statsmodels (python only, very poor performance, lots of issues getting it to work) https://www.statsmodels.org/stable/glm.html
- [celer](https://github.com/mathurinm/celer): Fast solver for Lasso and sparse logistic regression.
