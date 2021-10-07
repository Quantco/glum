Yet another GLM package?
========================

``glum`` was inspired by a desire to have a fast, maintainable, Python-first library for fitting GLMs with an extensive feature set.

At the beginning, we thoroughly examined all the existing contenders. The two mostly feature-complete options were glmnet and H2O. In many ways, the R package "glmnet" is the gold standard for regularized glm implementations. However, it is missing several useful features like built-in support for Tweedie and Gamma distributions. It also suffers from `impossible-to-maintain source <https://github.com/cran/glmnet/blob/b1a4b50de01e0cd24343959d7cf86452bac17b26/src/glmnet5dpclean.f>`_ and thus has frequent bugs and segfaults. Although Python-to-glmnet interfaces exist, none is complete and well maintained. We also looked into the H2O implementation. It’s more feature-complete than glmnet, but there are serious integration issues with Python. As we discovered, there is also substantial room to improve performance beyond the level of glmnet or H2O. 

So we decided to improve an existing package. Which one? To be a bit more precise, the question we wanted to answer was "Which library will be the least work to make feature-complete, high performance and correct?" To decide, we began by building a suite of benchmarks to compare the different libraries, and compared the libraries in terms of speed, the number of benchmarks that ran successfully, and code quality. In the end, we went with the code from `an sklearn pull request <https://github.com/scikit-learn/scikit-learn/pull/9405>`_. We called it "sklearn-fork" and actually gave our code that name for quite a while too. sklearn-fork had decent, understandable code, converged in most situations, and included many of the features that we wanted. But it was slow. We figured it would be easier to speed up a functioning library than fix a broken but fast library. So we decided to start improving sklearn-fork. As a side note, a huge thank you to `Christian <https://github.com/lorentzenchr>`_ for producing the baseline code for ``glum``.

Ultimately, improving sklearn-fork seems to have been the right choice. We feel we have achieved our goals and ``glum`` is now :doc:`feature-complete </glm>`, :doc:`high-performance </benchmarks>` and correct. However, over time, we uncovered more flaws in the optimizer than expected and, like most projects, building sklearn-fork into a feature-complete, fast, GLM library was a `harder task <https://github.com/Quantco/glum/issues?q=is%3Aissue+is%3Aclosed>`_ than `we predicted <https://github.com/Quantco/glum/pulls?q=is%3Apr+is%3Aclosed>`_. When we started, sklearn-fork successfully converged for most problems. But, it was slow, taking hundreds or thousands of iteratively reweighted least squares (IRLS) iterations, many more than other similar libraries. Overall, the improvements we’ve made separate into three categories: algorithmic improvements, detailed software optimizations, and new features. 

Algorithmic improvements
-------------------------

At the beginning, the lowest-hanging fruit came from debugging the implementation of IRLS and coordinate descent (CD) because those components were quite buggy and suboptimal. The algorithm we use is from [Yuan2012]_. We started by understanding the paper and relating it back to the code. This led to a circuitous chase around the code base, an effort that paid off when we noticed a hard-coded value in the optimizer was far too high. Fixing this was a one-line change that gave us 2-4X faster convergence! 

Another large algorithmic improvement to the optimizer came from centering the predictor matrix to have mean zero. Coordinate descent cycles through one feature at a time, which is a strategy that works poorly with non-centered predictors because changing any coefficient changes the mean. In several cases, zero-centering reduced the total number of IRLS iterations by a factor of two, while leaving solutions unchanged. As we discuss below, centering is nontrivial in the case of a sparse matrix because we don’t want to modify the zero entries and destroy the sparsity. This was a major impetus for starting a tabular matrix handling library, `tabmat <https://github.com/Quantco/tabmat>`_, as an extension of ``glum``.

Much later on, we made major improvements to the quality of the quadratic approximations for binomial, gamma, and Tweedie distributions, where the the original Hessian approximations turned out to be suboptimal. For the first couple months, we took for granted that the quadratic log-likelihood approximations from sklearn-fork were correct. However, after substantial investigation, it turned out that we were using a Fisher information matrix-based approximation to the hessian rather than the true Hessian. This was done in sklearn-fork because the Fisher information matrix (FIM) is guaranteed to be positive definite for any link function or distribution, a necessary condition for guaranteed convergence. However, in cases where the true Hessian is also positive definite, using it will result in much faster convergence. It turned out that switching to using the true Hessian for these special cases (linear, Poisson, gamma, logistic regression and Tweedie regression for 1 < p < 2) gave huge reductions in the number of IRLS iterations. Some gamma regression problems dropped from taking 50-100 iterations to taking just 5-10 iterations. 

Other important improvements:

* Using numerically stable log-likelihood, gradient and hessian formulas for the binomial distribution. In the naive version, we encounter floating point infinities for large parameter values in intermediate calculations.
* Exploring the use of an ADMM iterative L1 solver compared to our current CD solver. We ended up sticking with CD. This helped identify some crucial differences between glum and H2O, which uses an ADMM solver.
* Active set iteration where we use heuristics to improve performance in L1-regularized problems by predicting, at the beginning of each iteration, which coefficients are likely to remain zero. This effectively reduces the set of predictors and significantly improves performance in severely L1-regularized problems.
* Making sure that we could correctly compare objective functions between libraries. The meaning of the regularization strength varies depending on the constant factors that multiply the log-likelihood. 

Software optimizations
----------------------

Substantial performance improvements came from many places.

* Removing redundant calculations and storing intermediate results to re-use later. The line search step had a particularly large number of such optimization opportunities. 
* Cython-izing the coordinate descent implementation based on a version from sklearn’s Lasso implementation. Several optimizations were possible even beyond the sklearn Lasso implementation and we hope to contribute some of these upstream.
* Hand-optimizing the formulas and Cython-izing the code for common distributions' log likelihood, gradients, and hessians. We did this for normal, Poisson, gamma, Tweedie, binomial distributions.

The largest performance improvements have come from better tabular matrix handling. Initially, we were only handling uniformly dense or sparse matrices and using numpy and scipy.sparse to perform matrix operation. Now, we handle general "split" matrices that can be represented by a combination of dense, sparse, and categorical subcomponents. In addition, we built a ``StandardizedMatrix`` which handles the offsetting and multiplication needed to standardize a matrix to have mean zero and standard deviation one. We store the offsets and multipliers to perform this operation without modifying the underlying matrices. 

We took our first step into developing custom matrix classes when we realized that even the pure dense and sparse matrix implementations were suboptimal. The default scipy.sparse matrix-multiply and matrix-vector product implementations are not parallel. Furthermore, many matrix-vector products only involve a small subset of rows or columns. As a result, we now have custom implementations of these operations that are parallelized and allow operating on a restricted set of rows and columns. 

Before continuing, a quick summary of the only three matrix operations that we care about for GLM estimation:

* Matrix-vector products. ``X.dot(v)`` in numpy notation
* Transpose-matrix-vector products. ``X.T.dot(v)``
* Sandwich products. ``X.T @ diag(d) @ X``

As a matrix multiplication, the sandwich products are higher-dimensional operations than the matrix-vector products and, as such, are particularly expensive. Not only that, but the default implementation in numpy or scipy.sparse is going to be very inefficient. With dense numpy arrays, if we perform ``X.T @ diag(d)``, that will allocate and create a whole new matrix that’s just as large as the original ``X`` matrix. Then, we still need to perform a matrix multiply! As a result, we implemented a parallelized, cache-friendly, SIMD-optimized sandwich product operation that avoids the copy and performs the operation as a single matrix-multiply-like operation. We are in the process of contributing an implementation to the `BLIS library <https://github.com/flame/blis>`_.

The next big matrix optimization came from realizing that most data matrices are neither fully dense nor fully sparse. Some columns will be very sparse (e.g. number of parrots owned), some columns will be one-hot encoded categoricals (e.g. preferred parrot species) while other columns will be dense (e.g. volume in liters of the most recently seen parrot). So we built a SplitMatrix class that splits a matrix into dense and sparse subcomponents. A threshold of around 90% sparsity seems to be about the level at which it is beneficial to use a simple CSR sparse matrix instead of a dense matrix. The benefit of this split matrix was large, improving performance across all the matrix operations by 2-5x.

Later on, we also added categorical matrix handling to the mix. Many categorical columns will be very sparse. If there are 100 evenly distributed categories, each column will have 99% sparse. However, simply treating them as a general sparse matrix is leaving a lot on the table. Beyond just being sparse, we know that every non-zero entry is a one and that every row has only a single non-zero column. This is particularly beneficial for sandwich products where the output ends up being diagonal. But, despite the clear gains, adding categorical matrices was quite a large undertaking. We needed to modify our data generation process to produce categoricals instead of one-hot-encoded columns, add and optimize each of our matrix operations for categoricals, and specify "sandwich" interactions between categorical matrices, sparse matrices, and dense matrices. The result was a large improvement in runtime, with some sandwich and matrix-transpose-dot operations sped up by more than an order of magnitude.

The end result of all these matrix optimizations is that we now have a fairly complete library for handling simple sandwich, dot and transpose-dot operations on a mix of dense, sparse and categorical matrices. This is perfect for most tabular data! So, we’ve split this component off into its own library, `tabmat <https://github.com/Quantco/tabmat>`_.

New Features
-------------

In addition to the heavy focus on optimization and algorithmic correctness, we’ve also added a few important features to glum beyond what was already available in sklearn-fork.

* Automatic cross validation and regularization path handling similar in behavior to glmnet.
* Linear inequality constraints on coefficients. 
* A step size convergence criterion in addition to the typical gradient-norm based criterion.
* The binomial distribution, and as a result, L1 and L2-regularized logistic regression.
* Standard errors. 

References
----------

.. [Yuan2012] Yuan, G. X., Ho, C. H., & Lin, C. J. (2012). An improved glmnet for l1-regularized logistic regression. The Journal of Machine Learning Research, 13(1), 1999-2030.
