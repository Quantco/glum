# An introduction to the algorithms used in `glum` { #background }

Before continuing, please take a look at the [sklearn documentation](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression) for a high-level intro to generalized linear models.

In addition, please take a look at [the tutorials.](tutorials/tutorials.md)

For mathematical and algorithmic details, see below:

## What is a GLM?

This package is intended to fit L1 and L2-norm penalized generalized linear models (GLMs). What is a GLM?

A GLM is a linear model ($\eta = x^\top \beta$) wrapped in a transformation (link function) and equipped with a response distribution from an exponential family. The choice of link function and response distribution is very flexible. In a GLM, a predictive distribution for the response variable $Y$ is associated with a vector of observed predictors $x$.  The distribution has the form:

$$
\begin{align}
  p(y \, |\, x) &= m(y, \phi) \exp\left(\frac{\theta\, T(y) - A(\theta)}{\phi}\right)  & & \textrm{random component / distribution family} \\
  \theta &:= g(\eta) && \textrm{systematic component / link function} \\
  \eta &:= x^\top \beta && \textrm{parametric link component / linear predictor}
\end{align}
$$

Here $\beta$ are the parameters; $\phi$ is a parameter representing dispersion ("variance"); and $m$, $T$, $A$ are characterized by the user-specified model family; and $g$ is the **link function.** For background on the exponential family, and how common distributions can be represented in the form of the "random component" above, see [Wikipedia | Exponential family](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions).

The expected mean of $Y$ depends on $x$ by composition of **linear response** $\eta$ and (inverse) link function, i.e.:

$$
E[y | \eta] := \mu = g^{-1}(\eta).
$$

Proving this involves use of [moment-generating functions and cumulants](https://en.wikipedia.org/wiki/Exponential_family#Moments_and_cumulants_of_the_sufficient_statistic).

Defining an offset of $\gamma$, we can write the linear predictor as $\eta = X^T \beta + \gamma$.

## Examples of GLMs

### Normal

Many of the "regression" models we commonly work with are GLMs. For example, the normal distribution with a linear link function is a simple GLM: $y | x \sim N(x^T \beta, \sigma^2)$. If we fit $\beta$ with maximum likelihood, this is least-squares regression. If we replace the linear link function with an exponential link function, we get a simple multiplicative model that is also a GLM: $y | x \sim N(e^{x^T \beta}, \sigma^2)$.

### Tweedie

In the insurance context, we usually work with a Tweedie distribution, which is a generalization of Poisson ($p=1$) and Gamma ($p=2$):

$$
\begin{align*}
\theta &= \left\{\begin{array}{ll}
\frac{\mu^{1-p}}{1-p} & \text{if } p \neq 1\\
\log \mu              & \text{if } p = 1
\end{array}\right.,\\
T(y) &= y, \\
A(\theta) &= \left\{\begin{array}{ll}
\frac{\mu^{2-p}}{2-p} & \text{if } p \neq 2\\
\log \mu              & \text{if } p = 2.
\end{array}\right.,
\end{align*}
$$

With $1 < p < 2$, the Tweedie distribution is a [compound Poisson-gamma](https://en.wikipedia.org/wiki/Compound_Poisson_distribution#Compound_Poisson_Gamma_distribution) distribution. $y$ is distributed as if a number $N$ is drawn from a Poisson distribution, and then $N$ draws are taken IID from a gamma distribution and added. This distribution might model the total amount of a claim in an insurance context: There are $N$ incidents, and each incident $i$ has an amount $X_i$, and the total amount is the sum of $X_i$.

## Objective function

To fit a GLM, we minimize the negative log-likelihood (or typically the unit deviance) subject to an elastic net constraint involving a mix of L1 and L2 penalty terms:

$$
\min_{\beta} \mathcal L + \alpha \rho ||\beta||_1  + \frac{\alpha (1-\rho)}{2} ||\beta||_2^2
$$

## Optimizers/solvers

There are three solvers implemented in `glum`:

* `lbfgs` - This solver uses the scipy `fmin_l_bfgs_b` optimizer to minimize L2-penalized GLMs. The L-BFGS solver does not work with L1-penalties. Because L-BFGS does not store the full Hessian, it can be particularly effective for very high dimensional problems with several thousand or more columns. For more details, see [the scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html).
* `irls-cd` and `irls-ls` - These solvers are both based on Iteratively Reweighted Least Squares (IRLS). IRLS proceeds by iteratively approximating the objective function with a quadratic, then solving that quadratic for the optimal update. For purely L2-penalized settings, the `irls-ls` uses a least squares inner solver for each quadratic subproblem. For problems that have any L1-penalty component, the `irls-cd` uses a coordinate descent inner solver for each quadratic subproblem. The IRLS-LS and IRLS-CD implementations largely follow the algorithm described in `newglmnet` (see references below).

### Convexity
Our objective function will generally be convex, because the log-likelihoods of members of the exponential family are convex in their "natural parameterizations." The natural parameterization may not be the most obvious one. Example: The log-likelihood of the normal distribution is convex in one over the variance, which is its natural parameterization. It is not convex in the variance. We can generally assume we are solving convex problems.

An exception is multicollinearity (rank deficiency) in the design matrix, without an L2 component to the penalty. In that case, the problem will be only weakly convex and will have no unique miminum. This is not an arcane consideration, since we frequently generate rank deficiency by constructing multiple sets of one-hot encoded categorical variables. This can make evaluating different optimizers tricky, since they could converge to different equally good optima. The original `glmnet` paper suggests using at least a small L2 regularization component to remove "degeneracies and wild behavior."

### IRLS

We minimize the log likelihood using Iteratively Reweighted Least Squares (IRLS). IRLS can be justified by seeing it as taking a Newton step, but replacing the Hessian with the expected Hessian.

In the `irls-cd` and `irls-ls` solvers, the outer loop is an IRLS iteration that forms a quadratic approximation to the negative loglikelihood. That is, we find `w` and `z` so that the problem can be expressed as:

```
min sum_i w_i (z_i - x_i beta)^2 + penalty
```

We exit when either the gradient is small (`gradient_tol`) or the step size is small (`step_size_tol`). Both of these tolerances are user configurable.

Once we have formed this quadratic approximation, an "inner solver" finds the minimum of the quadratic. In the `irls-ls` solver, the inner solver is simply a direct least squares solve.

See the `glmintro` reference for an excellent discussion of IRLS in the context of GLMs.

### Coordinate Descent
With an L1 penalty, we use the `irls-cd` solver where the inner solver finds the minimum of the quadratic using coordinate descent. We exit the inner loop when the quadratic problem's gradient is small. The classic reference here is the `glmnet` paper.

However, coordinate descent is older than the glmnet paper, and is a simple idea. In a problem with data $y$ and $x$ and parameters "params", Coordinate Descent involves repeatedly optimizing one or several parameters while holding the rest fixed. Interestingly, Wikipedia [claims](https://en.wikipedia.org/wiki/Coordinate_descent#Applications) that coordinate descent is often overlooked among researchers because it is simple to implement, and they would rather work on something more interesting.

```python
"""
Cyclical coordinate descent with Newton-like steps and a
twice-differentiable objective function.
"""
def coordinate_descent_inner(y, x, params, grad_func, hess_func):
    for i, elt in enumerate(params):
        step = -grad_func(y, x, params) / hess_func(y, x, params)
        params[i] += step
    return params

def coordinate_descent(y, x, params, grad_func, hess_func,
                       convergence_func):
    while not convergence_func():
        params = coordinate_descent_inner(y, x, params, grad_func,
                                         hess_func)
    return params
```

In practice, the implementation is a bit more complicated when the objective function is not differentiable. See also the `coordinate_descent` reference below.

### Active set tracking

When penalizing with an L1-norm, it is common for many coefficients to be exactly zero. And, it is possible to predict during a given iteration which of those coefficients will stay zero. As a result, we track the "active set" consisting of all the coefficients that are either currently non-zero or likely to remain non-zero. We follow the outer loop active set tracking algorithm in the `newglmnet` reference. That paper refers to the same concept as "shrinkage", whereas the `glmnet` reference calls this the "active set". Currently, we have not yet implemented the inner loop active set tracking from the `newglmnet` reference.

## Matrix Types

Along with the GLM solvers, this package supports dense, sparse, categorical matrix types and mixtures of these types. Using the most efficient matrix representations massively improves performances.

For more details, see the [README for tabmat](https://github.com/Quantco/tabmat)

* We support dense matrices via standard numpy arrays.
* We support sparse CSR and CSC matrices via standard `scipy.sparse` objects. However, we have extended these operations with custom matrix-vector and sandwich product routines that are optimized and parallelized. A user does not need to modify their code to take advantage of this optimization. If a `scipy.sparse.csc_matrix` object is passed in, it will be automatically converted to a `SparseMatrix` object. This operation is almost free because no data needs to be copied.
* We implement a `CategoricalMatrix` object that efficiently represents these matrices without nearly as much overhead as a normal CSC or CSR sparse matrix.
* Finally, `SplitMatrix` allows mixing different matrix types for different columns to minimize overhead.

## Standardization

Internal to our solvers, all matrix types are wrapped in a `tabmat.StandardizedMatrix` which offsets columns to have mean zero and standard deviation one without modifying the matrix data itself. This avoids situations where modifying a matrix to have mean zero would result in losing the sparsity structure. It also avoids ever needing to copy or modify the input data matrix. As a result, excess memory usage is very low in `glum`.

References
----------

`glmnet` - [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://web.stanford.edu/~hastie/Papers/glmnet.pdf)

`newglmnet` - [An Improved GLMNET for L1-regularized LogisticRegression](https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf)

`glmintro` - [Bryan Lewis on GLMs](https://bwlewis.github.io/GLM/)

`coordinate_descent` - [Coordinate Descent Algorithms](http://www.optimization-online.org/DB_FILE/2014/12/4679.pdf)

`glmbook` - [Generalized Linear Models](http://www.utstat.toronto.edu/~brunner/oldclass/2201s11/readings/glmbook.pdf), McCullagh and Nelder
