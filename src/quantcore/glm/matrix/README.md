# Efficient matrix representations for working with tabular data

## Use case
Data used in economics, actuarial science, and many other fields is often tabular,
containing rows and columns. Further properties are also common:
- Tabular data often contains categorical data, 
often represented after processing as many columns of indicator values
created by "one-hot encoding."
- It often contains a mix of dense columns and sparse columns, 
perhaps due to one-hot encoding.
- It often is very sparse.

High-performance statistical applications often require fast computation of certain
operations, such as
- Operating on one column at a time
- Computing "sandwich products" of the data, `transpose(X) @ diag(d) @ X`. A sandwich
product shows up in the solution to Weighted Least Squares, as well as in the Hessian
of the likelihood in Generalized Linear Models such as Poisson regression.
- Matrix-vector products

Additionally, it is often desirable to normalize predictors for greater optimizer 
efficiency and numerical stability in
Coordinate Descent and in other machine learning algorithms.

## This library and its design

We designed this library with these use cases in mind. We built this library first for
estimating Generalized Linear Models, but expect it will be useful in a variety of
econometric and statistical use cases. This library was borne out of our need for 
speed, and its unified API is motivated by the annoyance by having to write repeated
checks for which type of matrix-like object you are operating on.

Design principles:
- Speed and memory efficiency are paramount.
- You don't need to sacrifice functionality by using this library: DenseGLMDataMatrix 
and MKLSparseMatrix subclass Numpy arrays and Scipy csc sparse matrices, respectively, 
and inherit their behavior wherever it is not improved on.
- As much as possible, syntax follows Numpy syntax, and dimension-reducing
  operations (like `sum`) return Numpy arrays, following Numpy dimensions
  about the dimensions of results. The aim is to make these classes
  as close as possible to being drop-in replacements for numpy ndarray.
  This is not always possible, however, due to the differing APIs of numpy ndarray
  and scipy sparse.
- Other operations, such as `toarray`, mimic Scipy sparse syntax.
- All matrix classes support matrix products, sandwich products, and `getcol`.
Individual subclasses may support significantly more operations.

## Matrix types
- `DenseGLMDataMatrix` represents dense matrices, subclassing numpy nparray. 
    It additionally supports methods `getcol`, `toarray`, `sandwich`, `standardize`, 
    and `unstandardize`.
- `MKLSparseMatrix` represents column-major sparse data, subclassing 
    `scipy.sparse.csc_matrix`. It additionally supports methods `sandwich`
    and `standardize`, and it's `dot` method (e.g. `@`) calls MKL's sparse dot product
    in the case of matrix-vector products, which is faster.
- `ColScaledSpMat` represents the sum of an n x k sparse matrix and a matrix
    of the form `ones((n, 1)) x shift`, where `shift` is `1 x k`. In other words,
    a matrix with a column-specific shifter applied. Such a matrix is dense, but
    `ColScaledSpMat` represents the sparse matrix and `shift` separately, allowing for
    efficient storage and computations.
- `SplitMatrix` represents matrices with both sparse and dense parts, allowing for
    a significant speedup in matrix multiplications.
    
## Benchmarks
???