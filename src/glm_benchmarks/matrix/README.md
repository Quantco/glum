Design principles:
- As much as possible, syntax follows Numpy syntax, and dimension-reducing
  operations (like `sum`) return Numpy arrays, following Numpy dimensions
  about the dimensions of results. The aim is to make these classes
  as close as possible to being drop-in replacements for numpy ndarray.
- Other operations, such as `toarray`, mimic Scipy sparse syntax.