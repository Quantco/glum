# glm_benchmarks

![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)

Python package to benchmark GLM implementations. 

## Running the benchmarks

After installing the package, you should have two CLI tools: `glm_benchmarks_run` and `glm_benchmarks_analyze`. Use the `--help` flag for full details. Look in `src/glum/problems.py` to see the list of problems that will be run through each library.

To run the full benchmarking suite, just run `glm_benchmarks_run` with no flags. This will probably take a very long time.

For a more advanced example: `glm_benchmarks_run --problem_name narrow-insurance-no-weights-l2-poisson --library_name glum --storage dense --num_rows 1000 --output_dir mydatadirname` will run just the first 1000 rows of the `narrow-insurance-no-weights-l2-poisson` problem through the `glum` library and save the output to `mydatadirname`. This demonstrates several capabilities that will speed development when you just want to run a subset of either data or problems or libraries. 

Demonstrating the command above:
```
(glum) ➜  glum git:(master) ✗ glm_benchmarks_run --problem_name narrow-insuranc
e-no-weights-l2-poisson --library_name glum --storage dense --num_rows 1000 --output_dir
 mydatadirname
running problem=narrow-insurance-no-weights-l2-poisson library=glum
Diagnostics:
         convergence  n_cycles  iteration_runtime  intercept
n_iter                                                      
0       1.444101e+00         0           0.001196  -1.843114
1       5.008199e-01         1           0.009937  -1.843114
2       8.087132e-02         2           0.001981  -2.311497
3       1.143680e-02         3           0.001860  -2.563429
4       3.526882e-04         4           0.001864  -2.607574
5       3.658644e-07         5           0.002538  -2.608770
ran problem narrow-insurance-no-weights-l2-poisson with library glum
ran in 0.045558929443359375
```

The `--problem_name` and `--library_name` flags take comma separated lists. This mean that if you want to run both `glum` and `r-glmnet`, you could run `glm_benchmarks_run --library_name glum,r-glmnet`.

The `glm_benchmarks_analyze` tool produces a dataframe comparing the correct and runtime of several runs/libraries. `glm_benchmarks_analyze` accepts an almost identical range of command line parameters as `glm_benchmarks_run`. You can use these CLI parameters to filter which problems and runs you would like to compare. 

For example:
```
(glum) ➜  glum git:(master) ✗ glm_benchmarks_analyze --problem_name narrow-insurance-no-weights-l2-poisson --library_name glum --storage dense --num_rows 1000 --output_dir mydatadirname --cols intercept,runtime,n_iter
                                                                                 library_name  intercept  runtime  n_iter
problem_name                           num_rows regularization_strength offset                                           
narrow-insurance-no-weights-l2-poisson 1000     0.001                   False   glum    -3.3194   0.0456       5
```

Benchmarks can be sped up by enabling caching of generated data. If you don't do this, you will spend a lot of time repeatedly generating the same data set. To enable caching, set the GLM_BENCHMARKS_CACHE environment variable to the directory you would like to write to.

We support several types of matrix storage, passed with the argument "--storage". The default is "auto" which splits the matrix into dense, sparse, and categorical subcomponents using `tabmat`. "dense" stores the data as a numpy array. "sparse" stores data as a CSC sparse matrix. "cat" splits the matrix into a dense component and categorical components. "split0.1" splits the matrix into sparse and dense parts, where any column with more than 10% nonzero elements is put into the dense part, and the rest is put into the sparse part.

## Profiling

For line-by-line profiling, mark any functions that you'd like to profile with the `@profile` decorator and then launch using line_profiler with `kernprof -lbv src/glum_benchmarks/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name glum`

For stack sampling profiling, use py-spy: `py-spy top -- python src/glum_benchmarks/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name glum`

## Memory profiling

To create a graph of memory usage:
```
mprof run --python -o mprofresults.dat --interval 0.01 src/glum_benchmarks/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name glum --num_rows 100000
mprof plot mprofresults.dat -o prof2.png
```

To do line-by-line memory profiling, add a `@profile` decorator to the functions you care about and then run:
```
python -m memory_profiler src/glum_benchmarks/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name glum --num_rows 100000
```
