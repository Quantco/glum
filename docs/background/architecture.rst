Architecture
============

quantcore.glm
--------------

The user interface of 

The GLM benchmark suite
------------------------

Before deciding to build a library custom built for our purposes, we did an thorough investigation of the various open source GLM implementations available. This resulted in an extensive suite of benchmarks for comparing the correctness, runtime and availability of features for these libraries. 

The benchmark suite has two command line entrypoints:

* ``glm_benchmarks_run``
* ``glm_benchmarks_analyze``

Both of these CLI tools take a range of arguments that specify the details of the benchmark problems and which libraries to benchmark.

For more details on the benchmark suite, see the README in the source at ``src/quantcore/glm_benchmarks/README.md``.



