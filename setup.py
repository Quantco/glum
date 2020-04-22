from os import path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

include_mkl = np.__config__.get_info("blas_mkl_info")["include_dirs"][0]
libraries_mkl = np.__config__.get_info("blas_mkl_info")["include_dirs"][0]

ext_modules = [
    Extension(
        name="glm_benchmarks.spblas.mkl_spblas",
        sources=["src/glm_benchmarks/spblas/mkl_spblas.pyx"],
        include_dirs=[np.get_include(), include_mkl],
        libraries=["mkl_rt"],
        library_dirs=["", libraries_mkl],
    ),
]

setup(
    name="glm_benchmarks",
    use_scm_version={"version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    description="Python package to benchmark GLM implementations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quantco/glm_benchmarks",
    author="QuantCo, Inc.",
    author_email="noreply@quantco.com",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    entry_points="""
        [console_scripts]
        glm_benchmarks_run = glm_benchmarks.main:cli_run
        glm_benchmarks_analyze = glm_benchmarks.main:cli_analyze
    """,
    ext_modules=cythonize(ext_modules, annotate=False),
    zip_safe=False,
)
