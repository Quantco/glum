import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

if sys.platform == "win32":
    allocator_libs = []  # type: ignore
    extra_compile_args = ["/openmp", "/O2"]
    extra_link_args = ["/openmp"]
elif sys.platform == "darwin":
    allocator_libs = ["jemalloc"]
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "--std=c++17",
    ]
    extra_link_args = [""]
else:
    allocator_libs = ["jemalloc"]
    extra_compile_args = [
        "-fopenmp",
        "-O3",
        "-ffast-math",
        "--std=c++17",
    ]
    extra_link_args = ["-fopenmp"]

architecture = os.environ.get("GLM_ARCHITECTURE", "native")
if architecture != "default":
    extra_compile_args.append("-march=" + architecture)

extension_args = dict(
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)
ext_modules = [
    Extension(
        name="glum._functions",
        sources=["src/glum/_functions.pyx"],
        **extension_args,
    ),
    Extension(
        name="glum._cd_fast",
        sources=["src/glum/_cd_fast.pyx"],
        **extension_args,
    ),
]

setup(
    name="glum",
    use_scm_version={"version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    description="High performance Python GLMs with all the features!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quantco/glum",
    author="QuantCo, Inc.",
    author_email="noreply@quantco.com",
    license="BSD",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["glum"]
        if os.environ.get("CONDA_BUILD")
        else ["glum", "glum_benchmarks"],
    ),
    install_requires=[
        "joblib",
        "numexpr",
        "numpy",
        "pandas",
        "scikit-learn>=0.23",
        "scipy",
        "tabmat>=3.1.0",
    ],
    entry_points=None
    if os.environ.get("CONDA_BUILD")
    else """
        [console_scripts]
        glm_benchmarks_run = glum_benchmarks.cli_run:cli_run
        glm_benchmarks_analyze = glum_benchmarks.cli_analyze:cli_analyze
    """,
    ext_modules=cythonize(ext_modules, annotate=False),
    zip_safe=False,
)
