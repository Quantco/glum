import io
import os
import sys
from os import path

import mako.runtime
import mako.template
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# TODO: this should be moved inside the compilation of the extension
print("templating C source")
templates = [
    "src/quantcore/glm/matrix/ext/dense_helpers-tmpl.cpp",
    "src/quantcore/glm/matrix/ext/sparse_helpers-tmpl.cpp",
]

for fn in templates:
    tmpl = mako.template.Template(filename=fn)

    buf = io.StringIO()
    ctx = mako.runtime.Context(buf)
    tmpl.render_context(ctx)
    rendered_src = buf.getvalue()

    out_fn = fn.split("-tmpl")[0] + ".cpp"

    # When the templated source code hasn't changed, we don't want to write the
    # file again because that'll touch the file and result in a rebuild
    write = True
    if path.exists(out_fn):
        with open(out_fn, "r") as f:
            out_fn_src = f.read()
            if out_fn_src == rendered_src:
                write = False

    if write:
        with open(out_fn, "w") as f:
            f.write(rendered_src)

if sys.platform == "win32":
    allocator_libs = []
    extra_compile_args = ["/openmp", "/O2"]
    extra_link_args = ["/openmp"]
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
        name="quantcore.glm.matrix.ext.sparse",
        sources=["src/quantcore/glm/matrix/ext/sparse.pyx"],
        libraries=allocator_libs,
        **extension_args,
    ),
    Extension(
        name="quantcore.glm.matrix.ext.dense",
        sources=["src/quantcore/glm/matrix/ext/dense.pyx"],
        libraries=allocator_libs,
        **extension_args,
    ),
    Extension(
        name="quantcore.glm.matrix.ext.categorical",
        sources=["src/quantcore/glm/matrix/ext/categorical.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="quantcore.glm.sklearn_fork._functions",
        sources=["src/quantcore/glm/sklearn_fork/_functions.pyx"],
        **extension_args,
    ),
    Extension(
        name="quantcore.glm.sklearn_fork._cd_fast",
        sources=["src/quantcore/glm/sklearn_fork/_cd_fast.pyx"],
        **extension_args,
    ),
]

setup(
    name="quantcore.glm",
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
        "Programming Language :: Python :: 3.8",
    ],
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[],
    entry_points="""
        [console_scripts]
        glm_benchmarks_run = quantcore.glm.cli_run:cli_run
        glm_benchmarks_analyze = quantcore.glm.cli_analyze:cli_analyze
    """,
    ext_modules=cythonize(ext_modules, annotate=False),
    zip_safe=False,
)
