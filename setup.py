import io
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
for fn in ["src/quantcore/glm/matrix/sandwich/dense-tmpl.cpp"]:
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

extension_args = dict(
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-fopenmp",
        "-O3",
        "-ffast-math",
        "-march=native",
        "--std=c++17",
    ],
    extra_link_args=["-fopenmp"],
    language="c++",
)
ext_modules = [
    Extension(
        name="quantcore.glm.matrix.sandwich.sandwich",
        sources=["src/quantcore/glm/matrix/sandwich/sandwich.pyx"],
        libraries=["jemalloc"],
        **extension_args,
    ),
    Extension(
        name="quantcore.glm.sklearn_fork._functions",
        sources=["src/quantcore/glm/sklearn_fork/_functions.pyx"],
        **extension_args,
    ),
    Extension(
        name="quantcore.glm.sklearn_fork._cd_fast",
        sources=["src/quantcore/glm/sklearn_fork/_cd_fast.pyx"],
        include_dirs=[np.get_include()],
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
