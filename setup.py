import io
from os import path

import mako.runtime
import mako.template
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# TODO: this should be moved inside the compilation of the extension
print("templating C source")
for fn in ["src/glm_benchmarks/sandwich/dense-tmpl.cpp"]:
    tmpl = mako.template.Template(filename=fn)

    buf = io.StringIO()
    ctx = mako.runtime.Context(buf)
    tmpl.render_context(ctx)
    rendered_src = buf.getvalue()

    out_fn = fn.split("-tmpl")[0] + ".cpp"

    # When the templated source code hasn't changed, we don't want to write the
    # file again because that'll touch the file and result in a rebuild
    write = True
    with open(out_fn, "r") as f:
        out_fn_src = f.read()
        if out_fn_src == rendered_src:
            write = False

    if write:
        with open(out_fn, "w") as f:
            f.write(rendered_src)

ext_modules = [
    Extension(
        name="glm_benchmarks.sandwich.sandwich",
        sources=["src/glm_benchmarks/sandwich/sandwich.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="glm_benchmarks.sklearn_fork._cd_fast",
        sources=["src/glm_benchmarks/sklearn_fork/_cd_fast.pyx"],
        include_dirs=[np.get_include()],
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
