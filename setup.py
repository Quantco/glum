from os import path
import io

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import mako.template
import mako.runtime

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

# TODO: this should be moved inside the compilation of the extension
print('templating C source')
for fn in ['src/glm_benchmarks/sandwich/dense-tmpl.c', 'src/glm_benchmarks/sandwich/sparse-tmpl.c']:
    tmpl = mako.template.Template(filename=fn)

    buf = io.StringIO()
    try:
        ctx = mako.runtime.Context(buf)
        rendered_src = tmpl.render_context(ctx)
    except:
        print(mako.exceptions.text_error_template().render())
        raise

    out_fn = fn.split('-tmpl')[0] + '.c'
    with open(out_fn, 'w') as f:
        f.write(buf.getvalue())

ext_modules = [
    Extension(
        name="glm_benchmarks.sandwich.sandwich",
        sources=[
            "src/glm_benchmarks/sandwich/sandwich.pyx"
        ],
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
