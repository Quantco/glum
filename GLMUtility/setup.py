import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glmutility",
    version="0.2.0",
    author="John S. Bogaardt",
    author_email="jbogaardt@gmail.com",
    description="Utility wrapper for statsmodels generalized linear model framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jbogaardt/GLMUtility",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT LIcense",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "pandas", "pyyaml", "scipy", "statsmodels", "patsy"],
    extras_require={"jupyter": ["bokeh", "ipywidgets"]},
)
