# glm_benchmarks

Python package to benchmark GLM implementations. 

[Link to Google Sheet that compares various existing implementations.](https://docs.google.com/spreadsheets/d/1C-n3YTzPR47Sf8M04eEaX4RbNomM13dk_BZaPHGgWXg/edit)

## Installation

You can install the package in development mode using:

```bash
git clone https://github.com/Quantco/glm_benchmarks
cd glm_benchmarks
pre-commit install
pip install --no-use-pep517 --disable-pip-version-check -e .
```

## Docker

To build the image, make sure you have a functioning Docker and docker-compose installation. Then, `docker-compose build`.

To run something, for example: `./run python benchmarks/sklearn_fork.py`
