FROM docker.pkg.github.com/quantco/miniforge3-qc-dev/miniconda3-qc-base:latest

RUN mkdir /app
WORKDIR /app

COPY environment.yml /app/environment.yml
RUN mamba env create

# We wait to copy the full app folder until now so that image caching still
# works for the previous slow-running install lines (conda and pip)
COPY . /app
RUN conda run -n quantcore.glm pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .

# set location for generated data caching
ENV GLM_BENCHMARKS_CACHE=/cache

ENTRYPOINT ["/app/build_and_launch"]
CMD ["bash"]
