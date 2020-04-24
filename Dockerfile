FROM continuumio/miniconda3:latest

RUN mkdir /app
WORKDIR /app

# To ensure we're installing anaconda package versions, let's install directly
# from anaconda first, then afterwards, install from conda-forge and pip
COPY requirements/ /app/requirements
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda install --file requirements/conda-requirements.txt
RUN pip install -r requirements/pip-requirements.txt


# We wait to copy the full app folder until now so that image caching still
# works for the previous slow-running install lines (conda and pip)
COPY . /app
RUN conda run -n base pip install --no-use-pep517 --no-deps --disable-pip-version-check -e .

ENTRYPOINT ["/app/build_and_launch"]
CMD ["bash"]
