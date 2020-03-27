FROM continuumio/miniconda3:latest

RUN apt-get install -y gfortran

RUN mkdir /app
WORKDIR /app

COPY conda-requirements.txt /app
RUN conda install -c conda-forge --file conda-requirements.txt

COPY pip-requirements.txt /app
RUN pip install -r pip-requirements.txt

# We wait to copy the full app folder until now so that image caching still
# works for the previous slow-running install lines (conda and pip)
COPY . /app
RUN pip install --no-use-pep517 --disable-pip-version-check -e .

CMD ["bash"]
