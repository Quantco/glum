FROM continuumio/miniconda3:latest

RUN mkdir /app
WORKDIR /app

COPY conda-requirements.txt /app
RUN conda install -c conda-forge --file conda-requirements.txt

COPY . /app
RUN pip install --no-use-pep517 --disable-pip-version-check -e .

CMD ["bash"]
