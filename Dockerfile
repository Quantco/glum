FROM continuumio/miniconda3:latest

# Installing Java for H2O
# the mkdir -p line is necessary to avoid a bug in default-jdk-headless
# installation: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=863199
RUN apt-get update && \
    mkdir -p /usr/share/man/man1 && \ 
    DEBIAN_FRONTEND=noninteractive apt-get -y install default-jdk-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

# To ensure we're installing anaconda package versions, let's install directly
# from anaconda first, then afterwards, install from conda-forge and pip
COPY requirements/ /app/requirements
RUN conda install --file requirements/anaconda-requirements.txt
RUN conda install -c conda-forge --file requirements/conda-forge-requirements.txt
RUN pip install -r requirements/pip-requirements.txt

# We wait to copy the full app folder until now so that image caching still
# works for the previous slow-running install lines (conda and pip)
COPY . /app
RUN pip install --no-use-pep517 --disable-pip-version-check -e .

CMD ["bash"]
