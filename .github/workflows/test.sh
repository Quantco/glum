#!/bin/bash

set -eo pipefail

docker-compose run work pre-commit install
docker-compose run work pre-commit run --all
docker-compose run work pytest
docker-compose run work pytest --doctest-modules src/
