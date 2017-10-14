#!/bin/bash

set -euo pipefail

docker build --tag snackable/classifier:local .
docker run -p 80:80 snackable/classifier:local

