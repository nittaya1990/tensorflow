#!/bin/bash

# Helper functions
die() {
  echo $@
  exit 1
}

if [[ $# != 1 ]]; then
  echo "Usage: $0 <docker_image_name>"
  exit 1
fi

DOCKER_IMG_NAME=$1

# Get current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Download mnist data to be included in the docker image
TMP_DATA_DIR="${DIR}/server/mnist-data"
mkdir -p "${TMP_DATA_DIR}" || \
    die "FAILED to create temporary directory for mnist-data"
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
     -P ${TMP_DATA_DIR}/
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
     -P ${TMP_DATA_DIR}/
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz \
     -P ${TMP_DATA_DIR}/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
     -P ${TMP_DATA_DIR}/

docker build -t "${DOCKER_IMG_NAME}" \
   -f "${DIR}/server/Dockerfile" \
   "${DIR}"

rm -rf ${TMP_DATA_DIR}
