#!/usr/bin/env bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Builds the test server for distributed (GRPC) TensorFlow
#
# Usage: build_server.sh <docker_image_name>
#
# This script obeys the following environment variables
#   TF_DIST_DOCKER_NO_CACHE:      do not use cache when building docker images

# Helper functions
die() {
  echo $@
  exit 1
}

# Check arguments
if [[ $# != 1 ]]; then
  die "Usage: $0 <docker_image_name>"
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

NO_CACHE_FLAG=""
if [[ ! -z "${TF_DIST_DOCKER_NO_CACHE}" ]] &&
   [[ "${TF_DIST_DOCKER_NO_CACHE}" != "0" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

# Call docker build
docker build ${NO_CACHE_FLAG} -t "${DOCKER_IMG_NAME}" \
   -f "${DIR}/server/Dockerfile" \
   "${DIR}"

rm -rf ${TMP_DATA_DIR}
