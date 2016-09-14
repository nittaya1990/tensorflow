#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Tests distributed TensorFlow on a locally running TF GRPC cluster.
#
# This script peforms the following steps:
# 1) Build the docker-in-docker (dind) image capable of running docker and
#    Kubernetes (k8s) cluster inside.
# 2) Run a container from the aforementioned image and start docker service
#    in it
# 3) Call a script to launch a k8s TensorFlow GRPC cluster inside the container
#    and run the distributed test suite.
#
# Usage: local_test.sh [--leave-container-running]
#                      [--model-name <MODEL_NAME>]
#                      [--num-workers <NUM_WORKERS>]
#                      [--num-parameter-servers <NUM_PARAMETER_SERVERS>]
#                      [--sync-replicas]
#
# E.g., local_test.sh --model-name CENSUS_WIDENDEEP
#       local_test.sh --num-workers 3 --num-parameter-servers 3
#
# Arguments:
# --leave-container-running:  Do not stop the docker-in-docker container after
#                             the termination of the tests, e.g., for debugging
#
# --num-workers <NUM_WORKERS>:
#   Specifies the number of worker pods to start
#
# --num-parameter-server <NUM_PARAMETER_SERVERS>:
#   Specifies the number of parameter servers to start
#
# --sync-replicas
#   Use the synchronized-replica mode. The parameter updates from the replicas
#   (workers) will be aggregated before applied, which avoids stale parameter
#   updates.
#
# In addition, this script obeys the following environment variables:
# TF_DIST_DOCKER_NO_CACHE:      do not use cache when building docker images


# Configurations
DOCKER_IMG_NAME="tensorflow/tf-dist-test-local-cluster"
LOCAL_K8S_CACHE=${HOME}/kubernetes

# Helper function
get_container_id_by_image_name() {
    # Get the id of a container by image name
    # Usage: get_docker_container_id_by_image_name <img_name>

    echo $(docker ps | grep $1 | awk '{print $1}')
}

# Parse input arguments
LEAVE_CONTAINER_RUNNING=0
MODEL_NAME=""
MODEL_NAME_FLAG=""
NUM_WORKERS=2
NUM_PARAMETER_SERVERS=2
SYNC_REPLICAS=0

while true; do
  if [[ $1 == "--leave-container-running" ]]; then
    LEAVE_CONTAINER_RUNNING=1
  elif [[ $1 == "--model-name" ]]; then
    MODEL_NAME="$2"
    MODEL_NAME_FLAG="--model-name ${MODEL_NAME}"
  elif [[ $1 == "--num-workers" ]]; then
    NUM_WORKERS=$2
  elif [[ $1 == "--num-parameter-servers" ]]; then
    NUM_PARAMETER_SERVERS=$2
  elif [[ $1 == "--sync-replicas" ]]; then
    SYNC_REPLICAS=1
  fi

  shift
  if [[ -z $1 ]]; then
    break
  fi
done

echo "LEAVE_CONTAINER_RUNNING: ${LEAVE_CONTAINER_RUNNING}"
echo "MODEL_NAME: \"${MODEL_NAME}\""
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "NUM_PARAMETER_SERVERS: ${NUM_PARAMETER_SERVERS}"
echo "SYNC_REPLICAS: \"${SYNC_REPLICAS}\""

# Current script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get utility functions
source ${DIR}/scripts/utils.sh


# First, make sure that no docker-in-docker container of the same image
# is already running
if [[ ! -z $(get_container_id_by_image_name ${DOCKER_IMG_NAME}) ]]; then
    die "It appears that there is already at least one Docker container "\
"of image name ${DOCKER_IMG_NAME} running. Please stop it before trying again"
fi

# Build docker-in-docker image for local k8s cluster
NO_CACHE_FLAG=""
if [[ ! -z "${TF_DIST_DOCKER_NO_CACHE}" ]] &&
   [[ "${TF_DIST_DOCKER_NO_CACHE}" != "0" ]]; then
  NO_CACHE_FLAG="--no-cache"
fi

docker build ${NO_CACHE_FLAG} -t ${DOCKER_IMG_NAME} \
   -f ${DIR}/Dockerfile.local ${DIR} || \
   die "Failed to build docker image: ${DOCKER_IMG_NAME}"

docker run ${DOCKER_IMG_NAME} \
    /var/tf_dist_test/scripts/dist_mnist_test.sh \
    --ps_hosts "localhost:2000,localhost:2001" \
    --worker_hosts "localhost:3000,localhost:3001" \
    --num_gpus 0
